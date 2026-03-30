# predict.py
import os
import shutil
import tempfile
from typing import Optional
import numpy as np
import requests
from PIL import Image
from cog import BasePredictor, Input, Path
import torch
from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video

MODEL_ID = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"

# Max pixel area per resolution tier
MAX_AREA = {
    "480p": 480 * 832,
    "720p": 720 * 1280,
}

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model from weights baked into the image"""
        print(f"Loading {MODEL_ID} from cache...")

        self.pipe = WanImageToVideoPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
        )
        # Cast VAE to float32 before CPU offload so offload hooks see the right dtype
        self.pipe.vae.to(dtype=torch.float32)
        # REQUIRED FOR 14B MoE MODEL: offloads weights to CPU RAM between uses
        self.pipe.enable_model_cpu_offload()

    def predict(
        self,
        image: Path = Input(description="Input image to animate."),
        prompt: str = Input(description="Prompt for video generation."),
        negative_prompt: str = Input(description="Negative prompt.", default=""),
        resolution: str = Input(
            description="Output resolution. 720p requires more VRAM and time.",
            default="480p",
            choices=["480p", "720p"],
        ),
        lora_url: Optional[str] = Input(
            description="Optional: URL to a .safetensors LoRA file (e.g., from CivitAI).",
            default=None
        ),
        civitai_token: Optional[str] = Input(
            description="Optional: CivitAI API token for gated LoRA downloads.",
            default=None
        ),
        lora_scale: float = Input(description="LoRA scale (0.0–2.0).", default=1.0, ge=0.0, le=2.0),
        num_frames: int = Input(
            description="Number of frames. Must satisfy (n-1) % 4 == 0, e.g. 17, 33, 49, 65, 81.",
            default=81, ge=5, le=121
        ),
        num_inference_steps: int = Input(description="Number of denoising steps.", default=50, ge=10, le=100),
        guidance_scale: float = Input(description="Guidance scale.", default=3.5, ge=1.0, le=20.0),
        seed: int = Input(description="Random seed. Set to -1 to randomize.", default=-1),
    ) -> Path:
        """Run a single prediction on the model"""

        if (num_frames - 1) % 4 != 0:
            raise ValueError(
                f"num_frames must satisfy (n-1) % 4 == 0 (e.g. 17, 33, 49, 65, 81). Got {num_frames}."
            )

        # Resize input image to Wan2.2-compatible dimensions (explicit int cast required)
        input_image = Image.open(str(image)).convert("RGB")
        max_area = MAX_AREA[resolution]
        aspect_ratio = input_image.height / input_image.width
        mod_value = int(
            self.pipe.vae_scale_factor_spatial
            * self.pipe.transformer.config.patch_size[1]
        )
        height = int(round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value)
        width = int(round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value)
        input_image = input_image.resize((width, height))

        if seed == -1:
            seed = int(torch.randint(0, 2**32, (1,)).item())
        generator = torch.Generator(device="cuda").manual_seed(seed)
        print(f"Seed: {seed}, resolution: {width}x{height}")

        temp_dir = None
        adapter_name = "custom_lora"
        lora_loaded = False

        try:
            # Download and load LoRA if provided
            if lora_url:
                download_url = lora_url
                if civitai_token:
                    delimiter = "&" if "?" in download_url else "?"
                    download_url += f"{delimiter}token={civitai_token}"

                print("Downloading LoRA...")  # don't log URL — may contain token
                temp_dir = tempfile.mkdtemp()
                lora_path = os.path.join(temp_dir, "lora.safetensors")

                response = requests.get(download_url, stream=True, allow_redirects=True, timeout=300)
                response.raise_for_status()

                with open(lora_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                print("Loading LoRA weights...")
                self.pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
                # Mark loaded immediately so finally block unloads it even if set_adapters fails
                lora_loaded = True
                self.pipe.set_adapters([adapter_name], adapter_weights=[lora_scale])
            else:
                self.pipe.unload_lora_weights()

            # Generate video
            kwargs = {
                "image": input_image,
                "prompt": prompt,
                "height": height,
                "width": width,
                "num_frames": num_frames,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "generator": generator,
                "output_type": "pil",
            }
            if negative_prompt:
                kwargs["negative_prompt"] = negative_prompt

            print(f"Generating {num_frames} frames at {width}x{height}...")
            result = self.pipe(**kwargs)

            if not result.frames or len(result.frames[0]) == 0:
                raise RuntimeError("Pipeline returned no frames.")

            output_frames = result.frames[0]

        finally:
            # Always clean up LoRA and temp files, even if generation failed
            if lora_loaded:
                self.pipe.unload_lora_weights()
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

        # Write to a unique temp path to avoid collisions on concurrent predictions
        out_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        out_path = out_file.name
        out_file.close()
        export_to_video(output_frames, out_path, fps=16)

        return Path(out_path)
