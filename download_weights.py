# download_weights.py
import os
from huggingface_hub import snapshot_download

# The official Diffusers format for the Wan 2.2 Image-to-Video MoE model
MODEL_ID = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Optional: set HF_TOKEN env var if the model is gated on HuggingFace
hf_token = os.environ.get("HF_TOKEN", None)

print(f"Downloading {MODEL_ID} weights to bake into the image...")
snapshot_download(
    repo_id=MODEL_ID,
    ignore_patterns=["*.pt", "*.bin"],  # We only need the .safetensors format
    token=hf_token,
)
print("Download complete.")
