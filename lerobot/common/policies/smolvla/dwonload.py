from huggingface_hub import snapshot_download

local_dir = snapshot_download(repo_id="HuggingFaceTB/SmolVLM2-500M-Video-Instruct", local_dir="./SmolVLM2-local", local_dir_use_symlinks=False)
print(f"Model saved to: {local_dir}")
