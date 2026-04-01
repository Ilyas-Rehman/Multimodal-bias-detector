# test_image.py
#
# Run this to figure out how images are stored in the dataset.
# This will tell us exactly how to fix get_data.py.
#
#     python test_image.py

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from pathlib import Path

print("Loading first 3 articles from dataset...")
dataset = load_dataset("vector-institute/newsmediabias-plus")
train = dataset["train"]

for i in range(3):
    sample = train[i]
    image_val = sample.get("image", None)
    unique_id = sample.get("unique_id", f"article_{i}")

    print(f"\n--- Article {i} ---")
    print(f"  unique_id: {unique_id}")
    print(f"  image type: {type(image_val).__name__}")
    print(f"  image value: {str(image_val)[:80]}")

    # Try hf_hub_download approach
    if isinstance(image_val, str) and image_val.startswith("images/"):
        print(f"\n  Trying hf_hub_download for: {image_val}")
        try:
            local_path = hf_hub_download(
                repo_id="vector-institute/newsmediabias-plus",
                filename=image_val,
                repo_type="dataset",
                local_dir="data/raw/test_images"
            )
            print(f"  SUCCESS: downloaded to {local_path}")
            from PIL import Image
            img = Image.open(local_path)
            print(f"  Image size: {img.size}, mode: {img.mode}")
        except Exception as e:
            print(f"  FAILED: {e}")
