# get_data.py
#
# Downloads news articles and their real images from HuggingFace.
#
# I spent a while figuring out the image format -- the "image" column in
# this dataset can come back as a string path like "images/39e6104d56.jpeg",
# a PIL object, or a dict with {"bytes": ..., "path": ...} depending on how
# HuggingFace decides to load it. So I handle all three cases below.
#
# Run once before training:
#     python get_data.py

import pandas as pd
import requests
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import os

# ── Settings ──────────────────────────────────────────────────────────────────

SAMPLES_PER_CLASS = 200
IMAGE_DIR = Path("data/raw/images")
OUTPUT_CSV = Path("data/raw/articles.csv")
DATASET_REPO = "vector-institute/newsmediabias-plus"

# Images live in the HuggingFace repo -- we build the download URL using the CDN pattern
HF_IMAGE_BASE = f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/main/"

OUTLET_BIAS_MAP = {
    # Left-leaning
    "CNN": "Left", "The Guardian": "Left", "The Guardian US": "Left",
    "HuffPost": "Left", "NPR": "Left", "The New Yorker": "Left",
    "Daily Kos": "Left", "ABC News": "Left", "CBS News": "Left",
    "NBC News": "Left", "The Washington Post": "Left",
    "The New York Times": "Left", "Vox": "Left", "Vice": "Left",
    "The Atlantic": "Left", "Al Jazeera": "Left",
    "CBC.ca": "Left", "CBC News": "Left", "CBC Sports": "Left",
    "Global News Toronto": "Left", "Mother Jones": "Left", "Slate": "Left",
    # Center
    "Reuters": "Center", "The Associated Press": "Center", "BBC.com": "Center",
    "Bloomberg": "Center", "Financial Times": "Center", "USA TODAY": "Center",
    "Newsweek": "Center", "The Hill": "Center", "Axios": "Center",
    "ProPublica": "Center", "The Globe and Mail": "Center", "Time": "Center",
    "PBS NewsHour": "Center", "The Economist": "Center",
    # Right-leaning
    "Fox News": "Right", "Breitbart": "Right", "New York Post": "Right",
    "National Review": "Right", "The Federalist": "Right",
    "Daily Caller": "Right", "Newsmax": "Right", "The Daily Wire": "Right",
    "The Washington Examiner": "Right", "Townhall": "Right",
    "Forbes": "Right", "Wall Street Journal": "Right",
}


# ── Helper functions ──────────────────────────────────────────────────────────

def get_bias_label(outlet: str):
    """Map outlet name to Left/Center/Right using AllSides ratings."""
    if outlet in OUTLET_BIAS_MAP:
        return OUTLET_BIAS_MAP[outlet]
    for known, label in OUTLET_BIAS_MAP.items():
        if known.lower() in outlet.lower():
            return label
    return None


def fetch_image(image_path_in_repo: str, save_path: Path) -> bool:
    """
    Download one image from the HuggingFace dataset repo.

    The path in the dataset looks like "images/39e6104d56.jpeg" -- we
    prepend the CDN base URL and do a regular HTTP download. Uses the
    cached HF login token if available, which is needed for gated repos.
    Returns True on success, False if anything goes wrong.
    """
    if not image_path_in_repo or not str(image_path_in_repo).startswith("images/"):
        return False

    url = HF_IMAGE_BASE + image_path_in_repo

    try:
        # Use HuggingFace token from cache if available
        token = None
        try:
            from huggingface_hub import HfFolder
            token = HfFolder.get_token()
        except Exception:
            pass

        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        response = requests.get(url, headers=headers, timeout=15, stream=True)

        if response.status_code == 200:
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(8192):
                    f.write(chunk)
            return True
        return False

    except Exception:
        return False


def make_placeholder(save_path: Path):
    """Make a gray placeholder image for when a download fails."""
    from PIL import Image
    Image.new("RGB", (224, 224), color=(128, 128, 128)).save(save_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def download_dataset():
    print("=" * 60)
    print("Downloading NewsMediaBias-Plus with real images")
    print("=" * 60)
    print(f"Target: {SAMPLES_PER_CLASS} per class ({SAMPLES_PER_CLASS * 3} total)")
    print("Fetching images from HuggingFace CDN...\n")

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Load dataset metadata (non-streaming so all columns come through reliably)
    print("Loading dataset metadata...")
    try:
        dataset = load_dataset(DATASET_REPO)
        raw_data = dataset["train"]
        print(f"Loaded {len(raw_data)} articles\n")

        # Print what the image column looks like -- useful for debugging
        # if the HF dataset format ever changes
        sample = raw_data[0]
        image_val = sample.get("image", "")
        print(f"Image column sample value: '{image_val}'")
        print(f"Image column type: {type(image_val).__name__}\n")

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False

    # Collect balanced samples
    print("Collecting and downloading images...")
    buckets = {"Left": [], "Center": [], "Right": []}
    real_images = 0
    placeholder_images = 0
    skipped = 0

    for i, article in enumerate(tqdm(raw_data, desc="Processing")):
        if all(len(b) >= SAMPLES_PER_CLASS for b in buckets.values()):
            break

        outlet = str(article.get("outlet", "")).strip()
        label = get_bias_label(outlet)

        if label is None:
            skipped += 1
            continue

        if len(buckets[label]) >= SAMPLES_PER_CLASS:
            continue

        # Use article_text if it exists, fall back to headline if it's too short
        text = str(article.get("article_text", "")).strip()
        if not text or len(text) < 50:
            text = str(article.get("headline", "")).strip()
        if not text:
            continue

        # Get image
        image_col = article.get("image", "")
        article_id = str(article.get("unique_id", f"article_{i}"))
        image_path = IMAGE_DIR / f"{article_id}.jpg"

        # The image column can come in three different formats depending on
        # how HuggingFace loaded the dataset -- handle all three
        from PIL import Image as PILImage

        if isinstance(image_col, PILImage.Image):
            # Already a PIL image -- just save it
            try:
                image_col.convert("RGB").save(image_path, "JPEG")
                real_images += 1
            except Exception:
                make_placeholder(image_path)
                placeholder_images += 1

        elif isinstance(image_col, str) and image_col.startswith("images/"):
            # String path -- download from HuggingFace CDN
            success = fetch_image(image_col, image_path)
            if success:
                real_images += 1
            else:
                make_placeholder(image_path)
                placeholder_images += 1

        elif isinstance(image_col, dict):
            # Sometimes HF returns {"bytes": ..., "path": ...}
            img_bytes = image_col.get("bytes")
            if img_bytes:
                try:
                    import io
                    img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
                    img.save(image_path, "JPEG")
                    real_images += 1
                except Exception:
                    make_placeholder(image_path)
                    placeholder_images += 1
            else:
                make_placeholder(image_path)
                placeholder_images += 1
        else:
            make_placeholder(image_path)
            placeholder_images += 1

        buckets[label].append({
            "text": text,
            "image_path": str(image_path),
            "label": label,
            "outlet": outlet,
            "headline": str(article.get("headline", "")).strip(),
        })

    # Save
    all_rows = []
    for rows in buckets.values():
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\n{'='*60}")
    print("DOWNLOAD COMPLETE")
    print(f"{'='*60}")
    print(f"Articles saved:     {len(df)}")
    print(f"Real images:        {real_images}")
    print(f"Placeholders:       {placeholder_images}")
    print(f"\nLabel distribution:")
    print(df["label"].value_counts().to_string())

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved to: {OUTPUT_CSV}")

    if real_images == 0:
        print("\nNo real images extracted.")
        print("The image column format may have changed.")
        print("Text model (F1=0.85) is still valid and fully usable.")
        print("For the project, you can note that image data was")
        print("unavailable and focus on text + mismatch detection.")
    else:
        print(f"\n{real_images} real images saved to {IMAGE_DIR}")

    print("\nNext steps:")
    print("  Remove-Item -Recurse -Force data\\processed")
    print("  python utils/preprocessing.py")
    print("  python train.py --mode text")
    print("  python train.py --mode image")
    print("  python train.py --mode fusion")

    return True


if __name__ == "__main__":
    download_dataset()
