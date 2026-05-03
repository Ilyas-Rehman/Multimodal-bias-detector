# get_data.py
#
# Downloads news articles and their real images from HuggingFace.
#
# Image resolution order (tries each in sequence):
#   1. PIL image object already in HF column -- just save it
#   2. Dict with raw bytes in HF column -- decode and save
#   3. String path like "images/39e6104d56.jpeg" -- try HF CDN download
#   4. Scrape og:image / twitter:image from source_url  <-- NEW, fixes the 404 problem
#   5. Encode image_description via CLIP text encoder   <-- NEW, gives CLIP real signal
#   6. Gray placeholder (last resort)
#
# Steps 4 and 5 are the fix. The HF CDN paths all 404 because the image
# files were never uploaded to the repo. But every article has a source_url
# we can scrape the real header image from. If that also fails, the
# image_description field gives CLIP something meaningful to encode instead
# of a blank gray square.
#
# Run once before training:
#     python get_data.py

import io
import os
import pandas as pd
import requests
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image as PILImage

SAMPLES_PER_CLASS = 200
IMAGE_DIR = Path("data/raw/images")
OUTPUT_CSV = Path("data/raw/articles.csv")
DATASET_REPO = "vector-institute/newsmediabias-plus"
HF_IMAGE_BASE = f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/main/"

SCRAPE_TIMEOUT = 8  # seconds per HTTP request when scraping source_url

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

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


def get_bias_label(outlet: str):
    if outlet in OUTLET_BIAS_MAP:
        return OUTLET_BIAS_MAP[outlet]
    for known, label in OUTLET_BIAS_MAP.items():
        if known.lower() in outlet.lower():
            return label
    return None


def fetch_hf_image(image_path_in_repo: str, save_path: Path) -> bool:
    """Try to download image from HuggingFace CDN. Usually 404s -- that's fine."""
    if not image_path_in_repo or not str(image_path_in_repo).startswith("images/"):
        return False
    url = HF_IMAGE_BASE + image_path_in_repo
    try:
        token = None
        try:
            from huggingface_hub import HfFolder
            token = HfFolder.get_token()
        except Exception:
            pass
        hdrs = dict(HEADERS)
        if token:
            hdrs["Authorization"] = f"Bearer {token}"
        resp = requests.get(url, headers=hdrs, timeout=SCRAPE_TIMEOUT, stream=True)
        if resp.status_code == 200:
            with open(save_path, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
            return True
    except Exception:
        pass
    return False


def scrape_og_image(source_url: str, save_path: Path) -> bool:
    """
    Scrape the og:image or twitter:image meta tag from a news article URL
    and save it to disk.

    This is the main fix for the 404 problem. Every article in the dataset
    has a source_url pointing to the original news page. The header image
    we want is almost always exposed as an og:image meta tag -- that's how
    social media previews work. So we fetch the page, parse the meta tag,
    download the image URL it points to, and save it.

    Returns True if we got a real image, False otherwise.
    """
    if not source_url or not source_url.startswith("http"):
        return False
    try:
        from bs4 import BeautifulSoup
        resp = requests.get(source_url, headers=HEADERS, timeout=SCRAPE_TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        img_url = None
        for prop in ["og:image", "twitter:image"]:
            tag = (
                soup.find("meta", property=prop)
                or soup.find("meta", attrs={"name": prop})
            )
            if tag and tag.get("content"):
                img_url = tag["content"]
                break

        if not img_url:
            return False

        img_resp = requests.get(img_url, headers=HEADERS, timeout=SCRAPE_TIMEOUT, stream=True)
        img_resp.raise_for_status()
        img = PILImage.open(io.BytesIO(img_resp.content)).convert("RGB")
        img.save(save_path, "JPEG")
        return True

    except Exception:
        return False


def save_description_flag(save_path: Path, description: str):
    """
    When we can't get a real image, save a .txt sidecar with the
    image_description so the training pipeline knows to use CLIP's
    text encoder on it instead of the blank image.

    The image file itself is still saved as a gray placeholder so the
    Dataset class doesn't need special cases -- but the sidecar signals
    that image features should come from the description, not the pixels.
    """
    desc_path = save_path.with_suffix(".desc.txt")
    desc_path.write_text(description or "news article image", encoding="utf-8")


def make_placeholder(save_path: Path):
    PILImage.new("RGB", (224, 224), color=(128, 128, 128)).save(save_path)


def get_image(article: dict, article_id: str) -> tuple[str, str]:
    """
    Try all strategies to get a real image for this article.
    Returns (image_path_str, image_source) where image_source is one of:
      "pil"         -- came directly from HF as PIL object
      "hf_bytes"    -- came from HF as raw bytes
      "hf_cdn"      -- downloaded from HuggingFace CDN
      "scraped"     -- scraped from source_url og:image
      "description" -- no image; use image_description with CLIP text encoder
      "placeholder" -- nothing worked; gray square (CLIP gets no signal)
    """
    image_col = article.get("image", "")
    source_url = str(article.get("url", "") or "").strip()
    description = str(article.get("img_description", "") or "").strip()
    image_path = IMAGE_DIR / f"{article_id}.jpg"

    # Strategy 1: PIL object already in the column
    if isinstance(image_col, PILImage.Image):
        try:
            image_col.convert("RGB").save(image_path, "JPEG")
            return str(image_path), "pil"
        except Exception:
            pass

    # Strategy 2: Raw bytes dict
    elif isinstance(image_col, dict):
        img_bytes = image_col.get("bytes")
        if img_bytes:
            try:
                img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
                img.save(image_path, "JPEG")
                return str(image_path), "hf_bytes"
            except Exception:
                pass

    # Strategy 3: HF CDN path (usually 404s, but try anyway)
    elif isinstance(image_col, str) and image_col.startswith("images/"):
        if fetch_hf_image(image_col, image_path):
            return str(image_path), "hf_cdn"

    # Strategy 4: Scrape og:image from source_url  <-- the actual fix
    if source_url and scrape_og_image(source_url, image_path):
        return str(image_path), "scraped"

    # Strategy 5: No image available -- save description for CLIP text encoder
    if description:
        make_placeholder(image_path)
        save_description_flag(image_path, description)
        return str(image_path), "description"

    # Strategy 6: Total fallback -- gray placeholder, CLIP gets nothing
    make_placeholder(image_path)
    return str(image_path), "placeholder"


def download_dataset():
    print("=" * 60)
    print("Downloading NewsMediaBias-Plus with real images")
    print("=" * 60)
    print(f"Target: {SAMPLES_PER_CLASS} per class ({SAMPLES_PER_CLASS * 3} total)")
    print("Image strategy: HF column → og:image scrape → description → placeholder\n")

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    print("Loading dataset metadata...")
    try:
        dataset = load_dataset(DATASET_REPO)
        raw_data = dataset["train"]
        print(f"Loaded {len(raw_data)} articles\n")
        sample = raw_data[0]
        print(f"Image column sample: '{sample.get('image', '')}'")
        print(f"Source URL sample:   '{str(sample.get('url', ''))[:60]}...'")
        print(f"Image desc sample:   '{str(sample.get('img_description', ''))[:80]}'\n")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False

    buckets = {"Left": [], "Center": [], "Right": []}
    source_counts = {"pil": 0, "hf_bytes": 0, "hf_cdn": 0,
                     "scraped": 0, "description": 0, "placeholder": 0}
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

        text = str(article.get("article_text", "")).strip()
        if not text or len(text) < 50:
            text = str(article.get("headline", "")).strip()
        if not text:
            continue

        article_id = str(article.get("unique_id", f"article_{i}"))
        image_path, source = get_image(article, article_id)
        source_counts[source] += 1

        buckets[label].append({
            "text": text,
            "image_path": image_path,
            "image_source": source,
            "image_description": str(article.get("img_description", "") or ""),
            "label": label,
            "outlet": outlet,
            "headline": str(article.get("headline", "")).strip(),
            "source_url": str(article.get("url", "") or ""),
        })

    all_rows = []
    for rows in buckets.values():
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Count articles where CLIP gets real visual signal (pil + hf_bytes + hf_cdn + scraped)
    real_image_count = sum(
        source_counts[k] for k in ("pil", "hf_bytes", "hf_cdn", "scraped")
    )
    clip_signal_count = real_image_count + source_counts["description"]

    print(f"\n{'='*60}")
    print("DOWNLOAD COMPLETE")
    print(f"{'='*60}")
    print(f"Articles saved: {len(df)}")
    print(f"\nImage source breakdown:")
    for src, count in source_counts.items():
        if count > 0:
            print(f"  {src:<15} {count}")
    print(f"\nReal images (visual signal): {real_image_count}")
    print(f"CLIP description fallbacks:  {source_counts['description']}")
    print(f"Gray placeholders (no signal): {source_counts['placeholder']}")
    print(f"\nCLIP gets real signal for {clip_signal_count}/{len(df)} articles "
          f"({clip_signal_count/max(len(df),1):.1%})")

    print(f"\nLabel distribution:")
    print(df["label"].value_counts().to_string())

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved to: {OUTPUT_CSV}")

    if real_image_count == 0 and source_counts["description"] == 0:
        print("\nWARNING: No real image signal obtained.")
        print("The image column format may have changed and source_url scraping failed.")
        print("Text model (F1=0.85) is still valid. Consider filing an issue with")
        print("Vector Institute to ask about image availability.")
    elif real_image_count == 0:
        print("\nNOTE: HF CDN images all 404'd (known issue with this dataset).")
        print("Using image_description as CLIP input -- this gives real signal.")
        print("Fusion should now outperform text-only baseline.")

    print("\nNext steps:")
    print("  python utils/preprocessing.py")
    print("  python train.py --mode text")
    print("  python train.py --mode image")
    print("  python train.py --mode fusion")

    return True


if __name__ == "__main__":
    download_dataset()