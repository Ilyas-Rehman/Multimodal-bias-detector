# utils/preprocessing.py
#
# This is where I prep the raw data before any training happens.
# Turns out this is way more work than I expected -- the raw articles
# from the web have HTML tags, broken URLs, and weird unicode all over them.
# Also had to figure out how to size and normalize images consistently.
#
# Main steps:
#   - Clean messy text (strip HTML, URLs, weird characters)
#   - Tokenize text so RoBERTa can read it (text → numbers)
#   - Resize and normalize images to 224x224
#   - Split everything into train / val / test
#
# Run directly to process raw data:
#     python utils/preprocessing.py
#
# Probably the most important lesson so far: bad preprocessing = bad results.
# Even a great model can't do much with garbage input.

import os
import re
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, CLIPProcessor
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


# ── Label mapping ─────────────────────────────────────────────────────────────
# ML models need numbers, not strings. So I map:
#   Left   → 0
#   Center → 1
#   Right  → 2
# This is "label encoding" -- one of the first things I learned about.

LABEL_MAP = {"Left": 0, "Center": 1, "Right": 2}
LABEL_NAMES = ["Left", "Center", "Right"]

# 128 tokens is enough for a headline + opening paragraph, and saves a ton
# of memory compared to the default 512. This was really important when
# training on CPU -- 512 was causing OOM errors.
MAX_TEXT_LENGTH = 128

# Standard size for vision models -- ResNet, CLIP, basically everything
# expects 224x224 so I just went with that.
IMAGE_SIZE = 224


# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Strip the junk out of raw article text.

    I ran a few raw articles through the tokenizer without cleaning first
    and the output was full of <div> tags and random URLs that obviously
    don't carry any political signal. The regex patterns here were
    confusing at first -- <[^>]+> means "match anything inside angle
    brackets", which catches all HTML tags. The rest handle URLs and
    non-standard characters.
    """
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^\w\s.,!?;:'\"-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_text(texts: list, tokenizer, max_length=MAX_TEXT_LENGTH):
    """
    Turn a list of strings into the token IDs that RoBERTa expects.

    I didn't really get tokenization at first. The model has a vocabulary
    of ~50k subword pieces and each one maps to a number. So "running"
    might become two tokens: "run" and "ning". padding="max_length" pads
    shorter sequences with [PAD] tokens so everything is the same length --
    needed to batch them together into a tensor. truncation=True just cuts
    off anything longer than max_length.
    """
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )


# ── Image transforms ──────────────────────────────────────────────────────────

def get_image_transforms(split="train"):
    """
    Set up image preprocessing -- different for train vs. val/test.

    For training I use data augmentation -- random crops and flips. This
    helps the model not just memorize exact images, and effectively makes
    the dataset bigger. Took me a while to understand why you wouldn't
    also do this at eval time -- but the point is you want consistent,
    reproducible results when checking accuracy, not random ones.

    The ImageNet mean/std values ([0.485...], [0.229...]) are the standard
    normalization for pretrained models. I copied these from the PyTorch
    docs and apparently they matter because the pretrained weights expect
    inputs normalized this way.
    """
    if split == "train":
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


def load_image(image_path: str, transform=None) -> torch.Tensor:
    """
    Load one image from disk and apply transforms.

    Some images in the dataset are corrupted or just missing entirely.
    I learned the hard way that one bad image can crash a whole training
    run, so I wrapped this in try/except. If an image fails it returns
    a black tensor and keeps going. Not ideal, but better than losing
    hours of training over one broken file.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        if transform:
            img = transform(img)
        return img
    except Exception as e:
        print(f"Warning: could not load image {image_path}: {e}")
        return torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)


# ── Dataset class ─────────────────────────────────────────────────────────────

class BiasDataset(Dataset):
    """
    Custom PyTorch Dataset for the multimodal bias data.

    Each sample gives back:
        input_ids, attention_mask  -- tokenized text for RoBERTa
        image                      -- preprocessed image tensor (or zeros if using desc fallback)
        pixel_values               -- same as image (alias for fusion model compatibility)
        clip_text_ids              -- CLIP-tokenized image description (used when no real image)
        use_image_flag             -- 1.0 = real image available, 0.0 = use description fallback
        label                      -- 0=Left, 1=Center, 2=Right
        raw_text                   -- keeping the original text for explainability later

    The use_image_flag is the key addition. When get_data.py couldn't get a
    real image (HF CDN 404'd and og:image scraping also failed), it saves
    the image_description to a .desc.txt sidecar file next to the image.
    We check for that sidecar here and route to CLIP's text encoder instead
    of the vision encoder, so CLIP gets real signal either way.
    """

    def __init__(self, dataframe, tokenizer, image_transform,
                 text_col="text", image_col="image_path", label_col="label"):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.text_col = text_col
        self.image_col = image_col
        self.label_col = label_col
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # Tokenize all article text upfront -- much faster than per-sample during training
        texts = [clean_text(str(t)) for t in self.data[text_col]]
        self.encodings = tokenize_text(texts, tokenizer)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = Path(str(self.data[self.image_col][idx]))
        desc_path = image_path.with_suffix(".desc.txt")

        if desc_path.exists():
            # No real image available -- use image_description with CLIP text encoder.
            # get_data.py writes this sidecar when og:image scraping fails but
            # image_description exists. CLIP's text encoder still gives meaningful
            # visual semantics from the description (e.g. "protesters march outside
            # capitol building" encodes differently than "stock chart trending up").
            description = desc_path.read_text(encoding="utf-8").strip()
            clip_inputs = self.clip_processor(
                text=description,
                return_tensors="pt",
                padding="max_length",
                max_length=77,
                truncation=True,
            )
            pixel_values = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
            clip_text_ids = clip_inputs["input_ids"].squeeze(0)
            use_image_flag = torch.tensor(0, dtype=torch.float)
        else:
            # Real image -- load and transform normally for CLIP vision encoder
            pixel_values = load_image(str(image_path), self.image_transform)
            clip_text_ids = torch.zeros(77, dtype=torch.long)
            use_image_flag = torch.tensor(1, dtype=torch.float)

        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "image": pixel_values,          # existing train.py expects this key
            "pixel_values": pixel_values,   # fusion model uses this key
            "clip_text_ids": clip_text_ids,
            "use_image_flag": use_image_flag,
            "label": torch.tensor(
                LABEL_MAP[self.data[self.label_col][idx]],
                dtype=torch.long
            ),
            "raw_text": self.data[self.text_col][idx]
        }


# ── Loading and splitting data ────────────────────────────────────────────────

def load_raw_data(data_dir="data/raw") -> pd.DataFrame:
    """
    Load raw data from data/raw. Looks for a CSV or JSON with columns:
    text, image_path, label. Falls back to sample data if nothing is there --
    useful for testing the pipeline without downloading everything first.
    """
    data_path = Path(data_dir)
    csv_files = list(data_path.glob("*.csv"))
    json_files = list(data_path.glob("*.json"))

    if csv_files:
        df = pd.read_csv(csv_files[0])
        print(f"Loaded {len(df)} samples from {csv_files[0]}")
    elif json_files:
        df = pd.read_json(json_files[0])
        print(f"Loaded {len(df)} samples from {json_files[0]}")
    else:
        print("No data found in data/raw. Creating sample data for testing.")
        df = create_sample_data()

    required = ["text", "image_path", "label"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: '{col}'")

    valid_labels = set(LABEL_MAP.keys())
    before = len(df)
    df = df[df["label"].isin(valid_labels)]
    if len(df) < before:
        print(f"Dropped {before - len(df)} rows with unrecognized labels.")

    return df


def create_sample_data() -> pd.DataFrame:
    """
    18 hardcoded examples so I can test the pipeline without real data.

    Writing these actually helped me understand how bias shows up in language --
    things like "undocumented immigrants" vs "illegal aliens", or whether the
    framing emphasizes victims or perpetrators. These are based on examples
    from the ViLBias paper and some Fox/CNN headlines I looked at.
    Obviously 18 samples is nowhere near enough to actually train a model --
    run get_data.py for the real thing.
    """
    print("NOTE: Using sample data. For real training, run: python get_data.py")
    samples = [
        # Left-leaning: humanizing language, social justice framing
        {"text": "Families torn apart by harsh immigration policies face impossible choices as communities suffer.", "image_path": "data/raw/placeholder.jpg", "label": "Left"},
        {"text": "Activists demand justice as undocumented immigrants are separated from their children at the border.", "image_path": "data/raw/placeholder.jpg", "label": "Left"},
        {"text": "Healthcare workers rally for universal coverage as millions remain uninsured under current system.", "image_path": "data/raw/placeholder.jpg", "label": "Left"},
        {"text": "Climate scientists warn of catastrophic consequences as government fails to act on emissions.", "image_path": "data/raw/placeholder.jpg", "label": "Left"},
        {"text": "Community organizers fight for affordable housing as families face eviction crisis.", "image_path": "data/raw/placeholder.jpg", "label": "Left"},
        {"text": "Protesters demand racial justice reforms as police brutality cases continue to mount.", "image_path": "data/raw/placeholder.jpg", "label": "Left"},
        # Center: balanced language, factual tone, both sides mentioned
        {"text": "Officials announced new border security measures today following bipartisan discussions.", "image_path": "data/raw/placeholder.jpg", "label": "Center"},
        {"text": "The committee reviewed both economic and environmental impacts before reaching a decision.", "image_path": "data/raw/placeholder.jpg", "label": "Center"},
        {"text": "Lawmakers on both sides of the aisle met to discuss the proposed infrastructure bill.", "image_path": "data/raw/placeholder.jpg", "label": "Center"},
        {"text": "The report found mixed results on the effectiveness of the new education policy.", "image_path": "data/raw/placeholder.jpg", "label": "Center"},
        {"text": "Analysts say the economic data presents a complex picture with both gains and challenges.", "image_path": "data/raw/placeholder.jpg", "label": "Center"},
        {"text": "The governor signed the legislation after months of negotiation between competing interests.", "image_path": "data/raw/placeholder.jpg", "label": "Center"},
        # Right-leaning: law enforcement framing, security emphasis, loaded terms
        {"text": "Illegal immigration surges as the administration fails to secure the border from invaders.", "image_path": "data/raw/placeholder.jpg", "label": "Right"},
        {"text": "Radical left policies destroying small businesses as socialist agenda threatens free market.", "image_path": "data/raw/placeholder.jpg", "label": "Right"},
        {"text": "Law enforcement officers overwhelmed as crime spikes in Democrat-run sanctuary cities.", "image_path": "data/raw/placeholder.jpg", "label": "Right"},
        {"text": "Second amendment rights under attack as Democrats push unconstitutional gun control measures.", "image_path": "data/raw/placeholder.jpg", "label": "Right"},
        {"text": "Taxpayers foot the bill for reckless spending as national debt reaches record levels.", "image_path": "data/raw/placeholder.jpg", "label": "Right"},
        {"text": "Parents fight back against radical curriculum as school boards push divisive ideology on children.", "image_path": "data/raw/placeholder.jpg", "label": "Right"},
    ]
    return pd.DataFrame(samples)


def split_data(df: pd.DataFrame, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    Split data into train / val / test.

    Standard 80/10/10 split:
    - Train  : what the model actually learns from
    - Val    : I check this during training to see if I'm overfitting
    - Test   : only touch this at the very end to get final numbers

    I kept confusing val and test at first. The key is: test set is
    completely held out until the end -- you don't use it to make any
    decisions about the model. stratify= keeps the class balance the
    same across all three splits, which matters with an uneven dataset.
    Falls back to a simple split if any class has too few samples to stratify.
    """
    min_class_size = df["label"].value_counts().min()

    if min_class_size < 3:
        print(f"Warning: only {min_class_size} sample(s) per class. Using simple split.")
        train, temp = train_test_split(df, test_size=(1 - train_ratio), random_state=seed)
        val, test = train_test_split(temp, test_size=0.5, random_state=seed)
    else:
        train, temp = train_test_split(
            df, test_size=(1 - train_ratio),
            stratify=df["label"], random_state=seed
        )
        val_size = val_ratio / (1 - train_ratio)
        if temp["label"].value_counts().min() < 2:
            val, test = train_test_split(temp, test_size=0.5, random_state=seed)
        else:
            val, test = train_test_split(
                temp, test_size=(1 - val_size),
                stratify=temp["label"], random_state=seed
            )

    print(f"Split sizes -- Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    return train, val, test


def save_splits(train, val, test, out_dir="data/processed"):
    """Save the three splits as CSVs so I don't have to re-split every time I run things."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    train.to_csv(f"{out_dir}/train.csv", index=False)
    val.to_csv(f"{out_dir}/val.csv", index=False)
    test.to_csv(f"{out_dir}/test.csv", index=False)
    print(f"Saved splits to {out_dir}/")


# ── Run directly ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading raw data...")
    df = load_raw_data()

    print("\nLabel distribution:")
    print(df["label"].value_counts())

    print("\nSplitting data...")
    train, val, test = split_data(df)

    print("\nSaving splits...")
    save_splits(train, val, test)

    print("\nPreprocessing complete.")
    print("Next: python train.py --mode text")