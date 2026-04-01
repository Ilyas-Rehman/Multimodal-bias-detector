# train.py
#
# Master training script. Run one phase at a time:
#
#     python train.py --mode text      # Phase 2: train text model
#     python train.py --mode image     # Phase 3: train image model
#     python train.py --mode fusion    # Phase 4: train fusion model
#     python train.py --mode all       # Run all phases in sequence

import argparse
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer

from utils.preprocessing import (
    BiasDataset, load_raw_data, split_data,
    save_splits, get_image_transforms
)
from utils.evaluation import compute_metrics, print_metrics, plot_confusion_matrix
from models.text_model import train_text_model, load_text_model
from models.image_model import train_image_model, load_image_model
from models.fusion_model import weighted_fusion, predict, detect_mismatch

import numpy as np


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_splits():
    """Load the already-processed train/val/test CSVs."""
    train = pd.read_csv("data/processed/train.csv")
    val = pd.read_csv("data/processed/val.csv")
    test = pd.read_csv("data/processed/test.csv")
    return train, val, test


def build_datasets(train_df, val_df, test_df, tokenizer):
    """Wrap the three dataframes into PyTorch Dataset objects."""
    train_ds = BiasDataset(train_df, tokenizer, get_image_transforms("train"))
    val_ds = BiasDataset(val_df, tokenizer, get_image_transforms("val"))
    test_ds = BiasDataset(test_df, tokenizer, get_image_transforms("test"))
    return train_ds, val_ds, test_ds


# ── Phase 1: Prepare data ─────────────────────────────────────────────────────

def phase1_data():
    print("\n" + "="*60)
    print("PHASE 1: DATA PREPARATION")
    print("="*60)

    processed_path = Path("data/processed/train.csv")
    if processed_path.exists():
        print("Processed data already exists. Skipping.")
        return

    df = load_raw_data()
    train, val, test = split_data(df)
    save_splits(train, val, test)
    print("Phase 1 complete.")


# ── Phase 2: Text model ───────────────────────────────────────────────────────

def phase2_text(train_ds, val_ds, test_ds, device):
    print("\n" + "="*60)
    print("PHASE 2: TEXT MODEL (RoBERTa)")
    print("="*60)

    model = train_text_model(
        train_ds, val_ds,
        epochs=3,
        batch_size=2,   # batch_size=2 because anything bigger runs out of RAM on CPU
        learning_rate=2e-5,
        device=device,
        save_path="models/text_model.pt"
    )

    # Check how the model does on the held-out test set
    print("\nEvaluating text model on test set...")
    model.eval()
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]
            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=-1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    print(f"\nTest predictions: {all_preds}")
    print(f"True labels:      {all_labels}")
    metrics = compute_metrics(all_labels, all_preds)
    print_metrics(metrics)
    print("\nNOTE: Scores are meaningless on 18-sample toy data.")
    print("This only confirms the pipeline runs without errors.")

    return model


# ── Phase 3: Image model ──────────────────────────────────────────────────────

def phase3_image(train_ds, val_ds, test_ds, device):
    print("\n" + "="*60)
    print("PHASE 3: IMAGE MODEL (CLIP)")
    print("="*60)

    model = train_image_model(
        train_ds, val_ds,
        epochs=5,
        batch_size=32,
        learning_rate=1e-4,
        device=device,
        save_path="models/image_model.pt"
    )

    # Check how the model does on the held-out test set
    print("\nEvaluating image model on test set...")
    model.eval()
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            labels = batch["label"]
            logits = model(images)
            preds = torch.argmax(logits, dim=-1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    metrics = compute_metrics(all_labels, all_preds)
    print_metrics(metrics)
    plot_confusion_matrix(all_labels, all_preds, save_path="image_confusion_matrix.png")

    return model


# ── Phase 4: Fusion ───────────────────────────────────────────────────────────

def phase4_fusion(text_model, image_model, test_ds, device):
    print("\n" + "="*60)
    print("PHASE 4: FUSION + MISMATCH DETECTION")
    print("="*60)

    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    all_text_probs = []
    all_image_probs = []
    all_labels = []
    mismatch_results = []

    text_model.eval()
    image_model.eval()

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            images = batch["image"].to(device)
            labels = batch["label"]

            text_probs = text_model.get_probabilities(input_ids, attention_mask).cpu()
            image_probs = image_model.get_probabilities(images).cpu()

            all_text_probs.append(text_probs)
            all_image_probs.append(image_probs)
            all_labels.extend(labels.numpy())

            # Check each article in the batch for text-image disagreement
            for i in range(len(labels)):
                mm = detect_mismatch(
                    text_probs[i].numpy(),
                    image_probs[i].numpy()
                )
                mismatch_results.append(mm)

    all_text_probs = torch.cat(all_text_probs, dim=0)
    all_image_probs = torch.cat(all_image_probs, dim=0)

    # Weighted fusion
    fused_probs = weighted_fusion(all_text_probs, all_image_probs)
    fused_preds = torch.argmax(fused_probs, dim=-1).numpy()

    print("\nFUSION RESULTS (weighted 0.7 text / 0.3 image):")
    metrics = compute_metrics(all_labels, fused_preds)
    print_metrics(metrics)
    plot_confusion_matrix(all_labels, fused_preds, save_path="fusion_confusion_matrix.png")

    # Mismatch report
    mismatches = [r for r in mismatch_results if r["mismatch"]]
    total = len(mismatch_results)
    print(f"\nCROSS-MODAL MISMATCH: {len(mismatches)}/{total} articles "
          f"({len(mismatches)/total:.1%}) show text-image disagreement")

    if mismatches:
        print("\nSample mismatches:")
        for r in mismatches[:3]:
            print(f"  {r['description']} "
                  f"(text conf: {r['text_confidence']}, "
                  f"image conf: {r['image_confidence']}, "
                  f"severity: {r['severity']})")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["text", "image", "fusion", "all"],
                        default="all")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # Make sure the data splits exist before trying to train anything
    phase1_data()

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    train_df, val_df, test_df = load_splits()
    train_ds, val_ds, test_ds = build_datasets(train_df, val_df, test_df, tokenizer)

    text_model = None
    image_model = None

    if args.mode in ("text", "all"):
        text_model = phase2_text(train_ds, val_ds, test_ds, device)

    if args.mode in ("image", "all"):
        image_model = phase3_image(train_ds, val_ds, test_ds, device)

    if args.mode in ("fusion", "all"):
        if text_model is None:
            text_model = load_text_model("models/text_model.pt", device)
        if image_model is None:
            image_model = load_image_model("models/image_model.pt", device)
        phase4_fusion(text_model, image_model, test_ds, device)

    print("\nAll phases complete.")
