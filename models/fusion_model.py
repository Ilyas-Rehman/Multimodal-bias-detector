# models/fusion_model.py
#
# Combines the text and image predictions into a final bias label.
# Also detects cross-modal mismatch -- when the text says one thing
# and the image says another. That's the main thing I'm adding
# beyond what ViLBias already did.
#
# Two ways to fuse:
#   1. Weighted average: 0.7 * text + 0.3 * image (no training needed, fast)
#   2. Learned fusion: small neural net figures out the best combination
#
# I started with weighted average because it's simpler. Learned fusion is
# probably better but needs more labeled data to train reliably.

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple

LABEL_NAMES = ["Left", "Center", "Right"]
# Text gets more weight because ViLBias showed images only add about
# 3-8 F1 points on top of text alone -- the text is carrying most of the signal
TEXT_WEIGHT = 0.7
IMAGE_WEIGHT = 0.3


# ── Weighted average fusion (no training required) ────────────────────────────

def weighted_fusion(
    text_probs: torch.Tensor,
    image_probs: torch.Tensor,
    text_weight: float = TEXT_WEIGHT,
    image_weight: float = IMAGE_WEIGHT
) -> torch.Tensor:
    """
    Combine text and image probabilities with a fixed weighted average.

    Simplest possible fusion strategy and honestly not bad as a baseline.
    text_probs and image_probs are both (batch, 3) tensors of softmax
    probabilities [Left, Center, Right]. Output is the same shape.
    """
    return text_weight * text_probs + image_weight * image_probs


# ── Learned fusion (trained neural net) ──────────────────────────────────────

class LearnedFusionModel(nn.Module):
    """
    A small neural net that learns how to combine text and image predictions.

    I was curious whether the model could learn a better weighting than my
    fixed 0.7/0.3 split. The input is just the concatenated probability
    vectors from both models (6 numbers total), and the output is the
    final 3-class prediction plus a mismatch confidence score.
    """

    def __init__(self, num_classes=3):
        super().__init__()

        self.fusion_net = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

        # Separate head to estimate how likely a mismatch is
        self.mismatch_head = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, text_probs: torch.Tensor, image_probs: torch.Tensor):
        """
        text_probs:  (batch, 3)
        image_probs: (batch, 3)

        returns logits (batch, 3) and mismatch_score (batch, 1)
        higher mismatch_score = model thinks there's a text-image disagreement
        """
        combined = torch.cat([text_probs, image_probs], dim=-1)
        logits = self.fusion_net(combined)
        mismatch_score = self.mismatch_head(combined)
        return logits, mismatch_score


def train_fusion_model(
    text_probs_train, image_probs_train, labels_train,
    text_probs_val, image_probs_val, labels_val,
    epochs=20,
    learning_rate=1e-3,
    save_path="models/fusion_model.pt"
):
    """
    Train the learned fusion model on pre-computed probability outputs.

    This is fast because we're not running RoBERTa or CLIP again -- we
    just feed in the probability vectors we already computed. The whole
    thing trains in seconds even on CPU.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LearnedFusionModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        logits, _ = model(
            text_probs_train.to(device),
            image_probs_train.to(device)
        )
        loss = criterion(logits, labels_train.to(device))
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits, _ = model(
                text_probs_val.to(device),
                image_probs_val.to(device)
            )
            val_preds = torch.argmax(val_logits, dim=-1).cpu()
            val_acc = (val_preds == labels_val).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}: loss={loss.item():.4f} val_acc={val_acc:.4f}")

    print(f"Fusion training complete. Best val acc: {best_val_acc:.4f}")
    return model


# ── Mismatch detection ────────────────────────────────────────────────────────

def detect_mismatch(
    text_probs: np.ndarray,
    image_probs: np.ndarray,
    confidence_threshold: float = 0.6
) -> Dict:
    """
    Check if a single article's text and image are pointing in opposite directions.

    I only call this a mismatch when both models are actually confident --
    if the image model barely leans Right at 0.35, that's probably noise.
    The 0.6 threshold felt right after looking at some examples manually.

    For severity I'm using Jensen-Shannon divergence, which measures how
    different two probability distributions are (0=identical, 1=maximally
    different). I hadn't heard of it before -- turns out it's a symmetric,
    bounded version of KL divergence that doesn't blow up near zero.
    """
    text_label = LABEL_NAMES[np.argmax(text_probs)]
    image_label = LABEL_NAMES[np.argmax(image_probs)]
    text_conf = float(np.max(text_probs))
    image_conf = float(np.max(image_probs))

    is_mismatch = (
        text_label != image_label and
        text_conf >= confidence_threshold and
        image_conf >= confidence_threshold
    )

    # JS divergence as a severity score -- how far apart are the distributions?
    avg = (text_probs + image_probs) / 2
    def kl(p, q): return np.sum(p * np.log(p / (q + 1e-10) + 1e-10))
    js_divergence = (kl(text_probs, avg) + kl(image_probs, avg)) / 2
    severity = float(np.clip(js_divergence, 0, 1))

    return {
        "mismatch": is_mismatch,
        "text_label": text_label,
        "image_label": image_label,
        "text_confidence": round(text_conf, 3),
        "image_confidence": round(image_conf, 3),
        "severity": round(severity, 3),
        "description": (
            f"Text leans {text_label} but image leans {image_label}"
            if is_mismatch else "Text and image agree"
        )
    }


# ── Full prediction pipeline ──────────────────────────────────────────────────

def predict(
    text_probs: np.ndarray,
    image_probs: np.ndarray,
    strategy: str = "weighted"
) -> Dict:
    """
    Run the complete fusion prediction for one article.

    Takes probability arrays from both models and returns a dict with the
    final label, confidence, per-model scores, and mismatch info. The
    learned strategy isn't wired in here yet -- I'd need to load the fusion
    model separately and pass it in, which I might add later.
    """
    if strategy == "weighted":
        fused = TEXT_WEIGHT * text_probs + IMAGE_WEIGHT * image_probs
    else:
        raise ValueError("For 'learned' strategy, use LearnedFusionModel directly.")

    final_label = LABEL_NAMES[np.argmax(fused)]
    confidence = float(np.max(fused))
    mismatch = detect_mismatch(text_probs, image_probs)

    return {
        "bias": final_label,
        "confidence": round(confidence, 3),
        "text_score": {
            "Left": round(float(text_probs[0]), 3),
            "Center": round(float(text_probs[1]), 3),
            "Right": round(float(text_probs[2]), 3)
        },
        "image_score": {
            "Left": round(float(image_probs[0]), 3),
            "Center": round(float(image_probs[1]), 3),
            "Right": round(float(image_probs[2]), 3)
        },
        "mismatch": mismatch["mismatch"],
        "mismatch_detail": mismatch
    }
