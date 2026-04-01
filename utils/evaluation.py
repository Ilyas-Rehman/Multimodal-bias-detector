# utils/evaluation.py
#
# All the metrics I'm tracking: accuracy, precision, recall, F1, Cohen's Kappa,
# and a confusion matrix. Also has the cross-modal mismatch report.
#
# I didn't know about Cohen's Kappa before this project -- it's a better
# measure than raw accuracy when classes might be imbalanced, because it
# accounts for how often you'd get the right answer just by guessing.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, cohen_kappa_score, classification_report
)

LABEL_NAMES = ["Left", "Center", "Right"]


def compute_metrics(y_true, y_pred):
    """
    Compute accuracy, precision, recall, F1, and Cohen's Kappa.

    I'm using macro-averaged F1 as the main metric because the classes
    might not be perfectly balanced. Macro treats each class equally,
    which makes sense here -- we care about getting Left, Center, and Right
    all right, not just predicting the most common one. Per-class numbers
    are also useful for seeing which direction the model struggles with most.
    """
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    kappa = cohen_kappa_score(y_true, y_pred)

    # Per-class metrics
    p_cls, r_cls, f_cls, support = precision_recall_fscore_support(
        y_true, y_pred, average=None,
        labels=[0, 1, 2], zero_division=0
    )
    per_class = {}
    for i, name in enumerate(LABEL_NAMES):
        per_class[name] = {
            "precision": round(p_cls[i], 4),
            "recall": round(r_cls[i], 4),
            "f1": round(f_cls[i], 4),
            "support": int(support[i])
        }

    return {
        "accuracy": round(acc, 4),
        "precision_macro": round(precision, 4),
        "recall_macro": round(recall, 4),
        "f1_macro": round(f1, 4),
        "cohen_kappa": round(kappa, 4),
        "per_class": per_class
    }


def print_metrics(metrics: dict):
    """Print all the metrics in a readable format."""
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy:         {metrics['accuracy']:.4f}")
    print(f"Precision (macro):{metrics['precision_macro']:.4f}")
    print(f"Recall (macro):   {metrics['recall_macro']:.4f}")
    print(f"F1 (macro):       {metrics['f1_macro']:.4f}")
    print(f"Cohen's Kappa:    {metrics['cohen_kappa']:.4f}")
    print(f"\n  Kappa interpretation:")
    k = metrics["cohen_kappa"]
    if k < 0.20:
        print(f"  {k:.2f} → Slight agreement")
    elif k < 0.40:
        print(f"  {k:.2f} → Fair agreement")
    elif k < 0.60:
        print(f"  {k:.2f} → Moderate agreement")
    elif k < 0.80:
        print(f"  {k:.2f} → Substantial agreement")
    else:
        print(f"  {k:.2f} → Almost perfect agreement")

    print("\nPer-class breakdown:")
    print(f"{'Label':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 50)
    for label, vals in metrics["per_class"].items():
        print(f"{label:<10} {vals['precision']:>10.4f} {vals['recall']:>10.4f} "
              f"{vals['f1']:>10.4f} {vals['support']:>10}")
    print("="*50)


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Plot a confusion matrix -- both raw counts and row-normalized percentages.

    I find the normalized version more useful because it shows how often each
    class gets predicted correctly regardless of how many samples it has.
    Showing both side by side was an idea I got from a tutorial and I liked it.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=axes[0])
    axes[0].set_title("Confusion Matrix (counts)")
    axes[0].set_ylabel("True label")
    axes[0].set_xlabel("Predicted label")

    # Normalized
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=axes[1])
    axes[1].set_title("Confusion Matrix (normalized)")
    axes[1].set_ylabel("True label")
    axes[1].set_xlabel("Predicted label")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")
    plt.show()


# ── Cross-modal mismatch analysis ─────────────────────────────────────────────

def compute_mismatch(text_preds, image_preds, threshold=0.3):
    """
    Find articles where the text and image are pointing in different directions.

    This is one of the things I'm most interested in -- cases where a headline
    reads neutral but the image is clearly partisan, or vice versa. I only
    flag it as a mismatch if both models are actually confident, otherwise
    there are too many false positives from weak predictions.

    Args:
        text_preds: list of dicts like {"Left": 0.7, "Center": 0.2, "Right": 0.1}
        image_preds: same format for image predictions
        threshold: how confident each model needs to be before I count it as a real lean
    """
    results = []
    for text_p, image_p in zip(text_preds, image_preds):
        text_label = max(text_p, key=text_p.get)
        image_label = max(image_p, key=image_p.get)

        text_conf = text_p[text_label]
        image_conf = image_p[image_label]

        # Only flag mismatch if both are confident about different labels
        is_mismatch = (
            text_label != image_label and
            text_conf > (0.5 + threshold) and
            image_conf > (0.5 + threshold)
        )

        results.append({
            "text_label": text_label,
            "image_label": image_label,
            "text_confidence": round(text_conf, 3),
            "image_confidence": round(image_conf, 3),
            "mismatch": is_mismatch,
            "mismatch_direction": f"{text_label} text / {image_label} image" if is_mismatch else None
        })

    return results


def mismatch_report(mismatch_results: list):
    """Print how many articles have text-image disagreement and in which direction."""
    total = len(mismatch_results)
    mismatches = [r for r in mismatch_results if r["mismatch"]]
    mismatch_rate = len(mismatches) / total if total > 0 else 0

    print("\n" + "="*50)
    print("CROSS-MODAL MISMATCH REPORT")
    print("="*50)
    print(f"Total samples:   {total}")
    print(f"Mismatches:      {len(mismatches)} ({mismatch_rate:.1%})")

    if mismatches:
        print("\nMismatch breakdown:")
        directions = {}
        for r in mismatches:
            d = r["mismatch_direction"]
            directions[d] = directions.get(d, 0) + 1
        for d, count in sorted(directions.items(), key=lambda x: -x[1]):
            print(f"  {d}: {count}")

    print("="*50)
    return mismatch_rate
