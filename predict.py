# predict.py
#
# Run the trained system on a new article.
#
# Usage:
#     python predict.py --text "Article text here" --image path/to/image.jpg
#     python predict.py --url https://example.com/article  (fetches text + image)

import argparse
import torch
import numpy as np
from PIL import Image
from transformers import RobertaTokenizer

from utils.preprocessing import clean_text, get_image_transforms, load_image, MAX_TEXT_LENGTH
from models.text_model import load_text_model
from models.image_model import load_image_model
from models.fusion_model import predict, weighted_fusion
import json


LABEL_NAMES = ["Left", "Center", "Right"]


def predict_article(text: str, image_path: str = None):
    """
    Run the full bias prediction on one article.

    Loads both models, runs text through RoBERTa and the image through CLIP,
    then combines them with weighted fusion and checks for mismatch. If no
    image is provided I use a uniform 1/3 distribution, which effectively
    makes it text-only -- the fusion still weights text at 0.7 so the
    Center pull from the uniform image is small.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    text_model = load_text_model("models/text_model.pt", device)
    image_model = load_image_model("models/image_model.pt", device)
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    # ── Text prediction ──
    cleaned = clean_text(text)
    encoding = tokenizer(
        cleaned,
        padding="max_length",
        truncation=True,
        max_length=MAX_TEXT_LENGTH,
        return_tensors="pt"
    )

    with torch.no_grad():
        text_probs = text_model.get_probabilities(
            encoding["input_ids"].to(device),
            encoding["attention_mask"].to(device)
        ).cpu().squeeze().numpy()

    # ── Image prediction ──
    if image_path:
        transform = get_image_transforms("val")
        image_tensor = load_image(image_path, transform).unsqueeze(0)

        with torch.no_grad():
            image_probs = image_model.get_probabilities(
                image_tensor.to(device)
            ).cpu().squeeze().numpy()
    else:
        # No image -- uniform distribution so text carries all the weight
        image_probs = np.array([1/3, 1/3, 1/3])
        print("No image provided. Using text-only prediction.")

    # ── Fusion and mismatch ──
    result = predict(text_probs, image_probs)

    # Add explanation hints
    result["explanation"] = generate_explanation(text, result)

    return result


def generate_explanation(text: str, result: dict) -> dict:
    """
    Simple keyword-based explanation hints -- not real explainability.
    I'll replace this with LIME or SHAP eventually, but this at least gives
    the user some idea of what words might be driving the prediction.
    """
    text_lower = text.lower()

    left_signals = [
        "families", "community", "rights", "justice", "equality",
        "climate", "healthcare", "undocumented", "asylum seekers"
    ]
    right_signals = [
        "border", "illegal", "enforcement", "security", "crime",
        "radical", "taxpayer", "freedom", "law and order"
    ]

    found_left = [w for w in left_signals if w in text_lower]
    found_right = [w for w in right_signals if w in text_lower]

    text_hint = ""
    if found_left:
        text_hint += f"Left-leaning language detected: {', '.join(found_left[:3])}. "
    if found_right:
        text_hint += f"Right-leaning language detected: {', '.join(found_right[:3])}."
    if not text_hint:
        text_hint = "No strong directional language signals detected."

    image_hint = "Image analysis based on visual content and composition."
    if result.get("mismatch"):
        image_hint = (
            f"Image leans {result['mismatch_detail']['image_label']} "
            f"while text leans {result['mismatch_detail']['text_label']}. "
            "Cross-modal mismatch detected."
        )

    return {"text": text_hint, "image": image_hint}


def format_output(result: dict) -> str:
    """Format the prediction result for terminal output."""
    lines = [
        "\n" + "="*55,
        "BIAS PREDICTION RESULT",
        "="*55,
        f"Bias label:    {result['bias']}",
        f"Confidence:    {result['confidence']:.1%}",
        "",
        "Score breakdown:",
        f"  Text  →  Left: {result['text_score']['Left']:.3f}  "
        f"Center: {result['text_score']['Center']:.3f}  "
        f"Right: {result['text_score']['Right']:.3f}",
        f"  Image →  Left: {result['image_score']['Left']:.3f}  "
        f"Center: {result['image_score']['Center']:.3f}  "
        f"Right: {result['image_score']['Right']:.3f}",
        "",
        f"Cross-modal mismatch: {'YES' if result['mismatch'] else 'No'}",
    ]

    if result["mismatch"]:
        d = result["mismatch_detail"]
        lines.append(f"  → {d['description']} (severity: {d['severity']:.3f})")

    lines += [
        "",
        "Explanation:",
        f"  Text:  {result['explanation']['text']}",
        f"  Image: {result['explanation']['image']}",
        "="*55,
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True, help="Article text")
    parser.add_argument("--image", type=str, default=None, help="Path to image file")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    result = predict_article(args.text, args.image)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(format_output(result))
