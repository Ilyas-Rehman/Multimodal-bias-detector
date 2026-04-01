# models/image_model.py
#
# Image-only bias classifier using CLIP as the feature extractor.
#
# I chose CLIP over plain ResNet because CLIP was trained on image-text pairs,
# so it understands what objects and scenes *mean* semantically, not just raw
# pixel patterns. The ViLBias paper tested both and CLIP-based models came out
# ahead by a few F1 points on image-only classification, which convinced me.

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np

NUM_CLASSES = 3

# ViT-B/32 gives 512-dim vectors -- looked this up in the open_clip docs
CLIP_EMBED_DIM = 512


class ImageBiasClassifier(nn.Module):
    """
    Frozen CLIP image encoder with a small trainable classification head.

    I keep CLIP frozen (freeze_clip=True) because:
    1. Our dataset is too small to fine-tune a model that big without overfitting
    2. Training is much faster when we're only updating the small head
    3. CLIP's features are already really good -- we just need to learn
       how to map them to Left/Center/Right

    Architecture:
        CLIP ViT-B/32 → 512-dim embedding → Dropout
        → Linear(512→256) → ReLU → Dropout → Linear(256→3)
    """

    def __init__(self, num_classes=NUM_CLASSES, dropout=0.3, freeze_clip=True):
        super().__init__()

        # Load CLIP -- first time downloads weights (~300MB)
        try:
            import open_clip
            self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai"
            )
        except ImportError:
            raise ImportError("Install open-clip-torch: pip install open-clip-torch")

        # Freeze CLIP so we don't accidentally destroy the pretrained representations
        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(CLIP_EMBED_DIM, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, images):
        """
        images: (batch, 3, 224, 224) tensor
        returns: logits of shape (batch, 3)
        """
        with torch.no_grad() if not self.training else torch.enable_grad():
            image_features = self.clip_model.encode_image(images)

        # L2-normalize the CLIP features -- standard practice and I found it
        # helps training stability. Without it the feature magnitudes vary a lot.
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = self.classifier(image_features.float())
        return logits

    def get_probabilities(self, images):
        """Run forward pass and return softmax probabilities instead of raw logits."""
        logits = self.forward(images)
        return torch.softmax(logits, dim=-1)


# ── Training ──────────────────────────────────────────────────────────────────

def train_image_model(
    train_dataset,
    val_dataset,
    epochs=5,
    batch_size=32,
    learning_rate=1e-4,
    device=None,
    save_path="models/image_model.pt"
):
    """
    Training loop for the image classifier.

    I use a higher LR here (1e-4) than in the text model (2e-5) because
    we're only training the small classification head, not the whole encoder.
    When the encoder is frozen you can afford a bigger LR -- I tried 2e-5
    and it was too slow to converge.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = ImageBiasClassifier().to(device)

    # Only pass the classifier head parameters to the optimizer -- CLIP is frozen
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05 * total_steps),
        num_training_steps=total_steps
    )

    criterion = nn.CrossEntropyLoss()
    best_val_f1 = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        all_preds, all_labels = [], []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        train_acc = np.mean(np.array(all_preds) == np.array(all_labels))

        model.eval()
        val_preds, val_labels = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [val]"):
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                logits = model(images)
                preds = torch.argmax(logits, dim=-1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = np.mean(np.array(val_preds) == np.array(val_labels))
        from sklearn.metrics import f1_score
        val_f1 = f1_score(val_labels, val_preds, average="macro")

        print(f"\nEpoch {epoch+1}: train_loss={train_loss/len(train_loader):.4f} "
              f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best model (val_f1={val_f1:.4f})")

    print(f"\nTraining complete. Best val F1: {best_val_f1:.4f}")
    return model


def load_image_model(model_path="models/image_model.pt", device=None):
    """Load a trained image model from disk for inference."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageBiasClassifier()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model
