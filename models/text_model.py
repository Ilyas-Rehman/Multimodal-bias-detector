# models/text_model.py
#
# Fine-tunes RoBERTa to classify political bias as Left / Center / Right.
# I picked RoBERTa over plain BERT because it was trained longer with more
# data and consistently outperforms BERT on text classification benchmarks.
# roberta-base is manageable on CPU -- the large version is too slow.

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import RobertaModel, RobertaTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np

NUM_CLASSES = 3
MODEL_NAME = "roberta-base"

# Had to drop MAX_LEN from 512 to 128 to get this running on CPU without
# running out of memory. 128 still covers a headline + first paragraph,
# which is where most of the bias signal is anyway.
MAX_LEN = 128   # default was 512 — 128 uses ~75% less memory, fine for headlines


class TextBiasClassifier(nn.Module):
    """
    RoBERTa with a small classification head on top.

    The way this works: RoBERTa produces a vector for every token, but
    I only use the [CLS] token (index 0), which is supposed to be a
    summary of the whole sequence. Then I pass it through two linear
    layers to get 3 class scores. Two layers gives a bit more capacity
    than one without being too heavy.

    Architecture:
        RoBERTa → [CLS] vector (768-dim) → Dropout → Linear(768→256)
        → ReLU → Dropout → Linear(256→3)
    """

    def __init__(self, num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        from transformers import RobertaConfig
        # Downloads ~500MB of pretrained weights the first time
        config = RobertaConfig.from_pretrained(MODEL_NAME)
        self.roberta = RobertaModel.from_pretrained(MODEL_NAME, config=config)
        hidden_size = self.roberta.config.hidden_size  # 768 for roberta-base

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # [CLS] is always the first token -- it's the sentence-level summary
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits

    def get_probabilities(self, input_ids, attention_mask):
        """Run forward pass and return softmax probabilities instead of raw logits."""
        logits = self.forward(input_ids, attention_mask)
        return torch.softmax(logits, dim=-1)


# ── Training ──────────────────────────────────────────────────────────────────

def train_text_model(
    train_dataset,
    val_dataset,
    epochs=3,
    batch_size=16,
    learning_rate=2e-5,
    device=None,
    save_path="models/text_model.pt"
):
    """
    Training loop for the text classifier.

    A few things I figured out writing this:
    - AdamW is the standard optimizer for transformers (weight_decay helps regularize)
    - The LR warmup schedule is important -- starting with a big LR damages the
      pretrained weights right away, which you don't want
    - I save based on val F1 instead of val loss because F1 is what actually
      matters for a 3-class classification problem
    - patience=3 means I stop early if F1 hasn't improved in 3 epochs
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = TextBiasClassifier().to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    criterion = nn.CrossEntropyLoss()
    best_val_f1 = 0
    patience = 3
    patience_counter = 0

    for epoch in range(epochs):
        # ── Training pass ──
        model.train()
        train_loss = 0
        all_preds, all_labels = [], []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()

            # Without gradient clipping the loss sometimes explodes to NaN
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        train_acc = np.mean(np.array(all_preds) == np.array(all_labels))

        # ── Validation pass ──
        model.eval()
        val_loss = 0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [val]"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                preds = torch.argmax(logits, dim=-1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = np.mean(np.array(val_preds) == np.array(val_labels))

        from sklearn.metrics import f1_score
        val_f1 = f1_score(val_labels, val_preds, average="macro")

        print(f"\nEpoch {epoch+1}: train_loss={train_loss/len(train_loader):.4f} "
              f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")

        # Save the best checkpoint and stop early if val F1 stops improving
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best model (val_f1={val_f1:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    print(f"\nTraining complete. Best val F1: {best_val_f1:.4f}")
    return model


def load_text_model(model_path="models/text_model.pt", device=None):
    """Load a trained text model from disk for inference."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextBiasClassifier()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model
