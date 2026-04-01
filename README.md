# Multimodal News Bias Detector

Detects political bias (Left / Center / Right) in news articles by analyzing
both the article text and its accompanying image. Also flags cross-modal
mismatch — when the image and text lean in different directions.

Based on findings from the ViLBias paper (Vector Institute, 2026).

---

## Setup

```bash
# 1. Clone / download the project
cd bias-detector

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Project structure

```
bias-detector/
├── data/
│   ├── raw/            # Original downloaded datasets
│   └── processed/      # Cleaned, tokenized, ready-to-train data
├── models/
│   ├── text_model.py   # RoBERTa fine-tuning
│   ├── image_model.py  # CLIP image classifier
│   └── fusion_model.py # Combines text + image scores
├── utils/
│   ├── preprocessing.py  # Cleaning, tokenizing, image transforms
│   └── evaluation.py     # Metrics, Cohen's Kappa, confusion matrix
├── notebooks/            # Exploratory analysis
├── app/
│   └── main.py           # Streamlit demo app
├── train.py              # Full training pipeline
├── predict.py            # Run inference on new articles
├── requirements.txt
└── README.md
```

---

## Phase-by-phase workflow

### Phase 1 — Data
```bash
python utils/preprocessing.py
```

### Phase 2 — Text model
```bash
python train.py --mode text
```

### Phase 3 — Image model
```bash
python train.py --mode image
```

### Phase 4 — Fusion model
```bash
python train.py --mode fusion
```

### Phase 5 — Run demo app
```bash
streamlit run app/main.py
```

---

## Sample output

```json
{
  "bias": "Left",
  "confidence": 0.85,
  "text_score": {"Left": 0.80, "Center": 0.10, "Right": 0.10},
  "image_score": {"Left": 0.60, "Center": 0.30, "Right": 0.10},
  "mismatch": false,
  "explanation": {
    "text": "Emotionally charged language, victim-centered framing",
    "image": "Sympathetic imagery, candlelight vigil"
  }
}
```

---

## Dataset sources

- AllSides Media Bias Ratings: https://www.allsides.com/media-bias/ratings
- Media Bias Fact Check: https://mediabiasfactcheck.com
- NewsMediaBias-Plus (HuggingFace): vector-institute/newsmediabias-plus

---

## Key design decisions

- **RoBERTa** for text — strongest small model per ViLBias results
- **CLIP** for images — best cross-modal encoder available
- **0.7 / 0.3 fusion weight** — text carries more signal than images (ViLBias: images add ~3-8 F1 points)
- **Mismatch detection** — flags when text and image lean different directions
- **Cohen's Kappa** reported alongside standard metrics to assess label quality
