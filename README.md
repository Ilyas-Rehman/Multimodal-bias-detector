# Multimodal News Bias Detector

**Ilyas Rehman** — research project supervised by Khaled (KMA) Solaiman, started March 2026

This is a work-in-progress research project, not a finished tool. I'm building it as part of my research work and writing this up as I go.

The goal is to detect political bias (Left / Center / Right) in news articles by looking at both the article text *and* the image that accompanies it. The idea is that bias doesn't just show up in the words — image selection can push a political frame even when the text seems neutral. I came across this in the ViLBias paper (arXiv:2412.17052) from the Vector Institute and wanted to build something similar.

The text classifier works well. The image model doesn't work yet — more on that below.

---

## What's working and what isn't

**Works:**
- Text-only bias classification using a fine-tuned RoBERTa model. F1 of 0.85, Cohen's Kappa of 0.78 on the test set.
- Data pipeline — downloading articles, cleaning text, splitting into train/val/test.
- Weighted fusion of text + image scores (70% text, 30% image).
- Cross-modal mismatch detection — flags when the text and image lean in opposite directions.
- Streamlit demo app for testing predictions.

**Doesn't work yet:**
- The image model. The dataset I'm using (NewsMediaBias-Plus on HuggingFace) lists image file paths in the metadata but the actual image files were never uploaded to the repo — they 404. So I can download the article text fine but end up with gray placeholder images for everything. The image model trains against these placeholders and learns nothing useful. I'm looking into alternative image sources.
- The learned fusion model is implemented but untrained — I need real image features before that makes sense.
- The explanation system is basic keyword matching right now. I want to replace it with LIME or SHAP eventually.

---

## Setup

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Running it

**Step 1 — Download data**

```bash
python get_data.py
```

This pulls articles from the NewsMediaBias-Plus dataset and tries to download images. Most images will be placeholders because of the 404 issue described above.

**Step 2 — Preprocess**

```bash
python utils/preprocessing.py
```

Cleans text, tokenizes everything, and saves train/val/test splits to `data/processed/`.

**Step 3 — Train**

```bash
python train.py --mode text     # just the text model
python train.py --mode image    # just the image model (broken for now)
python train.py --mode fusion   # combine both (needs trained text + image models)
python train.py --mode all      # run all phases in order
```

**Step 4 — Run the demo app**

```bash
streamlit run app/main.py
```

Paste in article text, optionally upload an image, and it returns a bias label with confidence scores and mismatch info.

**Run on a single article from the terminal**

```bash
python predict.py --text "Article text here" --image path/to/image.jpg
```

---

## Project structure

```
bias_detect/
├── data/
│   ├── raw/              # Downloaded articles and images
│   └── processed/        # Train/val/test splits
├── models/
│   ├── text_model.py     # RoBERTa fine-tuning
│   ├── image_model.py    # CLIP image classifier
│   └── fusion_model.py   # Weighted and learned fusion
├── utils/
│   ├── preprocessing.py  # Text cleaning, tokenization, image transforms
│   └── evaluation.py     # Metrics, confusion matrix, mismatch report
├── app/
│   └── main.py           # Streamlit demo
├── get_data.py           # Downloads the dataset
├── train.py              # Training pipeline
└── predict.py            # Inference on new articles
```

---

## Why these model choices

- **RoBERTa** — the ViLBias paper tested several text encoders and RoBERTa came out on top for this task. I'm using `roberta-base` because the large version is too slow on CPU.
- **CLIP** — trained on image-text pairs, so it understands visual semantics better than a plain ResNet. Again per ViLBias: CLIP-based image classifiers outperform ResNet-only approaches.
- **0.7 / 0.3 fusion** — text carries most of the signal. The ViLBias results show images only add about 3-8 F1 points on top of text alone, so text gets the higher weight.
- **Cohen's Kappa** — I'm reporting this alongside F1 because it accounts for class imbalance and gives a better picture of whether the model is actually learning or just guessing the majority class.

---

## The image data problem

The `vector-institute/newsmediabias-plus` dataset on HuggingFace has an `image` column that contains file paths like `images/39e6104d56.jpeg`. Those files don't exist in the repo — requests return 404. I didn't realize this until I ran `get_data.py` and noticed every image was coming back as a placeholder.

The text data is fine and the text model trained normally. For the image side I'm currently looking at whether there's another version of the dataset, or if I need to scrape images from the original article URLs myself.

---

## Dataset and references

- NewsMediaBias-Plus: `vector-institute/newsmediabias-plus` on HuggingFace
- ViLBias paper: arXiv:2412.17052
- Outlet bias labels based on AllSides Media Bias Ratings
