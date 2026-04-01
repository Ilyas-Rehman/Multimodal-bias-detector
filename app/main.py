# app/main.py
#
# Streamlit web interface for the bias detector.
# I added this so it's easier to demo than running predict.py from the terminal.
#
# Run with:
#     streamlit run app/main.py

import streamlit as st
import torch
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predict import predict_article

st.set_page_config(
    page_title="Multimodal Bias Detector",
    page_icon="",
    layout="wide"
)

# ── Styling ───────────────────────────────────────────────────────────────────

LABEL_COLORS = {
    "Left": "#2980b9",
    "Center": "#27ae60",
    "Right": "#c0392b"
}

st.title("Multimodal News Bias Detector")
st.caption("Analyzes both article text and image to classify political bias: Left / Center / Right")

st.divider()

# ── Input area ────────────────────────────────────────────────────────────────

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Article text")
    article_text = st.text_area(
        label="Paste the news article text here",
        height=300,
        placeholder="Paste article content here...",
        label_visibility="collapsed"
    )

with col2:
    st.subheader("Article image")
    uploaded_image = st.file_uploader(
        "Upload the article's image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )
    if uploaded_image:
        st.image(uploaded_image, use_container_width=True)
    else:
        st.info("No image uploaded — prediction will be text-only.")

st.divider()

# ── Run prediction ────────────────────────────────────────────────────────────

run_btn = st.button("Analyze bias", type="primary", use_container_width=False)

if run_btn:
    if not article_text.strip():
        st.error("Please paste some article text before analyzing.")
    else:
        with st.spinner("Running analysis..."):
            # Streamlit gives us a file object but predict_article needs a path,
            # so save to a temp file first
            image_path = None
            if uploaded_image:
                import tempfile
                from PIL import Image
                img = Image.open(uploaded_image).convert("RGB")
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                    img.save(f.name)
                    image_path = f.name

            try:
                result = predict_article(article_text, image_path)
            except Exception as e:
                st.error(f"Error running model: {e}")
                st.stop()

        # ── Results ──────────────────────────────────────────────────────────

        st.divider()
        st.subheader("Results")

        col_label, col_conf, col_mismatch = st.columns(3)

        bias = result["bias"]
        color = LABEL_COLORS.get(bias, "#888")

        with col_label:
            st.markdown(f"**Bias label**")
            st.markdown(
                f"<h2 style='color:{color};'>{bias}</h2>",
                unsafe_allow_html=True
            )

        with col_conf:
            st.markdown("**Confidence**")
            st.metric(label="", value=f"{result['confidence']:.1%}")

        with col_mismatch:
            st.markdown("**Cross-modal mismatch**")
            if result["mismatch"]:
                st.warning("Mismatch detected")
                d = result["mismatch_detail"]
                st.caption(d["description"])
            else:
                st.success("Text and image agree")

        st.divider()

        # Score bars
        col_text_scores, col_image_scores = st.columns(2)

        with col_text_scores:
            st.markdown("**Text model scores**")
            for label in ["Left", "Center", "Right"]:
                score = result["text_score"][label]
                st.markdown(f"{label}")
                st.progress(score, text=f"{score:.1%}")

        with col_image_scores:
            st.markdown("**Image model scores**")
            if image_path:
                for label in ["Left", "Center", "Right"]:
                    score = result["image_score"][label]
                    st.markdown(f"{label}")
                    st.progress(score, text=f"{score:.1%}")
            else:
                st.caption("No image provided — uniform distribution used.")

        st.divider()

        # Explanation
        st.subheader("Explanation")
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            st.markdown("**Text signals**")
            st.info(result["explanation"]["text"])
        with exp_col2:
            st.markdown("**Image signals**")
            st.info(result["explanation"]["image"])

        # Raw JSON for debugging
        with st.expander("Raw output (JSON)"):
            import json
            st.code(json.dumps(result, indent=2), language="json")

# ── Sidebar info ──────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("About")
    st.markdown("""
    This system classifies political bias in news articles using:

    - **RoBERTa** for text analysis
    - **CLIP** for image analysis
    - **Weighted fusion** (70% text, 30% image)
    - **Cross-modal mismatch** detection

    Based on findings from the ViLBias paper (Vector Institute, 2026).

    ---

    **Bias categories:**
    - Left — progressive/liberal framing
    - Center — balanced/neutral framing
    - Right — conservative framing

    ---

    **Mismatch flag:** raised when the text and image
    lean in clearly different political directions.
    This is a key signal that content may be using
    images to push a narrative beyond what the text says.
    """)
