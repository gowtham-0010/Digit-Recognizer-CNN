"""
Streamlit demo for the Digit Recognizer CNN.

Run with:
    streamlit run app/streamlit_app.py

Loads the pretrained best_model.pth (trained in Google Colab) and never
retrains — this file is purely an inference-time UI.
"""

import sys
import os

# Allow running via `streamlit run app/streamlit_app.py` from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from PIL import Image

from app.predictor import predict
from app.config import MODEL_PATH

st.set_page_config(page_title="Digit Recognizer (CNN)", page_icon="✏️", layout="centered")

st.title("✏️ Handwritten Digit Recognizer")
#st.caption("CNN trained on the Kaggle Digit Recognizer (MNIST) dataset — inference only, model trained in Google Colab.")

if not os.path.exists(MODEL_PATH):
    st.error(
        f"Model file not found at `{MODEL_PATH}`.\n\n"
        "Train the model in Google Colab (`notebooks/02_CNN_Training.ipynb` "
        "and `notebooks/03_Evaluation_and_Export.ipynb`), then download "
        "`best_model.pth` from your Google Drive into the local `models/` folder."
    )
    st.stop()

tab_draw, tab_upload = st.tabs(["🖌️ Draw a digit", "📁 Upload an image"])

image_to_predict = None

with tab_draw:
    st.write("Draw a single digit (0–9) below, then click **Predict**.")
    try:
        from streamlit_drawable_canvas import st_canvas

        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=18,
            stroke_color="white",
            background_color="black",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )
        if canvas_result.image_data is not None:
            image_to_predict = Image.fromarray(
                canvas_result.image_data.astype("uint8")
            ).convert("L")
    except ImportError:
        st.warning(
            "Drawing canvas requires the `streamlit-drawable-canvas` package "
            "(see requirements.txt). Use the **Upload an image** tab instead."
        )

with tab_upload:
    uploaded_file = st.file_uploader("Upload a digit image (PNG/JPG)", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image_to_predict = Image.open(uploaded_file)
        st.image(image_to_predict, caption="Uploaded image", width=150)

if st.button("Predict", type="primary"):
    if image_to_predict is None:
        st.warning("Please draw or upload a digit first.")
    else:
        result = predict(image_to_predict, top_k=3)

        st.subheader(f"Prediction: {result['predicted_class']}")
        st.write(f"Confidence: **{result['confidence'] * 100:.2f}%**")

        st.write("Top-3 predictions:")
        for label, confidence in result["top_k"]:
            st.write(f"- Digit **{label}**: {confidence * 100:.2f}%")
            st.progress(min(confidence, 1.0))

st.divider()
#st.caption("Model: SimpleCNN (2 conv blocks + BatchNorm + Dropout) — trained with data augmentation in Google Colab. See README.md for the full project write-up.")
