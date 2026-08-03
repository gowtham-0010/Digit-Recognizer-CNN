"""
High-level inference wrapper.

Loads the trained model once, then exposes a simple predict(image) API
used by the Streamlit app. No training happens here or anywhere in the
app/ package — it strictly consumes the .pth file exported from Colab.
"""

from PIL import Image
import torch

from app.model_loader import load_model
from app.preprocessing import preprocess_image
from app.utils import logits_to_probabilities, top_k_predictions
from app.config import CLASS_NAMES

_model = None  # lazy-loaded singleton


def get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model


def predict(image: Image.Image, top_k: int = 3) -> dict:
    """
    Run inference on a single PIL image of a handwritten digit.

    Returns:
        {
            "predicted_label": int,
            "predicted_class": str,
            "confidence": float,
            "top_k": [(label, confidence), ...]
        }
    """
    model = get_model()
    tensor = preprocess_image(image)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = logits_to_probabilities(logits)

    predicted_label = int(torch.argmax(probabilities, dim=1).item())
    confidence = float(probabilities[0, predicted_label].item())
    top_k_results = top_k_predictions(probabilities, k=top_k)

    return {
        "predicted_label": predicted_label,
        "predicted_class": CLASS_NAMES[predicted_label],
        "confidence": confidence,
        "top_k": top_k_results,
    }
