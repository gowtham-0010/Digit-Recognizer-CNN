"""
Preprocessing for inference.

Must exactly mirror how training data was prepared in the notebooks:
- grayscale, 28x28
- reshaped to (1, 1, 28, 28)
- pixel values scaled to [0, 1] via division by 255.0 (no mean/std
  normalization was used during training, so none is applied here)

MNIST-style training images are white digits on a black background.
Images from a Streamlit canvas or an uploaded photo are typically the
opposite (dark digit on a light background), so we auto-detect and invert
when needed.
"""

import numpy as np
from PIL import Image
import torch

from app.config import IMAGE_SIZE, PIXEL_MAX_VALUE


def _auto_invert_if_needed(img_array: np.ndarray) -> np.ndarray:
    """If the image looks like a dark digit on a light background
    (mean pixel value is high), invert it to match MNIST's light-on-dark
    convention used during training."""
    if img_array.mean() > (PIXEL_MAX_VALUE / 2):
        return PIXEL_MAX_VALUE - img_array
    return img_array


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Convert a PIL image into a (1, 1, 28, 28) float32 tensor ready for
    SimpleCNN, matching the training-time preprocessing exactly.
    """
    image = image.convert("L")  # grayscale
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)

    img_array = np.array(image).astype("float32")
    img_array = _auto_invert_if_needed(img_array)
    img_array = img_array / PIXEL_MAX_VALUE  # same normalization as training

    tensor = torch.tensor(img_array).unsqueeze(0).unsqueeze(0)  # (1, 1, 28, 28)
    return tensor
