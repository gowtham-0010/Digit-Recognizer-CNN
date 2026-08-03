"""
Loads the trained SimpleCNN weights (best_model.pth) exported from Colab.

This module NEVER trains — it only instantiates the architecture and
loads pretrained weights, in line with the two-environment design:
Colab trains, VS Code only performs inference.
"""

import os
import torch

from app.config import MODEL_PATH
from app.model_def import SimpleCNN


def load_model(model_path: str = MODEL_PATH, device: str = "cpu") -> SimpleCNN:
    """
    Instantiate SimpleCNN and load pretrained weights from model_path.

    Raises FileNotFoundError with a clear message if the .pth file hasn't
    been downloaded yet from Google Drive (Colab notebooks/03 output).
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at '{model_path}'.\n"
            "Train the model in Google Colab (notebooks/02_CNN_Training.ipynb "
            "and notebooks/03_Evaluation_and_Export.ipynb), then download "
            "'best_model.pth' from your Google Drive into the local "
            "'models/' folder."
        )

    model = SimpleCNN()
    state_dict = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(state_dict)
    model.eval()
    return model
