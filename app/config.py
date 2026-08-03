"""
Central configuration for the digit-recognizer app.

Edit MODEL_PATH if you place best_model.pth somewhere other than
the default `models/` folder produced by the Colab notebooks.
"""

import os

# Project root = one level up from this file's `app/` folder
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pth")

# Must match preprocessing used during training (see notebooks/01, 02)
IMAGE_SIZE = 28          # pixels, square
NUM_CLASSES = 10
PIXEL_MAX_VALUE = 255.0  # normalization divisor used in training

CLASS_NAMES = [str(i) for i in range(NUM_CLASSES)]
