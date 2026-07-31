"""
Project Configuration
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# =====================================================
# Root
# =====================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

# =====================================================
# Directories
# =====================================================

DATASET_DIR = ROOT_DIR / "dataset"

RAW_DATASET_DIR = DATASET_DIR / "raw"

PROCESSED_DATASET_DIR = DATASET_DIR / "processed"

MODELS_DIR = ROOT_DIR / "models"

RUNS_DIR = ROOT_DIR / "runs"

STATIC_DIR = ROOT_DIR / "static"

UPLOAD_DIR = ROOT_DIR / "uploads"

LOG_DIR = ROOT_DIR / "logs"

# =====================================================
# Create folders automatically
# =====================================================

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# Model Configuration
# =====================================================

# Pretrained model name
# Example:
# MODEL_NAME=yolo26n.pt
# MODEL_NAME=yolo11n.pt
MODEL_NAME = os.getenv("MODEL_NAME", "yolo26n.pt")

# Pretrained model path
PRETRAINED_MODEL = MODELS_DIR / MODEL_NAME

# Trained model path
TRAINED_MODEL = (
    RUNS_DIR
    / "waste_detection"
    / "weights"
    / "best.pt"
)

# Automatically use trained model if available
if TRAINED_MODEL.exists():
    MODEL_PATH = TRAINED_MODEL
else:
    MODEL_PATH = PRETRAINED_MODEL

# =====================================================
# Training
# =====================================================

IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", 640))

EPOCHS = int(os.getenv("EPOCHS", 100))

BATCH_SIZE = int(os.getenv("BATCH_SIZE", 8))

import torch

# =====================================================
# Device
# =====================================================

DEVICE = os.getenv("DEVICE", "auto").lower()

if DEVICE == "auto":
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"

# =====================================================
# Prediction
# =====================================================

CONFIDENCE = float(os.getenv("CONFIDENCE", 0.25))

# =====================================================
# Flask
# =====================================================

HOST = os.getenv("HOST", "127.0.0.1")

PORT = int(os.getenv("PORT", 8000))

DEBUG = os.getenv("DEBUG", "True").lower() == "true"

# =====================================================
# Allowed Image Extensions
# =====================================================

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png"
}

# =====================================================
# Final Classes
# =====================================================

FINAL_CLASSES = [
    "Plastic",
    "Paper",
    "Glass",
    "Metal",
    "Trash"
]