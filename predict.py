"""
Predict on an image.
"""

from pathlib import Path

from src.predictor import WastePredictor

predictor = WastePredictor()

IMAGE = Path("sample.jpg")

predictor.predict_image(IMAGE)

print("Prediction completed.")