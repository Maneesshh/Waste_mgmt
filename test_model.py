from ultralytics import YOLO
from pathlib import Path

MODEL_PATH = Path("models") / "yolo11m.pt"

print("Loading model...")

model = YOLO(MODEL_PATH)

print("✅ Model loaded successfully!")
print(model.model)