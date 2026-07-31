"""
Evaluate trained model.
"""

from ultralytics import YOLO

from src.config import *

model = YOLO(

    RUNS_DIR /
    "waste_detection" /
    "weights" /
    "best.pt"

)

metrics = model.val(

    data=str(PROCESSED_DATASET / "data.yaml")

)

print(metrics)