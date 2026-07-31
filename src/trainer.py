"""
YOLO Trainer
"""
import os


from ultralytics import YOLO

from src.config import (
    PRETRAINED_MODEL,
    PROCESSED_DATASET_DIR,
    EPOCHS,
    IMAGE_SIZE,
    BATCH_SIZE,
    DEVICE,
    RUNS_DIR
)


class WasteTrainer:

    def __init__(self):

        # Always start training from the pretrained model
        self.model = YOLO(str(PRETRAINED_MODEL))

    # -------------------------------------------------------

    def train(self):

        self.model.train(

            data=str(PROCESSED_DATASET_DIR / "data.yaml"),

            epochs=EPOCHS,

            imgsz=IMAGE_SIZE,

            batch=BATCH_SIZE,

            device=DEVICE,

            project=str(RUNS_DIR),

            name="waste_detection",

            exist_ok=True,


            workers = 0 if DEVICE == "mps" else min(8, os.cpu_count() or 4),

            verbose=True,

            cache="disk",      # Cache images after first epoch

            amp=True,            # Mixed precision (safe to enable)

            patience=15,         # Early stopping

            cos_lr=True,         # Cosine learning rate schedule

            pretrained=True,   

        )

    # -------------------------------------------------------

    def validate(self):

        return self.model.val()

    # -------------------------------------------------------

    def export(self):

        self.model.export(
            format="onnx"
        )