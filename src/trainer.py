"""
YOLO Trainer
"""

import os
import shutil
from pathlib import Path

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

DRIVE_BACKUP_DIR = "/content/drive/MyDrive/WasteDetection/checkpoints/runs"


def backup_to_drive(trainer):
    """Callback: copy runs/ to Drive after every epoch checkpoint save."""
    try:
        shutil.copytree(str(RUNS_DIR), DRIVE_BACKUP_DIR, dirs_exist_ok=True)
        print(f"[backup] Epoch {trainer.epoch + 1} checkpoint synced to Drive.")
    except Exception as e:
        print(f"[backup] Failed to sync to Drive: {e}")


class WasteTrainer:

    def __init__(self):
        self.pretrained_model = Path(PRETRAINED_MODEL)
        self.last_checkpoint = (
            RUNS_DIR / "waste_detection" / "weights" / "last.pt"
        )
        self.model = None

    # -------------------------------------------------------

    def train(self, resume=False):

        if resume:
            if not self.last_checkpoint.exists():
                raise FileNotFoundError(
                    f"Checkpoint not found:\n{self.last_checkpoint}"
                )

            print(f"\nResuming training from:\n{self.last_checkpoint}\n")

            self.model = YOLO(str(self.last_checkpoint))
            self.model.add_callback("on_model_save", backup_to_drive)

            self.model.train(resume=True)
            return

        # -------------------------------------------------------
        # Start a new training
        # -------------------------------------------------------

        print(f"\nStarting new training from:\n{self.pretrained_model}\n")

        self.model = YOLO(str(self.pretrained_model))
        self.model.add_callback("on_model_save", backup_to_drive)

        self.model.train(
            data=str(PROCESSED_DATASET_DIR / "data.yaml"),
            epochs=EPOCHS,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            device=DEVICE,
            project=str(RUNS_DIR),
            name="waste_detection",
            exist_ok=True,
            workers=0 if DEVICE == "mps" else min(8, os.cpu_count() or 4),
            verbose=True,
            cache="disk",
            amp=True,
            patience=15,
            cos_lr=True,
            pretrained=True
        )

    # -------------------------------------------------------

    def validate(self):
        return self.model.val()

    # -------------------------------------------------------

    def export(self):
        self.model.export(format="onnx")