"""
Main training script.
"""

import argparse

from src.trainer import WasteTrainer


def main():

    parser = argparse.ArgumentParser(description="YOLO26 Waste Detection Training")

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from runs/waste_detection/weights/last.pt"
    )

    args = parser.parse_args()

    trainer = WasteTrainer()

    trainer.train(resume=args.resume)


if __name__ == "__main__":
    main()