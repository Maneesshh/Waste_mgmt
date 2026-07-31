"""
Dataset preparation module.

Responsibilities:
1. Read original YOLO dataset
2. Remap 42 classes -> 5 classes
3. Merge all images
4. Shuffle
5. Split 70/15/15
6. Generate data.yaml
"""

from pathlib import Path
import random
import shutil
import yaml

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.config import *
from src.mappings import *
from src.utils import *


class DatasetPreparer:

    def __init__(self):

        self.raw = RAW_DATASET
        self.output = PROCESSED_DATASET

        self.logger = setup_logger(LOG_DIR / "dataset.log")

        create_directory(LOG_DIR)

        self.original_classes = []
        self.class_to_index = {}

        self.samples = []

    # ----------------------------------------------------------

    def load_yaml(self):

        yaml_file = self.raw / "data.yaml"

        if not yaml_file.exists():
            raise FileNotFoundError(yaml_file)

        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f)

        self.original_classes = data["names"]

        self.class_to_index = {
            name: idx
            for idx, name in enumerate(self.original_classes)
        }

        self.logger.info(
            f"Loaded {len(self.original_classes)} classes."
        )

    # ----------------------------------------------------------

    def collect_dataset(self):

        self.samples = []

        for split in ["train", "valid", "test"]:

            img_dir = self.raw / split / "images"
            lbl_dir = self.raw / split / "labels"

            for img in img_dir.iterdir():

                if img.suffix.lower() not in [
                    ".jpg",
                    ".jpeg",
                    ".png"
                ]:
                    continue

                label = lbl_dir / f"{img.stem}.txt"

                if label.exists():

                    self.samples.append(
                        (
                            img,
                            label
                        )
                    )

        self.logger.info(
            f"Collected {len(self.samples)} samples."
        )

      # ----------------------------------------------------------
    # Convert a single YOLO label file from 42 classes -> 5 classes
    # ----------------------------------------------------------

    def convert_label(self, label_path: Path):
        """
        Convert one label file from 42 classes to 5 classes.
        Returns a list of converted YOLO label lines.
        """

        converted = []

        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            old_class = int(parts[0])

            old_name = self.original_classes[old_class]

            # Skip classes that are not mapped
            if old_name not in CLASS_MAPPING:
                continue

            final_name = CLASS_MAPPING[old_name]

            new_class = FINAL_CLASS_TO_ID[final_name]

            converted.append(
                f"{new_class} {' '.join(parts[1:])}"
            )

        return converted

    # ----------------------------------------------------------
    # Convert every sample
    # ----------------------------------------------------------

    def remap_dataset(self):

        remapped = []

        self.logger.info("Remapping labels...")

        for image_path, label_path in tqdm(self.samples):

            labels = self.convert_label(label_path)

            if len(labels) == 0:
                continue

            remapped.append(
                (
                    image_path,
                    labels
                )
            )

        self.samples = remapped

        self.logger.info(
            f"Remaining samples: {len(self.samples)}"
        )      

            # ----------------------------------------------------------
    # Split dataset into Train / Validation / Test
    # ----------------------------------------------------------

    def split_dataset(self):

        random.seed(42)

        random.shuffle(self.samples)

        train, temp = train_test_split(
            self.samples,
            train_size=0.70,
            random_state=42
        )

        val, test = train_test_split(
            temp,
            test_size=0.50,
            random_state=42
        )

        self.train = train
        self.val = val
        self.test = test

        self.logger.info(f"Train : {len(train)}")
        self.logger.info(f"Val   : {len(val)}")
        self.logger.info(f"Test  : {len(test)}")


            # ----------------------------------------------------------
    # Copy images and labels
    # ----------------------------------------------------------

    def copy_split(self, dataset, split):

        image_dir = self.output / split / "images"
        label_dir = self.output / split / "labels"

        create_directory(image_dir)
        create_directory(label_dir)

        self.logger.info(f"Copying {split}...")

        for image_path, labels in tqdm(dataset):

            shutil.copy2(
                image_path,
                image_dir / image_path.name
            )

            label_path = label_dir / f"{image_path.stem}.txt"

            with open(label_path, "w") as f:

                for line in labels:
                    f.write(line + "\n")

                        # ----------------------------------------------------------
    # Create data.yaml
    # ----------------------------------------------------------

    def create_yaml(self):

        yaml_data = {
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "nc": len(FINAL_CLASSES),
            "names": FINAL_CLASSES
        }

        with open(self.output / "data.yaml", "w") as f:
            yaml.dump(
                yaml_data,
                f,
                sort_keys=False
            )

        self.logger.info("Created data.yaml")


            # ----------------------------------------------------------
    # Prepare complete dataset
    # ----------------------------------------------------------

    def prepare(self):

        delete_directory(self.output)

        create_directory(self.output)

        self.load_yaml()

        self.collect_dataset()

        self.remap_dataset()

        self.split_dataset()

        self.copy_split(
            self.train,
            "train"
        )

        self.copy_split(
            self.val,
            "val"
        )

        self.copy_split(
            self.test,
            "test"
        )

        self.create_yaml()

        self.logger.info("Dataset preparation completed.")