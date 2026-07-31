"""
Dataset preparation module.

Responsibilities
----------------
1. Read original YOLO dataset
2. Remap 42 classes -> 5 classes
3. Merge all images
4. Shuffle
5. Split into Train / Validation / Test
6. Generate data.yaml
"""

from pathlib import Path
import random
import shutil
import yaml

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.config import (
    RAW_DATASET_DIR,
    PROCESSED_DATASET_DIR,
    LOG_DIR,
    FINAL_CLASSES,
)

from src.mappings import CLASS_MAPPING, FINAL_CLASS_TO_ID
from src.utils import create_directory, delete_directory
from src.logger import setup_logger


class DatasetPreparer:

    def __init__(self):

        self.raw = RAW_DATASET_DIR
        self.output = PROCESSED_DATASET_DIR

        create_directory(LOG_DIR)

        self.logger = setup_logger(LOG_DIR / "dataset.log")

        self.original_classes = []
        self.class_to_index = {}

        self.samples = []

    # ----------------------------------------------------------

    def load_yaml(self):

        yaml_file = self.raw / "data.yaml"

        if not yaml_file.exists():
            raise FileNotFoundError(f"Cannot find {yaml_file}")

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

    def _find_validation_folder(self):

        if (self.raw / "valid").exists():
            return "valid"

        if (self.raw / "val").exists():
            return "val"

        raise FileNotFoundError(
            "Neither 'valid' nor 'val' folder exists."
        )

    # ----------------------------------------------------------

    def collect_dataset(self):

        self.samples = []

        validation_folder = self._find_validation_folder()

        splits = [
            "train",
            validation_folder,
            "test"
        ]

        for split in splits:

            img_dir = self.raw / split / "images"
            lbl_dir = self.raw / split / "labels"

            if not img_dir.exists():
                continue

            for img in img_dir.iterdir():

                if img.suffix.lower() not in (
                    ".jpg",
                    ".jpeg",
                    ".png"
                ):
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

    def convert_label(self, label_path: Path):

        converted = []

        with open(label_path) as f:

            for line in f:

                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                old_class = int(parts[0])

                old_name = self.original_classes[old_class]

                if old_name not in CLASS_MAPPING:
                    continue

                final_name = CLASS_MAPPING[old_name]

                new_class = FINAL_CLASS_TO_ID[final_name]

                converted.append(
                    f"{new_class} {' '.join(parts[1:])}"
                )

        return converted

    # ----------------------------------------------------------

    def remap_dataset(self):

        self.logger.info("Remapping labels...")

        remapped = []

        for image_path, label_path in tqdm(self.samples):

            labels = self.convert_label(label_path)

            if labels:

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

            with open(
                label_dir / f"{image_path.stem}.txt",
                "w"
            ) as f:

                f.write("\n".join(labels))

    # ----------------------------------------------------------

    def create_yaml(self):

        data = {

            "train": "train/images",

            "val": "val/images",

            "test": "test/images",

            "nc": len(FINAL_CLASSES),

            "names": FINAL_CLASSES

        }

        with open(
            self.output / "data.yaml",
            "w"
        ) as f:

            yaml.safe_dump(
                data,
                f,
                sort_keys=False
            )

        self.logger.info(
            "Created data.yaml"
        )

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

        self.logger.info(
            "Dataset preparation completed."
        )