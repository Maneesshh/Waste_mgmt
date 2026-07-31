"""
Utility functions for the Waste Detection project.
"""

from pathlib import Path
import logging
import shutil


def create_directory(path: Path) -> None:
    """
    Create a directory if it doesn't exist.
    """
    path.mkdir(parents=True, exist_ok=True)


def delete_directory(path: Path) -> None:
    """
    Delete a directory if it exists.
    """
    if path.exists():
        shutil.rmtree(path)


def setup_logger(log_file: Path) -> logging.Logger:
    """
    Configure and return a logger.
    """

    logger = logging.getLogger(log_file.stem)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger