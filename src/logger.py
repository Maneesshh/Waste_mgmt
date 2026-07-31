"""
Application Logger
"""

import logging
from pathlib import Path

from src.config import LOG_DIR

# Create log directory if it doesn't exist
LOG_DIR.mkdir(parents=True, exist_ok=True)


def setup_logger(log_file: Path, name: str = "WasteDetection") -> logging.Logger:
    """
    Create and return a logger that writes to the given log file.
    """

    logger = logging.getLogger(name)

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


# Default application logger
logger = setup_logger(LOG_DIR / "application.log")