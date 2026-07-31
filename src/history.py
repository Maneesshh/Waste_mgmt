"""
history.py

Save prediction history to CSV.
"""

from pathlib import Path
from datetime import datetime
import csv

from src.config import *

# -------------------------------------------------------
# History file location
# -------------------------------------------------------

HISTORY_FILE = LOG_DIR / "history.csv"


# -------------------------------------------------------
# Create history file if it doesn't exist
# -------------------------------------------------------

def initialize_history():

    LOG_DIR.mkdir(exist_ok=True)

    if HISTORY_FILE.exists():
        return

    with open(HISTORY_FILE, "w", newline="", encoding="utf-8") as file:

        writer = csv.writer(file)

        writer.writerow([
            "Date",
            "Image",
            "Class",
            "Confidence"
        ])


# -------------------------------------------------------
# Save prediction
# -------------------------------------------------------

def save_prediction(image_name, detected_class, confidence):

    initialize_history()

    with open(HISTORY_FILE, "a", newline="", encoding="utf-8") as file:

        writer = csv.writer(file)

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            image_name,
            detected_class,
            round(confidence * 100, 2)
        ])


# -------------------------------------------------------
# Get history
# -------------------------------------------------------

def get_history():

    initialize_history()

    rows = []

    with open(HISTORY_FILE, "r", encoding="utf-8") as file:

        reader = csv.DictReader(file)

        for row in reader:
            rows.append(row)

    return rows[::-1]