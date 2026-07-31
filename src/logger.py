"""
Application Logger
"""

import logging

from src.config import LOG_DIR

LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(

    filename=LOG_DIR / "application.log",

    level=logging.INFO,

    format="%(asctime)s | %(levelname)s | %(message)s"

)

logger = logging.getLogger("WasteDetection")