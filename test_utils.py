from pathlib import Path

from src.utils import (
    create_directory,
    delete_directory,
    setup_logger
)

# Create test folder
folder = Path("temp_test")

create_directory(folder)

logger = setup_logger(Path("logs/test.log"))

logger.info("Logger is working!")

delete_directory(folder)

print("✅ Utils working correctly.")