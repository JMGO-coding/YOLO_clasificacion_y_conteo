# ============================ #
#           IMPORTS            #
# ============================ #

# Standard library imports
import logging
import os
from pathlib import Path
from typing import Optional, Set
import numpy as np
from dotenv import load_dotenv

# Third-party imports
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort, Detection





# ============================= #
#         CONFIGURATION         #
# ============================= #

# Loading .env
load_dotenv()

# Reading 
DATASET_PATH = Path(os.getenv("DATASET_PATH"))
VIDEO_INPUT = Path(os.getenv("VIDEO_INPUT"))
VIDEO_OUTPUT = Path(os.getenv("VIDEO_OUTPUT"))
IMAGE_INPUT = Path(os.getenv("IMAGE_INPUT"))
IMAGE_OUTPUT = Path(os.getenv("IMAGE_OUTPUT"))
RESULTS_TXT = Path(os.getenv("RESULTS_TXT"))
ITEM_NAME = os.getenv("ITEM_NAME")
LINE_POSITION = int(os.getenv("LINE_POSITION", 500))

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)





# ============================= #
#           FUNCTIONS           #
# ============================= #

def export_metrics(
    total_count: int,
    frame_num: int,
    results_path: Path,
    item_name: str = ITEM_NAME
) -> None:
    """
    Export counting metrics to a text file.

    Args:
        total_count (int): Total number of counted items.
        frame_num (int): Total number of frames processed.
        results_path (Path): Path to save the metrics file.
        item_name (str): Name of the counted item class.
    """
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        f.write("===== COUNTING RESULTS =====\n")
        f.write(f"Item detected: {item_name}\n")
        f.write(f"Total detected in video: {total_count}\n")
        f.write(f"Frames processed: {frame_num}\n")
        f.write(f"Average per frame: {total_count/frame_num:.2f}\n")
    logging.info(f"Metrics exported to {results_path}")