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

def train_model(
    epochs: int = 5,
    imgsz: int = 640,
    batch: int = 6,
    save_path: Optional[Path] = Path("./models/yolo8_item_count.pt")
) -> YOLO:
    """
    Train a YOLOv8 model on a custom dataset with optional data augmentation
    and export it for easy use in Python.

    Args:
        epochs (int): Number of training epochs.
        imgsz (int): Input image size for training.
        batch (int): Batch size for training.
        save_path (Optional[Path]): File path to save the exported model (.pt format).

    Returns:
        YOLO: Trained YOLO model instance.
    """
    model = YOLO(model="yolov8s.pt", task="detect")

    logging.info(f"Starting training data from {DATASET_PATH}...")
    logging.info(f"{DATASET_PATH}/data.yaml")
 
    model.train(
        data=f"{DATASET_PATH}/data.yaml",
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name="yolo8_item_count",
        # augment=True
        # val=str(DATASET_PATH / "val"),  # Validation disabled due to few images
    )

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model.export(format="onnx", imgsz=imgsz, batch=batch) #, file=str(save_path))
        logging.info(f"Trained model exported to {save_path}")

    return model