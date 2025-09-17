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

# Custom imports
from src.training import train_model
from src.inference import process_video, process_image
from src.metrics import export_metrics





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
#        MAIN (EXECUTION)       #
# ============================= #

if __name__ == "__main__":
    runs_dir = Path("runs/detect/yolo11_item_count/weights/best.pt")
    if not runs_dir.exists():
        logging.info("Training YOLOv8 model...")
        model = train_model()
    else:
        logging.info("Loading pretrained YOLOv8 model...")
        model = YOLO(str(runs_dir))

    logging.info("Processing video with tracking and counting line...")
    #total_count = process_video(model, VIDEO_INPUT, VIDEO_OUTPUT)
    total_count = process_image(model, IMAGE_INPUT, IMAGE_OUTPUT, ITEM_NAME)

    if total_count is not None:
        cap = cv2.VideoCapture(str(VIDEO_INPUT))
        frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        export_metrics(total_count, frame_num, RESULTS_TXT)