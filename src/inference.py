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

def process_video(
    model: YOLO,
    input_path: Path,
    output_path: Path,
    item_name: str = ITEM_NAME,
    line_position: int = LINE_POSITION
) -> Optional[int]:
    """
    Process a video frame by frame, apply YOLO inference, track objects,
    count unique items crossing a counting line, and save an annotated output video.

    Args:
        model (YOLO): Trained YOLO model.
        input_path (Path): Path to the input video.
        output_path (Path): Path to save the processed video.
        item_name (str): Name of the target object class to count.
        line_position (int): Y-coordinate of the counting line in pixels.

    Returns:
        Optional[int]: Total number of unique items counted, or None if processing fails.
    """
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        logging.error(f"Cannot open video: {input_path}")
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(
        str(output_path),
        fourcc,
        cap.get(cv2.CAP_PROP_FPS),
        (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    )

    tracker = DeepSort(max_age=30)
    counted_ids: Set[int] = set()
    total_count: int = 0
    frame_num: int = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        results = model.predict(frame, conf=0.9, iou=0.8, verbose=False)
        detections = []

        # Prepare detections for tracker: [x1, y1, x2, y2, conf, cls]
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                conf = float(box.conf[0])
                if cls_name == item_name:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x = (x1+x2)/2
                    y = (y1+y2)/2
                    w = x2 - x1
                    h = y2 - y1
                    detections.append(([x1, y1, w, h], conf, cls_name))

        tracks = tracker.update_tracks(detections, frame=frame)

        # Draw bounding boxes, IDs, and count objects crossing the line
        for t in tracks:
            if not t.is_confirmed():
                continue
            track_id = t.track_id
            x1, y1, x2, y2 = map(int, t.to_ltrb())
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{track_id}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Count only if crossing line and not counted before
            if cy < line_position + 5 and cy > line_position - 5 and track_id not in counted_ids:
                total_count += 1
                counted_ids.add(track_id)

        # Draw counting line and total count
        cv2.line(frame, (0, line_position), (frame.shape[1], line_position), (0, 0, 255), 2)
        cv2.putText(frame, f"{item_name} count: {total_count}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

        out.write(frame)

    cap.release()
    out.release()
    logging.info(f"Processed {frame_num} frames, total {item_name}: {total_count}")
    return total_count



def process_image(
    model: YOLO,
    input_path: Path = IMAGE_INPUT,
    output_path: Path = IMAGE_OUTPUT,
    item_name: str = ITEM_NAME
) -> int:
    """
    Process a single image, apply YOLO inference, draw bounding boxes with class and confidence,
    and save the annotated image.

    Args:
        model (YOLO): Trained YOLO model.
        input_path (Path): Path to the input image.
        output_path (Path): Path to save the processed image.
        item_name (str): Name of the target object class to count.

    Returns:
        int: Number of detected items matching the target class.
    """
    image = cv2.imread(str(input_path))
    if image is None:
        logging.error(f"Cannot open image: {input_path}")
        return 0

    results = model.predict(image, conf=0.9, iou=0.8 ,verbose=False)
    total_count = 0

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            if cls_name == item_name:
                total_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    image,
                    f"{cls_name} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

    # Draw total count on the image
    cv2.putText(
        image,
        f"Total {item_name}: {total_count}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 0, 0),
        2
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    logging.info(f"Processed image saved at {output_path}, total {item_name}: {total_count}")

    return total_count