# https://docs.ultralytics.com/es/guides/object-counting/#what-is-object-counting
# https://youtu.be/A1V8yYlGEkI?t=1044



# ============================ #
#           IMPORTS            #
# ============================ #

# Standard library imports
import logging
from pathlib import Path
from typing import Optional, Set
import numpy as np

# Third-party imports
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort, Detection

# ============================= #
#         CONFIGURATION         #
# ============================= #

DATASET_PATH: Path = Path("./dataset2")
VIDEO_INPUT: Path = Path("./inputs/chumbos_vid_trim.mp4")
VIDEO_OUTPUT: Path = Path("./outputs/output_video_tracked_chumbos.avi")
IMAGE_INPUT: Path = Path("./inputs/3d8de7c7-1000042267.jpg")
IMAGE_OUTPUT: Path = Path("./outputs/output_img.jpg")
RESULTS_TXT: Path = Path("./outputs/conteo_resultados.txt")
ITEM_NAME: str = "Chumbo" #Apple"

# Counting line position (in pixels)
LINE_POSITION: int = 500

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
