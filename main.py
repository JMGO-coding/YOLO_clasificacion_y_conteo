# https://docs.ultralytics.com/es/guides/object-counting/#what-is-object-counting
# https://youtu.be/A1V8yYlGEkI?t=1044



# ============================ #
#           IMPORTS            #
# ============================ #

# Standard library imports
import logging
from pathlib import Path
from typing import Optional

# Third-party imports
import cv2
from ultralytics import YOLO





# ============================= #
#         CONFIGURATION         #
# ============================= #

# Constants
DATASET_PATH: Path = Path("./datasets/mi_dataset")
VIDEO_INPUT: Path = Path("./inputs/input_video.mp4")
VIDEO_OUTPUT: Path = Path("./outputs/output_video.mp4")
RESULTS_TXT: Path = Path("./outputs/conteo_resultados.txt")
ITEM_NAME: str = "mi_item"

# LOGGER CONFIGURATION
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)





# ============================= #
#           FUNCTIONS           #
# ============================= #
def train_model(epochs: int = 50, imgsz: int = 640, batch: int = 16) -> YOLO:
    """
    Train a YOLOv11 model on the custom dataset.

    Args:
        epochs (int): Number of training epochs.
        imgsz (int): Input image size for training.
        batch (int): Batch size.

    Returns:
        YOLO: The trained YOLOv11 model instance.
    """
    model = YOLO(model="yolov11s.pt", task="detect")

    model.train(
        data=str(DATASET_PATH / "data.yaml"),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name="yolo11_item_count"
    )

    return model


def process_video(model: YOLO, input_path: Path, output_path: Path) -> Optional[int]:
    """
    Process a video, run inference with YOLO, count instances of ITEM_NAME,
    and write the processed video to disk.

    Args:
        model (YOLO): YOLOv11 model instance.
        input_path (Path): Path to input video.
        output_path (Path): Path to output processed video.

    Returns:
        Optional[int]: Total count of detected items in the video, or None if failed.
    """
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        logging.error(f"Cannot open video: {input_path}")
        return None

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        str(output_path),
        fourcc,
        cap.get(cv2.CAP_PROP_FPS),
        (int(cap.get(3)), int(cap.get(4)))
    )

    total_count: int = 0
    frame_num: int = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        results = model(frame, conf=0.5)

        count = 0
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                if cls_name == ITEM_NAME:
                    count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        cls_name,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

        total_count += count

        cv2.putText(
            frame,
            f"{ITEM_NAME}: {count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 0, 0),
            2
        )

        out.write(frame)

    cap.release()
    out.release()

    logging.info(f"Processed {frame_num} frames from {input_path}")
    return total_count


def export_metrics(
    total_count: int,
    frame_num: int,
    results_path: Path,
    item_name: str = ITEM_NAME
) -> None:
    """
    Export detection metrics to a text file.

    Args:
        total_count (int): Total number of detected items.
        frame_num (int): Total number of frames processed.
        results_path (Path): Path to save the results file.
        item_name (str): Name of the item detected.
    """
    try:
        with open(results_path, "w", encoding="utf-8") as f:
            f.write("===== COUNTING RESULTS =====\n")
            f.write(f"Item detected: {item_name}\n")
            f.write(f"Total detected in video: {total_count}\n")
            f.write(f"Frames processed: {frame_num}\n")
            f.write(f"Average per frame: {total_count/frame_num:.2f}\n")

        logging.info(f"Metrics exported to {results_path}")
    except Exception as e:
        logging.error(f"Failed to write results file: {e}")





# ============================= #
#        MAIN (EXECUTION)       #
# ============================= #
if __name__ == "__main__":
    runs_dir = Path("runs/detect/yolo11_item_count")
    if not runs_dir.exists():
        logging.info("Training YOLOv11 model...")
        model = train_model()
    else:
        logging.info("Loading pretrained YOLOv11 model...")
        model = YOLO(str(runs_dir / "weights/best.pt"))

    logging.info("Processing video...")
    total_count = process_video(model, VIDEO_INPUT, VIDEO_OUTPUT)

    if total_count is not None:
        # For metrics we need frame count too → cv2 again
        cap = cv2.VideoCapture(str(VIDEO_INPUT))
        frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        export_metrics(total_count, frame_num, RESULTS_TXT)