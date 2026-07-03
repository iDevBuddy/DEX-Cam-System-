"""Shared YOLOv8 model. One model instance serves all camera threads (lock-guarded)."""
import threading

from ultralytics import YOLO

PERSON = 0
CELL_PHONE = 67  # COCO class

_lock = threading.Lock()
_model = None
_model_path = None


def init(model_path: str):
    global _model, _model_path
    _model_path = model_path
    _model = YOLO(model_path)
    # Warm-up so the first real frame isn't slow.
    import numpy as np
    _model.predict(np.zeros((480, 640, 3), dtype="uint8"), verbose=False)


def predict(frame, conf: float, imgsz: int):
    """Returns ultralytics result for persons + cell phones only."""
    with _lock:
        results = _model.predict(
            frame, conf=conf, imgsz=imgsz,
            classes=[PERSON, CELL_PHONE], verbose=False,
        )
    return results[0]


def device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"
