"""Download AI models into models/ (runs once; skipped if already present)."""
from pathlib import Path
from urllib.request import urlretrieve

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS = {
    "yolov8n.pt":
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
    "pose_landmarker_lite.task":
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
}

MODELS_DIR.mkdir(parents=True, exist_ok=True)
for name, url in MODELS.items():
    dest = MODELS_DIR / name
    if dest.exists():
        print(f"Already present: {name}")
    else:
        print(f"Downloading {name} ...")
        urlretrieve(url, dest)
        print(f"  saved ({dest.stat().st_size // 1024} KB)")
