"""Download AI models into models/ (runs once; skipped if already present)."""
from pathlib import Path
from urllib.request import urlretrieve

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS = {
    "yolo11s.pt":
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt",
    "yolo11n.pt":
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
    "pose_landmarker_lite.task":
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "pose_landmarker_full.task":
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    # OSNet person re-identification (torchreid model zoo, via Google Drive)
    "osnet_x1_0_msmt17.pt":
        "https://drive.google.com/uc?id=112EMUfBPYeYg70w-syK6V6Mx8-Qb9Q1M"
        "&export=download",
    "osnet_x0_25_msmt17.pt":
        "https://drive.google.com/uc?id=1sSwXSUlj4_tHZequ_iZ8w_Jh0VaRQMqF"
        "&export=download",
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
