"""Download YOLOv8n weights into models/ (runs once; skipped if already present)."""
from pathlib import Path
from urllib.request import urlretrieve

URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
DEST = Path(__file__).resolve().parent.parent / "models" / "yolov8n.pt"

if DEST.exists():
    print(f"Model already present: {DEST}")
else:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading yolov8n.pt ...")
    urlretrieve(URL, DEST)
    print(f"Saved to {DEST} ({DEST.stat().st_size // 1024} KB)")
