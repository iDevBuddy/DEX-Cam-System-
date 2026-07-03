"""Build sample.mp4 for offline demo testing: a slow pan over a real photo
containing people, so YOLO has something genuine to detect without any camera."""
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "models" / "_sample_src.jpg"
OUT = ROOT / "sample.mp4"

if not IMG.exists():
    print("Downloading source image (contains people)...")
    urlretrieve("https://ultralytics.com/images/bus.jpg", IMG)

img = cv2.imread(str(IMG))
H, W = img.shape[:2]
vw, vh = 640, 360
fps = 25
seconds = 30

writer = cv2.VideoWriter(str(OUT), cv2.VideoWriter_fourcc(*"mp4v"), fps, (vw, vh))
crop_w = int(W * 0.75)
crop_h = int(crop_w * vh / vw)
crop_h = min(crop_h, H)
max_x = W - crop_w
max_y = H - crop_h

n = fps * seconds
for i in range(n):
    # Slow sinusoidal pan — keeps "workers" moving in and out of frame.
    t = i / n
    x = int(max_x * (0.5 + 0.5 * np.sin(2 * np.pi * t * 2)))
    y = int(max_y * (0.5 + 0.5 * np.cos(2 * np.pi * t)))
    crop = img[y:y + crop_h, x:x + crop_w]
    writer.write(cv2.resize(crop, (vw, vh)))
writer.release()
print(f"Wrote {OUT} ({seconds}s, {vw}x{vh})")
