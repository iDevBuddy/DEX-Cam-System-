"""Record raw clips from the configured cameras for offline work
(threshold tuning, model evaluation, Kaggle experiments).

Records every camera in config.yaml in parallel (or just the ones named),
raw frames, no annotation, to clips/<camera>_<YYYYmmdd-HHMMSS>.mp4.
Server ka chalte rehna theek hai — DVR kai RTSP sessions deta hai.

Usage:
    python tools/record_clips.py                       # all cameras, 60s
    python tools/record_clips.py --seconds 300         # all cameras, 5 min
    python tools/record_clips.py factory-cam-3 factory-cam-4 --seconds 120
"""
import argparse
import os
import threading
import time
from datetime import datetime
from pathlib import Path

# Same transport settings as the live pipeline (TCP, 5s timeout).
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS",
                      "rtsp_transport;tcp|stimeout;5000000")

import cv2
import yaml

ROOT = Path(__file__).resolve().parent.parent
CLIPS = ROOT / "clips"


def record(name: str, source: str, seconds: float, results: dict):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        results[name] = "FAILED: stream open nahi hua"
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if not 1 <= fps <= 60:   # DVRs often report 0/garbage
        fps = 12.0
    out_path = CLIPS / f"{name}_{datetime.now():%Y%m%d-%H%M%S}.mp4"
    writer = None
    frames = 0
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.2)
            continue
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(
                str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(frame)
        frames += 1
    cap.release()
    if writer is None:
        results[name] = "FAILED: koi frame nahi mila (camera dead?)"
        return
    writer.release()
    mb = out_path.stat().st_size / 1e6
    results[name] = f"OK: {out_path.name} — {frames} frames, {mb:.1f} MB"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cameras", nargs="*",
                    help="camera names (default: all in config.yaml)")
    ap.add_argument("--seconds", type=float, default=60.0)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    cams = {c["name"]: c["source"] for c in cfg.get("cameras") or []}
    picked = args.cameras or list(cams)
    unknown = [n for n in picked if n not in cams]
    if unknown:
        raise SystemExit(f"config.yaml mein nahi hain: {unknown} "
                         f"(available: {list(cams)})")

    CLIPS.mkdir(exist_ok=True)
    print(f"Recording {len(picked)} camera(s), {args.seconds:.0f}s each -> {CLIPS}")
    results: dict = {}
    threads = [threading.Thread(target=record,
                                args=(n, cams[n], args.seconds, results))
               for n in picked]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print()
    for name in picked:
        print(f"  {name}: {results.get(name, 'FAILED: thread died')}")


if __name__ == "__main__":
    main()
