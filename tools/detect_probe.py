"""Raw-detection probe: EXACTLY what does the detector see on one camera,
before any filter can hide it?

Runs the detector at a rock-bottom 0.05 confidence for N seconds and logs
every single detection with a verdict for each stage of the pipeline:
  - min-height filter (bbox >= 5% of frame height)
  - track-start bar (camera's 'confidence' in config.yaml)
  - track-continue floor (0.12 — keeps existing tracks alive)
  - work-zone membership (feet point)
  - machine-zone membership
  - re-id crop size (identity only; never blocks boxes/counts)

Usage (on-site, cameras online):
    python tools/detect_probe.py factory-cam-5
    python tools/detect_probe.py factory-cam-5 --seconds 30
    python tools/detect_probe.py rtsp://user:pass@ip:554/... --imgsz 512
    python tools/detect_probe.py path/to/video.mp4

Also saves an annotated frame (EVERY raw detection drawn, even 0.05 conf)
to snapshots/probe_<name>_<time>.jpg — open it to see what YOLO saw.
The running server can stay up; Hikvision allows several RTSP sessions.
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FLOOR = 0.05          # probe floor — see everything the model even whispers
TRACK_CONTINUE = 0.12  # pipeline's DETECT_FLOOR (keeps existing tracks alive)
MIN_H_FRAC = 0.05      # pipeline's min-height filter
REID_MIN_H, REID_MIN_W = 96, 32


def load_camera(arg: str):
    """Camera name from config.yaml, or treat arg as a raw source."""
    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    inf = cfg["inference"]
    for cam in cfg.get("cameras") or []:
        if cam["name"] == arg:
            return cam, inf
    src = 0 if arg == "0" else arg
    return {"name": "adhoc", "source": src, "zone": None,
            "machine_zones": None, "process": True,
            "confidence": inf.get("confidence", 0.35)}, inf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("camera", help="camera name from config.yaml, RTSP URL, video file, or 0")
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--imgsz", type=int, default=None)
    args = ap.parse_args()

    cam, inf = load_camera(args.camera)
    imgsz = args.imgsz or int(inf.get("imgsz", 512))
    start_bar = float(cam.get("confidence") or inf.get("confidence", 0.35))

    print(f"PROBE: {cam['name']}  source={cam['source']}")
    print(f"  model={inf['model']}  imgsz={imgsz}  probe-floor={FLOOR}")
    print(f"  camera track-START bar={start_bar}  track-CONTINUE floor={TRACK_CONTINUE}")
    if cam.get("process", True) is False:
        print("  *** WARNING: is camera ka process=false hai — dashboard pe AI")
        print("  *** isi liye band hai. Probe phir bhi chalega (apna detector).")

    from app import detector, enhance
    from app.zones import MachineZones, Zone
    detector.init(inf["model"])
    zone = Zone(cam["zone"], 99) if cam.get("zone") else None
    machines = MachineZones(cam.get("machine_zones"))

    import os
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    cap = cv2.VideoCapture(cam["source"], cv2.CAP_FFMPEG) \
        if isinstance(cam["source"], str) else cv2.VideoCapture(cam["source"])
    if not cap.isOpened():
        print("!! Source nahi khula — camera online hai? URL sahi hai?")
        sys.exit(1)

    t_end = time.time() + args.seconds
    passes = 0
    all_confs: list[float] = []
    n_start = n_continue_only = n_too_small = n_in_zone = n_at_machine = 0
    best_frame, best_score = None, -1.0

    while time.time() < t_end:
        ok, frame = cap.read()
        if not ok:
            if isinstance(cam["source"], str) and not str(cam["source"]).startswith("rtsp"):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop video files
                continue
            print("  (frame drop — reconnect...)")
            time.sleep(1)
            cap.release()
            cap = cv2.VideoCapture(cam["source"], cv2.CAP_FFMPEG)
            continue
        if bool(inf.get("enhance", True)):
            frame = enhance.maybe_enhance(frame)
        h, w = frame.shape[:2]
        passes += 1
        r = detector.predict(frame, conf=FLOOR, imgsz=imgsz)
        boxes = r.boxes
        rows = []
        frame_score = 0.0
        draw = frame.copy()
        for b in boxes:
            cls = int(b.cls[0])
            if cls != detector.PERSON:
                continue
            conf = float(b.conf[0])
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            bh, bw = y2 - y1, x2 - x1
            feet = ((x1 + x2) / 2, y2)
            too_small = bh < MIN_H_FRAC * h
            starts = conf >= start_bar and not too_small
            continues = conf >= TRACK_CONTINUE and not too_small
            in_zone = zone.contains(feet, w, h) if zone else True
            at_machine = machines.at(feet, w, h) if machines else None
            reid_ok = bh >= REID_MIN_H and bw >= REID_MIN_W

            all_confs.append(conf)
            frame_score += conf
            if too_small:
                n_too_small += 1
            elif starts:
                n_start += 1
            elif continues:
                n_continue_only += 1
            if in_zone and not too_small:
                n_in_zone += 1
            if at_machine and not too_small:
                n_at_machine += 1

            verdict = ("DROP(too-small)" if too_small
                       else "STARTS-TRACK" if starts
                       else "continue-only" if continues
                       else "below-floor")
            rows.append(
                f"    person conf={conf:.2f} box={int(bw)}x{int(bh)}px "
                f"({100 * bh / h:.0f}% tall) {verdict}"
                + (" | in-zone" if in_zone else " | OUTSIDE-zone")
                + (f" | @{at_machine}" if at_machine else "")
                + ("" if reid_ok else " | crop-too-small-for-ID")
            )
            color = ((60, 200, 60) if starts else
                     (0, 200, 255) if continues else (0, 0, 230))
            cv2.rectangle(draw, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(draw, f"{conf:.2f}", (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        print(f"[{time.strftime('%H:%M:%S')}] pass {passes}: {len(rows)} person(s)")
        for line in rows:
            print(line)
        if frame_score > best_score:
            best_score, best_frame = frame_score, draw

    cap.release()
    print()
    print("=" * 60)
    print(f"SUMMARY ({args.seconds:.0f}s, {passes} passes)")
    if not all_confs:
        print("  ZERO person detections at conf 0.05 —")
        print("  => model ko banda dikh hi nahi raha (angle/roshni/size ka masla),")
        print("     filters ka qusoor NAHI hai.")
    else:
        print(f"  detections: {len(all_confs)}  "
              f"max conf={max(all_confs):.2f}  median={statistics.median(all_confs):.2f}")
        print(f"  would START a track (>= {start_bar}) : {n_start}")
        print(f"  continue-only (0.12..{start_bar})    : {n_continue_only}"
              f"   <- track sirf tab zinda jab pehle ban chuka ho")
        print(f"  dropped too-small (<5% frame height) : {n_too_small}")
        print(f"  feet inside work zone                : {n_in_zone}")
        print(f"  feet inside a machine zone           : {n_at_machine}")
        if n_start == 0 and n_continue_only > 0:
            print("  => VERDICT: detections hain magar START bar se neeche —")
            print(f"     is camera ki 'confidence' ({start_bar}) kam karni hogi.")
        elif n_start > 0:
            print("  => VERDICT: detections start-bar cross karti hain — agar dashboard")
            print("     pe phir bhi kuch nahi to process flag / server restart check karo.")
    if best_frame is not None:
        out = ROOT / "snapshots" / f"probe_{cam['name']}_{int(time.time())}.jpg"
        out.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(out), best_frame)
        print(f"  annotated frame saved: {out}")
        print("  (green=starts track, orange=continue-only, red=below floor)")


if __name__ == "__main__":
    main()
