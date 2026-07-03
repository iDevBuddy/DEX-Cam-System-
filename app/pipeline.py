"""Per-camera worker thread: capture → detect → track → classify → alert → annotate."""
import threading
import time

import cv2
import numpy as np
import supervision as sv

from . import db, detector
from .activity import ActivityTracker
from .alerts import AlertManager
from .camera import FrameSource, wait_backoff
from .zones import Zone

GREEN = (80, 200, 80)     # active
ORANGE = (0, 165, 255)    # idle
RED = (60, 60, 230)       # phone
CYAN = (200, 200, 40)     # zone


class CameraWorker(threading.Thread):
    def __init__(self, name: str, source, zone: Zone, cfg: dict):
        super().__init__(daemon=True, name=f"cam-{name}")
        self.cam_name = name
        self.source_str = str(source)
        self.zone = zone
        self.inf_cfg = cfg["inference"]
        self.tracker = sv.ByteTrack()
        self.activity = ActivityTracker(cfg["activity"])
        self.alerts = AlertManager(name, cfg["alerts"])
        self.status = "starting"
        self.counts = {"workers": 0, "active": 0, "idle": 0}
        self.stop_event = threading.Event()
        self._jpeg_lock = threading.Lock()
        self._latest_jpeg: bytes | None = None
        self._last_db_log = 0.0

    # ---------------- main loop ----------------

    def run(self):
        src = FrameSource(self.source_str)
        attempt = 0
        if not src.open():
            self.status = "reconnecting"
        interval = 1.0 / float(self.inf_cfg.get("infer_fps", 5))
        next_infer = 0.0

        while not self.stop_event.is_set():
            frame = src.read()
            if frame is None:
                self.status = "reconnecting"
                self._set_offline_frame()
                time.sleep(wait_backoff(attempt))
                attempt += 1
                src.open()
                continue
            attempt = 0
            self.status = "online"

            now = time.monotonic()
            if now < next_infer:
                if src.is_file:
                    time.sleep(0.03)  # pace file playback near real-time
                continue
            next_infer = now + interval

            try:
                self._process(frame)
            except Exception:
                # Never let one bad frame kill the camera thread.
                time.sleep(0.2)

        src.release()
        self.status = "stopped"

    def stop(self):
        self.stop_event.set()

    # ---------------- processing ----------------

    def _process(self, frame):
        h, w = frame.shape[:2]
        result = detector.predict(
            frame,
            conf=float(self.inf_cfg.get("confidence", 0.35)),
            imgsz=int(self.inf_cfg.get("imgsz", 640)),
        )
        det = sv.Detections.from_ultralytics(result)
        persons = det[det.class_id == detector.PERSON]
        phones = det[det.class_id == detector.CELL_PHONE]

        tracked = self.tracker.update_with_detections(persons)

        # Anchor = bottom-center of each box (feet position).
        anchors = []
        centroids = []
        for x1, y1, x2, y2 in tracked.xyxy:
            anchors.append(((x1 + x2) / 2, y2))
            centroids.append(((x1 + x2) / 2, (y1 + y2) / 2))

        track_ids = tracked.tracker_id if tracked.tracker_id is not None else []
        states = self.activity.update(track_ids, centroids, w)

        in_zone = sum(1 for a in anchors if self.zone.contains(a, w, h))
        active = sum(1 for s in states.values() if s == "active")
        idle = len(states) - active
        self.counts = {"workers": in_zone, "active": active, "idle": idle}

        annotated = self._annotate(frame, tracked, states, phones, w, h)
        self.alerts.check(in_zone, self.zone.max_workers, len(phones), annotated)
        self._encode(annotated)

        now = time.time()
        if now - self._last_db_log >= 1.0:
            self._last_db_log = now
            db.log_observation(self.cam_name, in_zone, active, idle)

    def _annotate(self, frame, tracked, states, phones, w, h):
        out = frame.copy()

        # Zone overlay (translucent fill + outline)
        poly = self.zone.pixels(w, h)
        overlay = out.copy()
        cv2.fillPoly(overlay, [poly], CYAN)
        cv2.addWeighted(overlay, 0.12, out, 0.88, 0, out)
        cv2.polylines(out, [poly], True, CYAN, 2)

        # Workers
        ids = tracked.tracker_id if tracked.tracker_id is not None else []
        for (x1, y1, x2, y2), tid in zip(tracked.xyxy, ids):
            state = states.get(int(tid), "active")
            color = GREEN if state == "active" else ORANGE
            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(out, p1, p2, color, 2)
            label = f"W{int(tid)} {state.upper()}"
            cv2.rectangle(out, (p1[0], p1[1] - 22), (p1[0] + 8 * len(label) + 8, p1[1]), color, -1)
            cv2.putText(out, label, (p1[0] + 4, p1[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Phones
        for x1, y1, x2, y2 in phones.xyxy:
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), RED, 2)
            cv2.putText(out, "PHONE", (int(x1), int(y1) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED, 2, cv2.LINE_AA)

        # Header bar
        c = self.counts
        bar = f"{self.cam_name}  |  workers: {c['workers']}  active: {c['active']}  idle: {c['idle']}"
        cv2.rectangle(out, (0, 0), (w, 30), (25, 25, 25), -1)
        cv2.putText(out, bar, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (240, 240, 240), 1, cv2.LINE_AA)
        return out

    # ---------------- frame output ----------------

    def _encode(self, frame):
        h, w = frame.shape[:2]
        if w > 960:  # keep MJPEG light
            frame = cv2.resize(frame, (960, int(h * 960 / w)))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            with self._jpeg_lock:
                self._latest_jpeg = buf.tobytes()

    def _set_offline_frame(self):
        img = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(img, f"{self.cam_name}: reconnecting...", (60, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2, cv2.LINE_AA)
        self._encode(img)

    def latest_jpeg(self) -> bytes | None:
        with self._jpeg_lock:
            return self._latest_jpeg
