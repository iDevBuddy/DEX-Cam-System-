"""Per-camera worker thread: capture → detect → track → classify → alert → annotate."""
import statistics
import threading
import time
from collections import deque

import cv2
import numpy as np
import supervision as sv

from . import db, detector
from .activity import ActivityTracker
from .alerts import AlertManager
from .camera import LatestFrameReader
from .pose import PostureClassifier
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
        self.live: dict[int, dict] = {}  # currently visible: tid -> state/posture
        self.stop_event = threading.Event()
        self._jpeg_lock = threading.Lock()
        self._latest_jpeg: bytes | None = None
        self._last_db_log = 0.0
        # Stability: median of recent counts kills 0→1→0 flicker.
        self._count_hist = deque(maxlen=5)
        self._active_hist = deque(maxlen=5)
        self._idle_hist = deque(maxlen=5)
        # Worker sessions: track_id -> accounting
        self._sessions: dict[int, dict] = {}
        # Posture (MediaPipe): checked at most once per second per worker
        self.posture_enabled = bool(cfg["activity"].get("use_pose", True))
        self._posture = PostureClassifier() if self.posture_enabled else None
        self._posture_cache: dict[int, dict] = {}  # tid -> {"p": str|None, "t": mono}
        self._idle_since: dict[int, float] = {}    # tid -> mono when idle streak began

    # ---------------- main loop ----------------

    def run(self):
        reader = LatestFrameReader(self.source_str)
        reader.start()
        interval = 1.0 / float(self.inf_cfg.get("infer_fps", 5))

        while not self.stop_event.is_set():
            if not reader.connected:
                self.status = "reconnecting"
                self._set_offline_frame()
                time.sleep(1.0)
                continue
            frame = reader.latest()
            if frame is None:
                time.sleep(0.05)
                continue
            self.status = "online"

            t0 = time.monotonic()
            try:
                self._process(frame)
            except Exception:
                # Never let one bad frame kill the camera thread.
                time.sleep(0.2)
            # Keep our own pace; the reader keeps the frame fresh meanwhile.
            time.sleep(max(0.0, interval - (time.monotonic() - t0)))

        reader.stop()
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

        # Drop implausibly small "people" (far-away noise, reflections).
        if len(persons) > 0:
            heights = persons.xyxy[:, 3] - persons.xyxy[:, 1]
            persons = persons[heights >= 0.05 * h]

        tracked = self.tracker.update_with_detections(persons)

        # Anchor = bottom-center of each box (feet position).
        anchors = []
        centroids = []
        for x1, y1, x2, y2 in tracked.xyxy:
            anchors.append(((x1 + x2) / 2, y2))
            centroids.append(((x1 + x2) / 2, (y1 + y2) / 2))

        track_ids = tracked.tracker_id if tracked.tracker_id is not None else []
        states = self.activity.update(track_ids, centroids, w)
        postures = self._update_postures(frame, tracked, w, h)
        # Posture overrides movement: a seated worker is idle even if fidgeting.
        for tid, p in postures.items():
            if p == "sitting":
                states[tid] = "idle"

        raw_in_zone = sum(1 for a in anchors if self.zone.contains(a, w, h))
        raw_active = sum(1 for s in states.values() if s == "active")
        raw_idle = len(states) - raw_active

        # Smoothed counts (median over ~1.2s) — steady numbers for UI + alerts.
        self._count_hist.append(raw_in_zone)
        self._active_hist.append(raw_active)
        self._idle_hist.append(raw_idle)
        in_zone = int(statistics.median(self._count_hist))
        active = int(statistics.median(self._active_hist))
        idle = int(statistics.median(self._idle_hist))
        self.counts = {"workers": in_zone, "active": active, "idle": idle}

        self._update_sessions(states, postures)
        self.live = {
            tid: {"state": st, "posture": postures.get(tid)}
            for tid, st in states.items()
        }

        annotated = self._annotate(frame, tracked, states, phones, w, h, postures)
        self.alerts.check(in_zone, self.zone.max_workers, len(phones), annotated)
        self._check_idle_workers(states, annotated)
        self._encode(annotated)

        now = time.time()
        if now - self._last_db_log >= 1.0:
            self._last_db_log = now
            db.log_observation(self.cam_name, in_zone, active, idle)

    def _update_sessions(self, states: dict, postures: dict | None = None):
        """Track each worker's visit: first seen → last seen, with active ratio
        and dominant posture. A session closes after the worker is gone for 5s;
        visits under 15s are noise and dropped."""
        now = time.time()
        postures = postures or {}
        for tid, state in states.items():
            s = self._sessions.setdefault(
                tid, {"start": now, "last": now, "active_n": 0, "total_n": 0,
                      "sit_n": 0, "stand_n": 0}
            )
            s["last"] = now
            s["total_n"] += 1
            if state == "active":
                s["active_n"] += 1
            p = postures.get(tid)
            if p == "sitting":
                s["sit_n"] += 1
            elif p == "standing":
                s["stand_n"] += 1

        for tid in [t for t, s in self._sessions.items() if now - s["last"] > 5.0]:
            s = self._sessions.pop(tid)
            duration = s["last"] - s["start"]
            if duration >= 15.0 and s["total_n"] > 0:
                if s["sit_n"] == 0 and s["stand_n"] == 0:
                    posture = None
                else:
                    posture = "sitting" if s["sit_n"] > s["stand_n"] else "standing"
                db.log_session(
                    self.cam_name, tid, s["start"], s["last"], duration,
                    round(100.0 * s["active_n"] / s["total_n"], 1), posture,
                )

    def _check_idle_workers(self, states: dict, frame):
        """Fire a photo alert for any worker idle past the threshold."""
        now = time.monotonic()
        for tid, state in states.items():
            if state == "idle":
                t0 = self._idle_since.setdefault(tid, now)
                elapsed = now - t0
                if elapsed >= self.alerts.idle_worker_after:
                    self.alerts.fire_idle_worker(tid, int(elapsed), frame)
            else:
                self._idle_since.pop(tid, None)
        for tid in [t for t in self._idle_since if t not in states]:
            self._idle_since.pop(tid, None)

    def _update_postures(self, frame, tracked, w, h) -> dict:
        """Posture per visible worker, refreshed at most 1x/sec per track."""
        if not self._posture or not self._posture.ok:
            return {}
        now = time.monotonic()
        postures = {}
        ids = tracked.tracker_id if tracked.tracker_id is not None else []
        for (x1, y1, x2, y2), tid in zip(tracked.xyxy, ids):
            tid = int(tid)
            cached = self._posture_cache.get(tid)
            if cached and now - cached["t"] < 1.0:
                postures[tid] = cached["p"]
                continue
            x1i, y1i = max(0, int(x1)), max(0, int(y1))
            x2i, y2i = min(w, int(x2)), min(h, int(y2))
            p = self._posture.posture(frame[y1i:y2i, x1i:x2i])
            self._posture_cache[tid] = {"p": p, "t": now}
            postures[tid] = p
        # forget stale tracks
        for tid in [t for t, c in self._posture_cache.items() if now - c["t"] > 10.0]:
            self._posture_cache.pop(tid, None)
        return postures

    def _annotate(self, frame, tracked, states, phones, w, h, postures=None):
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
            p = (postures or {}).get(int(tid))
            if p:
                label += " - SITTING" if p == "sitting" else " - STANDING"
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
