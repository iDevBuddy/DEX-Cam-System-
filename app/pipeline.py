"""Per-camera worker thread: capture → detect → track → identify → classify →
alert → annotate.

Active/idle rule (owner's definition): a worker standing at a machine is
ACTIVE — even motionless, because operating a machine barely moves the body.
Away from every machine zone (or sitting anywhere) the worker is IDLE, no
matter how much they wander.
"""
import statistics
import threading
import time
from collections import deque

import cv2
import numpy as np
import supervision as sv

from . import db, detector, ids, reid
from .activity import ActivityTracker
from .alerts import AlertManager
from .camera import LatestFrameReader
from .pose import PostureClassifier
from .zones import MachineZones, Zone

GREEN = (80, 200, 80)     # active
ORANGE = (0, 165, 255)    # idle
RED = (60, 60, 230)       # phone
CYAN = (200, 200, 40)     # work zone
MAGENTA = (200, 80, 220)  # machine zones
GREY = (140, 140, 140)    # suspected non-human

MACHINE_GRACE = 3.0       # s a worker stays "at machine" after stepping off
EMBED_EVERY = 2.0         # s between re-id samples per track

# Object filter: a "person" is reclassified as an object ONLY when every one
# of these holds for a full minute — pose never found, literally zero pixel
# movement, and weak detector confidence. Any single movement or one
# successful pose check makes the track immune for life. Deliberately biased:
# losing a phantom takes a minute; losing a real worker must never happen.
OBJECT_AFTER = 60.0       # s all conditions must hold before suspecting
OBJECT_MAX_CONF = 0.50    # mean YOLO confidence below this (machines score low)
OBJECT_MOVE_PX = 0.004    # movement above this fraction of width = human


class CameraWorker(threading.Thread):
    def __init__(self, name: str, source, zone: Zone, cfg: dict,
                 process: bool = True, confidence: float | None = None,
                 machine_zones: list[dict] | None = None):
        super().__init__(daemon=True, name=f"cam-{name}")
        self.cam_name = name
        self.source_str = str(source)
        self.process_enabled = process  # False => live view only, no AI
        self.conf = float(confidence) if confidence else float(
            cfg["inference"].get("confidence", 0.35))
        self.zone = zone
        self.machines = MachineZones(machine_zones)
        self.inf_cfg = cfg["inference"]
        self.tracker = sv.ByteTrack()
        self.activity = ActivityTracker(cfg["activity"])
        self.alerts = AlertManager(name, cfg["alerts"])
        self.status = "starting"
        self.counts = {"workers": 0, "active": 0, "idle": 0, "at_machine": 0}
        self.live: dict[int, dict] = {}  # raw tid -> display/state/posture/machine/pid
        self.stop_event = threading.Event()
        self._jpeg_lock = threading.Lock()
        self._latest_jpeg: bytes | None = None
        self._last_db_log = 0.0
        # Stability: median of recent counts kills 0→1→0 flicker.
        self._count_hist = deque(maxlen=5)
        self._active_hist = deque(maxlen=5)
        self._idle_hist = deque(maxlen=5)
        self._machine_hist = deque(maxlen=5)
        # Worker sessions: raw track id -> accounting
        self._sessions: dict[int, dict] = {}
        # Posture (MediaPipe): checked at most once per second per track
        self.posture_enabled = bool(cfg["activity"].get("use_pose", True))
        self._posture = PostureClassifier() if self.posture_enabled else None
        self._posture_cache: dict[int, dict] = {}  # tid -> {"p": str|None, "t": mono}
        self._idle_since: dict[int, float] = {}    # tid -> mono when idle streak began
        # Identity (OSNet re-id) — shared gallery across all cameras
        self.reid = reid.shared() if cfg["inference"].get("use_reid", True) else None
        if self.reid and not self.reid.ok:
            self.reid = None
        self._t_person: dict[int, int] = {}        # raw tid -> person id
        self._t_pending: dict[int, list] = {}      # raw tid -> unmatched embeddings
        self._t_embed_ts: dict[int, float] = {}    # raw tid -> last embed time
        self._t_machine_ts: dict[int, float] = {}  # raw tid -> last at-machine time
        self._t_machine_name: dict[int, str] = {}
        # Legacy display numbers (used only when re-id is unavailable)
        self._display: dict[int, int] = {}
        self._raw_seen: dict[int, float] = {}
        # Object filter (insan hai ya object?): see OBJECT_* constants above.
        self._t_first: dict[int, float] = {}       # tid -> mono when first seen
        self._t_conf: dict[int, list] = {}         # tid -> [sum, n] of YOLO conf
        self._t_human: set[int] = set()            # immune: moved or posed once
        self._suspect: set[int] = set()

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
                if self.process_enabled:
                    self._process(frame)
                else:
                    self._encode_view_only(frame)
            except Exception:
                # Never let one bad frame kill the camera thread.
                time.sleep(0.2)
            # Keep our own pace; the reader keeps the frame fresh meanwhile.
            pace = interval if self.process_enabled else 0.12
            time.sleep(max(0.0, pace - (time.monotonic() - t0)))

        reader.stop()
        if self.reid:
            self.reid.flush()
        self.status = "stopped"

    def stop(self):
        self.stop_event.set()

    # ---------------- processing ----------------

    def _process(self, frame):
        h, w = frame.shape[:2]
        result = detector.predict(
            frame,
            conf=self.conf,
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
        raw_ids = [int(r) for r in (
            tracked.tracker_id if tracked.tracker_id is not None else [])]

        # Anchor = bottom-center of each box (feet position).
        anchors, centroids = [], []
        for x1, y1, x2, y2 in tracked.xyxy:
            anchors.append(((x1 + x2) / 2, y2))
            centroids.append(((x1 + x2) / 2, (y1 + y2) / 2))

        move_states = self.activity.update(raw_ids, centroids, w)
        postures = self._update_postures(frame, tracked, raw_ids, w, h)
        self._update_object_filter(tracked, raw_ids, postures, w)
        self._update_identity(frame, tracked, raw_ids, w, h)

        # Who is standing at which machine right now (with a short grace so a
        # half-step back doesn't flicker the state).
        at_machine: dict[int, str | None] = {}
        now_m = time.monotonic()
        for tid, anchor in zip(raw_ids, anchors):
            name = self.machines.at(anchor, w, h) if self.machines else None
            if name:
                self._t_machine_ts[tid] = now_m
                self._t_machine_name[tid] = name
            elif now_m - self._t_machine_ts.get(tid, -1e9) <= MACHINE_GRACE:
                name = self._t_machine_name.get(tid)
            at_machine[tid] = name

        # ACTIVE/IDLE per the owner's rule.
        states: dict[int, str] = {}
        for tid in raw_ids:
            if postures.get(tid) == "sitting":
                states[tid] = "idle"          # sitting is never work here
            elif self.machines:
                states[tid] = "active" if at_machine[tid] else "idle"
            elif postures.get(tid) == "standing":
                states[tid] = "active"
            else:
                states[tid] = move_states.get(tid, "active")

        # Suspected non-humans drop out of every count and report.
        humans = [t for t in raw_ids if t not in self._suspect]
        h_states = {t: states[t] for t in humans}

        in_zone = raw_active = raw_idle = raw_machine = 0
        for tid, anchor in zip(raw_ids, anchors):
            if tid in self._suspect:
                continue
            if self.zone.contains(anchor, w, h):
                in_zone += 1
            if states[tid] == "active":
                raw_active += 1
            else:
                raw_idle += 1
            if at_machine.get(tid):
                raw_machine += 1

        # Smoothed counts (median over ~1.2s) — steady numbers for UI + alerts.
        self._count_hist.append(in_zone)
        self._active_hist.append(raw_active)
        self._idle_hist.append(raw_idle)
        self._machine_hist.append(raw_machine)
        self.counts = {
            "workers": int(statistics.median(self._count_hist)),
            "active": int(statistics.median(self._active_hist)),
            "idle": int(statistics.median(self._idle_hist)),
            "at_machine": int(statistics.median(self._machine_hist)),
        }

        self._update_sessions(h_states, postures, at_machine)
        self.live = {
            tid: {
                "display": self._label(tid),
                "pid": self._t_person.get(tid),
                "state": st,
                "posture": postures.get(tid),
                "machine": at_machine.get(tid),
            }
            for tid, st in h_states.items()
        }

        annotated = self._annotate(frame, tracked, raw_ids, states, phones,
                                   w, h, postures, at_machine)
        self.alerts.check(self.counts["workers"], self.zone.max_workers,
                          len(phones), annotated)
        self._check_idle_workers(h_states, annotated)
        self._encode(annotated)

        now = time.time()
        if now - self._last_db_log >= 1.0:
            self._last_db_log = now
            db.log_observation(self.cam_name, self.counts["workers"],
                               self.counts["active"], self.counts["idle"])
            if self.reid:
                for tid in humans:
                    pid = self._t_person.get(tid)
                    if pid is not None:
                        self.reid.tick(pid, 1.0, bool(at_machine.get(tid)))

    # ---------------- identity ----------------

    def _label(self, tid: int) -> str:
        """What to call this track on screen and in reports."""
        if self.reid:
            pid = self._t_person.get(tid)
            return self.reid.display(pid) if pid is not None else "..."
        return f"W{self._display.get(tid, tid)}"

    def _update_identity(self, frame, tracked, raw_ids, w, h):
        """Sample an appearance embedding every couple of seconds per track and
        resolve it to a persistent person (or eventually create a new one)."""
        now = time.monotonic()
        if not self.reid:
            # Legacy small display numbers (W1, W2...) per camera.
            for rid in raw_ids:
                self._raw_seen[rid] = now
                if rid not in self._display:
                    self._display[rid] = ids.acquire()
            for rid in [r for r, t in self._raw_seen.items() if now - t > 6.0]:
                self._raw_seen.pop(rid, None)
                disp = self._display.pop(rid, None)
                if disp is not None:
                    ids.release(disp)
            return

        for (x1, y1, x2, y2), tid in zip(tracked.xyxy, raw_ids):
            if tid in self._suspect:
                continue
            if now - self._t_embed_ts.get(tid, -1e9) < EMBED_EVERY:
                continue
            self._t_embed_ts[tid] = now
            x1i, y1i = max(0, int(x1)), max(0, int(y1))
            x2i, y2i = min(w, int(x2)), min(h, int(y2))
            crop = frame[y1i:y2i, x1i:x2i]
            emb = self.reid.embed(crop)
            if emb is None:
                continue
            pid = self._t_person.get(tid)
            if pid is None:
                pid = self.reid.assign(emb)
                if pid is not None:
                    self._t_person[tid] = pid
                    self._t_pending.pop(tid, None)
                else:
                    buf = self._t_pending.setdefault(tid, [])
                    buf.append(emb)
                    if len(buf) >= reid.NEW_AFTER:
                        pid = self.reid.create(buf)
                        self._t_person[tid] = pid
                        self._t_pending.pop(tid, None)
            if pid is not None:
                self.reid.offer_crop(pid, crop)

        # Forget tracks that left the scene.
        gone = [t for t in self._t_embed_ts
                if now - self._t_embed_ts.get(t, 0) > 30.0 and t not in raw_ids]
        for t in gone:
            for d in (self._t_person, self._t_pending, self._t_embed_ts,
                      self._t_machine_ts, self._t_machine_name):
                d.pop(t, None)

    # ---------------- sessions ----------------

    def _update_sessions(self, states: dict, postures: dict, at_machine: dict):
        """Track each worker's visit: first seen → last seen, with active ratio,
        machine-time ratio and dominant posture. A session closes after the
        worker is gone for 5s; visits under 15s are noise and dropped."""
        now = time.time()
        for tid, state in states.items():
            s = self._sessions.setdefault(
                tid, {"start": now, "last": now, "active_n": 0, "total_n": 0,
                      "sit_n": 0, "stand_n": 0, "machine_n": 0}
            )
            s["last"] = now
            s["total_n"] += 1
            if state == "active":
                s["active_n"] += 1
            if at_machine.get(tid):
                s["machine_n"] += 1
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
                    person_id=self._t_person.get(tid),
                    machine_pct=round(100.0 * s["machine_n"] / s["total_n"], 1),
                )

    def _check_idle_workers(self, states: dict, frame):
        """Fire a photo alert for any worker idle past the threshold."""
        now = time.monotonic()
        for tid, state in states.items():
            if state == "idle":
                t0 = self._idle_since.setdefault(tid, now)
                elapsed = now - t0
                if elapsed >= self.alerts.idle_worker_after:
                    self.alerts.fire_idle_worker(self._label(tid), int(elapsed), frame)
            else:
                self._idle_since.pop(tid, None)
        for tid in [t for t in self._idle_since if t not in states]:
            self._idle_since.pop(tid, None)

    # ---------------- posture & human check ----------------

    def _update_postures(self, frame, tracked, raw_ids, w, h) -> dict:
        """Posture per visible worker, refreshed at most 1x/sec per track."""
        if not self._posture or not self._posture.ok:
            return {}
        now = time.monotonic()
        postures = {}
        for (x1, y1, x2, y2), tid in zip(tracked.xyxy, raw_ids):
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

    def _update_object_filter(self, tracked, raw_ids, postures, w):
        """Insan hai ya object? A track earns permanent human status from a
        single detected pose or a single real movement. Only a track that does
        neither for a full minute AND scores low detector confidence is set
        aside as a machine part misread as a person."""
        now = time.monotonic()
        confs = (tracked.confidence if tracked.confidence is not None
                 else [1.0] * len(raw_ids))
        for tid, conf in zip(raw_ids, confs):
            self._t_first.setdefault(tid, now)
            c = self._t_conf.setdefault(tid, [0.0, 0])
            c[0] += float(conf)
            c[1] += 1
            if tid in self._t_human:
                continue
            spread = self.activity.spread(tid)
            moved = spread is not None and spread > OBJECT_MOVE_PX * w
            if moved or postures.get(tid) is not None:
                self._t_human.add(tid)
                self._suspect.discard(tid)
                continue
            if (now - self._t_first[tid] >= OBJECT_AFTER
                    and c[0] / max(c[1], 1) < OBJECT_MAX_CONF):
                self._suspect.add(tid)
        # Drop bookkeeping for tracks gone > 60s.
        live = set(raw_ids)
        for tid in [t for t in self._t_first
                    if t not in live and now - self._t_first[t] > 60.0]:
            self._t_first.pop(tid, None)
            self._t_conf.pop(tid, None)
            self._t_human.discard(tid)
            self._suspect.discard(tid)

    # ---------------- drawing ----------------

    def _annotate(self, frame, tracked, raw_ids, states, phones, w, h,
                  postures=None, at_machine=None):
        out = frame.copy()

        # Work zone (cyan) + machine zones (magenta)
        poly = self.zone.pixels(w, h)
        overlay = out.copy()
        cv2.fillPoly(overlay, [poly], CYAN)
        for name, z in (self.machines.zones if self.machines else []):
            cv2.fillPoly(overlay, [z.pixels(w, h)], MAGENTA)
        cv2.addWeighted(overlay, 0.12, out, 0.88, 0, out)
        cv2.polylines(out, [poly], True, CYAN, 2)
        for name, z in (self.machines.zones if self.machines else []):
            mp = z.pixels(w, h)
            cv2.polylines(out, [mp], True, MAGENTA, 2)
            tx, ty = mp[0][0], mp[0][1]
            cv2.putText(out, name, (int(tx) + 4, int(ty) + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, MAGENTA, 1, cv2.LINE_AA)

        # Workers
        for (x1, y1, x2, y2), tid in zip(tracked.xyxy, raw_ids):
            tid = int(tid)
            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            if tid in self._suspect:
                cv2.rectangle(out, p1, p2, GREY, 1)
                cv2.putText(out, "not a person", (p1[0], p1[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREY, 1, cv2.LINE_AA)
                continue
            state = states.get(tid, "active")
            color = GREEN if state == "active" else ORANGE
            cv2.rectangle(out, p1, p2, color, 2)
            label = f"{self._label(tid)} {state.upper()}"
            m = (at_machine or {}).get(tid)
            if m:
                label += f" @ {m}"
            p = (postures or {}).get(tid)
            if p:
                label += " - SITTING" if p == "sitting" else " - STANDING"
            cv2.rectangle(out, (p1[0], p1[1] - 22),
                          (p1[0] + 8 * len(label) + 8, p1[1]), color, -1)
            cv2.putText(out, label, (p1[0] + 4, p1[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Phones
        for x1, y1, x2, y2 in phones.xyxy:
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), RED, 2)
            cv2.putText(out, "PHONE", (int(x1), int(y1) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED, 2, cv2.LINE_AA)

        # Header bar
        c = self.counts
        bar = (f"{self.cam_name}  |  workers: {c['workers']}  "
               f"active: {c['active']}  idle: {c['idle']}")
        if self.machines:
            bar += f"  at machine: {c['at_machine']}"
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

    def _encode_view_only(self, frame):
        """No AI — just the live picture with a small header."""
        out = frame.copy()
        h, w = out.shape[:2]
        cv2.rectangle(out, (0, 0), (w, 30), (25, 25, 25), -1)
        cv2.putText(out, f"{self.cam_name}  |  LIVE VIEW", (10, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
        self._encode(out)

    def _set_offline_frame(self):
        img = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(img, f"{self.cam_name}: reconnecting...", (60, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2, cv2.LINE_AA)
        self._encode(img)

    def latest_jpeg(self) -> bytes | None:
        with self._jpeg_lock:
            return self._latest_jpeg
