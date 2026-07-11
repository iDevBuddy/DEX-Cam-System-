"""Per-camera worker thread: capture → detect → track → identify → classify →
alert → annotate.

Active/idle rules (v2 — owner's definition, machine-presence based):
  1. Person at a machine zone + machine RUNNING  = ACTIVE. Always.
  2. Person at a machine zone + machine STOPPED  = ACTIVE (setup/measuring).
  3. At no machine: NEUTRAL, turning IDLE after idle_after_minutes.
  4. Machine A -> B within machine_switch_window_seconds = one continuous
     ACTIVE story (re-id keeps identity across cameras), switch logged.
  5. Posture NEVER decides active/idle — it is a separate discipline layer:
     sitting at a machine past sitting_alert_minutes fires a POSTURE alert.
  6. Sticky states: entering ACTIVE takes 2 consecutive AI passes; leaving
     takes the full switch window of contrary evidence. No flicker.
"""
import statistics
import threading
import time
from collections import deque

import cv2
import numpy as np
import supervision as sv

from . import db, detector, enhance, ids, personstate, reid
from .activity import ActivityTracker
from .alerts import AlertManager
from .camera import LatestFrameReader
from .machinestate import MachineStateTracker
from .pose import PostureClassifier
from .zones import MachineZones, Zone

GREEN = (80, 200, 80)     # active
GREY = (150, 150, 150)    # neutral
ORANGE = (0, 165, 255)    # idle
RED = (60, 60, 230)       # phone
CYAN = (200, 200, 40)     # work zone
MAGENTA = (200, 80, 220)  # machine zone (stopped)
LIME = (90, 230, 130)     # machine zone (running)

MACHINE_GRACE = 3.0       # s a worker keeps a machine through a detection blip
EMBED_EVERY = 2.0         # s between re-id samples per track
ENTER_ACTIVE_PASSES = 2   # consecutive at-machine passes to turn ACTIVE (~1s)

# Posture compliance layer (never touches active/idle):
POSTURE_WINDOW = 30.0     # s of posture history considered
POSTURE_SIT_RATIO = 0.7   # sitting fraction of known samples to call "sitting"
POSTURE_MIN_SAMPLES = 20  # known samples needed — never guess from noise

# Two-threshold detection (the low-quality-camera fix). The detector runs at
# a floor confidence so weak evidence is never thrown away; ByteTrack only
# STARTS a track from a strong detection (per-camera threshold) but KEEPS an
# existing track alive on weak ones.
DETECT_FLOOR = 0.12       # detector floor; never used to create tracks
PHONE_CONF = 0.35         # phones still need a confident detection
DISPLAY_FPS = 8.0         # smooth video between (slower) AI passes

# Auto-pause on dead camera: a dead DVR channel either stops delivering
# frames (already handled as "reconnecting") or keeps serving a bit-identical
# "NO VIDEO" placeholder. If the picture doesn't CHANGE for PAUSE_AFTER, AI
# pauses itself and resumes the moment a live picture is back. Change =
# fraction of pixels that moved, so the DVR's ticking clock overlay (a few
# dozen pixels) doesn't count as life, while real sensor/compression noise
# (spread over the frame) does.
PAUSE_AFTER = 300.0       # s of frozen picture -> AI pauses itself
SIG_SAMPLE = 2.0          # s between picture-change samples
SIG_PIX_DELTA = 6         # gray-level delta for one pixel to count as changed
SIG_CHANGED_FRAC = 0.005  # >0.5% of pixels changed = picture is alive

# Object filter: a "person" is reclassified as an object when it stays weak
# (peak conf below OBJECT_MAX_CONF), pose is never found, and it does not
# move for OBJECT_AFTER — or, new, when it lives OBJECT_AFTER_JITTERY without
# EVER producing real human evidence, even if its box jitters (machine
# vibration / bbox noise used to grant lifetime immunity from one jiggle —
# the phantom-immunity bug). Lifetime immunity now needs real evidence:
# locomotion, a repeated pose, or one strong detection. Small jitter only
# defers the zero-movement clock (protects a fidgeting night worker whose
# pose MediaPipe can't find) and never grants immunity.
OBJECT_AFTER = 60.0        # s with zero movement -> suspect
OBJECT_AFTER_JITTERY = 600.0  # s alive with no evidence at all -> suspect
OBJECT_MAX_CONF = 0.45     # track's PEAK YOLO confidence stays below this
OBJECT_MOVE_PX = 0.004     # above this fraction of width = "did move" (defers clock)
WALK_SPREAD_PX = 0.012     # locomotion-level spread = instant lifetime immunity
POSE_EVIDENCE = 2          # pose found in this many samples (2s+ apart) = immunity
CONF_EVIDENCE = 0.50       # one detection this strong = immunity (phantoms never)


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
        act = cfg["activity"]
        fps = float(self.inf_cfg.get("infer_fps", 2))
        self.idle_after_s = float(act.get("idle_after_minutes", 7)) * 60.0
        self.switch_window = float(act.get("machine_switch_window_seconds", 45))
        self.sit_alert_s = float(act.get("sitting_alert_minutes", 3)) * 60.0
        self.mstate = MachineStateTracker(
            machine_zones, fps,
            float(act.get("machine_motion_threshold", 1.5)))
        self._mstate_logged: dict[str, str] = {}
        # Two-threshold tracking (ByteTrack internally demands
        # activation_threshold + 0.1 to CREATE a track — subtract to make the
        # config 'confidence' the real start bar).
        self.tracker = sv.ByteTrack(
            track_activation_threshold=max(0.05, self.conf - 0.1),
            lost_track_buffer=240,          # x frame_rate/30 => ~8s at any fps
            minimum_matching_threshold=0.8,
            frame_rate=int(round(fps)) or 1,
            minimum_consecutive_frames=2,
        )
        self.enhance_enabled = bool(self.inf_cfg.get("enhance", True))
        self.activity = ActivityTracker(act)   # movement memory (object filter)
        self.alerts = AlertManager(name, cfg["alerts"])
        self.status = "starting"
        # Auto-pause (see PAUSE_AFTER above)
        self.ai_paused = False
        self._sig_prev = None                   # last 96x96 gray sample
        self._sig_prev_ts = 0.0
        self._sig_change_ts = time.monotonic()  # picture last changed
        self.counts = {"workers": 0, "active": 0, "neutral": 0, "idle": 0,
                       "at_machine": 0}
        self.live: dict[int, dict] = {}
        self.stop_event = threading.Event()
        self._jpeg_lock = threading.Lock()
        self._latest_jpeg: bytes | None = None
        # Debug overlay (?debug=1 on the stream): rendered only while at
        # least one debug viewer is connected — zero cost otherwise.
        self.debug_viewers = 0
        self._latest_debug_jpeg: bytes | None = None
        self._last_raw_dets: list[tuple] = []   # (x1,y1,x2,y2,conf,too_small)
        self._t_lastconf: dict[int, float] = {}
        self._last_draw = None
        self._last_db_log = 0.0
        self._count_hist = {k: deque(maxlen=5) for k in self.counts}
        # Worker sessions: raw track id -> accounting
        self._sessions: dict[int, dict] = {}
        # Posture (MediaPipe): info + compliance only, never state
        self.posture_enabled = bool(act.get("use_pose", True))
        self._posture = PostureClassifier() if self.posture_enabled else None
        self._posture_cache: dict[int, dict] = {}
        self._post_hist: dict[int, deque] = {}   # tid -> (ts, 'sitting'/'standing')
        self._sit_since: dict[int, float] = {}   # tid -> wall ts sitting-at-machine began
        # State machine per track
        self._tstate: dict[int, dict] = {}
        self._t_lastseen: dict[int, float] = {}
        # Identity (OSNet re-id) — shared gallery across all cameras
        self.reid = reid.shared() if cfg["inference"].get("use_reid", True) else None
        if self.reid and not self.reid.ok:
            self.reid = None
        self.board = personstate.board(self.switch_window) if self.reid else None
        self._t_person: dict[int, int] = {}
        self._t_pending: dict[int, list] = {}
        self._t_embed_ts: dict[int, float] = {}
        self._t_machine_ts: dict[int, float] = {}
        self._t_machine_name: dict[int, str] = {}
        # Legacy display numbers (used only when re-id is unavailable)
        self._display: dict[int, int] = {}
        self._raw_seen: dict[int, float] = {}
        # Object filter
        self._t_first: dict[int, float] = {}
        self._t_conf: dict[int, list] = {}
        self._t_human: set[int] = set()
        self._suspect: set[int] = set()
        self._t_last_move: dict[int, float] = {}  # last small-jitter movement
        self._t_pose_ev: dict[int, tuple] = {}    # tid -> (pose hits, last ts)

    # ---------------- main loop ----------------

    def run(self):
        reader = LatestFrameReader(self.source_str)
        reader.start()
        interval = 1.0 / float(self.inf_cfg.get("infer_fps", 2))
        next_infer = 0.0

        while not self.stop_event.is_set():
            if not reader.connected:
                self.status = "reconnecting"
                self.counts = {k: 0 for k in self.counts}
                self.live = {}
                self._last_draw = None
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
                    self._signal_check(frame, t0)
                if not self.process_enabled:
                    self._encode_view_only(frame)
                elif self.ai_paused:
                    self._encode_paused(frame)
                elif t0 >= next_infer:
                    next_infer = t0 + interval
                    self._process(frame)
                else:
                    self._encode_cached(frame)
            except Exception:
                # Never let one bad frame kill the camera thread.
                time.sleep(0.2)
            time.sleep(max(0.0, 1.0 / DISPLAY_FPS - (time.monotonic() - t0)))

        reader.stop()
        if self.reid:
            self.reid.flush()
        if self.board:
            self.board.flush_all()
        self.status = "stopped"

    def stop(self):
        self.stop_event.set()

    # ---------------- processing ----------------

    def _process(self, frame):
        if self.enhance_enabled:
            frame = enhance.maybe_enhance(frame)
        h, w = frame.shape[:2]
        now = time.time()
        result = detector.predict(
            frame,
            conf=DETECT_FLOOR,
            imgsz=int(self.inf_cfg.get("imgsz", 512)),
        )
        det = sv.Detections.from_ultralytics(result)
        persons = det[det.class_id == detector.PERSON]
        phones = det[det.class_id == detector.CELL_PHONE]
        phones = phones[phones.confidence >= PHONE_CONF]

        # Every raw person detection, kept for the debug overlay/API before
        # any filter can hide it.
        self._last_raw_dets = [
            (float(x1), float(y1), float(x2), float(y2), float(c),
             bool((y2 - y1) < 0.05 * h))
            for (x1, y1, x2, y2), c in zip(persons.xyxy, persons.confidence)
        ] if len(persons) else []

        # Drop implausibly small "people" (far-away noise, reflections).
        if len(persons) > 0:
            heights = persons.xyxy[:, 3] - persons.xyxy[:, 1]
            persons = persons[heights >= 0.05 * h]

        tracked = self.tracker.update_with_detections(persons)
        raw_ids = [int(r) for r in (
            tracked.tracker_id if tracked.tracker_id is not None else [])]
        if tracked.confidence is not None:
            self._t_lastconf = {int(t): float(c) for t, c in
                                zip(raw_ids, tracked.confidence)}

        # Machine RUNNING/STOPPED — measured with people cut out of the ROI.
        mstates = self.mstate.update(frame, tracked.xyxy) if self.mstate else {}
        for mname, mst in mstates.items():
            if self._mstate_logged.get(mname) != mst["state"]:
                self._mstate_logged[mname] = mst["state"]
                db.log_machine_state(self.cam_name, mname,
                                     mst["state"], mst["energy"])

        anchors, centroids = [], []
        for x1, y1, x2, y2 in tracked.xyxy:
            anchors.append(((x1 + x2) / 2, y2))
            centroids.append(((x1 + x2) / 2, (y1 + y2) / 2))

        self.activity.update(raw_ids, centroids, w)  # movement memory only
        postures = self._update_postures(frame, tracked, raw_ids, w, h)
        self._update_object_filter(tracked, raw_ids, postures, w)
        self._update_identity(frame, tracked, raw_ids, w, h)

        # Who is at which machine (short grace bridges detection blips).
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

        states, idle_alerts = self._update_states(raw_ids, at_machine, now)
        sitting = self._posture_compliance(raw_ids, at_machine, postures,
                                           tracked, frame, w, h, now)

        # Suspected non-humans drop out of every count and report.
        humans = [t for t in raw_ids if t not in self._suspect]
        h_states = {t: states[t] for t in humans}

        raw = {"workers": 0, "active": 0, "neutral": 0, "idle": 0, "at_machine": 0}
        for tid, anchor in zip(raw_ids, anchors):
            if tid in self._suspect:
                continue
            if self.zone.contains(anchor, w, h):
                raw["workers"] += 1
            raw[states[tid]] += 1
            if at_machine.get(tid):
                raw["at_machine"] += 1

        for k, v in raw.items():
            self._count_hist[k].append(v)
        self.counts = {k: int(statistics.median(d))
                       for k, d in self._count_hist.items()}

        self._update_sessions(h_states, postures, at_machine, sitting)
        self.live = {
            tid: {
                "display": self._label(tid),
                "pid": self._t_person.get(tid),
                "state": st,
                "posture": postures.get(tid),
                "sitting": sitting.get(tid, False),
                "machine": at_machine.get(tid),
                "machine_running": self.mstate.running(at_machine.get(tid))
                                   if self.mstate else False,
            }
            for tid, st in h_states.items()
        }

        self._last_draw = (tracked.xyxy.copy(), list(raw_ids), dict(states),
                           phones.xyxy.copy(), dict(at_machine), dict(sitting),
                           dict(mstates), dict(self._tstate_public(now)))
        annotated = self._annotate(frame, *self._last_draw, w=w, h=h)
        if self.debug_viewers > 0:
            self._encode_debug(annotated)
        self.alerts.check(self.counts["workers"], self.zone.max_workers,
                          len(phones), annotated)
        for tid in idle_alerts:
            if tid in humans:
                self.alerts.fire_idle_worker(
                    self._label(tid), max(1, round(self.idle_after_s / 60)),
                    annotated)
        self._encode(annotated)

        if now - self._last_db_log >= 1.0:
            self._last_db_log = now
            db.log_observation(self.cam_name, self.counts["workers"],
                               self.counts["active"], self.counts["idle"])
            if self.reid:
                for tid in humans:
                    pid = self._t_person.get(tid)
                    if pid is None:
                        continue
                    at = at_machine.get(tid)
                    self.reid.tick(pid, 1.0, bool(at))
                    st = self._tstate.get(tid, {})
                    self.board.update(
                        pid, self._label(tid), self.cam_name,
                        st.get("state", "neutral"), st.get("state_since", now),
                        st.get("neutral_since"), st.get("away_since"),
                        at, self.mstate.running(at) if self.mstate else False,
                        sitting.get(tid, False), now)
                self.board.sweep(now)

    def _tstate_public(self, now) -> dict:
        """Per-track timing info the annotator needs (minutes in state)."""
        out = {}
        for tid, st in self._tstate.items():
            out[tid] = {"mins": (now - st.get("state_since", now)) / 60.0}
        return out

    # ---------------- state machine (rules 1-4, 6) ----------------

    def _update_states(self, raw_ids, at_machine, now) -> tuple[dict, list]:
        states, idle_alerts = {}, []
        for tid in raw_ids:
            self._t_lastseen[tid] = now
            st = self._tstate.get(tid)
            if st is None:
                st = self._tstate[tid] = {
                    "state": "neutral", "state_since": now,
                    "neutral_since": now, "away_since": None,
                    "at_count": 0, "inherited": False,
                }
            # One-time inheritance when identity resolves: continues the
            # person's story across cameras (machine-switch rule).
            if not st["inherited"] and self.board:
                pid = self._t_person.get(tid)
                if pid is not None:
                    st["inherited"] = True
                    inh = self.board.inherit(pid, now)
                    if inh:
                        st.update(state=inh["state"],
                                  state_since=inh["state_since"],
                                  neutral_since=inh["neutral_since"],
                                  away_since=inh["away_since"])

            if at_machine.get(tid):
                st["at_count"] += 1
                st["away_since"] = None
                if st["state"] != "active" and st["at_count"] >= ENTER_ACTIVE_PASSES:
                    st.update(state="active", state_since=now,
                              neutral_since=None)
            else:
                st["at_count"] = 0
                if st["state"] == "active":
                    if st["away_since"] is None:
                        st["away_since"] = now
                    elif now - st["away_since"] >= self.switch_window:
                        st.update(state="neutral", state_since=now,
                                  neutral_since=now, away_since=None)
                elif st["state"] == "neutral":
                    if st["neutral_since"] is None:
                        st["neutral_since"] = now
                    elif now - st["neutral_since"] >= self.idle_after_s:
                        st.update(state="idle", state_since=now)
                        idle_alerts.append(tid)
            states[tid] = st["state"]

        for tid in [t for t, ts in self._t_lastseen.items() if now - ts > 60.0]:
            self._t_lastseen.pop(tid, None)
            self._tstate.pop(tid, None)
            self._post_hist.pop(tid, None)
            self._sit_since.pop(tid, None)
        return states, idle_alerts

    # ---------------- posture compliance (discipline layer) ----------------

    def _posture_compliance(self, raw_ids, at_machine, postures, tracked,
                            frame, w, h, now) -> dict:
        """Smoothed sitting detection + POSTURE alert. Never touches state."""
        sitting = {}
        for (x1, y1, x2, y2), tid in zip(tracked.xyxy, raw_ids):
            p = postures.get(tid)
            hist = self._post_hist.setdefault(tid, deque())
            if p in ("sitting", "standing"):
                hist.append((now, p))
            while hist and now - hist[0][0] > POSTURE_WINDOW:
                hist.popleft()
            known = len(hist)
            ratio = (sum(1 for _, x in hist if x == "sitting") / known
                     if known else 0.0)
            is_sitting = known >= POSTURE_MIN_SAMPLES and ratio >= POSTURE_SIT_RATIO
            sitting[tid] = is_sitting

            if is_sitting and at_machine.get(tid) and tid not in self._suspect:
                t0 = self._sit_since.setdefault(tid, now)
                if now - t0 >= self.sit_alert_s:
                    x1i, y1i = max(0, int(x1)), max(0, int(y1))
                    x2i, y2i = min(w, int(x2)), min(h, int(y2))
                    crop = frame[y1i:y2i, x1i:x2i]
                    if crop.size:
                        self.alerts.fire_posture(
                            self._label(tid), int((now - t0) / 60) or 1, crop)
            else:
                self._sit_since.pop(tid, None)
        return sitting

    # ---------------- identity ----------------

    def _label(self, tid: int) -> str:
        if self.reid:
            pid = self._t_person.get(tid)
            # No identity yet (crop too small for re-id, or still pending)
            # => temporary ID. The person is tracked/classified regardless.
            return self.reid.display(pid) if pid is not None else f"T{tid}"
        return f"W{self._display.get(tid, tid)}"

    def _update_identity(self, frame, tracked, raw_ids, w, h):
        now = time.monotonic()
        if not self.reid:
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

        gone = [t for t in self._t_embed_ts
                if now - self._t_embed_ts.get(t, 0) > 30.0 and t not in raw_ids]
        for t in gone:
            for d in (self._t_person, self._t_pending, self._t_embed_ts,
                      self._t_machine_ts, self._t_machine_name):
                d.pop(t, None)

    # ---------------- sessions ----------------

    def _update_sessions(self, states: dict, postures: dict, at_machine: dict,
                         sitting: dict):
        now = time.time()
        for tid, state in states.items():
            s = self._sessions.setdefault(
                tid, {"start": now, "last": now, "active_n": 0, "idle_n": 0,
                      "total_n": 0, "sit_n": 0, "stand_n": 0, "machine_n": 0,
                      "sit_mach_n": 0}
            )
            s["last"] = now
            s["total_n"] += 1
            if state == "active":
                s["active_n"] += 1
            elif state == "idle":
                s["idle_n"] += 1
            if at_machine.get(tid):
                s["machine_n"] += 1
                if sitting.get(tid):
                    s["sit_mach_n"] += 1
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
                n = s["total_n"]
                db.log_session(
                    self.cam_name, tid, s["start"], s["last"], duration,
                    round(100.0 * s["active_n"] / n, 1), posture,
                    person_id=self._t_person.get(tid),
                    machine_pct=round(100.0 * s["machine_n"] / n, 1),
                    idle_pct=round(100.0 * s["idle_n"] / n, 1),
                    sit_machine_pct=round(100.0 * s["sit_mach_n"] / n, 1),
                )

    # ---------------- posture & human check ----------------

    def _update_postures(self, frame, tracked, raw_ids, w, h) -> dict:
        """Posture per visible worker, refreshed at most 1x/sec per track.
        Report info + compliance layer only — never decides active/idle."""
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

        for tid in [t for t, c in self._posture_cache.items() if now - c["t"] > 10.0]:
            self._posture_cache.pop(tid, None)
        return postures

    def _update_object_filter(self, tracked, raw_ids, postures, w):
        now = time.monotonic()
        confs = (tracked.confidence if tracked.confidence is not None
                 else [1.0] * len(raw_ids))
        for tid, conf in zip(raw_ids, confs):
            self._t_first.setdefault(tid, now)
            peak = self._t_conf.setdefault(tid, [0.0])
            peak[0] = max(peak[0], float(conf))
            if tid in self._t_human:
                continue
            spread = self.activity.spread(tid)
            # Lifetime immunity: locomotion or one strong detection.
            if ((spread is not None and spread > WALK_SPREAD_PX * w)
                    or float(conf) >= CONF_EVIDENCE):
                self._grant_human(tid)
                continue
            # Pose immunity needs POSE_EVIDENCE hits 2s+ apart — a single
            # MediaPipe hallucination on machinery no longer counts.
            if postures.get(tid) is not None:
                cnt, last = self._t_pose_ev.get(tid, (0, 0.0))
                if now - last >= 2.0:
                    cnt += 1
                    self._t_pose_ev[tid] = (cnt, now)
                if cnt >= POSE_EVIDENCE:
                    self._grant_human(tid)
                    continue
            # Small jitter defers the zero-movement clock, nothing more.
            if spread is not None and spread > OBJECT_MOVE_PX * w:
                self._t_last_move[tid] = now
            still_since = self._t_last_move.get(tid, self._t_first[tid])
            is_object = peak[0] < OBJECT_MAX_CONF and (
                now - still_since >= OBJECT_AFTER
                or now - self._t_first[tid] >= OBJECT_AFTER_JITTERY)
            if is_object:
                self._suspect.add(tid)
            else:
                self._suspect.discard(tid)
        live = set(raw_ids)
        for tid in [t for t in self._t_first
                    if t not in live and now - self._t_first[t] > 60.0]:
            self._t_first.pop(tid, None)
            self._t_conf.pop(tid, None)
            self._t_human.discard(tid)
            self._suspect.discard(tid)
            self._t_last_move.pop(tid, None)
            self._t_pose_ev.pop(tid, None)

    def _grant_human(self, tid: int):
        self._t_human.add(tid)
        self._suspect.discard(tid)
        self._t_pose_ev.pop(tid, None)
        self._t_last_move.pop(tid, None)

    # ---------------- drawing ----------------

    def _encode_cached(self, frame):
        if self.enhance_enabled:
            frame = enhance.maybe_enhance(frame)
        h, w = frame.shape[:2]
        if self._last_draw is None:
            self._encode(self._annotate(
                frame, np.empty((0, 4)), [], {}, np.empty((0, 4)),
                {}, {}, {}, {}, w=w, h=h))
            return
        self._encode(self._annotate(frame, *self._last_draw, w=w, h=h))

    def _annotate(self, frame, boxes, raw_ids, states, phone_boxes,
                  at_machine, sitting, mstates, tinfo, w, h):
        out = frame.copy()

        # Work zone (cyan) + machine zones (running=lime, stopped=magenta)
        poly = self.zone.pixels(w, h)
        overlay = out.copy()
        cv2.fillPoly(overlay, [poly], CYAN)
        for name, z in (self.machines.zones if self.machines else []):
            running = (mstates or {}).get(name, {}).get("state") == "running"
            cv2.fillPoly(overlay, [z.pixels(w, h)], LIME if running else MAGENTA)
        cv2.addWeighted(overlay, 0.12, out, 0.88, 0, out)
        cv2.polylines(out, [poly], True, CYAN, 2)
        for name, z in (self.machines.zones if self.machines else []):
            mst = (mstates or {}).get(name, {})
            running = mst.get("state") == "running"
            color = LIME if running else MAGENTA
            mp = z.pixels(w, h)
            cv2.polylines(out, [mp], True, color, 2)
            label = f"{name} - {'RUNNING' if running else 'stopped'}"
            cv2.putText(out, label, (int(mp[0][0]) + 4, int(mp[0][1]) + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # Workers
        for (x1, y1, x2, y2), tid in zip(boxes, raw_ids):
            tid = int(tid)
            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            if tid in self._suspect:
                cv2.rectangle(out, p1, p2, GREY, 1)
                cv2.putText(out, "not a person", (p1[0], p1[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREY, 1, cv2.LINE_AA)
                continue
            state = states.get(tid, "neutral")
            color = GREEN if state == "active" else (
                ORANGE if state == "idle" else GREY)
            cv2.rectangle(out, p1, p2, color, 2)
            label = f"{self._label(tid)} {state.upper()}"
            mins = (tinfo or {}).get(tid, {}).get("mins", 0)
            if state == "idle" and mins >= 1:
                label += f" {int(mins)}m"
            m = (at_machine or {}).get(tid)
            if state == "active" and m:
                running = (mstates or {}).get(m, {}).get("state") == "running"
                label += f" @ {m} ({'RUNNING' if running else 'stopped'})"
            if (sitting or {}).get(tid):
                label += " - SITTING"
            cv2.rectangle(out, (p1[0], p1[1] - 22),
                          (p1[0] + 8 * len(label) + 8, p1[1]), color, -1)
            cv2.putText(out, label, (p1[0] + 4, p1[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Phones
        for x1, y1, x2, y2 in phone_boxes:
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), RED, 2)
            cv2.putText(out, "PHONE", (int(x1), int(y1) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED, 2, cv2.LINE_AA)

        # Header bar
        c = self.counts
        bar = (f"{self.cam_name}  |  workers: {c['workers']}  "
               f"active: {c['active']}  neutral: {c['neutral']}  "
               f"idle: {c['idle']}")
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
        out = frame.copy()
        h, w = out.shape[:2]
        cv2.rectangle(out, (0, 0), (w, 30), (25, 25, 25), -1)
        cv2.putText(out, f"{self.cam_name}  |  LIVE VIEW", (10, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
        self._encode(out)

    def _encode_paused(self, frame):
        out = frame.copy()
        h, w = out.shape[:2]
        cv2.rectangle(out, (0, 0), (w, 30), (25, 25, 25), -1)
        cv2.putText(out, f"{self.cam_name}  |  AI PAUSED - no signal", (10, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, ORANGE, 1, cv2.LINE_AA)
        self._encode(out)

    # ---------------- signal watchdog (auto-pause) ----------------

    def _signal_check(self, frame, now: float):
        """Freeze detector: pause AI on a picture that hasn't changed for
        PAUSE_AFTER (dead channel placeholder), resume as soon as it moves."""
        if now - self._sig_prev_ts >= SIG_SAMPLE:
            small = cv2.cvtColor(cv2.resize(frame, (96, 96)), cv2.COLOR_BGR2GRAY)
            if self._sig_prev is None:
                self._sig_change_ts = now
            else:
                diff = cv2.absdiff(small, self._sig_prev)
                changed = float(np.mean(diff > SIG_PIX_DELTA))
                if changed > SIG_CHANGED_FRAC:
                    self._sig_change_ts = now
            self._sig_prev = small
            self._sig_prev_ts = now
        frozen = (now - self._sig_change_ts) >= PAUSE_AFTER
        if frozen and not self.ai_paused:
            self.ai_paused = True
            self.counts = {k: 0 for k in self.counts}
            self.live = {}
            self._last_draw = None
            print(f"[signal] {self.cam_name}: picture frozen "
                  f"{int(PAUSE_AFTER / 60)} min — AI paused (auto-resumes "
                  "when the signal returns)")
        elif self.ai_paused and not frozen:
            self.ai_paused = False
            print(f"[signal] {self.cam_name}: signal back — AI resumed")

    def _set_offline_frame(self):
        img = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(img, f"{self.cam_name}: reconnecting...", (60, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2, cv2.LINE_AA)
        self._encode(img)

    def _encode_debug(self, annotated):
        """Annotated frame + EVERY raw YOLO detection with its confidence:
        yellow = starts a track, orange = continue-only (below start bar),
        grey = dropped too-small. Legend shows this camera's thresholds."""
        out = annotated.copy()
        h, w = out.shape[:2]
        for x1, y1, x2, y2, conf, small in self._last_raw_dets:
            if small:
                color, tag = (140, 140, 140), "small"
            elif conf >= self.conf:
                color, tag = (60, 220, 255), "start"
            else:
                color, tag = (0, 165, 255), "weak"
            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(out, p1, p2, color, 1)
            cv2.putText(out, f"{conf:.2f} {tag}", (p1[0], max(12, p1[1] - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        legend = (f"DEBUG raw dets={len(self._last_raw_dets)} | "
                  f"floor {DETECT_FLOOR} | start bar {self.conf} | "
                  "yellow=start orange=weak grey=small")
        cv2.rectangle(out, (0, h - 26), (w, h), (25, 25, 25), -1)
        cv2.putText(out, legend, (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (240, 240, 240), 1, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self._jpeg_lock:
                self._latest_debug_jpeg = buf.tobytes()

    def latest_jpeg(self) -> bytes | None:
        with self._jpeg_lock:
            return self._latest_jpeg

    def latest_debug_jpeg(self) -> bytes | None:
        with self._jpeg_lock:
            return self._latest_debug_jpeg or self._latest_jpeg
