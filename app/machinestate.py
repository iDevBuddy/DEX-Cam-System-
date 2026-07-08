"""Machine RUNNING / STOPPED from the camera alone — no new hardware.

Signal 1: motion energy inside each machine-zone polygon (frame differencing
on a small grayscale image). Our machines visibly move when running (lathe
rotation, printing-machine motion).
Signal 2: sustained brightness shift in the zone (indicator lamps, glow).

Pixels inside person bounding boxes (inflated 10%) are cut out of the zone
before measuring, so a moving worker can never fake a running machine. If a
person hides most of the zone, that sample is skipped and the state holds.

The instantaneous signal is smoothed over a 10-second window; RUNNING needs
a majority of samples — the reported state cannot flicker.
"""
import time
from collections import deque

import cv2
import numpy as np

SCALE_W = 480          # analysis width in px — cheap (~2 ms) and sufficient
WINDOW_S = 10.0        # smoothing window (seconds)
PIXEL_DIFF = 12        # a pixel changing more than this gray = "moving pixel"
BRIGHT_SHIFT = 8.0     # sustained gray-level shift that counts as activity
BRIGHT_KEEP_S = 14.0   # brightness history horizon
MIN_VISIBLE = 0.30     # need this fraction of the zone person-free to sample
PERSON_PAD = 0.10      # inflate person boxes by 10% before cutting them out
MIN_SAMPLES = 4        # don't call RUNNING/STOPPED before this many samples

# Energy metric: PERCENT of zone pixels that moved (not mean difference) —
# a lathe chuck is a small part of a big zone; averaging dilutes it away,
# a moving-pixel fraction does not. Threshold is in percent of zone area.


class MachineStateTracker:
    """One per camera. Call update(frame, person_boxes) once per AI pass."""

    def __init__(self, machine_entries: list[dict] | None, fps: float,
                 default_threshold: float = 1.5):
        self.entries = []
        for i, e in enumerate(machine_entries or []):
            poly = e.get("zone") or e.get("poly")
            if poly:
                self.entries.append({
                    "name": e.get("name") or f"machine-{i + 1}",
                    "poly": poly,
                    "threshold": float(e.get("motion_threshold",
                                             default_threshold)),
                })
        n = max(MIN_SAMPLES, int(WINDOW_S * max(fps, 0.5)))
        self._hist = {e["name"]: deque(maxlen=n) for e in self.entries}
        self._bright = {e["name"]: deque() for e in self.entries}
        self.states = {e["name"]: {"state": "stopped", "energy": 0.0}
                       for e in self.entries}
        self._prev = None      # previous small grayscale frame
        self._prev_boxes = []  # person boxes of the previous frame — diffing
                               # against it leaves a "trail ghost" at the old
                               # position, which must be masked out too
        self._masks = None     # zone masks at analysis size
        self._size = None      # (w, h) the masks were built for

    def __bool__(self):
        return bool(self.entries)

    # ---------------- internals ----------------

    def _build_masks(self, sw: int, sh: int):
        self._masks = {}
        for e in self.entries:
            m = np.zeros((sh, sw), dtype=np.uint8)
            pts = np.array([[int(x * sw), int(y * sh)] for x, y in e["poly"]],
                           dtype=np.int32)
            cv2.fillPoly(m, [pts], 1)
            self._masks[e["name"]] = m

    def update(self, frame_bgr, person_xyxy) -> dict:
        """person_xyxy: full-frame person boxes (anything array-like of
        [x1,y1,x2,y2]). Returns {machine: {"state","energy"}}."""
        if not self.entries:
            return self.states
        try:
            h, w = frame_bgr.shape[:2]
            scale = SCALE_W / w
            sw, sh = SCALE_W, max(1, int(h * scale))
            if self._size != (sw, sh):
                self._size = (sw, sh)
                self._build_masks(sw, sh)
                self._prev = None
            small = cv2.cvtColor(
                cv2.resize(frame_bgr, (sw, sh), interpolation=cv2.INTER_AREA),
                cv2.COLOR_BGR2GRAY).astype(np.int16)

            # Person-free mask: current AND previous positions (people can't
            # pretend to be machines, not even their movement trail).
            cur_boxes = [list(map(float, b[:4]))
                         for b in (person_xyxy if person_xyxy is not None else [])]
            free = np.ones((sh, sw), dtype=bool)
            for x1, y1, x2, y2 in cur_boxes + self._prev_boxes:
                px, py = (x2 - x1) * PERSON_PAD, (y2 - y1) * PERSON_PAD
                a = max(0, int((x1 - px) * scale))
                b = max(0, int((y1 - py) * scale))
                c = min(sw, int((x2 + px) * scale) + 1)
                d = min(sh, int((y2 + py) * scale) + 1)
                free[b:d, a:c] = False
            self._prev_boxes = cur_boxes

            now = time.time()
            prev = self._prev
            self._prev = small
            for e in self.entries:
                name = e["name"]
                zone = self._masks[name].astype(bool)
                valid = zone & free
                if valid.sum() < MIN_VISIBLE * max(zone.sum(), 1):
                    continue  # zone mostly hidden — hold current state
                brightness = float(small[valid].mean())
                bh = self._bright[name]
                bh.append((now, brightness))
                while bh and now - bh[0][0] > BRIGHT_KEEP_S:
                    bh.popleft()
                old = [v for t, v in bh if now - t >= WINDOW_S * 0.8]
                bright_shift = abs(brightness - (sum(old) / len(old))) if old else 0.0

                if prev is None:
                    continue
                diff = np.abs(small[valid] - prev[valid])
                energy = float(100.0 * (diff > PIXEL_DIFF).mean())  # % moving px
                moving = energy > e["threshold"] or bright_shift > BRIGHT_SHIFT
                hist = self._hist[name]
                hist.append(moving)
                state = self.states[name]["state"]
                if len(hist) >= MIN_SAMPLES:
                    state = "running" if sum(hist) > 0.5 * len(hist) else "stopped"
                self.states[name] = {"state": state, "energy": round(energy, 2)}
        except Exception:
            pass  # measurement must never hurt the camera thread
        return self.states

    def running(self, machine: str | None) -> bool:
        if not machine:
            return False
        return self.states.get(machine, {}).get("state") == "running"
