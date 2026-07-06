"""Movement-based ACTIVE/IDLE classification.

A worker is IDLE when their position barely changes for `idle_after_seconds`.
(Phase 1 production adds pose estimation; movement is enough for the demo.)
"""
import time
from collections import deque


class ActivityTracker:
    def __init__(self, cfg: dict):
        self.idle_after = float(cfg.get("idle_after_seconds", 10))
        self.window = float(cfg.get("movement_window", 3.0))
        self.threshold = float(cfg.get("movement_threshold", 0.015))
        self.history = {}     # track_id -> deque[(t, x, y)]
        self.static_since = {}  # track_id -> t when movement stopped
        self.last_seen = {}

    def update(self, track_ids, centroids, frame_w: float) -> dict:
        """Returns {track_id: 'active' | 'idle'}."""
        now = time.monotonic()
        states = {}
        for tid, (cx, cy) in zip(track_ids, centroids):
            tid = int(tid)
            self.last_seen[tid] = now
            hist = self.history.setdefault(tid, deque())
            hist.append((now, cx, cy))
            while hist and now - hist[0][0] > self.window:
                hist.popleft()

            xs = [p[1] for p in hist]
            ys = [p[2] for p in hist]
            spread = max(max(xs) - min(xs), max(ys) - min(ys))
            moving = spread > self.threshold * frame_w

            if moving:
                self.static_since.pop(tid, None)
                states[tid] = "active"
            else:
                start = self.static_since.setdefault(tid, now)
                states[tid] = "idle" if now - start >= self.idle_after else "active"

        self._purge(now)
        return states

    def spread(self, tid: int) -> float | None:
        """Pixel spread of recent movement — None if we lack history."""
        hist = self.history.get(int(tid))
        if not hist or len(hist) < 3:
            return None
        xs = [p[1] for p in hist]
        ys = [p[2] for p in hist]
        return max(max(xs) - min(xs), max(ys) - min(ys))

    def _purge(self, now: float):
        gone = [tid for tid, t in self.last_seen.items() if now - t > 10.0]
        for tid in gone:
            self.last_seen.pop(tid, None)
            self.history.pop(tid, None)
            self.static_since.pop(tid, None)
