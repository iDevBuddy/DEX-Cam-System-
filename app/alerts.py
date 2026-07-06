"""Alert rules per camera: overcrowding, unmanned zone, mobile phone.
Cooldowns stop alert spam; snapshots saved for the dashboard log."""
import time
from pathlib import Path

import cv2

from . import db

SNAP_DIR = Path(__file__).resolve().parent.parent / "snapshots"


class AlertManager:
    def __init__(self, camera_name: str, cfg: dict):
        self.camera = camera_name
        self.cooldown = float(cfg.get("cooldown_seconds", 60))
        self.phone_cooldown = float(cfg.get("phone_cooldown_seconds", 30))
        self.unmanned_after = float(cfg.get("unmanned_after_seconds", 120))
        self.idle_worker_after = float(cfg.get("idle_worker_seconds", 60))
        self.idle_worker_cooldown = float(cfg.get("idle_worker_cooldown_seconds", 300))
        self.last_fired: dict[str, float] = {}
        self.last_occupied = time.monotonic()

    def _fire(self, alert_type: str, message: str, frame, cooldown: float,
              key: str | None = None):
        now = time.monotonic()
        key = key or alert_type
        if now - self.last_fired.get(key, -1e9) < cooldown:
            return
        self.last_fired[key] = now
        snapshot = self._save_snapshot(alert_type, frame)
        db.log_alert(self.camera, alert_type, message, snapshot)

    def _save_snapshot(self, alert_type: str, frame) -> str | None:
        try:
            SNAP_DIR.mkdir(parents=True, exist_ok=True)
            name = f"{int(time.time() * 1000)}_{self.camera}_{alert_type}.jpg"
            cv2.imwrite(str(SNAP_DIR / name), frame)
            return name
        except Exception:
            return None

    def check(self, in_zone: int, zone_max: int, phones: int, frame):
        now = time.monotonic()

        if in_zone > 0:
            self.last_occupied = now
        elif now - self.last_occupied >= self.unmanned_after:
            self._fire(
                "unmanned",
                f"Zone empty for {int(self.unmanned_after)}s on '{self.camera}'",
                frame, self.cooldown,
            )
            self.last_occupied = now  # reset so it re-alerts after another full window

        if in_zone > zone_max:
            self._fire(
                "overcrowding",
                f"{in_zone} workers in zone on '{self.camera}' (max {zone_max})",
                frame, self.cooldown,
            )

        if phones > 0:
            self._fire(
                "phone",
                f"Mobile phone detected on '{self.camera}'",
                frame, self.phone_cooldown,
            )

    def fire_idle_worker(self, who: str, idle_seconds: int, frame):
        """Photo evidence of a worker who has been idle past the threshold.
        Per-worker cooldown so one sleepy worker doesn't flood the log."""
        self._fire(
            "idle_worker",
            f"Worker {who} idle for {idle_seconds}s on '{self.camera}'",
            frame, self.idle_worker_cooldown, key=f"idle_{who}",
        )
