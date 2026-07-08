"""Cross-camera person state board.

Tracks live on one camera; PEOPLE move between cameras. This board keys
everything by re-id person, so an ACTIVE worker who walks from machine A
(camera 1) to machine B (camera 4) inside the switch window stays one
continuous ACTIVE story: the new track inherits the person's state, the
machine visit is closed/opened with `switched_from`, and no idle gap ever
appears in the record.

Also the single writer of the `machine_visits` table.
"""
import threading
import time

from . import db

VISIT_GRACE_S = 8.0    # machine flicker shorter than this doesn't end a visit
VISIT_MIN_S = 20.0     # visits shorter than this are noise and not logged
FORGET_S = 3600.0      # drop people unseen for an hour


class PersonBoard:
    def __init__(self, switch_window_s: float = 45.0):
        self.window = float(switch_window_s)
        self._lock = threading.Lock()
        self._p: dict[int, dict] = {}

    # ---------------- continuity ----------------

    def inherit(self, pid: int, now: float) -> dict | None:
        """Snapshot for a brand-new track of a known person. Within the
        switch window the new track continues the person's state (ACTIVE
        stays ACTIVE across cameras); beyond it, None => fresh NEUTRAL."""
        with self._lock:
            p = self._p.get(pid)
            if not p or now - p["last_seen"] > self.window:
                return None
            return {
                "state": p["state"],
                "state_since": p["state_since"],
                "neutral_since": p["neutral_since"],
                "away_since": p["away_since"],
            }

    # ---------------- accounting (call ~1x/sec per visible person) --------

    def update(self, pid: int, display: str, camera: str, state: str,
               state_since: float, neutral_since: float | None,
               away_since: float | None, machine: str | None,
               running: bool, sitting: bool, now: float):
        with self._lock:
            p = self._p.setdefault(pid, {
                "state": "neutral", "state_since": now, "neutral_since": now,
                "away_since": None, "last_seen": 0.0, "display": display,
                "visit": None, "last_machine": None, "last_machine_ts": 0.0,
            })
            p.update(state=state, state_since=state_since,
                     neutral_since=neutral_since, away_since=away_since,
                     last_seen=now, display=display)

            v = p["visit"]
            if machine:
                if v and (v["camera"], v["machine"]) != (camera, machine):
                    self._close_visit(p, pid)   # switching A -> B
                    v = None
                if v is None:
                    switched = None
                    if (p["last_machine"]
                            and now - p["last_machine_ts"] <= self.window
                            and p["last_machine"][1] != machine):
                        switched = p["last_machine"][1]
                    v = p["visit"] = {
                        "camera": camera, "machine": machine, "start": now,
                        "last": now, "n": 0, "run_n": 0, "sit_n": 0,
                        "switched_from": switched,
                    }
                v["last"] = now
                v["n"] += 1
                if running:
                    v["run_n"] += 1
                if sitting:
                    v["sit_n"] += 1
            elif v and now - v["last"] > VISIT_GRACE_S:
                self._close_visit(p, pid)

    def _close_visit(self, p: dict, pid: int):
        v = p.pop("visit", None) or None
        p["visit"] = None
        if not v:
            return
        end = v["last"]
        p["last_machine"] = (v["camera"], v["machine"])
        p["last_machine_ts"] = end
        if end - v["start"] >= VISIT_MIN_S and v["n"] > 0:
            db.log_machine_visit(
                v["camera"], v["machine"], pid, p.get("display"),
                v["start"], end,
                round(100.0 * v["run_n"] / v["n"], 1),
                round(100.0 * v["sit_n"] / v["n"], 1),
                v["switched_from"],
            )

    # ---------------- housekeeping ----------------

    def sweep(self, now: float):
        """Close visits of people who left every camera; forget old people."""
        with self._lock:
            for pid, p in list(self._p.items()):
                if p["visit"] and now - p["visit"]["last"] > VISIT_GRACE_S:
                    self._close_visit(p, pid)
                if now - p["last_seen"] > FORGET_S:
                    self._p.pop(pid, None)

    def flush_all(self):
        """Server shutdown: don't lose in-progress visits."""
        with self._lock:
            for pid, p in self._p.items():
                if p["visit"]:
                    self._close_visit(p, pid)


_board: PersonBoard | None = None
_board_lock = threading.Lock()


def board(switch_window_s: float = 45.0) -> PersonBoard:
    global _board
    with _board_lock:
        if _board is None:
            _board = PersonBoard(switch_window_s)
        return _board
