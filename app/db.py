"""SQLite storage. Single writer thread (sqlite likes one writer); readers open
their own short-lived connections."""
import queue
import sqlite3
import threading
import time
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "events.db"

_write_q: "queue.Queue[tuple]" = queue.Queue()
_writer_started = False

SCHEMA = """
CREATE TABLE IF NOT EXISTS observations (
    ts      REAL NOT NULL,          -- unix time
    camera  TEXT NOT NULL,
    workers INTEGER NOT NULL,       -- people in zone this second
    active  INTEGER NOT NULL,
    idle    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_obs_ts ON observations (ts);

CREATE TABLE IF NOT EXISTS alerts (
    ts        REAL NOT NULL,
    camera    TEXT NOT NULL,
    type      TEXT NOT NULL,        -- overcrowding | unmanned | phone
    message   TEXT NOT NULL,
    snapshot  TEXT
);
CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts (ts);

CREATE TABLE IF NOT EXISTS sessions (
    camera     TEXT NOT NULL,
    track_id   INTEGER NOT NULL,
    start_ts   REAL NOT NULL,
    end_ts     REAL NOT NULL,
    duration   REAL NOT NULL,       -- seconds
    active_pct REAL NOT NULL,
    posture    TEXT                 -- dominant: 'standing' | 'sitting' | NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_end ON sessions (end_ts);
"""


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.execute("PRAGMA journal_mode=WAL")
    return con


def start_writer():
    global _writer_started
    if _writer_started:
        return
    _writer_started = True
    con = _connect()
    con.executescript(SCHEMA)
    con.commit()
    con.close()
    threading.Thread(target=_writer_loop, daemon=True, name="db-writer").start()


def _writer_loop():
    con = _connect()
    while True:
        try:
            sql, params = _write_q.get(timeout=1.0)
            con.execute(sql, params)
            # Drain whatever else is queued, then commit once.
            while True:
                try:
                    sql, params = _write_q.get_nowait()
                    con.execute(sql, params)
                except queue.Empty:
                    break
            con.commit()
        except queue.Empty:
            continue
        except Exception:
            time.sleep(0.5)


def log_observation(camera: str, workers: int, active: int, idle: int):
    _write_q.put((
        "INSERT INTO observations (ts, camera, workers, active, idle) VALUES (?,?,?,?,?)",
        (time.time(), camera, workers, active, idle),
    ))


def log_alert(camera: str, alert_type: str, message: str, snapshot: str | None):
    _write_q.put((
        "INSERT INTO alerts (ts, camera, type, message, snapshot) VALUES (?,?,?,?,?)",
        (time.time(), camera, alert_type, message, snapshot),
    ))


def log_session(camera: str, track_id: int, start_ts: float, end_ts: float,
                duration: float, active_pct: float, posture: str | None = None):
    _write_q.put((
        "INSERT INTO sessions (camera, track_id, start_ts, end_ts, duration, active_pct, posture) "
        "VALUES (?,?,?,?,?,?,?)",
        (camera, int(track_id), start_ts, end_ts, duration, active_pct, posture),
    ))


def recent_sessions(limit: int = 30) -> list[dict]:
    rows = query(
        "SELECT camera, track_id, start_ts, end_ts, duration, active_pct, posture "
        "FROM sessions ORDER BY end_ts DESC LIMIT ?",
        (limit,),
    )
    out = []
    for r in rows:
        mins = r[4] / 60
        act_min = mins * (r[5] or 0) / 100
        out.append(
            {"camera": r[0], "worker": f"W{r[1]}", "start": r[2], "end": r[3],
             "minutes": round(mins, 1), "active_pct": r[5],
             "active_min": round(act_min, 1), "idle_min": round(mins - act_min, 1),
             "posture": r[6]}
        )
    return out


def history(minutes: int = 60) -> list[dict]:
    """Per-minute total workers across all cameras (sum of per-camera averages)."""
    since = time.time() - minutes * 60
    rows = query(
        """SELECT bucket, SUM(avg_w) FROM (
             SELECT CAST(ts/60 AS INTEGER)*60 AS bucket, camera, AVG(workers) AS avg_w
             FROM observations WHERE ts >= ? GROUP BY bucket, camera
           ) GROUP BY bucket ORDER BY bucket""",
        (since,),
    )
    return [{"t": r[0], "workers": round(r[1] or 0, 1)} for r in rows]


def query(sql: str, params: tuple = ()) -> list[tuple]:
    con = _connect()
    try:
        return con.execute(sql, params).fetchall()
    finally:
        con.close()


def recent_alerts(limit: int = 50) -> list[dict]:
    rows = query(
        "SELECT ts, camera, type, message, snapshot FROM alerts ORDER BY ts DESC LIMIT ?",
        (limit,),
    )
    return [
        {"ts": r[0], "camera": r[1], "type": r[2], "message": r[3], "snapshot": r[4]}
        for r in rows
    ]
