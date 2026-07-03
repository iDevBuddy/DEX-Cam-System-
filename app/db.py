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
