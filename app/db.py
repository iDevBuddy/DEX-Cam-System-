"""SQLite storage. Single writer thread (sqlite likes one writer); readers open
their own short-lived connections."""
import os
import queue
import sqlite3
import threading
import time
from pathlib import Path

# DEX_DB lets tests run against a scratch database without touching real data.
DB_PATH = Path(os.environ.get("DEX_DB")
               or Path(__file__).resolve().parent.parent / "data" / "events.db")

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

CREATE TABLE IF NOT EXISTS persons (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    label       TEXT NOT NULL DEFAULT 'unknown',  -- unknown | worker | visitor
    worker_no   INTEGER,            -- W1..Wn once approved as worker
    first_seen  REAL,
    last_seen   REAL,
    total_s     REAL NOT NULL DEFAULT 0,          -- seconds visible anywhere
    machine_s   REAL NOT NULL DEFAULT 0,          -- seconds standing at a machine
    best_crop   TEXT,               -- filename under snapshots/persons/
    best_crop_h INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS person_embs (
    person_id INTEGER NOT NULL,
    ts        REAL NOT NULL,
    emb       BLOB NOT NULL         -- 512 x float32, L2-normalized
);
CREATE INDEX IF NOT EXISTS idx_pembs_person ON person_embs (person_id);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS machine_states (
    ts      REAL NOT NULL,
    camera  TEXT NOT NULL,
    machine TEXT NOT NULL,
    state   TEXT NOT NULL,          -- 'running' | 'stopped' (change rows only)
    energy  REAL
);
CREATE INDEX IF NOT EXISTS idx_mstates ON machine_states (camera, machine, ts);

CREATE TABLE IF NOT EXISTS machine_visits (
    camera        TEXT NOT NULL,
    machine       TEXT NOT NULL,
    person_id     INTEGER,
    display       TEXT,             -- W1 / P5 at the time of the visit
    start_ts      REAL NOT NULL,
    end_ts        REAL,
    running_pct   REAL,             -- % of the visit the machine was RUNNING
    sitting_pct   REAL,             -- % of the visit spent sitting (compliance)
    switched_from TEXT              -- previous machine if this was a switch
);
CREATE INDEX IF NOT EXISTS idx_mvisits_end ON machine_visits (end_ts);
"""

MIGRATIONS = [
    "ALTER TABLE sessions ADD COLUMN person_id INTEGER",
    "ALTER TABLE sessions ADD COLUMN machine_pct REAL",
    "ALTER TABLE sessions ADD COLUMN idle_pct REAL",
    "ALTER TABLE sessions ADD COLUMN sit_machine_pct REAL",
]


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.execute("PRAGMA journal_mode=WAL")
    return con


def _backup_before_migration():
    """One-time safety copy before the v2 schema migrations touch the file.
    If anything goes wrong, backup_pre_v2.db still works with the old code."""
    bak = DB_PATH.parent / "backup_pre_v2.db"
    if not DB_PATH.exists() or bak.exists():
        return
    try:
        import shutil
        shutil.copy2(DB_PATH, bak)
        for ext in ("-wal", "-shm"):
            side = Path(str(DB_PATH) + ext)
            if side.exists():
                shutil.copy2(side, Path(str(bak) + ext))
    except Exception:
        pass  # backup failure must not block startup


def start_writer():
    global _writer_started
    if _writer_started:
        return
    _writer_started = True
    _backup_before_migration()
    con = _connect()
    con.executescript(SCHEMA)
    for mig in MIGRATIONS:
        try:
            con.execute(mig)
        except sqlite3.OperationalError:
            pass  # column already exists
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
                duration: float, active_pct: float, posture: str | None = None,
                person_id: int | None = None, machine_pct: float | None = None,
                idle_pct: float | None = None,
                sit_machine_pct: float | None = None):
    _write_q.put((
        "INSERT INTO sessions (camera, track_id, start_ts, end_ts, duration, "
        "active_pct, posture, person_id, machine_pct, idle_pct, sit_machine_pct) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (camera, int(track_id), start_ts, end_ts, duration, active_pct, posture,
         person_id, machine_pct, idle_pct, sit_machine_pct),
    ))


# ---------------- machines ----------------

def log_machine_state(camera: str, machine: str, state: str, energy: float):
    """One row per state CHANGE (not per second) — utilization is
    reconstructed from these change points."""
    _write_q.put((
        "INSERT INTO machine_states (ts, camera, machine, state, energy) "
        "VALUES (?,?,?,?,?)",
        (time.time(), camera, machine, state, energy),
    ))


def log_machine_visit(camera: str, machine: str, person_id, display: str,
                      start_ts: float, end_ts: float, running_pct: float,
                      sitting_pct: float, switched_from: str | None):
    _write_q.put((
        "INSERT INTO machine_visits (camera, machine, person_id, display, "
        "start_ts, end_ts, running_pct, sitting_pct, switched_from) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (camera, machine, person_id, display, start_ts, end_ts,
         running_pct, sitting_pct, switched_from),
    ))


def machine_utilization(since: float) -> list[dict]:
    """Running minutes per machine in the window, rebuilt from change rows.
    A machine's state persists from each change row until the next one."""
    now = time.time()
    out = []
    machines = query(
        "SELECT DISTINCT camera, machine FROM machine_states")
    for cam, mach in machines:
        # State at the window start = last change before it (default stopped).
        prev = query(
            "SELECT state FROM machine_states WHERE camera=? AND machine=? "
            "AND ts < ? ORDER BY ts DESC LIMIT 1", (cam, mach, since))
        state = prev[0][0] if prev else "stopped"
        t = since
        running = 0.0
        for (ts, st) in query(
            "SELECT ts, state FROM machine_states WHERE camera=? AND machine=? "
            "AND ts >= ? ORDER BY ts", (cam, mach, since)):
            if state == "running":
                running += ts - t
            t, state = ts, st
        if state == "running":
            running += now - t
        out.append({"camera": cam, "machine": mach,
                    "running_min": round(running / 60, 1),
                    "window_min": round((now - since) / 60, 1)})
    return out


def recent_machine_visits(since: float, limit: int = 20) -> list[dict]:
    rows = query(
        """SELECT camera, machine, display, start_ts, end_ts, running_pct,
                  sitting_pct, switched_from
           FROM machine_visits WHERE end_ts >= ?
           ORDER BY end_ts DESC LIMIT ?""", (since, limit))
    return [
        {"camera": r[0], "machine": r[1], "worker": r[2] or "?",
         "start": r[3], "end": r[4],
         "minutes": round((r[4] - r[3]) / 60, 1),
         "running_pct": r[5], "sitting_pct": r[6], "switched_from": r[7]}
        for r in rows
    ]


def person_display(label: str | None, worker_no, person_id) -> str:
    """W3 for approved workers, V7 for visitors, P5 for not-yet-classified."""
    if label == "worker" and worker_no:
        return f"W{worker_no}"
    if person_id is None:
        return "?"
    return ("V" if label == "visitor" else "P") + str(person_id)


def recent_sessions(limit: int = 30) -> list[dict]:
    rows = query(
        """SELECT s.camera, s.track_id, s.start_ts, s.end_ts, s.duration,
                  s.active_pct, s.posture, s.person_id, s.machine_pct,
                  p.label, p.worker_no
           FROM sessions s LEFT JOIN persons p ON p.id = s.person_id
           ORDER BY s.end_ts DESC LIMIT ?""",
        (limit,),
    )
    out = []
    for r in rows:
        mins = r[4] / 60
        act_min = mins * (r[5] or 0) / 100
        who = person_display(r[9], r[10], r[7]) if r[7] else f"W{r[1]}"
        out.append(
            {"camera": r[0], "worker": who, "start": r[2], "end": r[3],
             "minutes": round(mins, 1), "active_pct": r[5],
             "active_min": round(act_min, 1), "idle_min": round(mins - act_min, 1),
             "machine_min": round(mins * (r[8] or 0) / 100, 1) if r[8] is not None else None,
             "posture": r[6]}
        )
    return out


# ---------------- persons (identity gallery) ----------------

def reset_persons_if_model_changed(model_name: str, crop_dir) -> bool:
    """Embeddings only make sense within one re-id model's space. When the
    model changes, wipe the gallery (workers just get re-approved once).
    Runs synchronously at startup, before camera threads exist."""
    con = _connect()
    try:
        con.executescript(SCHEMA)
        row = con.execute("SELECT value FROM meta WHERE key='reid_model'").fetchone()
        # Galleries created before the meta table existed were built by x0_25.
        prev = row[0] if row else "osnet_x0_25_msmt17.pt"
        changed = prev != model_name
        if changed:
            con.execute("DELETE FROM persons")
            con.execute("DELETE FROM person_embs")
            con.execute("UPDATE sessions SET person_id = NULL")
            try:
                for f in Path(crop_dir).glob("P*.jpg"):
                    f.unlink()
            except Exception:
                pass
        con.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('reid_model', ?)",
            (model_name,),
        )
        con.commit()
        return changed
    finally:
        con.close()

def create_person(first_seen: float, embs: list[bytes]) -> int:
    """Synchronous insert — the new person id is needed immediately.
    Rare event (a few per day), so a short direct write is fine under WAL."""
    con = _connect()
    try:
        cur = con.execute(
            "INSERT INTO persons (label, first_seen, last_seen) VALUES ('unknown', ?, ?)",
            (first_seen, first_seen),
        )
        pid = cur.lastrowid
        for e in embs:
            con.execute(
                "INSERT INTO person_embs (person_id, ts, emb) VALUES (?,?,?)",
                (pid, first_seen, e),
            )
        con.commit()
        return pid
    finally:
        con.close()


def add_person_emb(person_id: int, emb: bytes, keep: int = 16):
    _write_q.put((
        "INSERT INTO person_embs (person_id, ts, emb) VALUES (?,?,?)",
        (person_id, time.time(), emb),
    ))
    _write_q.put((
        "DELETE FROM person_embs WHERE person_id = ? AND ts NOT IN "
        "(SELECT ts FROM person_embs WHERE person_id = ? ORDER BY ts DESC LIMIT ?)",
        (person_id, person_id, keep),
    ))


def update_person_time(person_id: int, total_s: float, machine_s: float,
                       last_seen: float):
    _write_q.put((
        "UPDATE persons SET total_s = ?, machine_s = ?, last_seen = ? WHERE id = ?",
        (total_s, machine_s, last_seen, person_id),
    ))


def update_person_crop(person_id: int, fname: str, crop_h: int):
    _write_q.put((
        "UPDATE persons SET best_crop = ?, best_crop_h = ? WHERE id = ?",
        (fname, crop_h, person_id),
    ))


def set_person_label(person_id: int, label: str, worker_no: int | None):
    _write_q.put((
        "UPDATE persons SET label = ?, worker_no = ? WHERE id = ?",
        (label, worker_no, person_id),
    ))


def load_persons() -> list[dict]:
    rows = query(
        "SELECT id, label, worker_no, first_seen, last_seen, total_s, machine_s, "
        "best_crop, best_crop_h FROM persons"
    )
    persons = []
    for r in rows:
        embs = [e for (e,) in query(
            "SELECT emb FROM person_embs WHERE person_id = ? ORDER BY ts DESC LIMIT 16",
            (r[0],),
        )]
        persons.append({
            "id": r[0], "label": r[1], "worker_no": r[2], "first_seen": r[3],
            "last_seen": r[4], "total_s": r[5] or 0, "machine_s": r[6] or 0,
            "best_crop": r[7], "best_crop_h": r[8] or 0, "embs": embs,
        })
    return persons


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
