"""FastAPI app: dashboard UI, MJPEG streams, camera management, alerts, reports."""
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

from . import config, db, detector, report
from .alerts import SNAP_DIR
from .pipeline import CameraWorker
from .zones import Zone

ROOT = Path(__file__).resolve().parent
DEFAULT_ZONE = [[0.03, 0.1], [0.97, 0.1], [0.97, 0.97], [0.03, 0.97]]

app = FastAPI(title="DEX AI Monitoring System")
workers: dict[str, CameraWorker] = {}
cfg: dict = {}


def start_camera(cam: dict):
    zone = Zone(cam.get("zone") or DEFAULT_ZONE, cam.get("max_workers", 3))
    w = CameraWorker(cam["name"], cam["source"], zone, cfg)
    workers[cam["name"]] = w
    w.start()


@app.on_event("startup")
def startup():
    global cfg
    cfg = config.load()
    db.start_writer()
    detector.init(cfg["inference"]["model"])
    for cam in cfg.get("cameras") or []:
        start_camera(cam)


# ---------------- UI ----------------

@app.get("/", response_class=HTMLResponse)
def index():
    return (ROOT / "web" / "index.html").read_text(encoding="utf-8")


# ---------------- cameras ----------------

class CameraIn(BaseModel):
    name: str
    source: str
    max_workers: int = 3


@app.get("/api/cameras")
def list_cameras():
    return [
        {
            "name": w.cam_name,
            "source": w.source_str,
            "status": w.status,
            "max_workers": w.zone.max_workers,
            "live_ids": [f"W{t}" for t in sorted(w.live)],
            **w.counts,
        }
        for w in workers.values()
    ]


@app.post("/api/cameras")
def add_camera(cam: CameraIn):
    name = cam.name.strip().replace(" ", "-")
    if not name:
        raise HTTPException(400, "Camera name required")
    if name in workers:
        raise HTTPException(400, f"Camera '{name}' already exists")
    entry = {"name": name, "source": cam.source.strip(),
             "max_workers": cam.max_workers, "zone": DEFAULT_ZONE}
    start_camera(entry)
    cfg.setdefault("cameras", []).append(entry)
    config.save(cfg)
    return {"ok": True, "name": name}


@app.delete("/api/cameras/{name}")
def remove_camera(name: str):
    w = workers.pop(name, None)
    if w is None:
        raise HTTPException(404, "No such camera")
    w.stop()
    cfg["cameras"] = [c for c in cfg.get("cameras") or [] if c["name"] != name]
    config.save(cfg)
    return {"ok": True}


# ---------------- video ----------------

@app.get("/stream/{name}")
def stream(name: str):
    if name not in workers:
        raise HTTPException(404, "No such camera")

    def gen():
        while name in workers:
            jpg = workers[name].latest_jpeg()
            if jpg:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
                       b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n"
                       + jpg + b"\r\n")
            time.sleep(0.15)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


# ---------------- alerts & stats ----------------

@app.get("/api/alerts")
def alerts(limit: int = 30):
    return db.recent_alerts(limit)


@app.get("/api/sessions")
def sessions(limit: int = 30):
    return db.recent_sessions(limit)


@app.get("/api/history")
def history(minutes: int = 60):
    return db.history(minutes)


@app.get("/snapshots/{fname}")
def snapshot(fname: str):
    path = (SNAP_DIR / fname).resolve()
    if not path.is_file() or path.parent != SNAP_DIR.resolve():
        raise HTTPException(404, "Not found")
    return FileResponse(path)


@app.get("/api/stats")
def stats():
    total = {"workers": 0, "active": 0, "idle": 0}
    online = 0
    for w in workers.values():
        if w.status == "online":
            online += 1
        for k in total:
            total[k] += w.counts[k]
    alerts_today = db.query(
        "SELECT COUNT(*) FROM alerts WHERE ts >= ?", (time.time() - 86400,)
    )[0][0]
    return {"cameras": len(workers), "online": online,
            "alerts_24h": alerts_today, "device": detector.device(), **total}


# ---------------- reports ----------------

class ReportIn(BaseModel):
    hours: float = 12.0
    email_to: str | None = None


@app.post("/api/report")
def make_report(req: ReportIn):
    return report.generate(req.hours, req.email_to)


@app.get("/api/worker/{wid}")
def worker_report(wid: str, hours: float = 12.0):
    """Everything we know about one worker ID: where they are right now,
    plus their visit history."""
    try:
        tid = int(wid.upper().lstrip("W"))
    except ValueError:
        raise HTTPException(400, "Worker id looks like: W3 or 3")

    from datetime import datetime
    lines = [f"WORKER W{tid} — REPORT", ""]

    # Live status across all cameras
    now_lines = []
    for w in workers.values():
        info = w.live.get(tid)
        if info:
            posture = f", {info['posture']}" if info["posture"] else ""
            now_lines.append(
                f"  RIGHT NOW on '{w.cam_name}': {info['state'].upper()}{posture}"
            )
    lines += now_lines if now_lines else ["  Not visible on any camera right now."]

    # Visit history
    import time as _t
    rows = db.query(
        """SELECT camera, start_ts, end_ts, duration, active_pct, posture
           FROM sessions WHERE track_id = ? AND end_ts >= ?
           ORDER BY end_ts DESC LIMIT 10""",
        (tid, _t.time() - hours * 3600),
    )
    lines.append("")
    if rows:
        lines.append(f"VISIT HISTORY (last {hours:g}h):")
        total_min = total_act = 0.0
        for cam, start, end, dur, act, posture in rows:
            mins = dur / 60
            act_min = mins * (act or 0) / 100
            total_min += mins
            total_act += act_min
            lines.append(
                f"  {cam}: {datetime.fromtimestamp(start).strftime('%H:%M')}-"
                f"{datetime.fromtimestamp(end).strftime('%H:%M')} "
                f"({mins:.1f} min: {act_min:.1f} active, {mins - act_min:.1f} idle"
                + (f", mostly {posture}" if posture else "") + ")"
            )
        lines += ["", f"TOTAL: {len(rows)} visit(s), {total_min:.1f} min — "
                      f"active {total_act:.1f} min, idle {total_min - total_act:.1f} min "
                      f"({100 * total_act / total_min:.0f}% active)" if total_min else ""]
    else:
        lines.append(f"No completed visits recorded in the last {hours:g}h.")

    return {"worker": f"W{tid}", "report": "\n".join(lines)}
