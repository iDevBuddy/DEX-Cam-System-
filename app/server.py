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
    w = CameraWorker(cam["name"], cam["source"], zone, cfg,
                     process=cam.get("process", True),
                     confidence=cam.get("confidence"),
                     machine_zones=cam.get("machine_zones"))
    workers[cam["name"]] = w
    # Loud and unmissable — a silent process:false once cost a debugging
    # session ("camera shows people but zero AI").
    mode = "ON" if w.process_enabled else "OFF  <-- view only, NO detection"
    print(f"[startup] {cam['name']}: AI processing {mode}", flush=True)
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
            "processing": w.process_enabled,
            "ai_paused": w.ai_paused,
            "max_workers": w.zone.max_workers,
            "live_ids": sorted({i["display"] for i in w.live.values()
                                if i["display"] != "..."}),
            "machines": [
                {"name": n, **s} for n, s in
                (w.mstate.states.items() if w.mstate else [])
            ],
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


@app.get("/crops/{fname}")
def person_crop(fname: str):
    from .reid import CROP_DIR
    path = (CROP_DIR / fname).resolve()
    if not path.is_file() or path.parent != CROP_DIR.resolve():
        raise HTTPException(404, "Not found")
    return FileResponse(path)


# ---------------- people (identity & approval) ----------------

CANDIDATE_MACHINE_MIN = 10.0  # unknown person with this much machine time
                              # is probably a worker — ask the owner


def _reid():
    from . import reid
    r = reid.shared()
    if not r.ok:
        raise HTTPException(503, "Re-identification is not available")
    return r


@app.get("/api/persons")
def list_persons():
    r = _reid()
    live_now = {}
    for w in workers.values():
        for info in w.live.values():
            if info.get("pid") is not None:
                live_now[info["pid"]] = {
                    "camera": w.cam_name, "state": info["state"],
                    "machine": info.get("machine"),
                }
    out = []
    for p in r.persons_snapshot():
        p["live"] = live_now.get(p["id"])
        p["candidate"] = (p["label"] == "unknown"
                          and p["machine_min"] >= CANDIDATE_MACHINE_MIN)
        out.append(p)
    # Workers first (by number), then candidates, then the rest by recency.
    out.sort(key=lambda p: (
        0 if p["label"] == "worker" else (1 if p["candidate"] else 2),
        p.get("worker_no") or 999,
        -(p.get("last_seen") or 0),
    ))
    return out


class LabelIn(BaseModel):
    label: str  # 'worker' | 'visitor' | 'unknown'
    worker_no: int | None = None


@app.post("/api/persons/{pid}/label")
def label_person(pid: int, body: LabelIn):
    if body.label not in ("worker", "visitor", "unknown"):
        raise HTTPException(400, "label must be worker, visitor or unknown")
    try:
        return _reid().set_label(pid, body.label, body.worker_no)
    except KeyError:
        raise HTTPException(404, "No such person")


@app.post("/api/persons/{keep}/merge/{absorb}")
def merge_persons(keep: int, absorb: int):
    """Same human got two IDs (clothing change / bad angle) — fold them."""
    try:
        return _reid().merge(keep, absorb)
    except KeyError:
        raise HTTPException(404, "No such person(s)")


@app.get("/api/stats")
def stats():
    total = {"workers": 0, "active": 0, "neutral": 0, "idle": 0}
    online = 0
    for w in workers.values():
        if w.status == "online":
            online += 1
        for k in total:
            total[k] += w.counts.get(k, 0)
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
def worker_report(wid: str, hours: float = 12.0, email: bool = False):
    """Everything we know about one person: where they are right now, machine
    time, visit history — and optionally email it with their photo attached.
    Accepts W1 (approved worker), P5 / V5 (person id), or a bare number."""
    import time as _t
    from datetime import datetime

    r = _reid()
    wid = wid.strip().upper()
    pid = None
    if wid.startswith("W"):
        try:
            pid = r.person_for_worker_no(int(wid[1:]))
        except ValueError:
            raise HTTPException(400, "Worker id looks like: W1, P5 or V5")
        if pid is None:
            raise HTTPException(404, f"No approved worker {wid}. "
                                     "Approve one in the People panel first.")
    else:
        try:
            pid = int(wid.lstrip("PV"))
        except ValueError:
            raise HTTPException(400, "Worker id looks like: W1, P5 or V5")

    people = {p["id"]: p for p in r.persons_snapshot()}
    person = people.get(pid)
    if not person:
        raise HTTPException(404, f"No person with id {pid}")
    disp = person["display"]

    lines = [f"{'WORKER' if person['label'] == 'worker' else 'PERSON'} {disp} — REPORT", ""]

    # Live status across all cameras
    now_lines = []
    for w in workers.values():
        for info in w.live.values():
            if info.get("pid") == pid:
                extra = ""
                if info.get("machine"):
                    extra += (f" at machine '{info['machine']}'"
                              f" ({'RUNNING' if info.get('machine_running') else 'stopped'})")
                if info.get("sitting"):
                    extra += ", SITTING"
                elif info.get("posture"):
                    extra += f", {info['posture']}"
                now_lines.append(
                    f"  RIGHT NOW on '{w.cam_name}': {info['state'].upper()}{extra}"
                )
    lines += now_lines if now_lines else ["  Not visible on any camera right now."]

    # Machine timeline (visits + switches)
    mv = db.query(
        """SELECT machine, camera, start_ts, end_ts, running_pct, sitting_pct,
                  switched_from
           FROM machine_visits WHERE person_id = ? AND end_ts >= ?
           ORDER BY end_ts DESC LIMIT 8""",
        (pid, _t.time() - hours * 3600),
    )
    if mv:
        lines += ["", f"MACHINE TIMELINE (last {hours:g}h):"]
        for mach, cam, ms, me, rp, sp, sw in mv:
            lines.append(
                f"  {mach} ({cam}): "
                f"{datetime.fromtimestamp(ms).strftime('%H:%M')}-"
                f"{datetime.fromtimestamp(me).strftime('%H:%M')} "
                f"({(me - ms) / 60:.1f} min, machine running {rp:g}%"
                + (f", sitting {sp:g}%" if sp else "")
                + (f") <- switched from {sw}" if sw else ")")
            )

    lines += ["", f"ALL-TIME: seen {person['total_min']:g} min total, "
                  f"{person['machine_min']:g} min at machines."]

    # Visit history
    rows = db.query(
        """SELECT camera, start_ts, end_ts, duration, active_pct, posture, machine_pct
           FROM sessions WHERE person_id = ? AND end_ts >= ?
           ORDER BY end_ts DESC LIMIT 10""",
        (pid, _t.time() - hours * 3600),
    )
    lines.append("")
    if rows:
        lines.append(f"VISIT HISTORY (last {hours:g}h):")
        total_min = total_act = total_mach = 0.0
        for cam, start, end, dur, act, posture, mach in rows:
            mins = dur / 60
            act_min = mins * (act or 0) / 100
            mach_min = mins * (mach or 0) / 100
            total_min += mins
            total_act += act_min
            total_mach += mach_min
            lines.append(
                f"  {cam}: {datetime.fromtimestamp(start).strftime('%H:%M')}-"
                f"{datetime.fromtimestamp(end).strftime('%H:%M')} "
                f"({mins:.1f} min: {act_min:.1f} active, {mins - act_min:.1f} idle, "
                f"{mach_min:.1f} at machine"
                + (f", mostly {posture}" if posture else "") + ")"
            )
        if total_min:
            lines += ["", f"TOTAL: {len(rows)} visit(s), {total_min:.1f} min — "
                          f"active {total_act:.1f} min, idle {total_min - total_act:.1f} min, "
                          f"at machine {total_mach:.1f} min "
                          f"({100 * total_act / total_min:.0f}% active)"]
    else:
        lines.append(f"No completed visits recorded in the last {hours:g}h.")

    body = "\n".join(lines)
    result = {"worker": disp, "person_id": pid, "crop": person.get("crop"),
              "report": body}

    if email:
        from .reid import CROP_DIR
        attachments = []
        if person.get("crop") and (CROP_DIR / person["crop"]).is_file():
            attachments.append(CROP_DIR / person["crop"])
        ok, msg = report.send_email(
            f"DEX AI — Report for {disp}", body, None, attachments)
        result["emailed"], result["email_status"] = ok, msg
    return result
