"""Shift report: SQLite stats → (optional) Gemini LLM narrative → (optional) email.
Always works: no API key => template report; no SMTP => saved to reports/ only."""
import smtplib
import time
from datetime import datetime
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import httpx

from . import db
from .alerts import SNAP_DIR
from .config import env

REPORT_DIR = Path(__file__).resolve().parent.parent / "reports"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def build_stats(hours: float = 12.0) -> dict:
    since = time.time() - hours * 3600
    cams = {}
    for cam, secs, avg_w, peak_w, act, idl in db.query(
        """SELECT camera, COUNT(*), AVG(workers), MAX(workers),
                  SUM(active), SUM(idle)
           FROM observations WHERE ts >= ? GROUP BY camera""",
        (since,),
    ):
        total = (act or 0) + (idl or 0)
        cams[cam] = {
            "observed_minutes": round(secs / 60, 1),
            "avg_workers": round(avg_w or 0, 1),
            "peak_workers": int(peak_w or 0),
            "active_pct": round(100 * (act or 0) / total, 1) if total else 0.0,
            "idle_pct": round(100 * (idl or 0) / total, 1) if total else 0.0,
            "alerts": {},
        }
    for cam, atype, n in db.query(
        "SELECT camera, type, COUNT(*) FROM alerts WHERE ts >= ? GROUP BY camera, type",
        (since,),
    ):
        cams.setdefault(cam, {"alerts": {}})["alerts"][atype] = n

    # Worker visit sessions per camera
    for cam, visits, avg_min, avg_act in db.query(
        """SELECT camera, COUNT(*), AVG(duration)/60.0, AVG(active_pct)
           FROM sessions WHERE end_ts >= ? GROUP BY camera""",
        (since,),
    ):
        cams.setdefault(cam, {"alerts": {}})["visits"] = {
            "count": visits,
            "avg_stay_min": round(avg_min or 0, 1),
            "avg_active_pct": round(avg_act or 0, 1),
        }

    # Hour-by-hour breakdown per camera (local time)
    for cam, hour, avg_w, act, idl in db.query(
        """SELECT camera, strftime('%H:00', ts, 'unixepoch', 'localtime') AS hr,
                  AVG(workers), SUM(active), SUM(idle)
           FROM observations WHERE ts >= ? GROUP BY camera, hr ORDER BY hr""",
        (since,),
    ):
        total = (act or 0) + (idl or 0)
        cams.setdefault(cam, {"alerts": {}}).setdefault("hourly", []).append({
            "hour": hour,
            "avg_workers": round(avg_w or 0, 1),
            "active_pct": round(100 * (act or 0) / total, 1) if total else 0.0,
        })
    # Recent individual worker sessions (most recent first)
    workers = []
    for cam, tid, start, end, dur, act, posture, pid, mach, plabel, wno in db.query(
        """SELECT s.camera, s.track_id, s.start_ts, s.end_ts, s.duration,
                  s.active_pct, s.posture, s.person_id, s.machine_pct,
                  p.label, p.worker_no
           FROM sessions s LEFT JOIN persons p ON p.id = s.person_id
           WHERE s.end_ts >= ? ORDER BY s.end_ts DESC LIMIT 15""",
        (since,),
    ):
        mins = dur / 60
        act_min = mins * (act or 0) / 100
        who = db.person_display(plabel, wno, pid) if pid else f"W{tid}"
        workers.append({
            "worker": who,
            "camera": cam,
            "from": datetime.fromtimestamp(start).strftime("%H:%M"),
            "to": datetime.fromtimestamp(end).strftime("%H:%M"),
            "minutes": round(mins, 1),
            "active_pct": act,
            "active_min": round(act_min, 1),
            "idle_min": round(mins - act_min, 1),
            "machine_min": round(mins * (mach or 0) / 100, 1),
            "posture": posture or "unknown",
        })

    # Known people: approved workers + anyone with real machine time
    people = []
    for pid, label, wno, total_s, machine_s, crop in db.query(
        """SELECT id, label, worker_no, total_s, machine_s, best_crop
           FROM persons
           WHERE label = 'worker' OR machine_s >= 120
           ORDER BY CASE label WHEN 'worker' THEN 0 ELSE 1 END,
                    worker_no, machine_s DESC LIMIT 12"""
    ):
        people.append({
            "id": db.person_display(label, wno, pid),
            "status": label,
            "total_min": round((total_s or 0) / 60, 1),
            "machine_min": round((machine_s or 0) / 60, 1),
            "crop": crop,
        })

    return {
        "window_hours": hours,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "cameras": cams,
        "workers": workers,
        "people": people,
    }


def render_template(stats: dict) -> str:
    lines = [
        "DEX AI — MONITORING SYSTEM REPORT",
        f"Generated: {stats['generated_at']}   Window: last {stats['window_hours']:g} hours",
        "=" * 55,
    ]
    if not stats["cameras"]:
        lines.append("No monitoring data recorded in this window yet.")
    for cam, s in stats["cameras"].items():
        lines += [
            "",
            f"CAMERA: {cam}",
            f"  Monitored time     : {s.get('observed_minutes', 0)} min",
            f"  Avg workers in zone: {s.get('avg_workers', 0)}   (peak {s.get('peak_workers', 0)})",
            f"  Active time        : {s.get('active_pct', 0)}%",
            f"  Idle time          : {s.get('idle_pct', 0)}%",
        ]
        v = s.get("visits")
        if v:
            lines.append(
                f"  Worker visits      : {v['count']} "
                f"(avg stay {v['avg_stay_min']} min, avg active {v['avg_active_pct']}%)"
            )
        alerts = s.get("alerts", {})
        if alerts:
            lines.append("  Alerts: " + ", ".join(f"{k}={v}" for k, v in alerts.items()))
        else:
            lines.append("  Alerts: none")
        hourly = s.get("hourly") or []
        if hourly:
            lines.append("  Hour-by-hour:")
            for hb in hourly:
                lines.append(
                    f"    {hb['hour']}  avg workers {hb['avg_workers']:<4}  "
                    f"active {hb['active_pct']}%"
                )
    people = stats.get("people") or []
    if people:
        lines += ["", "PEOPLE (identified by AI)", "-" * 55]
        for p in people:
            tag = p["status"].upper() if p["status"] != "unknown" else "UNCONFIRMED"
            lines.append(
                f"  {p['id']:<5} {tag:<12} seen {p['total_min']:>6} min, "
                f"at machines {p['machine_min']:>6} min"
            )
    workers = stats.get("workers") or []
    if workers:
        lines += ["", "WORKER DETAIL (recent visits)", "-" * 55]
        for wk in workers:
            lines.append(
                f"  {wk['worker']:<5} {wk['camera']:<16} {wk['from']}-{wk['to']}  "
                f"{wk['minutes']:>5} min (active {wk['active_min']}, "
                f"idle {wk['idle_min']}, machine {wk.get('machine_min', 0)})  "
                f"{wk['posture']}"
            )
    lines += ["", "-" * 55,
              "DEX AI Monitoring System — automated report."]
    return "\n".join(lines)


PROMPT = (
    "You are the reporting module of an AI factory worker-monitoring system "
    "built by DEX AI. Write a short, professional shift report (max 300 words) "
    "for a factory owner in simple English based on this data. Use short "
    "sections with clear headings. Describe individual workers by their IDs "
    "(W1, W2... are approved workers; P-numbers are unidentified people; "
    "V-numbers are visitors) — what they did, how long they stayed, how much "
    "time they spent working at machines versus idle. 'machine minutes' means "
    "time standing at a machine, which counts as productive work. Mention "
    "alerts, then end with one practical recommendation. Output plain text "
    "only, no markdown symbols. Data:\n"
)


def llm_report(stats: dict) -> str | None:
    """OpenRouter first, Gemini second, None (=template) if both unavailable."""
    prompt = PROMPT + str(stats)

    or_key = env("OPENROUTER_API_KEY")
    if or_key:
        # Free-tier models get rate-limited (429); try each until one answers.
        models = [m.strip() for m in env(
            "OPENROUTER_MODEL", "google/gemma-4-31b-it:free"
        ).split(",") if m.strip()]
        for fallback in ["google/gemma-4-26b-a4b-it:free",
                         "meta-llama/llama-3.3-70b-instruct:free"]:
            if fallback not in models:
                models.append(fallback)
        for model in models:
            try:
                r = httpx.post(
                    OPENROUTER_URL,
                    headers={"Authorization": f"Bearer {or_key}"},
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 700,
                    },
                    timeout=45,
                )
                r.raise_for_status()
                text = r.json()["choices"][0]["message"]["content"].strip()
                if text:
                    return text
            except Exception:
                continue  # next model

    g_key = env("GEMINI_API_KEY")
    if g_key:
        try:
            r = httpx.post(
                GEMINI_URL,
                params={"key": g_key},
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=25,
            )
            r.raise_for_status()
            return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception:
            pass

    return None  # template fallback


def snapshot_attachments(hours: float, limit: int = 7) -> list[Path]:
    """Photos worth attaching: each identified worker's own cropped photo
    first, then idle-worker alert snapshots, newest first. Missing files are
    skipped silently."""
    since = time.time() - hours * 3600
    paths: list[Path] = []
    seen = set()

    # Per-worker cropped photos (people seen in this window).
    crop_dir = SNAP_DIR / "persons"
    for (crop,) in db.query(
        """SELECT best_crop FROM persons
           WHERE best_crop IS NOT NULL AND last_seen >= ?
           ORDER BY CASE label WHEN 'worker' THEN 0 ELSE 1 END,
                    worker_no, machine_s DESC LIMIT 4""",
        (since,),
    ):
        p = crop_dir / crop
        if crop not in seen and p.is_file():
            seen.add(crop)
            paths.append(p)

    rows = db.query(
        """SELECT snapshot FROM alerts
           WHERE ts >= ? AND snapshot IS NOT NULL
           ORDER BY CASE type WHEN 'idle_worker' THEN 0 ELSE 1 END, ts DESC
           LIMIT ?""",
        (since, limit * 2),
    )
    for (name,) in rows:
        p = SNAP_DIR / name
        if name not in seen and p.is_file():
            seen.add(name)
            paths.append(p)
        if len(paths) >= limit:
            break
    return paths


def send_email(subject: str, body: str, to_addr: str | None = None,
               attachments: list[Path] | None = None) -> tuple[bool, str]:
    host, user, pwd = env("EMAIL_HOST"), env("EMAIL_USER"), env("EMAIL_PASS")
    to_addr = to_addr or env("EMAIL_TO") or user
    if not (host and user and pwd and to_addr):
        return False, "Email not configured (.env) — report saved locally."
    try:
        msg = MIMEMultipart()
        msg["Subject"] = subject
        msg["From"] = user
        msg["To"] = to_addr
        msg.attach(MIMEText(body, "plain", "utf-8"))
        for path in attachments or []:
            try:
                img = MIMEImage(path.read_bytes(), name=path.name)
                img.add_header("Content-Disposition", "attachment", filename=path.name)
                msg.attach(img)
            except Exception:
                continue  # one bad image must not kill the report email
        with smtplib.SMTP_SSL(host, int(env("EMAIL_PORT", "465")), timeout=20) as s:
            s.login(user, pwd)
            s.sendmail(user, [to_addr], msg.as_string())
        n = len(attachments or [])
        return True, f"Report emailed to {to_addr}" + (f" with {n} photo(s)." if n else ".")
    except Exception as e:
        return False, f"Email failed: {e}"


def generate(hours: float = 12.0, email_to: str | None = None) -> dict:
    stats = build_stats(hours)
    llm_text = llm_report(stats)
    body = llm_text or render_template(stats)
    source = "ai" if llm_text else "template"

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    (REPORT_DIR / fname).write_text(body, encoding="utf-8")

    attachments = snapshot_attachments(hours)
    emailed, email_msg = send_email("DEX AI — Shift Monitoring Report", body,
                                    email_to, attachments)
    return {"report": body, "source": source, "file": fname,
            "emailed": emailed, "email_status": email_msg,
            "attachments": [p.name for p in attachments]}
