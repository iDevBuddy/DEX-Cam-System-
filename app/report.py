"""Shift report: SQLite stats → (optional) Gemini LLM narrative → (optional) email.
Always works: no API key => template report; no SMTP => saved to reports/ only."""
import smtplib
import time
from datetime import datetime
from email.mime.text import MIMEText
from pathlib import Path

import httpx

from . import db
from .config import env

REPORT_DIR = Path(__file__).resolve().parent.parent / "reports"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"


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
    return {
        "window_hours": hours,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "cameras": cams,
    }


def render_template(stats: dict) -> str:
    lines = [
        "DEX AI — WORKER MONITORING REPORT (DEMO)",
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
        alerts = s.get("alerts", {})
        if alerts:
            lines.append("  Alerts: " + ", ".join(f"{k}={v}" for k, v in alerts.items()))
        else:
            lines.append("  Alerts: none")
    lines += ["", "-" * 55,
              "Demo build — Phase 1 adds pose-based activity, shift PDF reports,",
              "historical analytics, and Telegram alerts."]
    return "\n".join(lines)


def llm_report(stats: dict) -> str | None:
    key = env("GEMINI_API_KEY")
    if not key:
        return None
    prompt = (
        "You are the reporting module of an AI factory worker-monitoring system "
        "built by DEX AI. Write a short, professional shift report (max 250 words) "
        "for a factory owner in simple English based on this data. Use short "
        "sections with clear headings, mention notable idle time and alerts, and "
        "end with one practical recommendation. Data:\n" + str(stats)
    )
    try:
        r = httpx.post(
            GEMINI_URL,
            params={"key": key},
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=25,
        )
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        return None  # any failure -> template fallback


def send_email(subject: str, body: str, to_addr: str | None = None) -> tuple[bool, str]:
    host, user, pwd = env("EMAIL_HOST"), env("EMAIL_USER"), env("EMAIL_PASS")
    to_addr = to_addr or env("EMAIL_TO") or user
    if not (host and user and pwd and to_addr):
        return False, "Email not configured (.env) — report saved locally."
    try:
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = user
        msg["To"] = to_addr
        with smtplib.SMTP_SSL(host, int(env("EMAIL_PORT", "465")), timeout=20) as s:
            s.login(user, pwd)
            s.sendmail(user, [to_addr], msg.as_string())
        return True, f"Report emailed to {to_addr}."
    except Exception as e:
        return False, f"Email failed: {e}"


def generate(hours: float = 12.0, email_to: str | None = None) -> dict:
    stats = build_stats(hours)
    llm_text = llm_report(stats)
    body = llm_text or render_template(stats)
    source = "gemini" if llm_text else "template"

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    (REPORT_DIR / fname).write_text(body, encoding="utf-8")

    emailed, email_msg = send_email("DEX AI — Shift Monitoring Report (Demo)", body, email_to)
    return {"report": body, "source": source, "file": fname,
            "emailed": emailed, "email_status": email_msg}
