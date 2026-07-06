# DEX AI — Monitoring System

AI-powered factory worker monitoring using your **existing CCTV cameras**.
Connect any Hikvision (or ONVIF) NVR/DVR via RTSP and watch live:

- 👷 **Person re-identification (OSNet)** — the system *remembers* every person
  by appearance: same ID across cameras, across days, across restarts
- ✅ **Worker approval** — people who spend real time at machines are flagged;
  the owner approves each as **Worker (W1, W2…)** or **Visitor**, with photo
- 🏭 **Machine zones** — standing at a machine = ACTIVE (even motionless);
  away from every machine (or sitting anywhere) = IDLE
- 🧍 **Posture (MediaPipe full)** — sitting / standing per worker
- 🔍 **Human vs object filter** — a machine part misread as a person is
  quietly dropped after a minute of zero movement, zero pose, low confidence
- 📱 **Mobile phone detection** — instant alert with snapshot
- 📊 **Live web dashboard** — cameras, machine-time leaderboard, approval panel
- 📧 **AI-written reports** — per-shift and per-worker, each worker's own
  cropped photo attached, emailed on click

> The full production deployment (Phase 1) extends this with a 13-camera GPU
> pipeline, TimescaleDB analytics, scheduled PDF shift reports, Telegram
> alerts, and 24/7 on-premises operation.

---

## Quick Start (Windows / Linux)

Requires **Python 3.11 or 3.12** ([download](https://www.python.org/downloads/)).

```bash
# 1. Get the code
git clone <this-repo-url>
cd dexai-monitoring-demo

# 2. Create environment + install (one time, ~5 minutes)
python -m venv .venv
.venv\Scripts\activate          # Windows   (Linux: source .venv/bin/activate)
pip install -r requirements.txt

# 3. Download the AI models + generate offline test video (one time)
python tools/download_models.py
python tools/make_sample.py

# 4. Create your config
copy config.example.yaml config.yaml     # Linux: cp config.example.yaml config.yaml

# 5. Run
python main.py
```

Open **http://localhost:8000** — the dashboard starts with built-in sample footage.

## Connecting Your Hikvision Cameras

1. Find your NVR/DVR's IP address, username, and password.
2. Your RTSP URL format is:

   ```
   rtsp://USERNAME:PASSWORD@NVR_IP:554/Streaming/Channels/102
   ```

   | Channel code | Meaning |
   |---|---|
   | `101` | Camera 1, main stream (full quality) |
   | `102` | Camera 1, sub stream (**recommended** — lighter) |
   | `201` / `202` | Camera 2 main / sub |
   | `301`, `401`... | Camera 3, 4... |

3. Test it first (recommended):

   ```bash
   python tools/test_rtsp.py rtsp://admin:pass123@192.168.1.100:554/Streaming/Channels/102
   ```

4. In the dashboard, use **Add camera** → paste the URL → Connect.

Also accepted as a source: a video file path, or `0` for your laptop webcam.

## Optional Features (`.env`)

Copy `.env.example` to `.env`:

| Setting | What it enables |
|---|---|
| `GEMINI_API_KEY` | Reports written by AI in natural language ([free key](https://aistudio.google.com/apikey)). Without it, a clean built-in template is used. |
| `EMAIL_*` | Reports emailed on click (Gmail: use an App Password). Without it, reports are saved to `reports/` and shown on screen. |

## How It Works

```
NVR / DVR ──RTSP──▶ Frame capture ──▶ YOLO11 detection ──▶ ByteTrack IDs
(your cameras)      (auto-reconnect)   (person + phone)        │
                                                               ▼
                              OSNet re-id (who is this?) + MediaPipe posture
                                                               │
                                                               ▼
Dashboard ◀── FastAPI ◀── SQLite ◀── machine zones / active-idle / alerts
    │
    └─▶ Shift + per-worker reports (LLM) ──▶ Email with worker photos
```

Runs fully **on-premises** — no video ever leaves your network. GPU (NVIDIA)
is used automatically when available; otherwise runs on CPU.

## Troubleshooting

| Problem | Fix |
|---|---|
| RTSP won't connect | Check IP/user/pass; make sure RTSP is enabled on the NVR (Network → Advanced → Integration Protocol on Hikvision) |
| Stream connects but black | Try main stream `101` instead of `102` |
| Slow / laggy video | Normal on CPU — use sub-stream (`102`); production build uses GPU |
| Port 8000 busy | Edit `main.py`, change `port=8000` |

---

**DEX AI** · Monitoring System
