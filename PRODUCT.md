# PRODUCT.md

## What this is
DEX AI Monitoring System — an on-premises AI worker-monitoring dashboard for
packaging factories. Pulls RTSP streams from the factory's existing Hikvision
NVR/DVR, runs YOLOv8 person detection + ByteTrack tracking + MediaPipe posture,
and shows live annotated video, per-worker activity, alerts with photo
evidence, and AI-written shift reports (emailed).

## Who uses it
A factory owner or floor supervisor in Lahore, on a laptop (sometimes a wall
screen) inside an office next to the production floor. Ambient light is mixed;
sessions are long and glanceable — the video feeds are the hero, everything
else supports them. Dark theme is deliberate: annotated CCTV video reads best
on dark, and the tool runs all shift without fatiguing the eye.

## Register
product — design serves the task (monitoring), not the brand. Earned
familiarity over novelty; the closest reference points are NVR/VMS consoles
and ops dashboards, done with Linear-grade restraint.

## Design system (tokens live in app/web/index.html)
- Single-page vanilla HTML/CSS/JS served by FastAPI; no build step, no CDNs
  (must work offline inside a factory LAN).
- Color strategy: Restrained. Dark neutrals + one blue accent (#4f8cff) for
  primary actions/selection only. Semantic state colors: green=active,
  orange=idle/warning, red=phone/critical, cyan=zones/info.
- One type family (system-ui stack), tabular numerals for all counts.
- Spacing on a 4px base; cards only where content is genuinely card-shaped
  (camera feeds); lists elsewhere.
- Motion: 150–250ms state transitions only; reduced-motion honored.

## Constraints
- Client demo build; production Phase 1 adds GPU pipeline, TimescaleDB, PDF
  reports, Telegram.
- Must stay CPU-friendly and dependency-light; UI is one file.
- Urdu-speaking operators: copy stays short, concrete, jargon-free.
