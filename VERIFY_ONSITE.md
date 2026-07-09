# VERIFY ON-SITE — kal subah cameras ke saath (updated 2026-07-09 shaam)

> Pichhli visit ka natija: cam-5 HARDWARE DEAD (DVR "NO VIDEO", cam-2
> jaisa — dono electrician se checkwao, shared power/coax ka shak).
> cam-1/3/4 detection verified. CPU 99.8% saturated tha → `infer_fps`
> ab 1.0 hai + cam-5 `process: false`. Neeche kal ke steps.

## Pehle — 2 minute ka basic check

- [ ] Laptop factory WiFi (Fine Artos) pe, `ping 192.168.100.34` chalta hai
- [ ] `python main.py` → startup log: cam-1/3/4 `AI processing ON`,
      cam-5 `AI processing OFF <-- view only` (yeh jaan boojh ke OFF hai)
- [ ] Dashboard http://localhost:8000 — cam-1/3/4 GREEN

## STEP A: CPU verify @ infer_fps 1.0 (pichhli dafa adhura)

1.5 pe bhi 100% tha kyunke per-tick cost (0.68-0.83s) interval se bara
tha — 1.0 pe interval 1.0s hai, ab girna chahiye. Cameras online hone ke
10 min baad:

```
typeperf "\Processor(_Total)\% Processor Time" -si 5 -sc 36
```

- **CPU avg = ______ %** (target: 75-85%)
- Agar phir bhi 95%+ → Claude ko batao: agla lever torch thread tuning
  ya cam-4 per-camera fps hai (imgsz 448 REJECTED — accuracy tabah)
- Dashboard smooth? Report button waqt pe chalta hai?

## STEP B: Machine zones ke asli naam (Akif frames dekh kar batayega)

Workshop = engine rebuilding. Available naam:
`lathe-big, lathe-small, engine-stand, cylinder-boring, crank-grinder,
surface-facer (facer), grinder, drill`

Workflow:
1. Har AI camera ka annotated frame lo (zones drawn):
   ```
   .venv\Scripts\python tools\detect_probe.py factory-cam-1 --seconds 5
   .venv\Scripts\python tools\detect_probe.py factory-cam-3 --seconds 5
   .venv\Scripts\python tools\detect_probe.py factory-cam-4 --seconds 5
   ```
   → `snapshots\probe_*.jpg` Claude ko dikhao / khud dekho
2. Har zone ko naam do (e.g. "cam-4 left zone = lathe-big")
3. Har rename ke liye — history split se bachne ke liye DONO:
   ```
   .venv\Scripts\python tools\rename_machine.py factory-cam-4 top-machine lathe-big
   ```
   (pehle `--dry-run` laga ke ginti dekh lo) **+ config.yaml mein zone
   ka naam badlo** (Claude karega) → server restart
4. Dashboard badges + reports naya naam dikhate hain (dono DB se aate
   hain, rename tool ne history jor di)

Placeholder naam abhi: cam-1 `left-machine`/`right-machine`,
cam-3 `engine-stand`/`bench-right`, cam-4 `top-machine`/`lathe`.

## STEP C: Machine RUNNING/STOPPED calibration (naam final hone ke BAAD)

OFF baselines already pata hain: sab machines 0.0-0.27 energy.
Ab har machine EK EK kar ke ON karwao:

```
.venv\Scripts\python tools\detect_probe.py factory-cam-4 --seconds 30
```

- `machines:` line pe ON energy note karo (umeed: ~2-15)
- Per-zone `motion_threshold` OFF-max aur ON-min ke beech, safety
  margin ke saath (e.g. OFF 0.27, ON 4 → threshold ~1.5-2)
- Zone rang: RUNNING = lime green, stopped = magenta
- Koi machine chalti hui 0 energy de (smooth rotation, 1fps pe
  invisible) → Claude ko batao, us zone ka signal tune hoga.
  Worker ka ACTIVE is se kharab NAHI hota (machine pe khara = active)

## STEP D: Workers re-approve (gallery reset hui thi)

- [ ] People panel mein naye P# bante hain photo ke saath → Worker approve
- [ ] Duplicate P# banay (kapre change / bura angle) → us card pe
      **Duplicate?** dabao, phir asli card pe **✓ Yahan milao**
      (NAYA merge button — ab UI mein hai, sirf API nahi)

## Electrician (jab aaye)

- [ ] cam-2 + cam-5 power/coax check (dono DVR pe "NO VIDEO"; July 6 pe
      cam-5 zinda tha — 2-3 din mein mara, shared supply ka shak)
- [ ] Camera zinda ho jaye → config.yaml mein cam-5 `process: true`
      wapis + machine zones banana (frame capture → Claude)
- [ ] DVR ki date/time bhi theek karwao (06-15 dikha raha tha)

## Agar kuch bhi ajeeb ho

Probe ka output + `snapshots/probe_*.jpg` Claude ko do — exact bata
dega masla detection ka hai, threshold ka, ya zone ka.
