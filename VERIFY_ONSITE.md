# VERIFY ON-SITE — kal cameras ke saath yeh check karna

> Diagnosis 2026-07-09 (cameras offline the). Root cause mil chuka hai —
> **factory-cam-5 ka `process: false` hai** (config.yaml, aakhri camera ka
> aakhri line). Neeche ke steps us fix ko live confirm karne ke liye hain.

## Pehle — 2 minute ka basic check

- [ ] Laptop factory WiFi (Fine Artos) pe hai, `ping 192.168.100.34` chalta hai
- [ ] Dashboard http://localhost:8000 — charon cameras ka dot GREEN (online)
- [ ] cam-1/3/4 pe log dikhen to boxes aur labels aa rahe hain (yeh already
      theek the — regression check)

## Cam-5 fix ke baad (`process: true` ho chuka hai)

- [ ] Server start hote hi console mein har camera ki line dikhti hai:
      `[startup] factory-cam-5: AI processing ON` — koi OFF to nahi?
- [ ] Dashboard pe cam-5 ki line ab "live view only" ki jagah
      `X workers · Y active · Z idle` dikhati hai
- [ ] Cam-5 ke samne 2 bande khare karo → dono pe boxes + labels
- [ ] Counts strip mein workers count barhta hai

## REQUIRED: Cam-5 ke machine zones banana (agle task ki dependency!)

Naye active/idle rules machine zones pe chalte hain — cam-5 pe abhi
KOI machine zone nahi hai, is liye wahan sirf posture/movement rule
chal raha hai. Yeh step lazmi hai:

1. Cam-5 ka ek saaf frame capture karo (koi bhi tareeqa):
   ```
   .venv\Scripts\python tools\detect_probe.py factory-cam-5 --seconds 5
   ```
   → `snapshots\probe_factory-cam-5_*.jpg` ban jayega
2. Us photo mein dekho machines/kaam ki jagahen kahan hain
3. Frame Claude ko do ("cam-5 ke machine zones draw karo") — ya khud
   config.yaml mein cam-5 ke neeche `machine_zones:` add karo
   (coordinates 0..1 normalized, jaise baqi cameras mein hain)
4. Server restart → cam-5 ke video pe magenta zones machines pe
   baith rahe hain? Worker machine pe khara ho to `ACTIVE @ <naam>`?

## REQUIRED: CPU load check (4 cameras AI ke saath, 5 minute)

Ab pehli dafa CHAR cameras ek saath AI process kar rahe hain (pehle 3
the). Load naapna zaroori hai:

1. Charon cameras online + log kaam kar rahe hon (asli load)
2. Yeh command 5 minute chalao (har 5 sec sample, 60 samples):
   ```
   typeperf "\Processor(_Total)\% Processor Time" -si 5 -sc 60
   ```
3. Average number yahan likho: **CPU avg = ______ %**
4. Faisla:
   - 75% se neeche → sab theek, kuch nahi karna
   - 75-90% → cam-5 (ya kisi kam-zaroori camera) ki `infer_fps` per-camera
     kam karne ka socho, ya imgsz 448
   - 90%+ / laptop hang → foran batao, ek camera wapis view-only ya
     model tuning karenge
5. Sath mein dashboard use kar ke dekho — video streams atak to nahi
   rahin, report button waqt pe chalta hai?

## Probe tool — agar cam-5 pe boxes phir bhi na aayen

Terminal mein (server chalta rehne do):

```
cd C:\Users\iakif\dexai-monitoring-demo
.venv\Scripts\python tools\detect_probe.py factory-cam-5 --seconds 30
```

Yeh 30 second tak HAR raw detection print karega (0.05 conf tak) aur
`snapshots\probe_factory-cam-5_*.jpg` mein annotated frame save karega
(green = track shuru hoga, orange = sirf continue, red = bohot kamzor).

**Summary ka matlab:**
- `ZERO person detections` → model ko banda dikh hi nahi raha —
  camera angle/roshni ka masla, config ka nahi
- `would START a track: 0` magar `continue-only` mein numbers →
  cam-5 ki `confidence` (abhi global 0.35) kam karni hai (masalan 0.25)
- `dropped too-small` mein sab kuch → bande frame mein bohot chhote hain
  (camera bohot door/wide) — camera ya zoom ka faisla
- `OUTSIDE-zone` har jagah → zone polygon ghalat jagah hai (magar yeh
  sirf counts rokta hai, boxes nahi)

## Baqi cameras ke liye bhi (naya yolo11s pipeline pehli dafa live)

- [ ] cam-3: baitha hua mechanic — bench pe baithne ke baad bhi track
      zinda rehta hai? (walk-in pe box bane, baithne pe ORANGE ho jaye
      magar ghayab NA ho)
- [ ] cam-4: operator top-machine pe → `ACTIVE @ top-machine` label
- [ ] Har camera ka ek annotated frame dekh kar machine zones (magenta)
      asli machines pe baithte hain — nahi to coordinates batao, adjust
      kar doon ga
- [ ] People panel mein naye P-numbers bante hain, photo ke saath →
      workers ko approve karo (purani gallery reset ho chuki hai)
- [ ] Internet mile to: `git push` (commit 0bbcc62 pending hai)

## Agar kuch bhi ajeeb ho

Probe ka output + `snapshots/probe_*.jpg` mujhe bhej do — exact bata
doon ga masla detection ka hai, threshold ka, ya zone ka.
