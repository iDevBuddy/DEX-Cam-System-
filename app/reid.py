"""Person re-identification: who is this, across cameras and across restarts.

OSNet (pretrained on MSMT17, ~180k images of 4k+ people) turns a person crop
into a 512-dim appearance embedding. Cosine similarity against a persistent
gallery in SQLite gives every detected person a stable identity (P5) that the
owner can approve as a worker (W1) or dismiss as a visitor (V5).

Strictly optional: if torch, the model file, or any single call fails, the
system falls back to per-track IDs exactly as before. Nothing here may crash
a camera thread.
"""
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from . import db
from .alerts import SNAP_DIR

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
# Prefer the big OSNet when present (noticeably better identity matching on
# blurry CCTV crops); embeddings run ~1x/2s per worker so CPU cost is trivial.
_CANDIDATES = ["osnet_x1_0_msmt17.pt", "osnet_x0_25_msmt17.pt"]
MODEL_PATH = next((_MODELS_DIR / n for n in _CANDIDATES if (_MODELS_DIR / n).exists()),
                  _MODELS_DIR / _CANDIDATES[-1])
CROP_DIR = SNAP_DIR / "persons"

# Matching thresholds (cosine similarity of L2-normalized embeddings).
SIM_MATCH = 0.62      # >= this: same person
SIM_ADD_EMB = 0.87    # below this (but matched): add embedding for diversity
NEW_AFTER = 3         # unmatched samples before we declare a brand-new person
MIN_CROP_H = 96       # px; smaller crops give unreliable embeddings
MIN_CROP_W = 32

_lock = threading.Lock()
_instance = None


def shared():
    """Process-wide singleton (all cameras share one model + one gallery)."""
    global _instance
    with _lock:
        if _instance is None:
            _instance = ReID()
        return _instance


class ReID:
    def __init__(self):
        self.ok = False
        self._model = None
        self._infer_lock = threading.Lock()
        self._gal_lock = threading.Lock()
        self._persons: dict[int, dict] = {}   # pid -> {embs: np(N,512), meta...}
        self._dirty: set[int] = set()
        self._last_flush = time.time()
        try:
            import torch
            from . import osnet as osnet_mod

            builder = (osnet_mod.osnet_x1_0 if "x1_0" in MODEL_PATH.name
                       else osnet_mod.osnet_x0_25)
            model = builder(num_classes=1, pretrained=False)
            ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
            sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            sd = {k.replace("module.", ""): v for k, v in sd.items()
                  if not k.replace("module.", "").startswith("classifier")}
            model.load_state_dict(sd, strict=False)
            model.eval()
            self._torch = torch
            self._model = model
            # Embeddings from different OSNet variants live in different
            # spaces — a model change invalidates the whole gallery.
            db.reset_persons_if_model_changed(MODEL_PATH.name, CROP_DIR)
            self._load_gallery()
            CROP_DIR.mkdir(parents=True, exist_ok=True)
            self.ok = True
        except Exception:
            self.ok = False  # feature off, system fine

    # ---------------- gallery ----------------

    def _load_gallery(self):
        for p in db.load_persons():
            embs = [np.frombuffer(e, dtype=np.float32) for e in p["embs"]]
            embs = [e for e in embs if e.shape == (512,)]
            self._persons[p["id"]] = {
                "embs": np.stack(embs) if embs else np.zeros((0, 512), np.float32),
                "label": p["label"], "worker_no": p["worker_no"],
                "total_s": p["total_s"], "machine_s": p["machine_s"],
                "best_crop": p["best_crop"], "best_crop_h": p["best_crop_h"],
                "first_seen": p["first_seen"], "last_seen": p["last_seen"],
            }

    def display(self, pid: int | None) -> str:
        if pid is None:
            return "?"
        p = self._persons.get(pid)
        if not p:
            return f"P{pid}"
        return db.person_display(p["label"], p["worker_no"], pid)

    def person_for_worker_no(self, worker_no: int) -> int | None:
        with self._gal_lock:
            for pid, p in self._persons.items():
                if p["label"] == "worker" and p["worker_no"] == worker_no:
                    return pid
        return None

    def persons_snapshot(self) -> list[dict]:
        """Safe copy for the API layer."""
        with self._gal_lock:
            out = []
            for pid, p in self._persons.items():
                out.append({
                    "id": pid, "display": self.display(pid), "label": p["label"],
                    "worker_no": p["worker_no"],
                    "first_seen": p["first_seen"], "last_seen": p["last_seen"],
                    "total_min": round((p["total_s"] or 0) / 60, 1),
                    "machine_min": round((p["machine_s"] or 0) / 60, 1),
                    "crop": p["best_crop"],
                })
            return out

    def set_label(self, pid: int, label: str, worker_no: int | None = None) -> dict:
        with self._gal_lock:
            p = self._persons.get(pid)
            if not p:
                raise KeyError(pid)
            if label == "worker" and worker_no is None:
                used = {q["worker_no"] for q in self._persons.values()
                        if q["label"] == "worker" and q["worker_no"]}
                worker_no = next(n for n in range(1, 100) if n not in used)
            if label != "worker":
                worker_no = None
            p["label"], p["worker_no"] = label, worker_no
            db.set_person_label(pid, label, worker_no)
            return {"id": pid, "label": label, "worker_no": worker_no,
                    "display": self.display(pid)}

    def merge(self, keep: int, absorb: int) -> dict:
        """Fold person `absorb` into `keep` (same human seen as two IDs)."""
        with self._gal_lock:
            a, b = self._persons.get(keep), self._persons.get(absorb)
            if not a or not b or keep == absorb:
                raise KeyError((keep, absorb))
            a["embs"] = np.concatenate([a["embs"], b["embs"]])[-16:]
            a["total_s"] += b["total_s"]
            a["machine_s"] += b["machine_s"]
            a["first_seen"] = min(a["first_seen"] or 1e18, b["first_seen"] or 1e18)
            a["last_seen"] = max(a["last_seen"] or 0, b["last_seen"] or 0)
            if b["best_crop_h"] > a["best_crop_h"] and b["best_crop"]:
                a["best_crop"], a["best_crop_h"] = b["best_crop"], b["best_crop_h"]
            self._persons.pop(absorb)
        db.update_person_time(keep, a["total_s"], a["machine_s"], a["last_seen"] or time.time())
        db.update_person_crop(keep, a["best_crop"] or "", a["best_crop_h"])
        db._write_q.put(("UPDATE person_embs SET person_id = ? WHERE person_id = ?",
                         (keep, absorb)))
        db._write_q.put(("UPDATE sessions SET person_id = ? WHERE person_id = ?",
                         (keep, absorb)))
        db._write_q.put(("DELETE FROM persons WHERE id = ?", (absorb,)))
        return {"kept": keep, "absorbed": absorb}

    # ---------------- embedding ----------------

    def embed(self, crop_bgr) -> np.ndarray | None:
        """512-dim L2-normalized appearance embedding, or None."""
        if not self.ok or crop_bgr is None or crop_bgr.size == 0:
            return None
        h, w = crop_bgr.shape[:2]
        if h < MIN_CROP_H or w < MIN_CROP_W:
            return None
        try:
            img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (128, 256), interpolation=cv2.INTER_LINEAR)
            x = img.astype(np.float32) / 255.0
            x = (x - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)
            t = self._torch.from_numpy(x.transpose(2, 0, 1)[None]).float()
            with self._infer_lock, self._torch.no_grad():
                f = self._model(t)[0].numpy()
            n = np.linalg.norm(f)
            return (f / n).astype(np.float32) if n > 0 else None
        except Exception:
            return None

    # ---------------- matching ----------------

    def match(self, emb: np.ndarray) -> tuple[int | None, float]:
        """Best (person_id, similarity) across the gallery."""
        best_pid, best_sim = None, -1.0
        with self._gal_lock:
            for pid, p in self._persons.items():
                if p["embs"].shape[0] == 0:
                    continue
                sim = float(np.max(p["embs"] @ emb))
                if sim > best_sim:
                    best_pid, best_sim = pid, sim
        return best_pid, best_sim

    def assign(self, emb: np.ndarray) -> int | None:
        """Match or None (caller buffers unmatched embeddings per track)."""
        pid, sim = self.match(emb)
        if pid is not None and sim >= SIM_MATCH:
            if sim < SIM_ADD_EMB:
                with self._gal_lock:
                    p = self._persons[pid]
                    p["embs"] = np.concatenate([p["embs"], emb[None]])[-16:]
                db.add_person_emb(pid, emb.tobytes())
            return pid
        return None

    def create(self, embs: list[np.ndarray]) -> int:
        now = time.time()
        pid = db.create_person(now, [e.tobytes() for e in embs])
        with self._gal_lock:
            self._persons[pid] = {
                "embs": np.stack(embs), "label": "unknown", "worker_no": None,
                "total_s": 0.0, "machine_s": 0.0, "best_crop": None,
                "best_crop_h": 0, "first_seen": now, "last_seen": now,
            }
        return pid

    # ---------------- accounting ----------------

    def tick(self, pid: int, seconds: float, at_machine: bool):
        """Accumulate visible time (called ~1x/sec per person per camera)."""
        with self._gal_lock:
            p = self._persons.get(pid)
            if not p:
                return
            p["total_s"] += seconds
            if at_machine:
                p["machine_s"] += seconds
            p["last_seen"] = time.time()
            self._dirty.add(pid)
        now = time.time()
        if now - self._last_flush >= 10.0:
            self._last_flush = now
            self.flush()

    def flush(self):
        with self._gal_lock:
            dirty, self._dirty = self._dirty, set()
            for pid in dirty:
                p = self._persons.get(pid)
                if p:
                    db.update_person_time(pid, p["total_s"], p["machine_s"],
                                          p["last_seen"] or time.time())

    def offer_crop(self, pid: int, crop_bgr):
        """Keep the best (tallest) photo of each person for reports/approval."""
        if crop_bgr is None or crop_bgr.size == 0:
            return
        h = crop_bgr.shape[0]
        with self._gal_lock:
            p = self._persons.get(pid)
            if not p:
                return
            # Replace when clearly better, or refresh a stale photo.
            stale = (p.get("_crop_ts") or 0) < time.time() - 3600
            if h < p["best_crop_h"] and not (stale and h >= MIN_CROP_H):
                return
            fname = f"P{pid}.jpg"
            p["best_crop"], p["best_crop_h"] = fname, h
            p["_crop_ts"] = time.time()
        try:
            cv2.imwrite(str(CROP_DIR / f"P{pid}.jpg"), crop_bgr,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            db.update_person_crop(pid, f"P{pid}.jpg", h)
        except Exception:
            pass
