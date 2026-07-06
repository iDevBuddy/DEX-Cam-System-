"""MediaPipe posture classifier: SITTING / STANDING per worker.

Strictly optional — if MediaPipe, its model file, or any single call fails,
the system silently falls back to movement-only classification. Nothing here
is allowed to crash a camera thread.
"""
from pathlib import Path

import cv2

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
_FULL = _MODELS_DIR / "pose_landmarker_full.task"
_LITE = _MODELS_DIR / "pose_landmarker_lite.task"
# Full model when available (noticeably better on far/odd-angle workers;
# still fine on CPU at ~1 check per worker per second), else lite.
MODEL_PATH = _FULL if _FULL.exists() else _LITE

# Pose landmark indices (33-point model)
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26


class PostureClassifier:
    def __init__(self):
        self.ok = False
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            self._mp = mp
            options = vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1,
            )
            self._landmarker = vision.PoseLandmarker.create_from_options(options)
            self.ok = True
        except Exception:
            self.ok = False  # missing package/model => feature off, system fine

    def posture(self, person_crop_bgr) -> str | None:
        """Returns 'sitting', 'standing', or None when unsure."""
        if not self.ok or person_crop_bgr is None or person_crop_bgr.size == 0:
            return None
        h, w = person_crop_bgr.shape[:2]
        if h < 48 or w < 24:
            return None  # too small to judge
        try:
            rgb = cv2.cvtColor(person_crop_bgr, cv2.COLOR_BGR2RGB)
            image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb.copy())
            result = self._landmarker.detect(image)
            if not result.pose_landmarks:
                return None
            lm = result.pose_landmarks[0]

            needed = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_KNEE, R_KNEE]
            if any(lm[i].visibility is not None and lm[i].visibility < 0.3 for i in needed):
                return None

            shoulder_y = (lm[L_SHOULDER].y + lm[R_SHOULDER].y) / 2
            hip_y = (lm[L_HIP].y + lm[R_HIP].y) / 2
            knee_y = (lm[L_KNEE].y + lm[R_KNEE].y) / 2
            torso = hip_y - shoulder_y
            if torso <= 0.02:
                return None  # degenerate pose (lying / weird crop)

            # Standing: knees roughly a full torso-length below the hips.
            # Sitting: thighs horizontal, so knees end up near hip height.
            ratio = (knee_y - hip_y) / torso
            return "standing" if ratio > 0.55 else "sitting"
        except Exception:
            return None
