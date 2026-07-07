"""Frame enhancement for poor CCTV feeds: dark, hazy, low-contrast scenes.

Cheap, classical ops (gamma LUT + CLAHE) applied only when measurements say
the frame actually needs them — clean daylight frames pass through untouched.
Detection, posture, re-id crops and the dashboard stream all see the enhanced
frame, so every stage benefits consistently.
"""
import cv2
import numpy as np

DARK_MEAN = 90.0       # below this mean luma the frame counts as dark
FLAT_STD = 42.0        # below this luma spread the frame counts as washed-out
_clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
_gamma_luts: dict[float, np.ndarray] = {}


def _gamma_lut(g: float) -> np.ndarray:
    lut = _gamma_luts.get(g)
    if lut is None:
        lut = (np.power(np.arange(256) / 255.0, g) * 255).astype(np.uint8)
        _gamma_luts[g] = lut
    return lut


def maybe_enhance(frame_bgr):
    """Return the frame, brightened/contrast-stretched only if it needs it."""
    try:
        # Measure on a small grayscale thumbnail — sub-millisecond.
        small = cv2.resize(frame_bgr, (160, 90), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        mean = float(gray.mean())
        std = float(gray.std())

        dark = mean < DARK_MEAN
        flat = std < FLAT_STD
        if not (dark or flat):
            return frame_bgr

        out = frame_bgr
        if dark:
            # Gamma < 1 lifts shadows; deeper lift the darker the scene.
            g = 0.55 if mean < 55 else 0.7
            out = cv2.LUT(out, _gamma_lut(g))
        if flat:
            lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = _clahe.apply(lab[:, :, 0])
            out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return out
    except Exception:
        return frame_bgr  # enhancement must never break the pipeline
