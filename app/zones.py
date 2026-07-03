"""Work-zone polygon: normalized (0..1) coords scaled to each frame."""
import numpy as np
import cv2


class Zone:
    def __init__(self, polygon_norm, max_workers: int):
        self.polygon_norm = polygon_norm  # [[x, y], ...] in 0..1
        self.max_workers = int(max_workers)
        self._cache = {}  # (w, h) -> np.array pixel polygon

    def pixels(self, w: int, h: int) -> np.ndarray:
        key = (w, h)
        if key not in self._cache:
            self._cache[key] = np.array(
                [[int(x * w), int(y * h)] for x, y in self.polygon_norm],
                dtype=np.int32,
            )
        return self._cache[key]

    def contains(self, point, w: int, h: int) -> bool:
        poly = self.pixels(w, h)
        return cv2.pointPolygonTest(poly, (float(point[0]), float(point[1])), False) >= 0
