"""Frame source with auto-reconnect. Supports RTSP URLs, video files, and webcams."""
import os
import time
from pathlib import Path

# Force TCP for RTSP — UDP drops packets on busy factory networks.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;5000000")

import cv2


class FrameSource:
    def __init__(self, source):
        src = str(source).strip()
        self.is_webcam = src.isdigit()
        self.is_file = not self.is_webcam and Path(src).exists()
        self.is_rtsp = src.lower().startswith("rtsp://")
        self.source = int(src) if self.is_webcam else src
        self.cap = None

    def open(self) -> bool:
        self.release()
        self.cap = cv2.VideoCapture(self.source)
        if self.cap.isOpened():
            # Keep the buffer tiny so we always process the freshest frame.
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return True
        return False

    def read(self):
        if self.cap is None or not self.cap.isOpened():
            return None
        ok, frame = self.cap.read()
        if ok:
            return frame
        if self.is_file:
            # Loop video files forever — demo footage never "ends".
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
            if ok:
                return frame
        return None

    def release(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None


def wait_backoff(attempt: int) -> float:
    """Reconnect delay: 2s, 4s, 8s ... capped at 15s."""
    return min(2.0 * (2 ** min(attempt, 3)), 15.0)
