"""Quick RTSP connectivity test — run this BEFORE the demo meeting.

Usage:
    python tools/test_rtsp.py rtsp://user:pass@192.168.1.100:554/Streaming/Channels/102
"""
import sys
import time

import cv2

if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(1)

url = sys.argv[1]
print(f"Connecting to: {url}")
cap = cv2.VideoCapture(url)
if not cap.isOpened():
    print("FAILED: could not open stream. Check IP / username / password / channel.")
    sys.exit(2)

t0 = time.time()
frames = 0
while time.time() - t0 < 5:
    ok, frame = cap.read()
    if ok:
        frames += 1
cap.release()

if frames == 0:
    print("FAILED: connected but received no frames.")
    sys.exit(3)

h, w = frame.shape[:2]
print(f"OK: {frames} frames in 5s (~{frames/5:.0f} FPS), resolution {w}x{h}")
print("This URL is ready to paste into the dashboard.")
