from __future__ import annotations

import cv2
import time


def open_camera(
    index: int, width: int, height: int, fps: float | None = None
) -> cv2.VideoCapture:
    backend = cv2.CAP_AVFOUNDATION if hasattr(cv2, "CAP_AVFOUNDATION") else cv2.CAP_ANY
    capture = cv2.VideoCapture(index, backend)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps is not None:
        if fps <= 0:
            capture.release()
            raise ValueError("camera FPS must be positive")
        capture.set(cv2.CAP_PROP_FPS, fps)
    deadline = time.monotonic() + 10.0
    ok, frame = capture.read()
    while not ok and time.monotonic() < deadline:
        time.sleep(0.1)
        ok, frame = capture.read()
    if not ok:
        capture.release()
        raise RuntimeError(
            "camera did not return a frame; on macOS grant camera access to Terminal and retry"
        )
    actual = (frame.shape[1], frame.shape[0])
    if actual != (width, height):
        capture.release()
        raise RuntimeError(f"requested {width}x{height}, camera returned {actual[0]}x{actual[1]}")
    return capture
