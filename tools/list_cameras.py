#!/usr/bin/env python3
from __future__ import annotations

# Support direct execution from a source checkout.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe camera indices and actual frame sizes")
    parser.add_argument("--max-index", type=int, default=5)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    args = parser.parse_args()
    backend = cv2.CAP_AVFOUNDATION if hasattr(cv2, "CAP_AVFOUNDATION") else cv2.CAP_ANY
    for index in range(args.max_index + 1):
        capture = cv2.VideoCapture(index, backend)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        ok, frame = capture.read()
        capture.release()
        if ok:
            print(f"camera {index}: {frame.shape[1]}x{frame.shape[0]}")


if __name__ == "__main__":
    main()
