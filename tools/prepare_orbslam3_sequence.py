#!/usr/bin/env python3
"""Convert a recorded video to the TUM-style sequence read by ORB-SLAM3."""

from __future__ import annotations

# Support direct execution from a source checkout.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import cv2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    args = parser.parse_args()

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        raise SystemExit("Video does not report a valid frame rate")

    rgb_dir = args.output / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    rgb_lines = ["# color images", "# timestamp filename", "#"]
    frame_index = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        relative_path = Path("rgb") / f"{frame_index:06d}.jpg"
        target = args.output / relative_path
        if not cv2.imwrite(
            str(target), frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
        ):
            raise SystemExit(f"Cannot write image: {target}")
        timestamp = frame_index / fps
        rgb_lines.append(f"{timestamp:.9f} {relative_path.as_posix()}")
        frame_index += 1

    capture.release()
    (args.output / "rgb.txt").write_text("\n".join(rgb_lines) + "\n")
    print(f"Wrote {frame_index} frames at {fps:.3f} fps to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
