#!/usr/bin/env python3
from __future__ import annotations

# Support direct execution from a source checkout.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
from datetime import datetime
from pathlib import Path
import time

import cv2

from aruco_track.camera import open_camera
from aruco_track.camera_controls import UVCController, describe_camera_controls
from aruco_track.recording import LatestFrameCapture, RawRecordingResult, RawVideoRecorder


def draw_status(frame, title: str, detail: str):
    output = frame.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 120), (0, 0, 0), -1)
    cv2.putText(
        output, title, (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
        (255, 255, 255), 2, cv2.LINE_AA,
    )
    cv2.putText(
        output, detail, (24, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
        (0, 220, 255), 2, cv2.LINE_AA,
    )
    return output


def capture_phase(
    args,
    width: int,
    height: int,
    fps: float,
    run_id: str,
    wait_for_space: bool,
) -> RawRecordingResult:
    capture = open_camera(args.camera, width, height, fps)
    reported_fps = float(capture.get(cv2.CAP_PROP_FPS)) or fps
    settings = UVCController(args.uvc_device).apply_locked(
        exposure_us=args.exposure_us,
        gain=args.gain,
        white_balance=args.white_balance,
        power_line_frequency_hz=50,
        fps=reported_fps,
    )
    path = args.output_dir / f"fps_compare_{height}p{int(fps)}_{run_id}.avi"
    recorder = RawVideoRecorder(
        (width, height),
        reported_fps,
        metadata={"camera_controls": settings, "comparison_fps": fps},
    )
    stream = LatestFrameCapture(capture, recorder)
    stream.start()
    sequence = -1
    try:
        while wait_for_space:
            captured = stream.read_latest(sequence)
            if captured is None:
                raise RuntimeError("camera stopped while waiting to start")
            sequence, _, frame = captured
            preview = draw_status(
                frame,
                "AIM CAMERA AT WRISTS AND MARKERS",
                "Press SPACE to start the 1080p60 / 720p120 comparison",
            )
            cv2.imshow("60 vs 120 FPS capture", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                raise KeyboardInterrupt
            wait_for_space = key != ord(" ")

        countdown_end = time.monotonic() + args.countdown
        while time.monotonic() < countdown_end:
            captured = stream.read_latest(sequence)
            if captured is None:
                raise RuntimeError("camera stopped during countdown")
            sequence, _, frame = captured
            remaining = max(1, int(countdown_end - time.monotonic()) + 1)
            preview = draw_status(
                frame,
                f"NEXT: {int(fps)} FPS - START IN {remaining}",
                "Repeat the same fast wrist rotation and waving motion",
            )
            cv2.imshow("60 vs 120 FPS capture", preview)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                raise KeyboardInterrupt

        recorder.start(path)
        record_end = time.monotonic() + args.duration
        while time.monotonic() < record_end:
            captured = stream.read_latest(sequence)
            if captured is None:
                raise RuntimeError("camera stopped during recording")
            sequence, _, frame = captured
            remaining = max(0.0, record_end - time.monotonic())
            preview = draw_status(
                frame,
                f"RECORDING {int(fps)} FPS  {remaining:04.1f}s",
                f"{describe_camera_controls(settings)}  captured={recorder.frames}",
            )
            cv2.imshow("60 vs 120 FPS capture", preview)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                raise KeyboardInterrupt
    finally:
        stream.stop()
        result = recorder.stop()
    if result is None:
        raise RuntimeError(f"{int(fps)} FPS phase did not record")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Record matched 60/120 FPS camera tests")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--countdown", type=float, default=3.0)
    parser.add_argument("--exposure-us", type=int, default=8_000)
    parser.add_argument("--gain", type=int, default=160)
    parser.add_argument("--white-balance", type=int, default=128)
    parser.add_argument("--uvc-device", default="1d6b:0102")
    parser.add_argument("--output-dir", type=Path, default=Path("recordings"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = []
    try:
        phases = ((1920, 1080, 60.0), (1280, 720, 120.0))
        for index, (width, height, fps) in enumerate(phases):
            results.append(
                capture_phase(args, width, height, fps, run_id, wait_for_space=index == 0)
            )
    finally:
        cv2.destroyAllWindows()
    for result in results:
        effective_fps = result.frames / result.duration_s if result.duration_s else 0.0
        print(
            f"{result.path}: {result.frames} frames, {effective_fps:.2f} FPS, "
            f"writer drops={result.dropped_frames}"
        )


if __name__ == "__main__":
    main()
