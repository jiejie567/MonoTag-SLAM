#!/usr/bin/env python3
from __future__ import annotations

# Support direct execution from a source checkout.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from aruco_track.camera import open_camera
from aruco_track.capture_qc import CaptureQualitySummary, evaluate_capture_quality
from aruco_track.camera_controls import (
    UVCControlError,
    add_camera_control_arguments,
    apply_camera_control_arguments,
    describe_camera_controls,
)
from aruco_track.detector import ArucoDetector
from aruco_track.models import Calibration
from aruco_track.recording import LatestFrameCapture, RawVideoRecorder


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record full-rate raw camera video for offline tracking/SLAM"
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--calib", default="calib/camera_1920x1080.json")
    add_camera_control_arguments(parser)
    parser.add_argument("--output")
    parser.add_argument(
        "--show-marker-ids",
        action="store_true",
        help="show marker IDs in the preview while keeping the saved video unannotated",
    )
    parser.add_argument(
        "--capture-qc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="show lightweight blur/exposure/tag-size checks (default: enabled)",
    )
    args = parser.parse_args()
    output = Path(
        args.output or f"recordings/raw_{datetime.now():%Y%m%d_%H%M%S}.avi"
    )
    calibration = Calibration.load(args.calib)
    try:
        calibration.scaled_to((args.width, args.height))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    capture = open_camera(args.camera, args.width, args.height, args.fps)
    reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
    recording_fps = reported_fps if reported_fps > 1.0 else args.fps
    try:
        camera_controls = apply_camera_control_arguments(args, recording_fps)
    except UVCControlError as exc:
        capture.release()
        raise SystemExit(f"camera control failed: {exc}") from exc
    print(f"camera controls: {camera_controls}")
    camera_control_hud = describe_camera_controls(camera_controls)
    recorder = RawVideoRecorder(
        (args.width, args.height),
        recording_fps,
        metadata={
            "camera_controls": camera_controls,
            "calibration": str(Path(args.calib).resolve()),
            "requested_capture": {
                "width": args.width,
                "height": args.height,
                "fps": args.fps,
            },
        },
    )
    detector = ArucoDetector() if args.show_marker_ids or args.capture_qc else None
    qc_summary = CaptureQualitySummary()
    latest_quality = None
    next_qc_s = 0.0
    frame_capture = LatestFrameCapture(capture, recorder)
    last_sequence = -1
    recorder.start(output)
    frame_capture.start()
    try:
        while True:
            captured = frame_capture.read_latest(last_sequence)
            if captured is None:
                break
            last_sequence, timestamp_s, frame = captured
            preview = frame
            sample_qc = args.capture_qc and timestamp_s >= next_qc_s
            detections = detector.detect(frame) if detector is not None and (
                args.show_marker_ids or sample_qc
            ) else {}
            if sample_qc:
                latest_quality = evaluate_capture_quality(frame, detections)
                qc_summary.add(latest_quality)
                next_qc_s = timestamp_s + 0.25
            if args.show_marker_ids:
                preview = frame.copy()
                for marker_id, corners in detections.items():
                    polygon = np.rint(corners).astype(np.int32).reshape(-1, 1, 2)
                    cv2.polylines(preview, [polygon], True, (0, 220, 255), 2, cv2.LINE_AA)
                    x, y = polygon[0, 0]
                    cv2.putText(
                        preview,
                        str(marker_id),
                        (int(x), int(y) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 220, 255),
                        2,
                        cv2.LINE_AA,
                    )
            if latest_quality is not None:
                cv2.putText(
                    preview,
                    latest_quality.hud(),
                    (16, args.height - 74),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (80, 230, 80) if latest_quality.ready else (0, 180, 255),
                    2,
                    cv2.LINE_AA,
                )
            cv2.putText(
                preview,
                f"RAW REC {recorder.frames}   CAMERA {frame_capture.measured_fps:.1f} FPS"
                "   Q/Esc: stop",
                (16, args.height - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                preview,
                camera_control_hud,
                (16, args.height - 46),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (80, 230, 80) if camera_controls["mode"] == "locked" else (0, 220, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("raw recording - q to stop", preview)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        frame_capture.stop()
        recorder.metadata["capture_qc"] = qc_summary.to_dict()
        saved = recorder.stop()
        cv2.destroyAllWindows()
    if saved is None:
        raise SystemExit("recording stopped before any output was created")
    effective_fps = saved.frames / saved.duration_s if saved.duration_s > 0 else 0.0
    print(
        f"saved {saved.frames} raw frames to {saved.path} "
        f"({effective_fps:.1f} captured FPS, {saved.dropped_frames} writer drops)"
    )
    print(f"capture metadata: {saved.metadata_path}")


if __name__ == "__main__":
    main()
