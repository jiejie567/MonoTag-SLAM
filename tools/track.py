#!/usr/bin/env python3
from __future__ import annotations

# Support direct execution from a source checkout.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path
import subprocess
import sys
import time

import cv2

from aruco_track.camera import open_camera
from aruco_track.capture_qc import CaptureQualitySummary, evaluate_capture_quality
from aruco_track.camera_controls import (
    UVCControlError,
    add_camera_control_arguments,
    apply_camera_control_arguments,
    describe_camera_controls,
)
from aruco_track.hands import HandJointTracker
from aruco_track.hawor_backend import DEFAULT_HAWOR_CONFIG
from aruco_track.models import BandLayout, Calibration
from aruco_track.pipeline import TrackingPipeline
from aruco_track.recording import LatestFrameCapture, RawRecordingResult, RawVideoRecorder
from aruco_track.render import FadingTrajectory, draw_result


def parse_single(value: str) -> tuple[int, float]:
    marker_id, size_mm = value.split(":", 1)
    return int(marker_id), float(size_mm) / 1000.0


def print_recording(result: RawRecordingResult) -> None:
    effective_fps = result.frames / result.duration_s if result.duration_s > 0 else 0.0
    print(
        f"saved {result.frames} raw frames to {result.path} "
        f"({effective_fps:.1f} captured FPS, {result.dropped_frames} writer drops)"
    )
    print(f"capture metadata: {result.metadata_path}")


def offline_processing_command(
    video: Path,
    calibration: str,
    bands: list[str],
    world_board: str | None,
    hand_model: str,
    auto_marker_map: bool = True,
    static_marker_ids: str = "20-49",
    static_marker_size_mm: float = 48.0,
    hand_joints: bool = True,
    hand_backend: str = "hawor",
    hawor_config: str | Path = DEFAULT_HAWOR_CONFIG,
    hawor_device: str = "auto",
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "tools/export_action_labels.py"),
        str(video),
        "--calib",
        calibration,
        "--hand-model",
        hand_model,
        "--hand-backend",
        hand_backend,
    ]
    if hand_backend == "hawor":
        command.extend(("--hawor-config", str(hawor_config), "--hawor-device", hawor_device))
    if not hand_joints:
        command.append("--no-hand-joints")
    for band in bands:
        command.extend(("--band", band))
    if world_board:
        command.extend(
            (
                "--world-board",
                world_board,
                "--head-slam",
                "--graph-diagnostics",
                "--slam-debug-video",
                "--open-replay",
            )
        )
    elif auto_marker_map:
        command.extend(
            (
                "--auto-marker-map",
                "--static-marker-ids",
                static_marker_ids,
                "--static-marker-size-mm",
                str(static_marker_size_mm),
                "--graph-diagnostics",
                "--slam-debug-video",
                "--open-replay",
            )
        )
    return command


def process_recordings(paths: list[Path], args) -> None:
    if not paths or not args.auto_process:
        return
    if not args.band:
        print("automatic offline processing skipped: at least one --band is required")
        return
    for path in paths:
        print(f"starting offline full-frame processing: {path}", flush=True)
        completed = subprocess.run(
            offline_processing_command(
                path,
                args.calib,
                args.band,
                args.world_board,
                args.hand_model,
                args.auto_marker_map,
                args.static_marker_ids,
                args.static_marker_size_mm,
                args.hand_joints,
                args.offline_hand_backend,
                args.hawor_config,
                args.hawor_device,
            )
        )
        if completed.returncode != 0:
            print(
                f"offline processing failed for {path} (exit {completed.returncode}); "
                "the raw video is preserved"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live ArUco/hand preview with independent raw-video capture"
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--camera-fps", type=float, default=60.0)
    add_camera_control_arguments(parser)
    parser.add_argument("--calib", default="calib/camera_1920x1080.json")
    parser.add_argument(
        "--band", action="append", default=[], help="band layout JSON; repeat for both hands"
    )
    parser.add_argument("--world-board", help="fixed world-reference board layout JSON")
    parser.add_argument(
        "--auto-marker-map",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "after recording, treat static markers as independent 6DoF landmarks "
            "(default: enabled unless --world-board is supplied)"
        ),
    )
    parser.add_argument(
        "--static-marker-ids",
        default="20-49",
        help="fixed marker IDs for offline auto mapping (default: 20-49)",
    )
    parser.add_argument(
        "--static-marker-size-mm",
        type=float,
        default=48.0,
        help="printed fixed-marker side length (default: 48)",
    )
    parser.add_argument("--single", action="append", default=[], metavar="ID:SIZE_MM")
    parser.add_argument(
        "--show-marker-axes",
        action="store_true",
        help="also solve and draw a local coordinate frame for every marker in each band",
    )
    parser.add_argument("--sharpen", action="store_true")
    parser.add_argument(
        "--no-board-refine",
        action="store_true",
        help="disable constellation-assisted recovery of rejected marker candidates",
    )
    parser.add_argument(
        "--no-corner-tracking",
        action="store_true",
        help="disable optical-flow tracking across short marker detection gaps",
    )
    parser.add_argument(
        "--fixed-smoothing",
        action="store_true",
        help="use the previous fixed-gain pose smoother for comparison",
    )
    parser.add_argument(
        "--hand-joints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="detect and draw all 21 landmarks for both hands (default: enabled)",
    )
    parser.add_argument(
        "--hand-model",
        default="models/hand_landmarker.task",
        help="MediaPipe Hand Landmarker model bundle",
    )
    parser.add_argument(
        "--offline-hand-backend", choices=("hawor", "mediapipe"), default="hawor",
        help="offline hand reconstruction after recording (default: hawor); live preview stays MediaPipe",
    )
    parser.add_argument("--hawor-config", type=Path, default=DEFAULT_HAWOR_CONFIG)
    parser.add_argument("--hawor-device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--auto-process",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="after Q/Esc, process every recording offline at full source FPS (default: enabled)",
    )
    parser.add_argument(
        "--capture-qc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="show lightweight blur/exposure/tag-size checks (default: enabled)",
    )
    args = parser.parse_args()
    if args.world_board and args.auto_marker_map is True:
        raise SystemExit("--world-board and --auto-marker-map are mutually exclusive")
    if args.auto_marker_map is None:
        args.auto_marker_map = args.world_board is None

    calibration = Calibration.load(args.calib)
    if calibration.image_size != (args.width, args.height):
        try:
            calibration = calibration.scaled_to((args.width, args.height))
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    bands = [BandLayout.load(path) for path in args.band]
    world_board = BandLayout.load(args.world_board) if args.world_board else None
    pipeline = TrackingPipeline(
        calibration,
        bands,
        dict(parse_single(value) for value in args.single),
        args.sharpen,
        args.show_marker_axes,
        refine_markers=not args.no_board_refine,
        track_marker_gaps=0 if args.no_corner_tracking else 2,
        adaptive_smoothing=not args.fixed_smoothing,
        world_board=world_board,
    )
    trajectory = FadingTrajectory(
        [band.name for band in bands],
        world_only=world_board is not None,
    )

    capture = open_camera(args.camera, args.width, args.height, args.camera_fps)
    reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
    recording_fps = reported_fps if reported_fps > 1.0 else args.camera_fps
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
            "bands": [str(Path(path).resolve()) for path in args.band],
            "world_board": (
                str(Path(args.world_board).resolve()) if args.world_board else None
            ),
            "requested_capture": {
                "width": args.width,
                "height": args.height,
                "fps": args.camera_fps,
            },
        },
    )
    qc_summary = CaptureQualitySummary()
    latest_quality = None
    next_qc_s = 0.0
    frame_capture = LatestFrameCapture(capture, recorder)
    hand_tracker = None
    fps_started = time.monotonic()
    fps_frames = 0
    displayed_fps = 0.0
    first_timestamp_s = None
    last_timestamp_ms = -1
    last_sequence = -1
    recordings_to_process: list[Path] = []
    try:
        if args.hand_joints:
            hand_tracker = HandJointTracker(
                args.hand_model, calibration, [band.name for band in bands]
            )
        frame_capture.start()
        while True:
            captured = frame_capture.read_latest(last_sequence)
            if captured is None:
                break
            last_sequence, timestamp_s, frame = captured
            if first_timestamp_s is None:
                first_timestamp_s = timestamp_s
            result = pipeline.process(frame)
            if args.capture_qc and timestamp_s >= next_qc_s:
                latest_quality = evaluate_capture_quality(frame, result.detections)
                if recorder.active:
                    qc_summary.add(latest_quality)
                next_qc_s = timestamp_s + 0.25
            hand_joints = None
            if hand_tracker is not None:
                timestamp_ms = max(
                    last_timestamp_ms + 1,
                    round((timestamp_s - first_timestamp_s) * 1000.0),
                )
                last_timestamp_ms = timestamp_ms
                hand_joints = hand_tracker.process(
                    frame,
                    timestamp_ms,
                    result.raw_poses,
                    result.world_reference,
                )
            output = draw_result(frame, result, calibration, trajectory, hand_joints)

            fps_frames += 1
            now = time.monotonic()
            elapsed = now - fps_started
            if elapsed >= 0.5:
                displayed_fps = fps_frames / elapsed
                fps_started = now
                fps_frames = 0
            cv2.putText(
                output,
                f"PREVIEW {displayed_fps:.1f} FPS   CAMERA {frame_capture.measured_fps:.1f} FPS"
                "   R: raw AVI   Q/Esc: quit + process",
                (16, args.height - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                output,
                camera_control_hud,
                (16, args.height - 46),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (80, 230, 80) if camera_controls["mode"] == "locked" else (0, 220, 255),
                2,
                cv2.LINE_AA,
            )
            if latest_quality is not None:
                cv2.putText(
                    output,
                    latest_quality.hud(),
                    (16, args.height - 74),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (80, 230, 80) if latest_quality.ready else (0, 180, 255),
                    2,
                    cv2.LINE_AA,
                )
            if recorder.active:
                cv2.circle(output, (args.width - 165, 28), 8, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.putText(
                    output,
                    f"RAW REC {recorder.frames}",
                    (args.width - 145, 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            cv2.imshow("hand tracking preview - R records raw video", output)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                if recorder.active:
                    recorder.metadata["capture_qc"] = qc_summary.to_dict()
                    saved = recorder.stop()
                    if saved is not None:
                        print_recording(saved)
                        recordings_to_process.append(saved.path)
                else:
                    qc_summary.clear()
                    next_qc_s = 0.0
                    print(f"recording untouched camera frames to {recorder.start()}")
    finally:
        frame_capture.stop()
        recorder.metadata["capture_qc"] = qc_summary.to_dict()
        saved = recorder.stop()
        if saved is not None:
            print_recording(saved)
            recordings_to_process.append(saved.path)
        if hand_tracker is not None:
            hand_tracker.close()
        cv2.destroyAllWindows()
    process_recordings(recordings_to_process, args)


if __name__ == "__main__":
    main()
