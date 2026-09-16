#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter

import cv2

from aruco_track.hands import HandJointTracker
from aruco_track.models import BandLayout, Calibration
from aruco_track.pipeline import TrackingPipeline
from aruco_track.render import FadingTrajectory, draw_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a recording and report pose detection rates")
    parser.add_argument("video")
    parser.add_argument("--calib", default="calib/camera_1920x1080.json")
    parser.add_argument("--band", action="append", default=[])
    parser.add_argument("--world-board", help="fixed world-reference board layout JSON")
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--hand-joints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="detect and draw all 21 landmarks for both hands (default: enabled)",
    )
    parser.add_argument("--hand-model", default="models/hand_landmarker.task")
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
    args = parser.parse_args()
    calibration = Calibration.load(args.calib)
    bands = [BandLayout.load(path) for path in args.band]
    world_board = BandLayout.load(args.world_board) if args.world_board else None
    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    video_size = (
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    if video_size != calibration.image_size:
        try:
            calibration = calibration.scaled_to(video_size)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        print(f"scaled calibration to replay resolution {video_size}")
    pipeline = TrackingPipeline(
        calibration,
        bands,
        refine_markers=not args.no_board_refine,
        track_marker_gaps=0 if args.no_corner_tracking else 2,
        adaptive_smoothing=not args.fixed_smoothing,
        world_board=world_board,
    )
    trajectory = FadingTrajectory([band.name for band in bands])
    hand_tracker = (
        HandJointTracker(args.hand_model, calibration, [band.name for band in bands])
        if args.hand_joints
        else None
    )
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    frames = 0
    pose_frames: Counter[str] = Counter()
    ambiguous: Counter[str] = Counter()
    recovered: Counter[int] = Counter()
    tracked: Counter[int] = Counter()
    errors: dict[str, list[float]] = {band.name: [] for band in bands}
    world_reference_frames = 0
    world_pose_frames: Counter[str] = Counter()
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames += 1
        result = pipeline.process(frame)
        hand_joints = (
            hand_tracker.process(
                frame,
                round((frames - 1) * 1000.0 / fps),
                result.raw_poses,
                result.world_reference,
            )
            if hand_tracker is not None
            else None
        )
        world_reference_frames += int(result.world_reference is not None)
        world_pose_frames.update(result.world_poses.keys())
        recovered.update(result.recovered_ids)
        tracked.update(result.tracked_ids)
        for name, pose in result.poses.items():
            pose_frames[name] += 1
            ambiguous[name] += int(pose.ambiguous)
            errors.setdefault(name, []).append(pose.reprojection_error_px)
        if args.show:
            cv2.imshow(
                "replay - q to stop",
                draw_result(frame, result, calibration, trajectory, hand_joints),
            )
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    capture.release()
    if hand_tracker is not None:
        hand_tracker.close()
    cv2.destroyAllWindows()
    print(f"frames: {frames}")
    if world_board is not None:
        rate = 100.0 * world_reference_frames / frames if frames else 0.0
        print(f"world reference: {world_reference_frames}/{frames} ({rate:.1f}%)")
    print(
        "recovered marker observations: "
        + (", ".join(f"ID {marker_id}={count}" for marker_id, count in sorted(recovered.items())) or "0")
    )
    print(
        "optical-flow marker observations: "
        + (", ".join(f"ID {marker_id}={count}" for marker_id, count in sorted(tracked.items())) or "0")
    )
    for band in bands:
        count = pose_frames[band.name]
        rate = 100.0 * count / frames if frames else 0.0
        mean_error = sum(errors[band.name]) / count if count else float("nan")
        print(f"{band.name}: {count}/{frames} ({rate:.1f}%), ambiguous={ambiguous[band.name]}, mean_error={mean_error:.2f}px")
        if world_board is not None:
            world_count = world_pose_frames[band.name]
            world_rate = 100.0 * world_count / frames if frames else 0.0
            print(f"  world trajectory: {world_count}/{frames} ({world_rate:.1f}%)")


if __name__ == "__main__":
    main()
