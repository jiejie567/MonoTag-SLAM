#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import cv2

from aruco_track.models import BandLayout, Calibration, Pose
from aruco_track.pipeline import TrackingPipeline
from aruco_track.tracks import _matrix_to_quaternion


POSE_FIELDS = ("tx_m", "ty_m", "tz_m", "qw", "qx", "qy", "qz")


def pose_values(pose: Pose | None) -> list[float | str]:
    if pose is None:
        return [""] * len(POSE_FIELDS)
    translation = pose.tvec.reshape(3)
    quaternion = _matrix_to_quaternion(pose.rotation_matrix)
    return [*map(float, translation), *map(float, quaternion)]


def add_pose(row: dict[str, object], prefix: str, pose: Pose | None) -> None:
    row.update(
        {f"{prefix}_{field}": value for field, value in zip(POSE_FIELDS, pose_values(pose))}
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export hand and moving-camera trajectories in a fixed ArUco-board frame"
    )
    parser.add_argument("video")
    parser.add_argument("--calib", default="calib/camera_1920x1080.json")
    parser.add_argument("--world-board", required=True)
    parser.add_argument("--band", action="append", default=[], help="hand band JSON; repeat")
    parser.add_argument("--output", help="output CSV path")
    parser.add_argument("--no-board-refine", action="store_true")
    parser.add_argument("--no-corner-tracking", action="store_true")
    parser.add_argument("--fixed-smoothing", action="store_true")
    args = parser.parse_args()
    if not args.band:
        raise SystemExit("at least one --band is required")

    video_path = Path(args.video)
    output_path = Path(args.output) if args.output else video_path.with_name(
        f"{video_path.stem}_world_trajectory.csv"
    )
    metadata_path = output_path.with_suffix(".meta.json")
    calibration = Calibration.load(args.calib)
    world_board = BandLayout.load(args.world_board)
    bands = [BandLayout.load(path) for path in args.band]
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"cannot open {video_path}")
    video_size = (
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    if video_size != calibration.image_size:
        try:
            calibration = calibration.scaled_to(video_size)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    pipeline = TrackingPipeline(
        calibration,
        bands,
        refine_markers=not args.no_board_refine,
        track_marker_gaps=0 if args.no_corner_tracking else 2,
        adaptive_smoothing=not args.fixed_smoothing,
        world_board=world_board,
    )

    fixed_fields = [
        "frame",
        "timestamp_s",
        "hand",
        "selected_coordinate_frame",
        "trajectory_segment",
        "hand_camera_valid",
        "world_reference_valid",
        "world_hand_valid",
        "hand_marker_ids",
        "world_marker_ids",
        "hand_reprojection_error_px",
        "world_reprojection_error_px",
        "hand_ambiguous",
        "world_ambiguous",
        "hand_uses_recovered_marker",
        "world_uses_recovered_marker",
        "hand_uses_optical_flow_marker",
        "world_uses_optical_flow_marker",
        "recovered_ids",
        "tracked_ids",
    ]
    pose_fields = [
        f"{prefix}_{field}"
        for prefix in (
            "camera_hand_raw",
            "camera_hand_filtered",
            "world_hand_raw",
            "world_hand_filtered",
            "selected_hand_filtered",
            "world_camera_raw",
        )
        for field in POSE_FIELDS
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = 0
    reference_frames = 0
    camera_frames: Counter[str] = Counter()
    world_frames: Counter[str] = Counter()
    selected_frames: dict[str, str | None] = {band.name: None for band in bands}
    trajectory_segments: Counter[str] = Counter()
    with output_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fixed_fields + pose_fields)
        writer.writeheader()
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_count += 1
            result = pipeline.process(frame)
            reference = result.world_reference
            reference_frames += int(reference is not None)
            recovered = set(result.recovered_ids)
            tracked = set(result.tracked_ids)
            for band in bands:
                camera_pose = result.raw_poses.get(band.name)
                filtered_camera_pose = result.poses.get(band.name)
                raw_world_pose = result.raw_world_poses.get(band.name)
                filtered_world_pose = result.world_poses.get(band.name)
                if filtered_world_pose is not None:
                    selected_frame = "world"
                    selected_pose = filtered_world_pose
                elif filtered_camera_pose is not None:
                    selected_frame = "camera"
                    selected_pose = filtered_camera_pose
                else:
                    selected_frame = "invalid"
                    selected_pose = None
                previous_frame = selected_frames[band.name]
                if selected_frame == "invalid":
                    selected_frames[band.name] = None
                elif selected_frame != previous_frame:
                    trajectory_segments[band.name] += 1
                    selected_frames[band.name] = selected_frame
                camera_frames[band.name] += int(camera_pose is not None)
                world_frames[band.name] += int(filtered_world_pose is not None)
                hand_ids = set(camera_pose.marker_ids) if camera_pose is not None else set()
                world_ids = set(reference.marker_ids) if reference is not None else set()
                row: dict[str, object] = {
                    "frame": frame_count,
                    "timestamp_s": (frame_count - 1) / fps,
                    "hand": band.name,
                    "selected_coordinate_frame": selected_frame,
                    "trajectory_segment": (
                        trajectory_segments[band.name] if selected_pose is not None else ""
                    ),
                    "hand_camera_valid": int(camera_pose is not None),
                    "world_reference_valid": int(reference is not None),
                    "world_hand_valid": int(filtered_world_pose is not None),
                    "hand_marker_ids": ";".join(map(str, camera_pose.marker_ids)) if camera_pose else "",
                    "world_marker_ids": ";".join(map(str, reference.marker_ids)) if reference else "",
                    "hand_reprojection_error_px": camera_pose.reprojection_error_px if camera_pose else "",
                    "world_reprojection_error_px": reference.reprojection_error_px if reference else "",
                    "hand_ambiguous": int(camera_pose.ambiguous) if camera_pose else "",
                    "world_ambiguous": int(reference.ambiguous) if reference else "",
                    "hand_uses_recovered_marker": int(bool(hand_ids & recovered)),
                    "world_uses_recovered_marker": int(bool(world_ids & recovered)),
                    "hand_uses_optical_flow_marker": int(bool(hand_ids & tracked)),
                    "world_uses_optical_flow_marker": int(bool(world_ids & tracked)),
                    "recovered_ids": ";".join(map(str, sorted(recovered))),
                    "tracked_ids": ";".join(map(str, sorted(tracked))),
                }
                add_pose(row, "camera_hand_raw", camera_pose)
                add_pose(row, "camera_hand_filtered", filtered_camera_pose)
                add_pose(row, "world_hand_raw", raw_world_pose)
                add_pose(row, "world_hand_filtered", filtered_world_pose)
                add_pose(row, "selected_hand_filtered", selected_pose)
                add_pose(row, "world_camera_raw", result.camera_world_pose)
                writer.writerow(row)
    capture.release()

    metadata = {
        "video": str(video_path.resolve()),
        "calibration": str(Path(args.calib).resolve()),
        "world_board": str(Path(args.world_board).resolve()),
        "bands": [str(Path(path).resolve()) for path in args.band],
        "frames": frame_count,
        "fps": fps,
        "image_size": list(video_size),
        "world_reference_frames": reference_frames,
        "camera_pose_frames": dict(camera_frames),
        "world_hand_pose_frames": dict(world_frames),
        "world_frame": world_board.name,
        "policy": {
            "corner_tracking_max_gap_frames": 0 if args.no_corner_tracking else 2,
            "long_world_reference_gaps": "invalid; not interpolated",
            "world_hand_filtered": not args.fixed_smoothing,
            "selected_hand_filtered": (
                "world frame when world reference is valid; otherwise camera frame; "
                "trajectory_segment changes at every frame switch or invalid gap"
            ),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {frame_count * len(bands)} rows to {output_path}")
    print(f"wrote metadata to {metadata_path}")
    print(f"world reference valid: {reference_frames}/{frame_count}")
    for band in bands:
        print(
            f"{band.name}: camera={camera_frames[band.name]}/{frame_count}, "
            f"world={world_frames[band.name]}/{frame_count}"
        )


if __name__ == "__main__":
    main()
