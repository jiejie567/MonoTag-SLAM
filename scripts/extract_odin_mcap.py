#!/usr/bin/env python3
"""Extract Odin ROS 2 MCAP image, calibration, and odometry streams."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from mcap_ros2.reader import read_ros2_messages


IMAGE_TOPIC = "/odin1/image/undistorted/compressed"
CAMERA_INFO_TOPIC = "/odin1/image/undistorted/camera_info"
ODOMETRY_TOPIC = "/odin1/odometry"


def stamp_seconds(message: object) -> float:
    stamp = message.header.stamp
    return float(stamp.sec) + 1e-9 * float(stamp.nanosec)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mcap", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frame_dir = args.output / "frames"
    frame_dir.mkdir(exist_ok=True)

    image_rows: list[tuple[int, float, str]] = []
    odometry_rows: list[list[object]] = []
    camera_info = None
    for record in read_ros2_messages(args.mcap):
        topic = record.channel.topic
        message = record.ros_msg
        if topic == IMAGE_TOPIC:
            index = len(image_rows)
            name = f"{index:06d}.jpg"
            (frame_dir / name).write_bytes(bytes(message.data))
            image_rows.append((index, stamp_seconds(message), name))
        elif topic == ODOMETRY_TOPIC:
            pose = message.pose.pose
            odometry_rows.append([
                stamp_seconds(message),
                message.header.frame_id,
                message.child_frame_id,
                pose.position.x,
                pose.position.y,
                pose.position.z,
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
                *message.pose.covariance,
            ])
        elif topic == CAMERA_INFO_TOPIC and camera_info is None:
            camera_info = message

    if not image_rows or not odometry_rows or camera_info is None:
        raise RuntimeError("MCAP is missing Odin image, camera_info, or odometry")

    with (args.output / "image_timestamps.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame", "timestamp_s", "file"])
        writer.writerows(image_rows)
    with (args.output / "odometry.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "timestamp_s", "frame_id", "child_frame_id",
            "x_m", "y_m", "z_m", "qx", "qy", "qz", "qw",
            *[f"pose_covariance_{index}" for index in range(36)],
        ])
        writer.writerows(odometry_rows)

    calibration = {
        "image_size": [int(camera_info.width), int(camera_info.height)],
        "camera_matrix": np.asarray(camera_info.k, dtype=float).reshape(3, 3).tolist(),
        "dist_coeffs": list(camera_info.d),
        "source": str(args.mcap.resolve()),
        "topic": CAMERA_INFO_TOPIC,
        "distortion_model": camera_info.distortion_model,
        "input_is_undistorted": True,
    }
    (args.output / "camera.json").write_text(
        json.dumps(calibration, indent=2) + "\n"
    )

    image_times = np.asarray([row[1] for row in image_rows])
    odometry_times = np.asarray([row[0] for row in odometry_rows], dtype=float)
    summary = {
        "mcap": str(args.mcap.resolve()),
        "image_topic": IMAGE_TOPIC,
        "odometry_topic": ODOMETRY_TOPIC,
        "image_frames": len(image_rows),
        "odometry_samples": len(odometry_rows),
        "image_duration_s": float(image_times[-1] - image_times[0]),
        "odometry_duration_s": float(odometry_times[-1] - odometry_times[0]),
        "nominal_image_fps": float(1.0 / np.median(np.diff(image_times))),
        "nominal_odometry_hz": float(1.0 / np.median(np.diff(odometry_times))),
        "image_start_s": float(image_times[0]),
        "odometry_start_s": float(odometry_times[0]),
        "first_image_minus_first_odometry_s": float(image_times[0] - odometry_times[0]),
        "odometry_frame_id": str(odometry_rows[0][1]),
        "odometry_child_frame_id": str(odometry_rows[0][2]),
        "camera_frame_id": str(camera_info.header.frame_id),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
