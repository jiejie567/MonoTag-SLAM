#!/usr/bin/env python3
"""Compare exported camera poses with Odin ROS 2 odometry.

The bag publishes the Odin IMU pose but no camera-to-IMU transform.  This tool
therefore estimates one constant SE(3) transform on the first part of the
overlap and reports accuracy on the held-out tail.  It never estimates a
Sim(3), so metric scale errors remain visible.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp


def pose_matrix(translation: list[float], quaternion_wxyz: list[float]) -> np.ndarray:
    transform = np.eye(4)
    w, x, y, z = quaternion_wxyz
    transform[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    transform[:3, 3] = translation
    return transform


def parameters_to_transform(parameters: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_rotvec(parameters[:3]).as_matrix()
    transform[:3, 3] = parameters[3:6]
    return transform


def rotation_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    delta = first[:3, :3].T @ second[:3, :3]
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(delta).as_rotvec())))


class OdinTrajectory:
    def __init__(self, csv_path: Path):
        rows = list(csv.DictReader(csv_path.open()))
        self.times = np.asarray([float(row["timestamp_s"]) for row in rows])
        self.positions = np.asarray([
            [float(row["x_m"]), float(row["y_m"]), float(row["z_m"])]
            for row in rows
        ])
        self.rotations = Rotation.from_quat(np.asarray([
            [float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])]
            for row in rows
        ]))
        covariance = np.asarray([
            [float(row[f"pose_covariance_{index}"]) for index in range(36)]
            for row in rows
        ]).reshape(-1, 6, 6)
        diagonal = np.maximum(np.diagonal(covariance, axis1=1, axis2=2), 0.0)
        self.median_position_sigma_m = np.median(np.sqrt(diagonal[:, :3]), axis=0)
        self.median_rotation_sigma_deg = np.degrees(
            np.median(np.sqrt(diagonal[:, 3:]), axis=0)
        )
        self.slerp = Slerp(self.times, self.rotations)

    def sample(self, times: np.ndarray) -> list[np.ndarray]:
        clipped = np.clip(times, self.times[0], self.times[-1])
        positions = np.column_stack([
            np.interp(clipped, self.times, self.positions[:, axis])
            for axis in range(3)
        ])
        rotations = self.slerp(clipped).as_matrix()
        output = []
        for position, rotation in zip(positions, rotations):
            transform = np.eye(4)
            transform[:3, :3] = rotation
            transform[:3, 3] = position
            output.append(transform)
        return output


def residuals(
    parameters: np.ndarray,
    estimate_poses: list[np.ndarray],
    odin: OdinTrajectory,
    times: np.ndarray,
    estimate_extrinsic: bool,
) -> np.ndarray:
    world_alignment = parameters_to_transform(parameters[:6])
    if estimate_extrinsic:
        imu_from_camera = parameters_to_transform(parameters[6:12])
        time_offset = parameters[12]
    else:
        imu_from_camera = parameters_to_transform(parameters[6:12])
        time_offset = parameters[12]
    reference_poses = odin.sample(times + time_offset)
    values: list[float] = []
    for estimate, reference in zip(estimate_poses, reference_poses):
        predicted = world_alignment @ reference @ imu_from_camera
        values.extend((predicted[:3, 3] - estimate[:3, 3]) / 0.02)
        rotation_delta = predicted[:3, :3].T @ estimate[:3, :3]
        values.extend(Rotation.from_matrix(rotation_delta).as_rotvec() / np.radians(2.0))
    return np.asarray(values)


def fit_alignment(
    estimate_poses: list[np.ndarray],
    odin: OdinTrajectory,
    times: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    first_reference = odin.sample(times[:1])[0]
    initial_alignment = np.zeros(6)
    initial_alignment[:3] = Rotation.from_matrix(
        estimate_poses[0][:3, :3] @ first_reference[:3, :3].T
    ).as_rotvec()
    initial_alignment[3:6] = estimate_poses[0][:3, 3] - (
        parameters_to_transform(initial_alignment) @ first_reference
    )[:3, 3]
    initial = np.r_[initial_alignment, np.zeros(6), 0.0]
    lower = np.r_[np.full(3, -4 * np.pi), np.full(3, -10.0),
                  np.full(3, -4 * np.pi), np.full(3, -0.35), -0.15]
    upper = np.r_[np.full(3, 4 * np.pi), np.full(3, 10.0),
                  np.full(3, 4 * np.pi), np.full(3, 0.35), 0.15]
    result = least_squares(
        residuals,
        initial,
        args=(estimate_poses, odin, times, True),
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=600,
    )
    singular_values = np.linalg.svd(result.jac, compute_uv=False)
    diagnostics = {
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "iterations": int(result.nfev),
        "jacobian_condition": float(singular_values[0] / max(singular_values[-1], 1e-12)),
    }
    return result.x, diagnostics


def fit_world_only(
    estimate_poses: list[np.ndarray],
    odin: OdinTrajectory,
    times: np.ndarray,
    shared_parameters: np.ndarray,
) -> np.ndarray:
    initial = shared_parameters.copy()
    frozen = initial[6:].copy()

    def world_residual(world_parameters: np.ndarray) -> np.ndarray:
        parameters = np.r_[world_parameters, frozen]
        return residuals(parameters, estimate_poses, odin, times, False)

    result = least_squares(
        world_residual,
        initial[:6],
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=300,
    )
    initial[:6] = result.x
    return initial


def aligned_reference(
    parameters: np.ndarray, odin: OdinTrajectory, times: np.ndarray
) -> list[np.ndarray]:
    world_alignment = parameters_to_transform(parameters[:6])
    imu_from_camera = parameters_to_transform(parameters[6:12])
    return [
        world_alignment @ pose @ imu_from_camera
        for pose in odin.sample(times + parameters[12])
    ]


def metric_summary(
    estimates: list[np.ndarray], references: list[np.ndarray], times: np.ndarray
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    translation_errors = np.asarray([
        np.linalg.norm(estimate[:3, 3] - reference[:3, 3])
        for estimate, reference in zip(estimates, references)
    ])
    rotation_errors = np.asarray([
        rotation_error_deg(reference, estimate)
        for estimate, reference in zip(estimates, references)
    ])
    rpe_translation: list[float] = []
    rpe_rotation: list[float] = []
    scale_ratios: list[float] = []
    for first in range(len(times)):
        second = int(np.searchsorted(times, times[first] + 1.0))
        if second >= len(times):
            break
        reference_delta = np.linalg.inv(references[first]) @ references[second]
        estimate_delta = np.linalg.inv(estimates[first]) @ estimates[second]
        error = np.linalg.inv(reference_delta) @ estimate_delta
        rpe_translation.append(float(np.linalg.norm(error[:3, 3])))
        rpe_rotation.append(rotation_error_deg(np.eye(4), error))
        reference_distance = float(np.linalg.norm(
            references[second][:3, 3] - references[first][:3, 3]
        ))
        estimate_distance = float(np.linalg.norm(
            estimates[second][:3, 3] - estimates[first][:3, 3]
        ))
        if reference_distance >= 0.01:
            scale_ratios.append(estimate_distance / reference_distance)

    def stats(values: np.ndarray, prefix: str) -> dict[str, float]:
        if not len(values):
            return {}
        return {
            f"{prefix}_rmse": float(np.sqrt(np.mean(values ** 2))),
            f"{prefix}_median": float(np.median(values)),
            f"{prefix}_p95": float(np.percentile(values, 95)),
        }

    summary = {
        **stats(translation_errors, "ate_translation_m"),
        **stats(rotation_errors, "ate_rotation_deg"),
        **stats(np.asarray(rpe_translation), "rpe_1s_translation_m"),
        **stats(np.asarray(rpe_rotation), "rpe_1s_rotation_deg"),
        "scale_ratio_1s_median": float(np.median(scale_ratios)) if scale_ratios else float("nan"),
        "samples": len(estimates),
        "duration_s": float(times[-1] - times[0]) if len(times) > 1 else 0.0,
    }
    return summary, translation_errors, rotation_errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("actions", type=Path)
    parser.add_argument("image_timestamps", type=Path)
    parser.add_argument("odometry", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--calibration-fraction", type=float, default=0.55)
    args = parser.parse_args()

    image_rows = list(csv.DictReader(args.image_timestamps.open()))
    image_times = {int(row["frame"]): float(row["timestamp_s"]) for row in image_rows}
    actions = [json.loads(line) for line in args.actions.open()]
    fused_records = [row for row in actions if row.get("camera_world_pose_fused")]
    if len(fused_records) < 20:
        raise RuntimeError("too few valid fused camera poses")
    frames = np.asarray([int(row["frame"]) for row in fused_records])
    times = np.asarray([image_times[frame] for frame in frames])
    fused_poses = [
        pose_matrix(row["camera_world_pose_fused"]["translation_m"],
                    row["camera_world_pose_fused"]["quaternion_wxyz"])
        for row in fused_records
    ]
    split = max(12, min(len(fused_records) - 8,
                        int(round(len(fused_records) * args.calibration_fraction))))
    odin = OdinTrajectory(args.odometry)
    parameters, fit_diagnostics = fit_alignment(fused_poses[:split], odin, times[:split])
    reference_poses = aligned_reference(parameters, odin, times)
    all_metrics, all_translation, all_rotation = metric_summary(
        fused_poses, reference_poses, times
    )
    heldout_metrics, heldout_translation, heldout_rotation = metric_summary(
        fused_poses[split:], reference_poses[split:], times[split:]
    )

    marker_records = [row for row in actions if row.get("marker_camera_pose_observed")]
    marker_frames = np.asarray([int(row["frame"]) for row in marker_records])
    marker_times = np.asarray([image_times[frame] for frame in marker_frames])
    marker_poses = [
        pose_matrix(row["marker_camera_pose_observed"]["translation_m"],
                    row["marker_camera_pose_observed"]["quaternion_wxyz"])
        for row in marker_records
    ]
    marker_calibration_count = int(np.searchsorted(marker_times, times[split - 1], side="right"))
    marker_parameters = fit_world_only(
        marker_poses[:marker_calibration_count], odin,
        marker_times[:marker_calibration_count], parameters
    )
    marker_references = aligned_reference(marker_parameters, odin, marker_times)
    marker_test_start = marker_calibration_count
    marker_metrics, _, _ = metric_summary(
        marker_poses[marker_test_start:], marker_references[marker_test_start:],
        marker_times[marker_test_start:]
    )

    result = {
        "schema": "odin-marker-orb-accuracy/v1",
        "reference": {
            "name": "Odin1 odometry",
            "frame_id": "odom",
            "child_frame_id": "imu",
            "camera_imu_extrinsic_in_bag": False,
            "alignment": "SE(3) world alignment plus one constant camera-IMU SE(3); no Sim(3)",
            "time_offset_s": float(parameters[12]),
            "reported_median_position_sigma_m_xyz": odin.median_position_sigma_m.tolist(),
            "reported_median_rotation_sigma_deg_xyz": odin.median_rotation_sigma_deg.tolist(),
            "estimated_imu_from_camera_translation_m": parameters[9:12].tolist(),
            "estimated_imu_from_camera_rotation_quaternion_xyzw": Rotation.from_rotvec(
                parameters[6:9]
            ).as_quat().tolist(),
            "fit_diagnostics": fit_diagnostics,
        },
        "coverage": {
            "total_frames": len(actions),
            "valid_fused_frames": len(fused_records),
            "overall_fraction": len(fused_records) / len(actions),
            "first_valid_frame": int(frames[0]),
            "first_valid_time_s": float(fused_records[0]["timestamp_s"]),
            "valid_after_first_fraction": len(fused_records) / (len(actions) - int(frames[0])),
            "invalid_after_first_frames": int(sum(
                not row.get("camera_world_pose_fused") for row in actions[int(frames[0]):]
            )),
        },
        "split": {
            "calibration_samples": split,
            "heldout_samples": len(fused_records) - split,
            "calibration_end_frame": int(frames[split - 1]),
            "heldout_start_frame": int(frames[split]),
        },
        "fused_all_overlap": all_metrics,
        "fused_heldout": heldout_metrics,
        "marker_only_heldout": marker_metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")

    samples_path = args.output.with_name(args.output.stem + "_samples.csv")
    with samples_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "frame", "video_time_s", "ros_time_s", "source", "confidence",
            "estimate_x_m", "estimate_y_m", "estimate_z_m",
            "reference_x_m", "reference_y_m", "reference_z_m",
            "translation_error_m", "rotation_error_deg", "heldout",
        ])
        for index, (record, estimate, reference) in enumerate(
            zip(fused_records, fused_poses, reference_poses)
        ):
            writer.writerow([
                record["frame"], record["timestamp_s"], times[index],
                record.get("camera_world_source"), record.get("camera_world_confidence"),
                *estimate[:3, 3], *reference[:3, 3],
                all_translation[index], all_rotation[index], int(index >= split),
            ])
    print(json.dumps(result, indent=2))
    print(f"wrote {args.output}")
    print(f"wrote {samples_path}")


if __name__ == "__main__":
    main()
