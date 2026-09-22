#!/usr/bin/env python3
from __future__ import annotations

# Support direct execution from a source checkout.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from aruco_track.detector import ArucoDetector
from aruco_track.models import BandLayout, Calibration, Pose
from aruco_track.pose import solve_square_pose, square_object_points


PairFrame = tuple[np.ndarray, np.ndarray, Calibration]


def pose_matrix(pose: Pose) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = pose.rotation_matrix
    transform[:3, 3] = pose.tvec.reshape(3)
    return transform


def marker_transform(points: np.ndarray) -> np.ndarray:
    center = np.mean(points, axis=0)
    x_axis = points[1] - points[0]
    x_axis /= np.linalg.norm(x_axis)
    y_axis = points[0] - points[3]
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    transform[:3, 3] = center
    return transform


def marker_size(points: np.ndarray) -> float:
    edges = np.roll(points, -1, axis=0) - points
    return float(np.mean(np.linalg.norm(edges, axis=1)))


def relative_transform(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.linalg.inv(first) @ second


def rotation_angle_deg(rotation: np.ndarray) -> float:
    return float(np.degrees(np.linalg.norm(cv2.Rodrigues(rotation)[0])))


def robust_pair_transform(
    observations: list[np.ndarray], nominal: np.ndarray, min_frames: int
) -> tuple[np.ndarray, int] | None:
    plausible = []
    for observation in observations:
        delta = relative_transform(nominal, observation)
        if (
            np.linalg.norm(delta[:3, 3]) <= 0.03
            and rotation_angle_deg(delta[:3, :3]) <= 25.0
        ):
            plausible.append(observation)
    if len(plausible) < min_frames:
        return None

    def median_transform(values: list[np.ndarray]) -> np.ndarray:
        transform = np.eye(4, dtype=np.float64)
        transform[:3, 3] = np.median(
            np.stack([value[:3, 3] for value in values]), axis=0
        )
        residuals = np.stack(
            [
                cv2.Rodrigues(nominal[:3, :3].T @ value[:3, :3])[0].reshape(3)
                for value in values
            ]
        )
        transform[:3, :3] = nominal[:3, :3] @ cv2.Rodrigues(
            np.median(residuals, axis=0)
        )[0]
        return transform

    estimate = median_transform(plausible)
    translation_errors = np.asarray(
        [np.linalg.norm(value[:3, 3] - estimate[:3, 3]) for value in plausible]
    )
    rotation_errors = np.asarray(
        [
            rotation_angle_deg(estimate[:3, :3].T @ value[:3, :3])
            for value in plausible
        ]
    )
    translation_limit = max(
        0.003,
        float(np.median(translation_errors) + 3.0 * np.median(np.abs(translation_errors - np.median(translation_errors)))),
    )
    rotation_limit = max(
        2.0,
        float(np.median(rotation_errors) + 3.0 * np.median(np.abs(rotation_errors - np.median(rotation_errors)))),
    )
    inliers = [
        value
        for value, translation_error, rotation_error in zip(
            plausible, translation_errors, rotation_errors
        )
        if translation_error <= translation_limit and rotation_error <= rotation_limit
    ]
    if len(inliers) < min_frames:
        return None
    return median_transform(inliers), len(inliers)


def bundle_adjust_pair(
    observations: list[PairFrame],
    first_size: float,
    second_size: float,
    initial: np.ndarray,
    max_frames: int = 40,
) -> tuple[np.ndarray, float, int] | None:
    """Jointly refine one marker-to-marker transform and per-frame camera poses."""
    if not observations:
        return None
    sample_indices = np.linspace(
        0, len(observations) - 1, min(max_frames, len(observations))
    ).round().astype(int)
    sampled = [observations[index] for index in sample_indices]
    first_points = square_object_points(first_size)
    second_points = square_object_points(second_size)
    variables = [
        *cv2.Rodrigues(initial[:3, :3])[0].reshape(3),
        *initial[:3, 3],
    ]
    usable: list[PairFrame] = []
    for first_corners, second_corners, calibration in sampled:
        pose = solve_square_pose(
            first_corners,
            first_size,
            calibration.camera_matrix,
            calibration.dist_coeffs,
        )
        if pose is None:
            continue
        usable.append((first_corners, second_corners, calibration))
        variables.extend((*pose.rvec.reshape(3), *pose.tvec.reshape(3)))
    if len(usable) < 4:
        return None
    variables_array = np.asarray(variables, dtype=np.float64)

    def residuals(values: np.ndarray) -> np.ndarray:
        first_from_second_rotation = cv2.Rodrigues(values[:3])[0]
        transformed_second = (
            first_from_second_rotation @ second_points.T
        ).T + values[3:6]
        result = np.empty((len(usable), 16), dtype=np.float64)
        for index, (first_corners, second_corners, calibration) in enumerate(usable):
            frame_pose = values[6 + 6 * index : 12 + 6 * index]
            projected_first, _ = cv2.projectPoints(
                first_points,
                frame_pose[:3],
                frame_pose[3:],
                calibration.camera_matrix,
                calibration.dist_coeffs,
            )
            projected_second, _ = cv2.projectPoints(
                transformed_second,
                frame_pose[:3],
                frame_pose[3:],
                calibration.camera_matrix,
                calibration.dist_coeffs,
            )
            result[index] = np.concatenate(
                (
                    projected_first.reshape(4, 2) - first_corners,
                    projected_second.reshape(4, 2) - second_corners,
                )
            ).reshape(16)
        return result.reshape(-1)

    jacobian = lil_matrix(
        (16 * len(usable), 6 + 6 * len(usable)), dtype=np.int8
    )
    for index in range(len(usable)):
        row = 16 * index
        column = 6 + 6 * index
        jacobian[row : row + 16, :6] = 1
        jacobian[row : row + 16, column : column + 6] = 1
    optimized = least_squares(
        residuals,
        variables_array,
        jac_sparsity=jacobian.tocsr(),
        loss="soft_l1",
        f_scale=1.0,
        x_scale="jac",
        max_nfev=40,
    )
    if not optimized.success or not np.all(np.isfinite(optimized.x[:6])):
        return None
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = cv2.Rodrigues(optimized.x[:3])[0]
    transform[:3, 3] = optimized.x[3:6]
    delta = relative_transform(initial, transform)
    if (
        np.linalg.norm(delta[:3, 3]) > 0.015
        or rotation_angle_deg(delta[:3, :3]) > 15.0
    ):
        return None
    rms = float(np.sqrt(np.mean(residuals(optimized.x) ** 2)))
    return transform, rms, len(usable)


def calibrate_layout(
    layout: BandLayout,
    pair_estimates: dict[tuple[int, int], tuple[np.ndarray, int]],
) -> tuple[BandLayout, int, set[int], list[tuple[int, int, int]]]:
    nominal = {
        marker_id: marker_transform(points)
        for marker_id, points in layout.markers.items()
    }
    weighted_degree = Counter()
    for (first, second), (_, count) in pair_estimates.items():
        weighted_degree[first] += count
        weighted_degree[second] += count
    anchor = max(layout.markers, key=lambda marker_id: (weighted_degree[marker_id], -marker_id))
    transforms = {anchor: nominal[anchor]}
    tree_edges: list[tuple[int, int, int]] = []
    while True:
        candidates = []
        for pair, (transform, count) in pair_estimates.items():
            first, second = pair
            if (first in transforms) == (second in transforms):
                continue
            candidates.append((count, first, second, transform))
        if not candidates:
            break
        count, first, second, first_from_second = max(
            candidates, key=lambda candidate: candidate[0]
        )
        if first in transforms:
            transforms[second] = transforms[first] @ first_from_second
        else:
            transforms[first] = transforms[second] @ np.linalg.inv(first_from_second)
        tree_edges.append((first, second, count))

    calibrated_markers = dict(layout.markers)
    for marker_id, transform in transforms.items():
        if marker_id == anchor:
            continue
        local_points = square_object_points(marker_size(layout.markers[marker_id]))
        calibrated_markers[marker_id] = (
            transform[:3, :3] @ local_points.T
        ).T + transform[:3, 3]
    return (
        BandLayout(layout.name, layout.dictionary, calibrated_markers),
        anchor,
        set(transforms),
        tree_edges,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate a physical rigid multi-marker fixture from pairwise co-visibility"
    )
    parser.add_argument("video", nargs="+")
    layout_group = parser.add_mutually_exclusive_group(required=True)
    layout_group.add_argument("--band", help="legacy name for the nominal fixture layout")
    layout_group.add_argument("--layout", help="nominal wrist or fixed-board layout JSON")
    parser.add_argument("--calib", default="calib/camera_1920x1080.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", help="calibration diagnostics JSON (default: beside output)")
    parser.add_argument("--min-pair-frames", type=int, default=20)
    parser.add_argument("--bundle-frames", type=int, default=40)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    video_paths = [Path(value) for value in args.video]
    layout_path = Path(args.layout or args.band)
    source_data = json.loads(layout_path.read_text())
    layout = BandLayout.load(layout_path)
    calibration = Calibration.load(args.calib)
    observations: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    pair_frames: dict[tuple[int, int], list[PairFrame]] = defaultdict(list)
    marker_frames: Counter[int] = Counter()
    frames = 0
    for video_path in video_paths:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise SystemExit(f"cannot open {video_path}")
        image_size = (
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        video_calibration = calibration.scaled_to(image_size)
        detector = ArucoDetector(layout.dictionary, track_marker_gaps=0)
        previous: dict[int, tuple[Pose, int]] = {}
        video_frame = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            video_frame += 1
            frames += 1
            detections = detector.detect(frame)
            poses: dict[int, Pose] = {}
            for marker_id, points in layout.markers.items():
                if marker_id not in detections:
                    continue
                prior = previous.get(marker_id)
                pose = solve_square_pose(
                    detections[marker_id],
                    marker_size(points),
                    video_calibration.camera_matrix,
                    video_calibration.dist_coeffs,
                    prior[0]
                    if prior is not None and video_frame - prior[1] <= 5
                    else None,
                )
                if pose is None or pose.reprojection_error_px > 3.0:
                    continue
                previous[marker_id] = (pose, video_frame)
                poses[marker_id] = pose
                marker_frames[marker_id] += 1
            visible = sorted(poses)
            for first_index, first in enumerate(visible):
                for second in visible[first_index + 1 :]:
                    observations[(first, second)].append(
                        relative_transform(
                            pose_matrix(poses[first]), pose_matrix(poses[second])
                        )
                    )
                    pair_frames[(first, second)].append(
                        (
                            detections[first].copy(),
                            detections[second].copy(),
                            video_calibration,
                        )
                    )
        capture.release()

    nominal = {
        marker_id: marker_transform(points)
        for marker_id, points in layout.markers.items()
    }
    pair_estimates = {}
    for pair, values in observations.items():
        estimate = robust_pair_transform(
            values,
            relative_transform(nominal[pair[0]], nominal[pair[1]]),
            args.min_pair_frames,
        )
        if estimate is not None:
            pair_estimates[pair] = estimate
    _, _, _, initial_tree_edges = calibrate_layout(
        layout, pair_estimates
    )
    bundle_results = []
    for first, second, count in initial_tree_edges:
        result = bundle_adjust_pair(
            pair_frames[(first, second)],
            marker_size(layout.markers[first]),
            marker_size(layout.markers[second]),
            pair_estimates[(first, second)][0],
            args.bundle_frames,
        )
        if result is None:
            continue
        transform, rms, used_frames = result
        pair_estimates[(first, second)] = (transform, count)
        bundle_results.append((first, second, used_frames, rms))
    calibrated, anchor, connected, tree_edges = calibrate_layout(
        layout, pair_estimates
    )
    print(f"frames: {frames}")
    print(
        "marker coverage: "
        + ", ".join(
            f"ID {marker_id}={marker_frames[marker_id]}"
            for marker_id in sorted(layout.markers)
        )
    )
    print(
        "usable pairs: "
        + (
            ", ".join(
                f"{first}-{second}={count}"
                for (first, second), (_, count) in sorted(pair_estimates.items())
            )
            or "none"
        )
    )
    print(
        "bundle-refined tree pairs: "
        + (
            ", ".join(
                f"{first}-{second}={frames} frames/{rms:.2f}px RMS"
                for first, second, frames, rms in bundle_results
            )
            or "none"
        )
    )
    missing = set(layout.markers) - connected
    if missing and not args.allow_partial:
        raise SystemExit(
            "insufficient co-visibility: calibrated component "
            f"{sorted(connected)}, missing {sorted(missing)}; record every adjacent face pair"
        )
    output_path = Path(args.output)
    coordinate_system = source_data.get("coordinate_system")
    calibrated.save(
        args.output,
        calibrated_from_videos=[str(path.resolve()) for path in video_paths],
        nominal_layout=str(layout_path.resolve()),
        calibration_anchor_marker_id=anchor,
        calibrated_marker_ids=sorted(connected),
        partial=bool(missing),
        calibration_method="pair_bundle_adjustment",
        calibration_tree=[
            {"first": first, "second": second, "inlier_frames": count}
            for first, second, count in tree_edges
        ],
        **({"coordinate_system": coordinate_system} if coordinate_system else {}),
    )
    marker_adjustments = []
    for marker_id in sorted(connected):
        nominal_transform = marker_transform(layout.markers[marker_id])
        calibrated_transform = marker_transform(calibrated.markers[marker_id])
        delta = relative_transform(nominal_transform, calibrated_transform)
        marker_adjustments.append({
            "id": marker_id,
            "translation_mm": 1000.0 * float(np.linalg.norm(delta[:3, 3])),
            "rotation_deg": rotation_angle_deg(delta[:3, :3]),
            "side_length_mm": 1000.0 * marker_size(calibrated.markers[marker_id]),
        })
    report = {
        "schema": "rigid-marker-layout-calibration/v1",
        "nominal_layout": str(layout_path.resolve()),
        "calibrated_layout": str(output_path.resolve()),
        "videos": [str(path.resolve()) for path in video_paths],
        "camera_calibration": str(Path(args.calib).resolve()),
        "frames": frames,
        "anchor_marker_id": anchor,
        "connected_marker_ids": sorted(connected),
        "missing_marker_ids": sorted(missing),
        "marker_coverage": {str(marker_id): marker_frames[marker_id]
                            for marker_id in sorted(layout.markers)},
        "bundle_tree": [
            {"first": first, "second": second, "frames": used_frames,
             "rms_px": rms}
            for first, second, used_frames, rms in bundle_results
        ],
        "marker_adjustments": marker_adjustments,
    }
    report_path = Path(args.report) if args.report else output_path.with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"saved calibrated layout to {output_path.resolve()}")
    print(f"saved calibration report to {report_path.resolve()}")


if __name__ == "__main__":
    main()
