"""Candidate-only marker evidence. Existing valid/partial hints take precedence.

A candidate is not a valid world pose: native multi-view admission still applies.
The validated profile merges this stream only into previously invalid hints.
"""
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from .models import Pose, BandLayout, Calibration
from .marker_corners import TrackedMarkerObservation
from .orbslam3_backend import MIN_MARKER_CONFIDENCE, MAX_MARKER_ERROR_PX

def write_candidate_hints(
    path: Path,
    marker_poses: list[Pose | None],
    marker_confidences: list[float],
    detections: list[dict[int, np.ndarray]],
    accepted_marker_ids: list[tuple[int, ...]],
    layout: BandLayout,
    calibration: Calibration,
    fps: float,
    start_frame: int,
    marker_weights: list[dict[int, float]] | None = None,
    include_ids: bool = False,
    tracked_observations: list[TrackedMarkerObservation] | None = None,
    marker_layouts: dict[str, BandLayout] | None = None,
    marker_component_ids: list[str | None] | None = None,
) -> None:
    if not (
        len(marker_poses)
        == len(marker_confidences)
        == len(detections)
        == len(accepted_marker_ids)
    ):
        raise ValueError("tag observation streams must have equal length")
    if marker_weights is not None and len(marker_weights) != len(detections):
        raise ValueError("tag weight and detection streams must have equal length")
    if tracked_observations is not None and len(tracked_observations) != len(detections):
        raise ValueError("tracked corner stream must have equal length")
    if marker_component_ids is not None and len(marker_component_ids) != len(detections):
        raise ValueError("marker component stream must have equal length")
    lines = ["# timestamp valid confidence Twc(tx ty tz qx qy qz qw) count X Y Z u v ..."]
    for frame_index in range(start_frame, len(marker_poses)):
        timestamp = frame_index / fps
        pose = marker_poses[frame_index]
        confidence = marker_confidences[frame_index]
        marker_ids = accepted_marker_ids[frame_index]
        component_id = (
            marker_component_ids[frame_index]
            if marker_component_ids is not None
            else None
        )
        frame_layout = (
            marker_layouts.get(component_id, layout)
            if marker_layouts is not None
            else layout
        )
        point_pairs: list[tuple[np.ndarray, np.ndarray]] = []
        point_weights: list[float] = []
        point_ids: list[int] = []
        for marker_id in marker_ids:
            if marker_id not in frame_layout.markers or marker_id not in detections[frame_index]:
                continue
            weight = 1.0 if marker_weights is None else marker_weights[frame_index].get(marker_id, 1.0)
            if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
                raise ValueError("marker information weights must be finite and in [0, 1]")
            if weight == 0.0:
                continue
            image_points = np.asarray(
                detections[frame_index][marker_id], dtype=np.float64
            ).reshape(-1, 1, 2)
            undistorted = cv2.undistortPoints(
                image_points,
                calibration.camera_matrix,
                calibration.dist_coeffs,
                P=calibration.camera_matrix,
            ).reshape(-1, 2)
            for world_point, image_point in zip(
                frame_layout.markers[marker_id], undistorted
            ):
                point_pairs.append((world_point, image_point))
                point_weights.append(weight)
                point_ids.append(marker_id)
        full_valid = (pose is not None and confidence >= MIN_MARKER_CONFIDENCE
                      and pose.reprojection_error_px <= MAX_MARKER_ERROR_PX and len(point_pairs) >= 4)
        candidate = (not full_valid and pose is not None and component_id is not None
                     and .15 <= confidence < MIN_MARKER_CONFIDENCE
                     and pose.reprojection_error_px <= MAX_MARKER_ERROR_PX
                     and len(point_pairs)==4 and len(set(point_ids))==1
                     and all(w >= .99 for w in point_weights))
        tracked = tracked_observations[frame_index] if tracked_observations and not candidate else None
        partial_only, tracked_count, track_age = False, 0, 0.
        if not full_valid and not candidate:
            point_pairs, point_weights, point_ids = [], [], []
            if tracked and tracked.partial_only and tracked.pose is not None:
                pose, confidence, partial_only = tracked.pose, tracked.confidence, True
        if tracked and tracked.pose is not None and (full_valid or partial_only):
            pixels = cv2.undistortPoints(np.asarray(tracked.image_points, float).reshape(-1,1,2),
                calibration.camera_matrix, calibration.dist_coeffs, P=calibration.camera_matrix).reshape(-1,2)
            tracked_weights = (
                tracked.point_weights
                if len(tracked.point_weights) == len(pixels)
                else [.25] * len(pixels)
            )
            for world, pixel, mid, weight in zip(
                tracked.world_points, pixels, tracked.marker_ids, tracked_weights
            ):
                point_pairs.append((world, pixel))
                point_weights.append(float(weight))
                point_ids.append(mid)
            tracked_count, track_age = len(pixels), tracked.age_s
        if not full_valid and not candidate and not (partial_only and len(point_pairs)>=3 and confidence>=.15 and track_age<=.30):
            lines.append(f"{timestamp:.9f} 0")
            continue
        quaternion = Rotation.from_matrix(pose.rotation_matrix).as_quat()
        translation = pose.tvec.reshape(3)
        values = [
            f"{timestamp:.9f}",
            "1",
            f"{confidence:.9g}",
            *(f"{value:.9g}" for value in translation),
            *(f"{value:.9g}" for value in quaternion),
            str(len(point_pairs)),
        ]
        for world_point, image_point in point_pairs:
            values.extend(f"{value:.9g}" for value in world_point)
            values.extend(f"{value:.9g}" for value in image_point)
        if marker_weights is not None or include_ids or tracked_observations is not None:
            values.append("weights")
            values.extend(f"{weight:.9g}" for weight in point_weights)
        if include_ids or tracked_observations is not None:
            values.append("ids")
            values.extend(str(mid) for mid in point_ids)
        if tracked_observations is not None:
            values.extend(["tracked", str(tracked_count), "partial", str(int(partial_only)),
                           "age", f"{track_age:.9g}"])
        if component_id is not None:
            values.extend(["component", str(component_id)])
        if candidate: values.extend(["candidate", "1"])
        lines.append(" ".join(values))
    path.write_text("\n".join(lines) + "\n")

