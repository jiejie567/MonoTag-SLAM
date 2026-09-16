from __future__ import annotations

import cv2
import numpy as np

from .models import BandLayout, Calibration, Pose


def _project(
    world_points: np.ndarray, parameters: np.ndarray, calibration: Calibration
) -> np.ndarray:
    projected, _ = cv2.projectPoints(
        world_points,
        parameters[:3].reshape(3, 1),
        parameters[3:].reshape(3, 1),
        calibration.camera_matrix,
        calibration.dist_coeffs,
    )
    return projected.reshape(-1, 2)


def marker_pose_uncertainty(
    camera_from_world: Pose | None,
    layout: BandLayout | None,
    detections: dict[int, np.ndarray],
    accepted_marker_ids: tuple[int, ...] | list[int],
    calibration: Calibration,
    marker_weights: dict[int, float] | None = None,
) -> tuple[list[dict], dict | None]:
    """Return auditable corner residuals and a local 6-DoF pose covariance.

    The covariance is the inverse weighted projection Hessian around the
    accepted marker solution.  It is a conditioning/precision estimate, not
    external absolute accuracy; a 0.25 px floor avoids claiming zero
    uncertainty for synthetic or exactly fitted corners.
    """
    if camera_from_world is None or layout is None:
        return [], None
    weights = marker_weights or {}
    world, image, point_weights, labels = [], [], [], []
    for marker_id in accepted_marker_ids:
        if marker_id not in layout.markers or marker_id not in detections:
            continue
        corners = np.asarray(detections[marker_id], dtype=np.float64).reshape(4, 2)
        geometry = np.asarray(layout.markers[marker_id], dtype=np.float64).reshape(4, 3)
        weight = float(weights.get(marker_id, 1.0))
        if not np.isfinite(weight) or weight <= 0:
            continue
        for corner_index in range(4):
            world.append(geometry[corner_index])
            image.append(corners[corner_index])
            point_weights.append(weight)
            labels.append((int(marker_id), corner_index))
    if len(world) < 4:
        return [], None
    world_array = np.asarray(world, dtype=np.float64)
    image_array = np.asarray(image, dtype=np.float64)
    parameters = np.concatenate((
        np.asarray(camera_from_world.rvec, dtype=np.float64).reshape(3),
        np.asarray(camera_from_world.tvec, dtype=np.float64).reshape(3),
    ))
    predicted = _project(world_array, parameters, calibration)
    residual_vectors = image_array - predicted
    residual_norms = np.linalg.norm(residual_vectors, axis=1)
    corner_diagnostics = [
        {
            "marker_id": marker_id,
            "corner_index": corner_index,
            "residual_px": float(residual),
            "information_weight": float(weight),
        }
        for (marker_id, corner_index), residual, weight in zip(
            labels, residual_norms, point_weights
        )
    ]

    # Central finite differences are stable here because the state is only six
    # dimensional and this runs on cached marker observations, not per ORB point.
    jacobian = np.empty((2 * len(world_array), 6), dtype=np.float64)
    steps = np.array([1e-6, 1e-6, 1e-6, 1e-5, 1e-5, 1e-5])
    for axis, step in enumerate(steps):
        plus, minus = parameters.copy(), parameters.copy()
        plus[axis] += step
        minus[axis] -= step
        jacobian[:, axis] = (
            _project(world_array, plus, calibration)
            - _project(world_array, minus, calibration)
        ).reshape(-1) / (2.0 * step)
    repeated_weights = np.repeat(np.asarray(point_weights, dtype=np.float64), 2)
    weighted_jacobian = jacobian * np.sqrt(repeated_weights)[:, None]
    information = weighted_jacobian.T @ weighted_jacobian
    singular_values = np.linalg.svd(information, compute_uv=False)
    if singular_values[-1] <= max(1e-12, singular_values[0] * 1e-12):
        return corner_diagnostics, {
            "valid": False,
            "reason": "degenerate_projection_geometry",
            "corners": len(world_array),
            "condition_number": None,
        }
    weighted_residual = residual_vectors.reshape(-1) * np.sqrt(repeated_weights)
    rms_px = float(np.sqrt(np.mean(weighted_residual ** 2)))
    pixel_sigma_px = max(0.25, rms_px)
    covariance = np.linalg.inv(information) * pixel_sigma_px ** 2
    diagonal = np.maximum(np.diag(covariance), 0.0)
    rotation_std_deg = np.rad2deg(np.sqrt(diagonal[:3]))
    translation_std_m = np.sqrt(diagonal[3:])
    return corner_diagnostics, {
        "valid": True,
        "model": "local_projection_hessian",
        "corners": len(world_array),
        "pixel_sigma_px": pixel_sigma_px,
        "condition_number": float(singular_values[0] / singular_values[-1]),
        "translation_std_m": translation_std_m.tolist(),
        "position_std_m": float(np.linalg.norm(translation_std_m)),
        "rotation_std_deg": rotation_std_deg.tolist(),
        "rotation_std_deg_rms": float(np.linalg.norm(rotation_std_deg)),
    }
