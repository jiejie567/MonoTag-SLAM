from __future__ import annotations

import cv2
import numpy as np

from .models import BandLayout, Pose
from .pose import reprojection_error, solve_planar_pose


def solve_band_pose(
    detections: dict[int, np.ndarray],
    layout: BandLayout,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    previous: Pose | None = None,
    max_error_px: float = 5.0,
    min_inlier_markers: int = 1,
    validate_planar_ambiguity: bool = False,
    assist_detections: dict[int, np.ndarray] | None = None,
) -> Pose | None:
    visible = sorted(set(detections).intersection(layout.markers))
    if not visible:
        return None
    object_points = np.concatenate([layout.markers[i] for i in visible]).astype(np.float64)
    image_points = np.concatenate([detections[i] for i in visible]).astype(np.float64)

    if len(visible) == 1 and validate_planar_ambiguity:
        if min_inlier_markers > 1:
            return None
        assist_visible = sorted(
            set(assist_detections or {}).intersection(layout.markers).difference(visible)
        )
        assist_object_points = (
            np.concatenate([layout.markers[i] for i in assist_visible])
            if assist_visible else None
        )
        assist_image_points = (
            np.concatenate([assist_detections[i] for i in assist_visible])
            if assist_visible else None
        )
        pose = solve_planar_pose(
            object_points, image_points, camera_matrix, dist_coeffs,
            previous, max_error_px, assist_object_points, assist_image_points,
        )
        if pose is not None:
            pose.marker_ids = tuple(visible)
            pose.inlier_count = 4
            # Preserve the existing single-marker provenance flag. This is
            # distinct from an unresolved two-solution ambiguity (rejected).
            pose.ambiguous = True
        return pose

    success = False
    rvec = tvec = inliers = None
    if previous is not None:
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            previous.rvec.copy(),
            previous.tvec.copy(),
            True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if success and tvec[2, 0] > 0:
            projected, _ = cv2.projectPoints(
                object_points, rvec, tvec, camera_matrix, dist_coeffs
            )
            residuals = np.linalg.norm(
                projected.reshape(-1, 2) - image_points.reshape(-1, 2), axis=1
            )
            inlier_idx = np.flatnonzero(residuals <= max_error_px)
            inliers = inlier_idx.reshape(-1, 1) if len(inlier_idx) >= 4 else None
            success = inliers is not None
        else:
            success = False

    if not success and len(visible) == 1:
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE,
        )
        inliers = np.arange(4).reshape(-1, 1) if success else None
    elif not success:
        kwargs = {}
        if previous is not None:
            kwargs = {"rvec": previous.rvec.copy(), "tvec": previous.tvec.copy(), "useExtrinsicGuess": True}
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            iterationsCount=150,
            reprojectionError=max_error_px,
            confidence=0.995,
            flags=cv2.SOLVEPNP_ITERATIVE,
            **kwargs,
        )
    if not success or tvec[2, 0] <= 0 or inliers is None or len(inliers) < 4:
        return None

    inlier_idx = inliers.reshape(-1)
    if len(inlier_idx) >= 6:
        rvec, tvec = cv2.solvePnPRefineLM(
            object_points[inlier_idx], image_points[inlier_idx], camera_matrix, dist_coeffs, rvec, tvec
        )
    error = reprojection_error(
        object_points[inlier_idx], image_points[inlier_idx], rvec, tvec, camera_matrix, dist_coeffs
    )
    if error > max_error_px:
        return None
    inlier_markers = tuple(
        marker_id
        for marker_index, marker_id in enumerate(visible)
        if np.count_nonzero((inlier_idx // 4) == marker_index) >= 3
    )
    if len(inlier_markers) < min_inlier_markers:
        return None
    return Pose(
        np.asarray(rvec).reshape(3, 1),
        np.asarray(tvec).reshape(3, 1),
        error,
        marker_ids=inlier_markers,
        inlier_count=len(inlier_idx),
        ambiguous=len(inlier_markers) == 1,
    )
