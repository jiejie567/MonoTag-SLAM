from __future__ import annotations

import cv2
import numpy as np

from .models import Pose


def square_object_points(size_m: float) -> np.ndarray:
    half = size_m / 2.0
    return np.array(
        [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]],
        dtype=np.float64,
    )


def reprojection_error(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> float:
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    residuals = projected.reshape(-1, 2) - image_points.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(residuals * residuals, axis=1))))


def solve_planar_pose(
    object_points: np.ndarray,
    corners: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    previous: Pose | None = None,
    max_error_px: float = 5.0,
    assist_object_points: np.ndarray | None = None,
    assist_image_points: np.ndarray | None = None,
) -> Pose | None:
    """Select a planar solution, not just the smallest four-corner residual.

    The caller must expire ``previous`` after a measurement gap. Close pixel
    errors with materially different poses are not an absolute initializer;
    a recent, geometrically compatible prediction may disambiguate them.
    """
    points = np.asarray(object_points, dtype=np.float64).reshape(4, 3)
    pixels = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    solved = cv2.solvePnPGeneric(points, pixels, camera_matrix, dist_coeffs,
                                 flags=cv2.SOLVEPNP_IPPE)
    candidates = []
    if not solved[0]:
        return None
    for rvec, tvec in zip(solved[1], solved[2]):
        if not np.all(np.isfinite(rvec)) or not np.all(np.isfinite(tvec)):
            continue
        pose = Pose(np.asarray(rvec).reshape(3, 1), np.asarray(tvec).reshape(3, 1), 0.)
        if np.any((pose.rotation_matrix @ points.T + pose.tvec)[2] <= 0):
            continue
        pose.reprojection_error_px = reprojection_error(
            points, pixels, pose.rvec, pose.tvec, camera_matrix, dist_coeffs)
        if np.isfinite(pose.reprojection_error_px) and pose.reprojection_error_px <= max_error_px:
            candidates.append(pose)
    if not candidates:
        # IPPE is the right ambiguity-aware solver for ordinary views, but it
        # is numerically degenerate for an exactly frontal square.  In that
        # case OpenCV may return duplicate zero-rotation hypotheses whose
        # reprojection error is large even though the iterative planar solve
        # has a unique, positive-depth solution.  Recover that edge case with
        # the image/object correspondence itself; do not use it to bypass the
        # two-hypothesis ambiguity gate when IPPE produced valid candidates.
        ok, rvec, tvec = cv2.solvePnP(
            points, pixels, camera_matrix, dist_coeffs,
            previous.rvec.copy() if previous is not None else None,
            previous.tvec.copy() if previous is not None else None,
            previous is not None,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok or not np.all(np.isfinite(rvec)) or not np.all(np.isfinite(tvec)):
            return None
        fallback = Pose(np.asarray(rvec).reshape(3, 1), np.asarray(tvec).reshape(3, 1), 0.)
        if np.any((fallback.rotation_matrix @ points.T + fallback.tvec)[2] <= 0):
            return None
        fallback.reprojection_error_px = reprojection_error(
            points, pixels, fallback.rvec, fallback.tvec, camera_matrix, dist_coeffs)
        if not np.isfinite(fallback.reprojection_error_px) or fallback.reprojection_error_px > max_error_px:
            return None
        fallback.ambiguous = previous is None
        return fallback
    candidates.sort(key=lambda pose: pose.reprojection_error_px)
    best = candidates[0]
    if len(candidates) < 2:
        return best
    other = candidates[1]
    angle = np.linalg.norm(cv2.Rodrigues(best.rotation_matrix.T @ other.rotation_matrix)[0])
    first_center = -best.rotation_matrix.T @ best.tvec
    second_center = -other.rotation_matrix.T @ other.tvec
    distinct = angle > np.deg2rad(5.) or np.linalg.norm(first_center-second_center) > .02
    # 0.5 px is a residual separation gate, not a claim of metric accuracy.
    unresolved = distinct and other.reprojection_error_px-best.reprojection_error_px < .5
    if not unresolved:
        return best
    if assist_object_points is not None and assist_image_points is not None:
        assist_world = np.asarray(assist_object_points, dtype=np.float64).reshape(-1, 3)
        assist_pixels = np.asarray(assist_image_points, dtype=np.float64).reshape(-1, 2)
        if len(assist_world) >= 4 and len(assist_world) == len(assist_pixels):
            scored = sorted(
                (
                    reprojection_error(
                        assist_world, assist_pixels, pose.rvec, pose.tvec,
                        camera_matrix, dist_coeffs,
                    ),
                    pose,
                )
                for pose in candidates
            )
            # A rejected neighbouring marker is a vote, never a metric factor:
            # use it only when one branch clearly explains its approximate
            # location and neither pose parameter is subsequently refined by it.
            if scored[0][0] <= 12.0 and scored[1][0] - scored[0][0] >= 1.0:
                return scored[0][1]
    if previous is None:
        return None
    center = points.mean(axis=0).reshape(3, 1)
    predicted_center = previous.rotation_matrix @ center + previous.tvec
    compatible = []
    for pose in candidates:
        rotation_delta = np.linalg.norm(cv2.Rodrigues(previous.rotation_matrix.T @ pose.rotation_matrix)[0])
        center_delta = np.linalg.norm(pose.rotation_matrix @ center + pose.tvec-predicted_center)
        if rotation_delta <= np.deg2rad(25.) and center_delta <= .10:
            compatible.append((rotation_delta/.15 + center_delta/.03, pose))
    return min(compatible, key=lambda item: item[0])[1] if compatible else None


def solve_square_pose(
    corners: np.ndarray,
    size_m: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    previous: Pose | None = None,
) -> Pose | None:
    object_points = square_object_points(size_m)
    result = cv2.solvePnPGeneric(
        object_points,
        np.asarray(corners, dtype=np.float64).reshape(4, 2),
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not result[0]:
        return None

    candidates: list[tuple[float, Pose]] = []
    for rvec, tvec in zip(result[1], result[2]):
        rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
        if tvec[2, 0] <= 0:
            continue
        error = reprojection_error(
            object_points, corners, rvec, tvec, camera_matrix, dist_coeffs
        )
        score = error
        if previous is not None:
            score += 15.0 * float(np.linalg.norm(tvec - previous.tvec))
            relative = cv2.Rodrigues(rvec)[0] @ cv2.Rodrigues(previous.rvec)[0].T
            score += 0.15 * float(np.linalg.norm(cv2.Rodrigues(relative)[0]))
        candidates.append((score, Pose(rvec, tvec, error)))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    pose = candidates[0][1]
    pose.ambiguous = len(candidates) > 1 and abs(candidates[1][0] - candidates[0][0]) < 0.2
    return pose
