"""Disambiguate ONE static marker against an independent, fixed ORB trajectory.

The input poses must use one common map revision/gauge. Camera rotations are
fixed; only a positive translation scale and this marker's independent SE(3)
are fitted. No relative layout/coplanarity of different markers is assumed.
Returned camera-in-marker poses are branch-selection priors, NOT substitute
camera measurements. A caller must refit each original four-corner observation
and preserve this fit's evidence interval before publishing an offline result.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares

from .models import Calibration, Pose
from .pose import reprojection_error, solve_planar_pose, square_object_points


@dataclass(frozen=True)
class MarkerView:
    frame_id: int
    timestamp: float
    camera_pose: Pose  # camera-to-ORB-world, translation in the given ORB units
    corners: np.ndarray  # four original distorted pixels, decoded corner order
    information_weight: float = 1.0
    keyframe_id: int | None = None
    map_id: int | str = 0
    gauge_id: int | str = 0


@dataclass(frozen=True)
class MultiViewGates:
    minimum_views: int = 8
    minimum_keyframes: int = 3
    minimum_span_s: float = .15
    temporal_correlation_s: float = .05
    minimum_baseline_m: float = .04
    minimum_baseline_depth_ratio: float = .01
    maximum_rms_px: float = 1.5
    maximum_corner_error_px: float = 3.0
    pixel_noise_floor: float = .5
    maximum_relative_scale_std: float = .10
    maximum_rotation_std_deg: float = 5.0
    minimum_branch_gap_px: float = .35


@dataclass
class MultiViewMarkerFit:
    accepted: bool
    reason: str
    marker_id: int
    scale_m_per_unit: float | None = None
    marker_pose: Pose | None = None  # marker-to-world in metres, origin=C0
    origin_camera_center: np.ndarray | None = None
    camera_in_marker: dict[int, Pose] = field(default_factory=dict)
    training_frame_ids: tuple[int, ...] = ()
    validation_frame_ids: tuple[int, ...] = ()
    diagnostics: dict = field(default_factory=dict)


def fit_marker_multiview(
    views: list[MarkerView], calibration: Calibration, marker_size_m: float,
    marker_id: int, *, trajectory_is_independent: bool,
    fixed_scale: float | None = None, trajectory_marker_ids: tuple[int, ...] = (),
    gates: MultiViewGates = MultiViewGates(),
) -> MultiViewMarkerFit:
    """Fit independent ORB motion + repeated strong corners with held-out frames.

``fixed_scale`` is allowed for a separately calibrated metric trajectory, not
a scale previously estimated from the same target marker. Even with a fixed
scale, the motion baseline and branch/uncertainty gates remain in force.
``trajectory_is_independent`` must be explicitly established by the caller;
target-marker-constrained camera poses cannot validate that same marker.
"""
    out = MultiViewMarkerFit(False, 'insufficient_strong_views', marker_id)
    if not trajectory_is_independent or marker_id in trajectory_marker_ids:
        out.reason = 'trajectory_not_independent'
        return out
    if not np.isfinite(marker_size_m) or marker_size_m <= 0:
        raise ValueError('marker size must be positive and finite')
    if fixed_scale is not None and (not np.isfinite(fixed_scale) or fixed_scale <= 0):
        raise ValueError('fixed scale must be positive and finite')
    selected = sorted((v for v in views if np.isfinite(v.information_weight)
                       and .99 <= v.information_weight <= 1.0),
                      key=lambda v: v.timestamp)
    if len(selected) < gates.minimum_views:
        return out
    if len({v.frame_id for v in selected}) != len(selected):
        raise ValueError('duplicate frame IDs are not independent observations')
    if len({(v.map_id, v.gauge_id) for v in selected}) != 1:
        out.reason = 'mixed_map_or_gauge'
        return out
    times = np.array([v.timestamp for v in selected], float)
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError('observation timestamps must be finite and distinct')
    if times[-1] - times[0] < gates.minimum_span_s:
        out.reason = 'insufficient_time_span'
        return out
    keyframes = {v.keyframe_id for v in selected if v.keyframe_id is not None}
    if keyframes and len(keyframes) < gates.minimum_keyframes:
        out.reason = 'insufficient_keyframes'
        return out
    pixels = np.array([np.asarray(v.corners).reshape(4, 2) for v in selected], float)
    rotations = np.array([v.camera_pose.rotation_matrix for v in selected])
    centers = np.array([v.camera_pose.tvec.reshape(3) for v in selected], float)
    if not all(np.all(np.isfinite(a)) for a in (pixels, rotations, centers)):
        raise ValueError('all geometric inputs must be finite')
    origin = centers[0].copy()
    centers = centers - origin
    distances = np.linalg.norm(centers[:, None] - centers[None, :], axis=2)
    if float(distances.max()) <= 1e-7:
        out.reason = 'unobservable_translation_scale'
        return out
    # Adjacent high-rate exposures are not independent validation evidence.
    # Hold out WHOLE 50 ms blocks and cap each block's total information at one
    # observation. The noise floor/covariance remains conditional on ORB poses.
    blocks = np.floor((times - times[0]) / gates.temporal_correlation_s + 1e-8).astype(int)
    unique_blocks = sorted(set(blocks))
    if len(unique_blocks) < 6:
        out.reason = 'insufficient_independent_time_blocks'
        return out
    validation_blocks = set(unique_blocks[2::3])
    validation = np.flatnonzero(np.array([b in validation_blocks for b in blocks]))
    training = np.flatnonzero(np.array([b not in validation_blocks for b in blocks]))
    frame_weights = np.array([1. / np.sqrt(np.count_nonzero(blocks == b)) for b in blocks])
    training_weights = frame_weights[training, None, None]
    out.training_frame_ids = tuple(selected[i].frame_id for i in training)
    out.validation_frame_ids = tuple(selected[i].frame_id for i in validation)
    object_points = square_object_points(marker_size_m)
    camera_from_world = rotations.transpose(0, 2, 1)

    def geometry(x, indices):
        scale = fixed_scale if fixed_scale is not None else np.exp(x[6])
        world = object_points @ cv2.Rodrigues(x[:3])[0].T + x[3:6]
        camera = np.einsum('nij,nkj->nki', camera_from_world[indices],
                           world[None] - scale * centers[indices, None])
        projected = cv2.projectPoints(camera.reshape(-1, 3), np.zeros(3), np.zeros(3),
                                     calibration.camera_matrix, calibration.dist_coeffs)[0]
        return projected.reshape(-1, 4, 2), camera[:, :, 2], scale

    def residual(x):
        projected, depth, _ = geometry(x, training)
        return np.r_[((projected - pixels[training]) * training_weights).ravel(),
                     (1000 * np.minimum(depth - .005, 0) * training_weights[:, :, 0]).ravel()]

    # Keep BOTH positive-depth IPPE solutions, at separated training frames.
    anchors = sorted({int(training[0]), int(training[len(training) // 2]), int(training[-1])})
    ippe = {}
    for index in anchors:
        solved = cv2.solvePnPGeneric(object_points, pixels[index], calibration.camera_matrix,
                                    calibration.dist_coeffs, flags=cv2.SOLVEPNP_IPPE)
        candidates = []
        for rvec, tvec in zip(solved[1], solved[2]):
            rotation = cv2.Rodrigues(rvec)[0]
            if np.all(np.isfinite(tvec)) and np.all((object_points @ rotation.T + tvec.reshape(3))[:, 2] > 0):
                candidates.append((rotations[index] @ rotation,
                                   rotations[index] @ tvec.reshape(3)))
        ippe[index] = candidates
    if any(not values for values in ippe.values()):
        out.reason = 'ippe_initialization_failed'
        return out
    if fixed_scale is not None:
        scale_seeds = [fixed_scale]
    else:
        a, b = max(((a, b) for a in anchors for b in anchors if a < b),
                   key=lambda pair: distances[pair])
        delta = centers[b] - centers[a]
        scale_seeds = [float(np.dot(first[1] - second[1], delta) / np.dot(delta, delta))
                       for first in ippe[a] for second in ippe[b]]
        scale_seeds = [s for s in scale_seeds if np.isfinite(s) and 1e-6 < s < 1e6]
        if not scale_seeds:
            out.reason = 'no_positive_scale_initialization'
            return out
        scale_seeds = sorted(set(round(s, 8) for s in scale_seeds))
    fits = []
    for index in anchors:
        for rotation, translation in ippe[index]:
            for scale in scale_seeds:
                x = np.r_[cv2.Rodrigues(rotation)[0].ravel(), translation + scale * centers[index]]
                if fixed_scale is None:
                    x = np.r_[x, np.log(scale)]
                bounds = (np.r_[[-np.inf] * 6, np.log(1e-6)],
                          np.r_[[np.inf] * 6, np.log(1e6)]) if fixed_scale is None else (-np.inf, np.inf)
                fit = least_squares(residual, x, loss='soft_l1', f_scale=1.0,
                                    max_nfev=200, x_scale='jac', bounds=bounds)
                projected, depth, fitted_scale = geometry(fit.x, np.arange(len(selected)))
                errors = np.linalg.norm(projected - pixels, axis=2)
                if not fit.success or not np.all(np.isfinite(errors)) or depth.min() <= .02:
                    continue
                train_rms = float(np.sqrt(np.sum(errors[training] ** 2 * frame_weights[training, None] ** 2)
                                         / (4 * (len(unique_blocks) - len(validation_blocks)))))
                held_rms = float(np.sqrt(np.sum(errors[validation] ** 2 * frame_weights[validation, None] ** 2)
                                        / (4 * len(validation_blocks))))
                fits.append((train_rms, held_rms, fit.x, errors, depth, fitted_scale))
    if not fits:
        out.reason = 'optimization_failed'
        return out
    fits.sort(key=lambda f: f[0])  # held-out evidence does not select the winner
    train_rms, held_rms, x, errors, depth, scale = fits[0]
    rotation = cv2.Rodrigues(x[:3])[0]
    distinct = []
    for other in fits[1:]:
        angle = np.linalg.norm(cv2.Rodrigues(rotation.T @ cv2.Rodrigues(other[2][:3])[0])[0])
        if (angle > np.deg2rad(5) or np.linalg.norm(x[3:6] - other[2][3:6]) > .02
                or abs(np.log(scale / other[5])) > .10):
            distinct.append(other)
    baseline = float(distances.max() * scale)
    median_depth = float(np.median(depth))
    out.scale_m_per_unit = float(scale)
    out.origin_camera_center = origin
    out.diagnostics = {
        'views': len(selected), 'keyframes': len(keyframes),
        'training_rms_px': train_rms, 'validation_rms_px': held_rms,
        'maximum_corner_error_px': float(errors.max()), 'baseline_m': baseline,
        'baseline_depth_ratio': baseline / median_depth,
        'ippe_initializations': sum(len(v) for v in ippe.values()),
        'converged_starts': len(fits), 'distinct_competitors': len(distinct),
        'evidence_start_s': float(times[0]), 'evidence_end_s': float(times[-1]),
        'temporal_correlation_s': gates.temporal_correlation_s,
        'training_time_blocks': len(unique_blocks) - len(validation_blocks),
        'validation_time_blocks': len(validation_blocks),
        'scale_mode': 'independent_fixed' if fixed_scale is not None else 'estimated',
    }
    if baseline < gates.minimum_baseline_m or baseline / median_depth < gates.minimum_baseline_depth_ratio:
        out.reason = 'insufficient_motion_baseline'
        return out
    if train_rms > gates.maximum_rms_px or held_rms > gates.maximum_rms_px or errors.max() > gates.maximum_corner_error_px:
        out.reason = 'reprojection_validation_failed'
        return out
    if distinct:
        competitor = min(distinct, key=lambda f: f[1])
        out.diagnostics['competitor_validation_rms_px'] = competitor[1]
        if competitor[1] - held_rms < gates.minimum_branch_gap_px:
            out.reason = 'unresolved_multiview_branches'
            return out
    # Conditional covariance with a pixel-noise floor. Camera-pose uncertainty
    # is not included: callers must retain holdout residual and ORB provenance.
    base = ((geometry(x, training)[0] - pixels[training]) * training_weights).ravel()
    jacobian = np.empty((len(base), len(x)))
    for parameter in range(len(x)):
        step = 1e-6 * max(1.0, abs(x[parameter]))
        perturbed = x.copy()
        perturbed[parameter] += step
        jacobian[:, parameter] = (((geometry(perturbed, training)[0] - pixels[training])
                                   * training_weights).ravel() - base) / step
    norms = np.linalg.norm(jacobian, axis=0)
    singular = np.linalg.svd(jacobian / np.maximum(norms, 1e-12), compute_uv=False)
    out.diagnostics['normalized_jacobian_condition'] = float(singular[0] / max(singular[-1], 1e-15))
    if singular[-1] < singular[0] * 1e-6:
        out.reason = 'unobservable_geometry'
        return out
    sigma = max(gates.pixel_noise_floor, train_rms / np.sqrt(2))
    try:
        covariance = np.linalg.inv(jacobian.T @ jacobian) * sigma ** 2
    except np.linalg.LinAlgError:
        out.reason = 'unobservable_geometry'
        return out
    if not np.all(np.isfinite(covariance)) or np.any(np.diag(covariance) < 0):
        out.reason = 'unobservable_geometry'
        return out
    rotation_std = float(np.sqrt(np.linalg.eigvalsh(covariance[:3, :3]).max()) * 180 / np.pi)
    scale_std = float(np.sqrt(covariance[6, 6])) if fixed_scale is None else 0.0
    out.diagnostics.update(rotation_std_deg=rotation_std, relative_scale_std=scale_std,
                           uncertainty_conditional_on_fixed_orb_poses=True)
    if scale_std > gates.maximum_relative_scale_std or rotation_std > gates.maximum_rotation_std_deg:
        out.reason = 'uncertain_marker_pose_or_scale'
        return out
    out.marker_pose = Pose(x[:3].reshape(3, 1).copy(), x[3:6].reshape(3, 1).copy(), train_rms,
                           marker_ids=(marker_id,), inlier_count=4 * len(selected))
    for index, view in enumerate(selected):
        camera_rotation = rotation.T @ rotations[index]
        camera_translation = rotation.T @ (scale * centers[index] - x[3:6])
        out.camera_in_marker[view.frame_id] = Pose(cv2.Rodrigues(camera_rotation)[0],
                                                  camera_translation.reshape(3, 1),
                                                  float(np.sqrt(np.mean(errors[index] ** 2))),
                                                  marker_ids=(marker_id,), inlier_count=4)
    out.accepted, out.reason = True, 'accepted'
    return out


def refit_marker_view(
    view: MarkerView, fit: MultiViewMarkerFit, calibration: Calibration,
    marker_size_m: float, maximum_error_px: float = 2.5,
) -> Pose | None:
    """Refit actual corners after selecting an IPPE branch; never extrapolate.

The returned pose is camera-to-marker. The joint trajectory is used only as
an IPPE branch selector, not as an extra residual or a frozen output pose.
The caller must attach ``fit.diagnostics['evidence_end_s']`` as availability.
"""
    if not fit.accepted or view.information_weight < .99 or view.frame_id not in fit.camera_in_marker:
        return None
    prior = fit.camera_in_marker[view.frame_id]
    rotation = prior.rotation_matrix.T
    previous = Pose(cv2.Rodrigues(rotation)[0], -rotation @ prior.tvec, 0.)
    points = square_object_points(marker_size_m)
    measured = solve_planar_pose(points, view.corners, calibration.camera_matrix,
                                 calibration.dist_coeffs, previous, maximum_error_px)
    if measured is None:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(points, np.asarray(view.corners, float),
                                    calibration.camera_matrix, calibration.dist_coeffs,
                                    measured.rvec.copy(), measured.tvec.copy())
    current_rotation = cv2.Rodrigues(rvec)[0]
    error = reprojection_error(points, view.corners, rvec, tvec,
                               calibration.camera_matrix, calibration.dist_coeffs)
    angle = np.linalg.norm(cv2.Rodrigues(rotation.T @ current_rotation)[0])
    if (not np.isfinite(error) or error > maximum_error_px or
            np.any((points @ current_rotation.T + tvec.reshape(3))[:, 2] <= 0) or
            angle > np.deg2rad(25) or np.linalg.norm(tvec - previous.tvec) > .10):
        return None
    inverse_rotation = current_rotation.T
    return Pose(cv2.Rodrigues(inverse_rotation)[0], -inverse_rotation @ tvec,
                error, marker_ids=(fit.marker_id,), inlier_count=4, ambiguous=True)
