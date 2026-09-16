from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .bandsolve import solve_band_pose
from .models import BandLayout, Calibration, Pose


@dataclass(frozen=True)
class TagPoseResult:
    pose: Pose | None
    accepted_marker_ids: tuple[int, ...]
    rejected_marker_ids: tuple[int, ...]
    marker_errors_px: dict[int, float]
    graph_reprojection_error_px: float | None
    confidence: float
    consensus_vetoed: bool = False


@dataclass(frozen=True)
class WristTrajectoryResult:
    poses: list[Pose | None]
    reprojection_errors_px: list[float | None]


@dataclass(frozen=True)
class WorldTrackingResult:
    poses: list[Pose | None]
    sources: list[str]


def _camera_from_world_wrist(
    world_from_camera: Pose,
    world_from_wrist: Pose,
) -> Pose:
    camera_rotation = world_from_camera.rotation_matrix.T
    return Pose(
        cv2.Rodrigues(camera_rotation @ world_from_wrist.rotation_matrix)[0],
        camera_rotation @ (world_from_wrist.tvec - world_from_camera.tvec),
        world_from_wrist.reprojection_error_px,
        world_from_wrist.marker_ids,
        world_from_wrist.inlier_count,
        world_from_wrist.ambiguous,
    )


def _world_from_camera_wrist(
    world_from_camera: Pose,
    camera_from_wrist: Pose,
) -> Pose:
    return Pose(
        cv2.Rodrigues(
            world_from_camera.rotation_matrix @ camera_from_wrist.rotation_matrix
        )[0],
        world_from_camera.rotation_matrix @ camera_from_wrist.tvec
        + world_from_camera.tvec,
        camera_from_wrist.reprojection_error_px,
        camera_from_wrist.marker_ids,
        camera_from_wrist.inlier_count,
        camera_from_wrist.ambiguous,
    )


def _implausible_single_marker_step(
    first: Pose,
    second: Pose,
    elapsed_s: float,
) -> bool:
    """Conservative human-motion gate for an ambiguous planar measurement."""
    translation_step = float(np.linalg.norm(second.tvec - first.tvec))
    rotation_step = float(
        np.linalg.norm(
            cv2.Rodrigues(first.rotation_matrix.T @ second.rotation_matrix)[0]
        )
    )
    translation_limit = min(0.08, 0.015 + 2.0 * elapsed_s)
    rotation_limit = min(
        np.deg2rad(45.0),
        np.deg2rad(5.0 + 1200.0 * elapsed_s),
    )
    return translation_step > translation_limit or rotation_step > rotation_limit


def _prefer_dominant_face_when_small_face_disagrees(
    result: TagPoseResult,
    predicted_pose: Pose,
    detections: dict[int, np.ndarray],
    layout: BandLayout,
    calibration: Calibration,
    assist_detections: dict[int, np.ndarray],
) -> TagPoseResult:
    """Keep a tiny second face as ambiguity evidence, not a pose driver."""
    if result.pose is None or len(result.accepted_marker_ids) < 2:
        return result
    by_area = sorted(
        result.accepted_marker_ids,
        key=lambda marker_id: _marker_area(detections[marker_id]),
        reverse=True,
    )
    dominant_id = by_area[0]
    if _marker_area(detections[by_area[1]]) >= 0.5 * _marker_area(
        detections[dominant_id]
    ):
        return result
    auxiliary = {
        marker_id: corners
        for marker_id, corners in {**assist_detections, **detections}.items()
        if marker_id != dominant_id and marker_id in layout.markers
    }
    dominant = optimize_tag_pose(
        {dominant_id: detections[dominant_id]},
        layout,
        calibration,
        predicted_pose,
        validate_planar_ambiguity=True,
        assist_detections=auxiliary,
    )
    if dominant.pose is None:
        return result
    translation_disagreement = float(np.linalg.norm(
        result.pose.tvec - dominant.pose.tvec
    ))
    rotation_disagreement = float(np.linalg.norm(cv2.Rodrigues(
        dominant.pose.rotation_matrix.T @ result.pose.rotation_matrix
    )[0]))
    if (
        translation_disagreement <= 0.015
        and rotation_disagreement <= np.deg2rad(5.0)
    ):
        return result
    visible_ids = tuple(sorted(detections))
    errors = _marker_errors(
        dominant.pose, detections, layout, calibration, visible_ids
    )
    # Size alone does not invalidate a measured second face. A stale planar
    # prediction can agree with the large face yet contradict the small one.
    # Preserve the already-admitted joint solution only with tight per-face
    # corroboration (2 px); retain the existing 5 px outlier boundary for the
    # competing single-face pose. This is not admission of rejected corners.
    if (
        all(result.marker_errors_px.get(marker_id, np.inf) <= 2.0
            for marker_id in result.accepted_marker_ids)
        and any(errors[marker_id] > 5.0
                for marker_id in result.accepted_marker_ids)
    ):
        return result
    rejected = tuple(marker_id for marker_id in visible_ids if marker_id != dominant_id)
    return TagPoseResult(
        dominant.pose,
        (dominant_id,),
        rejected,
        errors,
        dominant.graph_reprojection_error_px,
        dominant.confidence,
        True,
    )


def _pose_vector(pose: Pose) -> np.ndarray:
    return np.concatenate((pose.rvec.reshape(3), pose.tvec.reshape(3)))


def _vector_pose(value: np.ndarray, error: float = 0.0) -> Pose:
    value = np.asarray(value, dtype=np.float64).reshape(6)
    return Pose(value[:3].reshape(3, 1), value[3:].reshape(3, 1), error)


def _marker_area(corners: np.ndarray) -> float:
    return abs(cv2.contourArea(np.asarray(corners, dtype=np.float32).reshape(4, 2)))


def _marker_sigma_px(
    marker_points: np.ndarray,
    image_corners: np.ndarray,
    pose: Pose,
    maximum_area: float,
) -> float:
    area = max(_marker_area(image_corners), 100.0)
    area_penalty = np.sqrt(maximum_area / area)
    first = marker_points[1] - marker_points[0]
    second = marker_points[3] - marker_points[0]
    normal = np.cross(first, second)
    norm = np.linalg.norm(normal)
    if norm <= 1e-12:
        viewing_cosine = 0.2
    else:
        camera_normal = pose.rotation_matrix @ (normal / norm)
        viewing_cosine = max(abs(float(camera_normal[2])), 0.2)
    viewing_penalty = 1.0 / np.sqrt(viewing_cosine)
    return float(np.clip(area_penalty * viewing_penalty, 1.0, 3.0))


def projection_residuals(
    pose_vector: np.ndarray,
    detections: dict[int, np.ndarray],
    layout: BandLayout,
    calibration: Calibration,
    marker_ids: tuple[int, ...],
    marker_sigmas_px: dict[int, float] | None = None,
) -> np.ndarray:
    pose = _vector_pose(pose_vector)
    residuals: list[np.ndarray] = []
    for marker_id in marker_ids:
        projected, _ = cv2.projectPoints(
            layout.markers[marker_id],
            pose.rvec,
            pose.tvec,
            calibration.camera_matrix,
            calibration.dist_coeffs,
        )
        residual = projected.reshape(4, 2) - detections[marker_id].reshape(4, 2)
        sigma = 1.0 if marker_sigmas_px is None else marker_sigmas_px[marker_id]
        residuals.append((residual / sigma).reshape(-1))
    return np.concatenate(residuals) if residuals else np.empty(0, dtype=np.float64)


def _optimization_residuals(
    pose_vector: np.ndarray,
    detections: dict[int, np.ndarray],
    layout: BandLayout,
    calibration: Calibration,
    marker_ids: tuple[int, ...],
    marker_sigmas_px: dict[int, float],
    predicted_pose: Pose | None,
) -> np.ndarray:
    residual = projection_residuals(
        pose_vector,
        detections,
        layout,
        calibration,
        marker_ids,
        marker_sigmas_px,
    )
    if predicted_pose is None or len(marker_ids) != 1:
        return residual
    pose = _vector_pose(pose_vector)
    rotation_error = cv2.Rodrigues(
        predicted_pose.rotation_matrix.T @ pose.rotation_matrix
    )[0].reshape(3)
    translation_error = pose.tvec.reshape(3) - predicted_pose.tvec.reshape(3)
    return np.concatenate(
        (residual, rotation_error / 0.12, translation_error / 0.02)
    )


def _marker_errors(
    pose: Pose,
    detections: dict[int, np.ndarray],
    layout: BandLayout,
    calibration: Calibration,
    marker_ids: tuple[int, ...],
) -> dict[int, float]:
    errors: dict[int, float] = {}
    for marker_id in marker_ids:
        projected, _ = cv2.projectPoints(
            layout.markers[marker_id],
            pose.rvec,
            pose.tvec,
            calibration.camera_matrix,
            calibration.dist_coeffs,
        )
        residual = projected.reshape(4, 2) - detections[marker_id].reshape(4, 2)
        errors[marker_id] = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    return errors


def optimize_tag_pose(
    detections: dict[int, np.ndarray],
    layout: BandLayout,
    calibration: Calibration,
    predicted_pose: Pose | None = None,
    max_marker_error_px: float = 5.0,
    max_graph_error_px: float = 4.0,
    marker_weights: dict[int, float] | None = None,
    validate_planar_ambiguity: bool = False,
    assist_detections: dict[int, np.ndarray] | None = None,
) -> TagPoseResult:
    weights = marker_weights or {}
    visible = tuple(sorted(
        marker_id for marker_id in set(detections).intersection(layout.markers)
        if weights.get(marker_id, 1.0) > 0.0
    ))
    if not visible:
        return TagPoseResult(None, (), (), {}, None, 0.0)

    # Weak observations may refine a known pose but cannot overrule a reliable
    # tag when selecting the initial planar solution, or initialize on their own.
    strong = {marker_id: detections[marker_id] for marker_id in visible
              if weights.get(marker_id, 1.0) >= 1.0}
    if not strong and predicted_pose is None:
        return TagPoseResult(None, (), visible, {}, None, 0.0)
    initial = solve_band_pose(
        strong or {marker_id: detections[marker_id] for marker_id in visible},
        layout,
        calibration.camera_matrix,
        calibration.dist_coeffs,
        predicted_pose,
        max_error_px=max_marker_error_px,
        validate_planar_ambiguity=validate_planar_ambiguity,
        assist_detections=assist_detections,
    )
    if initial is None:
        return TagPoseResult(None, (), visible, {}, None, 0.0)

    maximum_area = max(_marker_area(detections[marker_id]) for marker_id in visible)
    sigmas = {
        marker_id: _marker_sigma_px(
            layout.markers[marker_id], detections[marker_id], initial, maximum_area
        ) / np.sqrt(weights.get(marker_id, 1.0))
        for marker_id in visible
    }

    initial_errors = _marker_errors(initial, detections, layout, calibration, visible)
    seed_ids = tuple(marker_id for marker_id in initial.marker_ids if marker_id in visible)
    if not seed_ids:
        seed_ids = visible
    seed_error = float(np.median([initial_errors[marker_id] for marker_id in seed_ids]))
    seed_cutoff = max(max_marker_error_px, 2.5 * seed_error)
    initial_accepted = tuple(
        marker_id for marker_id in visible if initial_errors[marker_id] <= seed_cutoff
    )
    if not initial_accepted:
        return TagPoseResult(None, (), visible, initial_errors, None, 0.0)

    first = least_squares(
        _optimization_residuals,
        _pose_vector(initial),
        args=(
            detections,
            layout,
            calibration,
            initial_accepted,
            sigmas,
            predicted_pose,
        ),
        loss="huber",
        f_scale=2.0,
        max_nfev=80,
    )
    first_pose = _vector_pose(first.x)
    first_errors = _marker_errors(first_pose, detections, layout, calibration, visible)
    median_error = float(
        np.median([first_errors[marker_id] for marker_id in initial_accepted])
    )
    cutoff = max(max_marker_error_px, 2.5 * median_error)
    accepted = tuple(
        marker_id for marker_id in visible if first_errors[marker_id] <= cutoff
    )
    rejected = tuple(marker_id for marker_id in visible if marker_id not in accepted)
    if not accepted:
        return TagPoseResult(None, (), rejected, first_errors, None, 0.0)

    refined = least_squares(
        _optimization_residuals,
        first.x,
        args=(detections, layout, calibration, accepted, sigmas, predicted_pose),
        loss="huber",
        f_scale=2.0,
        max_nfev=80,
    )
    pose = _vector_pose(refined.x)
    errors = _marker_errors(pose, detections, layout, calibration, visible)
    accepted_residuals = projection_residuals(
        refined.x, detections, layout, calibration, accepted, None
    )
    graph_error = float(np.sqrt(np.mean(accepted_residuals * accepted_residuals)))
    if pose.tvec[2, 0] <= 0 or graph_error > max_graph_error_px:
        return TagPoseResult(None, (), visible, errors, graph_error, 0.0)

    pose.reprojection_error_px = graph_error
    pose.marker_ids = accepted
    pose.inlier_count = 4 * len(accepted)
    pose.ambiguous = len(accepted) == 1
    if predicted_pose is not None and pose.ambiguous:
        translation_delta = float(np.linalg.norm(pose.tvec - predicted_pose.tvec))
        rotation_delta = cv2.Rodrigues(
            predicted_pose.rotation_matrix.T @ pose.rotation_matrix
        )[0]
        if translation_delta > 0.25 or np.linalg.norm(rotation_delta) > np.deg2rad(45.0):
            return TagPoseResult(None, (), visible, errors, graph_error, 0.0)
    reliable = tuple(mid for mid in accepted if weights.get(mid, 1.0) >= 1.0)
    if validate_planar_ambiguity and len(reliable) == 1 and len(strong) > 1:
        # RANSAC/robust fitting can turn a multi-tag input into a single-tag
        # solution. Admission depends on the retained evidence, not how many
        # candidates were originally visible.
        mid = reliable[0]
        single = solve_band_pose(
            {mid: detections[mid]}, layout, calibration.camera_matrix,
            calibration.dist_coeffs, predicted_pose,
            max_error_px=max_marker_error_px, validate_planar_ambiguity=True,
            assist_detections=assist_detections,
        )
        if single is None:
            return TagPoseResult(None, (), visible, errors, graph_error, 0.0)
    confidence_ids = reliable or accepted
    confidence_area = max(_marker_area(detections[mid]) for mid in confidence_ids)
    confidence_error = float(np.sqrt(np.mean([errors[mid] ** 2 for mid in confidence_ids]) / 2.0))
    area_confidence = float(np.clip(confidence_area / 2000.0, 0.0, 1.0))
    error_confidence = float(np.exp(-confidence_error / 3.0))
    marker_confidence = min(1.0, 0.55 + 0.2 * len(confidence_ids))
    confidence = area_confidence * error_confidence * marker_confidence
    # The optional weak tag must not demote the *reliable* tags in the same
    # frame. Its lower information is already applied per corner in PnP/BA.
    # With no reliable tag, keep the whole pose below the absolute-anchor gate.
    if not reliable:
        confidence *= max(weights.get(mid, 1.0) for mid in accepted)
    if ((pose.ambiguous and predicted_pose is None)
            or (validate_planar_ambiguity and len(reliable) == 1)):
        # A recent pose can select an IPPE branch, but it is not another
        # measured tag. Do not let it promote a low-quality fixed single tag
        # across the absolute-anchor confidence gate (or increase BA weight).
        confidence *= 0.6
    return TagPoseResult(
        pose,
        accepted,
        rejected,
        errors,
        graph_error,
        float(np.clip(confidence, 0.0, 1.0)),
    )


def refine_wrist_pose_sequence(
    camera_poses: list[Pose | None],
    metric_flags: list[bool],
    map_ids: list[str | None],
    timestamps_s: list[float] | np.ndarray,
    detections: list[dict[int, np.ndarray]],
    initial_results: list[TagPoseResult],
    layout: BandLayout,
    calibration: Calibration,
    assist_detections: list[dict[int, np.ndarray]] | None = None,
    maximum_prediction_gap_s: float = 0.15,
) -> list[TagPoseResult]:
    """Re-select short-gap planar wrist poses using metric camera motion.

    A missing wrist observation remains missing. The last measured world pose
    only selects among IPPE candidates; it is never emitted as a replacement
    measurement.
    """
    frame_count = len(camera_poses)
    if not (
        len(metric_flags)
        == len(map_ids)
        == len(timestamps_s)
        == len(detections)
        == len(initial_results)
        == frame_count
    ):
        raise ValueError("wrist sequence inputs must have the same length")
    if assist_detections is None:
        assist_detections = [{} for _ in range(frame_count)]
    if len(assist_detections) != frame_count:
        raise ValueError("wrist assist detections must match the frame count")
    if maximum_prediction_gap_s <= 0.0:
        raise ValueError("wrist prediction gap must be positive")

    output: list[TagPoseResult] = []
    previous_world: dict[str, Pose] = {}
    previous_time: dict[str, float] = {}
    previous_evidence_time: dict[str, float] = {}
    for camera, metric, map_id, timestamp, frame_detections, initial, assist in zip(
        camera_poses,
        metric_flags,
        map_ids,
        timestamps_s,
        detections,
        initial_results,
        assist_detections,
    ):
        timestamp = float(timestamp)
        if camera is None or not metric or map_id is None:
            output.append(initial)
            continue
        last_world = previous_world.get(map_id)
        last_timestamp = previous_time.get(map_id)
        recent = (
            last_world is not None
            and last_timestamp is not None
            and 0.0 < timestamp - last_timestamp <= maximum_prediction_gap_s
            and 0.0 < timestamp - previous_evidence_time.get(map_id, -np.inf)
            <= maximum_prediction_gap_s
        )
        result = initial
        if recent and not (initial.pose is None and initial.consensus_vetoed):
            predicted = _camera_from_world_wrist(camera, last_world)
            visible = {
                marker_id: corners
                for marker_id, corners in frame_detections.items()
                if marker_id in layout.markers
            }
            if visible:
                resolved = optimize_tag_pose(
                    visible,
                    layout,
                    calibration,
                    predicted,
                    validate_planar_ambiguity=True,
                    assist_detections=assist,
                )
                if resolved.pose is not None:
                    result = resolved
                elif initial.pose is not None and initial.pose.ambiguous:
                    # Low four-corner error does not make a single planar
                    # branch valid when it contradicts the metric prediction.
                    result = TagPoseResult(
                        None,
                        (),
                        tuple(sorted(visible)),
                        resolved.marker_errors_px or initial.marker_errors_px,
                        resolved.graph_reprojection_error_px,
                        0.0,
                        True,
                    )
                if result.pose is not None:
                    result = _prefer_dominant_face_when_small_face_disagrees(
                        result,
                        predicted,
                        visible,
                        layout,
                        calibration,
                        assist,
                    )
        if result.pose is not None:
            world_pose = _world_from_camera_wrist(camera, result.pose)
            if recent and result.pose.ambiguous:
                elapsed = timestamp - last_timestamp
                if _implausible_single_marker_step(
                    last_world, world_pose, elapsed
                ):
                    result = TagPoseResult(
                        None,
                        (),
                        tuple(sorted(result.accepted_marker_ids)),
                        result.marker_errors_px,
                        result.graph_reprojection_error_px,
                        0.0,
                        True,
                    )
                else:
                    previous_world[map_id] = world_pose
                    previous_time[map_id] = timestamp
            else:
                previous_world[map_id] = world_pose
                previous_time[map_id] = timestamp
        output.append(result)
        # Restoring an ambiguous observation does not restart the evidence
        # clock. A new independently admitted or multi-face measurement does.
        if result.pose is not None and (
            initial.pose is not None or not result.pose.ambiguous
        ):
            previous_evidence_time[map_id] = timestamp

    # Offline processing can use the next measured pose as well.  This repairs
    # the short ambiguous tail immediately before a reliable second face
    # appears, without interpolating any missing wrist measurement.
    next_world: dict[str, Pose] = {}
    next_time: dict[str, float] = {}
    for index in range(frame_count - 1, -1, -1):
        camera = camera_poses[index]
        map_id = map_ids[index]
        timestamp = float(timestamps_s[index])
        result = output[index]
        forward_measured = result.pose is not None and (
            initial_results[index].pose is not None or not result.pose.ambiguous
        )
        if camera is None or not metric_flags[index] or map_id is None:
            continue
        future_world = next_world.get(map_id)
        future_timestamp = next_time.get(map_id)
        recent = (
            future_world is not None
            and future_timestamp is not None
            and 0.0 < future_timestamp - timestamp <= maximum_prediction_gap_s
        )
        # Do not resurrect a single-tag pose that the forward metric pass
        # explicitly rejected. A truly missing initial measurement may still
        # be recovered from a reliable future pose.
        backward_candidate = (
            result.pose is None and initial_results[index].pose is None
            and not result.consensus_vetoed
            and not initial_results[index].consensus_vetoed
        )
        if recent and backward_candidate:
            visible = {
                marker_id: corners
                for marker_id, corners in detections[index].items()
                if marker_id in layout.markers
            }
            if visible:
                predicted = _camera_from_world_wrist(camera, future_world)
                resolved = optimize_tag_pose(
                    visible,
                    layout,
                    calibration,
                    predicted,
                    validate_planar_ambiguity=True,
                    assist_detections=assist_detections[index],
                )
                if resolved.pose is not None:
                    world_pose = _world_from_camera_wrist(camera, resolved.pose)
                    if not _implausible_single_marker_step(
                        world_pose,
                        future_world,
                        future_timestamp - timestamp,
                    ):
                        result = resolved
                        output[index] = result
                elif result.pose is not None and result.pose.ambiguous:
                    result = TagPoseResult(
                        None,
                        (),
                        tuple(sorted(visible)),
                        resolved.marker_errors_px or result.marker_errors_px,
                        resolved.graph_reprojection_error_px,
                        0.0,
                        True,
                    )
                    output[index] = result
        # A recovered frame is not a fresh temporal anchor. Otherwise each
        # backward step restarts the horizon and an arbitrarily long chain of
        # ambiguous observations can be admitted from one future measurement.
        if forward_measured:
            next_world[map_id] = _world_from_camera_wrist(camera, result.pose)
            next_time[map_id] = timestamp
    return output


class MarkerPoseTracker:
    """Fixed-marker localization with a short-lived measured pose prediction."""

    def __init__(self, layout: BandLayout, calibration: Calibration):
        self.layout = layout
        self.calibration = calibration
        self._previous: Pose | None = None
        self._last_reliable_time: float | None = None

    def update(self, detections: dict[int, np.ndarray], timestamp_s: float,
               marker_weights: dict[int, float] | None = None,
               assist_detections: dict[int, np.ndarray] | None = None) -> TagPoseResult:
        if not np.isfinite(timestamp_s):
            raise ValueError("marker timestamp must be finite")
        if (self._last_reliable_time is None
                or not 0.0 < timestamp_s-self._last_reliable_time <= .15):
            self._previous = None
        result = optimize_tag_pose(
            detections, self.layout, self.calibration, self._previous,
            max_graph_error_px=2.5, marker_weights=marker_weights,
            validate_planar_ambiguity=True,
            assist_detections=assist_detections,
        )
        if result.pose is not None and self._previous is not None:
            primary = {
                marker_id: corners
                for marker_id, corners in detections.items()
                if marker_id in self.layout.markers
                and (marker_weights is None or marker_weights.get(marker_id, 1.0) > 0.0)
            }
            auxiliary = {
                marker_id: corners
                for marker_id, corners in (assist_detections or {}).items()
                if marker_id in self.layout.markers and marker_id not in primary
            }
            observed = {**auxiliary, **primary}
            visible = tuple(sorted(
                marker_id
                for marker_id in observed
            ))
            if len(visible) >= 3:
                previous_center = (
                    -self._previous.rotation_matrix.T @ self._previous.tvec
                )
                result_center = -result.pose.rotation_matrix.T @ result.pose.tvec
                center_step = float(np.linalg.norm(result_center - previous_center))
                if center_step > 0.05:
                    previous_errors = _marker_errors(
                        self._previous, observed, self.layout,
                        self.calibration, visible,
                    )
                    result_errors = _marker_errors(
                        result.pose, observed, self.layout,
                        self.calibration, visible,
                    )
                    previous_support = sum(
                        previous_errors[marker_id] <= (
                            8.0 if marker_id in auxiliary else 5.0
                        )
                        for marker_id in visible
                    )
                    result_support = sum(
                        result_errors[marker_id] <= (
                            8.0 if marker_id in auxiliary else 5.0
                        )
                        for marker_id in visible
                    )
                    # A robust fit may explain a small marker subset perfectly
                    # while jumping to the wrong planar solution. Other decoded
                    # markers are useful negative evidence even when the fit
                    # rejected them. If the recent measured pose explains at
                    # least two more visible markers, reject this update and let
                    # SLAM publish (or report invalid); never freeze the old pose.
                    if previous_support >= 3 and previous_support >= result_support + 2:
                        return TagPoseResult(
                            None,
                            (),
                            visible,
                            result_errors,
                            result.graph_reprojection_error_px,
                            0.0,
                            True,
                        )
        # Weak-only or rejected observations cannot keep a prediction alive.
        if result.pose is not None and result.confidence >= .35:
            self._previous = result.pose
            self._last_reliable_time = timestamp_s
        return result


def _world_wrist_projection_residuals(
    world_from_wrist: Pose,
    world_from_camera: Pose,
    detections: dict[int, np.ndarray],
    layout: BandLayout,
    calibration: Calibration,
    marker_ids: tuple[int, ...],
) -> np.ndarray:
    camera_rotation = world_from_camera.rotation_matrix.T
    camera_from_wrist_rotation = camera_rotation @ world_from_wrist.rotation_matrix
    camera_from_wrist_translation = camera_rotation @ (
        world_from_wrist.tvec - world_from_camera.tvec
    )
    residuals: list[np.ndarray] = []
    for marker_id in marker_ids:
        projected, _ = cv2.projectPoints(
            layout.markers[marker_id],
            cv2.Rodrigues(camera_from_wrist_rotation)[0],
            camera_from_wrist_translation,
            calibration.camera_matrix,
            calibration.dist_coeffs,
        )
        residuals.append(
            (projected.reshape(4, 2) - detections[marker_id].reshape(4, 2)).reshape(-1)
        )
    return np.concatenate(residuals) if residuals else np.empty(0, dtype=np.float64)


def _temporal_acceleration_residuals(
    rotations: list[np.ndarray],
    translations: np.ndarray,
    timestamps_s: np.ndarray,
    reference_fps: float = 60.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Second-order motion in physical time, normalized to the old 60 FPS units."""
    if len(rotations) < 3:
        empty = np.empty((0, 3), dtype=np.float64)
        return empty, empty
    reference_dt = 1.0 / reference_fps
    translation_values: list[np.ndarray] = []
    rotation_values: list[np.ndarray] = []
    for index in range(len(rotations) - 2):
        first_dt = timestamps_s[index + 1] - timestamps_s[index]
        second_dt = timestamps_s[index + 2] - timestamps_s[index + 1]
        mean_dt = 0.5 * (first_dt + second_dt)
        first_velocity = (
            translations[index + 1] - translations[index]
        ) / first_dt
        second_velocity = (
            translations[index + 2] - translations[index + 1]
        ) / second_dt
        translation_values.append(
            (second_velocity - first_velocity) / mean_dt * reference_dt**2
        )
        first_step = cv2.Rodrigues(
            rotations[index].T @ rotations[index + 1]
        )[0].reshape(3) / first_dt
        second_step = cv2.Rodrigues(
            rotations[index + 1].T @ rotations[index + 2]
        )[0].reshape(3) / second_dt
        rotation_values.append(
            (second_step - first_step) / mean_dt * reference_dt**2
        )
    return np.asarray(translation_values), np.asarray(rotation_values)


def _huberized_residual(values: np.ndarray, delta: float) -> np.ndarray:
    """Transform residuals so linear least squares has Huber cost."""
    absolute = np.abs(values)
    transformed = np.where(
        absolute <= delta,
        absolute,
        np.sqrt(np.maximum(2.0 * delta * absolute - delta * delta, 0.0)),
    )
    return np.copysign(transformed, values)


def _optimize_wrist_window(
    frame_indices: list[int],
    initial_poses: list[Pose],
    camera_poses: list[Pose | None],
    detections: list[dict[int, np.ndarray]],
    accepted_ids: list[tuple[int, ...]],
    assist_ids: list[tuple[int, ...]],
    layout: BandLayout,
    calibration: Calibration,
    timestamps_s: np.ndarray,
) -> tuple[list[Pose], list[float]]:
    pose_count = len(frame_indices)
    initial_state = np.concatenate([_pose_vector(pose) for pose in initial_poses])
    projection_ids = [
        accepted_ids[index] + assist_ids[index] for index in frame_indices
    ]
    projection_sizes = [8 * len(projection_ids[index]) for index in frame_indices]
    # One planar marker observes lateral motion well but has substantially
    # weaker depth/tilt conditioning than two faces of the wrist constellation.
    # Keep it usable, while letting neighbouring multi-marker frames and the
    # physical-time motion prior resolve that ambiguity.
    projection_sigmas = []
    for local_index, index in enumerate(frame_indices):
        ids = projection_ids[index]
        maximum_area = max(_marker_area(detections[index][marker_id]) for marker_id in ids)
        camera_pose = camera_poses[index]
        assert camera_pose is not None
        camera_from_wrist = _camera_from_world_wrist(
            camera_pose, initial_poses[local_index]
        )
        single_face_penalty = 2.0 if len(accepted_ids[index]) == 1 else 1.0
        strong_sigmas = [
            single_face_penalty * _marker_sigma_px(
                layout.markers[marker_id],
                detections[index][marker_id],
                camera_from_wrist,
                maximum_area,
            ) if len(accepted_ids[index]) <= 2 else single_face_penalty
            for marker_id in accepted_ids[index]
        ]
        # Keep the same area/view-angle conditioning used by the per-frame
        # pose solver.  Previously this stage silently restored equal weight,
        # so a tiny newly visible face could drag the whole temporal window.
        assist_sigmas = [
            max(
                np.sqrt(20.0),
                _marker_sigma_px(
                    layout.markers[marker_id],
                    detections[index][marker_id],
                    camera_from_wrist,
                    maximum_area,
                ) * np.sqrt(20.0),
            )
            for marker_id in assist_ids[index]
        ]
        projection_sigmas.append(np.repeat(strong_sigmas + assist_sigmas, 4))
    camera_rotations: list[np.ndarray] = []
    camera_translations: list[np.ndarray] = []
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    for frame_index in frame_indices:
        camera_pose = camera_poses[frame_index]
        assert camera_pose is not None
        camera_rotations.append(camera_pose.rotation_matrix.T)
        camera_translations.append(camera_pose.tvec.reshape(3))
        ids = projection_ids[frame_index]
        object_points.append(
            np.concatenate([layout.markers[marker_id] for marker_id in ids])
        )
        image_points.append(
            np.concatenate(
                [detections[frame_index][marker_id].reshape(4, 2) for marker_id in ids]
            )
        )
    acceleration_factors = max(0, pose_count - 2)
    residual_count = sum(projection_sizes) + 6 * acceleration_factors
    sparsity = lil_matrix((residual_count, 6 * pose_count), dtype=np.int8)
    row = 0
    for local_index, size in enumerate(projection_sizes):
        sparsity[row : row + size, 6 * local_index : 6 * local_index + 6] = 1
        row += size
    for local_index in range(acceleration_factors):
        sparsity[
            row : row + 6,
            6 * local_index : 6 * local_index + 18,
        ] = 1
        row += 6

    def residual(value: np.ndarray) -> np.ndarray:
        vectors = value.reshape(pose_count, 6)
        rotations = [cv2.Rodrigues(vector[:3])[0] for vector in vectors]
        translations = vectors[:, 3:]
        output: list[np.ndarray] = []
        for local_index in range(pose_count):
            camera_rotation = camera_rotations[local_index]
            camera_from_wrist_rotation = (
                camera_rotation @ rotations[local_index]
            )
            camera_from_wrist_translation = camera_rotation @ (
                translations[local_index] - camera_translations[local_index]
            )
            projected, _ = cv2.projectPoints(
                object_points[local_index],
                cv2.Rodrigues(camera_from_wrist_rotation)[0],
                camera_from_wrist_translation,
                calibration.camera_matrix,
                calibration.dist_coeffs,
            )
            projection_residual = (
                (projected.reshape(-1, 2) - image_points[local_index]) /
                projection_sigmas[local_index][:, None]
            ).reshape(-1)
            output.append(_huberized_residual(projection_residual, 10.0))
        translation_accelerations, rotation_accelerations = (
            _temporal_acceleration_residuals(
                rotations, translations, timestamps_s, reference_fps=60.0
            )
        )
        for translation_acceleration, rotation_acceleration in zip(
            translation_accelerations, rotation_accelerations
        ):
            output.append(translation_acceleration / 0.0008)
            output.append(rotation_acceleration / 0.008)
        return np.concatenate(output)

    optimized = least_squares(
        residual,
        initial_state,
        jac_sparsity=sparsity.tocsr(),
        # Pixel outliers are robustified above. Keep physical acceleration
        # quadratic so a bad planar depth estimate cannot evade the motion
        # model by crossing the global Huber threshold.
        loss="linear",
        max_nfev=80,
    )
    poses = [
        _vector_pose(optimized.x[6 * index : 6 * index + 6])
        for index in range(pose_count)
    ]
    errors: list[float] = []
    for pose, frame_index in zip(poses, frame_indices):
        camera_pose = camera_poses[frame_index]
        assert camera_pose is not None
        values = _world_wrist_projection_residuals(
            pose,
            camera_pose,
            detections[frame_index],
            layout,
            calibration,
            accepted_ids[frame_index],
        )
        errors.append(float(np.sqrt(np.mean(values * values))))
    return poses, errors


def _blend_pose(first: Pose, second: Pose, alpha: float) -> Pose:
    relative = first.rotation_matrix.T @ second.rotation_matrix
    rotation_step = cv2.Rodrigues(relative)[0] * alpha
    rotation = first.rotation_matrix @ cv2.Rodrigues(rotation_step)[0]
    translation = (1.0 - alpha) * first.tvec + alpha * second.tvec
    return Pose(cv2.Rodrigues(rotation)[0], translation, 0.0)


def optimize_wrist_trajectory(
    camera_poses: list[Pose | None],
    camera_from_wrist: list[Pose | None],
    detections: list[dict[int, np.ndarray]],
    accepted_ids: list[tuple[int, ...]],
    layout: BandLayout,
    calibration: Calibration,
    fps: float = 60.0,
    timestamps_s: list[float] | np.ndarray | None = None,
    maximum_window_s: float = 1.5,
    overlap_s: float = 0.15,
    parallel_workers: int = 4,
    assist_detections: list[dict[int, np.ndarray]] | None = None,
) -> WristTrajectoryResult:
    frame_count = len(camera_poses)
    if not (
        len(camera_from_wrist)
        == len(detections)
        == len(accepted_ids)
        == frame_count
    ):
        raise ValueError("wrist trajectory inputs must have the same length")
    if assist_detections is None:
        assist_detections = [{} for _ in range(frame_count)]
    if len(assist_detections) != frame_count:
        raise ValueError("wrist assist detections must match the frame count")
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError("wrist trajectory fps must be positive and finite")
    if not np.isfinite(maximum_window_s) or maximum_window_s <= 0.0:
        raise ValueError("wrist trajectory window duration must be positive")
    if not np.isfinite(overlap_s) or not 0.0 <= overlap_s < maximum_window_s:
        raise ValueError("wrist trajectory overlap must be shorter than its window")
    if not 1 <= parallel_workers <= 8:
        raise ValueError("wrist trajectory parallel_workers must be between 1 and 8")
    timestamps = (
        np.arange(frame_count, dtype=np.float64) / fps
        if timestamps_s is None
        else np.asarray(timestamps_s, dtype=np.float64)
    )
    if timestamps.shape != (frame_count,) or not np.all(np.isfinite(timestamps)):
        raise ValueError("wrist trajectory timestamps must match the frame count")
    if frame_count > 1 and np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("wrist trajectory timestamps must be strictly increasing")
    initial: list[Pose | None] = []
    for camera_pose, wrist_pose, ids in zip(
        camera_poses, camera_from_wrist, accepted_ids
    ):
        if camera_pose is None or wrist_pose is None or not ids:
            initial.append(None)
            continue
        rotation = camera_pose.rotation_matrix @ wrist_pose.rotation_matrix
        translation = (
            camera_pose.rotation_matrix @ wrist_pose.tvec + camera_pose.tvec
        )
        initial.append(
            Pose(
                cv2.Rodrigues(rotation)[0],
                translation,
                wrist_pose.reprojection_error_px,
                wrist_pose.marker_ids,
                wrist_pose.inlier_count,
                wrist_pose.ambiguous,
            )
        )

    output: list[Pose | None] = [None] * frame_count
    output_errors: list[float | None] = [None] * frame_count
    valid_indices = [index for index, pose in enumerate(initial) if pose is not None]
    segments: list[list[int]] = []
    for index in valid_indices:
        # A one- or two-frame marker miss is not a new physical motion
        # segment. Keep nearby *measured* poses in one optimization problem so
        # the single-face depth nullspace cannot restart as arbitrary constant
        # velocity after every miss. Missing frames remain None in the output.
        if (
            not segments
            or timestamps[index] - timestamps[segments[-1][-1]] > 0.15
        ):
            segments.append([index])
        else:
            segments[-1].append(index)

    window_groups: list[list[list[int]]] = []
    jobs = []
    for segment in segments:
        if len(segment) < 3:
            # A decoded planar marker still supplies identity and image
            # corners, but one or two isolated frames do not provide enough
            # evidence for a new 6-DoF wrist segment. Multi-face geometry is
            # independently constrained and remains publishable.
            if all(len(accepted_ids[index]) == 1 for index in segment):
                continue
            for frame_index in segment:
                output[frame_index] = initial[frame_index]
            continue
        windows: list[list[int]] = []
        start = 0
        while start < len(segment):
            segment_times = timestamps[segment]
            stop = int(np.searchsorted(
                segment_times,
                segment_times[start] + maximum_window_s,
                side="left",
            ))
            stop = min(len(segment), max(stop, start + 3))
            window = segment[start:stop]
            if len(window) < 3:
                break
            windows.append(window)
            if stop == len(segment):
                break
            next_time = segment_times[stop - 1] - overlap_s
            next_start = int(np.searchsorted(
                segment_times, np.nextafter(next_time, -np.inf), side="left"
            ))
            start = max(start + 1, min(next_start, stop - 1))

        window_groups.append(windows)
        for window in windows:
            window_initial = [initial[index] for index in window]
            assert all(pose is not None for pose in window_initial)
            window_detections = []
            window_assist_ids = []
            for index in window:
                weak = {
                    marker_id: corners
                    for marker_id, corners in assist_detections[index].items()
                    if marker_id in layout.markers
                    and marker_id not in accepted_ids[index]
                }
                window_detections.append({**detections[index], **weak})
                window_assist_ids.append(tuple(sorted(weak)))
            jobs.append((
                list(range(len(window))),
                [pose for pose in window_initial if pose is not None],
                [camera_poses[index] for index in window],
                window_detections,
                [accepted_ids[index] for index in window],
                window_assist_ids,
                layout,
                calibration,
                timestamps[window],
            ))

    if len(jobs) > 1 and parallel_workers > 1:
        with ProcessPoolExecutor(max_workers=min(parallel_workers, len(jobs))) as executor:
            futures = [executor.submit(_optimize_wrist_window, *job) for job in jobs]
            solved = [future.result() for future in futures]
    else:
        solved = [_optimize_wrist_window(*job) for job in jobs]

    solved_index = 0
    for windows in window_groups:
        group_results = solved[solved_index : solved_index + len(windows)]
        solved_index += len(windows)
        for window, (optimized, errors) in zip(windows, group_results):
            overlap_indices = [
                local_index
                for local_index, frame_index in enumerate(window)
                if output[frame_index] is not None
            ]
            overlap_rank = {local_index: rank for rank, local_index in enumerate(overlap_indices)}
            for local_index, (frame_index, pose, error) in enumerate(
                zip(window, optimized, errors)
            ):
                previous = output[frame_index]
                if previous is not None:
                    alpha = (overlap_rank[local_index] + 1) / (len(overlap_indices) + 1)
                    pose = _blend_pose(previous, pose, alpha)
                    previous_error = output_errors[frame_index]
                    if previous_error is not None:
                        error = (1.0 - alpha) * previous_error + alpha * error
                pose.reprojection_error_px = error
                pose.marker_ids = accepted_ids[frame_index]
                pose.inlier_count = 4 * len(accepted_ids[frame_index])
                pose.ambiguous = len(accepted_ids[frame_index]) == 1
                output[frame_index] = pose
                output_errors[frame_index] = error
    return WristTrajectoryResult(output, output_errors)
