from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .bandsolve import solve_band_pose
from .detector import ArucoDetector
from .models import BandLayout, Calibration, Pose
from .marker_quality import MarkerBoundaryQuality
from .pose import solve_square_pose
from .tag_graph import optimize_tag_pose
from .tracks import AdaptivePoseSmoother, PoseSmoother, WorldPoseSmoother


@dataclass
class FrameResult:
    detections: dict[int, np.ndarray]
    poses: dict[str, Pose]
    recovered_ids: tuple[int, ...] = ()
    tracked_ids: tuple[int, ...] = ()
    raw_poses: dict[str, Pose] = field(default_factory=dict)
    world_poses: dict[str, Pose] = field(default_factory=dict)
    raw_world_poses: dict[str, Pose] = field(default_factory=dict)
    world_reference: Pose | None = None
    camera_world_pose: Pose | None = None
    rejected_detections: dict[int, np.ndarray] = field(default_factory=dict)
    boundary_quality: dict[int, MarkerBoundaryQuality] = field(default_factory=dict)


def inverse_pose(pose: Pose) -> Pose:
    rotation = cv2.Rodrigues(pose.rvec)[0]
    inverse_rotation = rotation.T
    inverse_translation = -inverse_rotation @ pose.tvec
    return Pose(
        cv2.Rodrigues(inverse_rotation)[0],
        inverse_translation,
        pose.reprojection_error_px,
        pose.marker_ids,
        pose.inlier_count,
        pose.ambiguous,
    )


def relative_pose(reference: Pose, target: Pose) -> Pose:
    camera_from_reference = cv2.Rodrigues(reference.rvec)[0]
    camera_from_target = cv2.Rodrigues(target.rvec)[0]
    reference_from_target = camera_from_reference.T @ camera_from_target
    translation = camera_from_reference.T @ (target.tvec - reference.tvec)
    return Pose(
        cv2.Rodrigues(reference_from_target)[0],
        translation,
        float(np.hypot(reference.reprojection_error_px, target.reprojection_error_px)),
        target.marker_ids,
        target.inlier_count,
        reference.ambiguous or target.ambiguous,
    )


def compose_pose(reference: Pose, relative: Pose) -> Pose:
    camera_from_reference = reference.rotation_matrix
    reference_from_target = relative.rotation_matrix
    return Pose(
        cv2.Rodrigues(camera_from_reference @ reference_from_target)[0],
        camera_from_reference @ relative.tvec + reference.tvec,
        float(np.hypot(reference.reprojection_error_px, relative.reprojection_error_px)),
        relative.marker_ids,
        relative.inlier_count,
        reference.ambiguous or relative.ambiguous,
    )


def marker_pose_from_band(
    points: np.ndarray, band_pose: Pose, origin_band_pose: Pose | None = None
) -> Pose:
    center = np.mean(points, axis=0).reshape(3, 1)
    x_axis = points[1] - points[0]
    x_axis /= np.linalg.norm(x_axis)
    y_axis = points[0] - points[3]
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)
    band_from_marker = np.column_stack((x_axis, y_axis, z_axis))
    camera_from_band = cv2.Rodrigues(band_pose.rvec)[0]
    camera_from_marker = camera_from_band @ band_from_marker
    origin_band_pose = origin_band_pose or band_pose
    origin_camera_from_band = cv2.Rodrigues(origin_band_pose.rvec)[0]
    tvec = origin_camera_from_band @ center + origin_band_pose.tvec
    return Pose(
        cv2.Rodrigues(camera_from_marker)[0],
        tvec,
        band_pose.reprojection_error_px,
        band_pose.marker_ids,
        band_pose.inlier_count,
        band_pose.ambiguous,
    )


class TrackingPipeline:
    def __init__(
        self,
        calibration: Calibration,
        bands: list[BandLayout] | None = None,
        singles: dict[int, float] | None = None,
        sharpen: bool = False,
        show_band_marker_axes: bool = False,
        refine_markers: bool = True,
        track_marker_gaps: int = 2,
        adaptive_smoothing: bool = True,
        world_board: BandLayout | None = None,
        boundary_marker_ids: set[int] | None = None,
        allow_soft_marker_corners: bool = True,
    ):
        self.calibration = calibration
        self.bands = bands or []
        self._band_names = {band.name for band in self.bands}
        self.singles = singles or {}
        self.world_board = world_board
        self.show_band_marker_axes = show_band_marker_axes
        self.adaptive_smoothing = adaptive_smoothing
        layouts = self.bands + ([world_board] if world_board is not None else [])
        dictionaries = {layout.dictionary for layout in layouts}
        if len(dictionaries) > 1:
            raise ValueError("all configured layouts must use the same ArUco dictionary")
        marker_sets = [set(layout.markers) for layout in layouts]
        if len(set().union(*marker_sets)) != sum(map(len, marker_sets)):
            raise ValueError("marker IDs must not overlap between hands and world board")
        dictionary = next(iter(dictionaries), "DICT_4X4_50")
        if boundary_marker_ids is None:
            boundary_marker_ids = set(world_board.markers) if world_board is not None else set()
        self.detector = ArucoDetector(
            dictionary,
            sharpen=sharpen,
            board_markers=[layout.markers for layout in layouts] if refine_markers else None,
            camera_matrix=calibration.camera_matrix,
            dist_coeffs=calibration.dist_coeffs,
            track_marker_gaps=track_marker_gaps,
            validate_corners=bool(boundary_marker_ids),
            boundary_marker_ids=boundary_marker_ids,
            allow_soft_marker_corners=allow_soft_marker_corners,
            wrist_marker_ids={mid for band in self.bands for mid in band.markers},
        )
        self._raw_poses: dict[str, Pose] = {}
        self._smoothers: dict[str, PoseSmoother] = {}

    def process(self, frame: np.ndarray) -> FrameResult:
        detections = self.detector.detect(frame)
        poses: dict[str, Pose] = {}
        raw_poses: dict[str, Pose] = {}
        world_poses: dict[str, Pose] = {}
        raw_world_poses: dict[str, Pose] = {}
        world_reference = None
        camera_world_pose = None
        if self.world_board is not None:
            world_reference = solve_band_pose(
                detections,
                self.world_board,
                self.calibration.camera_matrix,
                self.calibration.dist_coeffs,
                self._raw_poses.get("@world"),
                validate_planar_ambiguity=True,
            )
            weights = {mid: quality.information_weight
                       for mid, quality in self.detector.last_boundary_quality.items()}
            if any(0.0 < weight < 1.0 for weight in weights.values()):
                world_reference = optimize_tag_pose(
                    detections, self.world_board, self.calibration,
                    self._raw_poses.get("@world"), marker_weights=weights,
                    validate_planar_ambiguity=True,
                ).pose
            if world_reference is not None:
                self._raw_poses["@world"] = world_reference
                camera_world_pose = inverse_pose(world_reference)
            else:
                self._raw_poses.pop("@world", None)
        for band in self.bands:
            pose = solve_band_pose(
                detections,
                band,
                self.calibration.camera_matrix,
                self.calibration.dist_coeffs,
                self._raw_poses.get(band.name),
            )
            if pose is not None:
                raw_poses[band.name] = pose
                smoothed = self._smooth(band.name, pose)
                poses[band.name] = Pose(
                    smoothed.rvec,
                    smoothed.tvec,
                    pose.reprojection_error_px,
                    pose.marker_ids,
                    pose.inlier_count,
                    pose.ambiguous,
                )
                self._raw_poses[band.name] = pose
                if world_reference is not None:
                    raw_world_pose = relative_pose(world_reference, pose)
                    raw_world_poses[band.name] = raw_world_pose
                    world_poses[band.name] = self._smooth(
                        f"world/{band.name}", raw_world_pose
                    )
                if self.show_band_marker_axes:
                    for marker_id in sorted(set(detections).intersection(band.markers)):
                        poses[f"marker-{marker_id}"] = marker_pose_from_band(
                            band.markers[marker_id], smoothed, pose
                        )
            else:
                self._raw_poses.pop(band.name, None)
                self._smoothers.pop(band.name, None)
                self._smoothers.pop(f"world/{band.name}", None)
        for marker_id, size_m in self.singles.items():
            if marker_id not in detections:
                continue
            name = f"marker-{marker_id}"
            pose = solve_square_pose(
                detections[marker_id],
                size_m,
                self.calibration.camera_matrix,
                self.calibration.dist_coeffs,
                self._raw_poses.get(name),
            )
            if pose is not None:
                pose.marker_ids = (marker_id,)
                poses[name] = self._smooth(name, pose)
                self._raw_poses[name] = pose
        return FrameResult(
            detections=detections,
            poses=poses,
            recovered_ids=self.detector.last_recovered_ids,
            tracked_ids=self.detector.last_tracked_ids,
            raw_poses=raw_poses,
            world_poses=world_poses,
            raw_world_poses=raw_world_poses,
            world_reference=world_reference,
            camera_world_pose=camera_world_pose,
            rejected_detections=self.detector.last_rejected_detections,
            boundary_quality=self.detector.last_boundary_quality,
        )

    def _smooth(self, name: str, pose: Pose) -> Pose:
        smoother = self._smoothers.get(name)
        if smoother is None:
            if name.startswith("world/"):
                smoother = (
                    WorldPoseSmoother()
                    if self.adaptive_smoothing
                    else PoseSmoother(translation_alpha=0.18, rotation_alpha=0.12)
                )
            elif name in self._band_names:
                smoother = (
                    AdaptivePoseSmoother()
                    if self.adaptive_smoothing
                    else PoseSmoother(translation_alpha=0.18, rotation_alpha=0.12)
                )
            else:
                smoother = PoseSmoother()
            self._smoothers[name] = smoother
        return smoother.update(pose)
