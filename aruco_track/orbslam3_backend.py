from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import gzip
import io
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
import zstandard as zstd
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .camera_state import FusedCameraFrame
from .models import BandLayout, Calibration, Pose
from .marker_corners import TrackedMarkerObservation


MIN_MARKER_CONFIDENCE = 0.35
MAX_MARKER_ERROR_PX = 2.5


@dataclass(frozen=True)
class OrbSlamObservation:
    inliers: int
    tracked_points: np.ndarray
    rejected_points: np.ndarray
    state: int
    map_point_ids: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )


@dataclass(frozen=True)
class MetricOrbSlamResult:
    frames: list[FusedCameraFrame]
    world_points: np.ndarray
    keyframe_poses: tuple[Pose, ...]
    observations: list[OrbSlamObservation | None]
    initialization_frame: int | None
    scale_m_per_slam_unit: float | None
    anchor_count: int
    median_anchor_position_error_m: float | None
    median_anchor_rotation_error_deg: float | None
    timing: dict[str, float]
    history: list[dict] = field(default_factory=list)
    maps: dict[str, dict] = field(default_factory=dict)
    offline_marker_bootstrap: dict | None = None


def first_reliable_marker_frame(
    marker_poses: list[Pose | None], marker_confidences: list[float]
) -> int | None:
    for index, (pose, confidence) in enumerate(
        zip(marker_poses, marker_confidences)
    ):
        if (
            pose is not None
            and confidence >= MIN_MARKER_CONFIDENCE
            and pose.reprojection_error_px <= MAX_MARKER_ERROR_PX
        ):
            return index
    return None


def write_orbslam3_settings(
    path: Path, calibration: Calibration, fps: float,
    slam_init: str = "auto", load_atlas: Path | None = None,
    save_atlas: Path | None = None,
    dynamic_geometry: bool = False,
    rigid_marker_layout: bool = False,
) -> None:
    distortion = np.pad(
        calibration.dist_coeffs.reshape(-1),
        (0, max(0, 5 - calibration.dist_coeffs.size)),
    )
    camera = calibration.camera_matrix
    width, height = calibration.image_size
    path.write_text(
        "%YAML:1.0\n\n"
        'File.version: "1.0"\n'
        'Camera.type: "PinHole"\n'
        f"Camera1.fx: {camera[0, 0]:.12f}\n"
        f"Camera1.fy: {camera[1, 1]:.12f}\n"
        f"Camera1.cx: {camera[0, 2]:.12f}\n"
        f"Camera1.cy: {camera[1, 2]:.12f}\n"
        f"Camera1.k1: {distortion[0]:.12f}\n"
        f"Camera1.k2: {distortion[1]:.12f}\n"
        f"Camera1.p1: {distortion[2]:.12f}\n"
        f"Camera1.p2: {distortion[3]:.12f}\n"
        f"Camera1.k3: {distortion[4]:.12f}\n"
        f"Camera.fps: {max(1, round(fps))}\n"
        "Camera.RGB: 0\n"
        f"Camera.width: {width}\n"
        f"Camera.height: {height}\n"
        "ORBextractor.nFeatures: 2000\n"
        "ORBextractor.scaleFactor: 1.2\n"
        "ORBextractor.nLevels: 8\n"
        "ORBextractor.iniThFAST: 20\n"
        "ORBextractor.minThFAST: 7\n"
        f"ORBextractor.dynamicGeometry: {int(dynamic_geometry)}\n"
        "Offline.synchronousMapping: 1\n"
        "Viewer.KeyFrameSize: 0.05\n"
        "Viewer.KeyFrameLineWidth: 1.0\n"
        "Viewer.GraphLineWidth: 0.9\n"
        "Viewer.PointSize: 2.0\n"
        "Viewer.CameraSize: 0.08\n"
        "Viewer.CameraLineWidth: 3.0\n"
        "Viewer.ViewpointX: 0.0\n"
        "Viewer.ViewpointY: -0.7\n"
        "Viewer.ViewpointZ: -1.8\n"
        "Viewer.ViewpointF: 500.0\n"
        f"TagFusion.markerOnlyInitialization: {int(slam_init == 'marker')}\n"
        "TagFusion.enabled: 1\n"
        f"TagFusion.rigidMarkerLayout: {int(rigid_marker_layout)}\n"
        "TagFusion.minimumScaleBaselineM: 0.04\n"
        "TagFusion.minimumKeyFrameIntervalS: 0.35\n"
        "TagFusion.minimumTranslationM: 0.03\n"
        "TagFusion.minimumRotationDeg: 7.0\n"
        "TagFusion.minimumTrackedRatio: 0.50\n"
        "TagFusion.poseWeight: 0.85\n"
        "TagFusion.maxAlignmentPositionResidualM: 0.03\n"
        "TagFusion.maxAlignmentRotationResidualDeg: 10.0\n"
        + (f"System.LoadAtlasFromFile: {json.dumps(str(load_atlas.resolve()))}\n" if load_atlas else "")
        + (f"System.SaveAtlasToFile: {json.dumps(str(save_atlas.resolve()))}\n" if save_atlas else "")
    )


def write_tag_observation_hints(
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
        tracked = tracked_observations[frame_index] if tracked_observations else None
        partial_only, tracked_count, track_age = False, 0, 0.
        if not full_valid:
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
        if not full_valid and not (partial_only and len(point_pairs)>=3 and confidence>=.15 and track_age<=.30):
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
        lines.append(" ".join(values))
    path.write_text("\n".join(lines) + "\n")


def run_orbslam3_sequence(
    project_dir: Path,
    sequence_dir: Path,
    settings_path: Path,
    output_dir: Path,
    tag_observations_path: Path | None = None,
    compact_history: bool = False,
    environment_overrides: dict[str, str] | None = None,
) -> tuple[
    dict[int, Pose],
    dict[int, Pose],
    np.ndarray,
    dict[int, OrbSlamObservation],
    dict[str, float],
]:
    root = project_dir / "third_party" / "ORB_SLAM3"
    binary = root / "Examples" / "Monocular" / "mono_tum_headless"
    vocabulary = root / "Vocabulary" / "ORBvoc.txt"
    if not binary.exists():
        raise RuntimeError(
            f"official ORB-SLAM3 runner is missing: {binary}; build mono_tum_headless"
        )
    frames_path = output_dir / "frames.txt"
    keyframes_path = output_dir / "keyframes.txt"
    points_path = output_dir / "points.xyz"
    observations_path = output_dir / "observations.txt"
    timing_path = output_dir / "timing.txt"
    library_paths = [
        project_dir / "third_party" / "opencv-4.10-install" / "lib",
        project_dir / "third_party" / "Pangolin" / "install" / "lib",
        root / "lib",
        root / "Thirdparty" / "DBoW2" / "lib",
        root / "Thirdparty" / "g2o" / "lib",
        Path("/opt/homebrew/lib"),
        Path("/opt/homebrew/opt/openssl@3/lib"),
    ]
    environment = os.environ.copy()
    library_key = 'DYLD_LIBRARY_PATH' if sys.platform == 'darwin' else 'LD_LIBRARY_PATH'
    environment[library_key] = ":".join(
        [str(path) for path in library_paths if path.is_dir()]
        + ([environment[library_key]] if environment.get(library_key) else []))
    # Bounded LK-to-ORB/PnP rescue is enabled in the offline export workflow.
    # Respect an explicit zero for ablations; this does not enable the separate
    # temporal-flow frontend or replace native geometric validation.
    environment.setdefault("ORB_SLAM3_FLOW_RECOVERY", "1")
    # OpenCV's robust estimators are seeded by the offline runner.  Keep the
    # BLAS/OpenMP execution width fixed as well so repeated processing of the
    # same immutable frame cache does not acquire avoidable scheduling noise.
    environment["OMP_NUM_THREADS"] = "1"
    environment["VECLIB_MAXIMUM_THREADS"] = "1"
    if compact_history:
        environment["ORB_SLAM3_COMPACT_HISTORY"] = "1"
    if environment_overrides:
        environment.update(environment_overrides)
    command = [
            str(binary),
            str(vocabulary),
            str(settings_path),
            str(sequence_dir),
            str(frames_path),
            str(keyframes_path),
            str(points_path),
            str(observations_path),
            str(timing_path),
        ]
    if tag_observations_path is not None:
        command.append(str(tag_observations_path))
    completed = subprocess.run(
        command,
        cwd=project_dir,
        env=environment,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60 + 2 * len((sequence_dir / "rgb.txt").read_text().splitlines()),
    )
    (output_dir / "native.log").write_text(completed.stdout)
    if completed.returncode != 0:
        tail = "\n".join(completed.stdout.splitlines()[-30:])
        raise RuntimeError(
            f"official ORB-SLAM3 failed with exit code {completed.returncode}:\n{tail}"
        )
    return (
        _read_tum_poses(frames_path),
        _read_tum_poses(keyframes_path),
        _read_points(points_path),
        _read_observations(observations_path),
        _read_timing(timing_path),
    )


def _timestamp_to_frame(timestamp: float, fps: float) -> int:
    return int(round(timestamp * fps))


def _read_tum_poses(path: Path, fps: float | None = None) -> dict[float | int, Pose]:
    poses: dict[float | int, Pose] = {}
    if not path.exists():
        return poses
    for line in path.read_text().splitlines():
        values = line.split()
        if len(values) != 8:
            continue
        timestamp, tx, ty, tz, qx, qy, qz, qw = map(float, values)
        rotation = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        key = _timestamp_to_frame(timestamp, fps) if fps is not None else timestamp
        poses[key] = Pose(
            cv2.Rodrigues(rotation)[0],
            np.array([[tx], [ty], [tz]], dtype=np.float64),
            0.0,
        )
    return poses


def remap_timestamp_poses(poses: dict[float, Pose], fps: float) -> dict[int, Pose]:
    return {_timestamp_to_frame(timestamp, fps): pose for timestamp, pose in poses.items()}


def _read_points(path: Path) -> np.ndarray:
    if not path.exists() or path.stat().st_size == 0:
        return np.empty((0, 3), dtype=np.float64)
    points = np.loadtxt(path, dtype=np.float64)
    return np.asarray(points, dtype=np.float64).reshape(-1, 3)


def _read_observations(path: Path) -> dict[float, OrbSlamObservation]:
    observations: dict[float, OrbSlamObservation] = {}
    if not path.exists():
        return observations
    for line in path.read_text().splitlines():
        values = line.split()
        if len(values) < 3:
            continue
        timestamp = float(values[0])
        state = int(values[1])
        count = int(values[2])
        payload = values[3:]
        if len(payload) == 3 * count:
            ids = np.asarray(payload[0::3], dtype=np.int64)
            points = np.column_stack((
                np.asarray(payload[1::3], dtype=np.float64),
                np.asarray(payload[2::3], dtype=np.float64),
            ))
        elif len(payload) == 2 * count:
            # Backward compatibility with milestones written before matched
            # MapPoint identities were exported.
            ids = np.empty(0, dtype=np.int64)
            points = np.asarray(payload, dtype=np.float64).reshape(-1, 2)
        else:
            continue
        observations[timestamp] = OrbSlamObservation(
            min(count, len(points)), points, np.empty((0, 2), dtype=np.float64), state, ids
        )
    return observations


def remap_timestamp_observations(
    observations: dict[float, OrbSlamObservation], fps: float
) -> dict[int, OrbSlamObservation]:
    return {
        _timestamp_to_frame(timestamp, fps): observation
        for timestamp, observation in observations.items()
    }


def _read_timing(path: Path) -> dict[str, float]:
    timing: dict[str, float] = {}
    if not path.exists():
        return timing
    for line in path.read_text().splitlines():
        values = line.split()
        if len(values) == 2:
            timing[values[0]] = float(values[1])
    return timing


def pose_from_native(values: list[float] | None) -> Pose | None:
    if values is None:
        return None
    values = np.asarray(values, dtype=float)
    if values.shape != (7,) or not np.all(np.isfinite(values)):
        raise ValueError("invalid native SE3")
    return Pose(cv2.Rodrigues(Rotation.from_quat(values[3:]).as_matrix())[0],
                values[:3].reshape(3, 1), 0.0)


def _apply_marker_graph_delta(pose: Pose, captured, committed, *, rigid_gauge=False) -> Pose | None:
    """Apply G_committed * inverse(G_captured) from an explicit native stamp.

    Stamps contain [sequence, scale, tx,ty,tz,qx,qy,qz,qw]. New marker-world
    gauges are rigid (both scales must be one), and change only on a committed
    source-to-target merge. Local Sim(3) graph stamps remain readable solely
    for legacy graph-only logs; they must not rescale a new metric marker pose.
    """
    try:
        captured, committed = np.asarray(captured, float), np.asarray(committed, float)
        if (captured.shape != (9,) or committed.shape != (9,)
                or not np.all(np.isfinite(captured)) or not np.all(np.isfinite(committed))
                or captured[1] <= 0.0 or committed[1] <= 0.0
                or captured[0] < 0 or committed[0] < captured[0]):
            return None
        if rigid_gauge and (captured[1] != 1.0 or committed[1] != 1.0):
            return None
        old_rotation = Rotation.from_quat(captured[5:]).as_matrix()
        new_rotation = Rotation.from_quat(committed[5:]).as_matrix()
    except (TypeError, ValueError):
        return None
    scale = committed[1] / captured[1]
    rotation = new_rotation @ old_rotation.T
    translation = committed[2:5] - scale * rotation @ captured[2:5]
    return Pose(cv2.Rodrigues(rotation @ pose.rotation_matrix)[0],
                scale * rotation @ pose.tvec + translation.reshape(3, 1), 0.0)


def camera_at_revision(
    frame: dict, revision: dict, references: dict | None = None
) -> tuple[Pose | None, str | None]:
    """Resolve a measured frame against a committed revision; never fill LOST.

    Metric tag observations retain their measured coordinates during local
    scale re-anchoring. Their independent rigid correction chain follows map
    merge and joint marker-pose BA without inheriting monocular scale changes.
    Tag-free visual frames follow reference poses and local depth units.
    """
    if frame["pose"] is None or frame["state"] not in (2, 6):
        return None, None
    reference = frame.get("reference")
    if references is None:
        references = {value[0]: value for value in revision.get("references", [])}
    mapping = next((m for m in frame.get("maps", []) if m["id"] == frame["active_map"]), {})
    reference_map = references.get(reference, [None, frame["active_map"]])[1]
    same_publication = frame is revision or (frame.get("timestamp") == revision.get("timestamp") and frame == revision)
    if frame.get("tag_anchored", frame["state"] == 6) and mapping.get("metric"):
        if same_publication:
            return pose_from_native(frame["pose"]), f"atlas_{frame['active_map']}"
        gauge_protocol = ("reference_marker_gauge" in frame or "reference_marker_gauge" in revision
                          or any(len(value) > 5 for value in references.values()))
        if gauge_protocol:
            value = references.get(reference)
            if value is None or len(value) < 6:
                return None, None
            # Missing/invalid new-protocol gauges must not silently fall back
            # to the visual Sim(3), even when a legacy graph stamp is present.
            pose = _apply_marker_graph_delta(pose_from_native(frame["pose"]),
                                             frame.get("reference_marker_gauge"), value[5], rigid_gauge=True)
            return (pose, f"atlas_{reference_map}") if pose is not None else (None, None)
        if "reference_marker_graph" in frame:
            # Compatibility with previously recorded graph-only publications;
            # newly written logs use the independent rigid-gauge branch above.
            value = references.get(reference)
            if value is None or len(value) < 5:
                # The new protocol promises an explicit correction chain.
                # An unresolved historical reference is not evidence that its
                # old absolute coordinates survived a merge/re-anchoring.
                return None, None
            pose = _apply_marker_graph_delta(pose_from_native(frame["pose"]),
                                             frame["reference_marker_graph"], value[4])
            return (pose, f"atlas_{reference_map}") if pose is not None else (None, None)
        # Direct marker and successfully tag-constrained poses already have
        # fixed-world factors. They are not purely relative visual poses:
        # rigidly dragging them with a background KF would violate those
        # factors. Legacy logs without the marker-graph protocol retain their
        # previous same-map absolute / cross-map relative reading policy.
        if reference_map == frame["active_map"]:
            return pose_from_native(frame["pose"]), f"atlas_{reference_map}"
    if reference in references and frame.get("relative") is not None:
        value = references[reference]
        world_from_ref = pose_from_native(value[2])
        camera_from_ref = pose_from_native(frame["relative"])
        camera_from_ref.tvec *= (value[3] if len(value) > 3 else 1.0) / frame.get("reference_scale", 1.0)
        from .pipeline import compose_pose, inverse_pose
        return compose_pose(world_from_ref, inverse_pose(camera_from_ref)), f"atlas_{value[1]}"
    if same_publication:
        # Legacy snapshots without replay references still carry a valid
        # CURRENT native measurement. A different publication may have changed
        # units/gauge: never relabel its old raw pose with that new map's scale.
        return pose_from_native(frame["pose"]), f"atlas_{frame['active_map']}"
    return None, None


def visual_camera_at_revision(
    frame: dict, revision: dict, references: dict | None = None
) -> tuple[Pose | None, str | None]:
    """Resolve the pre-marker visual pose in the final reference-keyframe gauge.

    New native logs archive ``visual_relative`` before applying the current
    frame's marker factor.  Keeping it relative to the same keyframe and unit
    stamp as the published pose lets offline optimization use an independent
    visual measurement without counting the marker observation twice.
    """
    if frame.get("visual_relative") is None or frame.get("state") not in (2, 6):
        return None, None
    reference = frame.get("reference")
    if references is None:
        references = {value[0]: value for value in revision.get("references", [])}
    value = references.get(reference)
    if value is None or len(value) < 3:
        return None, None
    try:
        acquired_scale = float(frame.get("reference_scale", 1.0))
        committed_scale = float(value[3] if len(value) > 3 else 1.0)
        if not (np.isfinite(acquired_scale) and acquired_scale > 0.0
                and np.isfinite(committed_scale) and committed_scale > 0.0):
            return None, None
        world_from_reference = pose_from_native(value[2])
        camera_from_reference = pose_from_native(frame["visual_relative"])
    except (KeyError, TypeError, ValueError):
        return None, None
    camera_from_reference.tvec *= committed_scale / acquired_scale
    from .pipeline import compose_pose, inverse_pose
    return (
        compose_pose(world_from_reference, inverse_pose(camera_from_reference)),
        f"atlas_{value[1]}",
    )


def select_camera_frame(
    frame: dict | None, revision: dict, marker: Pose | None = None,
    marker_confidence: float = 0.0, slam_inliers: int = 0,
    accepted_marker_ids: tuple[int, ...] | None = None,
    marker_weights: dict[int, float] | None = None,
    revision_references: dict | None = None,
) -> FusedCameraFrame:
    """Choose a metre pose from this observation and this committed revision.

    Publish one Atlas world frame only. Raw fixed-marker PnP is an input to the
    native tracker, never an independent output-pose fallback. A marker pose is
    published only after native marker tracking has registered it to an Atlas;
    otherwise that frame remains invalid. Callers must supply the original
    observation, never an already fused/final output pose.
    """
    measured, map_id = (camera_at_revision(frame, revision, revision_references)
                        if frame else (None, None))
    mapping = next((m for m in revision.get("maps", []) if f"atlas_{m['id']}" == map_id), {})
    capture_mapping = next((m for m in (frame or {}).get("maps", [])
                            if m.get("id") == (frame or {}).get("active_map")), {})
    tracked = (frame or {}).get("marker_tracking", {})
    marker_valid = (marker is not None and np.isfinite(marker_confidence)
                    and marker_confidence >= MIN_MARKER_CONFIDENCE
                    and np.isfinite(marker.reprojection_error_px)
                    and 0.0 <= marker.reprojection_error_px <= MAX_MARKER_ERROR_PX
                    and np.all(np.isfinite(marker.tvec)) and np.all(np.isfinite(marker.rvec)))
    if marker_valid and accepted_marker_ids is not None:
        marker_valid = bool(marker.marker_ids) and set(marker.marker_ids).issubset(accepted_marker_ids)
    if marker_valid and marker_weights is not None:
        # Weak corners may assist a known pose, not establish an independent
        # metric reference when the native map cannot localize.
        weights = [marker_weights.get(mid, 1.0) for mid in marker.marker_ids]
        marker_valid = (all(np.isfinite(weight) and 0.0 < weight <= 1.0 for weight in weights)
                        and any(weight >= .99 for weight in weights))
    if (not tracked.get("partial") and not tracked.get("accepted")
            and tracked.get("reason") == "geometry_rejected"):
        marker_valid = False
    source, confidence = "invalid", 0.0
    metric = bool(mapping.get("metric", False))
    metric_recovered_later = bool(
        measured is not None and metric and capture_mapping
        and not capture_mapping.get("metric", False)
    )
    if measured is not None and metric:
        # The native pose already includes accepted tag-corner factors. Never
        # replace it with a separate PnP world pose: that would switch gauges
        # at exactly the frames where the two estimates disagree.
        source = "marker" if frame["state"] == 6 else (
            "marker+slam" if frame.get("tag_anchored", marker_valid) else "head-slam")
        confidence = (tracked.get("confidence", marker_confidence) if source == "marker"
                      else float(np.clip(slam_inliers / 80, .35, 1)))
    else:
        # Raw PnP and arbitrary-unit poses stay diagnostic inputs. Publishing
        # either here would bypass native tag-corner admission and switch the
        # world gauge at source transitions.
        measured = None
    return FusedCameraFrame(
        measured, source, confidence, slam_inliers,
        tracked.get("reprojection_px") if tracked.get("accepted")
        else marker.reprojection_error_px if marker_valid else None,
        map_id, int(mapping.get("revision", 0)), metric,
        "marker" if mapping.get("seed") else "monocular", bool(mapping.get("background", False)),
        metric_recovered_later)


def resolve_native_history_path(directory: Path) -> Path:
    """Prefer the compact current format while retaining legacy replay support."""
    for name in (
        "native_history.jsonl.zst",
        "native_history.jsonl.gz",
        "native_history.jsonl",
    ):
        path = directory / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"native replay history is missing from {directory}")


def annotate_anchor_consistency(
    frame: FusedCameraFrame, mapping: dict, calibration: Calibration,
    detections: dict[int, np.ndarray], accepted_ids: tuple[int, ...],
    weights: dict[int, float] | None = None,
) -> FusedCameraFrame:
    """Audit the selected revision's pose, never a stale native/PnP pose delta.

    A metre gauge is not a guarantee of anchor accuracy. Only decoded strong
    corners registered in THIS map can contradict its pose. This diagnostic
    does not move a camera, bridge a gap or feed back into SLAM. Gross conflict
    uses the native local-BA 100 px admission bound, not a new pose jump gate.
    """
    quality = {"status": "unobserved", "marker_ids": [],
               "max_rms_px": None, "minimum_depth_m": None,
               "map_id": frame.map_id, "revision": frame.revision}
    if (frame.pose is None or not frame.metric or
            not np.all(np.isfinite(frame.pose.rvec)) or not np.all(np.isfinite(frame.pose.tvec)) or
            frame.map_id != f"atlas_{mapping.get('id')}" or
            (mapping.get('revision') is not None and mapping['revision'] != frame.revision)):
        quality["status"] = "unavailable"
        return replace(frame, anchor_consistency=quality)
    from .pipeline import inverse_pose
    tcw = inverse_pose(frame.pose)
    errors, depths = [], []
    for marker_id in sorted(set(accepted_ids)):
        weight = float((weights or {}).get(marker_id, 1.0))
        if not np.isfinite(weight) or not .99 <= weight <= 1.:
            continue
        world = mapping.get("markers", {}).get(str(marker_id))
        raw = detections.get(marker_id)
        if world is None or raw is None:
            continue
        world = np.asarray(world, dtype=np.float64).reshape(-1, 3)
        raw = np.asarray(raw, dtype=np.float64).reshape(-1, 2)
        if (world.shape != (4, 3) or raw.shape != (4, 2) or
                not np.all(np.isfinite(world)) or not np.all(np.isfinite(raw))):
            continue
        quality["marker_ids"].append(marker_id)
        camera_points = world @ tcw.rotation_matrix.T + tcw.tvec.reshape(3)
        depths.append(float(np.min(camera_points[:, 2])))
        if depths[-1] <= 1e-6:
            continue
        pixels = cv2.undistortPoints(raw, calibration.camera_matrix,
                                    calibration.dist_coeffs, P=calibration.camera_matrix).reshape(4, 2)
        projected = camera_points @ calibration.camera_matrix.T
        error = projected[:, :2] / projected[:, 2:] - pixels
        errors.append(float(np.sqrt(np.mean(np.sum(error * error, axis=1)))))
    if not depths:
        return replace(frame, anchor_consistency=quality)
    quality["minimum_depth_m"] = min(depths)
    quality["max_rms_px"] = max(errors) if errors and all(np.isfinite(errors)) else None
    gross_conflict = min(depths) <= 1e-6 or any(not np.isfinite(e) or e > 100. for e in errors)
    quality["status"] = ("conflict" if gross_conflict else "consistent"
                         if max(errors) <= MAX_MARKER_ERROR_PX else "residual_warning")
    # Preserve the measured trajectory even when it is inconsistent; mark its
    # quality honestly rather than making a failure disappear from playback.
    return replace(frame, confidence=0.0 if gross_conflict else frame.confidence,
                   anchor_consistency=quality)


def read_native_history(history_path: Path) -> list[dict]:
    """Read native JSONL one record at a time without duplicating file text."""
    history = []
    if history_path.suffix == ".zst":
        with history_path.open("rb") as source:
            with zstd.ZstdDecompressor().stream_reader(source) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8") as stream:
                    for line in stream:
                        if line.strip():
                            history.append(json.loads(line))
    else:
        opener = gzip.open if history_path.suffix == ".gz" else open
        with opener(history_path, "rt") as stream:
            for line in stream:
                if line.strip():
                    history.append(json.loads(line))
    return history


def read_native_result(
    history_path: Path, marker_poses: list[Pose | None],
    marker_confidences: list[float], observations: list[OrbSlamObservation | None],
    fps: float, timing: dict[str, float],
    accepted_marker_ids: list[tuple[int, ...]] | None = None,
    marker_weights: list[dict[int, float]] | None = None,
) -> MetricOrbSlamResult:
    history = read_native_history(history_path)
    if not history or not history[-1].get("final"):
        raise RuntimeError("native SLAM did not finish publishing its final Atlas")
    final = history[-1]
    if any(mapping.get("points_mode") == "delta" for mapping in final["maps"]):
        raise RuntimeError("native SLAM final publication must contain full map points")
    maps = {f"atlas_{m['id']}": m for m in final["maps"]}
    final_references = {value[0]: value for value in final.get("references", [])}
    indexed = {round(h["timestamp"] * fps): h for h in history if not h.get("final")}
    frames = []
    for index, (marker, confidence, observation) in enumerate(zip(
            marker_poses, marker_confidences, observations)):
        state = indexed.get(index)
        inliers = observation.inliers if observation else 0
        frames.append(select_camera_frame(
            state, final, marker, confidence, inliers,
            accepted_marker_ids[index] if accepted_marker_ids is not None else None,
            marker_weights[index] if marker_weights is not None else None,
            final_references))
    active = maps.get(f"atlas_{final['active_map']}", {})
    points = np.asarray([p[1:] for p in active.get("points", [])], float).reshape(-1, 3)
    keyframes = tuple(pose_from_native(k[2]) for k in active.get("keyframes", []))
    init = next((i for i, h in sorted(indexed.items()) if h["pose"] is not None), None)
    return MetricOrbSlamResult(
        frames, points, keyframes, observations, init,
        active.get("scale") if active.get("metric") else None,
        sum(p is not None and c >= MIN_MARKER_CONFIDENCE
            for p, c in zip(marker_poses, marker_confidences)),
        None, None, timing, history, maps)


def _validate_large_frame_refinement(camera_points, pixels, information, camera, candidate,
                                     background_count=None):
    """Cross-fit existing background matches; never infer motion from stillness.

    Only the exceptional 20--40 mm correction path pays for these two solves.
    Each half must predict pixels it did not optimize, and agree with the full
    solution. Marker corners are excluded so one strong square cannot dominate
    both the candidate and its independent background check.
    """
    background = (np.arange(background_count) if background_count is not None
                  else np.flatnonzero(np.asarray(information) == 1.))
    if len(background) < 40 or np.linalg.norm(candidate[3:]) > .040:
        return False
    zero = np.zeros(6)
    half_range = np.array([.06] * 3 + [.025] * 3)
    def residual(parameters, indices):
        projected = cv2.projectPoints(camera_points[indices], parameters[:3],
                                     parameters[3:], camera, np.zeros(5))[0].reshape(-1, 2)
        return projected - pixels[indices]
    # Deterministic interleaving retains the native feature ordering. Neither
    # fold uses the other's pixels during optimization.
    for parity in (0, 1):
        training = background[parity::2]
        held_out = background[1-parity::2]
        fit = least_squares(lambda x: residual(x, training).ravel(), zero,
                            method='trf', loss='huber', f_scale=2.45,
                            bounds=(-half_range, half_range), max_nfev=20,
                            xtol=1e-7, ftol=1e-7, gtol=1e-7)
        if not fit.success or not np.all(np.isfinite(fit.x)):
            return False
        # Agreement alone is insufficient if both halves are underconstrained.
        column_norms = np.linalg.norm(fit.jac, axis=0)
        if np.any(column_norms < 1e-9):
            return False
        singular = np.linalg.svd(fit.jac / column_norms, compute_uv=False)
        if singular[-1] < 1e-4 * singular[0]:
            return False
        before = float(np.sqrt(np.mean(residual(zero, held_out) ** 2)))
        after = float(np.sqrt(np.mean(residual(fit.x, held_out) ** 2)))
        rotation = Rotation.from_rotvec(fit.x[:3])
        candidate_rotation = Rotation.from_rotvec(candidate[:3])
        centre = -rotation.inv().apply(fit.x[3:])
        candidate_centre = -candidate_rotation.inv().apply(candidate[3:])
        depths = (rotation.apply(camera_points[background]) + fit.x[3:])[:, 2]
        if (not np.isfinite(after) or after > 3. or after > .8 * before
                or np.linalg.norm(centre - candidate_centre) > .005
                or np.degrees((rotation * candidate_rotation.inv()).magnitude()) > .5
                or np.mean(depths > 0.) < .98):
            return False
    return True


def refine_final_frame_poses(
    result: MetricOrbSlamResult,
    calibration: Calibration,
    detections: list[dict[int, np.ndarray]],
    accepted_marker_ids: list[tuple[int, ...]],
    marker_weights: list[dict[int, float]] | None = None,
    *, rematched: bool = False,
) -> MetricOrbSlamResult:
    """Refine measured frame poses against the final committed Atlas.

    ORB-SLAM publishes ordinary frames relative to their reference keyframe.
    A final global BA updates the keyframes and map points, but does not revisit
    every non-keyframe image.  The native runner therefore records the actual
    MapPoint identity behind each tracked pixel; this cheap pass optimizes only
    six camera parameters against those final points and final marker corners.

    The pass is deliberately conservative: it cannot create a pose, change a
    map, or accept a refinement over 40 mm / 3 degrees. Corrections over 20 mm
    additionally require two disjoint background cross-fits. Exact keyframe images
    use their already committed native KF pose; other poses are kept unless
    the same committed factors improve by at least 0.02 px.
    """
    if not (len(result.frames) == len(result.observations) == len(detections)
            == len(accepted_marker_ids)):
        raise ValueError("dense pose refinement streams must have equal length")
    if marker_weights is not None and len(marker_weights) != len(result.frames):
        raise ValueError("dense pose marker weights must match frames")

    from .pipeline import inverse_pose

    camera = np.asarray(calibration.camera_matrix, dtype=np.float64)
    zero_distortion = np.zeros(5, dtype=np.float64)
    point_lookup: dict[str, dict[int, np.ndarray]] = {}
    marker_lookup: dict[str, dict[int, np.ndarray]] = {}
    for map_id, mapping in result.maps.items():
        point_lookup[map_id] = {
            int(point[0]): np.asarray(point[1:], dtype=np.float64)
            for point in mapping.get("points", []) if len(point) == 4
        }
        marker_lookup[map_id] = {
            int(marker_id): np.asarray(corners, dtype=np.float64).reshape(4, 3)
            for marker_id, corners in mapping.get("markers", {}).items()
            if len(corners) == 12
        }

    started = time.perf_counter()
    refined_frames: list[FusedCameraFrame] = []
    accepted = attempted = 0
    keyframe_synchronized = marker_refined = 0
    large_attempted = large_accepted = 0
    history = [h for h in result.history if not h.get('final')]
    history = history if len(history) == len(result.frames) else []
    final_references = {v[0]: v for v in result.history[-1].get('references', [])} if result.history else {}
    keyframe_poses = {
        map_id: {(k[0], round(float(k[1]), 8)): k for k in mapping.get('keyframes', [])}
        for map_id, mapping in result.maps.items()
    }
    before_rms: list[float] = []
    after_rms: list[float] = []
    corrections_m: list[float] = []
    corrections_deg: list[float] = []

    for index, (frame, observation) in enumerate(zip(result.frames, result.observations)):
        if frame.pose is None or not frame.metric or frame.map_id not in point_lookup:
            refined_frames.append(frame)
            continue
        # A committed KF is authoritative for its own image, not a fresh pose
        # optimization. Never use this to populate an originally invalid frame.
        captured = history[index] if history else {}
        exact = keyframe_poses[frame.map_id].get((captured.get('reference'), round(float(captured.get('timestamp', -1.)), 8)))
        if (exact is not None and exact[0] == captured.get('reference')
                and captured.get('state') in (2, 6) and captured.get('pose') is not None):
            refined_frames.append(replace(frame, pose=pose_from_native(exact[2])))
            keyframe_synchronized += 1
            continue
        marker_changed = False
        if frame.source in ('marker', 'marker+slam') and captured:
            old_map = next((m for m in captured.get('maps', [])
                            if m.get('id') == captured.get('active_map')), {})
            ref = final_references.get(captured.get('reference'))
            if ref is not None and len(ref) >= 6 and f'atlas_{ref[1]}' == frame.map_id:
                gauge = _apply_marker_graph_delta(
                    Pose(np.zeros((3, 1)), np.zeros((3, 1)), 0.),
                    captured.get('reference_marker_gauge'), ref[5], rigid_gauge=True)
                if gauge is not None:
                    for mid in accepted_marker_ids[index]:
                        old = old_map.get('markers', {}).get(str(mid))
                        new = marker_lookup[frame.map_id].get(mid)
                        if old is None or len(old) != 12 or new is None:
                            continue
                        transformed = (gauge.rotation_matrix @ np.asarray(old).reshape(4, 3).T).T + gauge.tvec.reshape(1, 3)
                        if np.isfinite(transformed).all() and np.max(np.linalg.norm(transformed-new, axis=1)) > 1e-5:
                            marker_changed = True
        # Do not refit unchanged marker measurements just to follow pixel noise.
        if frame.source != 'head-slam' and not marker_changed and not rematched:
            refined_frames.append(frame)
            continue
        world, pixels, information = [], [], []
        if observation is not None and len(observation.map_point_ids) == len(observation.tracked_points):
            final_points = point_lookup[frame.map_id]
            seen_points = set()
            for point_id, pixel in zip(observation.map_point_ids, observation.tracked_points):
                point = final_points.get(int(point_id))
                if (int(point_id) not in seen_points and point is not None and np.all(np.isfinite(point))
                        and np.linalg.norm(point) < 1000.0 and np.all(np.isfinite(pixel))):
                    world.append(point); pixels.append(pixel); information.append(1.0)
                    seen_points.add(int(point_id))

        background_count = len(world)
        final_markers = marker_lookup[frame.map_id]
        weights = marker_weights[index] if marker_weights is not None else {}
        for marker_id in accepted_marker_ids[index]:
            if marker_id not in final_markers or marker_id not in detections[index]:
                continue
            image = cv2.undistortPoints(
                np.asarray(detections[index][marker_id], dtype=np.float64).reshape(-1, 1, 2),
                camera, calibration.dist_coeffs, P=camera,
            ).reshape(-1, 2)
            weight = float(weights.get(marker_id, 1.0))
            if image.shape != (4, 2) or not np.all(np.isfinite(image)) or not 0.0 < weight <= 1.0:
                continue
            for point, pixel in zip(final_markers[marker_id], image):
                world.append(point); pixels.append(pixel)
                # Match the native BA's 64x marker-corner information while
                # retaining the existing 25% weak-border convention.
                information.append(64.0 * weight)

        # Six-DoF pose-only optimization needs geometric support beyond a
        # nearly degenerate handful of background matches.
        marker_corner_count = sum(value >= 16.0 for value in information)
        strong_marker_corners = sum(value >= 63.0 for value in information)
        if (len(world) < 12 and marker_corner_count < 4) or (marker_changed and strong_marker_corners < 4):
            refined_frames.append(frame)
            continue
        attempted += 1
        world_array = np.asarray(world, dtype=np.float64).reshape(-1, 3)
        pixel_array = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
        sqrt_information = np.sqrt(np.asarray(information, dtype=np.float64)).reshape(-1, 1)
        initial = inverse_pose(frame.pose)  # Tcw for projection
        x0 = np.concatenate((initial.rvec.reshape(3), initial.tvec.reshape(3)))
        # Atlas exports may retain a finite sentinel pose for a frame that was
        # later rejected.  It is not a meaningful seed for local refinement;
        # reject it before projection arithmetic can overflow.
        if (not np.all(np.isfinite(x0))
                or np.linalg.norm(x0[:3]) > np.pi + 1e-3
                or np.linalg.norm(x0[3:]) >= 1000.0):
            refined_frames.append(frame)
            continue

        # Express the solve in the initial CAMERA frame. A change of world
        # origin/orientation must not change either the bounds or the answer.
        camera_points = (initial.rotation_matrix @ (world_array-frame.pose.tvec.reshape(1, 3)).T).T
        x0 = np.zeros(6)

        def raw_residual(parameters: np.ndarray) -> np.ndarray:
            projected = cv2.projectPoints(
                camera_points, parameters[:3], parameters[3:], camera, zero_distortion
            )[0].reshape(-1, 2)
            return projected - pixel_array

        initial_raw = raw_residual(x0)

        def weighted_residual(parameters: np.ndarray) -> np.ndarray:
            return (raw_residual(parameters) * sqrt_information).reshape(-1)

        # Keep every trial inside the same small correction envelope used by
        # the commit gate.  Besides being cheaper, this prevents a degenerate
        # point configuration from exploring an enormous-but-finite rotation
        # vector before the post-fit validity checks can reject it.
        parameter_half_range = np.array([0.06, 0.06, 0.06, 0.025, 0.025, 0.025])
        optimized = least_squares(
            weighted_residual, x0, method="trf", loss="huber", f_scale=2.45,
            bounds=(x0 - parameter_half_range, x0 + parameter_half_range),
            max_nfev=80 if rematched else 20, xtol=1e-7, ftol=1e-7, gtol=1e-7,
        )
        finite_solution = bool(np.all(np.isfinite(optimized.x)))
        final_raw = raw_residual(optimized.x) if finite_solution else initial_raw
        rms_before = float(np.sqrt(np.mean(np.square(initial_raw))))
        rms_after = float(np.sqrt(np.mean(np.square(final_raw))))
        if finite_solution:
            rotation_delta = (Rotation.from_rotvec(optimized.x[:3])
                              * Rotation.from_rotvec(x0[:3]).inv())
            rotation_deg = float(np.degrees(rotation_delta.magnitude()))
            delta_camera_pose = Pose(
                optimized.x[:3].reshape(3, 1),
                optimized.x[3:].reshape(3, 1), rms_after,
            )
            # Compose the camera-local delta back into the original world;
            # retain an explicit gate on the exported camera-centre motion.
            from .pipeline import compose_pose
            candidate_twc = compose_pose(frame.pose, inverse_pose(delta_camera_pose))
            translation_m = float(np.linalg.norm(
                candidate_twc.tvec - frame.pose.tvec
            ))
        else:
            rotation_deg = translation_m = float("inf")
        bounded_solution = finite_solution and translation_m <= 0.040 and rotation_deg <= 3.0
        depths = np.array([-1.0])
        if bounded_solution:
            rotation_matrix = Rotation.from_rotvec(optimized.x[:3]).as_matrix()
            numerically_safe = (
                np.all(np.isfinite(rotation_matrix))
                and np.max(np.abs(rotation_matrix)) <= 1.0 + 1e-6
                and np.max(np.abs(world_array)) < 1000.0
                and np.max(np.abs(optimized.x[3:])) < 1000.0
            )
            if numerically_safe:
                # A corrupt BLAS operand must remain a rejected candidate, not
                # turn a best-effort diagnostic pass into a noisy export.
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    depths = (rotation_matrix @ camera_points.T
                              + optimized.x[3:].reshape(3, 1))[2]
        valid = (
            optimized.success and bounded_solution
            and np.all(np.isfinite(depths))
            and np.mean(depths > 0.0) >= 0.98
            and rms_after + 0.02 <= rms_before
            and (not (marker_changed or rematched) or np.all(np.linalg.norm(final_raw[np.asarray(information) >= 16.], axis=1) <= 3.))
        )
        if rematched:
            # A strong square must not overwhelm the independently re-matched
            # background. Keep the adapter's native 3 px reprojection standard.
            valid = (valid and background_count >= 40 and float(np.sqrt(np.mean(
                np.sum(final_raw[:background_count] ** 2, axis=1)))) <= 3.)
        if valid and translation_m > .020:
            large_attempted += 1
            valid = _validate_large_frame_refinement(
                camera_points, pixel_array, information, camera, optimized.x,
                background_count=background_count)
            large_accepted += int(valid)
        if not valid:
            refined_frames.append(frame)
            continue
        refined_tcw = Pose(
            optimized.x[:3].reshape(3, 1), optimized.x[3:].reshape(3, 1),
            rms_after, frame.pose.marker_ids, len(world_array), frame.pose.ambiguous,
        )
        accepted += 1
        marker_refined += int(marker_changed)
        before_rms.append(rms_before); after_rms.append(rms_after)
        corrections_m.append(translation_m); corrections_deg.append(rotation_deg)
        refined_frames.append(replace(
            frame, pose=compose_pose(frame.pose, inverse_pose(refined_tcw)), graph_reprojection_error_px=rms_after
        ))

    timing = dict(result.timing)
    timing.update({
        "dense_pose_attempted": float(attempted),
        "dense_pose_accepted": float(accepted),
        "dense_pose_keyframes_synchronized": float(keyframe_synchronized),
        "dense_pose_changed_marker_refined": float(marker_refined),
        "dense_pose_large_attempted": float(large_attempted),
        "dense_pose_large_accepted": float(large_accepted),
        "dense_pose_seconds": time.perf_counter() - started,
        "dense_pose_before_rms_px": float(np.median(before_rms)) if before_rms else 0.0,
        "dense_pose_after_rms_px": float(np.median(after_rms)) if after_rms else 0.0,
        "dense_pose_correction_median_mm": (
            1000.0 * float(np.median(corrections_m)) if corrections_m else 0.0
        ),
        "dense_pose_correction_median_deg": (
            float(np.median(corrections_deg)) if corrections_deg else 0.0
        ),
    })
    return replace(result, frames=refined_frames, timing=timing)
