#!/usr/bin/env python3
from __future__ import annotations

import argparse
import atexit
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
import sys
import shutil
import subprocess
import time

import cv2
import numpy as np
import zstandard as zstd

from aruco_track.auto_marker_map import (
    AutoMarkerMap,
    build_auto_marker_map,
    localize_auto_marker_frames,
)
from aruco_track.camera_state import (
    FusedCameraFrame, make_exclusion_mask, observed_hands_for_mask, prepare_slam_frame,
)
from aruco_track.bandsolve import solve_band_pose
from aruco_track.hands import (
    HandJointTracker,
    RawHandJoints,
    TemporalHandAssignmentGate,
    assign_hands_to_bands,
    bind_landmarks_to_wrist,
    joint_pose_to_dict,
    raw_hand_from_dict,
    wrist_anchor_error_px,
)
from aruco_track.hand_recovery import hand_recovery_policy
from aruco_track.hawor_backend import (
    HaworHandTracker, add_hand_backend_arguments, hawor_policy, prepare_hawor_predictions,
)
from aruco_track.models import BandLayout, Calibration, Pose
from aruco_track.marker_quality import (
    MarkerBoundaryQuality,
    is_assist_only_marker,
    weak_corner_information_weights,
)
from aruco_track.marker_uncertainty import marker_pose_uncertainty
from aruco_track.wrist_precision import (
    add_wrist_precision_arguments, wrist_precision_fields, wrist_precision_policy,
)
from aruco_track.orbslam3_backend import (
    MetricOrbSlamResult,
    read_native_result,
    remap_timestamp_observations,
    run_orbslam3_sequence,
    write_orbslam3_settings,
    write_tag_observation_hints,
)
from aruco_track.slam_sequence_cache import SlamSequenceCache, sequence_cache_key, sequence_cache_root, snapshot_sequence
from aruco_track.pipeline import (
    TrackingPipeline,
    compose_pose,
    inverse_pose,
    relative_pose,
)
from aruco_track.tag_graph import (
    MarkerPoseTracker,
    TagPoseResult,
    WorldTrackingResult,
    WristTrajectoryResult,
    optimize_tag_pose,
    optimize_wrist_trajectory,
    refine_wrist_pose_sequence,
)
from aruco_track.tracks import _matrix_to_quaternion, _quaternion_to_matrix
from aruco_track.tracks import AdaptivePoseSmoother, WorldPoseSmoother


OBSERVATION_CACHE_SCHEMA = "aruco-image-hand-observations/v1"
OBSERVATION_ALGORITHM_VERSION = 3
_MONOTAG_GAP_RECOVERY = None  # Explicit validated-profile hook.
_MONOTAG_FINAL_FRAME_ADAPTER = None


def _hand_cache_requires_upgrade(metadata, enabled=True, min_confidence=0.4, hand_backend='mediapipe'):
    """An old full-frame cache is not evidence that recovery already ran."""
    return bool(enabled and (
        not metadata.get('hand_joints_enabled', True)
        or metadata.get('hand_backend', 'mediapipe') != hand_backend
        or metadata.get('hand_recovery_policy') != (
            hawor_policy() if hand_backend == 'hawor' else
            hand_recovery_policy(min_confidence=min_confidence))
    ))


def _cached_raw_hands(record):
    measurements = [hand.get('joints', {}) for hand in record.get('hands', {}).values()]
    measurements.extend(record.get('unassigned_hands', []))
    return [raw_hand_from_dict(joints) for joints in measurements if joints.get('valid')]


def _write_slam_frame_pair(
    image_path: Path,
    image: np.ndarray,
    allowed_mask: np.ndarray,
    frame_index: int,
) -> None:
    """Encode one prepared SLAM frame and mask outside the sequential tracker."""
    if not cv2.imwrite(
        str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, 92]
    ):
        raise RuntimeError(f"cannot write ORB frame {frame_index}")
    if not cv2.imwrite(str(image_path) + ".mask.png", allowed_mask):
        raise RuntimeError(f"cannot write exclusion mask {frame_index}")


def _assist_only_detections(
    rejected: dict[int, np.ndarray],
    qualities: dict[int, MarkerBoundaryQuality | dict[str, object]],
) -> dict[int, np.ndarray]:
    output: dict[int, np.ndarray] = {}
    for marker_id, corners in rejected.items():
        quality = qualities.get(marker_id)
        if isinstance(quality, dict):
            quality = MarkerBoundaryQuality(**quality)
        if quality is not None and is_assist_only_marker(quality):
            output[marker_id] = corners
    return output


def _source_fingerprint(path: str | Path, hash_contents: bool) -> dict[str, object]:
    source = Path(path)
    stat = source.stat()
    result: dict[str, object] = {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
    }
    if hash_contents:
        digest = hashlib.sha256()
        with source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        result["sha256"] = digest.hexdigest()
    return result


def _prepare_export_hawor_predictions(video, calibration_path, output_path, *, max_frames,
                                      config_path, device, cached_metadata=None):
    """Reuse a validated raw hand cache even when the action output name changes."""
    kwargs = dict(max_frames=max_frames, config_path=config_path, device=device)
    metadata = cached_metadata or {}
    provenance = metadata.get('hand_backend_provenance') or {}
    matching_policy = (metadata.get('hand_backend') == 'hawor'
                       and metadata.get('hand_joints_enabled', True)
                       and metadata.get('hand_recovery_policy') == hawor_policy()
                       and provenance.get('backend') == 'hawor')
    prediction = provenance.get('prediction_file', {})
    if matching_policy and prediction.get('path'):
        previous = Path(prediction['path']).resolve()
        if previous.is_file():
            def verify_integrity():
                # The backend's legacy signature error also covers corruption.
                # Rule corruption out before treating that error as a cache miss.
                original = json.loads(previous.with_suffix('.meta.json').read_text())
                if original != provenance or not isinstance(original.get('signature'), dict):
                    raise ValueError('HaWoR raw-cache metadata differs from observation provenance')
                checked_files = [(previous, prediction)]
                if provenance.get('execution') == 'ssh' or 'metrics_file' in provenance:
                    checked_files.append((previous.with_suffix('.metrics.json'),
                                          provenance.get('metrics_file', {})))
                for path, expected in checked_files:
                    actual = dict(path=str(path), **_source_fingerprint(path, 'sha256' in expected))
                    if actual != expected:
                        raise ValueError(f'HaWoR raw-cache content changed: {path}')

            verify_integrity()
            try:
                return prepare_hawor_predictions(video, calibration_path, previous, **kwargs)
            except ValueError as error:
                signature_errors = {
                    f'HaWoR observation cache is unverified or different: {previous}; choose a new output path',
                    'HaWoR cache differs or is unverified; refusing to overwrite it',
                }
                if str(error) not in signature_errors:
                    raise
                verify_integrity()
                print('HaWoR: raw-cache input signature changed; preparing a new prediction sidecar',
                      flush=True)
        else:
            print(f'HaWoR: previous raw prediction sidecar is missing: {previous}; '
                  'preparing a new prediction sidecar', flush=True)
    return prepare_hawor_predictions(video, calibration_path, output_path, **kwargs)


def _observation_input_fingerprints(
    video: str | Path,
    calibration_path: str | Path,
    bands: list[str | Path],
    board: str | Path | None,
    hand_model: str | Path,
    hand_joints: bool = True,
) -> dict[str, object]:
    return {
        # Avoid hashing a potentially multi-gigabyte video on every reuse;
        # ctime catches replacement even if size/mtime are deliberately kept.
        "video": _source_fingerprint(video, False),
        "calibration": _source_fingerprint(calibration_path, True),
        "bands": [_source_fingerprint(path, True) for path in bands],
        "world_board": (
            _source_fingerprint(board, True) if board is not None else None
        ),
        "hand_model": _source_fingerprint(hand_model, True) if hand_joints else None,
    }


def pose_to_dict(pose: Pose | None) -> dict[str, object] | None:
    if pose is None:
        return None
    return {
        "translation_m": pose.tvec.reshape(3).tolist(),
        "quaternion_wxyz": _matrix_to_quaternion(pose.rotation_matrix).tolist(),
        "reprojection_error_px": pose.reprojection_error_px,
        "marker_ids": list(pose.marker_ids),
        "inlier_count": pose.inlier_count,
        "ambiguous": pose.ambiguous,
    }


def _pose_to_internal_dict(pose: Pose | None) -> dict[str, object] | None:
    if pose is None:
        return None
    return {
        "rvec": pose.rvec.reshape(3).tolist(),
        "tvec": pose.tvec.reshape(3).tolist(),
        "reprojection_error_px": pose.reprojection_error_px,
        "marker_ids": list(pose.marker_ids),
        "inlier_count": pose.inlier_count,
        "ambiguous": pose.ambiguous,
    }


def _pose_from_internal_dict(data: dict[str, object] | None) -> Pose | None:
    if data is None:
        return None
    return Pose(
        np.asarray(data["rvec"], dtype=np.float64).reshape(3, 1),
        np.asarray(data["tvec"], dtype=np.float64).reshape(3, 1),
        float(data["reprojection_error_px"]),
        tuple(int(value) for value in data["marker_ids"]),
        int(data["inlier_count"]),
        bool(data["ambiguous"]),
    )


def _pose_from_output_dict(data: dict[str, object] | None) -> Pose | None:
    if data is None:
        return None
    rotation = _quaternion_to_matrix(
        np.asarray(data["quaternion_wxyz"], dtype=np.float64)
    )
    return Pose(
        cv2.Rodrigues(rotation)[0],
        np.asarray(data["translation_m"], dtype=np.float64).reshape(3, 1),
        float(data["reprojection_error_px"]),
        tuple(int(value) for value in data["marker_ids"]),
        int(data["inlier_count"]),
        bool(data["ambiguous"]),
    )


def _world_landmarks(
    camera_landmarks: object,
    world_from_camera: Pose | None,
    world_wrist: Pose | None = None,
) -> list[list[float]] | None:
    if camera_landmarks is None or world_from_camera is None:
        return None
    points = np.asarray(camera_landmarks, dtype=np.float64)
    if world_wrist is None:
        transformed = (
            world_from_camera.rotation_matrix @ points.T
        ).T + world_from_camera.tvec.reshape(1, 3)
    else:
        vectors = points - points[0]
        transformed = (
            world_from_camera.rotation_matrix @ vectors.T
        ).T + world_wrist.tvec.reshape(1, 3)
    return transformed.tolist()


def _rebind_cached_joints(
    joints: dict[str, object],
    wrist_pose: Pose | None,
    world_reference: Pose | None,
    calibration: Calibration,
) -> None:
    """Regenerate every wrist-dependent hand field from cached model/image measurements."""
    if not joints.get("valid"):
        joints.update(
            camera_landmarks_m=None,
            band_landmarks_m=None,
            world_landmarks_m=None,
            wrist_anchor_valid=False,
            wrist_anchor_error_px=None,
        )
        return
    model = np.asarray(joints.get("model_landmarks_m"), dtype=np.float64)
    image = np.asarray(joints.get("image_landmarks_normalized"), dtype=np.float64)
    if model.shape != (21, 3) or image.shape[0] != 21 or image.shape[1] < 2:
        raise ValueError("cached hand measurement is incomplete; run fresh analysis")
    anchor_error = None
    bound_pose = wrist_pose
    if bound_pose is not None:
        anchor_error = wrist_anchor_error_px(image, bound_pose, calibration)
        maximum_error = 0.12 * float(np.hypot(*calibration.image_size))
        if anchor_error > maximum_error:
            bound_pose = None
    camera_points, band_points, world_points = bind_landmarks_to_wrist(
        model, bound_pose, world_reference
    )
    joints.update(
        camera_landmarks_m=(camera_points.tolist() if camera_points is not None else None),
        band_landmarks_m=(band_points.tolist() if band_points is not None else None),
        world_landmarks_m=(world_points.tolist() if world_points is not None else None),
        wrist_anchor_valid=band_points is not None,
        wrist_anchor_error_px=anchor_error,
    )


def _reassign_cached_joints(
    record: dict[str, object],
    band_names: list[str],
    band_poses: dict[str, Pose],
    calibration: Calibration,
    assignment_gate: TemporalHandAssignmentGate | None = None,
) -> dict[str, dict[str, object]]:
    """Repeat current left/right assignment from cached raw hand measurements."""
    measurements: list[dict[str, object]] = []
    for hand in record.get("hands", {}).values():
        joints = hand.get("joints", {})
        if joints.get("valid"):
            measurements.append(joints)
    measurements.extend(
        joints
        for joints in record.get("unassigned_hands", [])
        if joints.get("valid")
    )
    raw_hands: list[RawHandJoints] = []
    source_by_identity: dict[int, dict[str, object]] = {}
    for joints in measurements:
        try:
            raw = RawHandJoints(
                str(joints["handedness"]),
                (float(joints["handedness_score"]) if joints.get("handedness_score") is not None else None),
                np.asarray(joints["image_landmarks_normalized"], dtype=np.float64),
                np.asarray(joints["model_landmarks_m"], dtype=np.float64),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "cached hand identity measurement is incomplete; run fresh analysis"
            ) from exc
        raw_hands.append(raw)
        source_by_identity[id(raw)] = joints
    assigned = assign_hands_to_bands(
        raw_hands, band_names, band_poses, calibration
    )
    geometric_band_by_hand = {
        id(raw): name for name, raw in assigned.items() if name in band_names
    }
    if assignment_gate is not None:
        assigned = assignment_gate.update(assigned)
    for name, raw in assigned.items():
        source = source_by_identity[id(raw)]
        matched_band = geometric_band_by_hand.get(id(raw))
        source["association_status"] = (
            "confirmed"
            if name in band_names
            else "temporal_unconfirmed"
            if matched_band is not None
            else "geometry_rejected"
        )
        source["temporal_confirmation_frames"] = (
            assignment_gate.streak(matched_band)
            if assignment_gate is not None and matched_band is not None
            else int(name in band_names)
        )
    return {
        name: source_by_identity[id(raw)] for name, raw in assigned.items()
    }


def _optional_output_path(
    value: str | None,
    video_path: Path,
    suffix: str,
) -> Path | None:
    if value is None:
        return None
    return Path(value) if value else video_path.with_name(f"{video_path.stem}{suffix}")


def _parse_marker_ids(value: str) -> set[int]:
    marker_ids: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            first, last = (int(part) for part in item.split("-", 1))
            if first > last:
                raise argparse.ArgumentTypeError("marker ID range must be ascending")
            marker_ids.update(range(first, last + 1))
        else:
            marker_ids.add(int(item))
    if not marker_ids:
        raise argparse.ArgumentTypeError("at least one marker ID is required")
    return marker_ids


def _sample_marker_map_frames(
    detections: list[dict[int, np.ndarray]],
    fps: float,
    target_hz: float = 20.0,
    minimum_pair_frames: int = 3,
) -> tuple[list[dict[int, np.ndarray]], list[int], int]:
    """Remove temporal redundancy while retaining every viable marker edge."""
    if not detections:
        return [], [], 1
    stride = max(1, int(round(float(fps) / target_hz)))
    if stride == 1:
        indices = list(range(len(detections)))
        return list(detections), indices, stride
    selected = set(range(0, len(detections), stride))
    pair_frames: dict[tuple[int, int], list[int]] = {}
    for frame_index, frame in enumerate(detections):
        marker_ids = sorted(frame)
        for first_index, first_id in enumerate(marker_ids):
            for second_id in marker_ids[first_index + 1 :]:
                pair_frames.setdefault((first_id, second_id), []).append(frame_index)
    for frames in pair_frames.values():
        if len(frames) < minimum_pair_frames:
            continue
        already_selected = [index for index in frames if index in selected]
        missing = minimum_pair_frames - len(already_selected)
        if missing <= 0:
            continue
        available = [index for index in frames if index not in selected]
        ranks = np.linspace(0, len(available) - 1, missing, dtype=int)
        selected.update(available[int(rank)] for rank in ranks)
    indices = sorted(selected)
    return [detections[index] for index in indices], indices, stride


def _tag_result_dict(result: TagPoseResult) -> dict[str, object]:
    return {
        "accepted_marker_ids": list(result.accepted_marker_ids),
        "rejected_marker_ids": list(result.rejected_marker_ids),
        "marker_errors_px": {
            str(marker_id): error for marker_id, error in result.marker_errors_px.items()
        },
        "graph_reprojection_error_px": result.graph_reprojection_error_px,
        "confidence": result.confidence,
        "marker_consensus_vetoed": result.consensus_vetoed,
    }


def _replace_cached_graph_measurement(
    hand: dict[str, object], result: TagPoseResult
) -> None:
    hand["wrist_camera_graph"] = pose_to_dict(result.pose)
    hand.update(_tag_result_dict(result))


def _empty_tag_result() -> TagPoseResult:
    return TagPoseResult(None, (), (), {}, None, 0.0)


def _cached_tag_result(hand: dict[str, object]) -> TagPoseResult:
    """Restore a versioned cached wrist image measurement without re-solving it."""
    return TagPoseResult(
        _pose_from_output_dict(hand.get("wrist_camera_graph")),
        tuple(int(value) for value in hand.get("accepted_marker_ids", ())),
        tuple(int(value) for value in hand.get("rejected_marker_ids", ())),
        {
            int(marker_id): float(error)
            for marker_id, error in hand.get("marker_errors_px", {}).items()
        },
        (
            float(hand["graph_reprojection_error_px"])
            if hand.get("graph_reprojection_error_px") is not None
            else None
        ),
        float(hand.get("confidence", 0.0)),
        bool(hand.get("marker_consensus_vetoed", False)),
    )


def _load_observation_cache(
    path, video, calibration_path, bands, board, fps, image_size,
    hand_model, min_hand_confidence, hand_joints=True,
    allow_hand_upgrade=False, hand_backend='mediapipe',
):
    """Reuse measured image/camera observations, never old SLAM world poses."""
    metadata = json.loads(path.with_suffix('.meta.json').read_text())
    contract = metadata.get('observation_cache_contract')
    if (metadata.get('schema') != 'aruco-full-hand-actions/v3'
            or not isinstance(contract, dict)
            or contract.get('schema') != OBSERVATION_CACHE_SCHEMA
            or contract.get('algorithm_version') != OBSERVATION_ALGORITHM_VERSION
            or not isinstance(contract.get('input_fingerprints'), dict)):
        raise ValueError('observation cache contract is missing or outdated; run fresh analysis')
    expected = {'video': video, 'calibration': calibration_path, 'world_board': board}
    for key, value in expected.items():
        cached_value = metadata.get(key)
        if value is None:
            if cached_value is not None:
                raise ValueError(f'observation cache {key} does not match this run')
            continue
        if not cached_value or Path(cached_value).resolve() != Path(value).resolve():
            raise ValueError(f'observation cache {key} does not match this run')
    if [str(Path(p).resolve()) for p in metadata['bands']] != [str(Path(p).resolve()) for p in bands]:
        raise ValueError('observation cache wrist layouts do not match this run')
    backend_upgrade = bool(allow_hand_upgrade and hand_joints and (
        not metadata.get('hand_joints_enabled', True)
        or metadata.get('hand_backend', 'mediapipe') != hand_backend))
    if not backend_upgrade and (not metadata.get('hand_model')
            or Path(metadata['hand_model']).resolve() != Path(hand_model).resolve()):
        raise ValueError('observation cache hand model does not match; run fresh analysis')
    if not backend_upgrade and not np.isclose(metadata.get('min_hand_confidence', np.nan), min_hand_confidence):
        raise ValueError('observation cache hand confidence does not match; run fresh analysis')
    cached_hand_joints = bool(metadata.get('hand_joints_enabled', True))
    requested_hand_joints = bool(hand_joints)
    if cached_hand_joints != requested_hand_joints:
        # A marker-only cache can be safely upgraded by running the hand backend on
        # the immutable source frames while reusing its ArUco measurements.
        # The reverse direction is intentionally rejected: silently dropping
        # cached measurements makes provenance ambiguous.
        if not (allow_hand_upgrade and requested_hand_joints and not cached_hand_joints):
            raise ValueError('observation cache hand-joint mode does not match; run fresh analysis')
    if not np.isclose(metadata['fps'], fps) or tuple(metadata['image_size']) != tuple(image_size):
        raise ValueError('observation cache video geometry/timing does not match')
    current_fingerprints = _observation_input_fingerprints(
        video, calibration_path, bands, board, metadata['hand_model'] if backend_upgrade else hand_model,
        hand_joints=cached_hand_joints,
    )
    cached_fingerprints = dict(contract['input_fingerprints'])
    if not cached_hand_joints:
        # Legacy marker-only caches recorded an unused model fingerprint.
        # On upgrade, the new model is validated by fresh hand detection.
        cached_fingerprints['hand_model'] = None
    if cached_fingerprints != current_fingerprints:
        raise ValueError('observation cache input content changed; run fresh analysis')
    records = [json.loads(line) for line in path.read_text().splitlines()]
    required = {'marker_camera_pose_observed', 'marker_camera_confidence', 'detected_marker_corners',
                'marker_boundary_quality', 'boundary_rejected_marker_corners', 'hands'}
    if not records or len(records) != metadata['frames']:
        raise ValueError('observation cache is empty or incomplete')
    for index, record in enumerate(records):
        if (record.get('frame') != index or not required.issubset(record)
                or not np.isclose(record.get('timestamp_s', -1), index/fps, atol=1e-6)):
            raise ValueError(f'invalid cached observation at frame {index}')
        if not all('joints' in hand for hand in record['hands'].values()):
            raise ValueError(f'missing cached hand measurement at frame {index}; run fresh analysis')
        for hand in record['hands'].values():
            joints = hand['joints']
            if joints.get('valid') and not {
                'image_landmarks_normalized', 'model_landmarks_m', 'bend_angles_rad'
            }.issubset(joints):
                raise ValueError(f'incomplete cached hand measurement at frame {index}; run fresh analysis')
    return records, metadata


def _submap_runs(
    submap_ids: list[str | None],
) -> list[tuple[int, int, str]]:
    runs: list[tuple[int, int, str]] = []
    start = 0
    while start < len(submap_ids):
        submap_id = submap_ids[start]
        stop = start + 1
        while stop < len(submap_ids) and submap_ids[stop] == submap_id:
            stop += 1
        if submap_id is not None:
            runs.append((start, stop, submap_id))
        start = stop
    return runs


def _final_resolved_map(result: MetricOrbSlamResult | None) -> dict[str, object]:
    if result is None or not result.history:
        return {}
    final = result.history[-1]
    map_id = final.get("active_map")
    aliases = final.get("marker_map_aliases", {})
    seen: set[object] = set()
    while map_id not in seen:
        seen.add(map_id)
        target = aliases.get(str(map_id), aliases.get(map_id))
        if target is None:
            break
        map_id = target
    return result.maps.get(f"atlas_{map_id}", {})


def _optimize_wrists_by_submap(
    fused_camera: list[FusedCameraFrame],
    active_submap_ids: list[str | None],
    timestamps_s: list[float],
    fps: float,
    wrist_poses: list[Pose | None],
    detections: list[dict[int, np.ndarray]],
    accepted_ids: list[tuple[int, ...]],
    assist_detections: list[dict[int, np.ndarray]],
    band: BandLayout,
    calibration: Calibration,
) -> WristTrajectoryResult:
    poses: list[Pose | None] = [None] * len(fused_camera)
    errors: list[float | None] = [None] * len(fused_camera)
    # Losing one camera measurement does not create a new world frame.
    # Preserve short, bracketed gaps in the optimization domain, while leaving
    # their camera poses missing. No measurement or trajectory is interpolated.
    runs: list[tuple[int, int, str]] = []
    for start, stop, map_id in _submap_runs(active_submap_ids):
        if (
            runs
            and runs[-1][2] == map_id
            and 0.0 < timestamps_s[start] - timestamps_s[runs[-1][1] - 1] <= 0.15
        ):
            runs[-1] = (runs[-1][0], stop, map_id)
        else:
            runs.append((start, stop, map_id))
    for start, stop, map_id in runs:
        result = optimize_wrist_trajectory(
            [frame.pose if active_submap_ids[index] == map_id and frame.metric else None
             for index, frame in enumerate(fused_camera[start:stop], start)],
            wrist_poses[start:stop],
            detections[start:stop],
            accepted_ids[start:stop],
            band,
            calibration,
            fps=fps,
            timestamps_s=timestamps_s[start:stop],
            assist_detections=assist_detections[start:stop],
        )
        poses[start:stop] = result.poses
        errors[start:stop] = result.reprojection_errors_px
    return WristTrajectoryResult(poses, errors)


def _track_world_by_submap(
    fused_camera: list[FusedCameraFrame],
    active_submap_ids: list[str | None],
    wrist_camera_poses: list[Pose | None],
    predicted_world_poses: list[Pose | None],
) -> WorldTrackingResult:
    # Same-frame measurement validity only; no time bridges or missing wrists.
    poses, sources = [], []
    for camera, wrist, optimized in zip(fused_camera, wrist_camera_poses, predicted_world_poses):
        valid = camera.pose is not None and camera.metric and wrist is not None
        poses.append(optimized if valid and optimized is not None else
                     compose_pose(camera.pose, wrist) if valid else None)
        sources.append(camera.source if valid else "invalid")
    return WorldTrackingResult(poses, sources)


def _weak_corner_factor_counts(path: Path) -> tuple[int, int]:
    points = frames = 0
    with path.open() as stream:
        for line in stream:
            fields = line.split()
            if not fields or fields[0].startswith("#") or "weights" not in fields:
                continue
            start = fields.index("weights") + 1
            stop = fields.index("ids", start) if "ids" in fields[start:] else len(fields)
            frame_points = sum(0.0 < float(value) < 0.249 for value in fields[start:stop])
            points += frame_points
            frames += frame_points > 0
    return points, frames


def _marker_layouts_by_component(
    marker_layout: BandLayout | None,
    marker_layouts: dict[str, BandLayout] | None,
    active_submap_ids: list[str | None],
) -> dict[str, BandLayout]:
    """Resolve layouts using the same component ids written to native SLAM."""
    if marker_layouts:
        return marker_layouts
    if marker_layout is None:
        return {}
    component_ids = {
        component_id for component_id in active_submap_ids
        if component_id is not None
    }
    key = next(iter(component_ids)) if len(component_ids) == 1 else marker_layout.name
    return {key: marker_layout}


def _persist_marker_bootstrap_artifacts(diagnostics: dict, replay_dir: Path) -> dict:
    """Keep audit evidence after the private native run directory is removed."""
    persisted = dict(diagnostics)
    for field, filename in (("probe_history_path", "marker_bootstrap_history.jsonl.zst"),
                            ("hints_path", "marker_bootstrap_hints.txt")):
        if not diagnostics.get(field):
            continue
        source_path = Path(diagnostics[field])
        if not source_path.is_file():
            persisted[field] = None
            persisted["missing_artifacts"] = {**persisted.get("missing_artifacts", {}),
                                               field: str(source_path)}
            continue
        replay_dir.mkdir(parents=True, exist_ok=True)
        destination_path = replay_dir / filename
        if field == "probe_history_path":
            with source_path.open("rb") as source, destination_path.open("wb") as destination:
                zstd.ZstdCompressor(level=1).copy_stream(source, destination)
        else:
            shutil.copy2(source_path, destination_path)
        persisted[field] = str(destination_path.resolve())
    return persisted


def _run_deferred_head_slam(
    video_path: Path,
    temporary_path: Path,
    calibration: Calibration,
    detections: list[dict[int, np.ndarray]],
    marker_poses: list[Pose | None],
    marker_confidences: list[float],
    accepted_marker_ids: list[tuple[int, ...]],
    marker_layout: BandLayout | None,
    active_submap_ids: list[str | None],
    replay_dir: Path,
    slam_init: str,
    load_atlas: Path | None,
    save_atlas: Path,
    marker_weights: list[dict[int, float]] | None = None,
    dynamic_geometry: bool = False,
    rigid_marker_layout: bool = False,
    marker_layouts: dict[str, BandLayout] | None = None,
    weak_marker_corners: bool = False,
    slam_replay: bool = True,
    excluded_marker_ids: list[set[int]] | None = None,
) -> MetricOrbSlamResult:
    frame_count = len(marker_poses)
    if excluded_marker_ids is not None and len(excluded_marker_ids) != frame_count:
        raise ValueError("excluded marker IDs must match the prepared frame count")
    fps_capture = cv2.VideoCapture(str(video_path))
    if not fps_capture.isOpened():
        raise RuntimeError(f"cannot reopen {video_path} for deferred ORB-SLAM3")
    fps = float(fps_capture.get(cv2.CAP_PROP_FPS)) or 30.0
    fps_capture.release()
    initialization_frame = 0
    project_dir = Path(__file__).resolve().parent
    sequence_cache = SlamSequenceCache(sequence_cache_root(project_dir))
    cache_key = sequence_cache_key(
        video_path,
        temporary_path,
        calibration,
        marker_layout,
        fps,
        frame_count,
        marker_poses,
        marker_confidences,
        accepted_marker_ids,
        marker_weights,
        detections=detections,
        marker_layouts=marker_layouts,
        active_submap_ids=active_submap_ids,
        weak_marker_corners=weak_marker_corners,
        excluded_marker_ids=excluded_marker_ids,
    )
    cache_started = time.perf_counter()
    cached_sequence = sequence_cache.lookup(cache_key)
    cache_lookup_seconds = time.perf_counter() - cache_started
    cache_hit = cached_sequence is not None
    preparation_seconds = 0.0
    with tempfile.TemporaryDirectory(prefix="orbslam3-") as temporary_directory:
        root = Path(temporary_directory)
        output_dir = root / "output"
        output_dir.mkdir()
        if cached_sequence is not None:
            sequence_dir = cached_sequence.path
        else:
            preparation_started = time.perf_counter()
            staging = sequence_cache.staging_directory(cache_key)
            sequence_dir = staging
            capture = None
            try:
                rgb_dir = sequence_dir / "rgb"
                rgb_dir.mkdir(parents=True)
                rgb_lines = ["# color images", "# timestamp filename", "#"]
                from aruco_track.marker_corners import MarkerCornerTracker, TrackedMarkerObservation
                # A calibrated rigid board uses the stable Atlas component id
                # ``world_board``; its descriptive JSON name need not match.
                available_layouts = _marker_layouts_by_component(
                    marker_layout, marker_layouts, active_submap_ids
                )
                corner_trackers = {
                    submap_id: MarkerCornerTracker(calibration, layout)
                    for submap_id, layout in available_layouts.items()
                }
                tracked_observations = []
                prepared_frames = 0
                capture = cv2.VideoCapture(str(video_path))
                if not capture.isOpened():
                    raise RuntimeError(f"cannot reopen {video_path} for deferred ORB-SLAM3")
                # Marker tracking stays ordered. Only independent JPEG/PNG
                # encoding runs concurrently, with a bounded memory footprint.
                encode_workers = 4
                pending_writes = deque()
                with ThreadPoolExecutor(
                    max_workers=encode_workers, thread_name_prefix="slam-encode"
                ) as encoder:
                    with temporary_path.open() as records:
                        for frame_index, line in enumerate(records):
                            ok, frame = capture.read()
                            if not ok:
                                break
                            if frame_index < initialization_frame:
                                continue
                            record = json.loads(line)
                            excluded_ids = (
                                excluded_marker_ids[frame_index]
                                if excluded_marker_ids is not None else set()
                            )
                            # A deferred decoded group must not return through
                            # a previous optical-flow seed or weak-corner streak.
                            if excluded_ids:
                                for candidate_tracker in corner_trackers.values():
                                    candidate_tracker.tracks = {
                                        key: value for key, value in candidate_tracker.tracks.items()
                                        if key[0] not in excluded_ids
                                    }
                                    candidate_tracker.weak_streaks = {
                                        key: value for key, value in candidate_tracker.weak_streaks.items()
                                        if key[0] not in excluded_ids
                                    }
                            frame_submap_id = active_submap_ids[frame_index]
                            frame_layout = available_layouts.get(frame_submap_id)
                            corner_tracker = corner_trackers.get(frame_submap_id)
                            if corner_tracker:
                                weak_detections = {
                                    int(marker_id): np.asarray(corners, dtype=float)
                                    for marker_id, corners in record.get(
                                        "boundary_rejected_marker_corners", {}
                                    ).items()
                                    if int(marker_id) not in excluded_ids
                                }
                                weak_corner_weights = {}
                                if weak_marker_corners:
                                    for marker_id, quality in record.get(
                                        "marker_boundary_quality", {}
                                    ).items():
                                        marker_id = int(marker_id)
                                        if marker_id not in weak_detections:
                                            continue
                                        weak_corner_weights[marker_id] = (
                                            weak_corner_information_weights(
                                                MarkerBoundaryQuality(**quality)
                                            )
                                        )
                                tracked_observations.append(corner_tracker.update(
                                    frame, frame_index/fps, detections[frame_index], accepted_marker_ids[frame_index],
                                    marker_poses[frame_index], marker_confidences[frame_index],
                                    marker_weights[frame_index] if marker_weights else None,
                                    weak_detections=weak_detections,
                                    weak_corner_weights=weak_corner_weights,
                                ))
                            else:
                                tracked_observations.append(TrackedMarkerObservation())
                            hands = observed_hands_for_mask(record)
                            mask_detections = {
                                int(marker_id): np.asarray(corners, dtype=np.float64)
                                for marker_id, corners in record.get(
                                    "boundary_rejected_marker_corners", {}
                                ).items()
                            }
                            # Admission filters factors, never image exclusion:
                            # keep even deferred raw quads out of background ORB.
                            mask_detections.update({
                                int(marker_id): np.asarray(corners, dtype=np.float64)
                                for marker_id, corners in record.get(
                                    "detected_marker_corners", {}
                                ).items()
                            })
                            mask_detections.update(detections[frame_index])
                            mask_detections.update({
                                int(mid): np.asarray(corners, dtype=np.float64)
                                for mid,corners in record.get('marker_mask_corners',{}).items()
                            })
                            # Keep tracked marker regions excluded from background ORB
                            # even when ArUco cannot decode a complete quad this frame.
                            if frame_layout and tracked_observations[-1].pose is not None:
                                tracked = tracked_observations[-1]
                                for mid in set(tracked.marker_ids):
                                    if mid not in mask_detections:
                                        pixels, depth = corner_tracker._project(frame_layout.markers[mid], tracked.pose)
                                        if np.all(depth > 0):
                                            mask_detections[mid] = pixels
                            allowed_mask = make_exclusion_mask(frame.shape, mask_detections, hands)
                            processed = prepare_slam_frame(frame, allowed_mask)
                            relative = Path("rgb") / f"{frame_index:06d}.jpg"
                            pending_writes.append(encoder.submit(
                                _write_slam_frame_pair,
                                sequence_dir / relative,
                                processed,
                                allowed_mask,
                                frame_index,
                            ))
                            if len(pending_writes) >= 2 * encode_workers:
                                pending_writes.popleft().result()
                            rgb_lines.append(
                                f"{frame_index / fps:.9f} {relative.as_posix()}"
                            )
                            prepared_frames += 1
                    while pending_writes:
                        pending_writes.popleft().result()
                if prepared_frames != frame_count - initialization_frame:
                    raise RuntimeError(
                        f"source video ended after {prepared_frames} of "
                        f"{frame_count - initialization_frame} SLAM frames"
                    )
                (sequence_dir / "rgb.txt").write_text("\n".join(rgb_lines) + "\n")
                tag_observations_path = sequence_dir / "tag_observations.txt"
                write_tag_observation_hints(
                    tag_observations_path,
                    marker_poses,
                    marker_confidences,
                    detections,
                    accepted_marker_ids,
                    marker_layout or BandLayout("unanchored", "DICT_4X4_50", {}),
                    calibration,
                    fps,
                    initialization_frame,
                    marker_weights=marker_weights,
                    include_ids=True,
                    tracked_observations=tracked_observations if available_layouts else None,
                    marker_layouts=available_layouts,
                    marker_component_ids=active_submap_ids,
                )
                cached_sequence = sequence_cache.publish(cache_key, staging)
                sequence_dir = cached_sequence.path
            except Exception:
                sequence_cache.discard(staging)
                raise
            finally:
                if capture is not None:
                    capture.release()
                preparation_seconds = time.perf_counter() - preparation_started
        # A cache entry may be pruned by another concurrently processed video.
        # Snapshot its immutable files into this run's private temporary tree
        # using hard links (copy only if the filesystems differ).  The native
        # runner then never observes a cache directory disappearing mid-run.
        cached_source_dir = sequence_dir
        run_sequence_dir = root / "sequence"
        snapshot_started = time.perf_counter()
        snapshot_sequence(cached_source_dir, run_sequence_dir)
        snapshot_seconds = time.perf_counter() - snapshot_started
        print(f"SLAM inputs ready: cache_hit={cache_hit}, lookup={cache_lookup_seconds:.1f}s, "
              f"prepare={preparation_seconds:.1f}s, snapshot={snapshot_seconds:.1f}s; "
              "starting native ORB-SLAM3", flush=True)
        sequence_dir = run_sequence_dir
        settings_path = root / "camera.yaml"
        write_orbslam3_settings(settings_path, calibration, fps, slam_init, load_atlas, save_atlas,
                               dynamic_geometry=dynamic_geometry,
                               rigid_marker_layout=rigid_marker_layout)
        tag_observations_path = sequence_dir / "tag_observations.txt"
        bootstrap_diagnostics = None
        if os.environ.get("ORB_SLAM3_OFFLINE_MARKER_BOOTSTRAP", "1") != "0":
            from aruco_track.marker_bootstrap import bootstrap_initial_marker_observations
            try:
                bootstrap = bootstrap_initial_marker_observations(
                    project_dir, sequence_dir, root, calibration, fps, detections,
                    marker_poses, marker_confidences, accepted_marker_ids,
                    marker_weights, _marker_layouts_by_component(
                        marker_layout, marker_layouts, active_submap_ids),
                    active_submap_ids, slam_init=slam_init, load_atlas=load_atlas,
                    dynamic_geometry=dynamic_geometry,
                )
                bootstrap_diagnostics = bootstrap.diagnostics
                if bootstrap.hints_path is not None:
                    tag_observations_path = bootstrap.hints_path
            except (RuntimeError, ValueError, cv2.error, subprocess.TimeoutExpired) as error:
                bootstrap_diagnostics = {"accepted": False, "reason": "probe_failed",
                                         "error": str(error), "published_observations": []}
            replay_dir.mkdir(parents=True, exist_ok=True)
            bootstrap_diagnostics = _persist_marker_bootstrap_artifacts(bootstrap_diagnostics, replay_dir)
            (replay_dir / "marker_bootstrap.json").write_text(
                json.dumps(bootstrap_diagnostics, indent=2, allow_nan=False))
            probe_log = root / "marker_bootstrap/native/native.log"
            if probe_log.is_file():
                shutil.copy2(probe_log, replay_dir / "marker_bootstrap_native.log")
            print(f"Offline marker bootstrap: {bootstrap_diagnostics['reason']}; "
                  f"{len(bootstrap_diagnostics.get('published_observations', []))} corner-validated initialization hints", flush=True)
        (
            timestamp_slam_poses,
            timestamp_keyframe_poses,
            slam_points,
            timestamp_observations,
            timing,
        ) = run_orbslam3_sequence(
            project_dir,
            sequence_dir,
            settings_path,
            output_dir,
            tag_observations_path,
            compact_history=not slam_replay,
        )
        timing["sequence_cache_hit"] = float(cache_hit)
        timing["sequence_cache_lookup_s"] = cache_lookup_seconds
        timing["sequence_preparation_s"] = preparation_seconds
        timing["sequence_snapshot_s"] = snapshot_seconds
        timing["marker_bootstrap_probe_seconds"] = (bootstrap_diagnostics or {}).get("probe_wall_seconds", 0.)
        timing["marker_bootstrap_observations"] = float(len((bootstrap_diagnostics or {}).get("published_observations", [])))
        weak_points, weak_frames = _weak_corner_factor_counts(tag_observations_path)
        timing["confirmed_weak_marker_corner_factors"] = float(weak_points)
        timing["confirmed_weak_marker_corner_frames"] = float(weak_frames)
        indexed_observations = remap_timestamp_observations(
            timestamp_observations, fps
        )
        observations = [
            indexed_observations.get(index) for index in range(frame_count)
        ]
        replay_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_dir / "native.log", replay_dir / "native.log")
        dynamic_log = output_dir / "frames.txt.dynamic.jsonl"
        if dynamic_log.is_file():
            shutil.copy2(dynamic_log, replay_dir / "dynamic_geometry.jsonl")
        history_path = replay_dir / "native_history.jsonl.zst"
        with (output_dir / "frames.txt.history.jsonl").open("rb") as source:
            with history_path.open("wb") as destination:
                zstd.ZstdCompressor(level=1).copy_stream(source, destination)
        result = read_native_result(history_path, marker_poses, marker_confidences,
                                    observations, fps, timing,
                                    accepted_marker_ids=accepted_marker_ids,
                                    marker_weights=marker_weights)
        result = replace(result, offline_marker_bootstrap=bootstrap_diagnostics)
        from aruco_track.orbslam3_backend import annotate_anchor_consistency, refine_final_frame_poses
        result = refine_final_frame_poses(
            result, calibration, detections, accepted_marker_ids, marker_weights
        )
        if os.environ.get('ORB_SLAM3_FINAL_FRAME_REMATCH', '1') != '0':
            from aruco_track.final_frame_rematch import refine_inconsistent_frames
            result = refine_inconsistent_frames(
                result, project_dir, video_path, save_atlas, temporary_path,
                calibration, detections, accepted_marker_ids,
                marker_weights if marker_weights is not None else [{} for _ in result.frames],
                fps, replay_dir,
                adapter_path=_MONOTAG_FINAL_FRAME_ADAPTER,
            )
        from aruco_track.offline_prefix import recover_initial_prefix
        if load_atlas is None and os.environ.get("ORB_SLAM3_PREFIX_RELOCALIZATION", "1") != "0":
            result = recover_initial_prefix(
                result, project_dir, video_path, save_atlas, temporary_path,
                calibration, fps, replay_dir,
            )
        if _MONOTAG_GAP_RECOVERY is not None and os.environ.get("ORB_SLAM3_SHORT_GAP_RECOVERY", "1") != "0":
            result = _MONOTAG_GAP_RECOVERY(result, project_dir, video_path, save_atlas,
                temporary_path, calibration, fps, replay_dir)
        annotated_frames = [
            annotate_anchor_consistency(frame, result.maps.get(frame.map_id, {}),
                calibration, detections[i], accepted_marker_ids[i],
                marker_weights[i] if marker_weights is not None else None)
            for i, frame in enumerate(result.frames)
        ]
        return replace(result, frames=annotated_frames)


def main() -> None:
    from aruco_track.processing_host import require_processing_host
    if "--help" not in sys.argv and "-h" not in sys.argv:
        require_processing_host()
    parser = argparse.ArgumentParser(
        description="Export wrist 6-DoF and full 21-landmark hand action labels"
    )
    parser.add_argument("video")
    parser.add_argument("--calib", default="calib/camera_1920x1080.json")
    parser.add_argument("--band", action="append", default=[], help="hand band JSON; repeat")
    parser.add_argument("--world-board", help="fixed world-reference board layout JSON")
    parser.add_argument(
        "--auto-marker-map",
        action="store_true",
        help=(
            "build independent metric Atlas maps from reliably connected "
            "fixed-marker components"
        ),
    )
    parser.add_argument(
        "--static-marker-ids",
        type=_parse_marker_ids,
        default=_parse_marker_ids("20-49"),
        metavar="IDS",
        help="fixed marker IDs for auto mapping, e.g. 20-49 or 20,21,30",
    )
    parser.add_argument(
        "--static-marker-size-mm",
        type=float,
        default=48.0,
        help="printed fixed-marker side length for auto mapping (default: 48)",
    )
    parser.add_argument(
        "--strict-marker-corners", action="store_true",
        help="comparison mode: reject slight grid differences instead of keeping low-weight corners",
    )
    parser.add_argument(
        "--marker-map-output",
        help="auto-generated local marker map JSON path",
    )
    parser.add_argument("--hand-model", default="models/hand_landmarker.task")
    add_wrist_precision_arguments(parser)
    add_hand_backend_arguments(parser)
    parser.add_argument(
        "--hand-joints", action=argparse.BooleanOptionalAction, default=True,
        help="run offline hand reconstruction with the selected backend (default: enabled)",
    )
    parser.add_argument("--slam-dynamic-filter", action=argparse.BooleanOptionalAction, default=False,
                        help="optional classical multi-frame geometric point masking (default: disabled; enable for strongly dynamic scenes)")
    parser.add_argument(
        "--slam-weak-marker-corners",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "experimental: add three-frame-confirmed, pose-consistent rejected "
            "fixed-marker corners to BA at 5%% information"
        ),
    )
    parser.add_argument("--output", help="output JSONL path")
    parser.add_argument(
        "--lerobot-output", type=Path, metavar="DIR",
        help="optional LeRobot v3 training export; disabled by default",
    )
    parser.add_argument(
        "--lerobot-task",
        help="natural-language episode instruction used with --lerobot-output",
    )
    parser.add_argument(
        "--lerobot-fps", type=int, default=20,
        help="optional LeRobot training FPS (default: 20)",
    )
    parser.add_argument("--lerobot-repo-id", help="optional LeRobot repository ID")
    parser.add_argument("--min-hand-confidence", type=float, default=0.4)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--head-slam", action="store_true",
                        help="native ORB-SLAM3 offline mapping, marker metric anchoring")
    parser.add_argument("--slam-init", choices=("auto", "marker"), default="auto")
    parser.add_argument("--reuse-observations", type=Path, metavar="ACTIONS_JSONL",
                        help="reuse prior ArUco/hand camera observations for the same video and fixed layouts; rerun native SLAM")
    parser.add_argument("--load-atlas", type=Path)
    parser.add_argument("--save-atlas", type=Path)
    parser.add_argument("--open-replay", action="store_true")
    parser.add_argument(
        "--slam-replay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "build the synchronized MP4/HTML replay after analysis "
            "(default: enabled; disable for trajectory-only batch audits)"
        ),
    )
    parser.add_argument("--slam-debug-video", "--vo-debug-video", dest="slam_debug_video",
                        nargs="?", const="", metavar="PATH",
                        help="write native SLAM process replay MP4")
    parser.add_argument(
        "--graph-diagnostics",
        nargs="?",
        const="",
        metavar="PATH",
        help="write per-frame factor diagnostics JSONL; optional output path",
    )
    removed = [arg.split("=")[0] for arg in sys.argv[1:]
               if arg.split("=")[0] in {"--ego-vo", "--max-vo-gap-s"}]
    if removed:
        parser.error("removed " + ", ".join(removed) +
                     "; use --head-slam (native ORB only, no marker-gap time limit)")
    from aruco_track.server_pipeline import DEFAULT_CONFIG, default_execution, run_remote_pipeline
    parser.add_argument('--execution', choices=('local', 'server'), default=default_execution(),
                        help='processing host; configured production default is server')
    parser.add_argument('--server-config', type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    if args.execution == 'server':
        run_remote_pipeline(args)
        return
    production = Path(__file__).resolve().parent/'config/production.json'
    if production.is_file():
        for name, value in json.loads(production.read_text())['environment'].items():
            os.environ.setdefault(name, value)
    if args.hand_joints and args.hand_backend == 'hawor':
        # The lightweight runtime manifest identifies all HaWoR resources.
        # The legacy --hand-model option remains exclusive to MediaPipe.
        args.hand_model = str(args.hawor_config.resolve())
    for atlas_path in (args.load_atlas, args.save_atlas):
        if atlas_path and atlas_path.suffix != ".osa":
            parser.error("Atlas paths must end in .osa")
    if args.load_atlas and not args.load_atlas.is_file():
        parser.error("--load-atlas does not exist")
    if args.load_atlas and args.save_atlas and args.load_atlas.resolve() == args.save_atlas.resolve():
        parser.error("input Atlas must not be overwritten")
    if args.save_atlas and args.save_atlas.exists():
        parser.error("--save-atlas already exists; choose a new output")
    if args.head_slam and not args.world_board:
        args.auto_marker_map = True
    head_slam_enabled = args.head_slam or args.auto_marker_map
    fusion_enabled = head_slam_enabled
    if args.reuse_observations and not head_slam_enabled:
        parser.error('--reuse-observations requires --head-slam or --auto-marker-map')
    if not args.band:
        raise SystemExit("at least one --band is required")
    if args.world_board and args.auto_marker_map:
        raise SystemExit("--world-board and --auto-marker-map are mutually exclusive")
    if fusion_enabled and not (args.world_board or args.auto_marker_map):
        raise SystemExit(
            "--head-slam requires --world-board or --auto-marker-map"
        )
    if args.static_marker_size_mm <= 0:
        raise SystemExit("--static-marker-size-mm must be positive")
    if (
        args.slam_debug_video is not None or args.graph_diagnostics is not None
        or args.open_replay or args.load_atlas or args.save_atlas
    ) and not fusion_enabled:
        raise SystemExit(
            "--slam-debug-video and --graph-diagnostics require --head-slam"
        )
    if not args.slam_replay and (args.open_replay or args.slam_debug_video is not None):
        parser.error("--open-replay/--slam-debug-video require --slam-replay")
    if args.lerobot_output is not None:
        if not args.lerobot_task:
            parser.error("--lerobot-output requires --lerobot-task")
        if args.lerobot_fps <= 0:
            parser.error("--lerobot-fps must be positive")
        if args.lerobot_output.exists():
            parser.error(f"LeRobot output already exists: {args.lerobot_output}")
    elif args.lerobot_task or args.lerobot_repo_id:
        parser.error("--lerobot-task/--lerobot-repo-id require --lerobot-output")

    video_path = Path(args.video)
    output_path = Path(args.output) if args.output else video_path.with_name(
        f"{video_path.stem}_actions.jsonl"
    )
    metadata_path = output_path.with_suffix(".meta.json")
    if args.reuse_observations and output_path.resolve() == args.reuse_observations.resolve():
        parser.error('observation cache input must not be overwritten; choose a new --output')
    replay_dir = output_path.with_name(output_path.stem + "_replay")
    save_atlas = args.save_atlas or replay_dir / "atlas.osa"
    if fusion_enabled and save_atlas.exists():
        parser.error(f"Atlas output exists: {save_atlas}; choose a new --output / --save-atlas")
    if fusion_enabled:
        save_atlas.parent.mkdir(parents=True, exist_ok=True)
    # The default video belongs to this output package, not a shared alias
    # beside the input video (which collides when reprocessing the same clip).
    debug_video_path = (replay_dir / "process.mp4" if args.slam_debug_video == ""
                        else _optional_output_path(args.slam_debug_video, video_path, "_slam_process.mp4"))
    if (debug_video_path is not None
            and debug_video_path.resolve() != (replay_dir / "process.mp4").resolve()
            and (debug_video_path.exists() or debug_video_path.is_symlink())):
        parser.error(f"debug video output exists: {debug_video_path}; choose a new path")
    orb_map_video_path: Path | None = None
    orb_map_viewer_path: Path | None = None
    diagnostics_path = _optional_output_path(
        args.graph_diagnostics, output_path, "_graph_diagnostics.jsonl"
    )
    marker_map_path = (
        Path(args.marker_map_output)
        if args.marker_map_output
        else output_path.with_name(f"{output_path.stem}_marker_map.json")
    )
    calibration = Calibration.load(args.calib)
    bands = [BandLayout.load(path) for path in args.band]
    if args.auto_marker_map:
        wrist_marker_ids = set().union(*(set(band.markers) for band in bands))
        overlap = wrist_marker_ids.intersection(args.static_marker_ids)
        if overlap:
            raise SystemExit(
                "static marker IDs overlap wrist bands: "
                + ",".join(str(marker_id) for marker_id in sorted(overlap))
            )
    world_board = BandLayout.load(args.world_board) if args.world_board else None
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"cannot open {video_path}")
    video_size = (
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    if video_size != calibration.image_size:
        try:
            calibration = calibration.scaled_to(video_size)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    cached_records, cached_metadata = None, None
    if args.reuse_observations:
        try:
            cached_records, cached_metadata = _load_observation_cache(
                args.reuse_observations, video_path, args.calib, args.band,
                args.world_board, fps, video_size, args.hand_model,
                args.min_hand_confidence, args.hand_joints,
                allow_hand_upgrade=True, hand_backend=args.hand_backend)
            cached_auto_map = cached_metadata.get('auto_marker_map', {})
            if bool(cached_auto_map.get('enabled')) != bool(args.auto_marker_map):
                raise ValueError('observation cache fixed-marker mode differs; run fresh analysis')
            if args.auto_marker_map:
                if set(cached_auto_map.get('static_marker_ids', ())) != args.static_marker_ids:
                    raise ValueError('observation cache static marker IDs differ; run fresh analysis')
                if not np.isclose(
                    cached_auto_map.get('marker_size_mm', np.nan),
                    args.static_marker_size_mm,
                ):
                    raise ValueError('observation cache static marker size differs; run fresh analysis')
            cached_mode = cached_metadata.get('marker_corner_policy', {}).get('mode')
            if cached_mode != ('strict' if args.strict_marker_corners else 'weighted'):
                raise ValueError('observation cache marker quality policy differs; run fresh analysis')
        except (ValueError, KeyError, OSError) as exc:
            capture.release()
            parser.error(str(exc))
        if args.max_frames is not None:
            cached_records = cached_records[:args.max_frames]
    cached_hand_upgrade = bool(
        cached_records is not None
        and _hand_cache_requires_upgrade(
            cached_metadata, args.hand_joints, args.min_hand_confidence, args.hand_backend)
    )
    cached_hand_fresh = bool(cached_records is not None and args.hand_joints and (
        not cached_metadata.get('hand_joints_enabled', True)
        or cached_metadata.get('hand_backend', 'mediapipe') != args.hand_backend))
    pipeline = None if cached_records is not None else TrackingPipeline(
        calibration,
        bands,
        world_board=world_board,
        boundary_marker_ids=args.static_marker_ids if args.auto_marker_map else None,
        allow_soft_marker_corners=not args.strict_marker_corners,
    )
    world_marker_tracker = MarkerPoseTracker(world_board, calibration) if world_board else None
    tracker = None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    stream_path = output_path
    if fusion_enabled:
        temporary = tempfile.NamedTemporaryFile(
            prefix=f".{output_path.stem}.",
            suffix=".slam-observations.jsonl",
            dir=output_path.parent,
            delete=False,
        )
        temporary.close()
        temporary_path = Path(temporary.name)
        # Native SLAM, trajectory optimization, and replay rendering all run
        # after observation extraction. If any of them raises before normal
        # cleanup, remove this recoverable scratch file at interpreter exit.
        atexit.register(temporary_path.unlink, missing_ok=True)
        stream_path = temporary_path
    frames = 0
    joint_frames: Counter[str] = Counter()
    fused_frames: Counter[str] = Counter()
    world_fused_frames: Counter[str] = Counter()
    selected_frame: dict[str, str | None] = {band.name: None for band in bands}
    segments: Counter[str] = Counter()
    orbslam3_result: MetricOrbSlamResult | None = None
    head_slam_map_points: list[int] = []
    head_slam_keyframes: list[int] = []
    marker_camera_poses: list[Pose | None] = []
    marker_confidences: list[float] = []
    slam_observations: list[object] = []
    board_detections: list[dict[int, np.ndarray]] = []
    board_assist_detections: list[dict[int, np.ndarray]] = []
    board_weights: list[dict[int, float]] = []
    board_accepted_ids: list[tuple[int, ...]] = []
    board_rejected_ids: list[tuple[int, ...]] = []
    world_graph_results: list[TagPoseResult] = []
    all_detections: list[dict[int, np.ndarray]] = []
    nondecoded_marker_ids: list[set[int]] = []
    wrist_graph_poses: dict[str, list[Pose | None]] = {
        band.name: [] for band in bands
    }
    wrist_graph_results: dict[str, list[TagPoseResult]] = {
        band.name: [] for band in bands
    }
    wrist_graph_accepted_ids: dict[str, list[tuple[int, ...]]] = {
        band.name: [] for band in bands
    }
    wrist_graph_assist_detections: dict[str, list[dict[int, np.ndarray]]] = {
        band.name: [] for band in bands
    }
    frame_timestamps_s: list[float] = []
    predicted_wrist_graph: dict[str, Pose] = {}
    hand_backend_provenance = None
    try:
        if args.hand_joints and args.hand_backend == 'hawor':
            prediction_path, hand_backend_provenance = _prepare_export_hawor_predictions(
                video_path, args.calib, output_path.with_name(f'{output_path.stem}.hawor_observations.jsonl'),
                max_frames=(len(cached_records) if cached_records is not None else args.max_frames),
                config_path=args.hawor_config, device=args.hawor_device,
                cached_metadata=cached_metadata)
            tracker = HaworHandTracker(prediction_path, calibration, [band.name for band in bands], fps)
        else:
            tracker = (
            HandJointTracker(
                args.hand_model,
                calibration,
                [band.name for band in bands],
                args.min_hand_confidence,
                initialize_full_frame_detector=(cached_records is None or cached_hand_fresh),
            )
            if args.hand_joints
            else None
            )
        with stream_path.open("w") as stream:
            while cached_records is None and (args.max_frames is None or frames < args.max_frames):
                ok, frame = capture.read()
                if not ok:
                    break
                timestamp_s = frames / fps
                frame_timestamps_s.append(timestamp_s)
                wrist = pipeline.process(frame)
                world_graph = None
                band_graph: dict[str, TagPoseResult] = {}
                if fusion_enabled:
                    assist_detections = _assist_only_detections(
                        wrist.rejected_detections, wrist.boundary_quality
                    )
                    if world_board is not None:
                        world_graph = world_marker_tracker.update(
                            wrist.detections, timestamp_s,
                            marker_weights={mid: quality.information_weight for mid, quality in
                                            wrist.boundary_quality.items()},
                            assist_detections=assist_detections,
                        )
                    for band in bands:
                        result = optimize_tag_pose(
                            wrist.detections,
                            band,
                            calibration,
                            predicted_wrist_graph.get(band.name),
                            validate_planar_ambiguity=True,
                            assist_detections=assist_detections,
                        )
                        band_graph[band.name] = result
                        if result.pose is not None:
                            predicted_wrist_graph[band.name] = result.pose
                        else:
                            # A rejected/occluded wrist is not a measurement.
                            # Do not let a stale camera-frame prior bias reacquisition.
                            predicted_wrist_graph.pop(band.name, None)
                wrist_poses_for_hands = {
                    band.name: (
                        band_graph[band.name].pose
                        if fusion_enabled and band_graph[band.name].pose is not None
                        else wrist.raw_poses.get(band.name)
                    )
                    for band in bands
                }
                joints = (
                    tracker.process(
                        frame,
                        round(timestamp_s * 1000.0),
                        {
                            name: pose
                            for name, pose in wrist_poses_for_hands.items()
                            if pose is not None
                        },
                        (
                            world_graph.pose
                            if fusion_enabled and world_graph is not None
                            else wrist.world_reference
                        ),
                    )
                    if tracker is not None
                    else {}
                )
                edge = None
                if fusion_enabled:
                    marker_pose = (
                        inverse_pose(world_graph.pose)
                        if world_graph is not None and world_graph.pose is not None
                        else None
                    )
                    marker_camera_poses.append(marker_pose)
                    marker_confidences.append(
                        world_graph.confidence
                        if marker_pose is not None and world_graph is not None
                        else 0.0
                    )
                    slam_observations.append(edge)
                    static_marker_ids = (
                        set(world_board.markers)
                        if world_board is not None
                        else args.static_marker_ids
                    )
                    board_detections.append(
                        {
                            marker_id: corners
                            for marker_id, corners in wrist.detections.items()
                            if marker_id in static_marker_ids
                        }
                    )
                    board_assist_detections.append({
                        marker_id: corners
                        for marker_id, corners in assist_detections.items()
                        if marker_id in static_marker_ids
                    })
                    board_weights.append({
                        mid: quality.information_weight for mid, quality in wrist.boundary_quality.items()
                        if mid in static_marker_ids
                    })
                    resolved_world_graph = world_graph or _empty_tag_result()
                    board_accepted_ids.append(
                        resolved_world_graph.accepted_marker_ids
                    )
                    board_rejected_ids.append(
                        resolved_world_graph.rejected_marker_ids
                    )
                    world_graph_results.append(resolved_world_graph)
                    all_detections.append(wrist.detections)
                    nondecoded_marker_ids.append(set(wrist.recovered_ids) | set(wrist.tracked_ids))
                    for band in bands:
                        wrist_graph_results[band.name].append(
                            band_graph[band.name]
                        )
                        wrist_graph_poses[band.name].append(
                            band_graph[band.name].pose
                        )
                        wrist_graph_accepted_ids[band.name].append(
                            band_graph[band.name].accepted_marker_ids
                        )
                        wrist_graph_assist_detections[band.name].append({
                            marker_id: corners
                            for marker_id, corners in assist_detections.items()
                            if marker_id in band.markers
                        })
                hands: dict[str, object] = {}
                internal_wrist_graph: dict[str, object] = {}
                for band in bands:
                    joint = joints.get(band.name)
                    camera_pose = wrist.poses.get(band.name)
                    world_pose = wrist.world_poses.get(band.name)
                    if world_pose is not None:
                        coordinate_frame = "world"
                        selected_pose = world_pose
                    elif camera_pose is not None:
                        coordinate_frame = "camera"
                        selected_pose = camera_pose
                    else:
                        coordinate_frame = "invalid"
                        selected_pose = None
                    if coordinate_frame == "invalid":
                        selected_frame[band.name] = None
                    elif coordinate_frame != selected_frame[band.name]:
                        segments[band.name] += 1
                        selected_frame[band.name] = coordinate_frame
                    joint_frames[band.name] += int(joint is not None)
                    fused_frames[band.name] += int(
                        joint is not None and joint.band_landmarks_m is not None
                    )
                    world_fused_frames[band.name] += int(
                        joint is not None and joint.world_landmarks_m is not None
                    )
                    hand_record = {
                        "wrist_precision_qualified": None,
                        "selected_coordinate_frame": coordinate_frame,
                        "trajectory_segment": (
                            segments[band.name] if selected_pose is not None else None
                        ),
                        "wrist_camera_raw": pose_to_dict(wrist.raw_poses.get(band.name)),
                        "wrist_camera_filtered": pose_to_dict(camera_pose),
                        "wrist_world_raw": pose_to_dict(
                            wrist.raw_world_poses.get(band.name)
                        ),
                        "wrist_world_filtered": pose_to_dict(world_pose),
                        "wrist_selected_filtered": pose_to_dict(selected_pose),
                        "joints": (
                            joint_pose_to_dict(joint)
                            if joint is not None
                            else {"valid": False, "wrist_anchor_valid": False}
                        ),
                    }
                    if fusion_enabled:
                        graph_result = band_graph[band.name]
                        hand_record.update(
                            {
                                "wrist_camera_graph": pose_to_dict(graph_result.pose),
                                **_tag_result_dict(graph_result),
                            }
                        )
                        internal_wrist_graph[band.name] = _pose_to_internal_dict(
                            graph_result.pose
                        )
                    hands[band.name] = hand_record
                unassigned = [
                    joint_pose_to_dict(joint)
                    for name, joint in joints.items()
                    if name.startswith("unassigned_")
                ]
                record = {
                    "frame": frames,
                    "timestamp_s": timestamp_s,
                    "world_reference": pose_to_dict(wrist.world_reference),
                    "camera_world_pose": pose_to_dict(wrist.camera_world_pose),
                    "recovered_marker_ids": list(wrist.recovered_ids),
                    "refined_static_marker_ids": list(pipeline.detector.last_refined_ids),
                    "marker_mask_corners": {str(mid): corners.tolist() for mid,corners in
                                            pipeline.detector.last_mask_detections.items()},
                    "optical_flow_marker_ids": list(wrist.tracked_ids),
                    "detected_marker_corners": {str(mid): corners.tolist() for mid, corners in wrist.detections.items()},
                    "boundary_rejected_marker_corners": {
                        str(marker_id): corners.tolist()
                        for marker_id, corners in wrist.rejected_detections.items()
                    },
                    "marker_boundary_quality": {
                        str(marker_id): asdict(quality)
                        for marker_id, quality in wrist.boundary_quality.items()
                    },
                    "hands": hands,
                    "unassigned_hands": unassigned,
                }
                if fusion_enabled:
                    resolved_world_graph = world_graph or _empty_tag_result()
                    record.update(
                        {
                            **_tag_result_dict(resolved_world_graph),
                            "__graph_wrist_camera": internal_wrist_graph,
                        }
                    )
                stream.write(
                    json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n"
                )
                frames += 1
    finally:
        if tracker is not None and cached_records is None:
            tracker.close()
        if not cached_hand_upgrade:
            capture.release()

    if cached_records is not None:
        cached_raw_poses: dict[str, Pose] = {}
        cached_camera_smoothers: dict[str, AdaptivePoseSmoother] = {}
        cached_world_smoothers: dict[str, WorldPoseSmoother] = {}
        cached_hand_assignment_gate = TemporalHandAssignmentGate(
            [band.name for band in bands]
        )
        with stream_path.open('w') as stream:
            for record in cached_records:
                frame = None
                if cached_hand_upgrade:
                    ok, frame = capture.read()
                    if not ok:
                        raise RuntimeError(
                            f'source video ended before cached frame {record["frame"]}'
                        )
                timestamp_s = float(record['timestamp_s'])
                frame_timestamps_s.append(timestamp_s)
                detections = {int(mid): np.asarray(c, dtype=float)
                              for mid, c in record['detected_marker_corners'].items()}
                rejected_detections = {
                    int(mid): np.asarray(c, dtype=float)
                    for mid, c in record['boundary_rejected_marker_corners'].items()
                }
                boundary_qualities = {
                    int(mid): quality
                    for mid, quality in record['marker_boundary_quality'].items()
                }
                assist_detections = _assist_only_detections(
                    rejected_detections, boundary_qualities
                )
                weights = {int(mid): q['information_weight'] for mid, q in record['marker_boundary_quality'].items()}
                # Reuse image/hand measurements, not an old PnP admission
                # decision that can bypass current planar-ambiguity checks.
                world_graph = (
                    world_marker_tracker.update(
                        detections, record['timestamp_s'], weights,
                        assist_detections=assist_detections,
                    )
                    if world_marker_tracker is not None
                    else _empty_tag_result()
                )
                marker = inverse_pose(world_graph.pose) if world_graph.pose is not None else None
                marker_camera_poses.append(marker)
                marker_confidences.append(world_graph.confidence)
                all_detections.append(detections)
                nondecoded_marker_ids.append(
                    set(record.get('recovered_marker_ids', ()))
                    | set(record.get('optical_flow_marker_ids', ()))
                )
                static_marker_ids = (
                    set(world_board.markers)
                    if world_board is not None
                    else args.static_marker_ids
                )
                board_detections.append({
                    mid: corners for mid, corners in detections.items()
                    if mid in static_marker_ids
                })
                board_assist_detections.append({
                    mid: corners for mid, corners in assist_detections.items()
                    if mid in static_marker_ids
                })
                board_weights.append({
                    mid: weight for mid, weight in weights.items()
                    if mid in static_marker_ids
                })
                board_accepted_ids.append(world_graph.accepted_marker_ids)
                board_rejected_ids.append(world_graph.rejected_marker_ids)
                world_graph_results.append(world_graph)
                record.update(_tag_result_dict(world_graph))
                record['marker_camera_pose_observed'] = pose_to_dict(marker)
                record['marker_camera_confidence'] = world_graph.confidence
                record['world_reference'] = pose_to_dict(world_graph.pose)
                record['camera_world_pose'] = pose_to_dict(marker)
                internal = {}
                solved_bands: dict[str, dict[str, object]] = {}
                for band in bands:
                    if cached_hand_fresh:
                        # The observation-cache version and all input
                        # fingerprints were validated above. Preserve these
                        # exact camera-space wrist measurements while adding
                        # only the previously omitted hand network output.
                        cached_hand = record['hands'][band.name]
                        graph_result = _cached_tag_result(cached_hand)
                        solved_bands[band.name] = {
                            "raw_pose": _pose_from_output_dict(
                                cached_hand.get("wrist_camera_raw")
                            ),
                            "camera_pose": _pose_from_output_dict(
                                cached_hand.get("wrist_camera_filtered")
                            ),
                            "graph_result": graph_result,
                            "graph_pose": graph_result.pose,
                            "raw_world_pose": _pose_from_output_dict(
                                cached_hand.get("wrist_world_raw")
                            ),
                            "world_pose": _pose_from_output_dict(
                                cached_hand.get("wrist_world_filtered")
                            ),
                        }
                        continue
                    raw_pose = solve_band_pose(
                        detections,
                        band,
                        calibration.camera_matrix,
                        calibration.dist_coeffs,
                        cached_raw_poses.get(band.name),
                    )
                    if raw_pose is not None:
                        cached_raw_poses[band.name] = raw_pose
                        camera_smoother = cached_camera_smoothers.setdefault(
                            band.name, AdaptivePoseSmoother()
                        )
                        camera_pose = camera_smoother.update(raw_pose)
                    else:
                        cached_raw_poses.pop(band.name, None)
                        cached_camera_smoothers.pop(band.name, None)
                        cached_world_smoothers.pop(band.name, None)
                        camera_pose = None
                    graph_result = optimize_tag_pose(
                        detections,
                        band,
                        calibration,
                        predicted_wrist_graph.get(band.name),
                        validate_planar_ambiguity=True,
                        assist_detections=assist_detections,
                    )
                    graph_pose = graph_result.pose
                    if graph_pose is not None:
                        predicted_wrist_graph[band.name] = graph_pose
                    else:
                        predicted_wrist_graph.pop(band.name, None)
                    raw_world_pose = (
                        relative_pose(world_graph.pose, raw_pose)
                        if world_graph.pose is not None and raw_pose is not None
                        else None
                    )
                    if raw_world_pose is not None:
                        world_smoother = cached_world_smoothers.setdefault(
                            band.name, WorldPoseSmoother()
                        )
                        world_pose = world_smoother.update(raw_world_pose)
                    else:
                        # Match the fresh pipeline: a missing world marker does
                        # not erase filter history while the wrist itself is
                        # still observed. Reset only when the wrist is lost
                        # (handled above), so marker reacquisition is identical.
                        world_pose = None
                    solved_bands[band.name] = {
                        "raw_pose": raw_pose,
                        "camera_pose": camera_pose,
                        "graph_result": graph_result,
                        "graph_pose": graph_pose,
                        "raw_world_pose": raw_world_pose,
                        "world_pose": world_pose,
                    }
                hand_band_poses = {
                    name: values["graph_pose"] or values["raw_pose"]
                    for name, values in solved_bands.items()
                    if values["graph_pose"] is not None
                    or values["raw_pose"] is not None
                }
                if tracker is not None:
                    if cached_hand_fresh:
                        detected_joints = tracker.process(
                            frame, round(timestamp_s * 1000.0),
                            hand_band_poses, world_graph.pose)
                    else:
                        detected_joints = tracker.process_observations(
                            frame, round(timestamp_s * 1000.0),
                            _cached_raw_hands(record), hand_band_poses, world_graph.pose)
                    assigned_joints = {
                        name: joint_pose_to_dict(joint)
                        for name, joint in detected_joints.items()
                    }
                else:
                    assigned_joints = _reassign_cached_joints(
                        record,
                        [band.name for band in bands],
                        hand_band_poses,
                        calibration,
                        cached_hand_assignment_gate,
                    )
                current_hands: dict[str, object] = {}
                for band in bands:
                    values = solved_bands[band.name]
                    raw_pose = values["raw_pose"]
                    camera_pose = values["camera_pose"]
                    graph_result = values["graph_result"]
                    graph_pose = values["graph_pose"]
                    raw_world_pose = values["raw_world_pose"]
                    world_pose = values["world_pose"]
                    if world_pose is not None:
                        coordinate_frame, selected_pose = 'world', world_pose
                    elif camera_pose is not None:
                        coordinate_frame, selected_pose = 'camera', camera_pose
                    else:
                        coordinate_frame, selected_pose = 'invalid', None
                    if coordinate_frame == 'invalid':
                        selected_frame[band.name] = None
                    elif coordinate_frame != selected_frame[band.name]:
                        segments[band.name] += 1
                        selected_frame[band.name] = coordinate_frame
                    joints = assigned_joints.get(
                        band.name,
                        {"valid": False, "wrist_anchor_valid": False},
                    )
                    hand = {
                        "selected_coordinate_frame": coordinate_frame,
                        "trajectory_segment": (
                            segments[band.name]
                            if selected_pose is not None
                            else None
                        ),
                        "wrist_camera_raw": pose_to_dict(raw_pose),
                        "wrist_camera_filtered": pose_to_dict(camera_pose),
                        "wrist_world_raw": pose_to_dict(raw_world_pose),
                        "wrist_world_filtered": pose_to_dict(world_pose),
                        "wrist_selected_filtered": pose_to_dict(selected_pose),
                        "joints": joints,
                    }
                    _replace_cached_graph_measurement(hand, graph_result)
                    binding_pose = graph_pose if graph_pose is not None else raw_pose
                    _rebind_cached_joints(
                        joints, binding_pose, world_graph.pose, calibration
                    )
                    current_hands[band.name] = hand
                    internal[band.name] = _pose_to_internal_dict(graph_pose)
                    wrist_graph_poses[band.name].append(graph_pose)
                    wrist_graph_results[band.name].append(graph_result)
                    wrist_graph_accepted_ids[band.name].append(
                        graph_result.accepted_marker_ids
                    )
                    wrist_graph_assist_detections[band.name].append({
                        marker_id: corners
                        for marker_id, corners in assist_detections.items()
                        if marker_id in band.markers
                    })
                    joint_frames[band.name] += int(bool(joints.get('valid')))
                    fused_frames[band.name] += int(joints.get('band_landmarks_m') is not None)
                    world_fused_frames[band.name] += int(joints.get('world_landmarks_m') is not None)
                current_unassigned = [
                    joints
                    for name, joints in assigned_joints.items()
                    if name.startswith("unassigned_")
                ]
                for joints in current_unassigned:
                    _rebind_cached_joints(joints, None, None, calibration)
                record['hands'] = current_hands
                record['unassigned_hands'] = current_unassigned
                record['__graph_wrist_camera'] = internal
                stream.write(json.dumps(record, separators=(',', ':'), allow_nan=False)+'\n')
        frames = len(cached_records)
        if tracker is not None:
            tracker.close()
        if cached_hand_upgrade:
            capture.release()
        if cached_hand_upgrade:
            print(
                f'reused {frames} cached ArUco observations; upgraded hand recovery '
                f'from source RGB using {args.hand_backend}; '
                'marker map and native SLAM recomputed'
            )
        else:
            print(
                f'reused {frames} cached ArUco/hand observations; '
                'marker map and native SLAM recomputed'
            )

    camera_sources: Counter[str] = Counter()
    graph_world_frames: Counter[str] = Counter()
    graph_world_joint_frames: Counter[str] = Counter()
    dual_world_wrist_frames = 0
    dual_world_joint_frames = 0
    fused_camera = []
    auto_marker_map: AutoMarkerMap | None = None
    auto_marker_map_sampling_stride = 1
    auto_marker_map_sample_count = frames
    camera_submap_ids: list[str | None] = [None] * frames
    submap_anchor_ids: dict[str, int | None] = {}
    marker_corner_diagnostics: list[list[dict]] = [[] for _ in range(frames)]
    marker_pose_uncertainties: list[dict | None] = [None for _ in range(frames)]
    marker_temporal_diagnostics: list[dict[int, dict]] = [{} for _ in range(frames)]
    marker_temporal_summary: dict = {"enabled": False}
    excluded_static_marker_ids: list[set[int]] = [set() for _ in range(frames)]
    if fusion_enabled:
        assert temporary_path is not None
        from aruco_track.marker_temporal_admission import review_static_marker_sequence

        static_marker_ids = set(world_board.markers) if world_board is not None else args.static_marker_ids
        temporal_review = review_static_marker_sequence(
            all_detections, frame_timestamps_s, calibration, static_marker_ids,
            marker_weights=board_weights, nondecoded_ids=nondecoded_marker_ids,
        )
        # Raw JSONL records remain immutable measurements. Every static factor
        # consumer below uses this shared fresh/reuse admission result instead.
        all_detections = temporal_review.detections
        marker_temporal_diagnostics = temporal_review.diagnostics
        marker_temporal_summary = temporal_review.summary
        excluded_static_marker_ids = temporal_review.excluded_ids
        board_detections = [
            {mid: corners for mid, corners in frame.items() if mid in static_marker_ids}
            for frame in all_detections
        ]
        board_assist_detections = [
            {mid: corners for mid, corners in frame.items() if mid not in excluded}
            for frame, excluded in zip(board_assist_detections, excluded_static_marker_ids)
        ]
        board_weights = [
            {mid: weight for mid, weight in weights.items() if mid not in excluded}
            for weights, excluded in zip(board_weights, excluded_static_marker_ids)
        ]
        if world_board is not None:
            world_marker_tracker = MarkerPoseTracker(world_board, calibration)
            world_graph_results = [
                world_marker_tracker.update(
                    frame, timestamp, marker_weights=weights, assist_detections=assist,
                )
                for frame, timestamp, weights, assist in zip(
                    board_detections, frame_timestamps_s, board_weights, board_assist_detections
                )
            ]
            marker_camera_poses = [
                inverse_pose(result.pose) if result.pose is not None else None
                for result in world_graph_results
            ]
            marker_confidences = [result.confidence for result in world_graph_results]
            board_accepted_ids = [result.accepted_marker_ids for result in world_graph_results]
            board_rejected_ids = [result.rejected_marker_ids for result in world_graph_results]
        if args.auto_marker_map:
            reliable_map_detections = [
                {
                    mid: corners
                    for mid, corners in frame.items()
                    if weights.get(mid, 1.0) >= 1.0
                }
                for frame, weights in zip(board_detections, board_weights)
            ]
            (
                sampled_map_detections,
                _,
                auto_marker_map_sampling_stride,
            ) = _sample_marker_map_frames(reliable_map_detections, fps)
            auto_marker_map_sample_count = len(sampled_map_detections)
            auto_marker_map = build_auto_marker_map(
                # Weak observations can localize against established geometry,
                # but cannot define the world origin or calibrate marker layout.
                sampled_map_detections,
                calibration,
                args.static_marker_ids,
                args.static_marker_size_mm / 1000.0,
                bands[0].dictionary,
            )
            auto_marker_map.save(marker_map_path)
            localized = localize_auto_marker_frames(
                auto_marker_map, board_detections, calibration, marker_weights=board_weights,
                assist_detections=board_assist_detections,
                fps=fps,
            )
            world_graph_results = [
                result or _empty_tag_result() for result in localized.results
            ]
            marker_camera_poses = [
                inverse_pose(result.pose)
                if result is not None and result.pose is not None
                else None
                for result in localized.results
            ]
            marker_confidences = [
                result.confidence if result is not None else 0.0
                for result in localized.results
            ]
            board_accepted_ids = [
                result.accepted_marker_ids if result is not None else ()
                for result in localized.results
            ]
            board_rejected_ids = [
                result.rejected_marker_ids if result is not None else ()
                for result in localized.results
            ]
            camera_submap_ids = list(localized.submap_ids)
            layouts = {
                submap.submap_id: submap.layout
                for submap in auto_marker_map.submaps
            }
            submap_anchor_ids = {
                submap.submap_id: submap.anchor_marker_id
                for submap in auto_marker_map.submaps
            }
            session_map_id = camera_submap_ids[0] if camera_submap_ids else None
        else:
            assert world_board is not None
            session_map_id = "world_board"
            layouts = {"world_board": world_board}
            camera_submap_ids = ["world_board"] * frames
            submap_anchor_ids = {"world_board": None}

        # Quantify the raw marker measurement before SLAM can improve or reject
        # it.  This keeps the reported precision tied to actual accepted image
        # corners and makes poor geometry auditable per frame.
        for frame_index, result in enumerate(world_graph_results):
            observation_map_id = camera_submap_ids[frame_index]
            observation_layout = layouts.get(observation_map_id)
            corners, uncertainty = marker_pose_uncertainty(
                result.pose,
                observation_layout,
                board_detections[frame_index],
                result.accepted_marker_ids,
                calibration,
                board_weights[frame_index],
            )
            marker_corner_diagnostics[frame_index] = corners
            marker_pose_uncertainties[frame_index] = uncertainty

        if head_slam_enabled:
            orbslam3_result = _run_deferred_head_slam(
                video_path,
                temporary_path,
                calibration,
                all_detections,
                marker_camera_poses,
                marker_confidences,
                board_accepted_ids,
                layouts.get(session_map_id),
                camera_submap_ids,
                replay_dir, args.slam_init, args.load_atlas, save_atlas,
                marker_weights=board_weights,
                dynamic_geometry=args.slam_dynamic_filter,
                rigid_marker_layout=not args.auto_marker_map,
                marker_layouts=layouts if args.auto_marker_map else None,
                weak_marker_corners=args.slam_weak_marker_corners,
                slam_replay=args.slam_replay,
                excluded_marker_ids=excluded_static_marker_ids,
            )
            fused_camera = orbslam3_result.frames
            bootstrap_rows = {
                row["frame_id"]: row for row in
                (orbslam3_result.offline_marker_bootstrap or {}).get("published_observations", [])
            }
            slam_observations = list(orbslam3_result.observations)
            camera_submap_ids = [frame.map_id for frame in fused_camera]
            native_frames = {round(h["timestamp"] * fps): h
                             for h in orbslam3_result.history if not h.get("final")}
            head_slam_map_points, head_slam_keyframes = [], []
            for index in range(frames):
                h = native_frames.get(index, {})
                m = next((m for m in h.get("maps", []) if m["id"] == h.get("active_map")), {})
                head_slam_map_points.append(
                    int(m.get("point_count", len(m.get("points", []))))
                )
                head_slam_keyframes.append(int(
                    m.get("keyframe_count", len(m.get("keyframes", [])))
                ))
            for map_id, m in orbslam3_result.maps.items():
                if m["metric"]:
                    layouts[map_id] = BandLayout(map_id, bands[0].dictionary,
                        {int(mid): np.asarray(corners).reshape(4, 3)
                         for mid, corners in m["markers"].items()})
            if session_map_id in layouts:
                layouts["marker_world"] = layouts[session_map_id]
        for band in bands:
            refined_results = refine_wrist_pose_sequence(
                [frame.pose for frame in fused_camera],
                [frame.metric for frame in fused_camera],
                camera_submap_ids,
                frame_timestamps_s,
                all_detections,
                wrist_graph_results[band.name],
                band,
                calibration,
                wrist_graph_assist_detections[band.name],
            )
            wrist_graph_results[band.name] = refined_results
            wrist_graph_poses[band.name] = [result.pose for result in refined_results]
            wrist_graph_accepted_ids[band.name] = [
                result.accepted_marker_ids for result in refined_results
            ]
        wrist_optimization_started = time.perf_counter()
        def optimize_band_trajectory(band: BandLayout) -> WristTrajectoryResult:
            return _optimize_wrists_by_submap(
                fused_camera,
                camera_submap_ids,
                frame_timestamps_s,
                fps,
                wrist_graph_poses[band.name],
                all_detections,
                wrist_graph_accepted_ids[band.name],
                wrist_graph_assist_detections[band.name],
                band,
                calibration,
            )

        # Left and right trajectories have no shared optimizer state.  Solve
        # them concurrently so their existing per-window process pools can use
        # the remaining performance cores instead of creating two pools in
        # sequence.  Keep the single-band path free of executor overhead.
        if len(bands) > 1:
            with ThreadPoolExecutor(max_workers=min(2, len(bands))) as executor:
                futures = {
                    band.name: executor.submit(optimize_band_trajectory, band)
                    for band in bands
                }
                wrist_trajectories = {
                    band.name: futures[band.name].result() for band in bands
                }
        else:
            wrist_trajectories = {
                band.name: optimize_band_trajectory(band) for band in bands
            }
        if orbslam3_result is not None:
            orbslam3_result.timing['wrist_trajectory_seconds'] = (
                time.perf_counter() - wrist_optimization_started
            )
        world_tracking = {}
        for band in bands:
            world_tracking[band.name] = _track_world_by_submap(
                fused_camera,
                camera_submap_ids,
                wrist_graph_poses[band.name],
                wrist_trajectories[band.name].poses,
            )
        graph_selected_frame: dict[str, str | None] = {
            band.name: None for band in bands
        }
        graph_segments: Counter[str] = Counter()
        diagnostics_stream = None
        if diagnostics_path is not None:
            diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
            diagnostics_stream = diagnostics_path.open("w")
        try:
            with temporary_path.open() as source, output_path.open("w") as stream:
                for line in source:
                    record = json.loads(line)
                    frame_index = int(record["frame"])
                    fused = fused_camera[frame_index]
                    submap_id = camera_submap_ids[frame_index]
                    world_graph_result = world_graph_results[frame_index]
                    record.pop("__graph_wrist_camera")
                    record.update(_tag_result_dict(world_graph_result))
                    record["marker_temporal_admission"] = {
                        str(mid): diagnostic
                        for mid, diagnostic in marker_temporal_diagnostics[frame_index].items()
                    }
                    record["assist_only_marker_ids"] = sorted(
                        board_assist_detections[frame_index]
                    )
                    record["marker_camera_pose_observed"] = pose_to_dict(marker_camera_poses[frame_index])
                    record["marker_camera_confidence"] = marker_confidences[frame_index]
                    record["marker_pose_offline_evidence"] = (
                        bootstrap_rows.get(frame_index) if head_slam_enabled else None
                    )
                    record["camera_world_pose_fused"] = pose_to_dict(fused.pose)
                    record["camera_world_source"] = fused.source
                    record["camera_world_confidence"] = fused.confidence
                    record["camera_anchor_consistency"] = fused.anchor_consistency
                    record["camera_submap_id"] = submap_id
                    record["world_frame_id"] = submap_id if fused.metric else None
                    record["map_revision"] = fused.revision
                    record["scale_status"] = "metric" if fused.metric else "unscaled"
                    record["camera_metric_recovered_later"] = (
                        fused.metric_recovered_later
                    )
                    record["camera_localization_recovery"] = fused.localization_recovery
                    record["initialization_source"] = fused.initialization_source
                    record["background_map_ready"] = fused.background_ready
                    record["slam_backend"] = "native-orb-slam3"
                    record["camera_submap_anchor_marker_id"] = (
                        submap_anchor_ids.get(submap_id)
                        if submap_id is not None
                        else None
                    )
                    record["slam_inliers"] = fused.slam_inliers
                    record["graph_reprojection_error_px"] = (
                        fused.graph_reprojection_error_px
                    )
                    record["marker_corner_diagnostics"] = (
                        marker_corner_diagnostics[frame_index]
                    )
                    record["marker_pose_uncertainty"] = (
                        marker_pose_uncertainties[frame_index]
                    )
                    if head_slam_enabled:
                        native = native_frames.get(frame_index, {})
                        final_map = orbslam3_result.maps.get(submap_id, {})
                        record["marker_graph_sequence"] = final_map.get("marker_graph_sequence", 0)
                        record["scale_anchor_keyframe"] = final_map.get("scale_anchor_keyframe", -1)
                        record["marker_tracking"] = native.get("marker_tracking", {})
                        record["marker_keyframe_event"] = native.get("marker_keyframe_event", "")
                        record["marker_event_keyframe_id"] = native.get("marker_event_keyframe_id", -1)
                        record["head_slam_map_points"] = head_slam_map_points[
                            frame_index
                        ]
                        record["head_slam_keyframes"] = head_slam_keyframes[
                            frame_index
                        ]
                    camera_sources[fused.source] += 1
                    for band in bands:
                        hand = record["hands"][band.name]
                        graph_result = wrist_graph_results[band.name][frame_index]
                        _replace_cached_graph_measurement(hand, graph_result)
                        camera_graph_pose = graph_result.pose
                        raw_world_graph_pose = (
                            compose_pose(fused.pose, camera_graph_pose)
                            if fused.pose is not None and camera_graph_pose is not None
                            else None
                        )
                        anchor_verified_world_pose = (
                            raw_world_graph_pose
                            if fused.source in {"marker", "marker+slam"}
                            else None
                        )
                        tracking = world_tracking[band.name]
                        world_tracking_pose = tracking.poses[frame_index]
                        world_tracking_source = tracking.sources[frame_index]
                        trajectory = wrist_trajectories[band.name]
                        optimized_world_graph_pose = trajectory.poses[frame_index]
                        world_graph_pose = optimized_world_graph_pose
                        if world_graph_pose is not None:
                            coordinate_frame = "world"
                            selected_graph_pose = world_graph_pose
                            graph_world_frames[band.name] += 1
                        elif camera_graph_pose is not None:
                            coordinate_frame = "camera"
                            selected_graph_pose = camera_graph_pose
                        else:
                            coordinate_frame = "invalid"
                            selected_graph_pose = None
                        selected_frame_key = (
                            f"world:{submap_id}"
                            if coordinate_frame == "world"
                            else coordinate_frame
                        )
                        if coordinate_frame == "invalid":
                            graph_selected_frame[band.name] = None
                        elif selected_frame_key != graph_selected_frame[band.name]:
                            graph_segments[band.name] += 1
                            graph_selected_frame[band.name] = selected_frame_key
                        hand["selected_coordinate_frame_graph"] = coordinate_frame
                        hand["world_submap_id"] = (
                            submap_id if world_graph_pose is not None else None
                        )
                        hand["trajectory_segment_graph"] = (
                            graph_segments[band.name]
                            if selected_graph_pose is not None
                            else None
                        )
                        hand["wrist_world_graph_raw"] = pose_to_dict(
                            raw_world_graph_pose
                        )
                        hand["wrist_world_anchor_verified"] = pose_to_dict(
                            anchor_verified_world_pose
                        )
                        hand["wrist_world_tracking"] = pose_to_dict(
                            world_tracking_pose
                        )
                        hand["wrist_world_tracking_source"] = (
                            world_tracking_source
                        )
                        hand["wrist_world_tracking_confidence"] = (
                            float(fused.confidence) * float(hand["confidence"])
                            if world_tracking_pose is not None
                            else 0.0
                        )
                        hand["wrist_world_graph"] = pose_to_dict(world_graph_pose)
                        hand.update(wrist_precision_fields(
                            all_detections[frame_index], graph_result.accepted_marker_ids,
                            band, calibration, camera_graph_pose,
                            world_valid=(world_graph_pose is not None
                                         and fused.pose is not None and fused.metric),
                            budget_mm=args.wrist_precision_budget_mm,
                            corner_sigma_px=args.wrist_corner_sigma_px,
                        ))
                        hand["wrist_world_recovered_later"] = bool(
                            fused.metric_recovered_later
                            and world_graph_pose is not None
                        )
                        hand["wrist_world_graph_optimized"] = pose_to_dict(
                            optimized_world_graph_pose
                        )
                        hand["trajectory_graph_reprojection_error_px"] = (
                            trajectory.reprojection_errors_px[frame_index]
                        )
                        # These IDs failed the strong boundary/template gate.
                        # They can only disambiguate an IPPE branch and enter
                        # the offline wrist fit at five per cent information;
                        # they never become accepted measurements.
                        hand["assist_only_marker_ids"] = sorted(
                            wrist_graph_assist_detections[band.name][frame_index]
                        )
                        hand["wrist_selected_graph"] = pose_to_dict(selected_graph_pose)
                        joints_record = hand["joints"]
                        joints_record["world_landmarks_graph_m"] = _world_landmarks(
                            joints_record.get("camera_landmarks_m"),
                            fused.pose,
                            world_graph_pose,
                        )
                        joints_record["world_landmarks_anchor_verified_m"] = (
                            _world_landmarks(
                                joints_record.get("camera_landmarks_m"),
                                fused.pose,
                                anchor_verified_world_pose,
                            )
                            if anchor_verified_world_pose is not None
                            else None
                        )
                        joints_record["world_landmarks_tracking_m"] = (
                            _world_landmarks(
                                joints_record.get("camera_landmarks_m"),
                                fused.pose,
                                world_tracking_pose,
                            )
                            if world_tracking_pose is not None
                            else None
                        )
                        graph_world_joint_frames[band.name] += int(
                            joints_record["world_landmarks_graph_m"] is not None
                        )
                    dual_world_wrist_frames += int(
                        all(
                            record["hands"][band.name]["wrist_world_graph"] is not None
                            for band in bands
                        )
                    )
                    dual_world_joint_frames += int(
                        all(
                            record["hands"][band.name]["joints"][
                                "world_landmarks_graph_m"
                            ] is not None
                            for band in bands
                        )
                    )
                    if diagnostics_stream is not None:
                        diagnostics = {
                            "frame": frame_index,
                            "timestamp_s": record["timestamp_s"],
                            "camera_world_source": fused.source,
                            "camera_world_confidence": fused.confidence,
                            "camera_metric_recovered_later": (
                                fused.metric_recovered_later
                            ),
                            "camera_submap_id": submap_id,
                            "camera_submap_anchor_marker_id": (
                                submap_anchor_ids.get(submap_id)
                                if submap_id is not None
                                else None
                            ),
                            "slam_inliers": fused.slam_inliers,
                            "accepted_marker_ids": record["accepted_marker_ids"],
                            "rejected_marker_ids": record["rejected_marker_ids"],
                            "assist_only_marker_ids": record[
                                "assist_only_marker_ids"
                            ],
                            "marker_consensus_vetoed": record[
                                "marker_consensus_vetoed"
                            ],
                            "boundary_rejected_marker_ids": sorted(
                                int(marker_id) for marker_id in record["boundary_rejected_marker_corners"]
                            ),
                            "marker_boundary_quality": record["marker_boundary_quality"],
                            "marker_errors_px": record["marker_errors_px"],
                            "marker_corner_diagnostics": record[
                                "marker_corner_diagnostics"
                            ],
                            "marker_pose_uncertainty": record[
                                "marker_pose_uncertainty"
                            ],
                            "graph_reprojection_error_px": record[
                                "graph_reprojection_error_px"
                            ],
                            "hands": {
                                name: {
                                    "accepted_marker_ids": hand[
                                        "accepted_marker_ids"
                                    ],
                                    "rejected_marker_ids": hand[
                                        "rejected_marker_ids"
                                    ],
                                    "assist_only_marker_ids": hand[
                                        "assist_only_marker_ids"
                                    ],
                                    "graph_reprojection_error_px": hand[
                                        "graph_reprojection_error_px"
                                    ],
                                    "trajectory_graph_reprojection_error_px": hand[
                                        "trajectory_graph_reprojection_error_px"
                                    ],
                                }
                                for name, hand in record["hands"].items()
                            },
                        }
                        diagnostics_stream.write(
                            json.dumps(
                                diagnostics,
                                separators=(",", ":"),
                                allow_nan=False,
                            )
                            + "\n"
                        )
                    stream.write(
                        json.dumps(record, separators=(",", ":"), allow_nan=False)
                        + "\n"
                    )
        finally:
            if diagnostics_stream is not None:
                diagnostics_stream.close()
            temporary_path.unlink(missing_ok=True)
        if args.slam_replay:
            from aruco_track.slam_replay import write_slam_replay
            from aruco_track.offline_replay_features import prepare_final_replay_features
            with output_path.open() as replay_stream:
                replay_actions = [json.loads(line) for line in replay_stream if line.strip()]
            replay_features = prepare_final_replay_features(
                Path(__file__).resolve().parent, video_path, replay_dir, replay_dir,
                orbslam3_result.history, replay_actions, calibration, fps,
                atlas_path=save_atlas)
            process_path, orb_map_viewer_path = write_slam_replay(
                video_path, output_path, replay_dir, orbslam3_result.history,
                calibration, all_detections, board_accepted_ids, fps, debug_video_path,
                marker_layout=world_board, actions=replay_actions,
                offline_features=replay_features, hybrid=True)
            del replay_actions, replay_features
            debug_video_path = process_path

    metadata = {
        "wrist_precision_policy": {
            **wrist_precision_policy(args.wrist_precision_budget_mm,
                                     args.wrist_corner_sigma_px),
            "enabled": bool(fusion_enabled),
        },
        "schema": "aruco-full-hand-actions/v3" if fusion_enabled else "aruco-full-hand-actions/v1",
        "video": str(video_path.resolve()),
        "calibration": str(Path(args.calib).resolve()),
        "bands": [str(Path(path).resolve()) for path in args.band],
        "world_board": str(Path(args.world_board).resolve()) if args.world_board else None,
        "marker_temporal_admission": {
            "enabled": bool(fusion_enabled),
            "policy": "short-window review of suspicious static-marker observations; raw pixels are preserved",
            "raw_observation_cache_policy": "v3 raw measurements remain reusable; admission is recomputed",
            "deferred_groups_can_seed_or_add_native_factors": False,
            "summary": marker_temporal_summary,
        },
        "marker_corner_policy": {
            "mode": "strict" if args.strict_marker_corners else "weighted",
            "fixed_marker_planar_validation": "IPPE dual candidates; reliable measured prediction expires after 0.15 s",
            "offline_multiview_bootstrap": (orbslam3_result.offline_marker_bootstrap
                                              if orbslam3_result is not None else None),
            "soft_information_weight": 0.25,
            "soft_can_initialize_map": False,
            "soft_observations": sum(0.0 < weight < 1.0 for frame in board_weights for weight in frame.values()),
            "soft_geometrically_accepted": sum(
                0.0 < weights.get(mid, 1.0) < 1.0
                for weights, accepted in zip(board_weights, board_accepted_ids) for mid in accepted
            ),
            "assist_only_observations": sum(
                len(frame) for frame in board_assist_detections
            ),
            "assist_only_frames": sum(bool(frame) for frame in board_assist_detections),
            "confirmed_weak_corners": {
                "enabled": bool(args.slam_weak_marker_corners),
                "role": (
                    "decoded rejected fixed-marker corners enter native local/global BA "
                    "at 5% information only after three consecutive frames agree within "
                    "2 px with a reliable same-frame marker pose"
                ),
                "can_initialize_or_set_scale": False,
                "information_weight": 0.05,
                "minimum_consecutive_frames": 3,
                "maximum_prediction_residual_px": 2.0,
                "factor_count": int(
                    orbslam3_result.timing.get(
                        "confirmed_weak_marker_corner_factors", 0
                    )
                ) if orbslam3_result is not None else 0,
                "frame_count": int(
                    orbslam3_result.timing.get(
                        "confirmed_weak_marker_corner_frames", 0
                    )
                ) if orbslam3_result is not None else 0,
            },
            "large_jump_consensus_veto": (
                "if a >50 mm update explains at least two fewer of three or more "
                "visible/assist markers than the recent measured pose, reject the marker "
                "update and defer to SLAM or invalid"
            ),
            "consensus_vetoed_frames": sum(
                result.consensus_vetoed for result in world_graph_results
            ),
            "uncertainty": {
                "model": "local_projection_hessian",
                "pixel_noise_floor_px": 0.25,
                "interpretation": (
                    "local marker-pose precision/conditioning estimate; "
                    "not external absolute accuracy"
                ),
                "valid_frames": sum(
                    bool(value and value.get("valid"))
                    for value in marker_pose_uncertainties
                ),
            },
        },
        "auto_marker_map": (
            {
                "enabled": True,
                "path": str(marker_map_path.resolve()),
                "static_marker_ids": sorted(args.static_marker_ids),
                "marker_size_mm": args.static_marker_size_mm,
                "geometry_sampling": {
                    "target_hz": 20.0,
                    "stride": auto_marker_map_sampling_stride,
                    "selected_frames": auto_marker_map_sample_count,
                    "source_frames": frames,
                    "pair_preservation": "at least three frames for every viable co-visible pair",
                },
                "submaps": len(auto_marker_map.submaps) if auto_marker_map else 0,
                "registered_marker_ids": (
                    sorted(auto_marker_map.marker_to_submap)
                    if auto_marker_map
                    else []
                ),
                "pending_marker_ids": (
                    list(auto_marker_map.pending_marker_ids)
                    if auto_marker_map
                    else []
                ),
                "policy": (
                    "only repeated reliable same-frame covisibility creates relative "
                    "marker constraints; every disconnected reliable marker component "
                    "can seed its own metric Atlas map; unrelated components retain "
                    "independent gauges until validated merge evidence exists"
                ),
            }
            if args.auto_marker_map
            else {"enabled": False}
        ),
        "hand_model": str(Path(args.hand_model).resolve()),
        "hand_backend": args.hand_backend,
        "hawor_config": str(args.hawor_config.resolve()) if args.hand_backend == 'hawor' else None,
        "hand_backend_provenance": hand_backend_provenance,
        "hand_joints_enabled": args.hand_joints,
        "hand_recovery_policy": (hawor_policy(enabled=args.hand_joints) if args.hand_backend == 'hawor'
                                 else hand_recovery_policy(enabled=args.hand_joints,
                                                           min_confidence=args.min_hand_confidence)),
        "min_hand_confidence": args.min_hand_confidence,
        "observation_cache_contract": {
            "schema": OBSERVATION_CACHE_SCHEMA,
            "algorithm_version": OBSERVATION_ALGORITHM_VERSION,
            "input_fingerprints": _observation_input_fingerprints(
                video_path,
                args.calib,
                args.band,
                args.world_board,
                args.hand_model,
                hand_joints=args.hand_joints,
            ),
        },
        "observation_cache": str(args.reuse_observations.resolve()) if args.reuse_observations else None,
        "frames": frames,
        "fps": fps,
        "image_size": list(video_size),
        "joint_detection_frames": dict(joint_frames),
        "wrist_anchored_joint_frames": dict(fused_frames),
        "world_joint_frames": dict(world_fused_frames),
        "graph_world_wrist_frames": dict(graph_world_frames),
        "graph_world_joint_frames": dict(graph_world_joint_frames),
        "valid_metric_label_yield": {
            "camera_frames": sum(
                frame.pose is not None and frame.metric for frame in fused_camera
            ),
            "camera_frames_recovered_after_metricization": sum(
                frame.pose is not None and frame.metric_recovered_later
                for frame in fused_camera
            ),
            "camera_frames_relocalized_short_gaps": sum(
                frame.pose is not None and (frame.localization_recovery or {}).get('method') == 'native-orb-final-atlas-short-gap-pnp'
                for frame in fused_camera),
            "camera_frames_relocalized_before_initialization": sum(
                frame.pose is not None and (frame.localization_recovery or {}).get('method') == 'native-orb-final-atlas-prefix-pnp'
                for frame in fused_camera
            ),
            "camera_fraction": (
                sum(frame.pose is not None and frame.metric for frame in fused_camera) / frames
                if frames
                else 0.0
            ),
            "wrist_frames": {
                band.name: graph_world_frames[band.name] for band in bands
            },
            "wrist_fraction": {
                band.name: graph_world_frames[band.name] / frames if frames else 0.0
                for band in bands
            },
            "dual_wrist_frames": dual_world_wrist_frames,
            "dual_wrist_fraction": dual_world_wrist_frames / frames if frames else 0.0,
            "conditional_hand_frames": {
                band.name: graph_world_joint_frames[band.name] for band in bands
            },
            "conditional_hand_fraction": {
                band.name: graph_world_joint_frames[band.name] / frames if frames else 0.0
                for band in bands
            },
            "dual_conditional_hand_frames": dual_world_joint_frames,
            "dual_conditional_hand_fraction": (
                dual_world_joint_frames / frames if frames else 0.0
            ),
        },
        "camera_world_sources": dict(camera_sources),
        "camera_anchor_consistency_summary": dict(Counter(
            (frame.anchor_consistency or {}).get("status", "unavailable")
            for frame in fused_camera
        )),
        "camera_confidence_policy": (
            "Visual confidence describes tracking support, not calibrated absolute accuracy. "
            "A strong decoded registered marker behind the selected revision's camera or "
            "with >100 px raw undistorted residual sets confidence to zero without moving "
            "or deleting the pose. Unobserved anchors do not certify metric accuracy; "
            "valid_metric_label_yield is pose availability before this quality screening."
        ),
        "slam": (
            {
                "enabled": True,
                "backend": (
                    "native-orb-slam3+tag-corners"
                    if head_slam_enabled
                    else None
                ),
                "dynamic_feature_policy": (
                    "exclude all detected hand/tag masks before ORB feature allocation; "
                    "native ORB geometric outlier rejection; "
                    + ("classical temporal geometry masking (no new learned model)"
                       if orbslam3_result.timing.get("dynamic_geometry", 0) else "temporal geometry masking disabled")
                ),
                "dynamic_geometry_enabled": bool(orbslam3_result.timing.get("dynamic_geometry", 0)),
                "dynamic_geometry_diagnostics": (str(replay_dir / "dynamic_geometry.jsonl")
                    if orbslam3_result.timing.get("dynamic_geometry", 0) else None),
                "camera_pose_policy": (
                    "marker-first native metric seed or ordinary monocular initialization; "
                    + (
                        "official monocular ORB-SLAM3 tracks, maps and relocalizes while markers are hidden; "
                        "offline low-support frames give queued local mapping a bounded catch-up window and "
                        "metric maps receive one final bounded projection search before declaring loss; "
                        "same-map pure-visual loops in metric marker maps keep all committed marker-factor "
                        "keyframes fixed as gauge anchors; cross-map pure-visual metric merges remain rejected; "
                        "validated native marker interval Sim3+corner BA and common-marker map merging enabled; "
                        "fixed-world tag-constrained frame poses "
                        "retain their native metre measurement except explicitly committed rigid map-merge gauges; "
                        "local background scale is not applied to those measurements; calibrated boards retain one "
                        "rigid SE(3) layout variable; tag-hidden visual frames may receive a conservative final-Atlas "
                        "pose-only refinement"
                        if head_slam_enabled
                        else ""
                    )
                ),
                "preinitialization_recovery_policy": (
                    "short initial NOT_INITIALIZED prefix only; read-only final Atlas ORB 2D-3D/PnP "
                    "and independent anchor checks; recovered final labels are marked with "
                    "camera_localization_recovery; no interpolation or historical tracking-state rewrite"
                ),
                "wrist_pose_policy": (
                    "valid metric camera and observed wrist only; short-gap single-marker "
                    "IPPE candidates are re-selected with committed camera motion and "
                    "bidirectional measured-pose context; no hole interpolation"
                ),
                "debug_video": (
                    str(debug_video_path.resolve()) if debug_video_path else None
                ),
                "orb_map_video": (
                    str(orb_map_video_path.resolve())
                    if orb_map_video_path is not None
                    else None
                ),
                "orb_map_viewer": (
                    str(orb_map_viewer_path.resolve())
                    if orb_map_viewer_path is not None
                    else None
                ),
                "diagnostics": (
                    str(diagnostics_path.resolve()) if diagnostics_path else None
                ),
            }
            if fusion_enabled
            else {"enabled": False}
        ),
        "head_slam": {
            "enabled": bool(head_slam_enabled),
            "backend": "official ORB-SLAM3 monocular" if head_slam_enabled else None,
            "marker_graph_capabilities": (
                orbslam3_result.history[-1].get("marker_graph_capabilities", {})
                if orbslam3_result is not None and orbslam3_result.history else {}
            ),
            "marker_graph_events": (
                orbslam3_result.history[-1].get("marker_graph_events", [])
                if orbslam3_result is not None and orbslam3_result.history else []
            ),
            "marker_map_aliases": (
                orbslam3_result.history[-1].get("marker_map_aliases", {})
                if orbslam3_result is not None and orbslam3_result.history else {}
            ),
            "initialization_policy": (
                args.slam_init
                if head_slam_enabled
                else None
            ),
            "initialization_frame": (
                orbslam3_result.initialization_frame
                if orbslam3_result is not None
                else None
            ),
            "metric_scale_m_per_slam_unit": (
                orbslam3_result.scale_m_per_slam_unit
                if orbslam3_result is not None
                else None
            ),
            "metric_anchor_count": (
                orbslam3_result.anchor_count
                if orbslam3_result is not None
                else 0
            ),
            "median_anchor_position_error_m": (
                orbslam3_result.median_anchor_position_error_m
                if orbslam3_result is not None
                else None
            ),
            "median_anchor_rotation_error_deg": (
                orbslam3_result.median_anchor_rotation_error_deg
                if orbslam3_result is not None
                else None
            ),
            "tracking_timing": (
                orbslam3_result.timing if orbslam3_result is not None else {}
            ),
            "map_points": len(_final_resolved_map(orbslam3_result).get("points", [])),
            "keyframes": len(_final_resolved_map(orbslam3_result).get("keyframes", [])),
            "feature_policy": (
                "native ORB Atlas; hand/tag regions excluded; fixed-layout tag corner BA; native frame-publication replay"
            ),
        },
        "joint_policy": {
            "backend": args.hand_backend,
            "landmarks": "21 landmarks in wrist/thumb/index/middle/ring/pinky order",
            "model_landmarks_m": "camera-oriented model estimates in model meters; not externally measured joint accuracy",
            "unsupported_predictions": "excluded from valid labels and SLAM hand masks",
            "future_context_frames_max": 15 if args.hand_backend == 'hawor' else 0,
            "camera_landmarks_m": "landmark 0 anchored to the raw measured wrist-band origin",
            "band_landmarks_m": "camera landmark vectors rotated into the raw wrist-band frame",
            "world_landmarks_m": "camera landmarks transformed by the directly observed fixed reference",
            "world_landmarks_graph_m": (
                "camera landmarks transformed by the final valid metric native ORB camera pose"
                if fusion_enabled
                else "not exported"
            ),
            "world_landmarks_anchor_verified_m": (
                "world landmarks emitted only when the fixed marker session map and wrist are directly observed in the same frame"
                if fusion_enabled
                else "not exported"
            ),
            "world_landmarks_tracking_m": (
                "measured wrist landmarks transformed by valid metric native camera poses"
                if fusion_enabled
                else "not exported"
            ),
            "missing_data": "null; no long-gap interpolation",
            "anatomical_wrist_offset": "zero in v1; calibrate a constant per-band offset for absolute fingertips",
            "band_assignment": (
                "wrist proximity plus wrist-to-palm alignment with the projected band Y-axis line; "
                "axis sign is ignored because the passive cuff is not keyed proximal/distal; "
                "two consecutive accepted frames required after every gap"
            ),
            "temporal_confirmation_frames": 2,
            "wrist_anchor_axis_ratio_max": 0.75,
            "wrist_axis_error_deg_max": 70.0,
            "wrist_anchor_gate_px": 0.12 * float(
                (video_size[0] ** 2 + video_size[1] ** 2) ** 0.5
            ),
        },
        "wrist_corner_policy": {
            "enabled": True,
            "mode": "resolution-aware boundary and template validation",
            "applies_to": "detected, recovered and optical-flow wrist corners",
            "rejected_measurements": "null wrist pose; no stale prior or gap filling",
            "assist_only": {
                "role": (
                    "IPPE branch vote and offline wrist fit only; cannot initialize, "
                    "set scale, raise confidence or become an accepted marker"
                ),
                "eligible_reasons": ["grid_mismatch", "wrist_grid_mismatch"],
                "minimum_contrast": 35.0,
                "minimum_edge_support": 0.9,
                "minimum_corner_support": 0.75,
                "maximum_template_interior_error_fraction": 0.02,
                "offline_information_weight": 0.05,
                "observations_by_band": {
                    band.name: sum(
                        len(frame)
                        for frame in wrist_graph_assist_detections[band.name]
                    )
                    for band in bands
                },
                "frames_by_band": {
                    band.name: sum(
                        bool(frame)
                        for frame in wrist_graph_assist_detections[band.name]
                    )
                    for band in bands
                },
            },
        },
    }
    if fusion_enabled:
        metadata["atlas"] = str(save_atlas.resolve())
        metadata["replay"] = (
            str(orb_map_viewer_path.resolve())
            if orb_map_viewer_path is not None
            else None
        )
        metadata["maps"] = {key: {field: value for field, value in m.items()
                                  if field not in {"points", "keyframes"}}
                            for key, m in orbslam3_result.maps.items()}
        # Each disconnected world has its own TUM; never concatenate gauges.
        from scipy.spatial.transform import Rotation
        for map_id in sorted({f.map_id for f in fused_camera if f.pose is not None}):
            lines = []
            for index, frame in enumerate(fused_camera):
                if frame.map_id != map_id or frame.pose is None:
                    continue
                values = [index/fps, *frame.pose.tvec.reshape(3),
                          *Rotation.from_matrix(frame.pose.rotation_matrix).as_quat()]
                lines.append(" ".join(f"{v:.9f}" for v in values))
            (replay_dir / f"camera_{map_id}.tum").write_text("\n".join(lines) + "\n")
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    if args.lerobot_output is not None:
        optional_python = Path(__file__).with_name(".venv-vla") / "bin" / "python"
        if not optional_python.is_file():
            raise SystemExit(
                "LeRobot export requires the isolated optional environment; run "
                "scripts/setup_vla_export.sh once"
            )
        command = [
            str(optional_python), str(Path(__file__).with_name("export_lerobot_dataset.py")),
            str(output_path), "--output", str(args.lerobot_output),
            "--task", args.lerobot_task, "--fps", str(args.lerobot_fps),
        ]
        if args.lerobot_repo_id:
            command.extend(["--repo-id", args.lerobot_repo_id])
        subprocess.run(command, check=True)
    if args.open_replay and fusion_enabled:
        subprocess.Popen([sys.executable, str(Path(__file__).with_name("replay_orb_slam.py")),
                          str(replay_dir)], start_new_session=True)
    print(f"wrote {frames} frames to {output_path}")
    print(f"wrote metadata to {metadata_path}")
    if args.lerobot_output is not None:
        print(f"wrote optional LeRobot dataset to {args.lerobot_output}")
    if diagnostics_path is not None:
        print(f"wrote graph diagnostics to {diagnostics_path}")
    if args.auto_marker_map:
        print(f"wrote auto marker map to {marker_map_path}")
    if debug_video_path is not None:
        print(f"wrote SLAM process video to {debug_video_path}")
    if orb_map_video_path is not None:
        print(f"wrote ORB map video to {orb_map_video_path}")
    if orb_map_viewer_path is not None:
        print(f"wrote interactive ORB map to {orb_map_viewer_path}")
    if fusion_enabled:
        print(f"camera world sources: {dict(camera_sources)}")
    for band in bands:
        message = (
            f"{band.name}: joints={joint_frames[band.name]}/{frames}, "
            f"wrist+joints={fused_frames[band.name]}/{frames}, "
            f"world={world_fused_frames[band.name]}/{frames}"
        )
        if fusion_enabled:
            message += f", graph_world={graph_world_frames[band.name]}/{frames}"
        print(message)


if __name__ == "__main__":
    main()
