"""Offline initial guesses from a short tag-free ORB + marker multiview fit.

The joint pose supplies native initialization, never an additional pose factor
or an independent single-frame PnP measurement. Native validates its own metric
proposal using original marker corners and background reprojections. Earlier
observations become available only AFTER prefix analysis has completed; callers
must preserve the returned availability metadata in offline exports.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Callable

import cv2
import numpy as np

from .marker_multiview import MarkerView, fit_marker_multiview, refit_marker_view
from .models import BandLayout, Calibration, Pose
from .orbslam3_backend import (
    MAX_MARKER_ERROR_PX, MIN_MARKER_CONFIDENCE, camera_at_revision,
    first_reliable_marker_frame, read_native_history, run_orbslam3_sequence,
    write_orbslam3_settings, write_tag_observation_hints,
)
from .pipeline import compose_pose
from .pose import square_object_points


@dataclass
class MarkerBootstrapResult:
    poses: list[Pose | None]
    confidences: list[float]
    accepted_marker_ids: list[tuple[int, ...]]
    component_ids: list[str | None]
    hints_path: Path | None
    diagnostics: dict


def _singleton_geometry(layout: BandLayout) -> tuple[int, float, Pose] | None:
    if len(layout.markers) != 1:
        return None
    marker_id, points = next(iter(layout.markers.items()))
    points = np.asarray(points, float)
    if points.shape != (4, 3) or not np.all(np.isfinite(points)):
        return None
    size = float(np.linalg.norm(points[1] - points[0]))
    if size <= 0:
        return None
    square = square_object_points(size)
    center = points.mean(axis=0)
    u, _, vt = np.linalg.svd(square.T @ (points - center))
    rotation = vt.T @ np.diag([1., 1., np.linalg.det(vt.T @ u.T)]) @ u.T
    if not np.allclose(square @ rotation.T + center, points, atol=1e-7, rtol=0):
        return None
    return marker_id, size, Pose(cv2.Rodrigues(rotation)[0], center.reshape(3, 1), 0.)


def bootstrap_initial_marker_observations(
    project_dir: Path, sequence_dir: Path, work_dir: Path,
    calibration: Calibration, fps: float,
    detections: list[dict[int, np.ndarray]], marker_poses: list[Pose | None],
    marker_confidences: list[float], accepted_marker_ids: list[tuple[int, ...]],
    marker_weights: list[dict[int, float]] | None,
    marker_layouts: dict[str, BandLayout], component_ids: list[str | None],
    *, slam_init: str = "auto", load_atlas: Path | None = None,
    max_probe_seconds: float = 8., dynamic_geometry: bool = False,
    probe_runner: Callable = run_orbslam3_sequence,
) -> MarkerBootstrapResult:
    """Return UNCHANGED measurement streams plus a separate native hint file.

Only unresolved strong observations of a singleton initial component qualify.
Missing weights cannot establish strong evidence and therefore skip probing.
``probe_runner`` has run_orbslam3_sequence's interface and must publish native
history; it exists for controlled tests, not for passing a metric trajectory.
"""
    count = len(marker_poses)
    if not all(len(values) == count for values in
               (detections, marker_confidences, accepted_marker_ids, component_ids)):
        raise ValueError("bootstrap observation streams must have equal length")
    if marker_weights is not None and len(marker_weights) != count:
        raise ValueError("bootstrap weight stream must have equal length")
    if not np.isfinite(fps) or fps <= 0 or not 0 < max_probe_seconds <= 8.:
        raise ValueError("bootstrap needs positive fps and a prefix of at most 8 seconds")
    diagnostic = {
        "mode": "offline-assisted-bootstrap", "accepted": False,
        "reason": "not_attempted", "causal_online": False,
        "probe_trajectory_used_as_optimization_factor": False,
        "no_added_pose_factors": True,
        "hint_kind": "multiview_initialization_hint",
        "independent_single_frame_pnp": False,
        "single_frame_refit_used_only_for_pixel_quality": True,
        "native_metric_commit_required": True,
        "historical_measurement_state_rewritten": False,
        "candidates": [], "published_observations": [],
    }
    result = MarkerBootstrapResult(list(marker_poses), list(marker_confidences),
                                   list(accepted_marker_ids), list(component_ids),
                                   None, diagnostic)
    initial_count = min(count, int(5. * fps) + 1)
    if load_atlas is not None or slam_init == "marker":
        diagnostic["reason"] = "atlas_load_or_marker_only_initialization"
        return result
    if first_reliable_marker_frame(marker_poses[:initial_count], marker_confidences[:initial_count]) is not None:
        diagnostic["reason"] = "initial_reliable_marker_already_available"
        return result
    if marker_weights is None:
        diagnostic["reason"] = "strong_corner_weights_unavailable"
        return result
    candidates = []
    evidence_end_frame = -1
    for component, layout in sorted(marker_layouts.items()):
        geometry = _singleton_geometry(layout)
        if geometry is None:
            continue
        marker_id, size, transform = geometry
        strong = [i for i in range(initial_count) if marker_poses[i] is None
                  and marker_id in detections[i]
                  and marker_weights[i].get(marker_id, 0.) >= .99
                  and np.asarray(detections[i][marker_id]).shape == (4, 2)
                  and np.all(np.isfinite(detections[i][marker_id]))]
        if len(strong) >= 8:
            candidates.append((component, marker_id, size, transform))
            evidence_end_frame = max(evidence_end_frame, strong[-1])
    if not candidates:
        diagnostic["reason"] = "insufficient_initial_strong_singleton_observations"
        return result

    # Native concatenates sequence_dir/filename, so use relative cache paths,
    # not absolute filenames. Adjacent immutable .mask.png files stay in place.
    probe_root = Path(work_dir) / "marker_bootstrap"
    probe_sequence, probe_output = probe_root / "sequence", probe_root / "native"
    probe_sequence.mkdir(parents=True, exist_ok=True)
    probe_output.mkdir(parents=True, exist_ok=True)
    rgb_lines, timestamps = [], []
    with (Path(sequence_dir) / "rgb.txt").open() as stream:
        for line in stream:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split()
            timestamp = float(fields[0])
            # End at the measured anchor window, not an arbitrary later time.
            # Later marker-free BA must not redefine the gauge used to validate
            # this initialization's old corner evidence (nor add probe cost).
            if (timestamp > max_probe_seconds or round(timestamp * fps) >= count
                    or round(timestamp * fps) > evidence_end_frame):
                break
            image = (Path(sequence_dir) / fields[1]).resolve()
            relative = os.path.relpath(image, probe_sequence.resolve())
            rgb_lines.append(f"{timestamp:.9f} {relative}")
            timestamps.append(timestamp)
    if not timestamps or abs(timestamps[0]) > .5 / fps:
        diagnostic["reason"] = "probe_sequence_does_not_start_at_frame_zero"
        return result
    (probe_sequence / "rgb.txt").write_text("\n".join(rgb_lines) + "\n")
    settings = probe_root / "settings.yaml"
    write_orbslam3_settings(settings, calibration, fps, slam_init="auto",
                            dynamic_geometry=dynamic_geometry)
    settings.write_text(settings.read_text().replace("TagFusion.enabled: 1", "TagFusion.enabled: 0")
                        + "loopClosing: 0\n")
    started = time.perf_counter()
    probe_runner(Path(project_dir), probe_sequence, settings, probe_output,
                 tag_observations_path=None, compact_history=True,
                 environment_overrides={
                     "ORB_SLAM3_DIAGNOSTIC_NO_FINAL_OPTIMIZATION": "1",
                     "ORB_SLAM3_OFFLINE_LOOP_SEARCH": "0",
                     "ORB_SLAM3_INCREMENTAL_LOOP_SEARCH": "0",
                     "ORB_SLAM3_MARKER_SIM3_LOOP": "0",
                 })
    diagnostic.update(probe_wall_seconds=time.perf_counter() - started,
                      available_after_s=timestamps[-1], probe_frames=len(timestamps),
                      probe_history_path=str(probe_output / "frames.txt.history.jsonl"),
                      probe_tag_fusion=False, probe_loop_closing=False,
                      probe_final_optimization=False)
    history = read_native_history(probe_output / "frames.txt.history.jsonl")
    if not history or not history[-1].get("final"):
        diagnostic["reason"] = "probe_final_publication_missing"
        return result
    if any(h.get("marker_pose_used") or h.get("marker_factor_eligible")
           or any(m.get("metric") for m in h.get("maps", [])) for h in history):
        diagnostic["reason"] = "probe_trajectory_not_marker_independent"
        return result
    final = history[-1]
    references = {v[0]: v for v in final.get("references", [])}
    indexed = {round(h["timestamp"] * fps): h for h in history if not h.get("final")}
    component_proposals: dict[str, dict[int, tuple[float, Pose, str, int, float]]] = {}
    for component, marker_id, size, transform in candidates:
        views = []
        for i, frame in sorted(indexed.items()):
            if i >= count or marker_poses[i] is not None or marker_id not in detections[i]:
                continue
            weight = marker_weights[i].get(marker_id, 0.)
            corners = np.asarray(detections[i][marker_id], float)
            if weight < .99 or corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
                continue
            camera, map_id = camera_at_revision(frame, final, references)
            if camera is not None:
                views.append(MarkerView(i, frame["timestamp"], camera, corners,
                                        weight, frame.get("reference"), map_id,
                                        final.get("timestamp", timestamps[-1])))
        fit = fit_marker_multiview(views, calibration, size, marker_id,
                                   trajectory_is_independent=True)
        entry = dict(marker_id=marker_id, component_id=component, accepted=fit.accepted,
                     reason=fit.reason, scale_m_per_unit=fit.scale_m_per_unit,
                     frame_ids=[v.frame_id for v in views], diagnostics=fit.diagnostics,
                     training_frames=list(fit.training_frame_ids),
                     validation_frames=list(fit.validation_frame_ids), hint_frames=[])
        diagnostic["candidates"].append(entry)
        if not fit.accepted:
            continue
        proposals = component_proposals.setdefault(component, {})
        for view in views:
            # A free per-frame six-DoF fit reintroduces the planar instability.
            # Use it ONLY to assess pixel quality, never as the pose factor or
            # initial guess. No interpolation or synthetic pre-init frame pose.
            measured_pose = refit_marker_view(view, fit, calibration, size)
            initial_pose = fit.camera_in_marker.get(view.frame_id)
            if (measured_pose is None or initial_pose is None
                    or not np.isfinite(initial_pose.reprojection_error_px)
                    or initial_pose.reprojection_error_px > MAX_MARKER_ERROR_PX):
                continue
            area = abs(cv2.contourArea(np.asarray(view.corners, np.float32)))
            confidence = float(min(area / 2000., 1.) *
                               np.exp(-measured_pose.reprojection_error_px / np.sqrt(2.) / 3.) * .75)
            if confidence < MIN_MARKER_CONFIDENCE or measured_pose.reprojection_error_px > MAX_MARKER_ERROR_PX:
                continue
            pose = compose_pose(transform, initial_pose)
            pose.reprojection_error_px = initial_pose.reprojection_error_px
            pose.marker_ids, pose.inlier_count = (marker_id,), 4
            entry["hint_frames"].append(view.frame_id)
            proposals[view.frame_id] = (confidence, pose, component, marker_id,
                                         measured_pose.reprojection_error_px)
    # Disconnected singleton coordinates are not a common world. Until native
    # metricization establishes a bridge, never interleave their pose factors.
    eligible = {component: values for component, values in component_proposals.items() if len(values) >= 8}
    selected = max(eligible, key=lambda component: (len(eligible[component]),
                   max(eligible[component]) - min(eligible[component])), default=None)
    proposals = eligible.get(selected, {})
    diagnostic["selected_component_id"] = selected
    diagnostic["component_selection_policy"] = "one_independent_world_with_at_least_eight_observed_corner_hints"
    for entry in diagnostic["candidates"]:
        entry["selected_for_native"] = entry["component_id"] == selected
    hint_poses, hint_confidences = list(marker_poses), list(marker_confidences)
    hint_ids, hint_components = list(accepted_marker_ids), list(component_ids)
    for i, (confidence, pose, component, marker_id, single_frame_error) in sorted(proposals.items()):
        hint_poses[i], hint_confidences[i] = pose, confidence
        hint_ids[i], hint_components[i] = (marker_id,), component
        diagnostic["published_observations"].append(dict(
            frame_id=i, observation_timestamp_s=i / fps, available_after_s=timestamps[-1],
            marker_id=marker_id, component_id=component, reprojection_error_px=pose.reprojection_error_px,
            confidence=confidence, kind="multiview_initialization_hint",
            single_frame_fit_rms_px=single_frame_error,
            joint_hint_reprojection_px=pose.reprojection_error_px,
            no_added_pose_factors=True,
            evidence_source="original_four_corners_with_joint_initial_guess"))
    if proposals:
        generated = probe_root / "initialization_observations.txt"
        write_tag_observation_hints(generated, hint_poses, hint_confidences,
                                    detections, hint_ids,
                                    next(iter(marker_layouts.values())), calibration,
                                    fps, 0, marker_weights, include_ids=True,
                                    marker_layouts=marker_layouts,
                                    marker_component_ids=hint_components)
        replacement = {round(float(line.split()[0]) * fps): line for line in generated.read_text().splitlines()
                       if line and not line.startswith("#")}
        original = Path(sequence_dir) / "tag_observations.txt"
        lines = original.read_text().splitlines()
        for n, line in enumerate(lines):
            if line.strip() and not line.lstrip().startswith("#"):
                i = round(float(line.split()[0]) * fps)
                if i in proposals:
                    lines[n] = replacement[i]
        result.hints_path = probe_root / "tag_observations.marker_bootstrap.txt"
        result.hints_path.write_text("\n".join(lines) + "\n")
        diagnostic.update(accepted=True, reason="multiview_initialization_hints_available",
                          hints_path=str(result.hints_path))
    else:
        diagnostic["reason"] = "no_validated_initialization_hints"
    (probe_root / "diagnostics.json").write_text(json.dumps(diagnostic, indent=2) + "\n")
    return result
