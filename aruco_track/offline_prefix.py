"""Recover only uninitialized image poses by read-only final-Atlas localization.

This is a separate, explicitly late result. Native process history is immutable:
recovered poses are final labels, never retroactively successful tracking events.
"""
from __future__ import annotations

from dataclasses import replace
from collections import Counter
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from .camera_state import FusedCameraFrame, observed_hands_for_mask
from .models import Calibration
from .orbslam3_backend import MetricOrbSlamResult, pose_from_native


def initial_prefix_request(result: MetricOrbSlamResult, fps: float) -> dict | None:
    """Select a short initial NOT_INITIALIZED prefix in one resolved metric map."""
    if not result.history or not result.history[-1].get("final") or fps <= 0:
        return None
    states = {round(h["timestamp"] * fps): h for h in result.history if not h.get("final")}
    boundary = next((i for i in sorted(states)
                     if states[i].get("pose") is not None and states[i].get("state") in (2, 6)), None)
    if boundary is None or boundary == 0 or boundary >= len(result.frames):
        return None
    anchor = result.frames[boundary]
    mapping = result.maps.get(anchor.map_id, {})
    if (anchor.pose is None or not anchor.metric or not mapping.get("metric")
            or not mapping.get("background") or anchor.revision != mapping.get("revision")):
        return None
    # No recovery of a later LOST segment or a prefix spanning a different map.
    prefix = [i for i in range(max(0, boundary - int(5.0 * fps)), boundary)
              if i in states and states[i].get("state") in (0, 1)
              and states[i].get("pose") is None and result.frames[i].pose is None
              and states[i].get("active_map") == states[boundary].get("active_map")]
    if not prefix or any(states.get(i, {}).get("state") not in (0, 1)
                         for i in range(boundary)):
        return None
    # A marker seed can precede the first usable background keyframe. Measure
    # this intervening image sequence as temporal support only: these frames
    # never become additional recovery targets, even if native tracking lost.
    support = []
    stride = max(1, int(np.ceil(fps / 30.0)))
    for i in range(boundary, min(len(result.frames), boundary + int(2.0 * fps) + 1)):
        state = states.get(i)
        if state is None or state.get("active_map") != states[boundary].get("active_map"):
            break
        if (i - boundary) % stride == 0:
            support.append(i)
    return {"map_id": mapping["id"], "map_revision": mapping["revision"],
            "boundary_frame": boundary, "boundary_time_s": boundary / fps,
            "frames": prefix, "support_frames": support}


def apply_prefix_candidates(result: MetricOrbSlamResult, request: dict, rows: list[dict]) -> MetricOrbSlamResult:
    """Validate the adapter contract; never overwrite a measured native pose."""
    frames = list(result.frames)
    eligible = set(request["frames"])
    rows = [row for row in rows if isinstance(row, dict)
            and type(row.get("frame")) is int and 0 <= row["frame"] < len(frames)]
    duplicates = {index for index, count in Counter(row["frame"] for row in rows).items() if count > 1}
    for row in rows:
        index = row.get("frame")
        if (index not in eligible or index in duplicates or row.get("accepted") is not True
                or row.get("support_only") is True
                or frames[index].pose is not None or row.get("map_id") != request["map_id"]
                or row.get("map_revision") != request["map_revision"]
                or type(row.get("inliers")) is not int):
            continue
        try:
            pose = pose_from_native(row.get("pose"))
            inliers, rms = int(row["inliers"]), float(row["rms_px"])
            if pose is None or inliers < 30 or not np.isfinite(rms) or not 0 <= rms <= 3.0:
                continue
        except (KeyError, TypeError, ValueError):
            continue
        recovery = {"method": "native-orb-final-atlas-prefix-pnp", "accepted": True,
                    "original_tracking_valid": False, "map_id": request["map_id"],
                    "map_revision": request["map_revision"],
                    "available_after_timestamp_s": result.history[-1]["timestamp"],
                    "inliers": inliers, "rms_px": rms}
        frames[index] = FusedCameraFrame(
            pose, "head-slam", float(np.clip(inliers / 100, .35, .9)), inliers, rms,
            f"atlas_{request['map_id']}", request["map_revision"], True,
            result.frames[request["boundary_frame"]].initialization_source, True,
            localization_recovery=recovery)
    return replace(result, frames=frames)


def recover_initial_prefix(
    result: MetricOrbSlamResult, project_dir: Path, video_path: Path,
    atlas_path: Path, records_path: Path, calibration: Calibration, fps: float,
    diagnostics_dir: Path,
) -> MetricOrbSlamResult:
    request = initial_prefix_request(result, fps)
    if request is None:
        return result
    started = time.perf_counter()
    binary = project_dir / "third_party/ORB_SLAM3/Examples/Monocular/relocalize_prefix_readonly"
    report = {"method": "native-orb-final-atlas-prefix-pnp", "request": request,
              "status": "not_run", "accepted_frames": 0,
              "history_policy": "final labels only; process history is not changed"}
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    if not binary.is_file() or not atlas_path.is_file():
        report["status"] = "missing_readonly_adapter_or_atlas"
    else:
        selected = set(request["frames"])
        support = set(request.get("support_frames", []))
        queries, support_queries, mask_frames = [], [], []
        mask_end = min(len(result.frames) - 1, request["boundary_frame"] + int(2.0 * fps))
        with records_path.open() as stream:
            for line in stream:
                record = json.loads(line)
                index = int(record["frame"])
                if index > mask_end:
                    break
                if index not in selected and index < request["boundary_frame"]:
                    continue
                corners = dict(record.get("boundary_rejected_marker_corners", {}))
                corners.update(record.get("detected_marker_corners", {}))
                polygons = list(corners.values())
                for hand in observed_hands_for_mask(record).values():
                    points = np.asarray(hand.get("image_landmarks_normalized", []), dtype=float)
                    if points.ndim == 2 and len(points) >= 3:
                        polygons.append((points[:, :2] * calibration.image_size).tolist())
                query = {"frame": index, "timestamp_s": index / fps,
                         "excluded_polygons": polygons,
                         "original_pose_valid": result.frames[index].pose is not None}
                if index in selected:
                    queries.append(query)
                elif index in support:
                    support_queries.append(query)
                if index >= request["boundary_frame"]:
                    mask_frames.append({"frame": index, "excluded_polygons": polygons})
        manifest = {key: value for key, value in request.items()
                    if key not in ("frames", "support_frames")}
        manifest.update(video=str(video_path.resolve()), camera_matrix=calibration.camera_matrix.tolist(),
                        dist_coeffs=calibration.dist_coeffs.reshape(-1).tolist(),
                        image_width=calibration.image_size[0], image_height=calibration.image_size[1],
                        queries=queries, support_queries=support_queries, mask_frames=mask_frames)
        # The native adapter runs with ``project_dir`` as its cwd.  Resolve
        # diagnostic paths before crossing that process boundary so a caller
        # using a relative --output directory cannot make an otherwise valid
        # prefix request look unreadable to the adapter.
        manifest_path = (diagnostics_dir / "prefix_localization_request.json").resolve()
        candidates_path = (diagnostics_dir / "prefix_localization_candidates.jsonl").resolve()
        manifest_path.write_text(json.dumps(manifest, separators=(",", ":"), allow_nan=False))
        try:
            completed = subprocess.run(
        [str(binary), str(project_dir / "third_party/ORB_SLAM3/Vocabulary/ORBvoc.txt"),
                 str(atlas_path.resolve()), str(manifest_path), str(candidates_path)],
                cwd=project_dir, capture_output=True, text=True, timeout=120, check=False)
            (diagnostics_dir / "prefix_localization.log").write_text(completed.stdout + completed.stderr)
            if completed.returncode:
                report["status"] = "adapter_failed"
                report["returncode"] = completed.returncode
            else:
                rows = [json.loads(line) for line in candidates_path.read_text().splitlines() if line.strip()]
                result = apply_prefix_candidates(result, request, rows)
                report["status"] = "completed"
                report["accepted_frames"] = sum(bool(frame.localization_recovery) for frame in result.frames)
                report["support_frames_evaluated"] = sum(row.get("type") == "support" for row in rows)
                report["support_frames_connected"] = sum(
                    row.get("type") == "support" and row.get("connected") is True for row in rows)
        except (OSError, ValueError, subprocess.TimeoutExpired) as error:
            report["status"] = "adapter_failed"
            report["reason"] = str(error)
    report["seconds"] = time.perf_counter() - started
    (diagnostics_dir / "prefix_localization.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    timing = dict(result.timing, prefix_localization_seconds=report["seconds"],
                  prefix_localization_recovered_frames=float(report["accepted_frames"]))
    print(f"Offline prefix localization: {report['status']}, "
          f"{report['accepted_frames']}/{len(request['frames'])} recovered, {report['seconds']:.2f}s", flush=True)
    return replace(result, timing=timing)
