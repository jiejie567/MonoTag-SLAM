#!/usr/bin/env python3
"""Re-run every usable source clip with the current marker--ORB backend.

The audit deliberately disables MediaPipe hand-joint inference, keeps wrist and
static-marker geometry, and normalizes old rigid-board runs to the current
independent-static-marker policy.  Existing observation caches are reused only
when that policy already matches; otherwise ArUco observations are recomputed.
"""

from __future__ import annotations

# Support direct execution from a source checkout.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Iterable

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT / "output" / "orb_only_audit_20260905"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _actions_for_meta(path: Path) -> Path:
    return path.with_name(path.name.replace(".meta.json", ".jsonl"))


def discover_sources() -> list[dict[str, Any]]:
    """Select the fullest native-SLAM result as configuration evidence per clip."""
    best: dict[Path, tuple[tuple[int, float], Path, dict[str, Any]]] = {}
    for meta_path in PROJECT.rglob("*.meta.json"):
        meta_text = str(meta_path)
        if any(token in meta_text for token in (
            "orb_only_audit_", "orb_repeatability_", "orb_cross_room_"
        )):
            continue
        try:
            meta = _load_json(meta_path)
        except (OSError, ValueError):
            continue
        video_value = meta.get("video")
        if not video_value:
            continue
        video = Path(video_value).resolve()
        if "calibration" in video.name.lower():
            continue
        actions = _actions_for_meta(meta_path)
        calibration = Path(str(meta.get("calibration", "")))
        bands = [Path(value) for value in meta.get("bands", [])]
        if not (video.is_file() and actions.is_file() and calibration.is_file()):
            continue
        if len(bands) < 1 or any(not band.is_file() for band in bands):
            continue
        score = (int(meta.get("frames") or 0), meta_path.stat().st_mtime)
        old = best.get(video)
        if old is None or score > old[0]:
            best[video] = (score, meta_path.resolve(), meta)

    rows: list[dict[str, Any]] = []
    for video, (_, meta_path, meta) in sorted(best.items(), key=lambda item: str(item[0])):
        digest = hashlib.sha1(str(video).encode()).hexdigest()[:8]
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", video.stem).strip("_") or "clip"
        rows.append(
            {
                "video": video,
                "source_meta": meta_path,
                "source_actions": _actions_for_meta(meta_path),
                "calibration": Path(meta["calibration"]),
                "bands": [Path(value) for value in meta["bands"]],
                "source_frames": int(meta.get("frames") or 0),
                "source_fps": float(meta.get("fps") or 0.0),
                "source_auto_marker_map": bool(meta.get("auto_marker_map", {}).get("enabled")),
                "source_hand_joints": bool(meta.get("hand_joints_enabled", True)),
                "static_marker_ids": list(
                    meta.get("auto_marker_map", {}).get("static_marker_ids")
                    or range(20, 50)
                ),
                "static_marker_size_mm": float(
                    meta.get("auto_marker_map", {}).get("marker_size_mm") or 48.0
                ),
                "strict_marker_corners": (
                    meta.get("marker_corner_policy", {}).get("mode") == "strict"
                ),
                "slug": f"{stem}_{digest}",
            }
        )
    return rows


def _pose(pose: Any) -> tuple[np.ndarray, np.ndarray] | None:
    if not isinstance(pose, dict):
        return None
    try:
        t = np.asarray(pose["translation_m"], dtype=float).reshape(3)
        q = np.asarray(pose["quaternion_wxyz"], dtype=float).reshape(4)
    except (KeyError, TypeError, ValueError):
        return None
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(q)):
        return None
    norm = float(np.linalg.norm(q))
    if norm < 1e-9:
        return None
    return t, q / norm


def _rotation_delta_deg(a: np.ndarray, b: np.ndarray) -> float:
    dot = min(1.0, max(-1.0, abs(float(np.dot(a, b)))))
    return math.degrees(2.0 * math.acos(dot))


def _contiguous_ranges(indices: Iterable[int], timestamps: list[float]) -> list[dict[str, Any]]:
    values = sorted(set(indices))
    if not values:
        return []
    runs: list[tuple[int, int]] = []
    start = previous = values[0]
    for index in values[1:]:
        if index != previous + 1:
            runs.append((start, previous))
            start = index
        previous = index
    runs.append((start, previous))
    return [
        {
            "start_frame": a,
            "end_frame": b,
            "start_s": timestamps[a],
            "end_s": timestamps[b],
            "frames": b - a + 1,
        }
        for a, b in runs
    ]


def _series_jumps(
    poses: list[tuple[np.ndarray, np.ndarray] | None],
    timestamps: list[float],
    map_ids: list[Any],
    minimum_m: float = 0.02,
    sources: list[str] | None = None,
) -> list[dict[str, Any]]:
    jumps = []
    for index in range(1, len(poses)):
        before, after = poses[index - 1], poses[index]
        if before is None or after is None or map_ids[index - 1] != map_ids[index]:
            continue
        dt = timestamps[index] - timestamps[index - 1]
        if dt <= 0 or dt > 0.25:
            continue
        translation = float(np.linalg.norm(after[0] - before[0]))
        rotation = _rotation_delta_deg(before[1], after[1])
        if translation >= minimum_m or rotation >= 8.0:
            jump = {
                    "frame": index,
                    "time_s": timestamps[index],
                    "translation_mm": 1000.0 * translation,
                    "rotation_deg": rotation,
                    "speed_m_s": translation / dt,
                }
            if sources is not None:
                jump["source_before"] = sources[index - 1]
                jump["source_after"] = sources[index]
            jumps.append(jump)
    jumps.sort(key=lambda row: (row["translation_mm"], row["rotation_deg"]), reverse=True)
    return jumps


def _single_frame_spikes(
    poses: list[tuple[np.ndarray, np.ndarray] | None],
    timestamps: list[float],
    map_ids: list[Any],
    minimum_m: float = 0.03,
    sources: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Find isolated translation outliers relative to adjacent valid poses.

    Unlike a raw frame-to-frame step, this remains useful during deliberate
    camera motion: a smooth fast trajectory predicts the middle sample from
    its two neighbours.  It is still a diagnostic, not accuracy evidence.
    """
    spikes = []
    for index in range(1, len(poses) - 1):
        before, current, after = poses[index - 1:index + 2]
        if before is None or current is None or after is None:
            continue
        if not (map_ids[index - 1] == map_ids[index] == map_ids[index + 1]):
            continue
        left = timestamps[index] - timestamps[index - 1]
        right = timestamps[index + 1] - timestamps[index]
        total = left + right
        if left <= 0 or right <= 0 or left > 0.25 or right > 0.25 or total <= 0:
            continue
        expected = before[0] + (left / total) * (after[0] - before[0])
        residual = float(np.linalg.norm(current[0] - expected))
        if residual < minimum_m:
            continue
        row = {
            "frame": index,
            "time_s": timestamps[index],
            "interpolation_residual_mm": 1000.0 * residual,
            "neighbour_span_mm": 1000.0 * float(np.linalg.norm(after[0] - before[0])),
        }
        if sources is not None:
            row["source"] = sources[index]
        spikes.append(row)
    spikes.sort(key=lambda row: row["interpolation_residual_mm"], reverse=True)
    return spikes


def _low_motion_windows(
    poses: list[tuple[np.ndarray, np.ndarray] | None],
    timestamps: list[float],
    map_ids: list[Any],
    fps: float,
) -> list[dict[str, Any]]:
    """Return conservative pose-cluster windows; these are not ground truth."""
    width = max(8, int(round(max(fps, 10.0) * 1.5)))
    stride = max(4, width // 3)
    windows = []
    for start in range(0, max(0, len(poses) - width + 1), stride):
        stop = start + width
        valid = [(i, poses[i]) for i in range(start, stop) if poses[i] is not None]
        if len(valid) < int(0.8 * width):
            continue
        ids = {map_ids[i] for i, _ in valid}
        if len(ids) != 1:
            continue
        xyz = np.asarray([value[0] for _, value in valid])
        centre = np.median(xyz, axis=0)
        radial = np.linalg.norm(xyz - centre, axis=1)
        endpoint = float(np.linalg.norm(xyz[-1] - xyz[0]))
        p95 = float(np.percentile(radial, 95))
        # Broad enough to retain noisy stationary periods, narrow enough to
        # exclude most deliberate wrist translations.
        if endpoint > 0.015 or p95 > 0.035:
            continue
        windows.append(
            {
                "start_s": timestamps[start],
                "end_s": timestamps[stop - 1],
                "map_id": next(iter(ids)),
                "valid_fraction": len(valid) / width,
                "rms_mm": 1000.0 * float(np.sqrt(np.mean(radial**2))),
                "p95_mm": 1000.0 * p95,
                "endpoint_mm": 1000.0 * endpoint,
            }
        )
    windows.sort(key=lambda row: row["rms_mm"], reverse=True)
    return windows


def analyze(actions_path: Path, meta_path: Path) -> dict[str, Any]:
    meta = _load_json(meta_path)
    records = [json.loads(line) for line in actions_path.open() if line.strip()]
    timestamps = [float(row.get("timestamp_s", index / max(float(meta.get("fps") or 30), 1)))
                  for index, row in enumerate(records)]
    map_ids = [row.get("camera_submap_id") for row in records]
    camera = [_pose(row.get("camera_world_pose_fused")) for row in records]
    invalid = [i for i, pose in enumerate(camera) if pose is None]

    source_counts: dict[str, int] = {}
    for row in records:
        key = str(row.get("camera_world_source", "unknown"))
        source_counts[key] = source_counts.get(key, 0) + 1

    wrist: dict[str, Any] = {}
    names = sorted({name for row in records for name in row.get("hands", {})})
    for name in names:
        poses = []
        wrist_maps = []
        for row in records:
            hand = row.get("hands", {}).get(name, {})
            poses.append(_pose(hand.get("wrist_world_graph")))
            wrist_maps.append(hand.get("world_submap_id") or row.get("camera_submap_id"))
        jumps = _series_jumps(poses, timestamps, wrist_maps)
        windows = _low_motion_windows(
            poses, timestamps, wrist_maps, float(meta.get("fps") or 30.0)
        )
        wrist[name] = {
            "valid_frames": sum(value is not None for value in poses),
            "valid_fraction": sum(value is not None for value in poses) / max(len(poses), 1),
            "jumps_over_20mm_or_8deg": len(jumps),
            "largest_jumps": jumps[:12],
            "probable_low_motion_windows": windows[:12],
            "note": "Low-motion windows are automatically inferred pose clusters, not ground-truth stationary labels.",
        }

    events = meta.get("head_slam", {}).get("marker_graph_events", [])
    event_counts: dict[str, int] = {}
    for event in events:
        key = f"{event.get('type')}:{event.get('status')}"
        event_counts[key] = event_counts.get(key, 0) + 1

    map_changes = []
    for index in range(1, len(map_ids)):
        if map_ids[index] != map_ids[index - 1]:
            map_changes.append(
                {
                    "frame": index,
                    "time_s": timestamps[index],
                    "from": map_ids[index - 1],
                    "to": map_ids[index],
                }
            )

    camera_sources = [str(row.get("camera_world_source", "unknown")) for row in records]
    camera_jumps = _series_jumps(camera, timestamps, map_ids, sources=camera_sources)
    camera_spikes = _single_frame_spikes(
        camera, timestamps, map_ids, sources=camera_sources
    )
    lost = _contiguous_ranges(invalid, timestamps)
    return {
        "video": meta.get("video"),
        "frames": len(records),
        "fps": float(meta.get("fps") or 0.0),
        "duration_s": timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0,
        "camera_valid_frames": len(records) - len(invalid),
        "camera_valid_fraction": (len(records) - len(invalid)) / max(len(records), 1),
        "camera_sources": source_counts,
        "maps_seen": sorted({str(value) for value in map_ids if value is not None}),
        "map_changes": map_changes,
        "lost_intervals": lost,
        "camera_jumps_over_20mm_or_8deg": len(camera_jumps),
        "largest_camera_jumps": camera_jumps[:20],
        "single_frame_camera_spikes_over_30mm": len(camera_spikes),
        "largest_single_frame_camera_spikes": camera_spikes[:20],
        "marker_graph_event_counts": event_counts,
        "marker_graph_events": events,
        "wrist": wrist,
        "tracking_timing": meta.get("head_slam", {}).get("tracking_timing", {}),
    }


def _command(row: dict[str, Any], destination: Path) -> list[str]:
    output = destination / "actions.jsonl"
    command = [
        str(PROJECT / ".venv" / "bin" / "python"),
        str(PROJECT / "export_action_labels.py"),
        str(row["video"]),
        "--calib", str(row["calibration"]),
        "--output", str(output),
        "--head-slam",
        "--no-slam-replay",
        "--static-marker-ids", ",".join(map(str, row["static_marker_ids"])),
        "--static-marker-size-mm", str(row["static_marker_size_mm"]),
    ]
    for band in row["bands"]:
        command.extend(["--band", str(band)])
    if row["strict_marker_corners"]:
        command.append("--strict-marker-corners")
    if row["source_auto_marker_map"]:
        # Matching the cache contract does not run MediaPipe again: cached
        # joints are merely carried through while the native SLAM is rerun.
        if not row["source_hand_joints"]:
            command.append("--no-hand-joints")
        command.extend(["--reuse-observations", str(row["source_actions"])])
    else:
        command.append("--no-hand-joints")
    return command


def write_report(root: Path, results: list[dict[str, Any]]) -> None:
    lines = [
        "# Current marker–ORB all-video audit",
        "",
        "All runs use the current native ORB-SLAM3 backend, independent static markers, "
        "and no MediaPipe hand-joint inference. Wrist-marker trajectories remain enabled. "
        "Automatically inferred low-motion windows are diagnostics, not ground truth.",
        "",
        "| clip | frames | valid | maps | losses | camera jumps | isolated spikes | re-anchor accepted/rejected |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        events = result.get("marker_graph_event_counts", {})
        accepted = events.get("scale_reanchor:accepted", 0)
        rejected = events.get("scale_reanchor:rejected", 0)
        lines.append(
            "| {name} | {frames} | {valid:.1%} | {maps} | {losses} | {jumps} | {spikes} | {accepted}/{rejected} |".format(
                name=Path(str(result.get("video"))).name,
                frames=result.get("frames", 0),
                valid=result.get("camera_valid_fraction", 0.0),
                maps=len(result.get("maps_seen", [])),
                losses=len(result.get("lost_intervals", [])),
                jumps=result.get("camera_jumps_over_20mm_or_8deg", 0),
                spikes=result.get("single_frame_camera_spikes_over_30mm", 0),
                accepted=accepted,
                rejected=rejected,
            )
        )
    lines.extend(["", "## Per-clip notes", ""])
    for result in results:
        lines.append(f"### {Path(str(result.get('video'))).name}")
        lines.append("")
        lines.append(
            f"- Camera valid: {result.get('camera_valid_fraction', 0):.1%}; "
            f"lost intervals: {len(result.get('lost_intervals', []))}; "
            f"map changes: {len(result.get('map_changes', []))}."
        )
        jumps = result.get("largest_camera_jumps", [])
        if jumps:
            top = jumps[0]
            lines.append(
                f"- Largest same-map camera step: {top['translation_mm']:.1f} mm / "
                f"{top['rotation_deg']:.2f} deg at {top['time_s']:.3f} s."
            )
        else:
            lines.append("- No same-map camera step exceeded 20 mm or 8 deg.")
        spikes = result.get("largest_single_frame_camera_spikes", [])
        if spikes:
            top = spikes[0]
            lines.append(
                f"- Largest isolated one-frame residual: "
                f"{top['interpolation_residual_mm']:.1f} mm at {top['time_s']:.3f} s "
                "(diagnostic, not ground truth)."
            )
        for name, wrist in result.get("wrist", {}).items():
            windows = wrist.get("probable_low_motion_windows", [])
            worst = windows[0] if windows else None
            detail = (
                f"; worst inferred low-motion RMS {worst['rms_mm']:.2f} mm "
                f"({worst['start_s']:.2f}–{worst['end_s']:.2f} s)"
                if worst else ""
            )
            lines.append(
                f"- {name}: {wrist.get('valid_fraction', 0):.1%} valid, "
                f"{wrist.get('jumps_over_20mm_or_8deg', 0)} large steps{detail}."
            )
        lines.append("")
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")


def _process_row(
    number: int,
    total: int,
    row: dict[str, Any],
    root: Path,
    analyze_only: bool,
) -> tuple[int, dict[str, Any], dict[str, Any] | None]:
    """Process one independent clip; all writes stay inside its own directory."""
    destination = root / row["slug"]
    destination.mkdir(parents=True, exist_ok=True)
    actions = destination / "actions.jsonl"
    meta = destination / "actions.meta.json"
    command = _command(row, destination)
    entry = {
        key: (str(value) if isinstance(value, Path) else [str(x) for x in value]
              if key == "bands" else value)
        for key, value in row.items()
    }
    entry["command"] = command
    entry["reused_observations"] = "--reuse-observations" in command
    entry["status"] = "pending"
    print(f"[{number}/{total}] {row['video']}", flush=True)
    if not analyze_only and not (actions.is_file() and meta.is_file()):
        started = time.perf_counter()
        with (destination / "run.log").open("w") as log:
            completed = subprocess.run(
                command,
                cwd=PROJECT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        # Old action files may predate the observation-cache contract. Fail
        # closed on reuse, then perform one explicit fresh ArUco-only pass.
        log_text = (destination / "run.log").read_text(errors="replace")
        cache_contract_failure = any(token in log_text for token in (
            "observation cache", "cached observation", "cached hand measurement"
        ))
        if (completed.returncode and "--reuse-observations" in command
                and cache_contract_failure):
            fallback = []
            skip_next = False
            for value in command:
                if skip_next:
                    skip_next = False
                    continue
                if value == "--reuse-observations":
                    skip_next = True
                    continue
                fallback.append(value)
            if "--no-hand-joints" not in fallback:
                fallback.append("--no-hand-joints")
            with (destination / "run.log").open("a") as log:
                log.write("\ncache reuse failed; retrying a fresh ArUco-only pass\n")
                completed = subprocess.run(
                    fallback,
                    cwd=PROJECT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            entry["fallback_command"] = fallback
        entry["wall_seconds"] = time.perf_counter() - started
        entry["returncode"] = completed.returncode
    result = None
    if actions.is_file() and meta.is_file():
        try:
            result = analyze(actions, meta)
            (destination / "audit.json").write_text(
                json.dumps(result, indent=2, allow_nan=False) + "\n"
            )
            entry["status"] = "complete"
        except Exception as exc:  # keep batch progress when one legacy clip is malformed
            entry["status"] = "analysis_failed"
            entry["error"] = repr(exc)
    elif entry.get("returncode"):
        entry["status"] = "run_failed"
    return number, entry, result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--match", help="regular expression matched against source path")
    parser.add_argument(
        "--jobs", type=int, default=1,
        help="independent videos to process concurrently (default: 1)",
    )
    args = parser.parse_args()

    rows = discover_sources()
    if args.match:
        pattern = re.compile(args.match)
        rows = [row for row in rows if pattern.search(str(row["video"]))]
    if args.limit is not None:
        rows = rows[: args.limit]
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    completed_entries: dict[int, dict[str, Any]] = {}
    completed_results: dict[int, dict[str, Any]] = {}

    def record(outcome: tuple[int, dict[str, Any], dict[str, Any] | None]) -> None:
        number, entry, result = outcome
        completed_entries[number] = entry
        if result is not None:
            completed_results[number] = result
        manifest = [completed_entries[index] for index in sorted(completed_entries)]
        results = [completed_results[index] for index in sorted(completed_results)]
        (root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n"
        )
        write_report(root, results)

    jobs = max(1, args.jobs)
    if jobs == 1:
        for number, row in enumerate(rows, 1):
            record(_process_row(number, len(rows), row, root, args.analyze_only))
    else:
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = [
                executor.submit(
                    _process_row, number, len(rows), row, root, args.analyze_only
                )
                for number, row in enumerate(rows, 1)
            ]
            for future in as_completed(futures):
                record(future.result())
    results = [completed_results[index] for index in sorted(completed_results)]
    print(f"completed {len(results)}/{len(rows)} clips; report: {root / 'REPORT.md'}")
    return 0 if len(results) == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
