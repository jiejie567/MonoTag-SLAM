"""Paired, deterministic corner-policy audit with a frozen metric marker map.

Reuses identical detections and wrist poses; does not run ORB or claim absolute
accuracy. The selected wrist must actually be stationary for spread to be useful.
"""
from __future__ import annotations

# Support direct execution from a source checkout.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from aruco_track.detector import ArucoDetector
from aruco_track.marker_quality import evaluate_marker_boundary
from aruco_track.models import BandLayout, Calibration, Pose
from aruco_track.pipeline import relative_pose
from aruco_track.tag_graph import optimize_tag_pose


def spread(poses: dict[int, Pose]) -> dict:
    if not poses:
        return {"frames": 0}
    indices = sorted(poses)
    positions = np.array([poses[index].tvec.reshape(3) for index in indices])
    distance = 1000 * np.linalg.norm(positions - np.median(positions, axis=0), axis=1)
    rotations = Rotation.from_matrix([poses[index].rotation_matrix for index in indices])
    angles = np.degrees((rotations.mean().inv() * rotations).magnitude())
    consecutive = np.diff(indices) == 1
    steps = 1000 * np.linalg.norm(np.diff(positions, axis=0), axis=1)[consecutive]
    return {
        "frames": len(poses),
        "position_radial_median_mm": float(np.median(distance)),
        "position_radial_p95_mm": float(np.percentile(distance, 95)),
        "rotation_radial_p95_deg": float(np.percentile(angles, 95)),
        "consecutive_step_p95_mm": float(np.percentile(steps, 95)) if len(steps) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--calib", type=Path, required=True)
    parser.add_argument("--marker-map", type=Path, required=True)
    parser.add_argument("--wrist-actions", type=Path, required=True)
    parser.add_argument("--wrist", default="strap_band_R")
    parser.add_argument("--max-frames", type=int, default=350)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    calibration = Calibration.load(args.calib)
    data = json.loads(args.marker_map.read_text())
    submap = data["submaps"][0]
    layout = BandLayout("frozen_world", data["dictionary"], {
        item["id"]: np.asarray(item["object_points_m"]) for item in submap["markers"]
    })
    records = [json.loads(line) for line in args.wrist_actions.read_text().splitlines()]
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, layout.dictionary))
    templates = {mid: cv2.aruco.generateImageMarker(dictionary, mid, (dictionary.markerSize + 2) * 20)
                 for mid in layout.markers}
    detector = ArucoDetector(layout.dictionary, camera_matrix=calibration.camera_matrix,
                             dist_coeffs=calibration.dist_coeffs)
    modes = ("strict", "weighted")
    previous = {mode: None for mode in modes}
    world = {mode: {} for mode in modes}
    localized = Counter()
    soft_detected = Counter()
    soft_used = Counter()
    retained = Counter()
    times = []
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {args.video}")
    try:
        for index in range(min(args.max_frames, len(records))):
            ok, frame = capture.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections = {mid: corners for mid, corners in detector.detect(gray).items()
                          if mid in templates}
            started = perf_counter()
            quality = {mid: evaluate_marker_boundary(*detector._boundary_image(gray, corners), templates[mid])
                       for mid, corners in detections.items()}
            times.append(1000 * (perf_counter() - started))
            soft = {mid for mid, item in quality.items() if item.reason == "soft_grid"}
            soft_detected.update(soft)
            hand = records[index]["hands"][args.wrist]
            item = hand.get("wrist_camera_graph") or hand.get("wrist_camera_raw")
            wrist = None
            if item is not None:
                quat = np.roll(item["quaternion_wxyz"], -1)
                wrist = Pose(Rotation.from_quat(quat).as_rotvec().reshape(3, 1),
                             np.asarray(item["translation_m"]).reshape(3, 1), 0.0)
            for mode in modes:
                selected = {mid: corners for mid, corners in detections.items() if quality[mid].accepted
                            and (mode == "weighted" or quality[mid].information_weight == 1.0)}
                retained[mode] += len(selected)
                result = optimize_tag_pose(selected, layout, calibration, previous[mode],
                                           max_graph_error_px=2.5,
                                           marker_weights={mid: quality[mid].information_weight for mid in selected})
                if result.pose is None:
                    continue
                previous[mode] = result.pose
                if mode == "weighted":
                    soft_used.update(soft.intersection(result.accepted_marker_ids))
                # Match the exporter's trusted-anchor confidence threshold.
                if result.confidence < 0.35:
                    continue
                localized[mode] += 1
                if wrist is not None:
                    world[mode][index] = relative_pose(result.pose, wrist)
    finally:
        capture.release()
    common = set(world["strict"]).intersection(world["weighted"])
    report = {
        "video": str(args.video.resolve()), "marker_map": str(args.marker_map.resolve()),
        "wrist": args.wrist, "frames": len(times), "note": "stationary-wrist stability proxy, not ground-truth accuracy",
        "retained_observations": dict(retained), "trusted_marker_pose_frames": dict(localized),
        "soft_detected_by_id": dict(soft_detected), "soft_geometrically_accepted_by_id": dict(soft_used),
        "all_valid_frames": {mode: spread(world[mode]) for mode in modes},
        "common_frames": {mode: spread({index: world[mode][index] for index in common}) for mode in modes},
        "boundary_check_ms_median_p95": np.percentile(times, [50, 95]).tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
