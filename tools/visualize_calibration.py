#!/usr/bin/env python3
from __future__ import annotations

# Support direct execution from a source checkout.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from aruco_track.models import Calibration


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render observed/reprojected ChArUco corners and undistortion QA"
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("--calib", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--square-mm", type=float, default=24.9)
    parser.add_argument("--marker-mm", type=float, default=19.0)
    args = parser.parse_args()

    calibration = Calibration.load(args.calib)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard(
        (7, 10), args.square_mm / 1000.0, args.marker_mm / 1000.0, dictionary
    )
    detector = cv2.aruco.CharucoDetector(board)
    board_points = np.asarray(board.getChessboardCorners(), dtype=np.float64)

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    input_fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    if (width, height) != calibration.image_size:
        calibration = calibration.scaled_to((width, height))
    output_fps = min(args.fps, input_fps)
    step = max(1, int(round(input_fps / output_fps)))
    actual_output_fps = input_fps / step

    panel_width = 960
    panel_height = round(panel_width * height / width)
    output_size = (panel_width * 2, panel_height)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), actual_output_fps, output_size
    )
    if not writer.isOpened():
        capture.release()
        raise SystemExit(f"cannot create {args.output}")

    decoded = rendered = detected = 0
    errors: list[float] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index = decoded
            decoded += 1
            if frame_index % step:
                continue

            observed = frame.copy()
            corners, ids, _, _ = detector.detectBoard(frame)
            error: float | None = None
            if ids is not None and len(ids) >= 6:
                indices = ids.reshape(-1).astype(np.int32)
                image_points = corners.reshape(-1, 2).astype(np.float64)
                object_points = board_points[indices]
                ok_pose, rvec, tvec = cv2.solvePnP(
                    object_points,
                    image_points,
                    calibration.camera_matrix,
                    calibration.dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                if ok_pose:
                    projected, _ = cv2.projectPoints(
                        object_points,
                        rvec,
                        tvec,
                        calibration.camera_matrix,
                        calibration.dist_coeffs,
                    )
                    projected = projected.reshape(-1, 2)
                    if np.all(np.isfinite(projected)) and np.max(np.abs(projected)) < 1e6:
                        residual = projected - image_points
                        error = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
                        errors.append(error)
                        detected += 1
                        for measured, prediction in zip(image_points, projected):
                            measured_i = tuple(np.rint(measured).astype(int))
                            prediction_i = tuple(np.rint(prediction).astype(int))
                            cv2.line(observed, measured_i, prediction_i, (190, 80, 190), 1, cv2.LINE_AA)
                            cv2.circle(observed, measured_i, 4, (70, 220, 90), 1, cv2.LINE_AA)
                            cv2.drawMarker(
                                observed,
                                prediction_i,
                                (190, 80, 190),
                                cv2.MARKER_CROSS,
                                7,
                                1,
                                cv2.LINE_AA,
                            )

            undistorted = cv2.undistort(
                frame,
                calibration.camera_matrix,
                calibration.dist_coeffs,
                None,
                calibration.camera_matrix,
            )
            left = cv2.resize(observed, (panel_width, panel_height), interpolation=cv2.INTER_AREA)
            right = cv2.resize(undistorted, (panel_width, panel_height), interpolation=cv2.INTER_AREA)
            cv2.rectangle(left, (0, 0), (panel_width, 58), (20, 20, 20), -1)
            cv2.rectangle(right, (0, 0), (panel_width, 58), (20, 20, 20), -1)
            status = (
                f"frame {frame_index}  corners {0 if ids is None else len(ids)}  "
                f"RMS {error:.2f} px"
                if error is not None
                else f"frame {frame_index}  board not usable"
            )
            cv2.putText(left, "Observed (green) vs reprojected (magenta)", (18, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 245, 245), 1, cv2.LINE_AA)
            cv2.putText(left, status, (18, 48), cv2.FONT_HERSHEY_SIMPLEX,
                        0.58, (245, 245, 245), 1, cv2.LINE_AA)
            cv2.putText(right, "Undistorted with the fitted phone calibration", (18, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 1, cv2.LINE_AA)
            writer.write(np.hstack([left, right]))
            rendered += 1
    finally:
        capture.release()
        writer.release()

    summary = {
        "video": str(args.video.resolve()),
        "calibration": str(args.calib.resolve()),
        "decoded_frames": decoded,
        "rendered_frames": rendered,
        "usable_board_frames": detected,
        "usable_fraction": detected / rendered if rendered else 0.0,
        "reprojection_median_px": float(np.median(errors)) if errors else None,
        "reprojection_p95_px": float(np.percentile(errors, 95)) if errors else None,
        "reprojection_max_px": float(np.max(errors)) if errors else None,
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.output} and {summary_path}")


if __name__ == "__main__":
    main()
