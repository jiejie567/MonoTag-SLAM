#!/usr/bin/env python3
from __future__ import annotations

# Support direct execution from a source checkout.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from aruco_track.camera import open_camera
from aruco_track.models import Calibration


def _view_descriptor(corners: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    """Describe where and how a ChArUco view covers the image."""
    points = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    width, height = image_size
    centre = points.mean(axis=0) / np.array([width, height], dtype=np.float64)
    span = np.ptp(points, axis=0) / np.array([width, height], dtype=np.float64)
    normalised = (points - points.mean(axis=0)) / np.array([width, height], dtype=np.float64)
    covariance = np.cov(normalised.T) if len(points) > 1 else np.zeros((2, 2))
    hull = cv2.convexHull(points.astype(np.float32))
    area = cv2.contourArea(hull) / float(width * height)
    return np.array([
        centre[0] * 2.0,
        centre[1] * 2.0,
        span[0] * 1.5,
        span[1] * 1.5,
        np.sqrt(max(area, 0.0)) * 2.0,
        covariance[0, 0] * 8.0,
        covariance[1, 1] * 8.0,
        covariance[0, 1] * 12.0,
    ])


def _select_diverse_views(candidates: list[dict[str, object]], count: int) -> list[dict[str, object]]:
    if len(candidates) <= count:
        return candidates
    descriptors = np.stack([candidate["descriptor"] for candidate in candidates])
    corner_counts = np.asarray([len(candidate["ids"]) for candidate in candidates])
    selected = [int(np.argmax(corner_counts))]
    remaining = np.ones(len(candidates), dtype=bool)
    remaining[selected[0]] = False
    minimum_distance = np.linalg.norm(descriptors - descriptors[selected[0]], axis=1)
    while len(selected) < count:
        # A small corner-count bonus avoids choosing a severely cropped view merely
        # because it is geometrically unusual.
        score = minimum_distance + 0.08 * corner_counts / max(float(corner_counts.max()), 1.0)
        score[~remaining] = -np.inf
        index = int(np.argmax(score))
        selected.append(index)
        remaining[index] = False
        minimum_distance = np.minimum(
            minimum_distance,
            np.linalg.norm(descriptors - descriptors[index], axis=1),
        )
    return [candidates[index] for index in selected]


def _collect_video_views(
    path: Path,
    detector: cv2.aruco.CharucoDetector,
    requested: int,
    sample_hz: float,
    min_corners: int,
) -> tuple[list[dict[str, object]], tuple[int, int], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise SystemExit(f"cannot open calibration video: {path}")
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if width <= 0 or height <= 0 or fps <= 0:
        capture.release()
        raise SystemExit(f"invalid calibration video metadata: {path}")
    step = max(1, int(round(fps / max(sample_hz, 0.1))))
    candidates: list[dict[str, object]] = []
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % step == 0:
            charuco_corners, charuco_ids, _, _ = detector.detectBoard(frame)
            if charuco_ids is not None and len(charuco_ids) >= min_corners:
                candidates.append({
                    "frame": frame_index,
                    "time_s": frame_index / fps,
                    "corners": charuco_corners.copy(),
                    "ids": charuco_ids.copy(),
                    "descriptor": _view_descriptor(charuco_corners, (width, height)),
                })
        frame_index += 1
    capture.release()
    if len(candidates) < 8:
        raise SystemExit(
            f"need at least 8 usable views; detected {len(candidates)} in {path}"
        )
    # Keep extra unseen views for a hold-out reprojection check.
    selected = _select_diverse_views(candidates, min(len(candidates), requested + 10))
    print(
        f"detected {len(candidates)} candidate views; selected "
        f"{min(requested, len(selected))} calibration + {max(0, len(selected) - requested)} hold-out",
        flush=True,
    )
    return selected, (width, height), fps


def _view_reprojection_error(
    board: cv2.aruco.CharucoBoard,
    corners: np.ndarray,
    ids: np.ndarray,
    matrix: np.ndarray,
    distortion: np.ndarray,
) -> float | None:
    indices = np.asarray(ids, dtype=np.int32).reshape(-1)
    object_points = np.asarray(board.getChessboardCorners(), dtype=np.float64)[indices]
    image_points = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    ok, rvec, tvec = cv2.solvePnP(
        object_points, image_points, matrix, distortion, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return None
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, matrix, distortion)
    residual = projected.reshape(-1, 2) - image_points
    return float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate a camera from a ChArUco board")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--video", type=Path, help="calibrate offline from a recorded video")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--square-mm", type=float, default=24.9)
    parser.add_argument("--marker-mm", type=float, default=19.0)
    parser.add_argument("--squares-x", type=int, default=7)
    parser.add_argument("--squares-y", type=int, default=10)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--sample-hz", type=float, default=5.0,
                        help="candidate detection rate for --video")
    parser.add_argument("--min-corners", type=int, default=12)
    parser.add_argument(
        "--max-view-rms-px",
        type=float,
        default=1.5,
        help="reject whole calibration views above this reprojection RMS",
    )
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--output", default="calib/camera_1920x1080.json")
    parser.add_argument("--window-x", type=int)
    parser.add_argument("--window-y", type=int, default=30)
    args = parser.parse_args()

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard(
        (args.squares_x, args.squares_y), args.square_mm / 1000.0, args.marker_mm / 1000.0, dictionary
    )
    detector = cv2.aruco.CharucoDetector(board)
    all_corners: list[np.ndarray] = []
    all_ids: list[np.ndarray] = []
    validation_views: list[dict[str, object]] = []
    video_fps: float | None = None
    if args.video is not None:
        selected, image_size, video_fps = _collect_video_views(
            args.video, detector, args.frames, args.sample_hz, args.min_corners
        )
        training_views = selected[:args.frames]
        validation_views = selected[args.frames:]
        all_corners = [view["corners"] for view in training_views]
        all_ids = [view["ids"] for view in training_views]
    else:
        image_size = (args.width, args.height)
        capture = open_camera(args.camera, args.width, args.height)
        last_capture = 0.0
        cv2.namedWindow("calibration", cv2.WINDOW_NORMAL)
        if args.window_x is not None:
            cv2.moveWindow("calibration", args.window_x, args.window_y)
        try:
            while len(all_corners) < args.frames:
                ok, frame = capture.read()
                if not ok:
                    break
                charuco_corners, charuco_ids, _, _ = detector.detectBoard(frame)
                preview = frame.copy()
                if charuco_ids is not None:
                    cv2.aruco.drawDetectedCornersCharuco(preview, charuco_corners, charuco_ids)
                cv2.putText(preview, f"views {len(all_corners)}/{args.frames} | space=capture q=quit", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 220, 30), 2)
                cv2.imshow("calibration", preview)
                key = cv2.waitKey(1) & 0xFF
                enough = charuco_ids is not None and len(charuco_ids) >= args.min_corners
                automatic = args.auto and enough and time.monotonic() - last_capture > 0.75
                if enough and (key == ord(" ") or automatic):
                    all_corners.append(charuco_corners.copy())
                    all_ids.append(charuco_ids.copy())
                    last_capture = time.monotonic()
                    print(
                        f"captured calibration view {len(all_corners)}/{args.frames} "
                        f"with {len(charuco_ids)} corners",
                        flush=True,
                    )
                if key in (ord("q"), 27):
                    break
        finally:
            capture.release()
            cv2.destroyAllWindows()
    if len(all_corners) < 8:
        raise SystemExit("need at least 8 usable views")
    if args.video is not None:
        training_metadata = training_views
    else:
        training_metadata = [
            {"frame": None, "time_s": None, "corners": corners, "ids": ids}
            for corners, ids in zip(all_corners, all_ids)
        ]
    rejected_training_views: list[dict[str, object]] = []
    while True:
        rms, matrix, distortion, _, _ = cv2.aruco.calibrateCameraCharuco(
            all_corners, all_ids, board, image_size, None, None
        )
        training_errors = [
            _view_reprojection_error(board, corners, ids, matrix, distortion)
            for corners, ids in zip(all_corners, all_ids)
        ]
        finite_errors = np.asarray([
            error if error is not None else np.inf for error in training_errors
        ])
        worst = int(np.argmax(finite_errors))
        if (
            finite_errors[worst] <= args.max_view_rms_px
            or len(all_corners) <= max(12, min(20, args.frames))
        ):
            break
        rejected = training_metadata.pop(worst)
        rejected_training_views.append({
            "frame": rejected.get("frame"),
            "time_s": rejected.get("time_s"),
            "rms_px": float(finite_errors[worst]),
        })
        all_corners.pop(worst)
        all_ids.pop(worst)
    if rejected_training_views:
        print(
            f"rejected {len(rejected_training_views)} high-residual calibration views; "
            f"kept {len(all_corners)}",
            flush=True,
        )
    validation_errors = [
        _view_reprojection_error(
            board, view["corners"], view["ids"], matrix, distortion
        )
        for view in validation_views
    ]
    valid_training = [error for error in training_errors if error is not None]
    valid_validation = [error for error in validation_errors if error is not None]
    Calibration(matrix, distortion, image_size).save(
        args.output,
        rms_reprojection_error=float(rms),
        frames=len(all_corners),
        square_mm=args.square_mm,
        marker_mm=args.marker_mm,
        source_video=str(args.video.resolve()) if args.video is not None else None,
        source_fps=video_fps,
        per_view_rms_px=valid_training,
        holdout_rms_px=valid_validation,
        holdout_median_px=(float(np.median(valid_validation)) if valid_validation else None),
        holdout_p95_px=(float(np.percentile(valid_validation, 95)) if valid_validation else None),
        rejected_training_views=rejected_training_views,
        max_view_rms_px=args.max_view_rms_px,
    )
    holdout_summary = (
        f"; hold-out median={np.median(valid_validation):.4f}px "
        f"p95={np.percentile(valid_validation, 95):.4f}px"
        if valid_validation else ""
    )
    print(
        f"saved {args.output}; RMS={rms:.4f}px using {len(all_corners)} views"
        f"{holdout_summary}"
    )


if __name__ == "__main__":
    main()
