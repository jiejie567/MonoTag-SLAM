from __future__ import annotations

import cv2
import numpy as np

from .marker_quality import MarkerBoundaryQuality, evaluate_marker_boundary


class _LegacyArucoDetector:
    """Compatibility wrapper for OpenCV 4.6's legacy ArUco bindings."""

    def __init__(self, dictionary, parameters):
        self._dictionary = dictionary
        self._parameters = parameters

    def detectMarkers(self, image):
        return cv2.aruco.detectMarkers(
            image, self._dictionary, parameters=self._parameters
        )

    def refineDetectedMarkers(
        self, image, board, corners, ids, rejected, camera_matrix, dist_coeffs
    ):
        # The 4.6 binding can segfault when passed a null camera matrix (the
        # modern class simply returns the current detections).  Board
        # refinement is only meaningful with calibration, so preserve the
        # decoded set and skip the unsafe legacy call in that case.
        if camera_matrix is None:
            return corners, ids, rejected, np.empty((0,), dtype=np.int32)
        return cv2.aruco.refineDetectedMarkers(
            image, board, corners, ids, rejected, camera_matrix, dist_coeffs
        )


def _make_aruco_detector(dictionary, corner_refinement):
    """Select the API supported by the installed OpenCV build."""
    if hasattr(cv2.aruco, "ArucoDetector"):
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = corner_refinement
        return cv2.aruco.ArucoDetector(dictionary, parameters)
    parameters = cv2.aruco.DetectorParameters_create()
    parameters.cornerRefinementMethod = corner_refinement
    return _LegacyArucoDetector(dictionary, parameters)


class ArucoDetector:
    def __init__(
        self,
        dictionary_name: str = "DICT_4X4_50",
        sharpen: bool = False,
        board_markers: list[dict[int, np.ndarray]] | None = None,
        camera_matrix: np.ndarray | None = None,
        dist_coeffs: np.ndarray | None = None,
        track_marker_gaps: int = 0,
        validate_corners: bool = False,
        boundary_marker_ids: set[int] | None = None,
        allow_soft_marker_corners: bool = True,
        wrist_marker_ids: set[int] | None = None,
        static_corner_refinement: bool = False,
    ):
        try:
            dictionary_id = getattr(cv2.aruco, dictionary_name)
        except AttributeError as exc:
            raise ValueError(f"unknown ArUco dictionary: {dictionary_name}") from exc
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self._dictionary = dictionary
        self._marker_templates: dict[int, np.ndarray] = {}
        self._detector = _make_aruco_detector(
            dictionary, cv2.aruco.CORNER_REFINE_SUBPIX
        )
        self._boards = [
            (
                cv2.aruco.Board(
                    [np.asarray(markers[marker_id], dtype=np.float32) for marker_id in sorted(markers)],
                    dictionary,
                    np.asarray(sorted(markers), dtype=np.int32),
                ),
                frozenset(markers),
            )
            for markers in board_markers or []
            if markers
        ]
        self._board_ids = frozenset().union(*(ids for _, ids in self._boards))
        self._camera_matrix = camera_matrix
        self._dist_coeffs = dist_coeffs
        self._quality_undistort_maps: tuple[np.ndarray, np.ndarray] | None = None
        self.last_recovered_ids: tuple[int, ...] = ()
        self.last_tracked_ids: tuple[int, ...] = ()
        self._track_marker_gaps = track_marker_gaps
        self._previous_gray: np.ndarray | None = None
        self._tracks: dict[int, tuple[np.ndarray, int, float]] = {}
        self.sharpen = sharpen
        self._wrist_marker_ids = set(wrist_marker_ids or ())
        self.validate_corners = validate_corners or bool(self._wrist_marker_ids)
        self._boundary_marker_ids = boundary_marker_ids if validate_corners else set()
        self.allow_soft_marker_corners = allow_soft_marker_corners
        self.last_boundary_quality: dict[int, MarkerBoundaryQuality] = {}
        self.last_rejected_detections: dict[int, np.ndarray] = {}
        self._static_corner_detector = None
        # A straight border in a rectified pinhole image need not be straight
        # in a raw distorted image. Keep SUBPIX there; do not reinterpret lens
        # curvature as a corner correction.
        if static_corner_refinement and (dist_coeffs is None or not np.any(dist_coeffs)):
            self._static_corner_detector = _make_aruco_detector(
                dictionary, cv2.aruco.CORNER_REFINE_APRILTAG
            )
        self.last_refined_ids: tuple[int, ...] = ()
        self.last_mask_detections: dict[int, np.ndarray] = {}

    def _refine_static_corners(self, gray, detections):
        """Refine decoded static IDs from current pixels, never from map poses.

        AprilTag's line-based corner localization is used with the SAME ArUco
        dictionary. No new identity, tracked-gap measurement, or wrist corner
        is introduced. Normal boundary/pose admission still runs afterwards.
        """
        self.last_refined_ids = ()
        self.last_mask_detections = {}
        allowed = set(self._boundary_marker_ids or ()) - self._wrist_marker_ids
        if self._static_corner_detector is None or not allowed.intersection(detections):
            return detections
        # Localization precision and ORB feature allocation are separate.
        # The existing padded exclusion already covers these bounded changes;
        # retain the original quads for its mask instead of changing which
        # background features receive the detector budget on every refinement.
        self.last_mask_detections = {mid: np.asarray(c).copy() for mid,c in detections.items()}
        corners, ids, _ = self._static_corner_detector.detectMarkers(gray)
        if ids is None:
            return detections
        refined = []
        flat_ids = ids.reshape(-1)
        for marker_id in sorted(allowed.intersection(detections)):
            indices = np.flatnonzero(flat_ids == marker_id)
            if len(indices) != 1:
                continue
            # AprilTag's edge coordinates use pixel-cell origins; OpenCV
            # projectPoints / SUBPIX use pixel centers. An axis-aligned square
            # occupying [30:130] has physical borders at 29.5 and 129.5, while
            # this detector reports 30 and 130. Do not mix those conventions.
            candidate = np.asarray(corners[int(indices[0])], dtype=np.float64).reshape(4, 2)-.5
            seed = detections[marker_id]
            side = float(np.min(np.linalg.norm(np.roll(seed, -1, axis=0)-seed, axis=1)))
            # Require eight pixels per dictionary cell on the shortest edge.
            # Tiny tags keep their original measured corners, not a noisier
            # alternative extrapolated from poorly resolved border segments.
            minimum_side = 8.*(self._dictionary.markerSize+2)
            mask_padding = max(8, int(round(.012*np.hypot(*gray.shape))))
            if (not np.isfinite(candidate).all() or side < minimum_side
                    or not cv2.isContourConvex(candidate.astype(np.float32))
                    or np.any(candidate < 0) or np.any(candidate >= [gray.shape[1], gray.shape[0]])
                    or np.max(np.linalg.norm(candidate-seed, axis=1)) > min(10., .1*side, mask_padding-1.)):
                continue
            detections[marker_id] = candidate
            refined.append(marker_id)
        self.last_refined_ids = tuple(refined)
        return detections

    def detect(self, frame: np.ndarray) -> dict[int, np.ndarray]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if self.sharpen:
            blurred = cv2.GaussianBlur(gray, (0, 0), 1.0)
            gray = cv2.addWeighted(gray, 1.7, blurred, -0.7, 0)
        corners, ids, rejected = self._detector.detectMarkers(gray)
        recovered_ids: set[int] = set()
        for board, board_ids in self._boards:
            detected_ids = set() if ids is None else set(ids.reshape(-1).tolist())
            if not detected_ids.intersection(board_ids):
                continue
            before = detected_ids
            corners, ids, rejected, _ = self._detector.refineDetectedMarkers(
                gray,
                board,
                corners,
                ids,
                rejected,
                self._camera_matrix,
                self._dist_coeffs,
            )
            after = set() if ids is None else set(ids.reshape(-1).tolist())
            recovered_ids.update((after - before).intersection(board_ids))
        self.last_recovered_ids = tuple(sorted(recovered_ids))
        detections = {} if ids is None else {
            int(marker_id): np.asarray(marker_corners, dtype=np.float64).reshape(4, 2)
            for marker_id, marker_corners in zip(ids.reshape(-1), corners)
        }
        detections = self._refine_static_corners(gray, detections)
        detections = self._track_short_gaps(gray, detections)
        self.last_boundary_quality = {}
        self.last_rejected_detections = {}
        if self.validate_corners:
            for marker_id, points in detections.items():
                if (self._boundary_marker_ids is not None
                        and marker_id not in self._boundary_marker_ids
                        and marker_id not in self._wrist_marker_ids):
                    continue
                if marker_id not in self._marker_templates:
                    self._marker_templates[marker_id] = cv2.aruco.generateImageMarker(
                        self._dictionary, marker_id, (self._dictionary.markerSize + 2) * 20
                    )
                quality = evaluate_marker_boundary(
                    *self._boundary_image(gray, points), self._marker_templates[marker_id],
                    allow_soft_grid=self.allow_soft_marker_corners,
                    wrist_marker=marker_id in self._wrist_marker_ids,
                )
                self.last_boundary_quality[marker_id] = quality
                if not quality.accepted:
                    self.last_rejected_detections[marker_id] = points.copy()
            for marker_id in self.last_rejected_detections:
                del detections[marker_id]
                self._tracks.pop(marker_id, None)
            self.last_recovered_ids = tuple(
                marker_id for marker_id in self.last_recovered_ids if marker_id in detections
            )
            self.last_tracked_ids = tuple(
                marker_id for marker_id in self.last_tracked_ids if marker_id in detections
            )
        return detections

    def _boundary_image(
        self, gray: np.ndarray, corners: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._camera_matrix is None or self._dist_coeffs is None or not np.any(self._dist_coeffs):
            return gray, corners
        if self._quality_undistort_maps is None or self._quality_undistort_maps[0].shape != gray.shape:
            self._quality_undistort_maps = cv2.initUndistortRectifyMap(
                self._camera_matrix, self._dist_coeffs, None, self._camera_matrix,
                (gray.shape[1], gray.shape[0]), cv2.CV_32FC1,
            )
        undistorted = cv2.undistortPoints(
            np.asarray(corners, dtype=np.float64).reshape(4, 1, 2),
            self._camera_matrix, self._dist_coeffs, P=self._camera_matrix,
        ).reshape(4, 2)
        margin = max(8.0, 0.3 * np.linalg.norm(np.ptp(undistorted, axis=0)))
        low = np.maximum(np.floor(undistorted.min(axis=0) - margin).astype(int), 0)
        high = np.minimum(
            np.ceil(undistorted.max(axis=0) + margin).astype(int),
            [gray.shape[1], gray.shape[0]],
        )
        if np.any(high <= low):
            return gray, corners
        maps = self._quality_undistort_maps
        # Undistort only the small marker ROI, not the whole video. Lens
        # distortion must not be mistaken for an occluded/non-planar tag grid.
        roi = cv2.remap(
            gray, maps[0][low[1]:high[1], low[0]:high[0]],
            maps[1][low[1]:high[1], low[0]:high[0]], cv2.INTER_LINEAR,
        )
        return roi, undistorted - low

    def _track_short_gaps(
        self, gray: np.ndarray, detections: dict[int, np.ndarray]
    ) -> dict[int, np.ndarray]:
        tracked: dict[int, tuple[np.ndarray, int, float]] = {}
        candidates = [
            (marker_id, corners, age, area)
            for marker_id, (corners, age, area) in self._tracks.items()
            if marker_id not in detections and age < self._track_marker_gaps
        ]
        if (
            candidates
            and self._previous_gray is not None
            and self._previous_gray.shape == gray.shape
        ):
            previous = np.concatenate(
                [corners.astype(np.float32).reshape(4, 1, 2) for _, corners, _, _ in candidates]
            )
            current, status, _ = cv2.calcOpticalFlowPyrLK(
                self._previous_gray, gray, previous, None, winSize=(21, 21), maxLevel=3
            )
            backward = backward_status = None
            if current is not None:
                backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
                    gray, self._previous_gray, current, None, winSize=(21, 21), maxLevel=3
                )
            if (
                current is not None
                and status is not None
                and backward is not None
                and backward_status is not None
            ):
                for index, (marker_id, _, age, previous_area) in enumerate(candidates):
                    section = slice(4 * index, 4 * index + 4)
                    corners = current[section].reshape(4, 2)
                    backward_error = np.linalg.norm(
                        backward[section].reshape(4, 2) - previous[section].reshape(4, 2),
                        axis=1,
                    )
                    area = abs(cv2.contourArea(corners))
                    polygon = np.rint(corners).astype(np.int32).reshape(-1, 1, 2)
                    valid = (
                        np.all(status[section])
                        and np.all(backward_status[section])
                        and np.max(backward_error) <= 0.5
                        and cv2.isContourConvex(polygon)
                        and area >= 100.0
                        and 0.6 * previous_area <= area <= 1.67 * previous_area
                        and np.all(corners[:, 0] >= 0)
                        and np.all(corners[:, 0] < gray.shape[1])
                        and np.all(corners[:, 1] >= 0)
                        and np.all(corners[:, 1] < gray.shape[0])
                    )
                    if valid:
                        detections[marker_id] = corners.astype(np.float64)
                        tracked[marker_id] = (corners, age + 1, area)
        fresh = {
            marker_id: (
                corners.astype(np.float32),
                0,
                abs(cv2.contourArea(corners.astype(np.float32))),
            )
            for marker_id, corners in detections.items()
            if marker_id in self._board_ids and marker_id not in tracked
        }
        fresh.update(tracked)
        self._tracks = fresh
        self._previous_gray = gray
        self.last_tracked_ids = tuple(sorted(tracked))
        return detections
