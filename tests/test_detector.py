#!/usr/bin/env python3
import unittest
from unittest.mock import patch

import numpy as np

from aruco_track.detector import ArucoDetector


class FakeCvDetector:
    def __init__(self, detected_corner, rejected_corner):
        self.detected_corner = detected_corner
        self.rejected_corner = rejected_corner
        self.refine_calls = 0

    def detectMarkers(self, gray):
        return [self.detected_corner], np.array([[0]], dtype=np.int32), [self.rejected_corner]

    def refineDetectedMarkers(
        self, gray, board, corners, ids, rejected, camera_matrix, dist_coeffs
    ):
        self.refine_calls += 1
        return (
            [self.detected_corner, self.rejected_corner],
            np.array([[0], [1]], dtype=np.int32),
            [],
            np.array([[0]], dtype=np.int32),
        )


class SequencedCvDetector:
    def __init__(self, corner):
        self.corner = corner
        self.calls = 0

    def detectMarkers(self, gray):
        self.calls += 1
        if self.calls == 1:
            return [self.corner], np.array([[0]], dtype=np.int32), []
        return [], None, []

    def refineDetectedMarkers(
        self, gray, board, corners, ids, rejected, camera_matrix, dist_coeffs
    ):
        return corners, ids, rejected, None


class ArucoDetectorTests(unittest.TestCase):
    def test_board_refinement_recovers_rejected_candidate(self):
        object_points = {
            0: np.array([[0, 0, 0], [0.01, 0, 0], [0.01, 0.01, 0], [0, 0.01, 0]]),
            1: np.array([[0.02, 0, 0], [0.03, 0, 0], [0.03, 0.01, 0], [0.02, 0.01, 0]]),
        }
        detected_corner = np.array([[[1, 1], [5, 1], [5, 5], [1, 5]]], dtype=np.float32)
        rejected_corner = np.array([[[7, 1], [11, 1], [11, 5], [7, 5]]], dtype=np.float32)
        detector = ArucoDetector(
            board_markers=[object_points],
            camera_matrix=np.eye(3),
            dist_coeffs=np.zeros(5),
            validate_corners=False,
        )
        fake = FakeCvDetector(detected_corner, rejected_corner)
        detector._detector = fake

        detections = detector.detect(np.zeros((16, 16), dtype=np.uint8))

        self.assertEqual(set(detections), {0, 1})
        self.assertEqual(detector.last_recovered_ids, (1,))
        self.assertEqual(fake.refine_calls, 1)

    def test_board_refinement_requires_an_anchor_marker(self):
        object_points = {
            1: np.array([[0, 0, 0], [0.01, 0, 0], [0.01, 0.01, 0], [0, 0.01, 0]])
        }
        corner = np.array([[[1, 1], [5, 1], [5, 5], [1, 5]]], dtype=np.float32)
        detector = ArucoDetector(board_markers=[object_points], validate_corners=False)
        fake = FakeCvDetector(corner, corner)
        detector._detector = fake

        detections = detector.detect(np.zeros((16, 16), dtype=np.uint8))

        self.assertEqual(set(detections), {0})
        self.assertEqual(detector.last_recovered_ids, ())
        self.assertEqual(fake.refine_calls, 0)

    def test_optical_flow_tracks_only_two_missing_frames(self):
        object_points = {
            0: np.array([[0, 0, 0], [0.01, 0, 0], [0.01, 0.01, 0], [0, 0.01, 0]])
        }
        corner = np.array([[[4, 4], [18, 4], [18, 18], [4, 18]]], dtype=np.float32)
        detector = ArucoDetector(
            board_markers=[object_points], track_marker_gaps=2, validate_corners=False
        )
        detector._detector = SequencedCvDetector(corner)
        flow_calls = 0

        def translate_forward_then_backward(previous, current, points, *args, **kwargs):
            nonlocal flow_calls
            flow_calls += 1
            offset = 1.0 if flow_calls % 2 else -1.0
            return points + np.array([[[offset, 0.0]]], dtype=np.float32), np.ones(
                (len(points), 1), dtype=np.uint8
            ), None

        frame = np.zeros((32, 32), dtype=np.uint8)
        with patch("aruco_track.detector.cv2.calcOpticalFlowPyrLK", translate_forward_then_backward):
            detected = detector.detect(frame)
            tracked_once = detector.detect(frame)
            tracked_twice = detector.detect(frame)
            expired = detector.detect(frame)

        self.assertEqual(set(detected), {0})
        self.assertEqual(set(tracked_once), {0})
        self.assertEqual(set(tracked_twice), {0})
        self.assertEqual(expired, {})
        self.assertEqual(detector.last_tracked_ids, ())


if __name__ == "__main__":
    unittest.main()
