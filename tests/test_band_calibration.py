#!/usr/bin/env python3
import unittest

import cv2
import numpy as np

from aruco_track.models import BandLayout, Calibration
from aruco_track.pose import square_object_points
from calibrate_band_layout import (
    bundle_adjust_pair,
    calibrate_layout,
    marker_size,
    marker_transform,
)


class BandCalibrationTests(unittest.TestCase):
    def test_marker_transform_round_trip(self):
        points = square_object_points(0.04)
        rotation = cv2.Rodrigues(np.array([[0.2], [-0.1], [0.3]]))[0]
        translation = np.array([0.1, -0.2, 0.5])
        band_points = (rotation @ points.T).T + translation

        transform = marker_transform(band_points)
        reconstructed = (
            transform[:3, :3] @ square_object_points(marker_size(band_points)).T
        ).T + transform[:3, 3]

        np.testing.assert_allclose(reconstructed, band_points)

    def test_pair_calibration_moves_connected_marker_only(self):
        first = square_object_points(0.04)
        second = first + np.array([0.05, 0.0, 0.0])
        third = first + np.array([0.10, 0.0, 0.0])
        layout = BandLayout("band", "DICT_4X4_50", {0: first, 1: second, 2: third})
        first_from_second = np.eye(4)
        first_from_second[0, 3] = 0.06

        calibrated, anchor, connected, _ = calibrate_layout(
            layout, {(0, 1): (first_from_second, 30)}
        )

        self.assertEqual(anchor, 0)
        self.assertEqual(connected, {0, 1})
        np.testing.assert_allclose(
            np.mean(calibrated.markers[1], axis=0), [0.06, 0.0, 0.0]
        )
        np.testing.assert_allclose(calibrated.markers[2], third)

    def test_bundle_adjust_pair_improves_relative_transform(self):
        camera_matrix = np.array(
            [[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]]
        )
        calibration = Calibration(camera_matrix, np.zeros(5), (1280, 720))
        marker_points = square_object_points(0.04)
        expected = np.eye(4)
        expected[:3, :3] = cv2.Rodrigues(np.array([0.0, 0.7, 0.0]))[0]
        expected[:3, 3] = [0.045, 0.0, 0.012]
        second_points = (
            expected[:3, :3] @ marker_points.T
        ).T + expected[:3, 3]
        observations = []
        for index in range(8):
            rvec = np.array([0.08 + 0.02 * index, -0.15, 0.03])
            tvec = np.array([0.01, -0.02 + 0.004 * index, 0.42 + 0.01 * index])
            first, _ = cv2.projectPoints(
                marker_points, rvec, tvec, camera_matrix, calibration.dist_coeffs
            )
            second, _ = cv2.projectPoints(
                second_points, rvec, tvec, camera_matrix, calibration.dist_coeffs
            )
            observations.append(
                (first.reshape(4, 2), second.reshape(4, 2), calibration)
            )
        initial = expected.copy()
        initial[:3, 3] += [0.004, -0.002, 0.003]

        result = bundle_adjust_pair(
            observations, 0.04, 0.04, initial, max_frames=8
        )

        self.assertIsNotNone(result)
        optimized, rms, used_frames = result
        self.assertEqual(used_frames, 8)
        self.assertLess(rms, 0.1)
        self.assertLess(
            np.linalg.norm(optimized[:3, 3] - expected[:3, 3]),
            np.linalg.norm(initial[:3, 3] - expected[:3, 3]),
        )


if __name__ == "__main__":
    unittest.main()
