#!/usr/bin/env python3
import unittest

import cv2
import numpy as np

from aruco_track.pose import solve_square_pose, square_object_points


class SquarePoseTests(unittest.TestCase):
    def setUp(self):
        self.camera = np.array([[900.0, 0, 640.0], [0, 900.0, 360.0], [0, 0, 1.0]])
        self.distortion = np.zeros(5)

    def test_recovers_metric_translation(self):
        expected_rvec = np.array([[0.12], [-0.2], [0.04]])
        expected_tvec = np.array([[0.03], [-0.02], [0.75]])
        image, _ = cv2.projectPoints(square_object_points(0.04), expected_rvec, expected_tvec, self.camera, self.distortion)
        pose = solve_square_pose(image.reshape(4, 2), 0.04, self.camera, self.distortion)
        self.assertIsNotNone(pose)
        np.testing.assert_allclose(pose.tvec, expected_tvec, atol=2e-5)
        self.assertLess(pose.reprojection_error_px, 1e-4)

    def test_wrong_size_changes_scale(self):
        rvec = np.array([[0.2], [0.1], [-0.05]])
        tvec = np.array([[0.0], [0.0], [0.8]])
        image, _ = cv2.projectPoints(square_object_points(0.04), rvec, tvec, self.camera, self.distortion)
        pose = solve_square_pose(image.reshape(4, 2), 0.08, self.camera, self.distortion)
        self.assertAlmostEqual(float(pose.tvec[2, 0]), 1.6, places=3)


if __name__ == "__main__":
    unittest.main()

