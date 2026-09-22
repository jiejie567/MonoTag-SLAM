#!/usr/bin/env python3
import unittest

import cv2
import numpy as np

from aruco_track.bandsolve import solve_band_pose
from aruco_track.models import Pose
from aruco_track.pipeline import marker_pose_from_band
from tools.make_band import band_layout


class BandPoseTests(unittest.TestCase):
    def setUp(self):
        self.layout = band_layout("left", 0, 69.0, 55.0, 56.0)
        self.camera = np.array([[1050.0, 0, 960.0], [0, 1040.0, 540.0], [0, 0, 1.0]])
        self.distortion = np.zeros(5)
        self.rvec = np.array([[0.25], [-0.35], [0.08]])
        self.tvec = np.array([[0.04], [-0.03], [0.72]])

    def detections(self, ids):
        result = {}
        for marker_id in ids:
            image, _ = cv2.projectPoints(self.layout.markers[marker_id], self.rvec, self.tvec, self.camera, self.distortion)
            result[marker_id] = image.reshape(4, 2)
        return result

    def test_multi_face_pose(self):
        pose = solve_band_pose(self.detections([0, 1, 2]), self.layout, self.camera, self.distortion)
        self.assertIsNotNone(pose)
        np.testing.assert_allclose(pose.tvec, self.tvec, atol=1e-5)
        self.assertFalse(pose.ambiguous)
        self.assertEqual(pose.marker_ids, (0, 1, 2))

    def test_rejects_marker_outlier(self):
        detections = self.detections([0, 1, 2])
        detections[2] += np.array([120.0, -80.0])
        pose = solve_band_pose(detections, self.layout, self.camera, self.distortion)
        self.assertIsNotNone(pose)
        np.testing.assert_allclose(pose.tvec, self.tvec, atol=2e-3)
        self.assertNotIn(2, pose.marker_ids)

    def test_one_face_is_flagged_ambiguous(self):
        pose = solve_band_pose(self.detections([1]), self.layout, self.camera, self.distortion)
        self.assertIsNotNone(pose)
        self.assertTrue(pose.ambiguous)

    def test_can_require_two_inlier_markers(self):
        pose = solve_band_pose(
            self.detections([1]),
            self.layout,
            self.camera,
            self.distortion,
            min_inlier_markers=2,
        )
        self.assertIsNone(pose)

    def test_previous_pose_keeps_single_face_solution_continuous(self):
        previous = Pose(self.rvec, self.tvec, 0.0)
        pose = solve_band_pose(
            self.detections([1]),
            self.layout,
            self.camera,
            self.distortion,
            previous=previous,
        )
        self.assertIsNotNone(pose)
        np.testing.assert_allclose(pose.tvec, self.tvec, atol=1e-6)
        relative = cv2.Rodrigues(pose.rvec)[0] @ cv2.Rodrigues(self.rvec)[0].T
        rotation_error = np.linalg.norm(cv2.Rodrigues(relative)[0])
        self.assertLess(rotation_error, 1e-5)

    def test_marker_frames_are_rigidly_derived_from_band_pose(self):
        band_pose = Pose(self.rvec, self.tvec, 0.5)
        marker_poses = [
            marker_pose_from_band(self.layout.markers[marker_id], band_pose)
            for marker_id in (0, 1, 2)
        ]
        camera_from_band = cv2.Rodrigues(self.rvec)[0]
        for marker_id, marker_pose in zip((0, 1, 2), marker_poses):
            center = np.mean(self.layout.markers[marker_id], axis=0).reshape(3, 1)
            expected = camera_from_band @ center + self.tvec
            np.testing.assert_allclose(marker_pose.tvec, expected, atol=1e-12)

    def test_marker_display_origin_can_follow_raw_pose(self):
        smoothed = Pose(np.zeros((3, 1)), np.zeros((3, 1)), 0.5)
        raw = Pose(self.rvec, self.tvec, 0.5)
        marker_pose = marker_pose_from_band(self.layout.markers[1], smoothed, raw)
        center = np.mean(self.layout.markers[1], axis=0).reshape(3, 1)
        expected = cv2.Rodrigues(self.rvec)[0] @ center + self.tvec
        np.testing.assert_allclose(marker_pose.tvec, expected, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
