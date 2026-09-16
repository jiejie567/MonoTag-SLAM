import unittest

import cv2
import numpy as np

from aruco_track.marker_uncertainty import marker_pose_uncertainty
from aruco_track.models import BandLayout, Calibration, Pose
from aruco_track.pose import square_object_points


class MarkerUncertaintyTests(unittest.TestCase):
    def setUp(self):
        self.calibration = Calibration(
            np.array([[900., 0, 640], [0, 900., 360], [0, 0, 1]]),
            np.zeros(5), (1280, 720),
        )
        self.camera_from_world = Pose(
            np.array([[.03], [-.08], [.01]]),
            np.array([[.01], [-.02], [.65]]), 0.,
        )

    def project(self, points):
        return cv2.projectPoints(
            points, self.camera_from_world.rvec, self.camera_from_world.tvec,
            self.calibration.camera_matrix, self.calibration.dist_coeffs,
        )[0].reshape(4, 2)

    def test_more_spatially_distributed_corners_reduce_pose_uncertainty(self):
        square = square_object_points(.048)
        one = BandLayout("one", "DICT_4X4_50", {20: square})
        two = BandLayout("two", "DICT_4X4_50", {
            20: square - [.07, 0, 0], 21: square + [.07, 0, 0],
        })
        _, first = marker_pose_uncertainty(
            self.camera_from_world, one, {20: self.project(square)}, (20,),
            self.calibration,
        )
        _, second = marker_pose_uncertainty(
            self.camera_from_world, two,
            {20: self.project(two.markers[20]), 21: self.project(two.markers[21])},
            (20, 21), self.calibration,
        )
        self.assertTrue(first["valid"] and second["valid"])
        self.assertLess(second["position_std_m"], first["position_std_m"])
        self.assertLess(second["rotation_std_deg_rms"], first["rotation_std_deg_rms"])

    def test_residual_and_weight_are_auditable_and_raise_uncertainty(self):
        square = square_object_points(.048)
        layout = BandLayout("one", "DICT_4X4_50", {20: square})
        exact = self.project(square)
        _, baseline = marker_pose_uncertainty(
            self.camera_from_world, layout, {20: exact}, (20,), self.calibration,
        )
        shifted = exact.copy(); shifted[0] += [1.5, -1.0]
        corners, noisy = marker_pose_uncertainty(
            self.camera_from_world, layout, {20: shifted}, (20,), self.calibration,
            {20: .25},
        )
        self.assertEqual([(item["marker_id"], item["corner_index"]) for item in corners],
                         [(20, 0), (20, 1), (20, 2), (20, 3)])
        self.assertGreater(corners[0]["residual_px"], 1.)
        self.assertTrue(all(item["information_weight"] == .25 for item in corners))
        self.assertGreater(noisy["position_std_m"], baseline["position_std_m"])


if __name__ == "__main__":
    unittest.main()
