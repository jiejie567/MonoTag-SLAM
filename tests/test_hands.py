#!/usr/bin/env python3
import unittest

import cv2
import numpy as np

from aruco_track.hands import (
    RawHandJoints,
    TemporalHandAssignmentGate,
    assign_hands_to_bands,
    band_side,
    bend_angles,
    bind_landmarks_to_wrist,
    wrist_anchor_error_px,
)
from aruco_track.models import Calibration, Pose


class HandJointTests(unittest.TestCase):
    def test_bend_angle_is_zero_when_straight_and_pi_over_two_when_bent(self):
        points = np.zeros((21, 3), dtype=np.float64)
        points[5] = (0.0, 0.0, 0.0)
        points[6] = (1.0, 0.0, 0.0)
        points[7] = (2.0, 0.0, 0.0)
        points[8] = (2.0, 1.0, 0.0)

        angles = bend_angles(points)

        self.assertAlmostEqual(angles["index_pip"], 0.0)
        self.assertAlmostEqual(angles["index_dip"], np.pi / 2.0)

    def test_landmarks_are_anchored_to_wrist_and_rotated_into_band(self):
        points = np.zeros((21, 3), dtype=np.float64)
        points[:] = (0.1, 0.2, 0.3)
        points[1] += (0.01, 0.0, 0.0)
        rotation = cv2.Rodrigues(np.array([[0.0], [0.0], [np.pi / 2.0]]))[0]
        band_pose = Pose(
            cv2.Rodrigues(rotation)[0],
            np.array([[1.0], [2.0], [3.0]]),
            0.1,
        )
        world_reference = Pose(
            np.zeros((3, 1)), np.array([[0.5], [1.0], [1.0]]), 0.1
        )

        camera, band, world = bind_landmarks_to_wrist(
            points, band_pose, world_reference
        )

        np.testing.assert_allclose(camera[0], (1.0, 2.0, 3.0))
        np.testing.assert_allclose(camera[1], (1.01, 2.0, 3.0))
        np.testing.assert_allclose(band[1], (0.0, -0.01, 0.0), atol=1e-12)
        np.testing.assert_allclose(world[0], (0.5, 1.0, 2.0))

    def test_handedness_does_not_bind_joints_when_band_pose_is_missing(self):
        calibration = Calibration(np.eye(3), np.zeros(5), (100, 100))
        points = np.zeros((21, 3), dtype=np.float64)
        left = RawHandJoints("Left", 0.9, points.copy(), points.copy())
        right = RawHandJoints("Right", 0.8, points.copy(), points.copy())

        assigned = assign_hands_to_bands(
            [right, left], ["strap_band_L", "strap_band_R"], {}, calibration
        )

        self.assertNotIn("strap_band_L", assigned)
        self.assertNotIn("strap_band_R", assigned)
        self.assertCountEqual(
            [id(hand) for hand in assigned.values()], [id(left), id(right)]
        )

    def test_visible_band_proximity_overrides_wrong_handedness(self):
        calibration = Calibration(
            np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]),
            np.zeros(5),
            (100, 100),
        )
        points = np.zeros((21, 3), dtype=np.float64)
        near_left = points.copy()
        near_left[0, :2] = (0.3, 0.5)
        near_right = points.copy()
        near_right[0, :2] = (0.7, 0.5)
        mislabeled_left = RawHandJoints("Right", 0.9, near_left, points.copy())
        mislabeled_right = RawHandJoints("Left", 0.9, near_right, points.copy())
        poses = {
            "strap_band_L": Pose(
                np.zeros((3, 1)), np.array([[-0.2], [0.0], [1.0]]), 0.1
            ),
            "strap_band_R": Pose(
                np.zeros((3, 1)), np.array([[0.2], [0.0], [1.0]]), 0.1
            ),
        }

        assigned = assign_hands_to_bands(
            [mislabeled_left, mislabeled_right],
            ["strap_band_L", "strap_band_R"],
            poses,
            calibration,
        )

        self.assertIs(assigned["strap_band_L"], mislabeled_left)
        self.assertIs(assigned["strap_band_R"], mislabeled_right)

    def test_visible_band_rejects_hand_perpendicular_to_band_axis(self):
        calibration = Calibration(
            np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]),
            np.zeros(5),
            (100, 100),
        )
        points = np.zeros((21, 3), dtype=np.float64)
        image = points.copy()
        image[0, :2] = (0.5, 0.5)
        image[[5, 9, 13, 17], :2] = (0.8, 0.5)
        false_hand = RawHandJoints("Left", 0.99, image, points)
        pose = Pose(
            np.zeros((3, 1)), np.array([[0.0], [0.0], [1.0]]), 0.1
        )

        assigned = assign_hands_to_bands(
            [false_hand], ["strap_band_L"], {"strap_band_L": pose}, calibration
        )

        self.assertNotIn("strap_band_L", assigned)
        self.assertIs(assigned["unassigned_0"], false_hand)

    def test_visible_band_accepts_either_axial_cuff_orientation(self):
        calibration = Calibration(
            np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]),
            np.zeros(5),
            (100, 100),
        )
        points = np.zeros((21, 3), dtype=np.float64)
        image = points.copy()
        image[0, :2] = (0.5, 0.5)
        image[[5, 9, 13, 17], :2] = (0.5, 0.8)
        hand = RawHandJoints("Left", 0.99, image, points)
        pose = Pose(
            np.zeros((3, 1)), np.array([[0.0], [0.0], [1.0]]), 0.1
        )

        assigned = assign_hands_to_bands(
            [hand], ["strap_band_L"], {"strap_band_L": pose}, calibration
        )

        self.assertIs(assigned["strap_band_L"], hand)

    def test_temporal_gate_rejects_isolated_match_and_confirms_next_frame(self):
        gate = TemporalHandAssignmentGate(["strap_band_L"])
        hand = object()

        first = gate.update({"strap_band_L": hand})
        missing = gate.update({})
        restart = gate.update({"strap_band_L": hand})
        confirmed = gate.update({"strap_band_L": hand})

        self.assertNotIn("strap_band_L", first)
        self.assertEqual(gate.min_consecutive_frames, 2)
        self.assertEqual(missing, {})
        self.assertNotIn("strap_band_L", restart)
        self.assertIs(confirmed["strap_band_L"], hand)

    def test_band_side_matches_current_layout_names(self):
        self.assertEqual(band_side("strap_band_L"), "Left")
        self.assertEqual(band_side("strap_band_R"), "Right")
        self.assertIsNone(band_side("wrist"))

    def test_wrist_anchor_error_compares_landmark_with_band_projection(self):
        calibration = Calibration(
            np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]),
            np.zeros(5),
            (100, 100),
        )
        pose = Pose(
            np.zeros((3, 1)), np.array([[0.1], [0.0], [1.0]]), 0.1
        )
        points = np.zeros((21, 3), dtype=np.float64)
        points[0, :2] = (0.6, 0.5)

        error = wrist_anchor_error_px(points, pose, calibration)

        self.assertAlmostEqual(error, 0.0)
        np.testing.assert_allclose(points[0, :2], (0.6, 0.5))


if __name__ == "__main__":
    unittest.main()
