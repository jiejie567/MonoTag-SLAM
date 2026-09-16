#!/usr/bin/env python3
import unittest
from unittest.mock import patch

import numpy as np
import cv2

from aruco_track.models import BandLayout, Calibration, Pose
from aruco_track.pipeline import (
    TrackingPipeline,
    compose_pose,
    inverse_pose,
    relative_pose,
)


class TrackingPipelineTests(unittest.TestCase):
    def test_fixed_and_wrist_markers_use_separate_boundary_profiles(self):
        calibration = Calibration(np.eye(3), np.zeros(5), (16, 16))
        hand = BandLayout("hand", "DICT_4X4_50", {0: np.zeros((4, 3))})
        world = BandLayout("world", "DICT_4X4_50", {20: np.zeros((4, 3))})
        pipeline = TrackingPipeline(calibration, [hand], world_board=world)
        self.assertTrue(pipeline.detector.validate_corners)
        self.assertEqual(pipeline.detector._boundary_marker_ids, {20})
        self.assertEqual(pipeline.detector._wrist_marker_ids, {0})
        wrist_only = TrackingPipeline(calibration, [hand])
        self.assertTrue(wrist_only.detector.validate_corners)
        self.assertEqual(wrist_only.detector._boundary_marker_ids, set())
        self.assertEqual(wrist_only.detector._wrist_marker_ids, {0})

    def test_missing_wrist_discards_prior_and_resets_smoothing(self):
        calibration = Calibration(np.eye(3), np.zeros(5), (16, 16))
        band = BandLayout("band", "DICT_4X4_50", {0: np.zeros((4, 3))})
        pipeline = TrackingPipeline(calibration, [band], adaptive_smoothing=False)
        pipeline.detector.detect = lambda frame: {}
        first = Pose(np.zeros((3, 1)), np.array([[0.0], [0.0], [1.0]]), 0.1)
        recovered = Pose(np.zeros((3, 1)), np.array([[1.0], [0.0], [1.0]]), 0.1)
        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        with patch("aruco_track.pipeline.solve_band_pose", side_effect=[first, None, recovered]) as solve:
            pipeline.process(frame)
            missing = pipeline.process(frame)
            self.assertNotIn("band", pipeline._raw_poses)
            self.assertNotIn("band", pipeline._smoothers)
            result = pipeline.process(frame)
        self.assertEqual(missing.poses, {})
        self.assertIsNone(solve.call_args.args[4])
        np.testing.assert_allclose(result.poses['band'].tvec, recovered.tvec)

    def test_band_output_uses_smoothed_translation_and_keeps_raw_prior(self):
        calibration = Calibration(np.eye(3), np.zeros(5), (16, 16))
        band = BandLayout("band", "DICT_4X4_50", {0: np.zeros((4, 3))})
        pipeline = TrackingPipeline(calibration, [band], adaptive_smoothing=False)
        pipeline.detector.detect = lambda frame: {}
        first = Pose(np.zeros((3, 1)), np.array([[0.0], [0.0], [1.0]]), 0.1)
        second = Pose(np.zeros((3, 1)), np.array([[1.0], [0.0], [1.0]]), 0.2)
        frame = np.zeros((16, 16, 3), dtype=np.uint8)

        with patch("aruco_track.pipeline.solve_band_pose", side_effect=[first, second]):
            pipeline.process(frame)
            result = pipeline.process(frame)

        np.testing.assert_allclose(
            result.poses["band"].tvec,
            np.array([[0.18], [0.0], [1.0]]),
        )
        self.assertIs(pipeline._raw_poses["band"], second)

    def test_relative_pose_converts_camera_poses_to_world_frame(self):
        camera_from_world = cv2.Rodrigues(np.array([[0.0], [0.0], [np.pi / 2.0]]))[0]
        world_from_hand = cv2.Rodrigues(np.array([[0.2], [-0.1], [0.05]]))[0]
        camera_world_translation = np.array([[1.0], [2.0], [3.0]])
        world_hand_translation = np.array([[0.4], [-0.2], [0.6]])
        reference = Pose(
            cv2.Rodrigues(camera_from_world)[0], camera_world_translation, 0.2
        )
        target = Pose(
            cv2.Rodrigues(camera_from_world @ world_from_hand)[0],
            camera_from_world @ world_hand_translation + camera_world_translation,
            0.3,
        )

        converted = relative_pose(reference, target)
        recomposed = compose_pose(reference, converted)
        camera_in_world = inverse_pose(reference)

        np.testing.assert_allclose(converted.rotation_matrix, world_from_hand, atol=1e-12)
        np.testing.assert_allclose(converted.tvec, world_hand_translation, atol=1e-12)
        np.testing.assert_allclose(recomposed.rotation_matrix, target.rotation_matrix, atol=1e-12)
        np.testing.assert_allclose(recomposed.tvec, target.tvec, atol=1e-12)
        np.testing.assert_allclose(
            camera_in_world.rotation_matrix, camera_from_world.T, atol=1e-12
        )
        np.testing.assert_allclose(
            camera_in_world.tvec,
            -camera_from_world.T @ camera_world_translation,
            atol=1e-12,
        )

    def test_pipeline_outputs_world_pose_only_with_current_reference(self):
        calibration = Calibration(np.eye(3), np.zeros(5), (16, 16))
        hand = BandLayout("hand", "DICT_4X4_50", {0: np.zeros((4, 3))})
        world = BandLayout("world", "DICT_4X4_50", {20: np.zeros((4, 3))})
        pipeline = TrackingPipeline(calibration, [hand], world_board=world)
        pipeline.detector.detect = lambda frame: {}
        reference = Pose(np.zeros((3, 1)), np.array([[1.0], [0.0], [0.0]]), 0.1)
        target = Pose(np.zeros((3, 1)), np.array([[1.5], [0.2], [0.3]]), 0.2)
        frame = np.zeros((16, 16, 3), dtype=np.uint8)

        with patch("aruco_track.pipeline.solve_band_pose", side_effect=[reference, target]):
            valid = pipeline.process(frame)
        with patch("aruco_track.pipeline.solve_band_pose", side_effect=[None, target]):
            invalid = pipeline.process(frame)

        np.testing.assert_allclose(
            valid.world_poses["hand"].tvec, np.array([[0.5], [0.2], [0.3]])
        )
        self.assertIs(valid.world_reference, reference)
        self.assertIsNotNone(valid.camera_world_pose)
        self.assertEqual(invalid.world_poses, {})
        self.assertIsNone(invalid.world_reference)
        self.assertIn("world/hand", pipeline._smoothers)

    def test_world_and_hand_marker_ids_must_not_overlap(self):
        calibration = Calibration(np.eye(3), np.zeros(5), (16, 16))
        hand = BandLayout("hand", "DICT_4X4_50", {0: np.zeros((4, 3))})
        world = BandLayout("world", "DICT_4X4_50", {0: np.zeros((4, 3))})

        with self.assertRaises(ValueError):
            TrackingPipeline(calibration, [hand], world_board=world)


if __name__ == "__main__":
    unittest.main()
