#!/usr/bin/env python3
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from aruco_track.models import Calibration, Pose
from aruco_track.tag_graph import (
    TagPoseResult,
    _temporal_acceleration_residuals,
    optimize_tag_pose,
    optimize_wrist_trajectory,
    projection_residuals,
    refine_wrist_pose_sequence,
)
from tools.make_band import band_layout


class TagGraphTests(unittest.TestCase):
    def setUp(self):
        self.layout = band_layout("left", 0, 69.0, 55.0, 56.0)
        self.calibration = Calibration(
            np.array(
                [[1050.0, 0.0, 960.0], [0.0, 1040.0, 540.0], [0.0, 0.0, 1.0]]
            ),
            np.zeros(5),
            (1920, 1080),
        )
        self.pose = Pose(
            np.array([[0.25], [-0.35], [0.08]]),
            np.array([[0.04], [-0.03], [0.72]]),
            0.0,
        )

    def detections(self, marker_ids):
        output = {}
        for marker_id in marker_ids:
            image, _ = cv2.projectPoints(
                self.layout.markers[marker_id],
                self.pose.rvec,
                self.pose.tvec,
                self.calibration.camera_matrix,
                self.calibration.dist_coeffs,
            )
            output[marker_id] = image.reshape(4, 2)
        return output

    def test_projection_factor_is_zero_at_ground_truth(self):
        detections = self.detections((0, 1))
        state = np.concatenate((self.pose.rvec.reshape(3), self.pose.tvec.reshape(3)))

        residual = projection_residuals(
            state, detections, self.layout, self.calibration, (0, 1)
        )

        np.testing.assert_allclose(residual, 0.0, atol=1e-9)

    def test_optimizes_all_visible_marker_corners(self):
        detections = self.detections((0, 1, 2))

        result = optimize_tag_pose(detections, self.layout, self.calibration)

        self.assertIsNotNone(result.pose)
        self.assertEqual(result.accepted_marker_ids, (0, 1, 2))
        self.assertEqual(result.rejected_marker_ids, ())
        np.testing.assert_allclose(result.pose.tvec, self.pose.tvec, atol=1e-5)
        self.assertLess(result.graph_reprojection_error_px, 1e-5)

    def test_rejects_a_marker_level_outlier(self):
        detections = self.detections((0, 1, 2))
        detections[2] += np.array([100.0, -80.0])

        result = optimize_tag_pose(detections, self.layout, self.calibration)

        self.assertIsNotNone(result.pose)
        self.assertIn(2, result.rejected_marker_ids)
        self.assertNotIn(2, result.accepted_marker_ids)
        np.testing.assert_allclose(result.pose.tvec, self.pose.tvec, atol=2e-3)

    def test_single_marker_remains_available_but_is_flagged_ambiguous(self):
        result = optimize_tag_pose(
            self.detections((1,)), self.layout, self.calibration, self.pose
        )

        self.assertIsNotNone(result.pose)
        self.assertTrue(result.pose.ambiguous)
        self.assertEqual(result.accepted_marker_ids, (1,))

    def test_metric_camera_motion_resolves_single_marker_after_short_gap(self):
        world_from_wrist = self.pose
        camera_poses = [
            Pose(np.zeros((3, 1)), np.zeros((3, 1)), 0.0),
            Pose(np.array([[0.0], [0.04], [0.0]]),
                 np.array([[0.01], [0.0], [0.0]]), 0.0),
            Pose(np.array([[0.0], [0.08], [0.0]]),
                 np.array([[0.02], [0.0], [0.0]]), 0.0),
        ]

        def camera_from_wrist(camera):
            rotation = camera.rotation_matrix.T @ world_from_wrist.rotation_matrix
            translation = camera.rotation_matrix.T @ (
                world_from_wrist.tvec - camera.tvec
            )
            return Pose(cv2.Rodrigues(rotation)[0], translation, 0.0)

        def project(camera, marker_ids):
            pose = camera_from_wrist(camera)
            return {
                marker_id: cv2.projectPoints(
                    self.layout.markers[marker_id],
                    pose.rvec,
                    pose.tvec,
                    self.calibration.camera_matrix,
                    self.calibration.dist_coeffs,
                )[0].reshape(4, 2)
                for marker_id in marker_ids
            }

        detections = [project(camera_poses[0], (0, 1)), {}, project(camera_poses[2], (1,))]
        seed = optimize_tag_pose(detections[0], self.layout, self.calibration)
        wrong = Pose(
            camera_from_wrist(camera_poses[2]).rvec + np.array([[1.8], [0.0], [0.0]]),
            camera_from_wrist(camera_poses[2]).tvec + np.array([[0.05], [0.0], [0.03]]),
            0.1,
            marker_ids=(1,),
            inlier_count=4,
            ambiguous=True,
        )
        initial = [
            seed,
            TagPoseResult(None, (), (), {}, None, 0.0),
            TagPoseResult(wrong, (1,), (), {1: 0.1}, 0.1, 0.2),
        ]

        refined = refine_wrist_pose_sequence(
            camera_poses,
            [True] * 3,
            ["atlas_0"] * 3,
            [0.0, 1.0 / 60.0, 2.0 / 60.0],
            detections,
            initial,
            self.layout,
            self.calibration,
        )

        self.assertIsNone(refined[1].pose)
        self.assertIsNotNone(refined[2].pose)
        resolved = refined[2].pose
        world_translation = (
            camera_poses[2].rotation_matrix @ resolved.tvec + camera_poses[2].tvec
        )
        world_rotation = camera_poses[2].rotation_matrix @ resolved.rotation_matrix
        np.testing.assert_allclose(world_translation, world_from_wrist.tvec, atol=2e-3)
        rotation_error = cv2.Rodrigues(
            world_from_wrist.rotation_matrix.T @ world_rotation
        )[0]
        self.assertLess(np.linalg.norm(rotation_error), np.deg2rad(2.0))

    def test_future_multi_marker_pose_repairs_short_single_marker_tail(self):
        camera = Pose(np.zeros((3, 1)), np.zeros((3, 1)), 0.0)
        single = self.detections((1,))
        multi = self.detections((0, 1))
        reliable = optimize_tag_pose(multi, self.layout, self.calibration)
        initial = [
            TagPoseResult(None, (), (1,), {}, None, 0.0),
            TagPoseResult(None, (), (), {}, None, 0.0),
            reliable,
        ]

        refined = refine_wrist_pose_sequence(
            [camera] * 3,
            [True] * 3,
            ["atlas_0"] * 3,
            [0.0, 1.0 / 60.0, 2.0 / 60.0],
            [single, {}, multi],
            initial,
            self.layout,
            self.calibration,
        )

        self.assertIsNotNone(refined[0].pose)
        np.testing.assert_allclose(refined[0].pose.tvec, self.pose.tvec, atol=2e-3)
        self.assertIsNone(refined[1].pose)

    def test_rejected_neighbour_votes_for_planar_branch_without_entering_ba(self):
        strong = self.detections((1,))
        unresolved = optimize_tag_pose(
            strong, self.layout, self.calibration,
            validate_planar_ambiguity=True,
        )
        self.assertIsNone(unresolved.pose)

        result = optimize_tag_pose(
            strong, self.layout, self.calibration,
            validate_planar_ambiguity=True,
            assist_detections=self.detections((2,)),
        )
        self.assertIsNotNone(result.pose)
        self.assertEqual(result.accepted_marker_ids, (1,))
        self.assertNotIn(2, result.accepted_marker_ids)
        self.assertNotIn(2, result.rejected_marker_ids)
        np.testing.assert_allclose(result.pose.tvec, self.pose.tvec, atol=1e-5)

        bad_assist = self.detections((2,))
        bad_assist[2] += 100.0
        rejected = optimize_tag_pose(
            strong, self.layout, self.calibration,
            validate_planar_ambiguity=True,
            assist_detections=bad_assist,
        )
        self.assertIsNone(rejected.pose)

    def test_soft_marker_has_less_influence_on_pose(self):
        detections = self.detections((0, 1, 2))
        detections[1] += np.array([2.0, -1.5])
        full = optimize_tag_pose(detections, self.layout, self.calibration)
        soft = optimize_tag_pose(detections, self.layout, self.calibration,
                                 marker_weights={1: 0.25})
        self.assertIsNotNone(full.pose)
        self.assertIsNotNone(soft.pose)
        self.assertIn(1, soft.accepted_marker_ids)
        self.assertLess(np.linalg.norm(soft.pose.tvec - self.pose.tvec),
                        np.linalg.norm(full.pose.tvec - self.pose.tvec))

    def test_adding_soft_tag_does_not_demote_reliable_anchor_confidence(self):
        base = optimize_tag_pose(self.detections((0, 2)), self.layout, self.calibration)
        extra = optimize_tag_pose(self.detections((0, 1, 2)), self.layout, self.calibration,
                                  marker_weights={1: 0.25})
        self.assertAlmostEqual(base.confidence, extra.confidence, places=6)

    def test_soft_marker_alone_cannot_initialize_but_can_follow_prior(self):
        detections = self.detections((1,))
        initial = optimize_tag_pose(detections, self.layout, self.calibration,
                                    marker_weights={1: 0.25})
        self.assertIsNone(initial.pose)
        tracked = optimize_tag_pose(detections, self.layout, self.calibration,
                                    self.pose, marker_weights={1: 0.25})
        self.assertIsNotNone(tracked.pose)
        self.assertLess(tracked.confidence, 0.35)

    def test_zero_weight_marker_cannot_influence_solution(self):
        detections = self.detections((0, 1, 2))
        detections[1] += 100.0
        result = optimize_tag_pose(detections, self.layout, self.calibration,
                                   marker_weights={1: 0.0})
        self.assertIsNotNone(result.pose)
        self.assertNotIn(1, result.accepted_marker_ids)
        np.testing.assert_allclose(result.pose.tvec, self.pose.tvec, atol=1e-5)

    def test_multi_frame_wrist_graph_suppresses_world_pose_jitter(self):
        rng = np.random.default_rng(8)
        frame_count = 12
        camera_poses = []
        wrist_camera_poses = []
        detections = []
        accepted_ids = []
        initial_world_translations = []
        true_world_wrist = np.array([0.0, 0.0, 1.0])
        for frame in range(frame_count):
            camera_translation = np.array([0.004 * frame, 0.0, 0.0])
            camera_pose = Pose(
                np.zeros((3, 1)), camera_translation.reshape(3, 1), 0.0
            )
            camera_poses.append(camera_pose)
            jitter = rng.normal(0.0, 0.004, size=3)
            initial_world = true_world_wrist + jitter
            initial_world_translations.append(initial_world)
            wrist_camera_poses.append(
                Pose(
                    rng.normal(0.0, 0.01, size=(3, 1)),
                    (initial_world - camera_translation).reshape(3, 1),
                    1.0,
                    marker_ids=(0, 1, 2),
                    inlier_count=12,
                )
            )
            frame_detections = {}
            true_camera_translation = true_world_wrist - camera_translation
            for marker_id in (0, 1, 2):
                image, _ = cv2.projectPoints(
                    self.layout.markers[marker_id],
                    np.zeros((3, 1)),
                    true_camera_translation.reshape(3, 1),
                    self.calibration.camera_matrix,
                    self.calibration.dist_coeffs,
                )
                frame_detections[marker_id] = image.reshape(4, 2) + rng.normal(
                    0.0, 0.35, size=(4, 2)
                )
            detections.append(frame_detections)
            accepted_ids.append((0, 1, 2))

        result = optimize_wrist_trajectory(
            camera_poses,
            wrist_camera_poses,
            detections,
            accepted_ids,
            self.layout,
            self.calibration,
        )

        optimized = np.array([pose.tvec.reshape(3) for pose in result.poses])
        initial_spread = np.linalg.norm(np.std(initial_world_translations, axis=0))
        optimized_spread = np.linalg.norm(np.std(optimized, axis=0))
        self.assertLess(optimized_spread, 0.5 * initial_spread)
        np.testing.assert_allclose(np.mean(optimized, axis=0), true_world_wrist, atol=2e-3)

    def test_assist_only_face_reduces_single_face_depth_bias_without_becoming_accepted(self):
        frame_count = 12
        camera = Pose(np.zeros((3, 1)), np.zeros((3, 1)), 0.0)
        strong = self.detections((1,))[1].copy()
        strong += np.array([[2.0, -1.0], [-1.0, 1.5], [1.0, 2.0], [-2.0, -1.5]])
        weak = self.detections((2,))[2]
        common = (
            [camera] * frame_count,
            [self.pose] * frame_count,
            [{1: strong}] * frame_count,
            [(1,)] * frame_count,
            self.layout,
            self.calibration,
        )
        baseline = optimize_wrist_trajectory(*common, parallel_workers=1)
        assisted = optimize_wrist_trajectory(
            *common,
            parallel_workers=1,
            assist_detections=[{2: weak}] * frame_count,
        )
        target = self.pose.tvec.reshape(3)
        baseline_error = np.mean([
            np.linalg.norm(pose.tvec.reshape(3) - target)
            for pose in baseline.poses
        ])
        assisted_error = np.mean([
            np.linalg.norm(pose.tvec.reshape(3) - target)
            for pose in assisted.poses
        ])
        self.assertLess(assisted_error, 0.8 * baseline_error)
        self.assertTrue(all(pose.marker_ids == (1,) for pose in assisted.poses))

    def test_tiny_conflicting_second_face_only_resolves_planar_ambiguity(self):
        camera = Pose(np.zeros((3, 1)), np.zeros((3, 1)), 0.0)
        first_detections = self.detections((0,))
        second_detections = self.detections((0, 1))
        second_detections[1] += np.array([8.0, 0.0])
        first = optimize_tag_pose(
            first_detections, self.layout, self.calibration, self.pose,
            validate_planar_ambiguity=True,
        )
        second = optimize_tag_pose(
            second_detections, self.layout, self.calibration, self.pose,
            validate_planar_ambiguity=True,
        )
        self.assertEqual(second.accepted_marker_ids, (0, 1))

        refined = refine_wrist_pose_sequence(
            [camera, camera],
            [True, True],
            ["map", "map"],
            [0.0, 0.01],
            [first_detections, second_detections],
            [first, second],
            self.layout,
            self.calibration,
        )

        self.assertEqual(refined[1].accepted_marker_ids, (0,))
        self.assertEqual(refined[1].rejected_marker_ids, (1,))
        np.testing.assert_allclose(refined[1].pose.tvec, self.pose.tvec, atol=1e-5)

    def test_isolated_single_marker_keeps_identity_but_not_pose(self):
        camera = Pose(np.zeros((3, 1)), np.zeros((3, 1)), 0.)
        wrist = Pose(
            np.zeros((3, 1)), np.array([[0.], [0.], [1.]]), 0., (0,)
        )
        result = optimize_wrist_trajectory(
            [camera, camera],
            [wrist, wrist],
            [self.detections((0,)), self.detections((0,))],
            [(0,), (0,)],
            self.layout,
            self.calibration,
            parallel_workers=1,
        )

        self.assertEqual(result.poses, [None, None])

    def test_short_marker_miss_does_not_restart_motion_segment(self):
        camera = Pose(np.zeros((3, 1)), np.zeros((3, 1)), 0.0)
        wrist = Pose(
            np.zeros((3, 1)), np.array([[0.0], [0.0], [1.0]]), 0.0, (0,)
        )
        result = optimize_wrist_trajectory(
            [camera] * 5,
            [wrist, wrist, None, wrist, wrist],
            [
                self.detections((0,)),
                self.detections((0,)),
                {},
                self.detections((0,)),
                self.detections((0,)),
            ],
            [(0,), (0,), (), (0,), (0,)],
            self.layout,
            self.calibration,
            fps=60.0,
            parallel_workers=1,
        )

        self.assertIsNone(result.poses[2])
        self.assertTrue(all(
            result.poses[index] is not None for index in (0, 1, 3, 4)
        ))

    def test_short_multi_marker_segment_remains_publishable(self):
        camera = Pose(np.zeros((3, 1)), np.zeros((3, 1)), 0.)
        wrist = Pose(
            np.zeros((3, 1)), np.array([[0.], [0.], [1.]]), 0., (0, 1)
        )
        result = optimize_wrist_trajectory(
            [camera, camera],
            [wrist, wrist],
            [self.detections((0, 1)), self.detections((0, 1))],
            [(0, 1), (0, 1)],
            self.layout,
            self.calibration,
            parallel_workers=1,
        )

        self.assertTrue(all(pose is not None for pose in result.poses))

    def test_temporal_regularizer_uses_physical_time_at_all_frame_rates(self):
        expected_translation = np.array([.8, -.4, .2]) / (60.0**2)
        expected_rotation = np.array([0., 0., .6]) / (60.0**2)
        for fps in (30., 60., 120.):
            times = np.arange(3, dtype=float) / fps
            translations = np.stack([
                .5 * np.array([.8, -.4, .2]) * time**2 for time in times
            ])
            rotations = [
                cv2.Rodrigues((.5 * np.array([0., 0., .6]) * time**2).reshape(3, 1))[0]
                for time in times
            ]
            translation, rotation = _temporal_acceleration_residuals(
                rotations, translations, times, reference_fps=60.)
            with self.subTest(fps=fps):
                np.testing.assert_allclose(translation[0], expected_translation, rtol=.04)
                np.testing.assert_allclose(rotation[0], expected_rotation, rtol=.04, atol=1e-10)

    def test_wrist_windows_use_seconds_and_blend_overlaps(self):
        fps = 120.
        count = 300
        timestamps = np.arange(count, dtype=float) / fps
        camera = Pose(np.zeros((3, 1)), np.zeros((3, 1)), 0.)
        wrist = Pose(np.zeros((3, 1)), np.array([[0.], [0.], [1.]]), 0., (0,))
        camera_poses = [camera] * count
        wrist_poses = [wrist] * count
        detections = [self.detections((0,)) for _ in range(count)]
        accepted = [(0,)] * count
        windows = []

        def fake_window(indices, initial, *args, **kwargs):
            windows.append(indices)
            offset = .01 * (len(windows) - 1)
            poses = [Pose(np.zeros((3, 1)), np.array([[offset], [0.], [1.]]), 0.)
                     for _ in indices]
            return poses, [0.] * len(indices)

        with patch('aruco_track.tag_graph._optimize_wrist_window', side_effect=fake_window):
            result = optimize_wrist_trajectory(
                camera_poses, wrist_poses, detections, accepted,
                self.layout, self.calibration, fps=fps, timestamps_s=timestamps,
                parallel_workers=1)

        self.assertGreater(len(windows), 1)
        self.assertTrue(all(timestamps[w[-1]] - timestamps[w[0]] < 1.5 for w in windows))
        self.assertGreater(len(windows[0]), 170)  # about 1.5 s, not the old fixed 90 frames
        values = np.array([pose.tvec[0, 0] for pose in result.poses])
        self.assertLess(np.max(np.abs(np.diff(values))), .003)

    def test_parallel_wrist_windows_match_serial_submission_order(self):
        count = 100
        camera = Pose(np.zeros((3, 1)), np.zeros((3, 1)), 0.)
        camera_poses = [camera] * count
        wrist_poses = [self.pose] * count
        detections = [self.detections((0, 1, 2)) for _ in range(count)]
        accepted = [(0, 1, 2)] * count
        arguments = (
            camera_poses, wrist_poses, detections, accepted,
            self.layout, self.calibration,
        )

        serial = optimize_wrist_trajectory(*arguments, parallel_workers=1)
        parallel = optimize_wrist_trajectory(*arguments, parallel_workers=4)

        for first, second in zip(serial.poses, parallel.poses):
            np.testing.assert_allclose(first.tvec, second.tvec, atol=1e-12)
            np.testing.assert_allclose(
                first.rotation_matrix, second.rotation_matrix, atol=1e-12
            )
        np.testing.assert_allclose(
            serial.reprojection_errors_px,
            parallel.reprojection_errors_px,
            atol=1e-12,
        )

    def test_sixty_fps_keeps_the_existing_ninety_frame_window_and_overlap(self):
        count = 250
        camera = Pose(np.zeros((3, 1)), np.zeros((3, 1)), 0.)
        wrists = [
            Pose(np.zeros((3, 1)), np.array([[index / 1000.], [0.], [1.]]), 0., (0,))
            for index in range(count)
        ]
        starts_and_lengths = []

        def fake_window(indices, initial, *args, **kwargs):
            starts_and_lengths.append((round(initial[0].tvec[0, 0] * 1000), len(indices)))
            return initial, [0.] * len(indices)

        with patch('aruco_track.tag_graph._optimize_wrist_window', side_effect=fake_window):
            optimize_wrist_trajectory(
                [camera] * count,
                wrists,
                [self.detections((0,)) for _ in range(count)],
                [(0,)] * count,
                self.layout,
                self.calibration,
                fps=60.,
                parallel_workers=1,
            )

        self.assertEqual(starts_and_lengths, [(0, 90), (80, 90), (160, 90)])


if __name__ == "__main__":
    unittest.main()
