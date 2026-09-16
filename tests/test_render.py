#!/usr/bin/env python3
import unittest
from unittest.mock import patch

import numpy as np

from aruco_track.models import Calibration, Pose
from aruco_track.pipeline import FrameResult
from aruco_track.render import (
    FadingTrajectory,
    LEFT_COLOR,
    RIGHT_COLOR,
    draw_frame_axes,
    draw_result,
    draw_rejected_marker_boundaries,
    draw_soft_marker,
    pose_color,
)


class FadingTrajectoryTests(unittest.TestCase):
    def test_rejection_label_distinguishes_grid_from_boundary(self):
        image = np.zeros((100, 100, 3), np.uint8)
        corners = np.array([[20, 20], [80, 20], [80, 80], [20, 80]])
        with patch("aruco_track.render.cv2.putText") as draw_text:
            draw_rejected_marker_boundaries(image, {21: corners}, {21: "grid_mismatch"})
        self.assertEqual(draw_text.call_args.args[1], "ID 21 REJECT: GRID MISMATCH")

    def test_soft_marker_label_does_not_claim_rejection(self):
        image = np.zeros((100, 100, 3), np.uint8)
        corners = np.array([[20, 20], [80, 20], [80, 80], [20, 80]])
        with patch("aruco_track.render.cv2.putText") as draw_text:
            draw_soft_marker(image, 21, corners, 0.25)
        self.assertEqual(draw_text.call_args.args[1], "ID 21 SOFT 25%")

    def test_hand_colors_are_semantic(self):
        self.assertEqual(pose_color("strap_band_L"), LEFT_COLOR)
        self.assertEqual(pose_color("marker-2"), LEFT_COLOR)
        self.assertEqual(pose_color("strap_band_R"), RIGHT_COLOR)
        self.assertEqual(pose_color("marker-8"), RIGHT_COLOR)

    def test_history_is_bounded_and_drawn(self):
        calibration = Calibration(
            np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]),
            np.zeros(5),
            (100, 100),
        )
        trajectory = FadingTrajectory(["band"], max_points=3)
        output = np.zeros((100, 100, 3), dtype=np.uint8)
        for x in (0.00, 0.05, 0.10, 0.15):
            pose = Pose(np.zeros((3, 1)), np.array([[x], [0.0], [1.0]]), 0.0)
            trajectory.draw(output, FrameResult({}, {"band": pose}), calibration)
        self.assertEqual(len(trajectory._points["band"]), 3)
        self.assertGreater(np.count_nonzero(output), 0)

    def test_missing_pose_holds_last_point(self):
        calibration = Calibration(
            np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]),
            np.zeros(5),
            (100, 100),
        )
        trajectory = FadingTrajectory(["band"], max_points=10)
        output = np.zeros((100, 100, 3), dtype=np.uint8)
        pose = Pose(np.zeros((3, 1)), np.array([[0.1], [0.0], [1.0]]), 0.0)
        trajectory.draw(output, FrameResult({}, {"band": pose}), calibration)
        last_valid = trajectory._points["band"][-1]
        trajectory.draw(output, FrameResult({}, {}), calibration)
        self.assertEqual(trajectory._points["band"][-1], last_valid)

    def test_trajectory_point_is_smoothed(self):
        calibration = Calibration(
            np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]),
            np.zeros(5),
            (100, 100),
        )
        trajectory = FadingTrajectory(["band"], smoothing_alpha=0.5)
        output = np.zeros((100, 100, 3), dtype=np.uint8)
        first = Pose(np.zeros((3, 1)), np.array([[0.0], [0.0], [1.0]]), 0.0)
        second = Pose(np.zeros((3, 1)), np.array([[0.2], [0.0], [1.0]]), 0.0)
        trajectory.draw(output, FrameResult({}, {"band": first}), calibration)
        trajectory.draw(output, FrameResult({}, {"band": second}), calibration)
        self.assertEqual(trajectory._points["band"][-1], (60, 50))

    def test_world_trajectory_rejects_an_impossible_single_frame_step(self):
        calibration = Calibration(
            np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]),
            np.zeros(5),
            (100, 100),
        )
        trajectory = FadingTrajectory(
            ["band"], world_smoothing_alpha=0.5, maximum_world_step_m=0.1
        )
        output = np.zeros((100, 100, 3), dtype=np.uint8)
        reference = Pose(
            np.zeros((3, 1)), np.array([[0.0], [0.0], [1.0]]), 0.0
        )
        first = Pose(np.zeros((3, 1)), np.zeros((3, 1)), 0.0)
        outlier = Pose(
            np.zeros((3, 1)), np.array([[0.5], [0.0], [0.0]]), 0.0
        )

        trajectory.draw(
            output,
            FrameResult({}, {}, world_poses={"band": first}, world_reference=reference),
            calibration,
        )
        trajectory.draw(
            output,
            FrameResult({}, {}, world_poses={"band": outlier}, world_reference=reference),
            calibration,
        )

        np.testing.assert_allclose(trajectory._world_points["band"][-1], [0.0, 0.0, 0.0])

    def test_camera_trajectory_endpoint_follows_raw_axis_origin(self):
        calibration = Calibration(
            np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]),
            np.zeros(5),
            (100, 100),
        )
        trajectory = FadingTrajectory(["band"], smoothing_alpha=0.3)
        output = np.zeros((100, 100, 3), dtype=np.uint8)
        first = Pose(np.zeros((3, 1)), np.array([[0.0], [0.0], [1.0]]), 0.0)
        filtered = Pose(np.zeros((3, 1)), np.array([[0.1], [0.0], [1.0]]), 0.0)
        raw = Pose(np.zeros((3, 1)), np.array([[0.2], [0.0], [1.0]]), 0.0)

        trajectory.draw(
            output,
            FrameResult({}, {"band": first}, raw_poses={"band": first}),
            calibration,
        )
        trajectory.draw(
            output,
            FrameResult({}, {"band": filtered}, raw_poses={"band": raw}),
            calibration,
        )

        self.assertEqual(trajectory._points["band"][-1], (53, 50))
        self.assertGreater(np.count_nonzero(output[49:52, 69:72]), 0)

    def test_world_trajectory_endpoint_follows_raw_axis_origin(self):
        calibration = Calibration(
            np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]),
            np.zeros(5),
            (100, 100),
        )
        trajectory = FadingTrajectory(["band"])
        output = np.zeros((100, 100, 3), dtype=np.uint8)
        reference = Pose(
            np.zeros((3, 1)), np.array([[0.0], [0.0], [1.0]]), 0.0
        )
        filtered_world = Pose(
            np.zeros((3, 1)), np.array([[0.1], [0.0], [0.0]]), 0.0
        )
        raw_camera = Pose(
            np.zeros((3, 1)), np.array([[0.2], [0.0], [1.0]]), 0.0
        )

        trajectory.draw(
            output,
            FrameResult(
                {},
                {"band": raw_camera},
                raw_poses={"band": raw_camera},
                world_poses={"band": filtered_world},
                world_reference=reference,
            ),
            calibration,
        )

        np.testing.assert_allclose(
            trajectory._world_points["band"][-1], [0.1, 0.0, 0.0]
        )
        self.assertEqual(trajectory._last_world_projection["band"][-1], (70, 50))

    def test_world_trajectory_ignores_points_behind_camera(self):
        calibration = Calibration(
            np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]),
            np.zeros(5),
            (100, 100),
        )
        trajectory = FadingTrajectory(["band"])
        output = np.zeros((100, 100, 3), dtype=np.uint8)
        reference = Pose(np.zeros((3, 1)), np.zeros((3, 1)), 0.0)
        behind = Pose(
            np.zeros((3, 1)), np.array([[0.1], [0.0], [-1.0]]), 0.0
        )

        trajectory.draw(
            output,
            FrameResult(
                {},
                {},
                world_poses={"band": behind},
                world_reference=reference,
            ),
            calibration,
        )

        self.assertIsNone(trajectory._last_world_projection["band"][-1])

    def test_short_world_gap_reconnects_world_trajectory(self):
        calibration = Calibration(
            np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]),
            np.zeros(5),
            (100, 100),
        )
        trajectory = FadingTrajectory(["band"], max_points=10)
        output = np.zeros((100, 100, 3), dtype=np.uint8)
        reference = Pose(
            np.zeros((3, 1)), np.array([[0.0], [0.0], [1.0]]), 0.0
        )
        world_pose = Pose(
            np.zeros((3, 1)), np.array([[0.1], [0.0], [0.0]]), 0.0
        )
        camera_pose = Pose(
            np.zeros((3, 1)), np.array([[0.1], [0.0], [1.0]]), 0.0
        )

        trajectory.draw(
            output,
            FrameResult(
                {},
                {"band": camera_pose},
                world_poses={"band": world_pose},
                world_reference=reference,
            ),
            calibration,
        )
        trajectory.draw(
            output,
            FrameResult(
                {},
                {"band": camera_pose},
                world_poses={"band": world_pose},
                world_reference=reference,
            ),
            calibration,
        )
        history_before_gap = list(trajectory._world_points["band"])
        output.fill(0)
        trajectory.draw(output, FrameResult({}, {"band": camera_pose}), calibration)
        self.assertEqual(trajectory.mode, "world_hold")
        self.assertEqual(list(trajectory._world_points["band"]), history_before_gap)
        self.assertGreater(np.count_nonzero(output), 0)
        trajectory.draw(
            output,
            FrameResult(
                {},
                {"band": camera_pose},
                world_poses={"band": world_pose},
                world_reference=reference,
            ),
            calibration,
        )

        self.assertEqual(trajectory.mode, "world")
        self.assertFalse(
            any(point is None for point in trajectory._world_points["band"])
        )
        self.assertFalse(trajectory._points["band"])

    def test_long_world_gap_breaks_world_trajectory(self):
        calibration = Calibration(
            np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]),
            np.zeros(5),
            (100, 100),
        )
        trajectory = FadingTrajectory(["band"], max_points=20)
        output = np.zeros((100, 100, 3), dtype=np.uint8)
        reference = Pose(
            np.zeros((3, 1)), np.array([[0.0], [0.0], [1.0]]), 0.0
        )
        world_pose = Pose(
            np.zeros((3, 1)), np.array([[0.1], [0.0], [0.0]]), 0.0
        )
        world_result = FrameResult(
            {},
            {},
            world_poses={"band": world_pose},
            world_reference=reference,
        )

        trajectory.draw(output, world_result, calibration)
        for _ in range(6):
            trajectory.draw(output, FrameResult({}, {}), calibration)
        trajectory.draw(output, world_result, calibration)

        self.assertTrue(
            any(point is None for point in trajectory._world_points["band"])
        )

    def test_world_only_mode_never_draws_camera_trajectory(self):
        calibration = Calibration(
            np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]),
            np.zeros(5),
            (100, 100),
        )
        trajectory = FadingTrajectory(
            ["band"], world_hold_frames=0, world_only=True
        )
        output = np.zeros((100, 100, 3), dtype=np.uint8)
        reference = Pose(
            np.zeros((3, 1)), np.array([[0.0], [0.0], [1.0]]), 0.0
        )
        world_pose = Pose(
            np.zeros((3, 1)), np.array([[0.1], [0.0], [0.0]]), 0.0
        )
        camera_pose = Pose(
            np.zeros((3, 1)), np.array([[0.1], [0.0], [1.0]]), 0.0
        )
        trajectory.draw(
            output,
            FrameResult(
                {},
                {"band": camera_pose},
                world_poses={"band": world_pose},
                world_reference=reference,
            ),
            calibration,
        )

        trajectory.draw(
            output,
            FrameResult({}, {"band": camera_pose}),
            calibration,
        )

        self.assertEqual(trajectory.mode, "world_unavailable")
        self.assertFalse(trajectory._points["band"])
        self.assertIsNone(trajectory._world_points["band"][-1])

    def test_world_only_display_does_not_depend_on_camera_pose(self):
        calibration = Calibration(
            np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]),
            np.zeros(5),
            (400, 300),
        )
        world_pose = Pose(
            np.zeros((3, 1)), np.array([[0.1], [-0.05], [0.0]]), 0.0
        )
        references = [
            Pose(np.zeros((3, 1)), np.array([[0.0], [0.0], [1.0]]), 0.0),
            Pose(np.array([[0.0], [0.3], [0.0]]), np.array([[0.4], [0.2], [0.8]]), 0.0),
        ]
        outputs = []
        for reference in references:
            trajectory = FadingTrajectory(["band"], world_only=True)
            output = np.zeros((300, 400, 3), dtype=np.uint8)
            trajectory.draw(
                output,
                FrameResult(
                    {},
                    {},
                    world_poses={"band": world_pose},
                    world_reference=reference,
                ),
                calibration,
            )
            outputs.append(output)

        np.testing.assert_array_equal(outputs[0], outputs[1])

    def test_hand_axis_uses_camera_pose_when_world_pose_is_available(self):
        calibration = Calibration(
            np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]),
            np.zeros(5),
            (100, 100),
        )
        trajectory = FadingTrajectory(["band"])
        camera_pose = Pose(
            np.zeros((3, 1)), np.array([[0.0], [0.0], [1.0]]), 0.0
        )
        world_reference = Pose(
            np.zeros((3, 1)), np.array([[0.0], [0.0], [1.0]]), 0.0
        )
        world_pose = Pose(
            np.zeros((3, 1)), np.array([[0.5], [0.0], [0.0]]), 0.0
        )

        draw_result(
            np.zeros((100, 100, 3), dtype=np.uint8),
            FrameResult(
                {},
                {"band": camera_pose},
                world_poses={"band": world_pose},
                world_reference=world_reference,
            ),
            calibration,
            trajectory,
        )

        np.testing.assert_allclose(trajectory._axis_points["band"][0], (50.0, 50.0))

    def test_axis_is_hidden_when_pose_is_missing(self):
        calibration = Calibration(
            np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]),
            np.zeros(5),
            (100, 100),
        )
        trajectory = FadingTrajectory(["band"])
        output = np.zeros((100, 100, 3), dtype=np.uint8)
        pose = Pose(np.zeros((3, 1)), np.array([[0.0], [0.0], [1.0]]), 0.0)

        self.assertTrue(
            trajectory.draw_pose_axes(output, "band", calibration, pose, None, 0.1, 2)
        )
        output.fill(0)
        self.assertFalse(
            trajectory.draw_pose_axes(output, "band", calibration, None, None, 0.1, 2)
        )
        self.assertEqual(np.count_nonzero(output), 0)
        self.assertIsNone(trajectory._axis_points["band"])

    def test_axis_origin_follows_raw_pose_without_screen_filter_lag(self):
        calibration = Calibration(
            np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]),
            np.zeros(5),
            (100, 100),
        )
        trajectory = FadingTrajectory(["band"])
        output = np.zeros((100, 100, 3), dtype=np.uint8)
        filtered = Pose(
            np.zeros((3, 1)), np.array([[0.0], [0.0], [1.0]]), 0.0
        )
        first_raw = Pose(
            np.zeros((3, 1)), np.array([[0.0], [0.0], [1.0]]), 0.0
        )
        jittered_raw = Pose(
            np.zeros((3, 1)), np.array([[0.01], [0.0], [1.0]]), 0.0
        )

        trajectory.draw_pose_axes(
            output, "band", calibration, filtered, first_raw, 0.1, 2
        )
        trajectory.draw_pose_axes(
            output, "band", calibration, filtered, jittered_raw, 0.1, 2
        )

        origin_x = trajectory._axis_points["band"][0, 0]
        self.assertEqual(origin_x, 51.0)

    def test_axis_uses_filtered_rotation_with_raw_translation(self):
        calibration = Calibration(
            np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]),
            np.zeros(5),
            (100, 100),
        )
        trajectory = FadingTrajectory(["band"])
        output = np.zeros((100, 100, 3), dtype=np.uint8)
        filtered = Pose(
            np.zeros((3, 1)), np.array([[0.0], [0.0], [1.0]]), 0.0
        )
        raw = Pose(
            np.array([[0.0], [0.0], [np.pi / 2.0]]),
            np.array([[0.2], [0.0], [1.0]]),
            0.0,
        )

        trajectory.draw_pose_axes(
            output, "band", calibration, filtered, raw, 0.1, 2
        )

        axes = trajectory._axis_points["band"]
        np.testing.assert_allclose(axes[0], (70.0, 50.0))
        self.assertGreater(axes[1, 0], axes[0, 0])
        self.assertAlmostEqual(axes[1, 1], axes[0, 1])

    def test_axis_hides_large_ambiguous_raw_jump(self):
        calibration = Calibration(
            np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]),
            np.zeros(5),
            (100, 100),
        )
        trajectory = FadingTrajectory(["band"])
        output = np.zeros((100, 100, 3), dtype=np.uint8)
        filtered = Pose(
            np.zeros((3, 1)), np.array([[0.0], [0.0], [1.0]]), 0.0
        )
        good_raw = Pose(
            np.zeros((3, 1)), np.array([[0.0], [0.0], [1.0]]), 0.0,
            ambiguous=True,
        )
        outlier = Pose(
            np.zeros((3, 1)), np.array([[0.5], [0.0], [1.0]]), 0.0,
            ambiguous=True,
        )

        trajectory.draw_pose_axes(
            output, "band", calibration, filtered, good_raw, 0.1, 2
        )
        output.fill(0)
        shown = trajectory.draw_pose_axes(
            output, "band", calibration, filtered, outlier, 0.1, 2
        )

        self.assertFalse(shown)
        self.assertEqual(np.count_nonzero(output), 0)
        self.assertIsNone(trajectory._axis_points["band"])

    def test_axes_are_clipped_at_image_boundary(self):
        calibration = Calibration(
            np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]),
            np.zeros(5),
            (100, 100),
        )
        output = np.zeros((100, 100, 3), dtype=np.uint8)

        draw_frame_axes(
            output,
            calibration,
            np.zeros((3, 1)),
            np.array([[0.45], [0.0], [1.0]]),
            0.2,
            2,
        )

        self.assertGreater(np.count_nonzero(output), 0)


if __name__ == "__main__":
    unittest.main()
