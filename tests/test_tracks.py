#!/usr/bin/env python3
import unittest

import numpy as np

from aruco_track.models import Pose
from aruco_track.tracks import AdaptivePoseSmoother, PoseSmoother, WorldPoseSmoother


class AdaptivePoseSmootherTests(unittest.TestCase):
    @staticmethod
    def pose(x: float, ambiguous: bool = False) -> Pose:
        return Pose(
            np.zeros((3, 1)),
            np.array([[x], [0.0], [1.0]]),
            0.5,
            marker_ids=(0,) if ambiguous else (0, 1),
            ambiguous=ambiguous,
        )

    def test_large_motion_follows_faster_than_fixed_smoothing(self):
        adaptive = AdaptivePoseSmoother()
        fixed = PoseSmoother(translation_alpha=0.18, rotation_alpha=0.12)
        adaptive.update(self.pose(0.0))
        fixed.update(self.pose(0.0))

        adaptive_result = adaptive.update(self.pose(0.01))
        fixed_result = fixed.update(self.pose(0.01))

        self.assertGreater(adaptive_result.tvec[0, 0], fixed_result.tvec[0, 0])

    def test_ambiguous_pose_is_followed_more_cautiously(self):
        confident = AdaptivePoseSmoother()
        ambiguous = AdaptivePoseSmoother()
        confident.update(self.pose(0.0))
        ambiguous.update(self.pose(0.0))

        confident_result = confident.update(self.pose(0.01))
        ambiguous_result = ambiguous.update(self.pose(0.01, ambiguous=True))

        self.assertLess(ambiguous_result.tvec[0, 0], confident_result.tvec[0, 0])


class WorldPoseSmootherTests(unittest.TestCase):
    @staticmethod
    def pose(x: float) -> Pose:
        return Pose(
            np.zeros((3, 1)),
            np.array([[x], [0.0], [1.0]]),
            0.5,
            marker_ids=(0, 1),
        )

    def test_stationary_millimetre_jitter_is_held(self):
        smoother = WorldPoseSmoother()
        outputs = [
            smoother.update(self.pose(x)).tvec[0, 0]
            for x in (0.0, 0.0015, -0.0012, 0.0010, -0.0014, 0.0011)
        ]

        self.assertEqual(outputs, [0.0] * len(outputs))

    def test_real_motion_crosses_deadband_and_is_followed(self):
        smoother = WorldPoseSmoother()
        for _ in range(5):
            smoother.update(self.pose(0.0))
        result = None
        for _ in range(3):
            result = smoother.update(self.pose(0.05))

        self.assertIsNotNone(result)
        self.assertGreater(result.tvec[0, 0], 0.02)


if __name__ == "__main__":
    unittest.main()
