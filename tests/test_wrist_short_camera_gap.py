"""Short camera gaps may share an optimization, never a fabricated pose."""
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from tools import export_action_labels as exporter
from aruco_track.camera_state import FusedCameraFrame
from aruco_track.models import Calibration, Pose
from aruco_track.tag_graph import WristTrajectoryResult, optimize_wrist_trajectory
from tools.make_band import band_layout


class WristShortCameraGapTests(unittest.TestCase):
    def calls(self, maps, times):
        camera = Pose(np.zeros((3, 1)), np.zeros((3, 1)), 0.)
        # Even a cached pose in an unassigned/non-metric frame must be omitted.
        frames = [FusedCameraFrame(camera, 'head-slam', 1., 40, 0., m,
                                   metric=m is not None) for m in maps]
        def stub(cameras, *args, **kwargs):
            return WristTrajectoryResult(cameras, [None] * len(cameras))
        with patch.object(exporter, 'optimize_wrist_trajectory', side_effect=stub) as solver:
            result = exporter._optimize_wrists_by_submap(
                frames, maps, times, 60., [camera] * len(maps),
                [{}] * len(maps), [(0,)] * len(maps), [{}] * len(maps), None, None)
        return solver.call_args_list, result

    def test_short_same_world_gap_is_joint_but_missing_stays_missing(self):
        calls, result = self.calls(['A', 'A', None, 'A', 'A'], [i / 60 for i in range(5)])
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0].args[0]), 5)
        self.assertIsNone(result.poses[2])
        self.assertTrue(all(result.poses[i] is not None for i in (0, 1, 3, 4)))

    def test_different_or_intervening_maps_never_join(self):
        for maps, expected in [(['A', None, 'B'], 2), (['A', 'B', 'A'], 3)]:
            with self.subTest(maps=maps):
                calls, _ = self.calls(maps, [0., .01, .02])
                self.assertEqual(len(calls), expected)

    def test_gap_uses_actual_time_and_is_bracketed(self):
        calls, _ = self.calls(['A', None, 'A'], [0., .01, .151])
        self.assertEqual(len(calls), 2)
        calls, out = self.calls([None, 'A', None], [0., .01, .02])
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0].args[0]), 1)
        self.assertIsNone(out.poses[0])
        self.assertIsNone(out.poses[2])

    def test_known_translation_and_rotation_not_flattened(self):
        band = band_layout('left', 0, 69., 55., 56.)
        cal = Calibration(np.array([[1050., 0., 960.], [0., 1040., 540.], [0., 0., 1.]]),
                          np.zeros(5), (1920, 1080))
        n, gap = 19, 9
        times = np.arange(n) / 60.
        frames, wrists, detections, truth = [], [], [], []
        maps = ['A' if i != gap else None for i in range(n)]
        for i, t in enumerate(times):
            camera = Pose(np.array([[0.], [.4 * t], [0.]]),
                          np.array([[.2 * t], [0.], [0.]]), 0.)
            world = Pose(np.array([[0.], [.8 * t], [0.]]),
                         np.array([[.6 * t], [-.02], [.72]]), 0., (0, 1, 2), 12)
            rotation = camera.rotation_matrix.T @ world.rotation_matrix
            relative = Pose(cv2.Rodrigues(rotation)[0],
                            camera.rotation_matrix.T @ (world.tvec - camera.tvec), 0., (0, 1, 2), 12)
            frames.append(FusedCameraFrame(camera if i != gap else None, 'head-slam', 1., 50, 0.,
                                           maps[i], metric=i != gap))
            wrists.append(relative)
            detections.append({m: cv2.projectPoints(band.markers[m], relative.rvec, relative.tvec,
                               cal.camera_matrix, cal.dist_coeffs)[0].reshape(4, 2) for m in (0, 1, 2)})
            truth.append(world)
        def serial(*args, **kwargs):
            return optimize_wrist_trajectory(*args, **kwargs, parallel_workers=1)
        with patch.object(exporter, 'optimize_wrist_trajectory', side_effect=serial):
            result = exporter._optimize_wrists_by_submap(
                frames, maps, times.tolist(), 60., wrists, detections,
                [(0, 1, 2)] * n, [{}] * n, band, cal)
        self.assertIsNone(result.poses[gap])
        for i, pose in enumerate(result.poses):
            if i == gap:
                continue
            np.testing.assert_allclose(pose.tvec, truth[i].tvec, atol=1e-5)
            np.testing.assert_allclose(pose.rotation_matrix, truth[i].rotation_matrix, atol=1e-5)
        self.assertGreater(float(result.poses[-1].tvec[0, 0] - result.poses[0].tvec[0, 0]), .179)


if __name__ == '__main__':
    unittest.main()
