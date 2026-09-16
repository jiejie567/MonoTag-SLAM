import unittest
import cv2
import numpy as np
from aruco_track.models import Pose, Calibration
from aruco_track.camera_state import FusedCameraFrame
from aruco_track.orbslam3_backend import (_validate_large_frame_refinement,
    refine_final_frame_poses, MetricOrbSlamResult, OrbSlamObservation)


class LargeFrameRefinementTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(73)
        self.points = rng.uniform([-.5, -.4, .7], [.5, .4, 2.], (80, 3))
        self.camera = np.array([[700., 0, 320], [0, 700, 240], [0, 0, 1]])
        self.candidate = np.array([.003, -.005, .002, .018, -.018, .015])
        self.pixels = cv2.projectPoints(self.points, self.candidate[:3],
                                      self.candidate[3:], self.camera, np.zeros(5))[0].reshape(-1, 2)

    def check_candidate(self, points=None, pixels=None, info=None, candidate=None):
        return _validate_large_frame_refinement(
            self.points if points is None else points,
            self.pixels if pixels is None else pixels,
            np.ones(80) if info is None else info, self.camera,
            self.candidate if candidate is None else candidate)

    def test_two_disjoint_fits_confirm_large_geometric_correction(self):
        self.assertTrue(self.check_candidate())

    def test_conflicting_pixel_halves_rejected(self):
        bad = self.pixels.copy(); bad[::2] += [20., -10.]
        self.assertFalse(self.check_candidate(pixels=bad))

    def test_markers_cannot_supply_independent_background_support(self):
        info = np.ones(80); info[30:] = 64.
        self.assertFalse(self.check_candidate(info=info))
        # Factor identity, not numerical weight, decides background support.
        self.assertFalse(_validate_large_frame_refinement(
            self.points, self.pixels, np.ones(80), self.camera, self.candidate,
            background_count=0))

    def test_large_or_disagreeing_candidate_rejected(self):
        bad = self.candidate.copy(); bad[3] = .05
        self.assertFalse(self.check_candidate(candidate=bad))
        bad[3] = -.018
        self.assertFalse(self.check_candidate(candidate=bad))

    def test_insufficient_or_negative_depth_support_rejected(self):
        self.assertFalse(self.check_candidate(points=self.points[:39], pixels=self.pixels[:39], info=np.ones(39)))
        self.assertFalse(self.check_candidate(points=-self.points))

    def test_degenerate_background_is_not_independent_support(self):
        points = np.tile([.1, .1, 1.], (80, 1))
        pixels = cv2.projectPoints(points, self.candidate[:3], self.candidate[3:],
                                  self.camera, np.zeros(5))[0].reshape(-1, 2)
        self.assertFalse(self.check_candidate(points=points, pixels=pixels))

    def test_full_refinement_commits_geometry_not_zero_motion(self):
        frame = FusedCameraFrame(Pose(np.zeros((3, 1)), np.zeros((3, 1)), 0.),
                                 'head-slam', 1., 80, None, 'atlas_0', metric=True)
        observation = OrbSlamObservation(80, self.pixels, np.empty((0, 2)), 2, np.arange(80))
        result = MetricOrbSlamResult([frame], self.points, (), [observation], 0, 1., 0,
            None, None, {}, [], {'atlas_0': {'points': [[i, *p] for i, p in enumerate(self.points)], 'markers': {}}})
        cal = Calibration(self.camera, np.zeros(5), (640, 480))
        fixed = refine_final_frame_poses(result, cal, [{}], [()])
        self.assertEqual(fixed.timing['dense_pose_large_accepted'], 1)
        expected = -cv2.Rodrigues(self.candidate[:3])[0].T @ self.candidate[3:]
        np.testing.assert_allclose(fixed.frames[0].pose.tvec.ravel(), expected, atol=1e-6)
        self.assertGreater(np.linalg.norm(fixed.frames[0].pose.tvec), .02)
        # Duplicate IDs cannot turn a handful of matches into independent support.
        result.observations[0] = OrbSlamObservation(80, self.pixels, np.empty((0, 2)), 2, np.arange(80) % 20)
        rejected = refine_final_frame_poses(result, cal, [{}], [()])
        self.assertEqual(rejected.timing['dense_pose_large_accepted'], 0)
