"""Regression cases for planar ambiguity and stale marker pose predictions."""
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from aruco_track.auto_marker_map import (
    AutoMarkerMap, AutoMarkerSubmap, localize_auto_marker_frames,
)
from aruco_track.models import BandLayout, Calibration, Pose
from aruco_track.pipeline import inverse_pose
from aruco_track.pose import square_object_points
from aruco_track.tag_graph import MarkerPoseTracker, TagPoseResult, optimize_tag_pose


class MarkerLocalizationSafetyTests(unittest.TestCase):
    def setUp(self):
        self.calibration = Calibration(
            np.array([[1100., 0., 960.], [0., 1100., 540.], [0., 0., 1.]]),
            np.zeros(5), (1920, 1080),
        )
        self.layout = BandLayout('test', 'DICT_4X4_50', {20: square_object_points(.048)})

    def project(self, pose):
        return cv2.projectPoints(
            self.layout.markers[20], pose.rvec, pose.tvec,
            self.calibration.camera_matrix, self.calibration.dist_coeffs,
        )[0].reshape(4, 2)

    def pose(self, z=.65, tilt=.3):
        return Pose(np.array([np.pi-tilt, .01, 0.]).reshape(3, 1),
                    np.array([0., 0., z]).reshape(3, 1), 0.)

    def marker_map(self):
        identity = Pose(np.zeros((3, 1)), np.zeros((3, 1)), 0.)
        submap = AutoMarkerSubmap('test', 20, {20: identity}, self.layout, (), 0.)
        return AutoMarkerMap('DICT_4X4_50', .048, (submap,))

    def solve(self, pixels, previous=None):
        return optimize_tag_pose({20: pixels}, self.layout, self.calibration, previous,
                                 validate_planar_ambiguity=True)

    def test_distinct_near_equal_planar_solutions_cannot_seed_a_world(self):
        truth = self.pose(tilt=.02)
        pixels = self.project(truth) + np.random.default_rng(13).normal(0, .20, (4, 2))
        result = self.solve(pixels)
        self.assertIsNone(result.pose, 'small reprojection error is not planar disambiguation')
        self.assertEqual(result.confidence, 0.)

    def test_clear_single_marker_still_initializes_without_motion(self):
        truth = self.pose(tilt=.6)
        result = self.solve(self.project(truth))
        self.assertIsNotNone(result.pose)
        self.assertGreaterEqual(result.confidence, .35)
        np.testing.assert_allclose(result.pose.tvec, truth.tvec, atol=1e-5)
        np.testing.assert_allclose(result.pose.rotation_matrix, truth.rotation_matrix, atol=1e-5)

    def test_coincident_frontal_solutions_are_not_rejected_by_tag_count(self):
        truth = Pose(np.array([np.pi, 0., 0.]).reshape(3, 1), np.array([[0.], [0.], [.65]]), 0.)
        result = self.solve(self.project(truth))
        self.assertIsNotNone(result.pose)
        self.assertGreaterEqual(result.confidence, .35)

    def test_recent_good_prediction_disambiguates_noisy_single_marker(self):
        truth = self.pose(tilt=.02)
        pixels = self.project(truth) + np.random.default_rng(13).normal(0, .20, (4, 2))
        result = self.solve(pixels, truth)
        self.assertIsNotNone(result.pose)
        self.assertGreaterEqual(result.confidence, .35)
        self.assertLess(np.linalg.norm(inverse_pose(result.pose).tvec-inverse_pose(truth).tvec), .02)

    def test_prior_does_not_count_as_extra_fixed_marker_evidence(self):
        truth = self.pose(tilt=.6)
        pixels = self.project(truth)
        first, repeated = self.solve(pixels), self.solve(pixels, truth)
        self.assertIsNotNone(first.pose)
        self.assertIsNotNone(repeated.pose)
        self.assertAlmostEqual(repeated.confidence, first.confidence, places=6)

    def test_prior_cannot_promote_a_small_single_tag_to_a_strong_anchor(self):
        calibration = Calibration(
            np.array([[550., 0., 960.], [0., 550., 540.], [0., 0., 1.]]),
            np.zeros(5), (1920, 1080),
        )
        truth = self.pose(tilt=.6)
        pixels = cv2.projectPoints(self.layout.markers[20], truth.rvec, truth.tvec,
                                   calibration.camera_matrix, calibration.dist_coeffs)[0].reshape(4, 2)
        result = optimize_tag_pose({20: pixels}, self.layout, calibration, truth,
                                   validate_planar_ambiguity=True)
        self.assertIsNotNone(result.pose)
        self.assertLess(result.confidence, .35)

    def test_long_occlusion_does_not_leave_an_unusable_prior_forever(self):
        # Use well-separated planar solutions so this isolates stale state,
        # rather than accepting a genuinely ambiguous far/frontal view.
        first, recovered = self.pose(.5, .6), self.pose(.9, .6)
        frames = [{20: self.project(first)}] + [{}]*60 + [{20: self.project(recovered)}]*8
        results = localize_auto_marker_frames(self.marker_map(), frames, self.calibration)
        self.assertIsNotNone(results.results[0])
        self.assertTrue(all(result is None for result in results.results[1:61]))
        self.assertTrue(all(result is not None for result in results.results[-8:]))
        np.testing.assert_allclose(results.results[-1].pose.tvec, recovered.tvec, atol=1e-4)

    def test_repeated_rejection_also_expires_the_prediction(self):
        first, recovered = self.pose(.5, .6), self.pose(.9, .6)
        frames = [{20: self.project(first)}] + [{20: self.project(recovered)}]*30
        results = localize_auto_marker_frames(self.marker_map(), frames, self.calibration)
        self.assertIsNotNone(results.results[-1], 'visible but rejected frames must not refresh the prior age')
        np.testing.assert_allclose(results.results[-1].pose.tvec, recovered.tvec, atol=1e-4)

    def test_fixed_board_tracker_expires_rejected_and_missing_observations(self):
        tracker = MarkerPoseTracker(self.layout, self.calibration)
        self.assertIsNotNone(tracker.update({20: self.project(self.pose(.5, .6))}, 0.).pose)
        self.assertIsNone(tracker.update({}, .1).pose)
        recovered = self.pose(.9, .6)
        result = tracker.update({20: self.project(recovered)}, 3.)
        self.assertIsNotNone(result.pose)
        np.testing.assert_allclose(result.pose.tvec, recovered.tvec, atol=1e-4)

    def test_weak_observations_cannot_extend_the_reliable_prediction(self):
        tracker = MarkerPoseTracker(self.layout, self.calibration)
        pixels = self.project(self.pose(tilt=.6))
        self.assertIsNotNone(tracker.update({20: pixels}, 0.).pose)
        self.assertIsNotNone(tracker.update({20: pixels}, .1, {20: .25}).pose)
        self.assertIsNone(tracker.update({20: pixels}, .2, {20: .25}).pose)

    def test_rejecting_other_tags_cannot_bypass_single_marker_validation(self):
        truth = self.pose(tilt=.02)
        pixels = self.project(truth) + np.random.default_rng(13).normal(0, .20, (4, 2))
        layout = BandLayout('test', 'DICT_4X4_50', {
            20: self.layout.markers[20], 21: self.layout.markers[20]+[.14, 0., 0.],
        })
        bad = cv2.projectPoints(layout.markers[21], truth.rvec, truth.tvec,
                               self.calibration.camera_matrix, self.calibration.dist_coeffs)[0].reshape(4, 2)
        bad[:3] += np.random.default_rng(0).normal(0, 40., (3, 2))
        cv2.setRNGSeed(0)
        result = optimize_tag_pose({20: pixels, 21: bad}, layout, self.calibration,
                                   validate_planar_ambiguity=True)
        self.assertIsNone(result.pose)
        self.assertEqual(result.confidence, 0.)

    def test_auto_map_weak_chain_cannot_keep_a_prediction_alive(self):
        pixels = self.project(self.pose(tilt=.6))
        frames = [{20: pixels}]*12
        weights = [{20: 1.}]+[{20: .25}]*11
        results = localize_auto_marker_frames(self.marker_map(), frames, self.calibration,
                                             marker_weights=weights, fps=30.)
        self.assertTrue(all(result is None for result in results.results[5:]))

    def test_rejected_marker_consensus_vetoes_large_subset_jump(self):
        markers = {
            20: square_object_points(.048),
            21: square_object_points(.048) + [.065, 0., 0.],
            22: square_object_points(.048) + [0., .065, 0.],
        }
        layout = BandLayout('multi', 'DICT_4X4_50', markers)
        previous = self.pose(.65, .35)
        detections = {
            marker_id: cv2.projectPoints(
                points, previous.rvec, previous.tvec,
                self.calibration.camera_matrix, self.calibration.dist_coeffs,
            )[0].reshape(4, 2)
            for marker_id, points in markers.items()
        }
        jumped = Pose(
            previous.rvec.copy(), previous.tvec + [[.08], [0.], [0.]], 0.,
            marker_ids=(20,), inlier_count=4,
        )
        tracker = MarkerPoseTracker(layout, self.calibration)
        tracker._previous = previous
        tracker._last_reliable_time = 0.0
        mocked = TagPoseResult(
            jumped, (20,), (21, 22), {20: 0., 21: 20., 22: 20.}, 0.5, 0.8
        )
        with patch('aruco_track.tag_graph.optimize_tag_pose', return_value=mocked):
            result = tracker.update(detections, .01)
        self.assertIsNone(result.pose)
        self.assertEqual(result.accepted_marker_ids, ())
        self.assertEqual(result.rejected_marker_ids, (20, 21, 22))

    def test_consensus_gate_keeps_a_real_large_motion(self):
        markers = {
            20: square_object_points(.048),
            21: square_object_points(.048) + [.065, 0., 0.],
            22: square_object_points(.048) + [0., .065, 0.],
        }
        layout = BandLayout('multi', 'DICT_4X4_50', markers)
        previous = self.pose(.65, .35)
        moved = Pose(
            previous.rvec.copy(), previous.tvec + [[.08], [0.], [0.]], 0.,
            marker_ids=(20, 21, 22), inlier_count=12,
        )
        detections = {
            marker_id: cv2.projectPoints(
                points, moved.rvec, moved.tvec,
                self.calibration.camera_matrix, self.calibration.dist_coeffs,
            )[0].reshape(4, 2)
            for marker_id, points in markers.items()
        }
        tracker = MarkerPoseTracker(layout, self.calibration)
        tracker._previous = previous
        tracker._last_reliable_time = 0.0
        mocked = TagPoseResult(moved, (20, 21, 22), (), {}, 0.0, 1.0)
        with patch('aruco_track.tag_graph.optimize_tag_pose', return_value=mocked):
            result = tracker.update(detections, .01)
        self.assertIs(result.pose, moved)
        self.assertFalse(result.consensus_vetoed)


if __name__ == '__main__':
    unittest.main()
