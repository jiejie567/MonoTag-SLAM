"""Quality must use the selected map revision without changing measurements."""
import copy
from dataclasses import replace
import unittest

import cv2
import numpy as np

from aruco_track.camera_state import FusedCameraFrame
from aruco_track.models import Calibration, Pose
from aruco_track.orbslam3_backend import annotate_anchor_consistency


class AnchorConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.calibration = Calibration(np.array([[500., 0., 320.], [0., 500., 240.], [0., 0., 1.]]),
                                       np.zeros(5), (640, 480))
        self.world = np.array([[-.024, .024, 0], [.024, .024, 0],
                               [.024, -.024, 0], [-.024, -.024, 0]])
        self.pixels = cv2.projectPoints(self.world, np.zeros(3), np.array([0., 0., 1.]),
                                       self.calibration.camera_matrix, np.zeros(5))[0].reshape(4, 2)
        self.mapping = {'id': 0, 'markers': {'49': self.world.ravel().tolist()}}
        self.frame = FusedCameraFrame(Pose(np.zeros((3, 1)), np.array([[0.], [0.], [-1.]]), .1),
                                     'head-slam', 1., 200, None, 'atlas_0', 7, True)

    def audit(self, frame=None, mapping=None, accepted=(49,), weights=None):
        return annotate_anchor_consistency(frame or self.frame, mapping or self.mapping,
                                           self.calibration, {49: self.pixels}, accepted, weights)

    def test_consistent_anchor_preserves_every_measurement(self):
        output = self.audit()
        self.assertEqual(output.anchor_consistency['status'], 'consistent')
        self.assertLess(output.anchor_consistency['max_rms_px'], 1e-8)
        self.assertEqual(output.confidence, 1.)
        self.assertIs(output.pose, self.frame.pose)

    def test_large_conflict_is_not_full_confidence_or_pose_replacement(self):
        pose = Pose(np.zeros((3, 1)), np.array([[4.5], [0.], [-1.]]), .1)
        frame = replace(self.frame, pose=pose)
        mapping_before = copy.deepcopy(self.mapping)
        output = self.audit(frame)
        self.assertEqual(output.anchor_consistency['status'], 'conflict')
        self.assertEqual(output.confidence, 0.)
        self.assertIs(output.pose, pose)
        self.assertEqual(output.source, 'head-slam')
        self.assertEqual(self.mapping, mapping_before)

    def test_low_projected_error_behind_camera_is_a_conflict(self):
        pose = Pose(np.zeros((3, 1)), np.array([[0.], [0.], [1.]]), .1)
        output = self.audit(replace(self.frame, pose=pose))
        self.assertEqual(output.anchor_consistency['status'], 'conflict')
        self.assertLess(output.anchor_consistency['minimum_depth_m'], 0)
        self.assertIsNone(output.anchor_consistency['max_rms_px'])

    def test_rejected_weak_nan_or_unknown_marker_does_not_veto(self):
        for accepted, weights in [((), None), ((49,), {49: .25}), ((49,), {49: float('nan')}), ((20,), None)]:
            with self.subTest(accepted=accepted, weights=weights):
                output = self.audit(accepted=accepted, weights=weights)
                self.assertEqual(output.anchor_consistency['status'], 'unobserved')
                self.assertEqual(output.confidence, 1.)

    def test_other_map_is_not_an_anchor_in_this_gauge(self):
        other = copy.deepcopy(self.mapping)
        other['id'] = 9
        self.assertEqual(self.audit(mapping=other).anchor_consistency['status'], 'unavailable')

    def test_other_revision_cannot_report_false_conflict(self):
        stale = copy.deepcopy(self.mapping)
        stale['revision'] = self.frame.revision - 1
        stale['markers']['49'] = (self.world + [4., 0., 0.]).ravel().tolist()
        result = self.audit(mapping=stale)
        self.assertEqual(result.anchor_consistency['status'], 'unavailable')
        self.assertEqual(result.confidence, self.frame.confidence)
        self.assertIs(result.pose, self.frame.pose)

    def test_rotated_distorted_pose_and_rigid_world_revision(self):
        from aruco_track.pipeline import inverse_pose
        calibration = Calibration(self.calibration.camera_matrix,
                                  np.array([.1, -.03, .002, -.001, .01]), (640, 480))
        camera = Pose(np.array([[.1], [.2], [-.1]]), np.array([[.05], [.02], [1.]]), .1)
        frame = replace(self.frame, pose=inverse_pose(camera))
        pixels = cv2.projectPoints(self.world, camera.rvec, camera.tvec,
                                   calibration.camera_matrix, calibration.dist_coeffs)[0].reshape(4, 2)
        result = annotate_anchor_consistency(frame, self.mapping, calibration, {49: pixels}, (49,))
        self.assertLess(result.anchor_consistency['max_rms_px'], 1e-5)
        rotation = cv2.Rodrigues(np.array([.2, -.1, .3]))[0]
        translation = np.array([.3, .2, -.1])
        moved_map = {'id': 0, 'revision': 8, 'markers': {
            '49': (self.world @ rotation.T + translation).ravel().tolist()}}
        moved_pose = Pose(cv2.Rodrigues(rotation @ frame.pose.rotation_matrix)[0],
                          (rotation @ frame.pose.tvec.reshape(3) + translation).reshape(3, 1), .1)
        revised = replace(frame, pose=moved_pose, revision=8)
        result = annotate_anchor_consistency(revised, moved_map, calibration, {49: pixels}, (49,))
        self.assertEqual(result.anchor_consistency['status'], 'consistent')
        self.assertLess(result.anchor_consistency['max_rms_px'], 1e-5)

    def test_successful_revision_is_rechecked_not_stale_conflict(self):
        moved = replace(self.frame, pose=Pose(np.zeros((3, 1)), np.array([[4.5], [0.], [-1.]]), .1))
        previous = self.audit(moved)
        corrected = replace(self.frame, revision=8, anchor_consistency=previous.anchor_consistency)
        result = self.audit(corrected)
        self.assertEqual(result.anchor_consistency['revision'], 8)
        self.assertEqual(result.anchor_consistency['status'], 'consistent')
        self.assertEqual(result.confidence, 1.)

    def test_missing_pose_is_not_recovered_by_quality_annotation(self):
        result = self.audit(replace(self.frame, pose=None, source='invalid', confidence=0.))
        self.assertEqual(result.anchor_consistency['status'], 'unavailable')
        self.assertIsNone(result.pose)


if __name__ == '__main__':
    unittest.main()
