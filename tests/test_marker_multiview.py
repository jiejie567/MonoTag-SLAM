import copy
import unittest

import cv2
import numpy as np

from aruco_track.marker_multiview import MarkerView, fit_marker_multiview, refit_marker_view
from aruco_track.models import Calibration, Pose
from aruco_track.pose import square_object_points


class MarkerMultiViewTests(unittest.TestCase):
    def make_views(self, baseline=.35, noise=.12, count=24,
                   marker_rvec=(.12, .28, -.08), marker_translation=(.05, .02, 1.2)):
        calibration = Calibration(np.array([[744., 0, 960.], [0, 744., 540.], [0, 0, 1.]]),
                                  np.zeros(5), (1920, 1080))
        marker_rotation = cv2.Rodrigues(np.array(marker_rvec, float))[0]
        marker_translation = np.array(marker_translation, float)
        scale = 2.4
        world = square_object_points(.048) @ marker_rotation.T + marker_translation
        rng = np.random.default_rng(83)
        views = []
        for index in range(count):
            fraction = index / (count - 1)
            center = np.array([baseline * fraction, .12 * baseline * np.sin(fraction * np.pi), 0.])
            rotation = cv2.Rodrigues(np.array([.015 * fraction, .04 * fraction, -.01 * fraction]))[0]
            camera = (world - center) @ rotation
            pixels = cv2.projectPoints(camera, np.zeros(3), np.zeros(3),
                                       calibration.camera_matrix, calibration.dist_coeffs)[0].reshape(4, 2)
            pixels += rng.normal(0, noise, pixels.shape)
            views.append(MarkerView(index, index / 30.,
                                    Pose(cv2.Rodrigues(rotation)[0], (center / scale).reshape(3, 1), 0),
                                    pixels, keyframe_id=index // 3))
        return views, calibration, scale, marker_rotation, marker_translation

    def fit(self, views, calibration, **kwargs):
        return fit_marker_multiview(views, calibration, .048, 28,
                                    trajectory_is_independent=True, **kwargs)

    def test_recovers_positive_scale_and_marker_pose_from_independent_motion(self):
        views, calibration, scale, rotation, translation = self.make_views()
        before = copy.deepcopy(views)
        result = self.fit(views, calibration)
        self.assertTrue(result.accepted, (result.reason, result.diagnostics))
        self.assertAlmostEqual(result.scale_m_per_unit, scale, delta=.08)
        self.assertLess(np.linalg.norm(result.marker_pose.tvec.ravel() - translation), .03)
        angle = np.linalg.norm(cv2.Rodrigues(rotation.T @ result.marker_pose.rotation_matrix)[0])
        self.assertLess(angle, np.deg2rad(4))
        self.assertGreaterEqual(result.diagnostics['ippe_initializations'], 6)
        self.assertGreaterEqual(len(result.validation_frame_ids), 2)
        self.assertFalse(set(result.training_frame_ids) & set(result.validation_frame_ids))
        self.assertEqual(set(result.camera_in_marker), set(range(24)))
        for old, current in zip(before, views):
            np.testing.assert_array_equal(old.corners, current.corners)
            np.testing.assert_array_equal(old.camera_pose.tvec, current.camera_pose.tvec)

    def test_independently_known_scale_can_localize_unknown_marker(self):
        views, calibration, scale, _, _ = self.make_views()
        result = self.fit(views, calibration, fixed_scale=scale, trajectory_marker_ids=(26, 45))
        self.assertTrue(result.accepted, (result.reason, result.diagnostics))
        self.assertEqual(result.scale_m_per_unit, scale)
        self.assertEqual(result.diagnostics['scale_mode'], 'independent_fixed')

    def test_refit_uses_real_pixels_and_does_not_extrapolate_or_freeze_prior(self):
        views, calibration, *_ = self.make_views()
        fit = self.fit(views, calibration)
        measured = refit_marker_view(views[8], fit, calibration, .048)
        self.assertIsNotNone(measured)
        self.assertLess(measured.reprojection_error_px, .5)
        self.assertFalse(np.array_equal(measured.tvec, fit.camera_in_marker[8].tvec))
        missing = MarkerView(100, 4., views[8].camera_pose, views[8].corners)
        self.assertIsNone(refit_marker_view(missing, fit, calibration, .048))

    def test_target_marker_cannot_validate_its_own_camera_trajectory(self):
        views, calibration, scale, _, _ = self.make_views()
        result = self.fit(views, calibration, fixed_scale=scale, trajectory_marker_ids=(28,))
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, 'trajectory_not_independent')
        result = fit_marker_multiview(views, calibration, .048, 28,
                                     trajectory_is_independent=False)
        self.assertEqual(result.reason, 'trajectory_not_independent')

    def test_rotation_only_cannot_estimate_translation_scale(self):
        views, calibration, *_ = self.make_views(baseline=0)
        result = self.fit(views, calibration)
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, 'unobservable_translation_scale')

    def test_held_out_inconsistent_corners_are_not_fit_or_admitted(self):
        views, calibration, *_ = self.make_views()
        clean = self.fit(views, calibration)
        for index in clean.validation_frame_ids:
            views[index].corners[:] += [4.0, -3.0]
        result = self.fit(views, calibration)
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, 'reprojection_validation_failed')
        self.assertLess(result.diagnostics['training_rms_px'], .5)
        self.assertGreater(result.diagnostics['validation_rms_px'], 4.)

    def test_weak_views_duplicate_frames_and_mixed_gauges_do_not_supply_evidence(self):
        views, calibration, *_ = self.make_views(count=8)
        weak = [MarkerView(v.frame_id, v.timestamp, v.camera_pose, v.corners, .25) for v in views]
        self.assertEqual(self.fit(weak, calibration).reason, 'insufficient_strong_views')
        with self.assertRaises(ValueError):
            self.fit(views + [views[0]], calibration)
        mixed = [*views[:-1], MarkerView(7, views[-1].timestamp, views[-1].camera_pose,
                                       views[-1].corners, gauge_id=1)]
        self.assertEqual(self.fit(mixed, calibration).reason, 'mixed_map_or_gauge')
        one_kf = [MarkerView(v.frame_id, v.timestamp, v.camera_pose, v.corners, keyframe_id=0)
                  for v in views]
        self.assertEqual(self.fit(one_kf, calibration).reason, 'insufficient_keyframes')

    def test_subcentimetre_baseline_does_not_become_metric_evidence(self):
        views, calibration, *_ = self.make_views(baseline=.003, noise=0)
        result = self.fit(views, calibration)
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, 'insufficient_motion_baseline')

    def test_two_markers_have_independent_non_coplanar_poses(self):
        first, calibration, scale, _, _ = self.make_views()
        second, _, _, second_rotation, second_translation = self.make_views(
            marker_rvec=(.55, -.20, .05), marker_translation=(-.20, .18, 1.7))
        a = self.fit(first, calibration)
        b = fit_marker_multiview(second, calibration, .048, 29,
                                trajectory_is_independent=True)
        self.assertTrue(a.accepted, (a.reason, a.diagnostics))
        self.assertTrue(b.accepted, (b.reason, b.diagnostics))
        self.assertAlmostEqual(b.scale_m_per_unit, scale, delta=.10)
        self.assertLess(np.linalg.norm(b.marker_pose.tvec.ravel() - second_translation), .05)
        angle = np.linalg.norm(cv2.Rodrigues(second_rotation.T @ b.marker_pose.rotation_matrix)[0])
        self.assertLess(angle, np.deg2rad(5))
        normals = [fit.marker_pose.rotation_matrix[:, 2] for fit in (a, b)]
        self.assertGreater(np.linalg.norm(normals[0] - normals[1]), .4)
        self.assertNotEqual(a.marker_id, b.marker_id)

    def test_correlated_120hz_repeats_do_not_artificially_shrink_uncertainty(self):
        views, calibration, *_ = self.make_views()
        low_rate = self.fit(views, calibration)
        repeated = [MarkerView(4 * v.frame_id + j, v.timestamp + j / 120,
                               v.camera_pose, v.corners.copy(), keyframe_id=v.keyframe_id)
                    for v in views for j in range(4)]
        high_rate = self.fit(repeated, calibration)
        self.assertTrue(low_rate.accepted, low_rate.reason)
        self.assertTrue(high_rate.accepted, (high_rate.reason, high_rate.diagnostics))
        # IID counting would halve standard errors after quadrupling identical
        # measurements. Time-block information caps must prevent that.
        for field in ('relative_scale_std', 'rotation_std_deg'):
            self.assertGreater(high_rate.diagnostics[field], .8 * low_rate.diagnostics[field])
        self.assertEqual(high_rate.diagnostics['training_time_blocks'],
                         low_rate.diagnostics['training_time_blocks'])
        train_blocks = {int((repeated[i].timestamp + 1e-9) / .05)
                        for i in high_rate.training_frame_ids}
        held_blocks = {int((repeated[i].timestamp + 1e-9) / .05)
                       for i in high_rate.validation_frame_ids}
        self.assertFalse(train_blocks & held_blocks)

    def test_competing_branches_with_good_pixel_residuals_remain_rejected(self):
        views, calibration, *_ = self.make_views(
            baseline=.07, noise=.5, marker_rvec=(.01, .03, 0))
        result = self.fit(views, calibration)
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, 'unresolved_multiview_branches')
        self.assertLess(result.diagnostics['validation_rms_px'], 1.)
        self.assertLess(result.diagnostics['competitor_validation_rms_px'], 1.)


if __name__ == '__main__':
    unittest.main()
