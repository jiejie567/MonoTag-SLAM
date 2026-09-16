import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from aruco_track.marker_bootstrap import bootstrap_initial_marker_observations
from aruco_track.marker_multiview import MultiViewMarkerFit
from aruco_track.models import BandLayout, Calibration, Pose
from aruco_track.pose import square_object_points


class MarkerBootstrapContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.sequence = self.root / 'cache'
        self.sequence.mkdir()
        self.work = self.root / 'work'
        self.count, self.fps = 20, 10.
        self.tracked_count = 10
        (self.sequence / 'rgb.txt').write_text(''.join(f'{i / self.fps:.9f} rgb/{i}.jpg\n'
                                                      for i in range(self.count)))
        (self.sequence / 'tag_observations.txt').write_text('# original cache hints\n' +
            ''.join(f'{i / self.fps:.9f} 0\n' for i in range(self.count)))
        self.original_rgb = (self.sequence / 'rgb.txt').read_bytes()
        self.original_hints = (self.sequence / 'tag_observations.txt').read_bytes()
        self.calibration = Calibration(np.array([[500., 0, 320.], [0, 500., 240.], [0, 0, 1.]]),
                                       np.zeros(5), (640, 480))
        self.layouts = {'single28': BandLayout('single28', 'DICT_4X4_50', {28: square_object_points(.048)})}
        self.detections = [{28: np.array([[300., 210.], [360., 210.], [360., 270.], [300., 270.]])}
                           for _ in range(self.count)]
        self.poses = [None] * self.count
        self.confidences = [0.] * self.count
        self.accepted = [()] * self.count
        self.components = [None] * self.count
        self.weights = [{28: .995 if i % 2 else 1.} for i in range(self.count)]
        self.calls = []

    def runner(self, project, sequence, settings, output, **kwargs):
        self.calls.append((sequence, settings.read_text(), kwargs))
        timestamps = [float(line.split()[0]) for line in (sequence / 'rgb.txt').read_text().splitlines()
                      if line.strip() and not line.startswith('#')]
        rows = [dict(timestamp=i / self.fps, state=2 if i < self.tracked_count else 1,
                     pose=[0, 0, 0, 0, 0, 0, 1] if i < self.tracked_count else None,
                     reference=i // 3, maps=[{'metric': False}], final=False)
                for i in (round(timestamp * self.fps) for timestamp in timestamps)]
        rows.append(dict(timestamp=timestamps[-1], maps=[{'metric': False}], final=True, references=[]))
        (output / 'frames.txt.history.jsonl').write_text('\n'.join(json.dumps(row) for row in rows))

    def run_bootstrap(self, **kwargs):
        return bootstrap_initial_marker_observations(
            self.root, self.sequence, self.work, self.calibration, self.fps,
            self.detections, self.poses, self.confidences, self.accepted,
            self.weights, self.layouts, self.components, probe_runner=self.runner, **kwargs)

    @staticmethod
    def camera(frame, final, references):
        return ((Pose(np.zeros((3, 1)), np.zeros((3, 1)), 0.), 'atlas_0')
                if frame['state'] == 2 and frame['pose'] else (None, None))

    def fitted(self, views, calibration, size, marker_id, **kwargs):
        self.assertTrue(kwargs['trajectory_is_independent'])
        return MultiViewMarkerFit(
            True, 'accepted', marker_id, 2.4,
            camera_in_marker={v.frame_id: Pose(np.zeros((3, 1)),
                np.array([[v.frame_id * .001], [0.], [-.4]]), .3) for v in views},
            diagnostics={'evidence_end_s': max(v.timestamp for v in views)})

    @staticmethod
    def refit(view, fit, calibration, size):
        # Deliberately different six-DoF result: only its pixel error may be
        # used; native initialization must use the common multiview solution.
        return Pose(np.array([[.4], [0.], [0.]]), np.array([[9.], [-8.], [7.]]), .2,
                    marker_ids=(28,), inlier_count=4)

    def assert_measurement_streams_unchanged(self, result):
        for field, original in (('poses', self.poses), ('confidences', self.confidences),
                                ('accepted_marker_ids', self.accepted),
                                ('component_ids', self.components)):
            self.assertEqual(getattr(result, field), original)
            self.assertIsNot(getattr(result, field), original)

    def test_probe_is_marker_free_and_hints_do_not_mutate_cache_or_extrapolate(self):
        self.poses[18] = Pose(np.zeros((3, 1)), np.array([[1.], [2.], [3.]]), .8)
        self.confidences[18], self.accepted[18], self.components[18] = .2, (29,), 'existing'
        for index, detection in enumerate(self.detections):
            detection[28] = detection[28] + [index, 0.]
        original_corners = [d[28].copy() for d in self.detections]
        original_weights = [dict(weights) for weights in self.weights]

        def refit(view, fit, calibration, size):
            if view.frame_id == 4:
                return None  # failed real-corner quality check must remain invalid
            return self.refit(view, fit, calibration, size)

        with patch('aruco_track.marker_bootstrap.camera_at_revision', side_effect=self.camera), \
             patch('aruco_track.marker_bootstrap.fit_marker_multiview', side_effect=self.fitted), \
             patch('aruco_track.marker_bootstrap.refit_marker_view', side_effect=refit):
            result = self.run_bootstrap()
        self.assertEqual(len(self.calls), 1)
        _, settings, arguments = self.calls[0]
        self.assertIn('TagFusion.enabled: 0', settings)
        self.assertIn('loopClosing: 0', settings)
        self.assertIsNone(arguments['tag_observations_path'])
        self.assertEqual(arguments['environment_overrides']['ORB_SLAM3_DIAGNOSTIC_NO_FINAL_OPTIMIZATION'], '1')
        self.assertEqual(arguments['environment_overrides']['ORB_SLAM3_OFFLINE_LOOP_SEARCH'], '0')
        self.assertEqual(arguments['environment_overrides']['ORB_SLAM3_INCREMENTAL_LOOP_SEARCH'], '0')
        self.assertEqual(arguments['environment_overrides']['ORB_SLAM3_MARKER_SIM3_LOOP'], '0')
        self.assertTrue(result.diagnostics['accepted'])
        self.assertIsNotNone(result.hints_path)
        self.assertNotEqual(result.hints_path, self.sequence / 'tag_observations.txt')
        self.assertEqual((self.sequence / 'tag_observations.txt').read_bytes(), self.original_hints)
        self.assertEqual((self.sequence / 'rgb.txt').read_bytes(), self.original_rgb)
        self.assert_measurement_streams_unchanged(result)
        self.assertTrue(all(pose is None for pose in result.poses[:18]))
        np.testing.assert_array_equal(self.poses[18].tvec, [[1.], [2.], [3.]])
        self.assertEqual(self.weights, original_weights)
        expected = set(range(10)) - {4}
        published = result.diagnostics['published_observations']
        self.assertEqual({row['frame_id'] for row in published}, expected)
        hints = {round(float(line.split()[0]) * self.fps): line.split()
                 for line in result.hints_path.read_text().splitlines() if not line.startswith('#')}
        original_lines = self.original_hints.decode().splitlines()[1:]
        for index in range(self.count):
            np.testing.assert_array_equal(self.detections[index][28], original_corners[index])
            values = hints[index]
            if index not in expected:
                self.assertEqual(' '.join(values), original_lines[index])
                continue
            self.assertEqual(values[1], '1')
            np.testing.assert_allclose(np.asarray(values[3:6], float), [index * .001, 0., -.4])
            np.testing.assert_allclose(np.asarray(values[6:10], float), [0., 0., 0., 1.], atol=1e-12)
            self.assertEqual(values[10], '4')
            corners = np.asarray(values[11:31], float).reshape(4, 5)
            np.testing.assert_allclose(corners[:, :3], self.layouts['single28'].markers[28])
            np.testing.assert_allclose(corners[:, 3:], original_corners[index])
            offset = values.index('weights')
            np.testing.assert_allclose(np.asarray(values[offset+1:offset+5], float),
                                       [original_weights[index][28]] * 4)
            self.assertEqual(values[values.index('ids')+1:values.index('ids')+5], ['28'] * 4)
        for row in result.diagnostics['published_observations']:
            self.assertEqual(row['available_after_s'], 1.9)
            self.assertLess(row['observation_timestamp_s'], row['available_after_s'])
            self.assertEqual(row['kind'], 'multiview_initialization_hint')
            self.assertEqual(row['evidence_source'], 'original_four_corners_with_joint_initial_guess')
            self.assertEqual(row['single_frame_fit_rms_px'], .2)
            self.assertEqual(row['joint_hint_reprojection_px'], .3)
            self.assertTrue(row['no_added_pose_factors'])
        self.assertFalse(result.diagnostics['causal_online'])
        self.assertFalse(result.diagnostics['historical_measurement_state_rewritten'])
        self.assertFalse(result.diagnostics['independent_single_frame_pnp'])
        self.assertTrue(result.diagnostics['single_frame_refit_used_only_for_pixel_quality'])
        self.assertTrue(result.diagnostics['native_metric_commit_required'])
        self.assertFalse(result.diagnostics['probe_trajectory_used_as_optimization_factor'])

    def test_only_same_frame_strong_corners_with_a_tracked_camera_supply_hints(self):
        self.tracked_count = 16
        self.weights[1] = {28: .25}
        self.detections[2] = {}
        self.detections[3][28][0, 0] = np.nan
        self.detections[4][28] = np.zeros((3, 2))
        expected = set(range(16)) - {1, 2, 3, 4}

        def fitted(views, *args, **kwargs):
            self.assertEqual({v.frame_id for v in views}, expected)
            result = self.fitted(views, *args, **kwargs)
            # Extra joint poses must not fill weak, missing or untracked frames.
            for index in range(self.count):
                result.camera_in_marker.setdefault(index, Pose(np.zeros((3, 1)), np.zeros((3, 1)), .3))
            return result

        with patch('aruco_track.marker_bootstrap.camera_at_revision', side_effect=self.camera), \
             patch('aruco_track.marker_bootstrap.fit_marker_multiview', side_effect=fitted), \
             patch('aruco_track.marker_bootstrap.refit_marker_view', side_effect=self.refit) as refit:
            result = self.run_bootstrap()
        self.assertTrue(result.diagnostics['accepted'])
        self.assertEqual({row['frame_id'] for row in result.diagnostics['published_observations']}, expected)
        self.assertEqual({call.args[0].frame_id for call in refit.call_args_list}, expected)
        self.assert_measurement_streams_unchanged(result)

    def test_missing_or_bad_joint_initialization_does_not_use_the_independent_pnp_pose(self):
        for rejected_error in (None, 2.5001, float('nan')):
            with self.subTest(joint_error=rejected_error):
                def fitted(views, *args, **kwargs):
                    result = self.fitted(views, *args, **kwargs)
                    if rejected_error is None:
                        del result.camera_in_marker[4]
                    else:
                        result.camera_in_marker[4].reprojection_error_px = rejected_error
                    return result

                with patch('aruco_track.marker_bootstrap.camera_at_revision', side_effect=self.camera), \
                     patch('aruco_track.marker_bootstrap.fit_marker_multiview', side_effect=fitted), \
                     patch('aruco_track.marker_bootstrap.refit_marker_view', side_effect=self.refit):
                    result = self.run_bootstrap()
                self.assertTrue(result.diagnostics['accepted'])
                self.assertEqual({row['frame_id'] for row in result.diagnostics['published_observations']},
                                 set(range(10)) - {4})
                self.assert_measurement_streams_unchanged(result)

    def test_fewer_than_eight_valid_joint_hints_do_not_publish_a_candidate(self):
        def fitted(views, *args, **kwargs):
            result = self.fitted(views, *args, **kwargs)
            result.camera_in_marker = {i: pose for i, pose in result.camera_in_marker.items() if i < 7}
            return result

        with patch('aruco_track.marker_bootstrap.camera_at_revision', side_effect=self.camera), \
             patch('aruco_track.marker_bootstrap.fit_marker_multiview', side_effect=fitted), \
             patch('aruco_track.marker_bootstrap.refit_marker_view', side_effect=self.refit):
            result = self.run_bootstrap()
        self.assertFalse(result.diagnostics['accepted'])
        self.assertEqual(result.diagnostics['reason'], 'no_validated_initialization_hints')
        self.assertIsNone(result.hints_path)
        self.assertFalse(result.diagnostics['published_observations'])
        self.assert_measurement_streams_unchanged(result)

    def test_unobserved_tail_does_not_extend_the_probe_or_final_gauge(self):
        # The cached video continues for one second after the evidence window.
        # Later weak-only detections must not extend that window either.
        for index in range(10, self.count):
            self.weights[index] = {28: .25}
        final_timestamps = []

        def camera(frame, final, references):
            final_timestamps.append(final['timestamp'])
            return self.camera(frame, final, references)

        with patch('aruco_track.marker_bootstrap.camera_at_revision', side_effect=camera), \
             patch('aruco_track.marker_bootstrap.fit_marker_multiview', side_effect=self.fitted), \
             patch('aruco_track.marker_bootstrap.refit_marker_view', side_effect=self.refit):
            result = self.run_bootstrap()
        self.assertTrue(result.diagnostics['accepted'])
        self.assertEqual(result.diagnostics['probe_frames'], 10)
        self.assertEqual(result.diagnostics['available_after_s'], .9)
        self.assertEqual(set(final_timestamps), {.9})
        self.assertEqual({row['frame_id'] for row in result.diagnostics['published_observations']}, set(range(10)))
        self.assertTrue(all(row['available_after_s'] == .9
                            for row in result.diagnostics['published_observations']))
        probe_sequence = self.calls[0][0]
        probe_times = [float(line.split()[0]) for line in (probe_sequence / 'rgb.txt').read_text().splitlines()]
        self.assertEqual(probe_times, [index / self.fps for index in range(10)])
        self.assertEqual((self.sequence / 'rgb.txt').read_bytes(), self.original_rgb)
        self.assertEqual((self.sequence / 'tag_observations.txt').read_bytes(), self.original_hints)
        self.assert_measurement_streams_unchanged(result)

    def test_existing_reliable_initial_marker_skips_probe(self):
        self.poses[1] = Pose(np.zeros((3, 1)), np.zeros((3, 1)), .2)
        self.confidences[1] = .8
        result = self.run_bootstrap()
        self.assertEqual(result.diagnostics['reason'], 'initial_reliable_marker_already_available')
        self.assertFalse(self.calls)
        self.assertIsNone(result.hints_path)

    def test_weak_only_evidence_and_loaded_atlas_do_not_start_probe(self):
        self.weights = [{28: .25}] * self.count
        result = self.run_bootstrap()
        self.assertEqual(result.diagnostics['reason'], 'insufficient_initial_strong_singleton_observations')
        self.assertFalse(self.calls)
        result = self.run_bootstrap(load_atlas=Path('existing.osa'))
        self.assertEqual(result.diagnostics['reason'], 'atlas_load_or_marker_only_initialization')
        self.assertFalse(self.calls)

    def test_marker_constrained_probe_cannot_validate_target(self):
        original_runner = self.runner

        def contaminated(*args, **kwargs):
            original_runner(*args, **kwargs)
            history = args[3] / 'frames.txt.history.jsonl'
            rows = [json.loads(line) for line in history.read_text().splitlines()]
            rows[0]['marker_factor_eligible'] = True
            history.write_text('\n'.join(json.dumps(row) for row in rows))

        self.runner = contaminated
        with patch('aruco_track.marker_bootstrap.fit_marker_multiview') as fit:
            result = self.run_bootstrap()
        self.assertEqual(result.diagnostics['reason'], 'probe_trajectory_not_marker_independent')
        fit.assert_not_called()
        self.assertIsNone(result.hints_path)
        self.assertEqual((self.sequence / 'tag_observations.txt').read_bytes(), self.original_hints)


if __name__ == '__main__':
    unittest.main()
