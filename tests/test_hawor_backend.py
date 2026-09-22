import json
from pathlib import Path
import tempfile
import unittest
import sys
from unittest.mock import patch, MagicMock

import numpy as np

from aruco_track.hawor_backend import HaworHandTracker, _read_predictions, _effective_calibration, _stamp, hawor_policy, prepare_hawor_predictions
import cv2
from aruco_track.hands import joint_pose_to_dict, raw_hand_from_dict
from aruco_track.models import Calibration, Pose
from tools.export_action_labels import _load_observation_cache, _prepare_export_hawor_predictions
from tests import test_cached_observations


class HaworBackendTests(unittest.TestCase):
    def test_local_runner_keeps_virtualenv_python_symlink(self):
        root=Path(self.temp.name); interpreter=root/'venv/bin/python'
        interpreter.parent.mkdir(parents=True);interpreter.symlink_to(Path(sys.executable).resolve())
        repo=root/'repo';(repo/'_DATA/data').mkdir(parents=True)
        (repo/'_DATA/data/mano_mean_params.npz').write_bytes(b'mean')
        mano=root/'mano';mano.mkdir()
        for side in ['LEFT','RIGHT']:(mano/f'MANO_{side}.pkl').write_bytes(b'model')
        resources=dict(python=str(interpreter),repo=str(repo),mano_dir=str(mano),execution='local')
        for name in ['checkpoint','model_config','detector']:
            p=root/name;p.write_bytes(b'fixture');resources[name]=str(p)
        config=root/'runtime.json';config.write_text(json.dumps(resources))
        video=root/'clip.mp4';video.write_bytes(b'video')
        calib=root/'camera.json';self.calibration.save(calib)
        capture=MagicMock();capture.get.side_effect=lambda k:{cv2.CAP_PROP_FRAME_WIDTH:640,
            cv2.CAP_PROP_FRAME_HEIGHT:480,cv2.CAP_PROP_FRAME_COUNT:2,cv2.CAP_PROP_FPS:30}[k]
        with patch('aruco_track.hawor_backend.cv2.VideoCapture',return_value=capture), \
             patch('aruco_track.hawor_remote._engine_hashes',return_value={'lib/models/hawor.py':'abc'}), \
             patch('aruco_track.hawor_backend.subprocess.run',side_effect=RuntimeError('captured')) as run:
            with self.assertRaisesRegex(RuntimeError,'captured'):
                prepare_hawor_predictions(video,calib,root/'new.jsonl',config_path=config)
        self.assertEqual(run.call_args[0][0][0],str(interpreter))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'predictions.jsonl'
        self.calibration = Calibration(np.array([[500., 0, 320], [0, 500, 240], [0, 0, 1]]),
                                       np.zeros(5), (640, 480))
        self.pose = Pose(np.zeros((3, 1)), np.array([[0.], [0.], [1.]]), .2)

    def write(self, support=(True, True)):
        uv = np.tile([.5, .42], (21, 1)); uv[0] = [.5, .5]
        xyz = np.c_[np.zeros(21), np.arange(21)*.002, np.zeros(21)]
        rows = [dict(frame=i, timestamp_s=i/30, hands=[dict(
            handedness='Left', detection_score=.8, image_landmarks_normalized=uv.tolist(),
            model_landmarks_m=xyz.tolist(), detector_supported=ok, image_supported=ok,
            future_frames_used=1-i)]) for i, ok in enumerate(support)]
        self.path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        return rows

    def test_actual_support_and_two_frame_confirmation(self):
        self.write()
        tracker = HaworHandTracker(self.path, self.calibration, ['strap_band_L'], 30)
        self.assertNotIn('strap_band_L', tracker.process(None, 0, {'strap_band_L':self.pose}))
        joint = tracker.process(None, 33, {'strap_band_L':self.pose})['strap_band_L']
        value = joint_pose_to_dict(joint)
        self.assertIsNone(value['handedness_score'])
        self.assertIsNone(value['wrist_association_confidence'])
        self.assertEqual(value['detector_confidence'], .8)
        self.assertEqual(value['prediction_backend'], 'hawor')
        self.assertTrue(value['image_supported'])
        self.assertTrue(value['wrist_anchor_valid'])
        self.assertIsNone(raw_hand_from_dict(value).handedness_score)
        json.dumps(value, allow_nan=False)

    def test_unsupported_pose_is_not_output_or_used_for_confirmation(self):
        self.write((False, True))
        tracker = HaworHandTracker(self.path, self.calibration, ['strap_band_L'], 30)
        self.assertEqual(tracker.process(None, 0, {'strap_band_L':self.pose}), {})
        self.assertNotIn('strap_band_L', tracker.process(None, 33, {'strap_band_L':self.pose}))

    def test_missing_wrist_has_no_camera_or_world_hand(self):
        self.write()
        tracker = HaworHandTracker(self.path, self.calibration, ['strap_band_L'], 30)
        values = tracker.process(None, 0, {})
        self.assertTrue(values)
        for joint in values.values():
            self.assertIsNone(joint.camera_landmarks_m)
            self.assertIsNone(joint.world_landmarks_m)

    def test_frame_gap_and_corrupt_coordinates_fail_explicitly(self):
        rows = self.write(); rows[1]['frame'] = 4
        self.path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        with self.assertRaises(ValueError):
            _read_predictions(self.path, 30, expected_count=2, expected_start=0)

    def test_hand_calibration_uses_exporter_resize_without_mutating_source(self):
        source = Path(self.temp.name)/'camera.json'
        self.calibration.save(source)
        original_bytes = source.read_bytes()
        capture = MagicMock()
        capture.get.side_effect = lambda key: {cv2.CAP_PROP_FRAME_WIDTH:320,
                                               cv2.CAP_PROP_FRAME_HEIGHT:240}[key]
        with patch('aruco_track.hawor_backend.cv2.VideoCapture', return_value=capture):
            result = _effective_calibration('video.mp4', source, self.path)
            repeated = _effective_calibration('video.mp4', source, self.path)
        self.assertEqual(result, repeated)
        self.assertEqual(source.read_bytes(), original_bytes)
        derived = Calibration.load(result)
        self.assertEqual(derived.image_size, (320, 240))
        np.testing.assert_array_equal(derived.camera_matrix, self.calibration.scaled_to((320,240)).camera_matrix)

    def test_hand_calibration_rejects_crop_before_network(self):
        source = Path(self.temp.name)/'camera.json'
        self.calibration.save(source)
        capture = MagicMock()
        capture.get.side_effect = lambda key: {cv2.CAP_PROP_FRAME_WIDTH:320,
                                               cv2.CAP_PROP_FRAME_HEIGHT:320}[key]
        with patch('aruco_track.hawor_backend.cv2.VideoCapture', return_value=capture):
            with self.assertRaisesRegex(ValueError, 'aspect ratio changed'):
                _effective_calibration('video.mp4', source, self.path)

    def test_backend_upgrade_keeps_marker_cache_without_accepting_stale_hands(self):
        cache, inputs, original = test_cached_observations.CachedObservationTests().fixture(Path(self.temp.name))
        config = Path(self.temp.name)/'hawor.json'; config.write_text('{}')
        rows, meta = _load_observation_cache(cache, inputs[0], inputs[1], [inputs[2]], inputs[3],
                                            60, (640,480), config, .4,
                                            allow_hand_upgrade=True, hand_backend='hawor')
        self.assertEqual(rows, [original])
        self.assertEqual(meta['hand_model'], str(inputs[4]))
        with self.assertRaisesRegex(ValueError, 'hand model'):
            _load_observation_cache(cache, inputs[0], inputs[1], [inputs[2]], inputs[3],
                                    60, (640,480), config, .4, hand_backend='hawor')


class HaworExportCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name).resolve()
        self.previous = root/'previous.hawor_observations.jsonl'
        self.previous.write_text('{"frame":0,"timestamp_s":0,"hands":[]}\n')
        metrics = self.previous.with_suffix('.metrics.json')
        metrics.write_text('{"status":"complete"}\n')
        self.provenance = dict(backend='hawor', execution='ssh', signature={'frames':1},
                               prediction_file=_stamp(self.previous, True),
                               metrics_file=_stamp(metrics, True))
        self.previous.with_suffix('.meta.json').write_text(json.dumps(self.provenance))
        self.metadata = dict(hand_backend='hawor', hand_joints_enabled=True,
                             hand_recovery_policy=hawor_policy(),
                             hand_backend_provenance=self.provenance)
        self.output = root/'new.hawor_observations.jsonl'

    def prepare(self):
        return _prepare_export_hawor_predictions(
            'video.mp4', 'camera.json', self.output, max_frames=1,
            config_path='runtime.json', device='auto', cached_metadata=self.metadata)

    def test_same_policy_reuses_original_path_through_strict_prepare(self):
        with patch('tools.export_action_labels.prepare_hawor_predictions',
                   return_value=(self.previous, self.provenance)) as prepare:
            self.assertEqual(self.prepare(), (self.previous, self.provenance))
        prepare.assert_called_once_with('video.mp4', 'camera.json', self.previous,
                                        max_frames=1, config_path='runtime.json', device='auto')
        self.assertFalse(self.output.exists())

    def test_only_signature_change_prepares_new_output_for_each_backend(self):
        messages = ['HaWoR cache differs or is unverified; refusing to overwrite it',
                    f'HaWoR observation cache is unverified or different: {self.previous}; choose a new output path']
        for message in messages:
            with self.subTest(message=message), patch('tools.export_action_labels.prepare_hawor_predictions',
                    side_effect=[ValueError(message), (self.output, {})]) as prepare:
                self.assertEqual(self.prepare(), (self.output, {}))
                self.assertEqual([call.args[2] for call in prepare.call_args_list],
                                 [self.previous, self.output])

    def test_network_and_other_validation_errors_do_not_trigger_another_inference(self):
        for error in (RuntimeError('SSH connection failed'), ValueError('incomplete HaWoR prediction cache')):
            with self.subTest(error=error), patch('tools.export_action_labels.prepare_hawor_predictions',
                                                   side_effect=error) as prepare:
                with self.assertRaises(type(error)) as raised:
                    self.prepare()
                self.assertIs(raised.exception, error)
                prepare.assert_called_once()

    def test_prediction_corruption_is_not_treated_as_signature_change(self):
        self.previous.write_text('changed raw predictions')
        with patch('tools.export_action_labels.prepare_hawor_predictions') as prepare:
            with self.assertRaisesRegex(ValueError, 'raw-cache content changed'):
                self.prepare()
            prepare.assert_not_called()

    def test_metrics_corruption_is_not_treated_as_signature_change(self):
        self.previous.with_suffix('.metrics.json').write_text('changed metrics')
        with patch('tools.export_action_labels.prepare_hawor_predictions') as prepare:
            with self.assertRaisesRegex(ValueError, 'raw-cache content changed'):
                self.prepare()
            prepare.assert_not_called()

    def test_provenance_mismatch_fails_explicitly(self):
        self.previous.with_suffix('.meta.json').write_text('{}')
        with patch('tools.export_action_labels.prepare_hawor_predictions') as prepare:
            with self.assertRaisesRegex(ValueError, 'metadata differs'):
                self.prepare()
            prepare.assert_not_called()

    def test_missing_raw_path_prepares_new_output(self):
        self.previous.unlink()
        with patch('tools.export_action_labels.prepare_hawor_predictions', return_value=(self.output, {})) as prepare:
            self.prepare()
            self.assertEqual(prepare.call_args.args[2], self.output)
            prepare.assert_called_once()

    def test_changed_policy_does_not_reuse_prior_hawor_cache(self):
        self.metadata['hand_recovery_policy'] = {'version':'old-policy'}
        with patch('tools.export_action_labels.prepare_hawor_predictions', return_value=(self.output, {})) as prepare:
            self.prepare()
            self.assertEqual(prepare.call_args.args[2], self.output)
            prepare.assert_called_once()


if __name__ == '__main__':
    unittest.main()
