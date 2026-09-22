import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from aruco_track.hands import HandJointPose
from aruco_track.hand_recovery import hand_recovery_policy
from aruco_track.models import Calibration
from tools.export_action_labels import _hand_cache_requires_upgrade, _cached_raw_hands
from tools.refresh_hand_labels import frozen_fields, refresh_record
from tools.refresh_hand_labels import main as refresh_main


def pose(t):
    return dict(translation_m=t, quaternion_wxyz=[1, 0, 0, 0])


class CachedTracker:
    def process_observations(self, frame, timestamp, raw, poses, protected_assignments=None):
        self.protected = protected_assignments
        return {name: HandJointPose(name, hand.handedness, hand.handedness_score,
                                   hand.image_landmarks_normalized,
                                   hand.model_landmarks_m, {})
                for name, hand in protected_assignments.items()}


class PredictedTracker:
    backend = 'hawor'
    recovery_diagnostics = {}

    def __init__(self, joints=None):
        self.joints = joints or {}

    def process_observations(self, frame, timestamp, raw, poses, protected_assignments=None):
        self.raw, self.protected = raw, protected_assignments
        return self.joints

    def process(self, frame, timestamp, poses):
        self.redetected = True
        return self.joints

    def close(self):
        pass


class RefreshHandTests(unittest.TestCase):
    def setUp(self):
        self.calibration = Calibration(np.array([[400., 0, 320], [0, 400, 240], [0, 0, 1]]),
                                       np.zeros(5), (640, 480))
        self.joints = dict(valid=True, handedness='Left', handedness_score=.9,
                           image_landmarks_normalized=[[.5, .5, 0]] * 21,
                           model_landmarks_m=[[0., index * .001, 0.] for index in range(21)],
                           world_landmarks_graph_m=[[999., 0., 0.]] * 21)
        self.row = dict(frame=0, timestamp_s=0, camera_world_source='head-slam',
                        scale_status='metric', camera_submap_id='atlas_0',
                        camera_world_pose_fused=pose([10, 0, 0]),
                        detected_marker_corners={'20': [[1, 2]] * 4},
                        hands={'left': dict(joints=self.joints, world_submap_id='atlas_0',
                                             wrist_camera_graph=pose([0, 0, 1]),
                                             wrist_world_graph=pose([10, 0, 1]),
                                             wrist_world_tracking=pose([10, .1, 1]),
                                             wrist_world_anchor_verified=None)},
                        unassigned_hands=[])

    def test_old_enabled_cache_is_automatically_upgraded(self):
        self.assertTrue(_hand_cache_requires_upgrade({'hand_joints_enabled': True}))
        self.assertTrue(_hand_cache_requires_upgrade({'hand_joints_enabled': False}))
        self.assertFalse(_hand_cache_requires_upgrade({}, enabled=False))

    def test_matching_policy_reuses_candidates_but_changed_policy_upgrades(self):
        metadata = dict(hand_joints_enabled=True, hand_recovery_policy=hand_recovery_policy())
        self.assertFalse(_hand_cache_requires_upgrade(metadata))
        self.assertTrue(_hand_cache_requires_upgrade(metadata, min_confidence=.7))
        metadata['hand_recovery_policy']['version'] = 'old'
        self.assertTrue(_hand_cache_requires_upgrade(metadata))

    def test_frozen_slam_and_source_unchanged_and_no_second_scale(self):
        original = copy.deepcopy(self.row)
        result = refresh_record(self.row, None, CachedTracker(), self.calibration)
        self.assertEqual(self.row, original)
        self.assertEqual(frozen_fields(result), frozen_fields(original))
        joints = result['hands']['left']['joints']
        self.assertEqual(joints['model_landmarks_m'], original['hands']['left']['joints']['model_landmarks_m'])
        np.testing.assert_allclose(joints['world_landmarks_graph_m'][0], [10, 0, 1])
        np.testing.assert_allclose(joints['world_landmarks_tracking_m'][0], [10, .1, 1])
        np.testing.assert_allclose(joints['world_landmarks_graph_m'][20], [10, .020, 1])

    def test_missing_final_wrist_clears_stale_world_but_keeps_image_measurement(self):
        hand = self.row['hands']['left']
        hand.update(wrist_camera_graph=None, wrist_world_graph=None, wrist_world_tracking=None)
        result = refresh_record(self.row, None, CachedTracker(), self.calibration)
        joints = result['hands']['left']['joints']
        self.assertTrue(joints['valid'])
        self.assertIsNone(joints['camera_landmarks_m'])
        self.assertIsNone(joints['world_landmarks_graph_m'])
        self.assertIsNone(joints['world_landmarks_tracking_m'])

    def test_invalid_camera_or_unaligned_map_does_not_create_world_joints(self):
        for change in ({'camera_world_source': 'invalid'}, {'scale_status': 'arbitrary-scale'},
                       {'camera_submap_id': 'atlas_1'}):
            row = {**self.row, **change}
            result = refresh_record(row, None, CachedTracker(), self.calibration)
            self.assertIsNone(result['hands']['left']['joints']['world_landmarks_graph_m'])

    def test_cache_reuse_preserves_roi_provenance(self):
        self.row['hands']['left']['joints']['detection_source'] = 'wrist_roi'
        raw = _cached_raw_hands(self.row)
        self.assertEqual(raw[0].detection_source, 'wrist_roi')

    def test_hawor_does_not_protect_or_retain_old_mediapipe_labels(self):
        tracker = PredictedTracker()
        result = refresh_record(self.row, None, tracker, self.calibration)
        self.assertEqual(tracker.raw, [])
        self.assertEqual(tracker.protected, {})
        self.assertFalse(result['hands']['left']['joints']['valid'])
        self.assertEqual(frozen_fields(result), frozen_fields(self.row))

    def test_hawor_new_vectors_use_frozen_wrist_not_previous_model_or_predicted_depth(self):
        model = np.array([[0., i * .002, 0.] for i in range(21)])
        tracker = PredictedTracker({'left': HandJointPose(
            'left', 'Left', None, np.array(self.joints['image_landmarks_normalized']), model, {})})
        result = refresh_record(self.row, None, tracker, self.calibration)
        self.assertIsNone(result['hands']['left']['joints']['handedness_score'])
        np.testing.assert_allclose(result['hands']['left']['joints']['world_landmarks_graph_m'][20],
                                   [10, .040, 1])
        self.row['hands']['left']['wrist_camera_graph'] = None
        self.row['hands']['left']['wrist_world_graph'] = None
        self.row['hands']['left']['wrist_world_tracking'] = None
        result = refresh_record(self.row, None, tracker, self.calibration)
        self.assertIsNone(result['hands']['left']['joints']['world_landmarks_graph_m'])

    def test_backend_switch_to_mediapipe_runs_fresh_detection(self):
        tracker = PredictedTracker()
        tracker.backend = 'mediapipe'
        result = refresh_record(self.row, None, tracker, self.calibration, redetect=True)
        self.assertTrue(tracker.redetected)
        self.assertFalse(result['hands']['left']['joints']['valid'])

    def test_default_refresh_uses_hawor_and_updates_model_contract_without_decoding_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.jsonl'
            output = Path(directory) / 'new' / 'actions.jsonl'
            source.write_text(json.dumps(self.row) + '\n')
            config = Path(directory) / 'hawor.json'
            config.write_text('{}')
            old_inputs = {'video': 'source-pixels', 'hand_model': 'old-mediapipe'}
            new_inputs = {'video': 'source-pixels', 'hand_model': 'new-hawor'}
            metadata = dict(video='raw.mp4', calibration='calib.json', bands=['left.json'],
                            hand_model='old.task', frames=1, fps=90., hand_joints_enabled=True,
                            replay={'directory': 'source_replay'},
                            quality_control={'valid': True},
                            observation_cache_contract={'input_fingerprints': old_inputs})
            source.with_suffix('.meta.json').write_text(json.dumps(metadata))
            tracker = PredictedTracker()
            with patch('sys.argv', ['tools/refresh_hand_labels.py', str(source), '--output', str(output),
                                    '--hawor-config', str(config)]), \
                    patch('tools.refresh_hand_labels._observation_input_fingerprints',
                          side_effect=[old_inputs, new_inputs]), \
                    patch('tools.refresh_hand_labels.load_replay_calibration', return_value=self.calibration), \
                    patch('tools.refresh_hand_labels.BandLayout.load') as layout, \
                    patch('tools.refresh_hand_labels.prepare_hawor_predictions',
                          return_value=(Path('predictions.jsonl'), {'device': 'mps'})) as prepare, \
                    patch('tools.refresh_hand_labels.HaworHandTracker', return_value=tracker), \
                    patch('tools.refresh_hand_labels.cv2.VideoCapture') as capture:
                layout.return_value.name = 'left'
                refresh_main()
            capture.assert_not_called()
            self.assertEqual(prepare.call_args.kwargs['max_frames'], 1)
            self.assertNotEqual(prepare.call_args.args[2], output)
            actual = json.loads(output.with_suffix('.meta.json').read_text())
            self.assertEqual(actual['hand_backend'], 'hawor')
            self.assertEqual(actual['hand_model'], str(config.resolve()))
            self.assertEqual(actual['observation_cache_contract']['input_fingerprints'], new_inputs)
            self.assertIsNone(actual['replay'])
            self.assertTrue(actual['quality_control']['hand_refresh_requires_reassessment'])
            self.assertEqual(frozen_fields(json.loads(output.read_text())), frozen_fields(self.row))


if __name__ == '__main__':
    unittest.main()
