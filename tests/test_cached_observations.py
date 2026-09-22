import json
import os
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from aruco_track.models import Calibration, Pose
from aruco_track.tag_graph import TagPoseResult
from tools.export_action_labels import (
    OBSERVATION_ALGORITHM_VERSION,
    OBSERVATION_CACHE_SCHEMA,
    _cached_tag_result,
    _final_resolved_map,
    _load_observation_cache,
    _observation_input_fingerprints,
    _rebind_cached_joints,
    _reassign_cached_joints,
    _replace_cached_graph_measurement,
    _sample_marker_map_frames,
    pose_to_dict,
)


class CachedObservationTests(unittest.TestCase):
    def test_disabled_hands_do_not_require_model_fingerprint(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache, inputs, _ = self.fixture(Path(temporary))
            inputs[4].unlink()
            result = _observation_input_fingerprints(
                inputs[0], inputs[1], [inputs[2]], inputs[3], inputs[4],
                hand_joints=False,
            )
            self.assertIsNone(result['hand_model'])
            with self.assertRaises(FileNotFoundError):
                _observation_input_fingerprints(
                    inputs[0], inputs[1], [inputs[2]], inputs[3], inputs[4],
                )

    def test_legacy_marker_only_cache_needs_no_unused_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache, inputs, record = self.fixture(Path(temporary))
            metadata_path = cache.with_suffix('.meta.json')
            metadata = json.loads(metadata_path.read_text())
            metadata['hand_joints_enabled'] = False
            metadata_path.write_text(json.dumps(metadata))
            inputs[4].unlink()
            records, _ = _load_observation_cache(
                cache, inputs[0], inputs[1], [inputs[2]], inputs[3],
                60., (640, 480), inputs[4], .4, hand_joints=False,
            )
            self.assertEqual(records, [record])

    def fixture(self, root):
        inputs = [root/name for name in (
            'video.avi', 'calibration.json', 'band.json', 'board.json', 'hand.task')]
        for p in inputs:
            p.write_text('unchanged input')
            os.utime(p, (100, 100))
        cache = root/'actions.jsonl'
        record = dict(frame=0, timestamp_s=0, marker_camera_pose_observed=None,
                      marker_camera_confidence=0, detected_marker_corners={}, marker_boundary_quality={},
                      boundary_rejected_marker_corners={}, hands={'right': {'wrist_camera_graph': None, 'joints': {}}},
                      camera_world_pose_fused={'deliberately': 'old result, not an input measurement'})
        cache.write_text(json.dumps(record)+'\n')
        input_fingerprints = _observation_input_fingerprints(
            inputs[0], inputs[1], [inputs[2]], inputs[3], inputs[4]
        )
        cache.with_suffix('.meta.json').write_text(json.dumps(dict(video=str(inputs[0]),
            calibration=str(inputs[1]), bands=[str(inputs[2])], world_board=str(inputs[3]),
            hand_model=str(inputs[4]), min_hand_confidence=.4,
            schema='aruco-full-hand-actions/v3',
            observation_cache_contract={
                'schema': OBSERVATION_CACHE_SCHEMA,
                'algorithm_version': OBSERVATION_ALGORITHM_VERSION,
                'input_fingerprints': input_fingerprints,
            }, frames=1, fps=60., image_size=[640,480])))
        return cache, inputs, record

    def load(self, cache, inputs):
        return _load_observation_cache(
            cache, inputs[0], inputs[1], [inputs[2]], inputs[3], 60., (640,480),
            inputs[4], .4)

    def test_load_preserves_original_measurements_and_source_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache, inputs, record = self.fixture(Path(temporary))
            before = cache.read_bytes()
            records, _ = self.load(cache, inputs)
            self.assertEqual(records, [record])
            records[0]['hands'].clear()
            self.assertEqual(cache.read_bytes(), before)

    def test_mismatched_source_or_timing_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache, inputs, _ = self.fixture(Path(temporary))
            with self.assertRaisesRegex(ValueError, 'video does not match'):
                self.load(cache, [Path('other.avi'), *inputs[1:]])
            with self.assertRaisesRegex(ValueError, 'geometry/timing'):
                _load_observation_cache(
                    cache, inputs[0], inputs[1], [inputs[2]], inputs[3], 30.,
                    (640,480), inputs[4], .4)

    def test_cache_contract_hand_model_and_confidence_are_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache, inputs, _ = self.fixture(Path(temporary))
            metadata_path = cache.with_suffix('.meta.json')
            original = json.loads(metadata_path.read_text())
            variants = []
            missing = dict(original); missing.pop('observation_cache_contract')
            variants.append((missing, 'fresh analysis'))
            wrong_schema = json.loads(json.dumps(original))
            wrong_schema['observation_cache_contract']['schema'] = 'old-observations'
            variants.append((wrong_schema, 'fresh analysis'))
            wrong_version = json.loads(json.dumps(original))
            wrong_version['observation_cache_contract']['algorithm_version'] += 1
            variants.append((wrong_version, 'fresh analysis'))
            wrong_confidence = dict(original); wrong_confidence['min_hand_confidence'] = .7
            variants.append((wrong_confidence, 'hand confidence'))
            wrong_model = dict(original); wrong_model['hand_model'] = str(Path(temporary)/'other.task')
            variants.append((wrong_model, 'hand model'))
            for metadata, message in variants:
                metadata_path.write_text(json.dumps(metadata))
                with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                    self.load(cache, inputs)

    def test_hand_joint_mode_must_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache, inputs, _ = self.fixture(Path(temporary))
            with self.assertRaisesRegex(ValueError, 'hand-joint mode'):
                _load_observation_cache(
                    cache, inputs[0], inputs[1], [inputs[2]], inputs[3],
                    60., (640, 480), inputs[4], .4, hand_joints=False,
                )

    def test_marker_only_cache_can_be_upgraded_with_fresh_hands(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache, inputs, record = self.fixture(Path(temporary))
            metadata_path = cache.with_suffix('.meta.json')
            metadata = json.loads(metadata_path.read_text())
            metadata['hand_joints_enabled'] = False
            metadata_path.write_text(json.dumps(metadata))
            records, loaded = _load_observation_cache(
                cache, inputs[0], inputs[1], [inputs[2]], inputs[3],
                60., (640, 480), inputs[4], .4,
                hand_joints=True, allow_hand_upgrade=True,
            )
            self.assertEqual(records, [record])
            self.assertFalse(loaded['hand_joints_enabled'])

    def test_hand_cache_cannot_be_downgraded_via_upgrade_flag(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache, inputs, _ = self.fixture(Path(temporary))
            with self.assertRaisesRegex(ValueError, 'hand-joint mode'):
                _load_observation_cache(
                    cache, inputs[0], inputs[1], [inputs[2]], inputs[3],
                    60., (640, 480), inputs[4], .4,
                    hand_joints=False, allow_hand_upgrade=True,
                )

    def test_changed_configuration_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache, inputs, _ = self.fixture(Path(temporary))
            original = inputs[1].stat()
            inputs[1].write_text('changed configuration')
            os.utime(inputs[1], ns=(original.st_atime_ns, original.st_mtime_ns))
            with self.assertRaisesRegex(ValueError, 'input content changed'):
                self.load(cache, inputs)

    def test_incomplete_and_wrong_frame_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache, inputs, record = self.fixture(Path(temporary))
            cache.write_text('')
            with self.assertRaisesRegex(ValueError, 'empty or incomplete'):
                self.load(cache, inputs)
            record['frame'] = 2
            cache.write_text(json.dumps(record)+'\n')
            with self.assertRaisesRegex(ValueError, 'frame 0'):
                self.load(cache, inputs)

    def test_cached_hand_3d_is_rebound_to_current_wrist_pose(self):
        calibration = Calibration(
            np.array([[500., 0., 320.], [0., 500., 240.], [0., 0., 1.]]),
            np.zeros(5), (640, 480))
        wrist = Pose(np.zeros((3, 1)), np.array([[.1], [.02], [1.]]), 0.)
        world_to_camera = Pose(np.zeros((3, 1)), np.array([[0.], [0.], [.5]]), 0.)
        projected = cv2.projectPoints(
            np.zeros((1, 3)), wrist.rvec, wrist.tvec,
            calibration.camera_matrix, calibration.dist_coeffs)[0].reshape(2)
        model = np.zeros((21, 3), dtype=float)
        model[:, 0] = np.linspace(0., .1, 21)
        joints = {
            'valid': True,
            'image_landmarks_normalized': np.tile(projected / calibration.image_size, (21, 1)).tolist(),
            'model_landmarks_m': model.tolist(),
            'camera_landmarks_m': np.full((21, 3), 99.).tolist(),
            'band_landmarks_m': np.full((21, 3), 99.).tolist(),
            'world_landmarks_m': np.full((21, 3), 99.).tolist(),
            'wrist_anchor_valid': True,
            'wrist_anchor_error_px': 999.,
        }

        _rebind_cached_joints(joints, wrist, world_to_camera, calibration)

        np.testing.assert_allclose(joints['camera_landmarks_m'][0], wrist.tvec.ravel())
        np.testing.assert_allclose(joints['band_landmarks_m'][0], 0.)
        np.testing.assert_allclose(joints['world_landmarks_m'][0], [.1, .02, .5])
        self.assertTrue(joints['wrist_anchor_valid'])
        self.assertAlmostEqual(joints['wrist_anchor_error_px'], 0.)

        joints['image_landmarks_normalized'] = np.zeros((21, 2)).tolist()
        _rebind_cached_joints(joints, wrist, world_to_camera, calibration)
        self.assertIsNone(joints['camera_landmarks_m'])
        self.assertIsNone(joints['band_landmarks_m'])
        self.assertIsNone(joints['world_landmarks_m'])
        self.assertFalse(joints['wrist_anchor_valid'])

    def test_cached_wrist_pose_and_admission_are_replaced_not_reused(self):
        poisoned = Pose(
            np.zeros((3, 1)), np.full((3, 1), 99.), 99., (5,), 4, True
        )
        current = Pose(
            np.zeros((3, 1)), np.array([[.1], [.2], [1.]]), .4, (6, 7), 8, False
        )
        hand = {
            'wrist_camera_graph': {
                'translation_m': poisoned.tvec.ravel().tolist(),
            },
            'accepted_marker_ids': [5],
            'rejected_marker_ids': [6, 7],
        }
        result = TagPoseResult(current, (6, 7), (8,), {8: 12.}, .4, .9)

        _replace_cached_graph_measurement(hand, result)

        np.testing.assert_allclose(hand['wrist_camera_graph']['translation_m'], [.1, .2, 1.])
        self.assertEqual(hand['accepted_marker_ids'], [6, 7])
        self.assertEqual(hand['rejected_marker_ids'], [8])

    def test_versioned_cached_tag_measurement_round_trips(self):
        pose = Pose(
            np.array([[0.1], [-0.2], [0.3]]),
            np.array([[0.4], [0.5], [0.6]]),
            0.7,
            (1, 2),
            8,
            False,
        )
        hand = {
            'wrist_camera_graph': pose_to_dict(pose),
            'accepted_marker_ids': [1, 2],
            'rejected_marker_ids': [3],
            'marker_errors_px': {'1': 0.5, '2': 0.8},
            'graph_reprojection_error_px': 0.65,
            'confidence': 0.9,
        }

        restored = _cached_tag_result(hand)

        np.testing.assert_allclose(restored.pose.tvec, pose.tvec)
        np.testing.assert_allclose(
            restored.pose.rotation_matrix, pose.rotation_matrix, atol=1e-12
        )
        self.assertEqual(restored.accepted_marker_ids, (1, 2))
        self.assertEqual(restored.rejected_marker_ids, (3,))
        self.assertEqual(restored.marker_errors_px, {1: 0.5, 2: 0.8})

    def test_marker_map_sampling_preserves_brief_viable_pair(self):
        frames = [{} for _ in range(12)]
        for index in (1, 5, 7):
            frames[index] = {
                20: np.zeros((4, 2)),
                21: np.ones((4, 2)),
            }

        sampled, indices, stride = _sample_marker_map_frames(frames, 60.0)

        self.assertEqual(stride, 3)
        self.assertTrue({1, 5, 7}.issubset(indices))
        self.assertEqual(len(sampled), len(indices))

    def test_cached_hand_identity_is_reassigned_from_current_wrist_geometry(self):
        calibration = Calibration(
            np.array([[500., 0., 320.], [0., 500., 240.], [0., 0., 1.]]),
            np.zeros(5), (640, 480))
        left_pose = Pose(
            np.zeros((3, 1)), np.array([[-.1], [0.], [1.]]), 0.
        )
        right_pose = Pose(
            np.zeros((3, 1)), np.array([[.1], [0.], [1.]]), 0.
        )

        def joints(handedness, pixel_x):
            image = np.zeros((21, 3), dtype=float)
            image[:, :2] = [pixel_x / 640., 240. / 480.]
            return {
                'valid': True,
                'handedness': handedness,
                'handedness_score': .9,
                'image_landmarks_normalized': image.tolist(),
                'model_landmarks_m': np.zeros((21, 3)).tolist(),
                'bend_angles_rad': {},
            }

        left_measurement = joints('Left', 270.)
        right_measurement = joints('Right', 370.)
        record = {
            # Deliberately poison the old action identity. Current wrist
            # geometry must win when the raw measurements are reused.
            'hands': {
                'strap_band_L': {'joints': right_measurement},
                'strap_band_R': {'joints': left_measurement},
            },
            'unassigned_hands': [],
        }

        assigned = _reassign_cached_joints(
            record,
            ['strap_band_L', 'strap_band_R'],
            {'strap_band_L': left_pose, 'strap_band_R': right_pose},
            calibration,
        )

        self.assertIs(assigned['strap_band_L'], left_measurement)
        self.assertIs(assigned['strap_band_R'], right_measurement)

    def test_final_map_summary_follows_alias_and_shutdown_state(self):
        class Result:
            history = [{
                'final': True,
                'active_map': 3,
                'marker_map_aliases': {'3': 1},
            }]
            maps = {
                'atlas_1': {'points': list(range(7)), 'keyframes': list(range(2))},
                'atlas_2': {'points': list(range(99)), 'keyframes': list(range(99))},
            }

        mapping = _final_resolved_map(Result())
        self.assertEqual((len(mapping['points']), len(mapping['keyframes'])), (7, 2))


if __name__ == '__main__':
    unittest.main()
