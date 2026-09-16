"""Certified prefix measurements are display evidence, not projected decoration."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from aruco_track.offline_replay_features import (
    _prepare_prefix_features, prefix_feature_request, validate_prefix_feature_rows,
)
from tests.test_offline_replay_features import fixture, request_for, correspondence


class PrefixReplayFeatureTests(unittest.TestCase):
    def setUp(self):
        self.history, self.actions = fixture()
        for action in self.actions[:-1]:
            action['camera_localization_recovery'] = {
                'method': 'native-orb-final-atlas-prefix-pnp', 'accepted': True,
                'original_tracking_valid': False, 'map_id': 0, 'map_revision': 9,
            }
        self.request = request_for(self.history, self.actions)
        self.prefix = prefix_feature_request(self.history, self.actions, self.request, {0})
        self.row = {**correspondence(self.request),
                    'source': 'offline-prefix-relocalization', 'candidate': True,
                    'connected': True, 'support_only': False,
                    'matched_feature_count': 30, 'matches': 60, 'inlier_fraction': .5}
        self.other = {**copy.deepcopy(self.row), 'frame': 1, 'timestamp_s': 1 / 60.}
        self.metadata = {
            'type': 'metadata', 'schema': 'readonly-prefix-localization/v1',
            'map_id': 0, 'map_revision': 9, 'anchor_valid': True, 'atlas_modified': False,
            'anchor_translation_difference_m': .001, 'anchor_rotation_difference_deg': .1,
        }
        self.history[-1]['maps'][0]['points'] = [
            [pid, (u - 320.) / 500. * 2., (v - 240.) / 500. * 2., 1.]
            for u, v, pid, error in self.row['matched_features']
        ]

    def validate(self, row=None, metadata=None, final=None):
        return validate_prefix_feature_rows(
            [metadata or self.metadata, row or self.row, self.other],
            self.prefix, self.request, final or self.history[-1])

    def test_request_requires_actual_late_recovery_evidence(self):
        original = copy.deepcopy((self.history, self.actions, self.request))
        self.assertEqual([q['frame'] for q in self.prefix['queries']], list(range(6)))
        self.assertTrue(all(q['original_pose_valid'] is False for q in self.prefix['queries']))
        self.assertEqual((self.history, self.actions, self.request), original)
        for field, value in [('accepted', False), ('original_tracking_valid', True),
                             ('method', 'interpolated'), ('map_revision', 10), ('map_id', 1)]:
            actions = copy.deepcopy(self.actions)
            for action in actions:
                if 'camera_localization_recovery' in action:
                    action['camera_localization_recovery'][field] = value
            self.assertIsNone(prefix_feature_request(self.history, actions, self.request, {0}))

    def test_no_fallback_when_all_regular_matches_succeed(self):
        self.assertIsNone(prefix_feature_request(self.history, self.actions, self.request, set()))

    def test_existing_native_pose_cannot_masquerade_as_prefix(self):
        self.history[0]['pose'] = [0., 0., -1., 0., 0., 0., 1.]
        self.history[0]['state'] = 2
        self.assertIsNone(prefix_feature_request(self.history, self.actions, self.request, {0}))

    def test_marker_seed_uses_intervening_support_frames(self):
        self.history[2]['pose'] = [0., 0., -1., 0., 0., 0., 1.]
        self.history[2]['state'] = 6
        request = prefix_feature_request(self.history, self.actions, self.request, {0})
        self.assertEqual(request['boundary_frame'], 2)
        self.assertEqual([q['frame'] for q in request['queries']], [0, 1])
        self.assertEqual([q['frame'] for q in request['support_queries']], [2, 4, 6])

    def test_prefix_matches_are_reprojected_in_unchanged_frozen_pose(self):
        original = copy.deepcopy((self.row, self.request, self.history))
        output = self.validate()
        self.assertEqual(output[0]['inliers'], 30)
        self.assertLess(output[0]['rms_px'], 1e-9)
        self.assertEqual(output[0]['correspondence_origin'], 'validated-prefix-pnp')
        self.assertEqual(output[0]['source'], 'offline-final-map-correspondence')
        self.assertEqual((self.row, self.request, self.history), original)

    def test_anchor_and_original_pose_gates_remain_mandatory(self):
        for field, value in [('anchor_valid', False), ('atlas_modified', True),
                             ('anchor_translation_difference_m', .05),
                             ('anchor_rotation_difference_deg', 5.), ('map_revision', 10)]:
            self.assertEqual(self.validate(metadata={**self.metadata, field: value}), {})
        for field, value in [('connected', False), ('accepted', False), ('candidate', False),
                             ('support_only', True), ('source', 'unknown'), ('inliers', 29),
                             ('matches', 100), ('inlier_fraction', .449), ('occupied_cells', 4),
                             ('hull_fraction', .059), ('rms_px', 3.01), ('matched_feature_count', 29)]:
            self.assertEqual(self.validate(row={**self.row, field: value}), {})

    def test_single_prefix_and_malformed_frame_are_not_accepted(self):
        self.assertEqual(validate_prefix_feature_rows([self.metadata, self.row], self.prefix,
                         self.request, self.history[-1]), {})
        self.assertEqual(self.validate(row={**self.row, 'frame': []}), {})

    def test_frozen_pose_disagreement_cannot_be_fixed_by_changing_labels(self):
        self.request['queries'][0]['T_world_camera'][0][3] += .5
        self.assertEqual(self.validate(), {})

    def test_missing_bad_or_behind_camera_final_map_points_are_rejected(self):
        for variation in ('missing', 'behind', 'wrong_map', 'wrong_revision', 'delta'):
            final = copy.deepcopy(self.history[-1])
            mapping = final['maps'][0]
            if variation == 'missing':
                mapping['points'].pop()
            elif variation == 'behind':
                for point in mapping['points']:
                    point[3] = -5.
            elif variation == 'wrong_map':
                mapping['id'] = 1
            elif variation == 'wrong_revision':
                mapping['revision'] = 8
            else:
                mapping['points_mode'] = 'delta'
            self.assertEqual(self.validate(final=final), {})

    def test_updated_hand_marker_mask_cannot_admit_old_background_points(self):
        self.request['queries'][0]['excluded_polygons'] = [[[0, 0], [640, 0], [640, 480], [0, 480]]]
        self.assertEqual(self.validate(), {})

    def test_duplicate_pixel_or_point_evidence_is_rejected(self):
        for field in ('point', 'pixel'):
            row = copy.deepcopy(self.row)
            if field == 'point':
                row['matched_features'][1][2] = row['matched_features'][0][2]
            else:
                row['matched_features'][1][:2] = row['matched_features'][0][:2]
            self.assertEqual(self.validate(row=row), {})

    def test_optional_failure_preserves_regular_matches(self):
        accepted = {0: {'inliers': 36, 'source': 'regular'}}
        for error in (OSError('missing'), subprocess.TimeoutExpired('adapter', 1),
                      json.JSONDecodeError('bad cache', 'x', 0)):
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                with patch('aruco_track.offline_replay_features._run_prefix_features', side_effect=error):
                    result = _prepare_prefix_features(Path('.'), directory, Path('binary'), Path('atlas'),
                        {}, self.request, self.history, self.actions, accepted)
                self.assertIs(result, accepted)
                self.assertEqual(json.loads((directory / 'prefix_feature_failure.json').read_text())['status'],
                                 'optional_prefix_features_failed')

    def test_successful_fallback_never_replaces_regular_measurements(self):
        regular = correspondence(self.request)
        accepted = {0: regular}
        prefix = prefix_feature_request(self.history, self.actions, self.request, {2, 4})
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / 'prefix_feature_matches.meta.json').write_text(json.dumps({
                'inputs': {'prefix_request': prefix, 'policy': 'validated-prefix-correspondence/v1'},
                'returncode': 0,
            }))
            (directory / 'prefix_feature_candidates.jsonl').write_text('\n'.join(
                json.dumps(row) for row in [self.metadata, self.row, self.other]))
            with patch('aruco_track.offline_replay_features.subprocess.run', side_effect=AssertionError('cache missed')):
                output = _prepare_prefix_features(Path('.'), directory, Path('binary'), Path('atlas'),
                    {}, self.request, self.history, self.actions, accepted)
            self.assertIs(output[0], regular)


if __name__ == '__main__':
    unittest.main()
