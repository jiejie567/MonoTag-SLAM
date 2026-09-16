"""Offline replay-only ORB correspondence contracts; no detector or SLAM run."""
import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from aruco_track.models import Calibration
from aruco_track.offline_replay_features import (
    replay_feature_request, validate_replay_feature_rows,
)
from aruco_track.slam_replay import _draw_offline_orb_features, write_slam_replay


def calibration():
    return Calibration(np.array([[500., 0., 320.], [0., 500., 240.], [0., 0., 1.]]),
                       np.zeros(5), (640, 480))


def fixture(boundary=6, fps=60.):
    history, actions = [], []
    for index in range(boundary + 1):
        ready = index == boundary
        mapping = {'id': 0, 'revision': 9 if ready else 1, 'metric': True,
                   'seed': True, 'background': ready, 'scale': 1.,
                   'points': [[1, 0., 0., 1.]], 'keyframes': [], 'markers': {}}
        history.append({'timestamp': index / fps, 'final': False,
                        'state': 2 if ready else 1, 'active_map': 0,
                        'pose': [0., 0., -1., 0., 0., 0., 1.] if ready else None,
                        'maps': [mapping], 'feature_count': 10,
                        'matched_features': [[300., 200., 1]] if ready else []})
        actions.append({'frame': index, 'timestamp_s': index / fps,
                        'camera_world_pose_fused': {'translation_m': [0., 0., -1.],
                                                    'quaternion_wxyz': [1., 0., 0., 0.]},
                        'camera_world_source': 'head-slam', 'camera_world_confidence': .8,
                        'camera_submap_id': 'atlas_0', 'world_frame_id': 'atlas_0',
                        'map_revision': 9, 'scale_status': 'metric',
                        'slam_inliers': 80, 'graph_reprojection_error_px': .4,
                        'detected_marker_corners': {}, 'accepted_marker_ids': [],
                        'boundary_rejected_marker_corners': {}, 'hands': {}})
    history.append(copy.deepcopy(history[-1]))
    history[-1]['final'] = True
    return history, actions


def request_for(history, actions, fps=60.):
    return replay_feature_request(history, actions, calibration(), fps, Path('mock.avi'))


def correspondence(request, frame=0):
    query = next(query for query in request['queries'] if query['frame'] == frame)
    features = [[60. + 100 * column, 50. + 80 * row, 100 + row * 6 + column, .5]
                for row in range(5) for column in range(6)]
    return {'type': 'frame', 'frame': frame, 'timestamp_s': query['timestamp_s'],
            'map_id': request['map_id'], 'map_revision': request['map_revision'],
            'source': 'offline-final-map-correspondence', 'accepted': True,
            'validated_after_final_atlas': True,
            'validation_effective_time_s': request['validation_effective_time_s'],
            'matched_features': features, 'features': 80, 'matches': 30, 'inliers': 30,
            'occupied_cells': 12, 'hull_fraction': 500 * 320 / (640 * 480),
            'rms_px': .5, 'p95_px': .5, 'inlier_fraction': 1.}


class ReplayFeatureRequestTests(unittest.TestCase):
    def test_request_samples_prefix_at_30fps_and_preserves_frozen_camera_labels(self):
        history, actions = fixture()
        history[4]['state'] = 6
        history[4]['pose'] = [50., 0., -1., 0., 0., 0., 1.]
        original = copy.deepcopy((history, actions))
        request = request_for(history, actions)
        self.assertEqual(request['purpose'], 'replay-feature-correspondence')
        self.assertEqual((request['map_id'], request['map_revision']), (0, 9))
        self.assertEqual(request['boundary_frame'], 6)
        self.assertEqual([query['frame'] for query in request['queries']], [0, 2, 4])
        for query in request['queries']:
            self.assertAlmostEqual(query['timestamp_s'], query['frame'] / 60.)
            expected = np.eye(4)
            expected[2, 3] = -1.
            np.testing.assert_allclose(query['T_world_camera'], expected)
        self.assertEqual((history, actions), original)

    def test_no_request_without_final_snapshot_or_successful_background_boundary(self):
        for variation in ('no_final', 'no_background', 'historical_background_missing',
                          'no_native_pose', 'no_state2'):
            history, actions = fixture()
            if variation == 'no_final':
                history.pop()
            elif variation == 'no_background':
                for frame in history:
                    frame['maps'][0]['background'] = False
            elif variation == 'historical_background_missing':
                history[-2]['maps'][0]['background'] = False
            elif variation == 'no_native_pose':
                history[-2]['pose'] = None
            else:
                history[-2]['state'] = 6
            with self.subTest(variation=variation):
                self.assertIsNone(request_for(history, actions))

    def test_queries_exclude_invalid_final_labels_but_allow_recovered_prefix_states(self):
        for field, value in [('camera_world_source', 'invalid'),
                             ('camera_world_pose_fused', None),
                             ('map_revision', 8), ('scale_status', 'unscaled'),
                             ('camera_submap_id', 'atlas_4'), ('world_frame_id', 'atlas_4')]:
            history, actions = fixture()
            for record in actions[:-1]:
                record[field] = value
            with self.subTest(field=field):
                self.assertIsNone(request_for(history, actions))
        history, actions = fixture()
        history[0]['state'], history[2]['state'] = 3, 4
        request = request_for(history, actions)
        self.assertEqual([query['frame'] for query in request['queries']], [0, 2, 4])

    def test_lost_prefix_without_valid_final_pose_remains_excluded(self):
        history, actions = fixture()
        history[0]['state'] = 4
        actions[0]['camera_world_pose_fused'] = None
        request = request_for(history, actions)
        self.assertEqual([query['frame'] for query in request['queries']], [2, 4])

    def test_request_is_limited_to_five_seconds_before_initial_background(self):
        history, actions = fixture(boundary=480)
        request = request_for(history, actions)
        indices = [query['frame'] for query in request['queries']]
        self.assertTrue(indices)
        self.assertGreaterEqual(min(indices), 180)
        self.assertLess(max(indices), 480)
        self.assertLessEqual(len(indices), 150)
        self.assertTrue(all(b - a >= 2 for a, b in zip(indices, indices[1:])))

    def test_markers_rejected_markers_and_observed_hands_are_excluded(self):
        history, actions = fixture()
        marker = [[10., 10.], [40., 10.], [40., 40.], [10., 40.]]
        rejected = [[50., 50.], [80., 50.], [80., 80.], [50., 80.]]
        hand = {'valid': True,
                'image_landmarks_normalized': [[.25, .25, 0.], [.5, .25, 0.], [.5, .5, 0.]]}
        actions[0]['detected_marker_corners'] = {'20': marker}
        actions[0]['boundary_rejected_marker_corners'] = {'21': rejected}
        actions[0]['hands'] = {'right': {'joints': hand}}
        actions[0]['unassigned_hands'] = [dict(hand, image_landmarks_normalized=[
            [.6, .6, 0.], [.8, .6, 0.], [.8, .8, 0.]])]
        actions[-1]['detected_marker_corners'] = {'20': marker}
        original = copy.deepcopy(actions)
        request = request_for(history, actions)
        excluded = request['queries'][0]['excluded_polygons']
        self.assertIn(marker, excluded)
        self.assertIn(rejected, excluded)
        self.assertIn([[160., 120.], [320., 120.], [320., 240.]], excluded)
        self.assertIn([[384., 288.], [512., 288.], [512., 384.]], excluded)
        masks = {row['frame']: row['excluded_polygons'] for row in request['mask_frames']}
        self.assertIn(request['boundary_frame'], masks)
        self.assertIn(marker, masks[request['boundary_frame']])
        self.assertEqual(actions, original)


class ReplayFeatureRowTests(unittest.TestCase):
    def setUp(self):
        self.request = request_for(*fixture())
        self.row = correspondence(self.request)

    def test_valid_measured_correspondences_are_accepted_without_mutating_inputs(self):
        original = copy.deepcopy((self.request, self.row))
        accepted = validate_replay_feature_rows([self.row], self.request)
        self.assertEqual(accepted, {0: self.row})
        self.assertEqual((self.request, self.row), original)

    def test_rows_require_exact_query_source_map_revision_and_explicit_acceptance(self):
        for field, value in [('frame', 1), ('frame', False), ('frame', 0.),
                             ('source', 'offline-prefix-relocalization'),
                             ('accepted', False), ('accepted', 'true'),
                             ('map_id', 4), ('map_revision', 8),
                             ('timestamp_s', .001), ('validation_effective_time_s', -1.)]:
            with self.subTest(field=field, value=value):
                self.assertEqual(validate_replay_feature_rows(
                    [dict(self.row, **{field: value})], self.request), {})

    def test_nonfinite_clocks_and_malformed_rows_are_rejected_not_raised(self):
        variants = [None, [], {}, dict(self.row, frame=[]), dict(self.row, frame={}),
                    dict(self.row, matched_features=None), dict(self.row, occupied_cells=float('inf'))]
        for field in ('timestamp_s', 'validation_effective_time_s', 'rms_px'):
            variants.extend(dict(self.row, **{field: value}) for value in (float('nan'), float('inf')))
        for row in variants:
            with self.subTest(row=row):
                self.assertEqual(validate_replay_feature_rows([row], self.request), {})

    def test_invalid_point_rejects_whole_frame_instead_of_filtering_to_thirty_valid_points(self):
        for index, value in [(0, float('nan')), (0, -1.), (0, 640.), (1, 480.),
                             (2, True), (2, 140.), (2, -1), (3, -1.), (3, 3.01)]:
            row = copy.deepcopy(self.row)
            extra = [250., 250., 140, .5]
            extra[index] = value
            row['matched_features'].append(extra)
            row['inliers'] = 31
            with self.subTest(index=index, value=value):
                self.assertEqual(validate_replay_feature_rows([row], self.request), {})

    def test_duplicate_points_or_frames_are_not_last_row_wins(self):
        for duplicate_column in ('id', 'pixel'):
            row = copy.deepcopy(self.row)
            if duplicate_column == 'id':
                row['matched_features'][1][2] = row['matched_features'][0][2]
            else:
                row['matched_features'][1][:2] = row['matched_features'][0][:2]
            with self.subTest(duplicate_column=duplicate_column):
                self.assertEqual(validate_replay_feature_rows([row], self.request), {})
        self.assertEqual(validate_replay_feature_rows([self.row, copy.deepcopy(self.row)], self.request), {})
        self.assertEqual(validate_replay_feature_rows(
            [self.row, dict(self.row, accepted=False)], self.request), {})

    def test_geometry_coverage_is_recomputed_and_not_trusted_from_reported_metrics(self):
        for coordinates in ('cluster', 'line'):
            row = copy.deepcopy(self.row)
            for index, point in enumerate(row['matched_features']):
                point[:2] = [20. + index, 20. + index % 2] if coordinates == 'cluster' else [10. + 20 * index, 200.]
            with self.subTest(coordinates=coordinates):
                self.assertEqual(validate_replay_feature_rows([row], self.request), {})
        row = copy.deepcopy(self.row)
        row['matched_features'].pop()
        row['inliers'] = 29
        self.assertEqual(validate_replay_feature_rows([row], self.request), {})

    def test_correspondence_statistics_must_satisfy_reported_gate_as_well(self):
        for field, value in [('inliers', 29), ('occupied_cells', 4),
                             ('hull_fraction', .059), ('rms_px', 3.01)]:
            with self.subTest(field=field):
                self.assertEqual(validate_replay_feature_rows(
                    [dict(self.row, **{field: value})], self.request), {})


class ReplayFeatureDrawingTests(unittest.TestCase):
    def test_cyan_rings_are_drawn_at_measured_pixels_not_projected_map_locations(self):
        features = [[100.25, 200.75, 11, .4], [300.75, 150.25, 12, .5]]
        original = copy.deepcopy(features)
        with patch('aruco_track.slam_replay.cv2.circle') as circle:
            _draw_offline_orb_features(np.zeros((480, 640, 3), np.uint8), features)
        rings = [call.args for call in circle.call_args_list if call.args[2] == 5]
        self.assertEqual([args[1] for args in rings], [(100, 201), (301, 150)])
        self.assertTrue(all(args[3:5] == ((255, 210, 0), 2) for args in rings))
        self.assertEqual(features, original)

    def render(self, *, final_map=True, invalid_camera=False, wrong_revision=False):
        history, actions = fixture(boundary=2, fps=30.)
        request = request_for(history, actions, fps=30.)
        row = correspondence(request)
        if invalid_camera:
            actions[0]['camera_world_source'] = 'invalid'
        if wrong_revision:
            row['map_revision'] = 8
        original = copy.deepcopy((history, actions, row))
        capture, encoder = MagicMock(), MagicMock()
        capture.isOpened.return_value = True
        capture.read.return_value = (True, np.zeros((480, 640, 3), np.uint8))
        encoder.wait.return_value = 0
        encoder.stdin.write = lambda data: len(data)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with patch('aruco_track.slam_replay.cv2.VideoCapture', return_value=capture), \
                 patch('aruco_track.slam_replay.subprocess.Popen', return_value=encoder), \
                 patch('aruco_track.slam_replay.shutil.which', return_value='/mock/ffmpeg'), \
                 patch('aruco_track.slam_replay.FINAL_REPLAY_HOLD_SECONDS', 0.), \
                 patch('aruco_track.slam_replay._draw_offline_orb_features') as draw:
                write_slam_replay(Path('mock.avi'), directory / 'unused.jsonl', directory,
                                  history, calibration(), [{}, {}, {}], [(), (), ()], 30.,
                                  actions=actions, final_map=final_map, offline_features={0: row})
            with gzip.open(directory / 'video_frames.json.gz', 'rt') as stream:
                frames = json.load(stream)
            with gzip.open(directory / 'timeline.json.gz', 'rt') as stream:
                timeline = json.load(stream)
        self.assertEqual((history, actions, row), original)
        return frames, timeline, draw

    def test_final_replay_adds_separate_offline_counts_and_synchronized_point_ids(self):
        frames, timeline, draw = self.render()
        self.assertEqual([frame['orb_tracked'] for frame in frames], [0, 0, 1])
        self.assertEqual([frame['orb_detected'] for frame in frames], [10, 10, 10])
        self.assertEqual([frame['orb_offline_matched'] for frame in frames], [30, 0, 0])
        self.assertEqual(frames[0]['orb_offline_match_rms_px'], .5)
        self.assertEqual(frames[0]['orb_offline_map_point_ids'], list(range(100, 130)))
        self.assertEqual(frames[0]['orb_map_point_ids'], list(range(100, 130)))
        self.assertEqual(frames[2]['orb_map_point_ids'], [1])
        self.assertEqual(frames[0]['orb_offline_match_source'], 'offline-final-map-correspondence')
        self.assertEqual([row['state'] for row in timeline], [1, 1, 2, 2])
        draw.assert_called_once()

    def test_offline_rows_do_not_enable_invalid_cameras_wrong_revisions_or_process_mode(self):
        for options in ({'invalid_camera': True}, {'wrong_revision': True}, {'final_map': False}):
            with self.subTest(options=options):
                frames, _, draw = self.render(**options)
                self.assertEqual(frames[0]['orb_tracked'], 0)
                self.assertEqual(frames[0]['orb_offline_matched'], 0)
                self.assertEqual(frames[0]['orb_offline_map_point_ids'], [])
                draw.assert_not_called()
                if options.get('invalid_camera'):
                    self.assertIsNone(frames[0]['camera'])
                    self.assertEqual(frames[0]['source'], 'invalid')


if __name__ == '__main__':
    unittest.main()
