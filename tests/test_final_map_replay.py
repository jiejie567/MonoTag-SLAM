"""Final-map replay regressions without native SLAM, a camera, or encoding."""
import copy
import gzip
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from aruco_track.models import Calibration
from aruco_track.replay_browser import decode_timeline
from aruco_track.slam_replay import (
    _TrailReplayCache, final_label_camera_frame, write_slam_replay,
)
import render_slam_replay


def pose(x=.1, y=0., z=-1.):
    return {'translation_m': [x, y, z],
            'quaternion_wxyz': [1., 0., 0., 0.],
            'reprojection_error_px': .2, 'marker_ids': [20]}


def snapshot(index=0, *, final=False, map_id=0, revision=9, metric=True, state=2):
    return {'timestamp': index / 30., 'final': final, 'state': state,
            'active_map': map_id,
            'pose': [-10., 0., -1., 0., 0., 0., 1.] if state == 2 else None,
            'reference': 10, 'reference_scale': 1.,
            'relative': [10., 0., 1., 0., 0., 0., 1.],
            'references': [[10, map_id, [0., 0., 0., 0., 0., 0., 1.], 1.]],
            'feature_count': index + 7, 'matched_features': [[10., 20., index + 11]],
            'maps': [{'id': map_id, 'metric': metric, 'seed': metric,
                      'background': True, 'revision': revision, 'scale': 5.,
                      'points': [[1, float(index), 0., 1.]],
                      'keyframes': [], 'markers': {}}]}


def action(index=0, *, map_id=0, revision=9, source='head-slam'):
    world = f'atlas_{map_id}'
    return {'frame': index, 'timestamp_s': index / 30.,
            'camera_world_pose_fused': pose(.1 + index / 10.),
            'camera_world_source': source, 'camera_world_confidence': .8,
            'camera_submap_id': world, 'world_frame_id': world,
            'map_revision': revision, 'scale_status': 'metric',
            'slam_inliers': 80, 'graph_reprojection_error_px': .3,
            'initialization_source': 'marker', 'background_map_ready': True,
            'camera_metric_recovered_later': False,
            'camera_localization_recovery': None,
            'marker_camera_pose_observed': pose(99.),
            'marker_camera_confidence': .99,
            'accepted_marker_ids': [], 'rejected_marker_ids': [],
            'detected_marker_corners': {},
            'hands': {'right': {'world_submap_id': world,
                               'wrist_camera_graph': pose(10., 0., 1.2),
                               'wrist_world_graph': pose(.5, .1, .2),
                               'joints': {'valid': True,
                                          'camera_landmarks_m': [[100., 100., 100.]] * 21,
                                          'world_landmarks_graph_m': [[.7, .2, .4]] * 21}}}}


def prefix_recovery():
    return {'method': 'native-orb-final-atlas-prefix-pnp', 'accepted': True,
            'original_tracking_valid': False, 'map_id': 0, 'map_revision': 9,
            'available_after_timestamp_s': 2 / 30., 'inliers': 80, 'rms_px': .3}


class FinalLabelCameraTests(unittest.TestCase):
    def test_final_exported_pose_and_diagnostics_are_preserved(self):
        record, final = action(), snapshot(final=True)
        record['camera_world_pose_fused']['quaternion_wxyz'] = [2 ** -.5, 0., 0., 2 ** -.5]
        record['camera_metric_recovered_later'] = True
        original = copy.deepcopy((record, final))
        with patch('aruco_track.slam_replay.select_camera_frame',
                   side_effect=AssertionError('final labels must not be reselected')):
            selected = final_label_camera_frame(record, final)
        np.testing.assert_allclose(selected.pose.tvec.ravel(), [.1, 0., -1.])
        np.testing.assert_allclose(selected.pose.rotation_matrix,
                                   [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], atol=1e-12)
        self.assertEqual((selected.source, selected.map_id, selected.revision),
                         ('head-slam', 'atlas_0', 9))
        self.assertEqual((selected.confidence, selected.slam_inliers,
                          selected.graph_reprojection_error_px), (.8, 80, .3))
        self.assertTrue(selected.metric)
        self.assertTrue(selected.metric_recovered_later)
        self.assertEqual((record, final), original)

    def test_offline_prefix_recovery_provenance_survives(self):
        record = action()
        record['camera_localization_recovery'] = prefix_recovery()
        selected = final_label_camera_frame(record, snapshot(final=True))
        self.assertIsNotNone(selected.pose)
        self.assertEqual(selected.localization_recovery, prefix_recovery())
        self.assertFalse(selected.localization_recovery['original_tracking_valid'])

    def test_invalid_or_missing_final_pose_never_revives_raw_marker_pose(self):
        for source, value in [('invalid', pose()), ('head-slam', None), ('slam', pose())]:
            record = action(source=source)
            record['camera_world_pose_fused'] = value
            record['camera_localization_recovery'] = prefix_recovery()
            with self.subTest(source=source, pose=value):
                selected = final_label_camera_frame(record, snapshot(final=True))
                self.assertIsNone(selected.pose)
                self.assertEqual((selected.source, selected.confidence), ('invalid', 0.))

    def test_revision_mismatch_or_missing_revision_is_invalid(self):
        for revision in (8, 10, None):
            record = action(revision=revision)
            with self.subTest(revision=revision):
                selected = final_label_camera_frame(record, snapshot(final=True))
                self.assertIsNone(selected.pose)
                self.assertEqual(selected.source, 'invalid')

    def test_unknown_map_or_inconsistent_world_ownership_is_invalid(self):
        for field, value in [('camera_submap_id', 'atlas_4'),
                             ('camera_submap_id', None),
                             ('world_frame_id', 'atlas_4')]:
            record = action()
            record[field] = value
            with self.subTest(field=field, value=value):
                self.assertIsNone(final_label_camera_frame(record, snapshot(final=True)).pose)
        record = action()
        record['world_frame_id'] = None
        self.assertIsNotNone(final_label_camera_frame(record, snapshot(final=True)).pose)

    def test_unscaled_label_or_nonmetric_final_map_is_invalid(self):
        for scale_status, metric in [('unscaled', True), (None, True), ('metric', False)]:
            record = action()
            record['scale_status'] = scale_status
            with self.subTest(scale_status=scale_status, metric=metric):
                selected = final_label_camera_frame(record, snapshot(final=True, metric=metric))
                self.assertIsNone(selected.pose)
                self.assertFalse(selected.metric)

    def test_multimap_labels_use_final_ownership_without_alias_or_scale_transform(self):
        final = snapshot(final=True, map_id=2, revision=42)
        final['maps'].append(snapshot(map_id=4, revision=55)['maps'][0])
        final['marker_map_aliases'] = {'0': 2}
        final['references'][0][2][0] = 200.
        final['references'][0][3] = 7.
        final['maps'][0]['scale'] = 11.
        record = action(map_id=2, revision=42)
        selected = final_label_camera_frame(record, final)
        self.assertEqual((selected.map_id, selected.revision), ('atlas_2', 42))
        np.testing.assert_allclose(selected.pose.tvec.ravel(), [.1, 0., -1.])
        # Surviving maps are not implicitly merged into the active map either.
        other = final_label_camera_frame(action(map_id=4, revision=55), final)
        self.assertEqual(other.map_id, 'atlas_4')
        np.testing.assert_allclose(other.pose.tvec.ravel(), [.1, 0., -1.])
        self.assertIsNone(final_label_camera_frame(action(map_id=0, revision=42), final).pose)

    def test_nonfinite_or_malformed_final_pose_is_invalid(self):
        for field, value in [('translation_m', [float('nan'), 0., 1.]),
                             ('translation_m', [1., 2.]),
                             ('quaternion_wxyz', [0., 0., 0., 0.])]:
            record = action()
            record['camera_world_pose_fused'][field] = value
            with self.subTest(field=field, value=value):
                selected = final_label_camera_frame(record, snapshot(final=True))
                self.assertIsNone(selected.pose)
                self.assertEqual(selected.source, 'invalid')


class FinalTrailCacheTests(unittest.TestCase):
    def test_recovered_prefix_uses_final_camera_despite_invalid_native_snapshot(self):
        record = action()
        record['camera_localization_recovery'] = prefix_recovery()
        cache = _TrailReplayCache([snapshot(state=1)], [record], 30., final_labels=True)
        with patch('aruco_track.slam_replay.replay_camera_frame',
                   side_effect=AssertionError('final cache must use final labels')):
            selected, trails, camera, map_id = cache.resolve(0, snapshot(final=True))
        self.assertEqual(map_id, 'atlas_0')
        self.assertEqual(selected.localization_recovery, prefix_recovery())
        np.testing.assert_allclose(camera.tvec.ravel(), [.1, 0., -1.])
        np.testing.assert_allclose(trails['right'], [[.5, .1, .2]])

    def test_invalid_final_camera_does_not_fall_back_to_native_pose(self):
        cache = _TrailReplayCache([snapshot()], [action(source='invalid')], 30., final_labels=True)
        with patch('aruco_track.slam_replay.camera_at_revision',
                   side_effect=AssertionError('an invalid final label is a hard gap')):
            selected, trails, camera, _ = cache.resolve(0, snapshot(final=True))
        self.assertEqual(selected.source, 'invalid')
        self.assertIsNone(camera)
        self.assertEqual(trails, {})

    def test_final_wrist_is_not_scaled_again_and_only_past_samples_are_used(self):
        records = [action(i) for i in range(3)]
        records[2]['hands']['right']['wrist_world_graph'] = pose(999., 999., 999.)
        final = snapshot(2, final=True)
        final['references'][0][3] = 20.
        final['maps'][0]['scale'] = 50.
        original = copy.deepcopy(records)
        cache = _TrailReplayCache([snapshot(i) for i in range(3)], records, 30., final_labels=True)
        _, trails, camera, _ = cache.resolve(1, final)
        np.testing.assert_allclose(camera.tvec.ravel(), [.2, 0., -1.])
        np.testing.assert_allclose(trails['right'], [[.5, .1, .2]] * 2)
        self.assertEqual(cache.trail_timestamps, [0., 1 / 30.])
        self.assertEqual(records, original)

    def test_cross_map_and_invalid_camera_gaps_are_never_bridged(self):
        records = [action(0), action(1, map_id=2), action(2, source='invalid'), action(3)]
        final = snapshot(3, final=True)
        final['maps'].append(snapshot(map_id=2)['maps'][0])
        cache = _TrailReplayCache([snapshot(i) for i in range(4)], records, 30., final_labels=True)
        _, trails, _, map_id = cache.resolve(3, final)
        self.assertEqual(map_id, 'atlas_0')
        self.assertEqual(trails['right'][1:3], [None, None])
        np.testing.assert_allclose([trails['right'][0], trails['right'][3]], [[.5, .1, .2]] * 2)
        self.assertEqual(cache.display_bridges['right'], [])


class FinalMapPackageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.records = [action(i) for i in range(3)]
        self.records[0]['camera_localization_recovery'] = prefix_recovery()
        self.records[1]['camera_world_source'] = 'invalid'
        self.history = [snapshot(0, state=1, metric=False, revision=1),
                        snapshot(1, state=4, metric=False, revision=2),
                        snapshot(2, revision=7), snapshot(2, final=True)]
        self.history[0]['maps'][0]['points'] = []
        self.history[-1]['maps'][0]['points'] = [[5, 2., 3., 4.], [6, 7., 8., 9.]]

    def render(self):
        records_path = self.directory / 'actions.jsonl'
        records_path.write_text(''.join(json.dumps(record) + '\n' for record in self.records))
        self.original_action_bytes = records_path.read_bytes()
        original = copy.deepcopy((self.history, self.records))
        capture, encoder = MagicMock(), MagicMock()
        capture.isOpened.return_value = True
        capture.read.return_value = (True, np.zeros((480, 640, 3), np.uint8))
        encoder.wait.return_value = 0
        # Do not retain raw frame buffers in MagicMock call history.
        encoder.stdin.write = lambda data: len(data)
        calibration = Calibration(np.array([[500., 0., 320.], [0., 500., 240.], [0., 0., 1.]]),
                                  np.zeros(5), (640, 480))
        with patch('aruco_track.slam_replay.cv2.VideoCapture', return_value=capture), \
             patch('aruco_track.slam_replay.subprocess.Popen', return_value=encoder), \
             patch('aruco_track.slam_replay.shutil.which', return_value='/mock/ffmpeg'):
            write_slam_replay(Path('mock.avi'), records_path, self.directory, self.history,
                              calibration, [{}, {}, {}], [(), (), ()], 30.,
                              actions=self.records, final_map=True)
        self.assertEqual((self.history, self.records), original)
        self.assertEqual(records_path.read_bytes(), self.original_action_bytes)
        with gzip.open(self.directory / 'video_frames.json.gz', 'rt') as stream:
            frames = json.load(stream)
        with gzip.open(self.directory / 'timeline.json.gz', 'rt') as stream:
            timeline = json.load(stream)
        manifest = json.loads((self.directory / 'manifest.json').read_text())
        return frames, timeline, manifest

    def test_final_cloud_is_selected_from_zero_without_rewriting_history_or_adding_tail(self):
        frames, timeline, manifest = self.render()
        self.assertEqual(len(frames), 3)
        self.assertEqual([frame['source_frame'] for frame in frames], [0, 1, 2])
        self.assertEqual([frame['sequence'] for frame in frames], [3, 3, 3])
        self.assertEqual([frame['observation_sequence'] for frame in frames], [0, 1, 2])
        self.assertTrue(all(not frame['tail'] for frame in frames))
        self.assertEqual([row['sequence'] for row in timeline], [0, 1, 2, 3])
        self.assertEqual([row['state'] for row in timeline], [1, 4, 2, 2])
        self.assertEqual([row['maps'][0]['revision'] for row in timeline], [1, 2, 7, 9])
        self.assertEqual(timeline[0]['count'], 0)
        selected = timeline[frames[0]['sequence']]
        self.assertTrue(selected['checkpoint'])
        data = gzip.decompress((self.directory / 'points.bin.gz').read_bytes())
        points = [struct.unpack_from('<QQfff', data, selected['offset'] + index * 28)
                  for index in range(selected['count'])]
        self.assertEqual(points, [(0, 5, 2., 3., 4.), (0, 6, 7., 8., 9.)])
        with gzip.open(self.directory / manifest['browser_timeline'], 'rt') as stream:
            self.assertEqual(decode_timeline(json.load(stream)), timeline)

    def test_final_camera_hands_and_wrist_trails_use_exported_world_geometry(self):
        frames, _, _ = self.render()
        self.assertEqual(frames[0]['localization_recovery'], prefix_recovery())
        for index in (0, 2):
            frame = frames[index]
            np.testing.assert_allclose(frame['camera']['translation'], [.1 + index / 10., 0., -1.])
            self.assertEqual(frame['map_revision'], 9)
            np.testing.assert_allclose(frame['hands']['right'], [[.7, .2, .4]] * 21)
            np.testing.assert_allclose(frame['trails']['right'][-1], [.5, .1, .2])
        self.assertIsNone(frames[1]['camera'])
        self.assertEqual(frames[1]['source'], 'invalid')
        self.assertEqual(frames[1]['hands'], {})
        self.assertEqual(frames[1]['trails'], {})
        self.assertIsNone(frames[2]['trails']['right'][1])
        self.assertEqual([frame['orb_detected'] for frame in frames], [7, 8, 9])
        self.assertEqual([frame['orb_tracked'] for frame in frames], [0, 0, 1])
        self.assertEqual([frame['orb_map_point_ids'] for frame in frames], [[], [], [13]])
        # Missing final hand labels must not be reconstructed from the deliberately
        # inconsistent camera-local landmarks still present in the observations.
        self.records[0]['hands']['right']['joints']['world_landmarks_graph_m'] = None
        self.records[2]['hands']['right']['world_submap_id'] = 'atlas_4'
        without_final_hands, _, _ = self.render()
        self.assertEqual(without_final_hands[0]['hands'], {})
        self.assertEqual(without_final_hands[2]['hands'], {})

    def test_manifest_explicitly_identifies_final_map_mode_and_has_no_final_hold(self):
        _, _, manifest = self.render()
        self.assertEqual(manifest['replay_mode'], 'final-map')
        self.assertEqual(manifest['final_hold_seconds'], 0.)
        self.assertEqual(manifest['analysis_frames'], 3)
        self.assertEqual(manifest['frames'], 3)
        self.assertEqual(manifest['offline_preprocessing']['prefix_relocalized_final_labels'], 1)


class FinalMapCliTests(unittest.TestCase):
    def test_final_cli_uses_sibling_directory_and_never_replaces_original_package(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'actions_replay'
            source.mkdir()
            for name, value in [('process.mp4', 'original video'), ('index.html', 'original HTML'),
                                ('native_history.jsonl', 'cached history'), ('atlas.osa', 'cached Atlas')]:
                (source / name).write_text(value)
            actions_path = root / 'actions.jsonl'
            actions_path.write_text(json.dumps(action()) + '\n')
            metadata_path = actions_path.with_suffix('.meta.json')
            metadata_path.write_text(json.dumps({'replay': str(source / 'index.html'),
                                                  'video': 'mock.avi', 'fps': 30.}))
            originals = {path: path.read_bytes() for path in [*source.iterdir(), actions_path, metadata_path]}
            target = root / 'actions_replay_final'
            history = [snapshot(), snapshot(final=True)]
            with patch('sys.argv', ['render_slam_replay.py', str(actions_path), '--final-map']), \
                 patch('render_slam_replay.load_replay_calibration', return_value=MagicMock()), \
                 patch('render_slam_replay.resolve_native_history_path',
                       return_value=source / 'native_history.jsonl') as resolve_history, \
                 patch('render_slam_replay.read_native_history', return_value=history), \
                 patch('render_slam_replay.write_slam_replay', return_value=(
                     target / 'process.mp4', target / 'index.html')) as render, \
                 patch('builtins.print'):
                render_slam_replay.main()
            resolve_history.assert_called_once_with(source)
            self.assertEqual(render.call_args.args[2], target)
            self.assertIs(render.call_args.args[3], history)
            self.assertTrue(render.call_args.kwargs['final_map'])
            self.assertEqual((target / 'native_history.jsonl').read_bytes(),
                             originals[source / 'native_history.jsonl'])
            self.assertEqual((target / 'atlas.osa').read_bytes(), originals[source / 'atlas.osa'])
            for path, data in originals.items():
                self.assertEqual(path.read_bytes(), data)
            (target / 'process.mp4').write_text('existing final render')
            with patch('sys.argv', ['render_slam_replay.py', str(actions_path), '--final-map']), \
                 patch('render_slam_replay.write_slam_replay') as render, \
                 patch('sys.stderr', new=MagicMock()), \
                 self.assertRaises(SystemExit) as error:
                render_slam_replay.main()
            self.assertEqual(error.exception.code, 2)
            render.assert_not_called()
            self.assertEqual((target / 'process.mp4').read_text(), 'existing final render')
            for path, data in originals.items():
                self.assertEqual(path.read_bytes(), data)


if __name__ == '__main__':
    unittest.main()
