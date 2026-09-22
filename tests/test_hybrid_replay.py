"""Hybrid replay contracts: final reconstruction and causal local view stay separate."""
import ast
import copy
import gzip
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from aruco_track.camera_state import FusedCameraFrame
from aruco_track.models import Calibration
from aruco_track.orbslam3_backend import pose_from_native
from aruco_track.replay_browser import decode_timeline
from aruco_track.slam_replay import write_slam_replay
from tools import render_slam_replay
from tools import verify_slam_replay
def _pose(x=.1, y=0., z=-1.):
    return {'translation_m': [x, y, z], 'quaternion_wxyz': [1., 0., 0., 0.],
            'reprojection_error_px': .2, 'marker_ids': [20]}


def _snapshot(index, *, final=False, state=2, map_id=0, revision=9):
    return {'timestamp': index / 30., 'final': final, 'state': state,
            'active_map': map_id, 'pose': [-10., 0., -1., 0., 0., 0., 1.] if state == 2 else None,
            'reference': None, 'relative': None, 'feature_count': index + 7,
            'matched_features': [[10., 20., index + 11]],
            'maps': [{'id': map_id, 'metric': state == 2, 'seed': state == 2,
                      'background': state == 2, 'revision': revision, 'scale': 5.,
                      'points': [[index + 11, float(index), 0., 1.]] if state == 2 else [],
                      'keyframes': [], 'markers': {}}]}


def _action(index, *, map_id=0, revision=9, source='head-slam'):
    world = f'atlas_{map_id}'
    return {'frame': index, 'timestamp_s': index / 30.,
            'camera_world_pose_fused': _pose(.1 + index / 10.),
            'camera_world_source': source, 'camera_world_confidence': .8,
            'camera_submap_id': world, 'world_frame_id': world,
            'map_revision': revision, 'scale_status': 'metric',
            'slam_inliers': 80, 'graph_reprojection_error_px': .3,
            'initialization_source': 'marker', 'background_map_ready': True,
            'camera_metric_recovered_later': False, 'camera_localization_recovery': None,
            'accepted_marker_ids': [], 'rejected_marker_ids': [],
            'detected_marker_corners': {},
            'hands': {'right': {'world_submap_id': world,
                               'wrist_camera_graph': _pose(100., 100., 100.),
                               'wrist_world_graph': _pose(.5, .1, .2),
                               'joints': {'valid': True,
                                          'camera_landmarks_m': [[100., 100., 100.]] * 21,
                                          'world_landmarks_graph_m': [[.7, .2, .4]] * 21}}}}


def _process_camera(snapshot, revision, action, *args, **kwargs):
    """Deliberately different camera/gauge from the optimized action labels."""
    if snapshot['state'] != 2:
        return FusedCameraFrame(None, 'invalid', 0., 0, None)
    return FusedCameraFrame(pose_from_native(snapshot['pose']), 'head-slam', .7, 60, .4,
                            f"atlas_{snapshot['active_map']}",
                            revision['maps'][0]['revision'], True)


class HybridReplayPackageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.records = [_action(index) for index in range(3)]
        self.records[0]['camera_localization_recovery'] = {
            'method': 'native-orb-final-atlas-prefix-pnp', 'accepted': True,
            'original_tracking_valid': False, 'map_id': 0, 'map_revision': 9,
            'available_after_timestamp_s': 2 / 30., 'inliers': 80, 'rms_px': .3}
        self.records[1]['camera_world_source'] = 'invalid'
        self.history = [_snapshot(0, state=1, revision=1),
                        _snapshot(1, revision=2), _snapshot(2, revision=3),
                        _snapshot(2, final=True)]
        self.history[-1]['maps'][0]['points'] = [[13, 2., 3., 4.], [101, 7., 8., 9.]]
        self.offline = {0: {'accepted': True, 'source': 'offline-final-map-correspondence',
                            'frame': 0, 'map_id': 0, 'map_revision': 9, 'rms_px': .2,
                            'matched_features': [[20., 30., 101, .2]]}}

    def render(self):
        records_path = self.directory / 'actions.jsonl'
        records_path.write_text(''.join(json.dumps(record) + '\n' for record in self.records))
        original_bytes = records_path.read_bytes()
        original_objects = copy.deepcopy((self.history, self.records, self.offline))
        capture, encoder = MagicMock(), MagicMock()
        capture.isOpened.return_value = True
        capture.read.return_value = (True, np.zeros((480, 640, 3), np.uint8))
        encoder.wait.return_value = 0
        encoder.stdin.write = lambda data: len(data)
        calibration = Calibration(np.array([[500., 0., 320.], [0., 500., 240.], [0., 0., 1.]]),
                                  np.zeros(5), (640, 480))
        with patch('aruco_track.slam_replay.cv2.VideoCapture', return_value=capture), \
             patch('aruco_track.slam_replay.subprocess.Popen', return_value=encoder), \
             patch('aruco_track.slam_replay.shutil.which', return_value='/mock/ffmpeg'), \
             patch('aruco_track.slam_replay.replay_camera_frame', side_effect=_process_camera), \
             patch('aruco_track.slam_replay.FINAL_REPLAY_HOLD_SECONDS', .1):
            write_slam_replay(Path('mock.avi'), records_path, self.directory, self.history,
                              calibration, [{}, {}, {}], [(), (), ()], 30.,
                              actions=self.records, hybrid=True, offline_features=self.offline)
        self.assertEqual(records_path.read_bytes(), original_bytes)
        self.assertEqual((self.history, self.records, self.offline), original_objects)
        with gzip.open(self.directory / 'video_frames.json.gz', 'rt') as stream:
            frames = json.load(stream)
        with gzip.open(self.directory / 'timeline.json.gz', 'rt') as stream:
            timeline = json.load(stream)
        manifest = json.loads((self.directory / 'manifest.json').read_text())
        return frames, timeline, manifest

    def test_global_cloud_is_final_from_zero_while_local_history_is_untouched(self):
        frames, timeline, manifest = self.render()
        analysis = [frame for frame in frames if not frame['tail']]
        self.assertEqual([frame['source_frame'] for frame in analysis], [0, 1, 2])
        self.assertEqual([frame['sequence'] for frame in analysis], [0, 1, 2])
        self.assertEqual([frame['observation_sequence'] for frame in analysis], [0, 1, 2])
        self.assertEqual([frame['global_sequence'] for frame in frames], [3] * len(frames))
        self.assertEqual([row['state'] for row in timeline], [1, 2, 2, 2])
        self.assertEqual([row['maps'][0]['revision'] for row in timeline], [1, 2, 3, 9])
        self.assertEqual(timeline[frames[0]['sequence']]['count'], 0)
        final = timeline[frames[0]['global_sequence']]
        self.assertTrue(final['checkpoint'])
        data = gzip.decompress((self.directory / 'points.bin.gz').read_bytes())
        points = [struct.unpack_from('<QQfff', data, final['offset'] + index * 28)
                  for index in range(final['count'])]
        self.assertEqual(points, [(0, 13, 2., 3., 4.), (0, 101, 7., 8., 9.)])
        with gzip.open(self.directory / manifest['browser_timeline'], 'rt') as stream:
            self.assertEqual(decode_timeline(json.load(stream)), timeline)

    def test_future_prefix_recovery_does_not_make_process_tracking_valid(self):
        frames, _, _ = self.render()
        first = frames[0]
        np.testing.assert_allclose(first['camera']['translation'], [.1, 0., -1.])
        self.assertTrue(first['localization_recovery']['accepted'])
        self.assertEqual(first['map_revision'], 9)
        self.assertIsNone(first['process_camera'])
        self.assertEqual(first['process_source'], 'invalid')
        self.assertFalse(first['process_metric'])
        self.assertEqual(first['process_orb_map_point_ids'], [])

    def test_process_valid_camera_does_not_resurrect_invalid_final_action(self):
        frames, _, _ = self.render()
        frame = frames[1]
        self.assertIsNone(frame['camera'])
        self.assertEqual(frame['source'], 'invalid')
        self.assertEqual(frame['hands'], {})
        self.assertEqual(frame['trails'], {})
        np.testing.assert_allclose(frame['process_camera']['translation'], [-10., 0., -1.])
        self.assertEqual(frame['process_map_revision'], 2)
        self.assertEqual(frame['process_source'], 'head-slam')

    def test_final_hands_and_world_trails_are_not_transformed_by_process_camera(self):
        frames, _, _ = self.render()
        frame = frames[2]
        np.testing.assert_allclose(frame['hands']['right'], [[.7, .2, .4]] * 21)
        np.testing.assert_allclose(frame['trails']['right'][-1], [.5, .1, .2])
        self.assertIsNone(frame['trails']['right'][1])
        np.testing.assert_allclose(frame['camera']['translation'], [.3, 0., -1.])
        np.testing.assert_allclose(frame['process_camera']['translation'], [-10., 0., -1.])
        self.assertEqual((frame['map_revision'], frame['process_map_revision']), (9, 3))

    def test_offline_feature_correspondences_do_not_leak_into_process_active_points(self):
        frames, _, _ = self.render()
        self.assertEqual(frames[0]['orb_offline_matched'], 1)
        self.assertEqual(frames[0]['orb_map_point_ids'], [101])
        self.assertEqual(frames[0]['process_orb_map_point_ids'], [])
        self.assertEqual(frames[2]['process_orb_map_point_ids'], [13])

    def test_distinct_process_and_final_map_ownership_remains_explicit(self):
        self.history[-1]['active_map'] = 2
        self.history[-1]['maps'][0]['id'] = 2
        for record in self.records:
            record['camera_submap_id'] = record['world_frame_id'] = 'atlas_2'
            record['hands']['right']['world_submap_id'] = 'atlas_2'
        self.offline[0]['map_id'] = 2
        frames, _, _ = self.render()
        self.assertEqual((frames[2]['map_id'], frames[2]['process_map_id']), ('atlas_2', 'atlas_0'))
        self.assertEqual(frames[2]['orb_map_point_ids'], [])
        self.assertEqual(frames[2]['process_orb_map_point_ids'], [13])
        np.testing.assert_allclose(frames[2]['hands']['right'], [[.7, .2, .4]] * 21)

    def test_final_pose_and_hand_gaps_are_never_filled_just_to_show_frame_zero(self):
        self.records[0]['camera_world_pose_fused'] = None
        self.records[0]['camera_localization_recovery'] = None
        frames, _, _ = self.render()
        self.assertIsNone(frames[0]['camera'])
        self.assertIsNone(frames[0]['process_camera'])
        self.assertEqual(frames[0]['hands'], {})
        self.assertEqual(frames[0]['trails'], {})
        self.assertEqual(frames[0]['orb_offline_matched'], 0)
        self.assertEqual(frames[0]['global_sequence'], 3)

    def test_hybrid_manifest_and_tail_identify_both_view_semantics(self):
        frames, _, manifest = self.render()
        self.assertEqual(manifest['replay_mode'], 'hybrid')
        self.assertEqual(manifest['analysis_frames'], 3)
        tail = [frame for frame in frames if frame['tail']]
        self.assertEqual(len(tail), 3)
        self.assertTrue(all(frame['source_frame'] == 2 for frame in tail))
        self.assertTrue(all(frame['sequence'] == frame['global_sequence'] == 3 for frame in tail))
        self.assertTrue(all(frame['observation_sequence'] == 2 for frame in tail))

    def test_hybrid_and_final_map_modes_are_mutually_exclusive(self):
        with self.assertRaises(ValueError), \
             patch('aruco_track.slam_replay.subprocess.Popen') as encoder:
            write_slam_replay(Path('mock.avi'), Path('actions.jsonl'), self.directory,
                              self.history, MagicMock(), [], [], 30.,
                              actions=self.records, hybrid=True, final_map=True)
        encoder.assert_not_called()

    def test_package_verifier_accepts_hybrid_but_rejects_mixed_revisions(self):
        frames, _, _ = self.render()
        history_path = self.directory / 'native_history.jsonl'
        history_path.write_text('test history supplied by fixture')
        with patch('tools.verify_slam_replay.resolve_native_history_path', return_value=history_path), \
             patch('tools.verify_slam_replay.read_native_history', return_value=self.history):
            result = verify_slam_replay.verify(self.directory)
            self.assertEqual(result['verified_video_frames'], len(frames))
            for field, wrong in [('global_sequence', 1), ('process_map_revision', 9),
                                 ('map_revision', 3)]:
                corrupted = copy.deepcopy(frames)
                corrupted[2][field] = wrong
                with gzip.open(self.directory / 'video_frames.json.gz', 'wt') as stream:
                    json.dump(corrupted, stream)
                with self.subTest(field=field), self.assertRaises(AssertionError):
                    verify_slam_replay.verify(self.directory)


class HybridReplayCliTests(unittest.TestCase):
    def test_exporter_uses_hybrid_and_cached_final_feature_matches(self):
        source = Path(__file__).resolve().parents[1] / 'tools/export_action_labels.py'
        tree = ast.parse(source.read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == 'write_slam_replay']
        self.assertEqual(len(calls), 1)
        keywords = {keyword.arg: keyword.value for keyword in calls[0].keywords}
        self.assertIs(keywords['hybrid'].value, True)
        self.assertIsInstance(keywords['offline_features'], ast.Name)
        prepared_names = {target.id for node in ast.walk(tree)
                          if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                          and isinstance(node.value.func, ast.Name)
                          and node.value.func.id == 'prepare_final_replay_features'
                          for target in node.targets if isinstance(target, ast.Name)}
        self.assertIn(keywords['offline_features'].id, prepared_names)

    def test_default_render_is_a_new_hybrid_package_and_keeps_source_immutable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'actions_replay'
            source.mkdir()
            for name, value in [('process.mp4', 'original video'), ('index.html', 'original HTML'),
                                ('native_history.jsonl', 'history'), ('atlas.osa', 'Atlas')]:
                (source / name).write_text(value)
            actions_path = root / 'actions.jsonl'
            actions_path.write_text(json.dumps(_action(0)) + '\n')
            metadata_path = actions_path.with_suffix('.meta.json')
            metadata_path.write_text(json.dumps({'replay': str(source / 'index.html'),
                                                  'video': 'mock.avi', 'fps': 30.}))
            originals = {path: path.read_bytes() for path in [*source.iterdir(), actions_path, metadata_path]}
            target = root / 'actions_replay_hybrid'
            history = [_snapshot(0), _snapshot(0, final=True)]
            with patch('sys.argv', ['tools/render_slam_replay.py', str(actions_path)]), \
                 patch('tools.render_slam_replay.load_replay_calibration', return_value=MagicMock()), \
                 patch('tools.render_slam_replay.resolve_native_history_path',
                       return_value=source / 'native_history.jsonl'), \
                 patch('tools.render_slam_replay.read_native_history', return_value=history), \
                 patch('tools.render_slam_replay.prepare_final_replay_features', return_value={}), \
                 patch('tools.render_slam_replay.write_slam_replay',
                       return_value=(target / 'process.mp4', target / 'index.html')) as render, \
                 patch('builtins.print'):
                render_slam_replay.main()
            self.assertEqual(render.call_args.args[2], target)
            self.assertTrue(render.call_args.kwargs['hybrid'])
            self.assertFalse(render.call_args.kwargs['final_map'])
            for path, data in originals.items():
                self.assertEqual(path.read_bytes(), data)
            self.assertEqual((target / 'atlas.osa').read_bytes(), originals[source / 'atlas.osa'])


if __name__ == '__main__':
    unittest.main()
