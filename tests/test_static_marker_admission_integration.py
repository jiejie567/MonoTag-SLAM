"""Focused input-preparation guards: native runner and bootstrap are mocked."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import json
import os
import unittest

import cv2
import numpy as np

from aruco_track.models import BandLayout, Calibration, Pose
from aruco_track.marker_corners import TrackedMarkerObservation
from aruco_track.orbslam3_backend import MetricOrbSlamResult
from aruco_track.slam_sequence_cache import SlamSequenceCache, sequence_cache_key
from export_action_labels import _run_deferred_head_slam


class IntegrationGuards(unittest.TestCase):
    def test_excluded_groups_cannot_flow_or_bootstrap_but_remain_masked(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "tiny.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30., (64, 48))
            self.assertTrue(writer.isOpened())
            for _ in range(2):
                writer.write(np.zeros((48, 64, 3), np.uint8))
            writer.release()
            good = np.array([[6., 16], [16, 16], [16, 26], [6, 26]])
            deferred = np.array([[42., 16], [52, 16], [52, 26], [42, 26]])
            records = root / "raw.jsonl"
            raw = dict(hands={}, unassigned_hands=[], detected_marker_corners={
                "24": good.tolist(), "27": deferred.tolist()},
                boundary_rejected_marker_corners={"27": deferred.tolist()})
            records.write_text("\n".join(json.dumps(raw) for _ in range(2)) + "\n")
            original = records.read_bytes()
            calibration = Calibration(np.array([[50., 0, 32], [0, 50., 24], [0, 0, 1]]),
                                      np.zeros(5), (64, 48))
            world = np.array([[-.024, -.024, 0], [.024, -.024, 0], [.024, .024, 0], [-.024, .024, 0]])
            layout = BandLayout("fixed", "DICT_4X4_50", {24: world, 27: world + [.2, 0, 0]})
            pose = Pose(np.zeros((3, 1)), np.array([[0.], [0.], [-.5]]), .2)
            seen = dict(tracker=0, mask=0, bootstrap=0, native=0)
            check = self

            class FakeTracker:
                def __init__(self, *_args):
                    self.tracks = {(27, 0): (deferred[0], 0.), (24, 0): (good[0], 0.)}
                    self.weak_streaks = {(27, 0): (3, 0.)}

                def update(self, _image, _time, detections, accepted, *_args, **kwargs):
                    check.assertNotIn(27, detections)
                    check.assertNotIn(27, accepted)
                    check.assertNotIn((27, 0), self.tracks)
                    check.assertIn((24, 0), self.tracks)
                    check.assertNotIn((27, 0), self.weak_streaks)
                    check.assertNotIn(27, kwargs['weak_detections'])
                    seen['tracker'] += 1
                    return TrackedMarkerObservation()

            def prepare(frame, allowed_mask):
                self.assertEqual(allowed_mask[21, 47], 0)
                seen['mask'] += 1
                return frame.copy()

            def bootstrap(*args, **kwargs):
                self.assertTrue(all(27 not in frame for frame in args[5]))
                self.assertTrue(all(27 not in ids for ids in args[8]))
                seen['bootstrap'] += 1
                return SimpleNamespace(hints_path=None, diagnostics=dict(
                    accepted=False, reason='test_no_probe', published_observations=[]))

            def native(_project, sequence, settings, output, hints, **kwargs):
                for line in hints.read_text().splitlines():
                    if line.startswith('#'):
                        continue
                    fields = line.split()
                    ids = fields[fields.index('ids') + 1:fields.index('tracked')]
                    self.assertEqual(set(ids), {'24'})
                (output / 'native.log').write_text('mock only\n')
                (output / 'frames.txt.history.jsonl').write_text('{}\n')
                seen['native'] += 1
                return {}, {}, np.empty((0, 3)), {}, {}

            def result(*args, **kwargs):
                return MetricOrbSlamResult([], np.empty((0, 3)), (), [], None, None,
                                           0, None, None, dict(args[5]))

            with patch.dict(os.environ, {'ORB_SLAM3_OFFLINE_MARKER_BOOTSTRAP': '1', 'ORB_SLAM3_PREFIX_RELOCALIZATION': '0'}), \
                    patch('aruco_track.marker_corners.MarkerCornerTracker', FakeTracker), \
                    patch('aruco_track.marker_bootstrap.bootstrap_initial_marker_observations', side_effect=bootstrap), \
                    patch('export_action_labels.SlamSequenceCache', return_value=SlamSequenceCache(root / 'cache')), \
                    patch('export_action_labels.run_orbslam3_sequence', side_effect=native), \
                    patch('export_action_labels.read_native_result', side_effect=result), \
                    patch('aruco_track.orbslam3_backend.refine_final_frame_poses', side_effect=lambda value, *_args: value), \
                    patch('export_action_labels.prepare_slam_frame', side_effect=prepare):
                _run_deferred_head_slam(video, records, calibration, [{24: good}] * 2,
                    [pose] * 2, [.9] * 2, [(24,)] * 2, layout, ['world_board'] * 2,
                    root / 'replay', 'auto', None, root / 'atlas.osa',
                    marker_weights=[{24: 1.}] * 2, excluded_marker_ids=[{27}, {27}])
            self.assertEqual(seen, dict(tracker=2, mask=2, bootstrap=1, native=1))
            self.assertEqual(original, records.read_bytes())
            self.assertFalse((root / 'atlas.osa').exists())
            common = (video, records, calibration, layout, 30., 2)
            self.assertNotEqual(sequence_cache_key(*common, excluded_marker_ids=[set(), set()]),
                                sequence_cache_key(*common, excluded_marker_ids=[{27}, set()]))
            self.assertEqual(sequence_cache_key(*common, excluded_marker_ids=[{27, 24}, set()]),
                             sequence_cache_key(*common, excluded_marker_ids=[{24, 27}, set()]))
            with self.assertRaises(ValueError):
                sequence_cache_key(*common, excluded_marker_ids=[{27}])


if __name__ == '__main__':
    unittest.main()
