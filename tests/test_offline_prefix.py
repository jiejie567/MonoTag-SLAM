import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from aruco_track.camera_state import FusedCameraFrame
from aruco_track.models import Pose, Calibration
from aruco_track.offline_prefix import initial_prefix_request, apply_prefix_candidates, recover_initial_prefix
from aruco_track.orbslam3_backend import MetricOrbSlamResult
from aruco_track.pipeline import compose_pose


def fixture():
    pose = Pose(np.zeros((3, 1)), np.array([[.1], [0], [1.]]), 0.)
    mapping = dict(id=2, revision=42, metric=True, background=True)
    frames = [FusedCameraFrame(None, "invalid", 0., 0, None) for _ in range(3)]
    frames += [FusedCameraFrame(pose, "head-slam", 1., 100, 1., "atlas_2", 42, True)]
    frames += [FusedCameraFrame(None, "invalid", 0., 0, None)]
    history = [dict(timestamp=i / 60., state=1 if i < 3 else 2 if i == 3 else 3,
                    pose=None if i != 3 else [.1, 0, 1, 0, 0, 0, 1], active_map=2) for i in range(5)]
    history.append(dict(timestamp=4 / 60., final=True, active_map=2, maps=[mapping]))
    return MetricOrbSlamResult(frames, np.empty((0, 3)), (), [None] * 5, 3,
                               1., 0, None, None, {}, history, {"atlas_2": mapping})


class OfflinePrefixTests(unittest.TestCase):
    def test_only_initial_unmeasured_prefix_requested(self):
        request = initial_prefix_request(fixture(), 60.)
        self.assertEqual(request["frames"], [0, 1, 2])
        self.assertEqual(request["boundary_frame"], 3)

    def test_support_is_bounded_same_map_and_not_a_recovery_target(self):
        result = fixture()
        result.history[3]["state"] = 6  # Marker pose seed before background keyframe.
        request = initial_prefix_request(result, 60.)
        self.assertEqual(request["support_frames"], [3])
        self.assertNotIn(4, request["frames"])  # A later lost frame remains non-target.
        result.history[4]["active_map"] = 3
        self.assertEqual(initial_prefix_request(result, 60.)["support_frames"], [3])

    def test_support_sampling_does_not_cross_new_map_or_two_seconds(self):
        result = fixture()
        result.frames.extend([result.frames[-1]] * 400)
        result.history[:] = result.history[:-1] + [
            dict(timestamp=i / 90., state=6, pose=None, active_map=2)
            for i in range(5, 405)] + [result.history[-1]]
        for i, state in enumerate(result.history[:-1]):
            state["timestamp"] = i / 90.
        request = initial_prefix_request(result, 90.)
        self.assertEqual(request["support_frames"], list(range(3, 184, 3)))
        result.history[20]["active_map"] = 3
        self.assertEqual(initial_prefix_request(result, 90.)["support_frames"], list(range(3, 20, 3)))

    def test_support_rows_cannot_publish_poses_even_if_marked_accepted(self):
        result = fixture()
        request = initial_prefix_request(result, 60.)
        base = dict(map_id=2, map_revision=42, accepted=True,
                    pose=[.08, 0, 1, 0, 0, 0, 1], inliers=50, rms_px=1.1,
                    support_only=True)
        updated = apply_prefix_candidates(result, request, [dict(base, frame=i) for i in range(5)])
        self.assertEqual(updated.frames, result.frames)

    def test_manifest_keeps_support_separate_and_masks_background_anchor(self):
        result = fixture()
        calibration = Calibration(np.array([[100., 0., 32.], [0., 100., 24.], [0., 0., 1.]]),
                                  np.zeros(5), (64, 48))
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            records = directory / 'records.jsonl'
            rows = [dict(frame=i, hands={}, detected_marker_corners={
                '20': [[i, 0], [i+1, 0], [i+1, 1], [i, 1]]}) for i in range(5)]
            records.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            atlas = directory / 'atlas.osa'
            atlas.write_bytes(b'not opened by mocked adapter')

            def run(command, **kwargs):
                manifest = json.loads(Path(command[3]).read_text())
                self.assertEqual([q['frame'] for q in manifest['queries']], [0, 1, 2])
                self.assertEqual([q['frame'] for q in manifest['support_queries']], [3])
                self.assertTrue(manifest['support_queries'][0]['original_pose_valid'])
                self.assertEqual([q['frame'] for q in manifest['mask_frames']], [3, 4])
                self.assertTrue(manifest['mask_frames'][0]['excluded_polygons'])
                Path(command[4]).write_text(json.dumps(dict(type='support', frame=3,
                    connected=True, support_only=True, accepted=False))+'\n')
                return type('Completed', (), dict(returncode=0, stdout='', stderr=''))()

            with patch('aruco_track.offline_prefix.subprocess.run', side_effect=run), \
                    patch('aruco_track.offline_prefix.Path.is_file', return_value=True):
                updated = recover_initial_prefix(result, directory, directory/'raw.mp4', atlas,
                    records, calibration, 60., directory/'diagnostics')
            self.assertEqual(updated.frames, result.frames)
            self.assertIs(updated.history, result.history)
            report = json.loads((directory/'diagnostics/prefix_localization.json').read_text())
            self.assertEqual(report['support_frames_evaluated'], 1)
            self.assertEqual(report['support_frames_connected'], 1)

    def test_no_metric_or_wrong_map_no_recovery(self):
        for field, value in (("metric", False), ("background", False), ("revision", 43)):
            result = fixture()
            result.maps["atlas_2"][field] = value
            self.assertIsNone(initial_prefix_request(result, 60.))
        result = fixture()
        result.history[1]["state"] = 3
        self.assertIsNone(initial_prefix_request(result, 60.))

    def test_pose_recovery_is_late_and_history_unchanged(self):
        result = fixture()
        history = copy.deepcopy(result.history)
        row = dict(frame=1, map_id=2, map_revision=42, accepted=True,
                   pose=[.08, 0, 1, 0, 0, 0, 1], inliers=50, rms_px=1.1)
        updated = apply_prefix_candidates(result, initial_prefix_request(result, 60.), [row])
        self.assertIsNone(result.frames[1].pose)
        self.assertEqual(updated.history, history)
        self.assertIs(updated.frames[3], result.frames[3])
        self.assertIsNone(updated.frames[4].pose)  # later genuine LOST stays invalid
        recovered = updated.frames[1]
        self.assertFalse(recovered.localization_recovery["original_tracking_valid"])
        self.assertEqual(recovered.revision, 42)
        wrist = Pose(np.zeros((3, 1)), np.array([[.02], [0], [.5]]), 0.)
        np.testing.assert_allclose(compose_pose(recovered.pose, wrist).tvec.ravel(), [.10, 0, 1.5])

    def test_bad_or_duplicate_adapter_rows_not_accepted(self):
        base = dict(frame=1, map_id=2, map_revision=42, accepted=True,
                    pose=[.08, 0, 1, 0, 0, 0, 1], inliers=50, rms_px=1.1)
        for field, value in (("map_id", 3), ("map_revision", 41), ("inliers", 29),
                             ("rms_px", float("nan")), ("rms_px", 4), ("pose", None),
                             ("accepted", False), ("frame", 4)):
            result = fixture()
            updated = apply_prefix_candidates(result, initial_prefix_request(result, 60.),
                                              [dict(base, **{field: value})])
            self.assertIsNone(updated.frames[1].pose)
            self.assertIsNone(updated.frames[4].pose)
        result = fixture()
        updated = apply_prefix_candidates(result, initial_prefix_request(result, 60.), [base, base])
        self.assertIsNone(updated.frames[1].pose)
        for row in (None, [], dict(base, frame=True), dict(base, frame="1"),
                    dict(base, frame=[]), dict(base, inliers=50.9), dict(base, accepted="false")):
            updated = apply_prefix_candidates(result, initial_prefix_request(result, 60.), [row])
            self.assertIsNone(updated.frames[1].pose)


if __name__ == "__main__":
    unittest.main()
