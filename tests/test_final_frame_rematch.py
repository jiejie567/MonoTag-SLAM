import copy
from dataclasses import replace
import unittest

import cv2
import numpy as np

from aruco_track.camera_state import FusedCameraFrame
from aruco_track.models import Calibration, Pose
from aruco_track.orbslam3_backend import MetricOrbSlamResult, refine_final_frame_poses
from aruco_track.final_frame_rematch import validated_observation, suspect_windows


class FinalFrameRematchTests(unittest.TestCase):
    def fixture(self):
        cal = Calibration(np.array([[500., 0, 320], [0, 500., 240], [0, 0, 1]]), np.zeros(5), (640, 480))
        points = np.array([[x, y, z] for z in (2., 2.5) for x in np.linspace(-.9, .9, 8)
                           for y in np.linspace(-.7, .7, 5)])
        pixels = cv2.projectPoints(points, np.zeros(3), np.zeros(3), cal.camera_matrix, cal.dist_coeffs)[0].reshape(-1, 2)
        mapping = dict(id=0, revision=7, metric=True, background=True,
                       points=[[i, *p] for i, p in enumerate(points)], markers={})
        features = [[*uv, i, 0.] for i, uv in enumerate(pixels)]
        row = dict(source='offline-final-frame-pose-evidence', accepted=False, support_only=True,
                   candidate=True, connected=True, map_id=0, map_revision=7, gauge='final_metric_atlas',
                   validated_after_final_atlas=True, timestamp_s=1., evidence_pose=[0, 0, 0, 0, 0, 0, 1],
                   matched_features=features, matched_feature_count=len(features), inliers=len(features),
                   matches=len(features), rms_px=0.)
        return cal, mapping, row

    def test_real_matches_required_and_contract_checked(self):
        cal, mapping, row = self.fixture()
        self.assertIsNotNone(validated_observation(row, mapping, cal, 1.))
        variants = [dict(map_id=1), dict(map_revision=8), dict(accepted=True), dict(support_only=False),
                    dict(timestamp_s=2.), dict(evidence_pose=[0, 0, 0, 0, 0, 0, 2])]
        for changes in variants:
            self.assertIsNone(validated_observation(dict(row, **changes), mapping, cal, 1.))
        corrupt = copy.deepcopy(row)
        corrupt['matched_features'][0][0] += 10
        self.assertIsNone(validated_observation(corrupt, mapping, cal, 1.))
        corrupt['matched_features'][0] = corrupt['matched_features'][1]
        self.assertIsNone(validated_observation(corrupt, mapping, cal, 1.))

    def test_rematch_can_refit_marker_frame_without_freezing_or_filling(self):
        cal, mapping, row = self.fixture()
        obs = validated_observation(row, mapping, cal, 1.)
        initial = Pose(np.zeros((3, 1)), np.array([[.01], [0.], [0.]]), 0.)
        frame = FusedCameraFrame(initial, 'marker', 1., 0, 0., 'atlas_0', 7, True, 'marker', True)
        result = MetricOrbSlamResult([frame], np.empty((0, 3)), (), [obs], 0, 1., 0, None, None, {}, [], {'atlas_0': mapping})
        normal = refine_final_frame_poses(result, cal, [{}], [()], [{}])
        self.assertIs(normal.frames[0], frame)
        updated = refine_final_frame_poses(result, cal, [{}], [()], [{}], rematched=True)
        self.assertEqual(updated.timing['dense_pose_accepted'], 1)
        np.testing.assert_allclose(updated.frames[0].pose.tvec, 0., atol=1e-6)
        missing = replace(result, frames=[replace(frame, pose=None, source='invalid')])
        self.assertIsNone(refine_final_frame_poses(missing, cal, [{}], [()], [{}], rematched=True).frames[0].pose)

    def test_selection_excludes_other_maps_and_invalid_frames(self):
        cal, mapping, _ = self.fixture()
        marker = np.array([[-.1, -.1, 1.], [.1, -.1, 1.], [.1, .1, 1.], [-.1, .1, 1.]])
        mapping['markers'] = {'20': marker.ravel().tolist()}
        pixels = cv2.projectPoints(marker, np.zeros(3), np.zeros(3), cal.camera_matrix, cal.dist_coeffs)[0].reshape(4, 2)
        f = FusedCameraFrame(Pose(np.zeros((3, 1)), np.array([[.05], [0.], [0.]]), 0.), 'head-slam', 1., 40,
                            None, 'atlas_0', 7, True, 'marker', True)
        frames = [f, replace(f, pose=None, source='invalid'), replace(f, map_id='atlas_1'), f]
        result = MetricOrbSlamResult(frames, np.empty((0, 3)), (), [None]*4, 0, 1., 0, None, None, {},
                                    [dict(final=True)], {'atlas_0': mapping})
        windows = suspect_windows(result, cal, [{20: pixels}]*4, [(20,)]*4, [{}]*4, 10.)
        self.assertEqual([i for window in windows for i in window], [0, 3])

        # A low-residual single-marker camera is still depth/tilt ambiguous.
        steady = replace(f, pose=Pose(np.zeros((3, 1)), np.zeros((3, 1)), 0.), source='marker')
        single = replace(result, frames=[steady])
        self.assertEqual(suspect_windows(single, cal, [{20: pixels}], [(20,)], [{}], 10.), [[0]])
        self.assertEqual(suspect_windows(single, cal, [{20: pixels}], [(20,)], [{20: .25}], 10.), [])
        tracked = replace(single, frames=[replace(steady, source='marker+slam')])
        self.assertEqual(suspect_windows(tracked, cal, [{20: pixels}], [(20,)], [{}], 10.), [])


if __name__ == '__main__':
    unittest.main()
