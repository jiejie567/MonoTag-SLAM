import copy
import unittest
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from aruco_track.camera_state import FusedCameraFrame
from aruco_track.models import Calibration, Pose
from aruco_track.orbslam3_backend import MetricOrbSlamResult, OrbSlamObservation, refine_final_frame_poses


class FinalRefinementGaugeTests(unittest.TestCase):
    def fixture(self, shift, rotation):
        K = np.array([[500., 0, 320], [0, 500, 240], [0, 0, 1]])
        cal = Calibration(K, np.zeros(5), (640, 480))
        points = np.array([[x, y, z] for z in (2.5, 3., 3.5)
                           for x, y in ((-.4, -.3), (.4, -.3), (-.4, .3), (.4, .3))])
        target = Rotation.from_euler('y', 1., degrees=True).as_matrix()
        pixels = cv2.projectPoints(points, cv2.Rodrigues(target.T)[0], np.zeros(3), K, np.zeros(5))[0].reshape(-1, 2)
        world = (rotation @ points.T).T + shift
        frame = FusedCameraFrame(Pose(cv2.Rodrigues(rotation)[0], shift.reshape(3, 1), 0.),
                                 'head-slam', 1., 12, None, 'atlas_0', 0, True, 'marker', True)
        obs = OrbSlamObservation(12, pixels, np.empty((0, 2)), 2, np.arange(12))
        result = MetricOrbSlamResult([frame], world, (), [obs], 0, 1., 0, None, None, {}, [],
                                    {'atlas_0': {'points': [[i, *p] for i, p in enumerate(world)], 'markers': {}}})
        return cal, result, rotation @ target

    def test_rigid_world_change_keeps_identical_solution(self):
        for shift, r in [(np.zeros(3), np.eye(3)), (np.array([20., -7, 2]),
                         Rotation.from_euler('xyz', [30, 50, 90], degrees=True).as_matrix())]:
            cal, result, expected = self.fixture(shift, r)
            actual = refine_final_frame_poses(result, cal, [{}], [()], [{}])
            self.assertEqual(actual.timing['dense_pose_accepted'], 1)
            np.testing.assert_allclose(actual.frames[0].pose.rotation_matrix, expected, atol=1e-6)
            np.testing.assert_allclose(actual.frames[0].pose.tvec.ravel(), shift, atol=1e-6)

    def test_committed_keyframe_exact_and_lost_not_filled(self):
        cal, result, _ = self.fixture(np.zeros(3), np.eye(3))
        result.history[:] = [dict(timestamp=0., reference=42, state=6, pose=[0, 0, 0, 0, 0, 0, 1]), dict(final=True)]
        result.maps['atlas_0']['keyframes'] = [[42, 0., [.004, 0, 0, 0, 0, 0, 1]]]
        before = copy.deepcopy(result.history)
        actual = refine_final_frame_poses(result, cal, [{}], [()], [{}])
        np.testing.assert_allclose(actual.frames[0].pose.tvec.ravel(), [.004, 0, 0])
        self.assertEqual(result.history, before)
        result.history[0]['reference'] = 99  # A loaded Atlas may reuse timestamps.
        other = refine_final_frame_poses(result, cal, [{}], [()], [{}])
        self.assertEqual(other.timing['dense_pose_keyframes_synchronized'], 0)
        result.frames[0] = FusedCameraFrame(None, 'invalid', 0., 0, None)
        self.assertIsNone(refine_final_frame_poses(result, cal, [{}], [()], [{}]).frames[0].pose)

    def test_only_changed_marker_layout_triggers_marker_refinement(self):
        cal, result, _ = self.fixture(np.zeros(3), np.eye(3))
        old = np.array([[-.05, -.05, 1], [.05, -.05, 1], [.05, .05, 1], [-.05, .05, 1]])
        pixels = cv2.projectPoints(old, np.zeros(3), np.zeros(3), cal.camera_matrix, np.zeros(5))[0].reshape(4, 2)
        identity = [0, 1., 0., 0., 0., 0., 0., 0., 1.]
        result.frames[0] = FusedCameraFrame(Pose(np.zeros((3, 1)), np.zeros((3, 1)), 0.),
                                          'marker', 1., 0, 0., 'atlas_0', 0, True, 'marker', True)
        result.observations[:] = [None]
        result.maps['atlas_0'] = dict(markers={'20': old.ravel().tolist()}, points=[])
        result.history[:] = [dict(timestamp=0., state=6, pose=[0, 0, 0, 0, 0, 0, 1], active_map=0,
                               reference=42, reference_marker_gauge=identity,
                               maps=[dict(id=0, markers={'20': old.ravel().tolist()})]),
                          dict(final=True, references=[[42, 0, [0, 0, 0, 0, 0, 0, 1], 1., identity, identity]])]
        same = refine_final_frame_poses(result, cal, [{20: pixels}], [(20,)], [{20: 1.}])
        self.assertIs(same.frames[0], result.frames[0])
        result.maps['atlas_0']['markers']['20'] = (old + [.004, 0, 0]).ravel().tolist()
        changed = refine_final_frame_poses(result, cal, [{20: pixels}], [(20,)], [{20: 1.}])
        self.assertEqual(changed.timing['dense_pose_changed_marker_refined'], 1)
        np.testing.assert_allclose(changed.frames[0].pose.tvec.ravel(), [.004, 0, 0], atol=1e-5)
        weak = refine_final_frame_poses(result, cal, [{20: pixels}], [(20,)], [{20: .25}])
        self.assertIs(weak.frames[0], result.frames[0])
