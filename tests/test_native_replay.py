import copy
import gzip
import json
from pathlib import Path
import struct
import tempfile
import zstandard as zstd
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from aruco_track.camera_state import FusedCameraFrame, make_exclusion_mask, observed_hands_for_mask
from aruco_track.models import Calibration, Pose
from aruco_track.orbslam3_backend import (
    _read_observations, camera_at_revision, read_native_history, read_native_result,
    visual_camera_at_revision,
)
from aruco_track.slam_replay import (
    _draw_orb_features, _draw_world_axes, _world_axis_pixels, _smooth_world_trail, _draw_trails,
    _TrailReplayCache, pack_history, replay_camera_frame, trails_at_revision, _events,
)
from export_action_labels import _track_world_by_submap
from verify_slam_replay import verify


def snapshot(index=0, state=2, x=0, metric=True):
    return {'timestamp': index / 30, 'final': False, 'state': state, 'active_map': 0,
            'pose': [x, 0, 0, 0, 0, 0, 1] if state in (2, 6) else None,
            'reference': 10, 'relative': [-x, 0, 0, 0, 0, 0, 1],
            'references': [[10, 0, [0, 0, 0, 0, 0, 0, 1]]],
            'maps': [{'id': 0, 'metric': metric, 'scale': 1 if metric else 0,
                      'seed': True, 'background': True, 'revision': 0,
                      'points': [[1, 0, 0, 1]], 'keyframes': [], 'markers': {},
                      'loops': [], 'merges': []}]}


class NativeReplayTests(unittest.TestCase):
    def test_native_observations_keep_final_map_point_identity_and_read_legacy(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "observations.txt"
            path.write_text("0.0 2 2 7 10 20 8 30 40\n0.1 2 2 11 21 31 41\n")
            observations = _read_observations(path)
        np.testing.assert_array_equal(observations[0.0].map_point_ids, [7, 8])
        np.testing.assert_allclose(observations[0.0].tracked_points, [[10, 20], [30, 40]])
        self.assertEqual(observations[0.1].map_point_ids.size, 0)
        np.testing.assert_allclose(observations[0.1].tracked_points, [[11, 21], [31, 41]])

    def test_marker_events_describe_actual_keyframe_and_partial_tracking(self):
        current=snapshot()
        current.update(marker_keyframe_event='first_seen:20,relocalized:21,anchor_reobserved:49',marker_event_keyframe_id=7)
        events=_events(None,current)
        self.assertIn('关键帧 7',events[0])
        self.assertIn('首次可靠观测 marker 20',events[0])
        self.assertIn('持续失跟后由 marker 恢复定位 21',events[0])
        self.assertIn('固定 marker 重新可见：保留角点关键帧 49',events[0])
        current['final']=True
        self.assertFalse(any('关键帧 7' in event for event in _events(current,current)))
        current['final']=False
        current['marker_keyframe_event']=''
        current['marker_tracking']={'partial':True,'accepted':True,'corners':3}
        self.assertTrue(any('3 点' in event for event in _events(snapshot(),current)))
        self.assertFalse(any('3 点' in event for event in _events(current,current)))

    def test_marker_pose_gate_events_explain_graph_only_and_confirmation(self):
        previous = snapshot()
        current = snapshot(1)
        current['marker_tracking'] = {'pose_constraint_reason': 'single_marker_graph_only'}
        self.assertTrue(any('不单帧拉动 Atlas 位姿' in event
                            for event in _events(previous, current)))
        self.assertFalse(_events(current, current))

        partial = snapshot(2)
        partial['marker_tracking'] = {'pose_constraint_reason': 'partial_marker_graph_only'}
        self.assertTrue(any('弱权重角点进入图优化' in event
                            for event in _events(current, partial)))

        confirming = snapshot(3)
        confirming['marker_tracking'] = {'pose_constraint_reason': 'marker_set_unconfirmed'}
        self.assertTrue(any('等待连续 3 帧' in event
                            for event in _events(partial, confirming)))

        assisted = snapshot(4)
        assisted['marker_tracking'] = {'pose_constraint_reason': 'fused_low_visual_support'}
        self.assertTrue(any('ORB 有效匹配偏少' in event
                            for event in _events(confirming, assisted)))

        recovered = snapshot(5)
        recovered['marker_tracking'] = {'pose_constraint_reason': 'fused_marker_recovery'}
        self.assertTrue(any('避免两个世界系产生接缝' in event
                            for event in _events(assisted, recovered)))

        strong = snapshot(6)
        strong['marker_tracking'] = {'pose_constraint_reason': 'fused_three_marker_support'}
        self.assertTrue(any('12+ 强角点' in event
                            for event in _events(recovered, strong)))

        conflict = snapshot(7)
        conflict['marker_tracking'] = {
            'pose_constraint_reason': 'deferred_to_marker_graph_background_conflict'}
        self.assertTrue(any('撤销本次位姿更新，保留角点' in event
                            for event in _events(strong, conflict)))
        self.assertFalse(_events(conflict, conflict))

    def test_partial_native_measurement_keeps_confidence_without_decoded_pnp(self):
        first=snapshot(state=6,x=.2)
        first['marker_tracking']={'partial':True,'accepted':True,'confidence':.22,'reprojection_px':.5}
        final=copy.deepcopy(first);final['final']=True
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'history.jsonl'
            path.write_text(json.dumps(first)+'\n'+json.dumps(final)+'\n')
            result=read_native_result(path,[None],[0.],[None],30,{})
        self.assertEqual(result.frames[0].source,'marker')
        self.assertAlmostEqual(result.frames[0].confidence,.22)
        self.assertAlmostEqual(result.frames[0].pose.tvec[0,0],.2)

    def test_bootstrap_reference_change_is_logged_without_claiming_a_new_map(self):
        previous, current = snapshot(), snapshot(1)
        current['marker_bootstrap'] = dict(reason='insufficient_matches', reference_frame=2,
            next_reference_frame=15, reference_changed=True, matches=12, triangulated=0, baseline_m=.08)
        events = _events(previous, current)
        self.assertEqual(len(events), 1)
        self.assertIn('2 → 15', events[0])
        self.assertIn('世界原点不变', events[0])
        current['marker_bootstrap']['reference_changed'] = False
        self.assertEqual(_events(previous, current), [])

    def test_world_axes_use_inverse_camera_pose_and_physical_length(self):
        calibration = Calibration(np.array([[500., 0, 320], [0, 500., 240], [0, 0, 1]]),
                                  np.zeros(5), (640, 480))
        camera = Pose(np.zeros((3, 1)), np.array([[.1], [0.], [-1.]]), 0.)
        pixels = _world_axis_pixels(camera, calibration, True)
        np.testing.assert_allclose(pixels[:3], [[270, 240], [295, 240], [270, 265]])
        # Changing the estimated camera position MUST move this diagnostic
        # overlay; there is no independent marker solve or screen-space latch.
        camera.tvec[0, 0] += .02
        shifted = _world_axis_pixels(camera, calibration, True)
        np.testing.assert_allclose(shifted[0], pixels[0] + [-10, 0])

    def test_world_axes_follow_camera_rotation(self):
        calibration = Calibration(np.array([[500., 0, 320], [0, 500., 240], [0, 0, 1]]),
                                  np.zeros(5), (640, 480))
        camera = Pose(np.array([[0.], [0.], [np.pi/2]]), np.array([[0.], [0.], [-1.]]), 0.)
        np.testing.assert_allclose(_world_axis_pixels(camera, calibration, True)[:3],
                                   [[320, 240], [320, 215], [345, 240]], atol=1e-6)

    def test_world_axes_not_drawn_without_metric_pose_or_behind_camera(self):
        calibration = Calibration(np.eye(3), np.zeros(5), (80, 80))
        camera = Pose(np.zeros((3, 1)), np.array([[0.], [0.], [1.]]), 0.)
        for pose, metric in [(None, True), (camera, False), (camera, True)]:
            image = np.zeros((80, 80, 3), np.uint8)
            self.assertFalse(_draw_world_axes(image, pose, calibration, metric))
            self.assertFalse(np.any(image))

    def test_only_tracked_native_features_are_drawn_but_all_detections_counted(self):
        observation = {'state': 2, 'features': [[20, 20, None], [50, 50, 7]]}
        before = copy.deepcopy(observation)
        image = np.zeros((80, 80, 3), np.uint8)
        self.assertEqual(_draw_orb_features(image, observation), (2, 1))
        self.assertFalse(np.any(image[15:26, 15:26]))
        self.assertGreater(image[50, 50, 1], image[50, 50, 0])
        self.assertGreater(image[45, 50, 1], image[45, 50, 0])
        self.assertEqual(observation, before)

    def test_marker_only_does_not_display_failed_orb_candidates_as_tracked(self):
        image = np.zeros((80, 80, 3), np.uint8)
        self.assertEqual(_draw_orb_features(image, {'state': 6, 'features': [[20, 20, 7]]}), (1, 0))
        self.assertFalse(np.any(image))

    def test_compact_native_features_keep_total_count_and_only_matched_pixels(self):
        observation = {'state': 2, 'feature_count': 2000,
                       'matched_features': [[50, 50, 7]]}
        before = copy.deepcopy(observation)
        image = np.zeros((80, 80, 3), np.uint8)
        self.assertEqual(_draw_orb_features(image, observation), (2000, 1))
        self.assertGreater(image[50, 50, 1], image[50, 50, 0])
        self.assertEqual(observation, before)

        image[:] = 0
        observation['state'] = 6
        self.assertEqual(_draw_orb_features(image, observation), (2000, 0))
        self.assertFalse(np.any(image))

    def test_shared_exclusion_mask_survives_old_vo_removal(self):
        corners = np.array([[20, 20], [40, 20], [40, 40], [20, 40]], float)
        hand = np.zeros((21, 3)); hand[:, :2] = [.75, .5]
        mask = make_exclusion_mask((100, 100, 3), {0: corners},
                                   {'left': {'image_landmarks_normalized': hand}})
        self.assertEqual(mask[30, 30], 0)
        self.assertEqual(mask[50, 75], 0)
        self.assertEqual(mask[90, 5], 255)

    def test_unassigned_hand_is_masked_without_creating_an_action_identity(self):
        points = np.zeros((21, 3)); points[:, :2] = [.75, .5]
        record = {'hands': {'left': {'joints': {'valid': False}}},
                  'unassigned_hands': [{'valid': True, 'image_landmarks_normalized': points.tolist()},
                                       {'valid': False}]}
        before = copy.deepcopy(record)
        hands = observed_hands_for_mask(record)
        self.assertEqual(set(hands), {'unassigned_0'})
        mask = make_exclusion_mask((100, 100, 3), {}, hands)
        self.assertEqual(mask[50, 75], 0)
        self.assertEqual(mask[90, 5], 255)
        self.assertEqual(record, before)

    def test_lost_pose_never_filled_by_future_reference(self):
        lost = snapshot(1, 3)
        lost['pose'] = [99, 0, 0, 0, 0, 0, 1]
        self.assertEqual(camera_at_revision(lost, snapshot(2)), (None, None))

    def test_past_pose_recomputed_from_corrected_keyframe(self):
        frame, revised = snapshot(x=.2), snapshot()
        revised['references'][0][2][0] = .1
        pose, map_id = camera_at_revision(frame, revised)
        np.testing.assert_allclose(pose.tvec.reshape(3), [.3, 0, 0])
        self.assertEqual(map_id, 'atlas_0')

    def test_absolute_marker_pose_not_dragged_by_background_keyframe_ba(self):
        frame, revised = snapshot(state=6, x=.2), snapshot()
        revised['references'][0][2][0] = .1
        pose, map_id = camera_at_revision(frame, revised)
        np.testing.assert_allclose(pose.tvec.reshape(3), [.2, 0, 0])
        self.assertEqual(map_id, 'atlas_0')

    def test_marker_cross_map_transform_still_applied(self):
        frame, revised = snapshot(state=6, x=.2), snapshot()
        revised['references'][0][1] = 1
        revised['references'][0][2][0] = .1
        pose, map_id = camera_at_revision(frame, revised)
        np.testing.assert_allclose(pose.tvec.reshape(3), [.3, 0, 0])
        self.assertEqual(map_id, 'atlas_1')

    def test_joint_tag_orb_pose_keeps_its_absolute_constraint(self):
        frame, revised = snapshot(x=.2), snapshot()
        frame['tag_anchored'] = True
        revised['references'][0][2][0] = .1
        pose, _ = camera_at_revision(frame, revised)
        self.assertAlmostEqual(pose.tvec[0, 0], .2)
        frame['tag_anchored'] = False
        pose, _ = camera_at_revision(frame, revised)
        self.assertAlmostEqual(pose.tvec[0, 0], .3)

    def test_pre_metric_relative_translation_scaled_once(self):
        frame, revised = snapshot(x=2, metric=False), snapshot()
        frame["reference_scale"] = 1
        revised["references"][0].append(.4)
        pose, _ = camera_at_revision(frame, revised)
        self.assertAlmostEqual(pose.tvec[0, 0], .8)
        frame["relative"][0] = -.8
        frame["reference_scale"] = .4
        pose, _ = camera_at_revision(frame, revised)
        self.assertAlmostEqual(pose.tvec[0, 0], .8)

    def test_interval_corrections_use_each_reference_scale_without_scaling_metric_wrist(self):
        revised = snapshot()
        revised['timestamp'] = 1.
        revised['maps'][0]['revision'] = 3
        revised['references'] = [
            [10, 0, [0., 0., 0., 0., 0., 0., 1.], 1.],
            [11, 0, [.9, 0., 0., 0., 0., 0., 1.], .9],
            [12, 0, [1.6, 0., 0., 0., 0., 0., 1.], .8]]
        # The native commit changes different KFs differently; there is no
        # single global scale applicable to this interval or the metric hands.
        for i, expected_x in enumerate([.1, .99, 1.68]):
            frame = snapshot(i, x=i+.1)
            frame.update(reference=10+i, reference_scale=1.,
                         relative=[-.1, 0., 0., 0., 0., 0., 1.])
            record = {'hands': {'right': {'wrist_camera_graph': {
                'translation_m': [.25, .1, 1.]}}}}
            with self.subTest(reference=frame['reference']):
                pose, map_id = camera_at_revision(frame, revised)
                self.assertEqual(map_id, 'atlas_0')
                np.testing.assert_allclose(pose.tvec.ravel(), [expected_x, 0., 0.])
                trails, _, _ = trails_at_revision(0, [frame], [record], revised, 30.)
                np.testing.assert_allclose(trails['right'], [[expected_x+.25, .1, 1.]])

    def test_rejected_interval_event_does_not_itself_transform_any_camera(self):
        frame, revised = snapshot(x=.2), snapshot()
        revised['timestamp'] = 1.
        revised['marker_graph_events'] = [{'sequence': 1, 'type': 'scale_reanchor',
                                            'status': 'rejected', 'scale': .8}]
        pose, map_id = camera_at_revision(frame, revised)
        self.assertEqual(map_id, 'atlas_0')
        np.testing.assert_allclose(pose.tvec.ravel(), [.2, 0., 0.])

    def test_marker_history_follows_only_explicit_graph_delta_not_background_ba(self):
        frame, revised = snapshot(state=6, x=.2), snapshot()
        old_graph = [7, 1.2, .3, -.2, .1, 0., 0., np.sin(np.pi/12), np.cos(np.pi/12)]
        # Delta is s=.9, Rz=90deg, t=(1,2,0), composed on the LEFT.
        new_graph = [8, 1.08, 1.18, 2.27, .09, 0., 0., np.sin(np.pi/3), np.cos(np.pi/3)]
        frame['reference_marker_graph'] = old_graph
        frame['references'][0].extend([1., old_graph])
        revised['references'][0].extend([.9, new_graph])
        revised['references'][0][2][0] = 999.  # Unrelated ordinary background BA.
        revised['maps'][0]['revision'] = 8
        before = copy.deepcopy(frame)
        camera, map_id = camera_at_revision(frame, revised)
        self.assertEqual(map_id, 'atlas_0')
        np.testing.assert_allclose(camera.tvec.ravel(), [1., 2.18, 0.], atol=1e-12)
        np.testing.assert_allclose(camera.rotation_matrix, [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-12)
        self.assertEqual(frame, before)
        record = {'hands': {'right': {'wrist_camera_graph': {'translation_m': [.25, 0., 1.]}}}}
        trails, _, _ = trails_at_revision(0, [frame], [record], revised, 30.)
        # The camera graph correction scales its old world translation only;
        # physical wrist-to-camera geometry remains .25m and 1m, not .225/.9.
        np.testing.assert_allclose(trails['right'], [[1., 2.43, 1.]], atol=1e-12)

    def test_marker_map_merge_graph_delta_follows_target_without_rescaling_local_hand(self):
        frame, revised = snapshot(state=6, x=.2), snapshot()
        identity = [0, 1., 0., 0., 0., 0., 0., 0., 1.]
        frame['reference_marker_graph'] = identity
        frame['references'][0].extend([1., identity])
        revised['references'][0].extend([1., [1, 1., 3., 0., 0., 0., 0., 0., 1.]])
        revised['references'][0][1] = 2
        revised['references'][0][2][0] = 500.
        revised['active_map'] = 2
        revised['maps'][0]['id'] = 2
        camera, map_id = camera_at_revision(frame, revised)
        self.assertEqual(map_id, 'atlas_2')
        np.testing.assert_allclose(camera.tvec.ravel(), [3.2, 0., 0.])
        record = {'hands': {'right': {'wrist_camera_graph': {'translation_m': [.25, 0., 1.]}}}}
        trails, _, map_id = trails_at_revision(0, [frame], [record], revised, 30.)
        self.assertEqual(map_id, 'atlas_2')
        np.testing.assert_allclose(trails['right'], [[3.45, 0., 1.]])

    def test_marker_graph_capture_already_corrected_is_not_applied_twice(self):
        frame, revised = snapshot(state=6, x=.2), snapshot()
        graph = [5, .9, 2., 0., 0., 0., 0., 0., 1.]
        frame['reference_marker_graph'] = graph
        frame['references'][0].extend([.9, graph])
        revised['references'][0].extend([.9, graph])
        revised['references'][0][2][0] = 999.
        camera, _ = camera_at_revision(frame, revised)
        np.testing.assert_allclose(camera.tvec.ravel(), [.2, 0., 0.])

    def test_new_protocol_missing_or_invalid_graph_reference_does_not_relabel_old_pose(self):
        frame = snapshot(state=6, x=.2)
        frame['reference_marker_graph'] = [5, 1., 0., 0., 0., 0., 0., 0., 1.]
        for graph in (None, [4, 1., 0., 0., 0., 0., 0., 0., 1.],
                      [6, 0., 0., 0., 0., 0., 0., 0., 1.],
                      [6, float('nan'), 0., 0., 0., 0., 0., 0., 1.]):
            revised = snapshot()
            revised['timestamp'] = 1.
            revised['references'] = [] if graph is None else [
                [10, 0, [0., 0., 0., 0., 0., 0., 1.], 1., graph]]
            with self.subTest(graph=graph):
                self.assertEqual(camera_at_revision(frame, revised), (None, None))
        # This remains a genuine current marker measurement even if native
        # has no replayable KF yet. No historical correction is being claimed.
        camera, map_id = camera_at_revision(frame, frame)
        self.assertEqual(map_id, 'atlas_0')
        np.testing.assert_allclose(camera.tvec.ravel(), [.2, 0., 0.])

    def test_world_wrist_kept_after_more_than_two_seconds_without_marker(self):
        pose = Pose(np.zeros((3, 1)), np.zeros((3, 1)), 0)
        frames = [FusedCameraFrame(pose, 'head-slam', .9, 90, None, 'atlas_0', metric=True)] * 150
        wrists = [pose] * 150
        result = _track_world_by_submap(frames, ['atlas_0'] * 150, wrists, wrists)
        self.assertTrue(all(p is not None for p in result.poses))
        wrists[80] = None
        result = _track_world_by_submap(frames, ['atlas_0'] * 150, wrists, wrists)
        self.assertIsNone(result.poses[80])

    def test_static_world_hand_does_not_draw_camera_motion(self):
        history = [snapshot(i, x=.01 * i) for i in range(10)]
        actions = [{'hands': {'left': {'wrist_camera_graph': {'translation_m': [1-.01*i, 0, 1]}}}}
                   for i in range(10)]
        trails, _, _ = trails_at_revision(9, history, actions, history[-1], 30)
        np.testing.assert_allclose(trails['left'], [[1, 0, 1]] * 10)

    def test_replay_prefers_offline_world_wrist_fit(self):
        history = [snapshot(i, x=.1 * i) for i in range(2)]
        identity = [1.0, 0.0, 0.0, 0.0]
        actions = []
        for index in range(2):
            actions.append({
                'camera_submap_id': 'atlas_0',
                'camera_world_pose_fused': {
                    'translation_m': [.1 * index, 0.0, 0.0],
                    'quaternion_wxyz': identity,
                },
                'hands': {'right': {
                    'world_submap_id': 'atlas_0',
                    'wrist_camera_graph': {
                        'translation_m': [2.0 * index, 0.0, 1.0],
                        'quaternion_wxyz': identity,
                    },
                    'wrist_world_graph': {
                        'translation_m': [1.0, 0.0, 1.0],
                        'quaternion_wxyz': identity,
                    },
                }},
            })
        trails, _, _ = trails_at_revision(1, history, actions, history[-1], 30)
        np.testing.assert_allclose(trails['right'], [[1.0, 0.0, 1.0]] * 2)

    def test_static_wrist_with_camera_translation_and_rotation_projects_to_one_pixel(self):
        from scipy.spatial.transform import Rotation
        from aruco_track.pipeline import inverse_pose
        history, actions = [], []
        fixed = np.array([.15, -.1, 1.2])
        for i in range(20):
            angle = Rotation.from_euler('xyz', [i*.008, i*.01, -i*.004])
            rotation = angle.as_matrix()
            position = np.array([i*.003, -i*.001, .002*i])
            camera = Pose(cv2.Rodrigues(rotation)[0], position.reshape(3, 1), 0.)
            inverse = inverse_pose(camera)
            h = snapshot(i)
            h['pose'] = position.tolist() + angle.as_quat().tolist()
            h['relative'] = inverse.tvec.ravel().tolist() + Rotation.from_matrix(inverse.rotation_matrix).as_quat().tolist()
            history.append(h)
            measured = rotation.T @ (fixed-position)
            actions.append({'hands': {'right': {'wrist_camera_graph': {'translation_m': measured.tolist()}}}})
        trails, camera, map_id = trails_at_revision(19, history, actions, history[-1], 30)
        self.assertEqual(map_id, 'atlas_0')
        np.testing.assert_allclose(trails['right'], [fixed]*20, atol=1e-12)
        calibration = Calibration(np.array([[500., 0, 320], [0, 500., 240], [0, 0, 1]]),
                                  np.array([.03, -.01, .001, -.002, 0]), (640, 480))
        inverse = inverse_pose(camera)
        expected = tuple(np.rint(cv2.projectPoints(fixed.reshape(1, 3), inverse.rvec, inverse.tvec,
                               calibration.camera_matrix, calibration.dist_coeffs)[0].reshape(2)).astype(int))
        with patch('aruco_track.slam_replay.cv2.line') as draw:
            _draw_trails(np.zeros((480, 640, 3), np.uint8), trails, camera, calibration)
        self.assertEqual(draw.call_count, 19)
        for call in draw.call_args_list:
            self.assertEqual(call.args[1], expected)
            self.assertEqual(call.args[2], expected)

    def test_trails_never_fall_back_to_camera_coordinates_when_lost_or_unscaled(self):
        history = [snapshot(i, x=.01*i) for i in range(3)]
        actions = [{'hands': {'right': {'wrist_camera_graph': {'translation_m': [1, 0, 1]}}}}]*3
        for last in [snapshot(2, state=4), snapshot(2, metric=False)]:
            history[-1] = last
            trails, _, _ = trails_at_revision(2, history, actions, last, 30)
            self.assertEqual(trails, {})

    def test_historical_camera_failure_or_unrelated_map_breaks_world_trail(self):
        history = [snapshot(i) for i in range(4)]
        actions = [{'hands': {'right': {'wrist_camera_graph': {'translation_m': [1, 0, 1]}}}}]*4
        history[1] = snapshot(1, state=4)
        history[2]['reference'] = 99
        history[-1]['references'].append([99, 1, [0, 0, 0, 0, 0, 0, 1]])
        trails, _, _ = trails_at_revision(3, history, actions, history[-1], 30)
        self.assertEqual(trails['right'], [[1, 0, 1], None, None, [1, 0, 1]])

    def test_3d_map_keeps_marker_shapes_without_text_labels(self):
        template = (Path(__file__).resolve().parents[1]/'aruco_track/slam_replay.html').read_text()
        self.assertIn('Object.values(m.markers)', template)
        self.assertNotIn('ctx.fillText', template)

    def test_3d_map_keeps_origin_keyframe_identity_after_culling(self):
        template = (Path(__file__).resolve().parents[1]/'aruco_track/slam_replay.html').read_text()
        self.assertIn('indexOriginKeyFrames(timeline)', template)
        self.assertIn("orderedKeyframes=[...m.keyframes].sort((a,b)=>a[1]-b[1]||a[0]-b[0])", template)
        self.assertIn("if(isOrigin)camera(view,k[2],'#e52424',size*4.8,3)", template)
        self.assertIn("smoothLine(view,segment,'#1769dc',.85,3)", template)
        self.assertIn('context.bezierCurveTo', template)
        self.assertIn('lastKeyframePosition=pos', template)
        self.assertIn("originPresent?'':'（已剔除）'", template)
        self.assertNotIn("last?'#1769dc':'#e52424'", template)

    def test_world_trail_filter_reduces_jitter_without_mutating_observations(self):
        values = [[(-1)**i * .002, 0, 1] for i in range(200)]
        original = copy.deepcopy(values)
        filtered = np.array(_smooth_world_trail(values, 60))
        self.assertLess(filtered[20:, 0].std(), .001)
        self.assertEqual(values, original)

    def test_world_trail_filter_has_no_future_samples_and_does_not_fill_gaps(self):
        prefix = [[i*.001, 0, 1] for i in range(10)]
        values = prefix + [None, None, [5, 0, 1], None]
        filtered = _smooth_world_trail(values, 60)
        self.assertEqual(filtered[:10], _smooth_world_trail(prefix, 60))
        self.assertEqual(filtered[10:], [None, None, [5, 0, 1], None])

    def test_world_trail_motion_lag_is_below_one_output_frame(self):
        for fps in [30, 60, 120]:
            speed = .5
            values = np.array([[speed*i/fps, 0, 1] for i in range(fps*2)])
            filtered = np.array(_smooth_world_trail(values.tolist(), fps))
            lag_s = (values[:, 0] - filtered[:, 0]) / speed
            self.assertLess(float(lag_s.max()), 1/30)

    def test_world_trail_filter_is_recomputed_after_map_revision(self):
        history = [snapshot(i, x=.01*i) for i in range(10)]
        actions = [{'hands': {'left': {'wrist_camera_graph': {'translation_m': [1-.01*i, 0, 1]}}}}
                   for i in range(10)]
        original, _, _ = trails_at_revision(9, history, actions, history[-1], 30)
        revision = copy.deepcopy(history[-1])
        revision['references'][0][2][0] = .5
        corrected, _, _ = trails_at_revision(9, history, actions, revision, 30)
        np.testing.assert_allclose(np.array(corrected['left']) - original['left'], [[.5, 0, 0]]*10)

    def test_trail_cache_matches_direct_replay_and_reuses_revision_context(self):
        history = [snapshot(i, x=.001*i) for i in range(50)]
        actions = [{'hands': {'left': {'wrist_camera_graph': {
            'translation_m': [1-.001*i, 0, 1]}}}} for i in range(50)]
        expected = [trails_at_revision(i, history, actions, history[i], 30)
                    for i in range(50)]
        cache = _TrailReplayCache(history, actions, 30)
        with patch('aruco_track.slam_replay.replay_camera_frame', wraps=replay_camera_frame) as replay:
            actual = [cache.resolve(i, history[i])[1:] for i in range(50)]
            calls_after_first_pass = replay.call_count
            cache.resolve(49, history[49])
            self.assertEqual(replay.call_count, calls_after_first_pass)
        self.assertLess(calls_after_first_pass, 120)
        for wanted, got in zip(expected, actual):
            expected_trails, expected_camera, expected_map = wanted
            trails, camera, map_id = got
            self.assertEqual(map_id, expected_map)
            self.assertEqual(trails, expected_trails)
            np.testing.assert_allclose(camera.tvec, expected_camera.tvec)
            np.testing.assert_allclose(camera.rotation_matrix, expected_camera.rotation_matrix)

    def test_point_birth_update_delete_checkpoint_round_trip(self):
        history = [snapshot(0), snapshot(1), snapshot(2), snapshot(180)]
        history[0]['maps'][0]['points'] = []
        history[2]['maps'][0]['points'] = [[1, .5, 0, 1], [2, 100, 0, 1]]
        history[3]['maps'][0]['points'] = [[2, 100, 0, 1]]
        with tempfile.TemporaryDirectory() as directory:
            rows = pack_history(history, Path(directory), 30)
            data = gzip.decompress((Path(directory) / 'points.bin.gz').read_bytes())
            points = {}
            for row, expected in zip(rows, history):
                if row['checkpoint']:
                    points.clear()
                for i in range(row['count']):
                    m, pid, x, y, z = struct.unpack_from('<QQfff', data, row['offset'] + i * 28)
                    points[(m, pid)] = [x, y, z]
                for m, pid in row['deleted']:
                    del points[(m, pid)]
                self.assertEqual(points, {(0, p[0]): p[1:] for p in expected['maps'][0]['points']})
        self.assertTrue(rows[-1]['checkpoint'])

    def test_native_compact_point_publications_pack_to_same_viewer_state(self):
        history = [snapshot(i) for i in range(4)]
        history[0]['maps'][0].update(points_mode='full', point_count=1,
                                     points=[[1, 0, 0, 1]], deleted_points=[])
        history[1]['maps'][0].update(points_mode='delta', point_count=2,
                                     points=[[1, .5, 0, 1], [2, 2, 0, 1]], deleted_points=[])
        history[2]['maps'][0].update(points_mode='delta', point_count=1,
                                     points=[], deleted_points=[1])
        history[3]['final'] = True
        history[3]['maps'][0].update(points_mode='full', point_count=1,
                                     points=[[2, 2, 0, 1]], deleted_points=[])
        expected = [
            {(0, 1): [0, 0, 1]},
            {(0, 1): [.5, 0, 1], (0, 2): [2, 0, 1]},
            {(0, 2): [2, 0, 1]},
            {(0, 2): [2, 0, 1]},
        ]
        with tempfile.TemporaryDirectory() as directory:
            rows = pack_history(history, Path(directory), 30)
            data = gzip.decompress((Path(directory) / 'points.bin.gz').read_bytes())
            points = {}
            for row, wanted in zip(rows, expected):
                if row['checkpoint']:
                    points.clear()
                for i in range(row['count']):
                    m, pid, x, y, z = struct.unpack_from('<QQfff', data, row['offset'] + i * 28)
                    points[(m, pid)] = [x, y, z]
                for m, pid in row['deleted']:
                    del points[(m, pid)]
                self.assertEqual(points, wanted)
                self.assertEqual(row['maps'][0]['point_count'], len(wanted))
        self.assertTrue(rows[-1]['checkpoint'])

    def test_native_history_reader_streams_blank_lines_and_final_compact_checkpoint(self):
        frame, final = snapshot(), snapshot(1)
        frame['maps'][0].update(points_mode='full', point_count=1, deleted_points=[])
        final['final'] = True
        final['maps'][0].update(points_mode='full', point_count=1, deleted_points=[])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'native.jsonl.gz'
            with gzip.open(path, 'wt') as stream:
                stream.write(json.dumps(frame) + '\n\n' + json.dumps(final) + '\n')
            self.assertEqual(read_native_history(path), [frame, final])
            result = read_native_result(path, [None], [0], [None], 30, {})
        np.testing.assert_allclose(result.world_points, [[0, 0, 1]])

    def test_native_history_reader_supports_zstd_stream(self):
        frame, final = snapshot(), snapshot(1)
        frame['maps'][0].update(points_mode='full', point_count=1, deleted_points=[])
        final['final'] = True
        final['maps'][0].update(points_mode='full', point_count=1, deleted_points=[])
        payload = ('\n'.join(json.dumps(item) for item in (frame, final)) + '\n').encode()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'native_history.jsonl.zst'
            path.write_bytes(zstd.ZstdCompressor(level=1).compress(payload))
            self.assertEqual(read_native_history(path), [frame, final])

    def test_native_result_rejects_delta_only_final_map(self):
        frame, final = snapshot(), snapshot(1)
        final['final'] = True
        final['maps'][0].update(points_mode='delta', point_count=1, deleted_points=[])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'native.jsonl'
            path.write_text(json.dumps(frame) + '\n' + json.dumps(final) + '\n')
            with self.assertRaisesRegex(RuntimeError, 'final.*full'):
                read_native_result(path, [None], [0], [None], 30, {})

    def test_saved_replay_verifier_accepts_compact_native_point_history(self):
        frame, final = snapshot(), snapshot(1)
        frame['maps'][0].update(points_mode='full', point_count=1, deleted_points=[])
        final['final'] = True
        final['maps'][0].update(points_mode='full', point_count=1,
                                points=[[1, .25, 0, 1]], deleted_points=[])
        history = [frame, final]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            pack_history(history, directory, 30)
            with gzip.open(directory / 'native_history.jsonl.gz', 'wt') as stream:
                for publication in history:
                    stream.write(json.dumps(publication) + '\n')
            with gzip.open(directory / 'video_frames.json.gz', 'wt') as stream:
                json.dump([{'source_frame': 0, 'sequence': 0, 'tail': False}], stream)
            (directory / 'manifest.json').write_text(json.dumps({
                'frames': 1, 'analysis_frames': 1, 'source_fps': 30, 'fps': 30}))
            report = verify(directory)
        self.assertEqual(report['verified_publications'], 2)
        self.assertEqual(report['final_map_counts'][0]['points'], 1)

    def test_arbitrary_scale_never_labelled_metres(self):
        history = [snapshot(metric=False), snapshot(metric=False)]
        history[-1]['final'] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'native.jsonl'
            path.write_text('\n'.join(json.dumps(h) for h in history))
            result = read_native_result(path, [None], [0], [None], 30, {})
        self.assertIsNone(result.frames[0].pose)
        self.assertIsNone(result.scale_m_per_slam_unit)
        self.assertEqual(len(result.world_points), 1)

    def test_final_metricization_recovers_resolvable_historical_camera_pose(self):
        acquired = snapshot(metric=False, x=.2)
        final = copy.deepcopy(acquired)
        final['final'] = True
        final['maps'][0].update(metric=True, scale=.5, revision=1)
        final['references'][0].extend([.5])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'native.jsonl'
            path.write_text('\n'.join(json.dumps(h) for h in (acquired, final)))
            result = read_native_result(path, [None], [0], [None], 30, {})
        frame = result.frames[0]
        self.assertIsNotNone(frame.pose)
        self.assertTrue(frame.metric)
        self.assertTrue(frame.metric_recovered_later)
        self.assertEqual(frame.source, 'head-slam')

    def test_fusion_source_requires_native_constraint_not_just_visible_marker(self):
        marker = Pose(np.zeros((3, 1)), np.zeros((3, 1)), 0.)
        for applied, expected in [(False, 'head-slam'), (True, 'marker+slam')]:
            history = [snapshot(), snapshot()]
            history[0]['tag_anchored'] = applied
            history[-1]['final'] = True
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'native.jsonl'
                path.write_text('\n'.join(json.dumps(h) for h in history))
                result = read_native_result(path, [marker], [1.], [None], 30, {})
            self.assertEqual(result.frames[0].source, expected)


class MarkerGaugeReplayTests(unittest.TestCase):
    @staticmethod
    def frame(index=0, x=.12, reference=10, map_id=0, state=2):
        frame = snapshot(index, state, x)
        identity = [0, 1., 0., 0., 0., 0., 0., 0., 1.]
        frame.update(tag_anchored=True, reference=reference, active_map=map_id,
                     reference_scale=1., reference_marker_graph=identity.copy(),
                     reference_marker_gauge=identity.copy())
        frame['maps'][0]['id'] = map_id
        frame['references'] = [[reference, map_id, [.1, 0., 0., 0., 0., 0., 1.],
                                1., identity.copy(), identity.copy()]]
        return frame

    def test_metric_marker_pose_is_not_rescaled_about_an_unchanged_reference(self):
        for state in (2, 6):
            frame = self.frame(state=state)
            revised = self.frame(index=1, state=state)
            # The reference stays at .10m, but its background depth unit
            # changes by .85. Applying that visual Sim3 to .12m makes .117m.
            revised['references'][0][3] = .85
            revised['references'][0][4] = [1, .85, .015, 0., 0., 0., 0., 0., 1.]
            with self.subTest(state=state):
                camera, map_id = camera_at_revision(frame, revised)
                self.assertEqual(map_id, 'atlas_0')
                np.testing.assert_allclose(camera.tvec.ravel(), [.12, 0., 0.], atol=1e-12)

    def test_rigid_merge_moves_source_but_not_target_or_unrelated_marker_maps(self):
        source = self.frame(x=.2, reference=10, map_id=1)
        target = self.frame(x=2., reference=11)
        third = self.frame(x=-.3, reference=12, map_id=3)
        revised = self.frame(index=2)
        identity = source['reference_marker_gauge']
        merged = [1, 1., 1., 2., 0., 0., 0., np.sqrt(.5), np.sqrt(.5)]
        poison_graph = [1, .7, 99., 88., 77., 0., 0., 0., 1.]
        revised['maps'].append(copy.deepcopy(third['maps'][0]))
        revised['references'] = [
            [10, 0, [999., 0., 0., 0., 0., 0., 1.], .7, poison_graph, merged],
            [11, 0, [999., 0., 0., 0., 0., 0., 1.], .7, poison_graph, identity],
            [12, 3, [999., 0., 0., 0., 0., 0., 1.], 1., identity, identity]]
        before = copy.deepcopy([source, target, third, revised])
        for frame, expected, map_id in [(source, [1., 2.2, 0.], 'atlas_0'),
                                         (target, [2., 0., 0.], 'atlas_0'),
                                         (third, [-.3, 0., 0.], 'atlas_3')]:
            with self.subTest(source_map=frame['active_map']):
                camera, resolved = camera_at_revision(frame, revised)
                self.assertEqual(resolved, map_id)
                np.testing.assert_allclose(camera.tvec.ravel(), expected, atol=1e-12)
        camera, _ = camera_at_revision(source, revised)
        np.testing.assert_allclose(camera.rotation_matrix, [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-12)
        record = {'hands': {'right': {'wrist_camera_graph': {'translation_m': [.25, 0., 1.]}}}}
        trails, _, _ = trails_at_revision(0, [source], [record], revised, 30.)
        np.testing.assert_allclose(trails['right'], [[1., 2.45, 1.]], atol=1e-12)
        self.assertEqual([source, target, third, revised], before)

    def test_consecutive_rigid_merges_apply_only_the_uncaptured_gauge_delta(self):
        before = self.frame(x=.2, map_id=1)
        captured = self.frame(index=1)
        captured['pose'] = [1., 2.2, 0., 0., 0., np.sqrt(.5), np.sqrt(.5)]
        first_gauge = [1, 1., 1., 2., 0., 0., 0., np.sqrt(.5), np.sqrt(.5)]
        captured['reference_marker_gauge'] = first_gauge
        captured['references'][0][5] = first_gauge
        final = self.frame(index=2, map_id=2)
        # Second merge: Rz=-90deg, t=(3,0,.5); its cumulative composition
        # with the first merge is R=I, t=(5,-1,.5).
        final['references'][0][5] = [2, 1., 5., -1., .5, 0., 0., 0., 1.]
        for frame in (before, captured):
            camera, map_id = camera_at_revision(frame, final)
            self.assertEqual(map_id, 'atlas_2')
            np.testing.assert_allclose(camera.tvec.ravel(), [5.2, -1., .5], atol=1e-12)
            np.testing.assert_allclose(camera.rotation_matrix, np.eye(3), atol=1e-12)
        same_gauge = copy.deepcopy(captured)
        same_gauge['timestamp'] += 1.
        same_gauge['references'][0][2][0] = 999.
        camera, _ = camera_at_revision(captured, same_gauge)
        np.testing.assert_allclose(camera.tvec.ravel(), captured['pose'][:3], atol=1e-12)

    def test_invalid_rigid_gauges_fail_closed_without_visual_graph_fallback(self):
        invalid = [None, [0, 1.], [0, .85, 0., 0., 0., 0., 0., 0., 1.],
                   [0, 1., float('nan'), 0., 0., 0., 0., 0., 1.],
                   [0, 1., 0., 0., 0., 0., 0., 0., 0.]]
        for gauge in invalid:
            for side in ('captured', 'committed'):
                frame, revised = self.frame(), self.frame(index=1)
                if side == 'captured':
                    frame['reference_marker_gauge'] = gauge
                else:
                    revised['references'][0][5] = gauge
                with self.subTest(gauge=gauge, side=side):
                    self.assertEqual(camera_at_revision(frame, revised), (None, None))
        frame, revised = self.frame(), self.frame(index=1)
        frame['reference_marker_gauge'][0] = 2
        revised['references'][0][5][0] = 1
        self.assertEqual(camera_at_revision(frame, revised), (None, None))
        # A unit ratio of one is insufficient: neither absolute gauge is
        # allowed to contain scale, even if both contain the same wrong value.
        frame['reference_marker_gauge'] = [1, .85, 0., 0., 0., 0., 0., 0., 1.]
        revised['references'][0][5] = frame['reference_marker_gauge'].copy()
        self.assertEqual(camera_at_revision(frame, revised), (None, None))

    def test_new_gauge_protocol_missing_capture_or_reference_never_relabels_history(self):
        for missing in ('captured', 'reference', 'committed', 'committed_and_publication'):
            frame, revised = self.frame(), self.frame(index=1)
            if missing == 'captured':
                frame.pop('reference_marker_gauge')
            elif missing == 'reference':
                revised['references'] = []
            else:
                revised['references'][0].pop()
                if missing == 'committed_and_publication':
                    revised.pop('reference_marker_gauge')
            with self.subTest(missing=missing):
                self.assertEqual(camera_at_revision(frame, revised), (None, None))
        # Missing history is not the same as a missing CURRENT measurement.
        frame['reference_marker_gauge'] = None
        frame['reference'] = None
        frame['references'] = []
        camera, map_id = camera_at_revision(frame, frame)
        self.assertEqual(map_id, 'atlas_0')
        np.testing.assert_allclose(camera.tvec.ravel(), [.12, 0., 0.])

    def test_visual_history_still_uses_local_reference_scale_not_marker_gauge(self):
        frame, revised = self.frame(), self.frame(index=1)
        frame.update(tag_anchored=False, relative=[-.2, 0., 0., 0., 0., 0., 1.],
                     reference_marker_gauge=None)
        revised['references'][0][2][0] = 3.
        revised['references'][0][3] = .5
        revised['references'][0][5] = None
        camera, map_id = camera_at_revision(frame, revised)
        self.assertEqual(map_id, 'atlas_0')
        np.testing.assert_allclose(camera.tvec.ravel(), [3.1, 0., 0.])

    def test_pre_marker_visual_pose_resolves_independently_of_marker_pose(self):
        frame, revised = snapshot(state=6, x=.2), snapshot()
        frame.update(reference_scale=1., visual_relative=[-.35, 0., 0., 0., 0., 0., 1.])
        revised['references'][0][2][0] = .1
        marker_camera, _ = camera_at_revision(frame, revised)
        visual_camera, map_id = visual_camera_at_revision(frame, revised)
        self.assertEqual(map_id, 'atlas_0')
        np.testing.assert_allclose(marker_camera.tvec.ravel(), [.2, 0., 0.])
        np.testing.assert_allclose(visual_camera.tvec.ravel(), [.45, 0., 0.])

    def test_pre_marker_visual_pose_uses_reference_unit_stamp_once(self):
        frame, revised = snapshot(state=6, x=.2), snapshot()
        frame.update(reference_scale=2., visual_relative=[-1., 0., 0., 0., 0., 0., 1.])
        revised['references'][0].append(.5)
        visual_camera, _ = visual_camera_at_revision(frame, revised)
        np.testing.assert_allclose(visual_camera.tvec.ravel(), [.25, 0., 0.])

    def test_missing_pre_marker_visual_pose_is_not_invented(self):
        frame = snapshot(state=6, x=.2)
        self.assertEqual(visual_camera_at_revision(frame, snapshot()), (None, None))


if __name__ == '__main__':
    unittest.main()
