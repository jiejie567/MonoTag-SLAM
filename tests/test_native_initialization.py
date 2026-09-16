"""Native integration tests: RUN_NATIVE_SLAM_TESTS=1 python -m unittest ..."""
import json
import os
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from aruco_track.models import BandLayout, Calibration, Pose
from aruco_track.marker_corners import TrackedMarkerObservation
from aruco_track.orbslam3_backend import run_orbslam3_sequence, write_orbslam3_settings, write_tag_observation_hints


@unittest.skipUnless(os.environ.get('RUN_NATIVE_SLAM_TESTS') == '1', 'explicit native integration run')
class NativeInitializationTests(unittest.TestCase):
    def test_new_marker_component_metricizes_the_active_arbitrary_map(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sequence, output = root / "sequence", root / "output"
            sequence.mkdir(); output.mkdir()
            calibration = Calibration(
                np.array([[500., 0, 320], [0, 500., 240], [0, 0, 1]]),
                np.zeros(5), (640, 480),
            )
            rng = np.random.default_rng(654)
            background_points = np.column_stack((
                rng.uniform(-1.8, 1.8, 1400),
                rng.uniform(-1.2, 1.2, 1400),
                rng.uniform(1.8, 5.0, 1400),
            ))
            patches = rng.choice(
                np.array([35, 220], np.uint8), size=(1400, 9, 9)
            )
            count = 310
            camera_world_x = np.zeros(count)
            camera_world_x[120:] = np.linspace(0., .35, count - 120)
            marker_world_x = .35
            square = np.array([
                [-.024, -.024, 0], [.024, -.024, 0],
                [.024, .024, 0], [-.024, .024, 0],
            ])
            layouts = {
                "area_a": BandLayout("area_a", "DICT_4X4_50", {20: square}),
                "area_b": BandLayout("area_b", "DICT_4X4_50", {30: square}),
            }
            lines, poses, confidences, detections, accepted, components = [], [], [], [], [], []
            for index, world_x in enumerate(camera_world_x):
                relative = f"{index:04d}.png"
                image = np.full((480, 640), 127, np.uint8)
                if index >= 120:
                    camera_points = background_points - np.array([world_x, 0., 0.])
                    pixels = np.column_stack((
                        500. * camera_points[:, 0] / camera_points[:, 2] + 320.,
                        500. * camera_points[:, 1] / camera_points[:, 2] + 240.,
                    ))
                    for point_index, (u, v) in enumerate(pixels):
                        x, y = round(u), round(v)
                        if 4 <= x < 636 and 4 <= y < 476:
                            image[y - 4:y + 5, x - 4:x + 5] = patches[point_index]
                cv2.imwrite(str(sequence / relative), image)
                lines.append(f"{index / 30:.9f} {relative}")
                marker_id = 20 if index < 15 else (30 if index >= 230 else None)
                visible = marker_id is not None
                local_camera_x = 0. if marker_id == 20 else world_x - marker_world_x
                poses.append(Pose(
                    np.zeros((3, 1)),
                    np.array([[local_camera_x], [0.], [-.5]]), 0.,
                    marker_ids=(marker_id,),
                ) if visible else None)
                if visible:
                    pixels = cv2.projectPoints(
                        square, np.zeros(3),
                        np.array([-local_camera_x, 0., .5]),
                        calibration.camera_matrix, np.zeros(5),
                    )[0].reshape(4, 2)
                    detections.append({marker_id: pixels}); accepted.append((marker_id,))
                else:
                    detections.append({}); accepted.append(())
                confidences.append(1. if visible else 0.)
                components.append(
                    "area_a" if marker_id == 20 else
                    "area_b" if marker_id == 30 else None
                )
            (sequence / "rgb.txt").write_text("\n".join(lines) + "\n")
            write_tag_observation_hints(
                root / "tags.txt", poses, confidences, detections, accepted,
                BandLayout("empty", "DICT_4X4_50", {}), calibration, 30., 0,
                include_ids=True, marker_layouts=layouts,
                marker_component_ids=components,
            )
            write_orbslam3_settings(
                root / "camera.yaml", calibration, 30., save_atlas=root / "atlas.osa"
            )
            run_orbslam3_sequence(
                Path(__file__).resolve().parents[1], sequence,
                root / "camera.yaml", output, root / "tags.txt",
            )
            history = [
                json.loads(line)
                for line in (output / "frames.txt.history.jsonl").read_text().splitlines()
            ]
            live_maps = [mapping for mapping in history[-1]["maps"] if not mapping.get("bad")]
            self.assertEqual(len(live_maps), 2)
            active = next(
                mapping for mapping in live_maps
                if mapping["id"] == history[-1]["active_map"]
            )
            self.assertTrue(active["metric"])
            self.assertIn("30", active["markers"])
            self.assertIn(
                "by metricizing the continuously tracked map",
                (output / "native.log").read_text(),
            )

    def test_disconnected_marker_components_seed_separate_metric_maps(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sequence, output = root / "sequence", root / "output"
            sequence.mkdir(); output.mkdir()
            calibration = Calibration(
                np.array([[500., 0, 320], [0, 500., 240], [0, 0, 1]]),
                np.zeros(5), (640, 480),
            )
            image = np.zeros((480, 640, 3), np.uint8)
            cv2.imwrite(str(sequence / "frame.png"), image)
            count = 10
            (sequence / "rgb.txt").write_text(
                "\n".join(f"{i/30:.9f} frame.png" for i in range(count)) + "\n"
            )
            square = np.array([
                [-.024, -.024, 0], [.024, -.024, 0],
                [.024, .024, 0], [-.024, .024, 0],
            ])
            layouts = {
                "room_a": BandLayout("room_a", "DICT_4X4_50", {48: square}),
                "room_b": BandLayout("room_b", "DICT_4X4_50", {20: square}),
            }
            camera_pose = Pose(
                np.zeros((3, 1)), np.array([[0.], [0.], [-.5]]), 0.
            )
            pixels = cv2.projectPoints(
                square, np.zeros(3), np.array([0., 0., .5]),
                calibration.camera_matrix, np.zeros(5),
            )[0].reshape(4, 2)
            components = ["room_a"] * 4 + ["room_b"] * 6
            marker_ids = [48] * 4 + [20] * 6
            write_tag_observation_hints(
                root / "tags.txt", [camera_pose] * count, [1.] * count,
                [{mid: pixels} for mid in marker_ids],
                [(mid,) for mid in marker_ids],
                BandLayout("empty", "DICT_4X4_50", {}), calibration, 30., 0,
                include_ids=True, marker_layouts=layouts,
                marker_component_ids=components,
            )
            write_orbslam3_settings(
                root / "camera.yaml", calibration, 30., save_atlas=root / "atlas.osa"
            )
            run_orbslam3_sequence(
                Path(__file__).resolve().parents[1], sequence,
                root / "camera.yaml", output, root / "tags.txt",
            )
            history = [
                json.loads(line)
                for line in (output / "frames.txt.history.jsonl").read_text().splitlines()
            ]
            metric_maps = [mapping for mapping in history[-1]["maps"] if mapping["metric"]]
            self.assertEqual(len(metric_maps), 2)
            self.assertEqual(
                {frozenset(mapping["markers"]) for mapping in metric_maps},
                {frozenset({"48"}), frozenset({"20"})},
            )
            self.assertIn(
                "Disconnected marker component room_b: no metric visual bridge",
                (output / "native.log").read_text(),
            )

    def test_non_covisible_markers_register_through_continuous_metric_track(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sequence, output = root / "sequence", root / "output"
            sequence.mkdir(); output.mkdir()
            calibration = Calibration(
                np.array([[500., 0, 320], [0, 500., 240], [0, 0, 1]]),
                np.zeros(5), (640, 480),
            )
            texture = np.random.default_rng(321).integers(
                0, 256, (480, 2400), dtype=np.uint8
            )
            count = 120
            camera_world_x = np.concatenate((
                np.zeros(10), np.linspace(0., .12, 40),
                np.linspace(.12, .20, 30), np.full(40, .20)
            ))
            square = np.array([
                [-.024, -.024, 0], [.024, -.024, 0],
                [.024, .024, 0], [-.024, .024, 0],
            ])
            marker_b_world_x = .20
            layouts = {
                "area_a": BandLayout("area_a", "DICT_4X4_50", {20: square}),
                "area_b": BandLayout("area_b", "DICT_4X4_50", {30: square}),
            }
            lines, poses, confidences, detections, accepted, components = [], [], [], [], [], []
            for index, world_x in enumerate(camera_world_x):
                left = 800 + round(world_x * 1000)
                relative = f"{index:04d}.png"
                cv2.imwrite(str(sequence / relative), texture[:, left:left + 640])
                lines.append(f"{index / 30:.9f} {relative}")
                marker_id = 20 if index < 50 else (30 if index >= 80 else None)
                component = "area_a" if marker_id == 20 else ("area_b" if marker_id == 30 else None)
                local_camera_x = world_x - (marker_b_world_x if marker_id == 30 else 0.)
                visible = marker_id is not None
                pose = Pose(
                    np.zeros((3, 1)), np.array([[local_camera_x], [0.], [-.5]]), 0.
                ) if visible else None
                if visible:
                    pixels = cv2.projectPoints(
                        square, np.zeros(3), np.array([-local_camera_x, 0., .5]),
                        calibration.camera_matrix, np.zeros(5),
                    )[0].reshape(4, 2)
                    detections.append({marker_id: pixels})
                    accepted.append((marker_id,))
                else:
                    detections.append({}); accepted.append(())
                poses.append(pose)
                confidences.append(1. if visible else 0.)
                components.append(component)
            (sequence / "rgb.txt").write_text("\n".join(lines) + "\n")
            write_tag_observation_hints(
                root / "tags.txt", poses, confidences, detections, accepted,
                BandLayout("empty", "DICT_4X4_50", {}), calibration, 30., 0,
                include_ids=True, marker_layouts=layouts,
                marker_component_ids=components,
            )
            write_orbslam3_settings(
                root / "camera.yaml", calibration, 30., save_atlas=root / "atlas.osa"
            )
            run_orbslam3_sequence(
                Path(__file__).resolve().parents[1], sequence,
                root / "camera.yaml", output, root / "tags.txt",
            )
            history = [
                json.loads(line)
                for line in (output / "frames.txt.history.jsonl").read_text().splitlines()
            ]
            metric_maps = [mapping for mapping in history[-1]["maps"] if mapping["metric"]]
            native_log = (output / "native.log").read_text()
            self.assertEqual(
                len(metric_maps), 1,
                [(item["id"], item["metric"], item["background"],
                  len(item["keyframes"]), item.get("point_count"))
                 for item in history[-1]["maps"]] + [native_log[-2000:]],
            )
            self.assertEqual(set(metric_maps[0]["markers"]), {"20", "30"})
            registered = np.asarray(metric_maps[0]["markers"]["30"]).reshape(4, 3)
            np.testing.assert_allclose(registered[:, 0], square[:, 0] + marker_b_world_x, atol=.015)
            self.assertIn(
                "Registered marker component area_b through continuous metric SLAM trajectory",
                native_log,
            )

    def test_changed_seed_view_bootstraps_without_moving_world_origin(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sequence, output = root/'sequence', root/'output'
            sequence.mkdir(); output.mkdir()
            calibration = Calibration(np.array([[500., 0, 320], [0, 500., 240], [0, 0, 1]]),
                                      np.zeros(5), (640, 480))
            texture = np.random.default_rng(123).integers(0, 256, (480, 3200), dtype=np.uint8)
            # First look at a disjoint area, then stay still in a new view.
            # Both views have a known tag, but only the later view overlaps
            # subsequent translating frames. The old fixed seed cannot match.
            xs = [-.8]*10 + [0.]*16 + [i*.006 for i in range(1, 15)] + [.084]*100
            # A later marker-assisted recovery has a real 4 cm view change,
            # but no new/reappearing tag event. Ordinary visual keyframe rules
            # must still run on that early-return recovery path.
            xs[53:] = [.124]*(len(xs)-53)
            local = np.array([[-.024, -.024, 0], [.024, -.024, 0],
                              [.024, .024, 0], [-.024, .024, 0]])
            layout = BandLayout('world', 'DICT_4X4_50', {20: local+[-.8, 0, 0], 21: local})
            poses, confidences, detections, accepted, lines = [], [], [], [], []
            for i, x in enumerate(xs):
                left = 1280 + round(x*1000)
                image = texture[:, left:left+640]
                # Lose background features while the tag still localizes.
                # The next measured tag/partial-tag frame must recover ORB,
                # not remain stuck in a zero-association local-map search.
                if i in (53,62):
                    image=np.zeros_like(image)
                relative = f'{i:04d}.png'
                cv2.imwrite(str(sequence/relative), image)
                lines.append(f'{i/30:.9f} {relative}')
                mid = 20 if i < 10 else 21
                visible = i < 45 or i in (53,54,62)
                pose = Pose(np.zeros((3, 1)), np.array([[x], [0.], [-.5]]), 0.)
                corners = cv2.projectPoints(layout.markers[mid], np.zeros(3), np.array([-x, 0, .5]),
                                           calibration.camera_matrix, np.zeros(5))[0].reshape(4, 2)
                poses.append(pose if visible else None)
                # A reliable single-tag observation is capped below .5 by
                # the Python estimator when no temporal pose prior exists.
                confidences.append(.4 if visible else 0.)
                detections.append({mid: corners} if visible else {})
                accepted.append((mid,) if visible else ())
            (sequence/'rgb.txt').write_text('\n'.join(lines)+'\n')
            tracked=[TrackedMarkerObservation() for _ in xs]
            corners=cv2.projectPoints(layout.markers[21],np.zeros(3),np.array([-xs[63],0,.5]),
                                     calibration.camera_matrix,np.zeros(5))[0].reshape(4,2)
            tracked[63]=TrackedMarkerObservation(
                Pose(np.zeros((3,1)),np.array([[xs[63]],[0.],[-.5]]),0.),.3,True,
                layout.markers[21][1:].tolist(),corners[1:].tolist(),[21]*3,[1,2,3],'tracked',1/30)
            write_tag_observation_hints(root/'tags.txt', poses, confidences, detections, accepted,
                                        layout, calibration, 30, 0, include_ids=True,tracked_observations=tracked)
            write_orbslam3_settings(root/'camera.yaml', calibration, 30, save_atlas=root/'atlas.osa')
            run_orbslam3_sequence(Path(__file__).resolve().parents[1], sequence, root/'camera.yaml', output, root/'tags.txt')
            history = [json.loads(line) for line in (output/'frames.txt.history.jsonl').read_text().splitlines()]
            self.assertTrue(any(h.get('marker_bootstrap', {}).get('reference_changed')
                                and round(h['timestamp']*30) >= 10 for h in history),
                            [(h['timestamp'], h['state'], h.get('marker_bootstrap')) for h in history[:35]])
            for h in history[:26]:
                self.assertEqual(h['state'], 6)
                self.assertTrue(h['tag_anchored'])
                self.assertEqual(len(h['maps'][0]['keyframes']), 1 if round(h['timestamp']*30) < 10 else 2)
                self.assertEqual(h['maps'][0].get('point_count', len(h['maps'][0]['points'])), 0)
            ready = next(h for h in history if h['maps'][0]['background'])
            self.assertLess(ready['timestamp'], 40/30)
            self.assertGreaterEqual(ready['maps'][0].get('point_count', len(ready['maps'][0]['points'])), 50)
            recovery = history[54]
            self.assertFalse(recovery['marker_keyframe_event'])
            self.assertTrue(any(abs(k[1]-54/30)<1e-6
                                for h in history[54:58] for k in h['maps'][0]['keyframes']),
                            'visual recovery must consider motion-driven keyframes once the interval is met')
            for h in history[45:-1]:
                index=round(h['timestamp']*30)
                self.assertEqual(h['state'], 6 if index in (53,62) else 2, h['timestamp'])
                self.assertEqual(h['tag_anchored'],index in (53,54,62))
                self.assertEqual(h['marker_pose_used'], index in (53,54,62))
                self.assertEqual(h['marker_observation_valid'], index in (53,54,62,63))
                self.assertEqual(h['marker_factor_eligible'], index in (53,54,62))
                self.assertEqual(h['marker_pose_constraint_applied'], index == 54)
                self.assertEqual(h['active_map'], history[0]['active_map'])
                self.assertTrue(h['maps'][0]['metric'])
                self.assertAlmostEqual(h['maps'][0]['scale'], 1.)
                np.testing.assert_allclose(h['pose'][:3], [xs[index], 0, -.5], atol=.01)
            final = history[-1]['maps'][0]
            origin = next(k for k in final['keyframes'] if k[0] == history[0]['reference'])
            np.testing.assert_allclose(origin[2][:3], [-.8, 0, -.5], atol=1e-6)
            np.testing.assert_allclose(final['markers']['20'], layout.markers[20].ravel(), atol=1e-6)

    def run_sequence(self, root, anchored, load=None, name='run', masked=False,
                     visibility=None, final_marker_id=20, bad_corner=False, soft_only=False,
                     partial_frames=(), partial_id=20, partial_shift_m=0., marker_sequence=None):
        root = root / name
        root.mkdir()
        sequence = root / 'sequence'; sequence.mkdir()
        output = root / 'output'; output.mkdir()
        calibration = Calibration(np.array([[500., 0, 320], [0, 500., 240], [0, 0, 1]]), np.zeros(5), (640, 480))
        # A featureless stationary camera must need no visual baseline when a
        # measured, reliable tag pose and its corners are supplied.
        image = np.zeros((480, 640, 3), np.uint8)
        if masked:
            rng = np.random.default_rng(123)
            image = rng.integers(0, 256, image.shape, dtype=np.uint8)
            cv2.imwrite(str(sequence / 'frame.png.mask.png'), np.zeros(image.shape[:2], np.uint8))
        cv2.imwrite(str(sequence / 'frame.png'), image)
        visibility = visibility if visibility is not None else [anchored] * 4
        count = len(visibility)
        (sequence / 'rgb.txt').write_text('\n'.join(f'{i/30:.9f} frame.png' for i in range(count)))
        points = np.array([[-.024, -.024, 0], [.024, -.024, 0], [.024, .024, 0], [-.024, .024, 0]])
        ids = marker_sequence or [20] * (count-1) + [final_marker_id]
        layout = BandLayout('world', 'DICT_4X4_50',
                           {mid: points+[.08*(mid-20),0,0] for mid in set(ids)})
        pose = Pose(np.zeros((3, 1)), np.array([[0.], [0.], [-.5]]), 0.)
        hints = root / 'tags.txt'
        tracked = [TrackedMarkerObservation() for _ in visibility]
        for index in partial_frames:
            partial_pose = Pose(pose.rvec, pose.tvec+[[partial_shift_m],[0.],[0.]], 0.)
            partial_corners = cv2.projectPoints(points, np.zeros(3), -partial_pose.tvec,
                calibration.camera_matrix, np.zeros(5))[0].reshape(4,2)
            tracked[index] = TrackedMarkerObservation(partial_pose,.3,True,points[1:].tolist(),partial_corners[1:].tolist(),
                [partial_id]*3,[1,2,3],'tracked',max(1,index)/30)
        detected = []
        for mid, visible in zip(ids, visibility):
            pixels = cv2.projectPoints(layout.markers[mid], np.zeros(3), -pose.tvec,
                calibration.camera_matrix, np.zeros(5))[0].reshape(4,2)
            if bad_corner:
                pixels[0] += 40
            detected.append({mid: pixels} if visible else {})
        write_tag_observation_hints(hints, [pose if visible else None for visible in visibility],
                                    [1. if visible else 0. for visible in visibility],
                                    detected,
                                    [(mid,) if visible else () for mid, visible in zip(ids, visibility)],
                                    layout, calibration, 30, 0, include_ids=True,
                                    marker_weights=[{mid: .25 if soft_only else 1.} for mid in ids],
                                    tracked_observations=tracked if partial_frames else None)
        settings, atlas = root / 'camera.yaml', root / 'atlas.osa'
        write_orbslam3_settings(settings, calibration, 30, load_atlas=load, save_atlas=atlas)
        run_orbslam3_sequence(Path(__file__).resolve().parents[1], sequence, settings, output, hints)
        history = [json.loads(line) for line in (output / 'frames.txt.history.jsonl').read_text().splitlines()]
        return history, atlas

    def test_new_marker_at_zero_motion_gets_one_information_keyframe(self):
        with tempfile.TemporaryDirectory() as temporary:
            history,_=self.run_sequence(Path(temporary),True,visibility=[True]*8,final_marker_id=21)
        events=[h['marker_keyframe_event'] for h in history[:-1] if h['marker_keyframe_event']]
        self.assertEqual(events,['first_seen:20','first_seen:21'])
        self.assertEqual(len(history[-1]['maps'][0]['keyframes']),2)
        self.assertFalse(history[-1]['maps'][0]['points'])

    def test_reobservation_requests_keyframe_but_short_flicker_does_not(self):
        with tempfile.TemporaryDirectory() as temporary:
            for gap,expected in [(2,1),(18,2)]:
                history,_=self.run_sequence(Path(temporary),True,name=f'gap{gap}',
                    visibility=[True]*3+[False]*gap+[True]*4)
                events=[h['marker_keyframe_event'] for h in history[:-1] if h['marker_keyframe_event']]
                self.assertEqual(events,['first_seen:20']+(['relocalized:20'] if expected==2 else []))
                self.assertEqual(len(history[-1]['maps'][0]['keyframes']),expected)
                self.assertTrue(all(h['pose'] is None for h in history[3:3+gap]))

    def test_old_marker_reappearing_during_valid_localization_adds_no_keyframe(self):
        ids = [20]*3 + [21]*20 + [20]*4
        with tempfile.TemporaryDirectory() as temporary:
            history,_=self.run_sequence(Path(temporary),True,visibility=[True]*len(ids),
                                       marker_sequence=ids)
        events=[h['marker_keyframe_event'] for h in history[:-1] if h['marker_keyframe_event']]
        self.assertEqual(events,['first_seen:20','first_seen:21'])
        self.assertEqual(len(history[-1]['maps'][0]['keyframes']),2)
        self.assertTrue(all(h['state']==6 and h['pose'] is not None for h in history))

    def test_three_corner_jump_is_rejected_even_with_zero_reprojection_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            history,_=self.run_sequence(Path(temporary),True,visibility=[True,False,True],
                                       partial_frames=[1],partial_shift_m=.0188)
        frame=history[1]
        self.assertLess(frame['marker_tracking']['reprojection_px'],.001)
        self.assertFalse(frame['marker_tracking']['accepted'])
        self.assertEqual(frame['marker_tracking']['reason'],'three_corner_motion_gate')
        self.assertIsNone(frame['pose'])
        self.assertIsNotNone(history[2]['pose'])

    def test_three_known_tracked_corners_hold_metric_pose_only_briefly(self):
        with tempfile.TemporaryDirectory() as temporary:
            history,_=self.run_sequence(Path(temporary),True,visibility=[True]+[False]*12,
                                       partial_frames=range(1,13))
        for h in history[1:4]:
            self.assertEqual(h['state'],6,h)
            self.assertTrue(h['tag_anchored'])
            self.assertTrue(h['marker_tracking']['partial'])
            self.assertEqual(h['marker_tracking']['corners'],3)
            self.assertTrue(h['marker_tracking']['accepted'])
            np.testing.assert_allclose(h['pose'][:3],[0,0,-.5],atol=1e-6)
            self.assertEqual(len(h['maps'][0]['keyframes']),1)
            self.assertFalse(h['marker_keyframe_event'])
        self.assertTrue(all(h['pose'] is None for h in history[4:]))
        self.assertEqual(history[4]['marker_tracking']['reason'],'three_corner_age_gate')

    def test_partial_cannot_initialize_or_claim_unknown_marker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            cold,_=self.run_sequence(root,False,name='cold',partial_frames=range(4))
            self.assertTrue(all(h['pose'] is None for h in cold))
            self.assertFalse(any(m['metric'] for h in cold for m in h['maps']))
            unknown,_=self.run_sequence(root,True,name='unknown',visibility=[True]+[False]*3,
                                       partial_frames=range(1,4),partial_id=21)
            self.assertTrue(all(h['pose'] is None for h in unknown[1:]))
            self.assertFalse(any(h['marker_tracking']['accepted'] for h in unknown[1:]))

    def test_stationary_marker_initializes_and_survives_serialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            history, atlas = self.run_sequence(root, True)
            self.assertTrue(atlas.is_file())
            for frame in history[:-1]:
                self.assertEqual(frame['state'], 6)
                np.testing.assert_allclose(frame['pose'][:3], [0, 0, -.5], atol=1e-6)
                mapping = frame['maps'][0]
                self.assertTrue(mapping['metric'])
                self.assertFalse(mapping['background'])
                self.assertEqual(len(mapping['keyframes']), 1)
                self.assertEqual(mapping['points'], [])
                self.assertIn('20', mapping['markers'])
            loaded, _ = self.run_sequence(root, True, load=atlas, name='loaded')
            preserved = [m for m in loaded[-1]['maps'] if m['id'] == history[-1]['maps'][0]['id']]
            self.assertEqual(len(preserved), 1)
            self.assertEqual(sum('20' in m['markers'] for m in loaded[-1]['maps']), 1)
            self.assertEqual(preserved[0]['markers'], history[-1]['maps'][0]['markers'])

    def test_bad_corner_and_soft_only_cannot_seed_metric_map(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for option in ('bad_corner', 'soft_only'):
                history, _ = self.run_sequence(root, True, name=option, **{option: True})
                self.assertTrue(all(h['pose'] is None for h in history))
                self.assertFalse(any(m['metric'] for h in history for m in h['maps']))

    def test_marker_loss_is_invalid_and_new_map_does_not_inherit_scale(self):
        with tempfile.TemporaryDirectory() as temporary:
            visibility = [True] + [False] * 105 + [True]
            history, _ = self.run_sequence(Path(temporary), True, visibility=visibility,
                                          final_marker_id=21)
            frames = history[:-1]
            self.assertTrue(all(h['pose'] is None for h in frames[1:-1]))
            self.assertTrue(any(not m['metric'] for h in frames[1:-1] for m in h['maps']))
            self.assertEqual(frames[-1]['state'], 6)
            maps = [m for m in history[-1]['maps'] if m['keyframes']]
            self.assertEqual(len(maps), 2)
            self.assertEqual({mid for m in maps for mid in m['markers']}, {'20', '21'})

    def test_same_marker_recovers_previous_seed_after_loss(self):
        with tempfile.TemporaryDirectory() as temporary:
            history, _ = self.run_sequence(Path(temporary), True,
                                          visibility=[True] + [False]*105 + [True])
            self.assertEqual(history[-1]['active_map'], history[0]['active_map'])
            self.assertEqual(history[-1]['state'], 6)
            self.assertEqual(sum('20' in m['markers'] for m in history[-1]['maps']), 1)

    def test_native_feature_mask_excludes_every_feature(self):
        with tempfile.TemporaryDirectory() as temporary:
            history, _ = self.run_sequence(Path(temporary), True, masked=True)
            self.assertTrue(all(h.get('feature_count', len(h.get('features', []))) == 0
                                and not h.get('matched_features') for h in history))

    def test_featureless_no_marker_does_not_output_pose(self):
        with tempfile.TemporaryDirectory() as temporary:
            history, _ = self.run_sequence(Path(temporary), False)
            self.assertTrue(all(h['pose'] is None for h in history))


if __name__ == '__main__':
    unittest.main()
