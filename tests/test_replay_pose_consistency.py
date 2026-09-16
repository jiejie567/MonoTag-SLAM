"""Small branch regressions; no SLAM run, camera access, or video encoding."""
import copy
import gzip
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from aruco_track.models import BandLayout, Calibration, Pose
from aruco_track.orbslam3_backend import read_native_result
from aruco_track.slam_replay import trails_at_revision, _world_axis_pixels
import render_slam_replay


def snapshot(state=4, metric=False):
    return {'timestamp': 0., 'final': False, 'state': state, 'active_map': 0,
            'pose': [2., 0., -1., 0., 0., 0., 1.] if state in (2, 6) else None,
            'reference': 10, 'reference_scale': 1., 'relative': [-2., 0., 1., 0., 0., 0., 1.],
            'references': [[10, 0, [0., 0., 0., 0., 0., 0., 1.], 1.]], 'tag_anchored': False,
            'maps': [{'id': 0, 'metric': metric, 'seed': metric, 'background': True,
                      'revision': 7, 'scale': 1. if metric else 0.,
                      'points': [[1, 10., 20., 30.]], 'keyframes': [], 'markers': {}}]}


def action(x=.1):
    return {'marker_camera_pose_observed': {
                'translation_m': [x, 0., -1.], 'quaternion_wxyz': [1., 0., 0., 0.],
                'reprojection_error_px': .2, 'marker_ids': [20], 'ambiguous': True},
            'marker_camera_confidence': .9, 'accepted_marker_ids': [20],
            'rejected_marker_ids': [],
            'marker_boundary_quality': {'20': {'information_weight': 1.}},
            'hands': {'right': {'wrist_camera_graph': {'translation_m': [.3-x, 0., 1.2]}}}}


def register_marker(frame, marker_id=20):
    frame['maps'][0]['markers'][str(marker_id)] = [0.] * 12
    return frame


def native_result(frame, record):
    raw = record['marker_camera_pose_observed']
    pose = Pose(np.zeros((3, 1)), np.array(raw['translation_m']).reshape(3, 1),
                raw['reprojection_error_px'], tuple(raw['marker_ids']), ambiguous=raw['ambiguous'])
    final = copy.deepcopy(frame)
    final['final'] = True
    stream = io.StringIO(json.dumps(frame) + '\n' + json.dumps(final) + '\n')
    with patch('aruco_track.orbslam3_backend.open', return_value=stream, create=True):
        return read_native_result(Path('in_memory.jsonl'), [pose],
                                  [record['marker_camera_confidence']], [None], 30., {},
                                  accepted_marker_ids=[tuple(record['accepted_marker_ids'])],
                                  marker_weights=[{int(mid): q['information_weight'] for mid, q in
                                                   record['marker_boundary_quality'].items()}]).frames[0]


class ReplayMarkerFallbackTests(unittest.TestCase):
    def test_lost_native_never_publishes_raw_marker_as_an_output_pose(self):
        frame, record = register_marker(snapshot(metric=True)), action()
        label = native_result(frame, record)
        trails, camera, map_id = trails_at_revision(0, [frame], [record], frame, 30.)
        self.assertEqual(label.source, 'invalid')
        self.assertIsNone(label.pose)
        self.assertIsNone(map_id)
        self.assertIsNone(camera)
        self.assertEqual(trails, {})
        calibration = Calibration(np.array([[500., 0., 320.], [0., 500., 240.], [0., 0., 1.]]),
                                  np.zeros(5), (640, 480))
        self.assertIsNone(_world_axis_pixels(camera, calibration, True))

    def test_unscaled_orb_does_not_publish_an_independent_marker_world(self):
        from aruco_track.slam_replay import replay_camera_frame
        frame, record = snapshot(state=2), action()
        final = copy.deepcopy(frame)
        final['maps'][0].update(metric=True, revision=9)
        current = replay_camera_frame(frame, frame, record)
        finished = replay_camera_frame(frame, final, record)
        self.assertEqual((current.map_id, current.source, current.revision), ('atlas_0', 'invalid', 7))
        self.assertIsNone(current.pose)
        self.assertEqual((finished.map_id, finished.revision), ('atlas_0', 9))
        # A final optimized label is deliberately poisonous: replay must never read it.
        record['camera_world_pose_fused'] = {'translation_m': [99., 99., 99.]}
        record['camera_world_confidence'] = 1.
        replay = replay_camera_frame(frame, frame, record)
        self.assertIsNone(replay.pose)
        self.assertEqual(replay.source, 'invalid')

    def test_reliable_marker_never_switches_an_inconsistent_metric_atlas_pose(self):
        frame, record = snapshot(state=2, metric=True), action(.1)
        frame.update(tag_anchored=True, pose=[.30, 0., -1., 0., 0., 0., 1.])
        frame['marker_tracking'] = {'accepted': True, 'partial': False}
        selected = native_result(frame, record)
        self.assertEqual((selected.source, selected.map_id), ('marker+slam', 'atlas_0'))
        np.testing.assert_allclose(selected.pose.tvec.ravel(), [.30, 0., -1.])
        following = copy.deepcopy(frame)
        following['pose'][0] = .31
        next_selected = native_result(following, record)
        self.assertEqual((next_selected.source, next_selected.map_id),
                         ('marker+slam', 'atlas_0'))
        self.assertAlmostEqual(
            np.linalg.norm(next_selected.pose.tvec - selected.pose.tvec), .01
        )

    def test_native_marker_tracking_publishes_marker_in_atlas_not_marker_world(self):
        frame, record = snapshot(state=6, metric=True), action(.1)
        frame.update(tag_anchored=True,
                     marker_tracking={'accepted': True, 'partial': False})
        selected = native_result(frame, record)
        self.assertEqual((selected.source, selected.map_id), ('marker', 'atlas_0'))

    def test_consistent_marker_and_metric_slam_keep_native_fused_pose(self):
        frame, record = snapshot(state=2, metric=True), action(.1)
        frame.update(tag_anchored=True, pose=[.11, 0., -1., 0., 0., 0., 1.])
        frame['marker_tracking'] = {'accepted': True, 'partial': False}
        selected = native_result(frame, record)
        self.assertEqual((selected.source, selected.map_id), ('marker+slam', 'atlas_0'))
        np.testing.assert_allclose(selected.pose.tvec.ravel(), [.11, 0., -1.])

    def test_native_marker_atlas_trail_breaks_only_for_missing_native_pose(self):
        records = [action(.01*i) for i in range(4)]
        snapshots = []
        for index in range(4):
            frame = register_marker(snapshot(state=6, metric=True))
            frame.update(timestamp=index / 30., pose=[.01*index, 0., -1., 0., 0., 0., 1.],
                         tag_anchored=True,
                         marker_tracking={'accepted': True, 'partial': False})
            snapshots.append(frame)
        snapshots[2] = register_marker(snapshot(metric=True))
        snapshots[2]['timestamp'] = 2 / 30.
        trails, _, map_id = trails_at_revision(3, snapshots, records, snapshots[3], 30.)
        self.assertEqual(map_id, 'atlas_0')
        np.testing.assert_allclose(trails['right'][1], [.3, 0., .2])
        self.assertIsNone(trails['right'][2])
        np.testing.assert_allclose([trails['right'][0], trails['right'][3]], [[.3, 0., .2]]*2)

    def test_bad_marker_does_not_reappear_via_fallback_or_final_fused_fields(self):
        from aruco_track.slam_replay import replay_camera_frame
        original = action()
        variants = []
        for confidence in [.34, float('nan')]:
            row = copy.deepcopy(original); row['marker_camera_confidence'] = confidence
            variants.append(row)
        row = copy.deepcopy(original); row['marker_camera_pose_observed']['reprojection_error_px'] = 3.
        variants.append(row)
        row = copy.deepcopy(original); row['accepted_marker_ids'] = []
        variants.append(row)
        row = copy.deepcopy(original); row['rejected_marker_ids'] = [20]
        variants.append(row)
        row = copy.deepcopy(original); row['marker_boundary_quality']['20']['information_weight'] = .25
        variants.append(row)
        row = copy.deepcopy(original); row['marker_camera_pose_observed'] = None
        variants.append(row)
        for row in variants:
            row['camera_world_confidence'] = 1.
            row['camera_world_pose_fused'] = original['marker_camera_pose_observed']
            with self.subTest(row=row):
                selected = replay_camera_frame(snapshot(), snapshot(), row)
                self.assertIsNone(selected.pose)
                self.assertEqual(selected.source, 'invalid')

    def test_full_geometry_rejection_is_not_revived_but_distinct_partial_rejection_is_not_used(self):
        from aruco_track.slam_replay import replay_camera_frame
        frame, record = register_marker(snapshot(metric=True)), action()
        frame['marker_tracking'] = {'accepted': False, 'partial': False, 'reason': 'geometry_rejected'}
        self.assertIsNone(native_result(frame, record).pose)
        self.assertIsNone(replay_camera_frame(frame, frame, record).pose)
        frame['marker_tracking']['partial'] = True
        self.assertIsNone(replay_camera_frame(frame, frame, record).pose)

    def test_label_and_replay_both_require_accepted_strong_marker_observations(self):
        from aruco_track.slam_replay import replay_camera_frame
        for accepted, weight in [([], 1.), ([20], .25), ([20], 0.), ([20], float('inf'))]:
            frame, record = snapshot(), action()
            record['accepted_marker_ids'] = accepted
            record['marker_boundary_quality']['20']['information_weight'] = weight
            with self.subTest(accepted=accepted, weight=weight):
                self.assertIsNone(native_result(frame, record).pose)
                self.assertIsNone(replay_camera_frame(frame, frame, record).pose)

    def test_marker_world_geometry_has_no_atlas_points_or_future_marker_layout(self):
        from aruco_track.slam_replay import replay_camera_frame, _marker_world_view
        from aruco_track.camera_state import FusedCameraFrame
        layout = BandLayout('fixed', 'DICT_4X4_50', {
            20: np.array([[0., 0., 0.], [.05, 0., 0.], [.05, .05, 0.], [0., .05, 0.]]),
            21: np.ones((4, 3))})
        raw = action()['marker_camera_pose_observed']
        pose = Pose(np.zeros((3, 1)), np.asarray(raw['translation_m']).reshape(3, 1),
                    raw['reprojection_error_px'], tuple(raw['marker_ids']))
        selected = FusedCameraFrame(pose, 'marker', .9, 0, .2,
                                    'marker_world', metric=True)
        view = _marker_world_view(selected, layout)
        self.assertEqual((view['id'], view['metric'], view['point_count']), ('marker_world', True, 0))
        self.assertEqual(view['keyframes'], [])
        self.assertEqual(set(view['markers']), {'20'})
        np.testing.assert_allclose(view['markers']['20'], layout.markers[20].ravel())
        self.assertEqual(_marker_world_view(selected, None)['markers'], {})

    def test_process_frames_and_final_tail_use_their_own_revisions_without_encoding(self):
        from aruco_track.replay_browser import decode_timeline
        from aruco_track.slam_replay import write_slam_replay
        frame, record = snapshot(state=2), action()
        frame.update(feature_count=1, matched_features=[[10., 20., 1]])
        final = copy.deepcopy(frame)
        final['final'] = True
        final['maps'][0].update(metric=True, revision=9)
        record['hands']['right']['joints'] = {'valid': False}
        record['hands']['right']['wrist_camera_graph']['quaternion_wxyz'] = [1., 0., 0., 0.]
        capture = MagicMock()
        capture.isOpened.return_value = True
        capture.read.return_value = (True, np.zeros((480, 640, 3), np.uint8))
        encoder = MagicMock()
        encoder.wait.return_value = 0
        # Do not keep 31 unencoded images in MagicMock's call argument history.
        encoder.stdin.write = lambda data: len(data)
        calibration = Calibration(np.array([[500., 0., 320.], [0., 500., 240.], [0., 0., 1.]]),
                                  np.zeros(5), (640, 480))
        layout = BandLayout('fixed', 'DICT_4X4_50', {20: np.zeros((4, 3))})
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            records_path = directory / 'actions.jsonl'
            records_path.write_text(json.dumps(record) + '\n')
            with patch('aruco_track.slam_replay.cv2.VideoCapture', return_value=capture), \
                 patch('aruco_track.slam_replay.subprocess.Popen', return_value=encoder), \
                 patch('aruco_track.slam_replay.shutil.which', return_value='/mock/ffmpeg'):
                write_slam_replay(Path('mock.avi'), records_path, directory, [frame, final],
                                  calibration, [{}], [(20,)], 30., marker_layout=layout)
            with gzip.open(directory / 'video_frames.json.gz', 'rt') as stream:
                frames = json.load(stream)
            with gzip.open(directory / 'timeline.json.gz', 'rt') as stream:
                timeline = json.load(stream)
            manifest = json.loads((directory / 'manifest.json').read_text())
            browser_timeline = directory / manifest['browser_timeline']
            self.assertTrue(browser_timeline.is_file())
            with gzip.open(browser_timeline, 'rt') as stream:
                self.assertEqual(decode_timeline(json.load(stream)), timeline)
        first = frames[0]
        self.assertEqual((first['source'], first['map_id'], first['map_revision']), ('invalid', 'atlas_0', 7))
        self.assertFalse(first['world_axes_visible'])
        self.assertFalse(first['metric'])
        self.assertFalse(first['metric_recovered_later'])
        self.assertEqual(first['orb_map_point_ids'], [1])
        self.assertEqual(first['visible_marker_ids'], [20])
        self.assertFalse(first['tail'])
        self.assertIsNone(first['marker_world_view'])
        self.assertFalse(timeline[0]['maps'][0]['metric'])
        self.assertEqual([m['id'] for m in timeline[0]['maps']], [0])
        self.assertEqual((frames[1]['map_id'], frames[1]['map_revision']), ('atlas_0', 9))
        self.assertEqual(len(frames[1:]), 120)
        self.assertTrue(all(f['tail'] for f in frames[1:]))
        self.assertTrue(all(f['metric_recovered_later'] for f in frames[1:]))
        self.assertTrue(all(f['marker_world_view'] is None for f in frames[1:]))
        self.assertTrue(manifest['capabilities']['visual_loop_merge_unscaled'])
        self.assertFalse(manifest['capabilities']['metric_tag_visual_loop'])
        self.assertFalse(manifest['capabilities']['metric_tag_pose_graph_factors'])
        self.assertFalse(manifest['capabilities']['metric_tag_visual_merge'])
        self.assertFalse(manifest['capabilities']['metric_tag_visual_loop_merge'])
        self.assertFalse(manifest['capabilities']['metric_tag_global_ba'])
        self.assertFalse(manifest['capabilities']['correction_rejection_events'])
        self.assertEqual(manifest['final_hold_seconds'], 4.0)

    def test_replay_defaults_to_double_speed_and_retains_manual_speed_control(self):
        html = (Path(__file__).resolve().parents[1] / 'aruco_track/slam_replay.html').read_text()
        options = html.split('<select id="speed">', 1)[1].split('</select>', 1)[0]
        self.assertIn('<option selected>2</option>', options)
        self.assertEqual(options.count('selected'), 1)
        initialization = html.split('<script>', 1)[1].split('const popup=', 1)[0]
        self.assertIn(
            "video.defaultPlaybackRate=video.playbackRate=Number($('speed').value);",
            initialization,
        )
        self.assertIn("$('speed').onchange=()=>video.playbackRate=Number($('speed').value);", html)

    def test_viewer_draws_native_loop_edges_and_explains_revised_history(self):
        html = (Path(__file__).resolve().parents[1] / 'aruco_track/slam_replay.html').read_text()
        self.assertIn('for(const pair of m.loops||[])', html)
        self.assertIn("红线回环边", html)
        self.assertIn('历史轨迹已按最终地图修订重绘', html)

    def test_hand_view_replaces_planar_grid_and_retains_playback_controls(self):
        html = (Path(__file__).resolve().parents[1] / 'aruco_track/slam_replay.html').read_text()
        for function in ('worldToHandCamera', 'projectHandPoint', 'handFrameState',
                         'handViewSegments', 'drawEgoHands'):
            self.assertIn(f'function {function}(', html)
        self.assertNotIn('二维世界轨迹', html)
        self.assertNotIn('function niceMetricStep(', html)
        self.assertNotIn('const trajectoryView=', html)
        self.assertNotIn('id="ego"', html)
        self.assertNotIn('syncEgoOrientation', html)
        self.assertNotIn('useFreeOrientation', html)
        self.assertIn('id="hand-frame-status"', html)
        self.assertIn('function smoothFadingLine(', html)
        self.assertIn("context.lineJoin='round'", html)
        self.assertIn('context.quadraticCurveTo(', html)
        self.assertIn('video.ended', html)
        self.assertIn('video.currentTime>=video.duration-.05', html)
        self.assertIn('视频播放失败', html)

    def test_replay_frame_label_explains_source_time_mapping_and_final_hold(self):
        html = (Path(__file__).resolve().parents[1] / 'aruco_track/slam_replay.html').read_text()
        self.assertIn('function frameClockLabel(frame)', html)
        self.assertIn('回放帧 ${frameIndex}', html)
        self.assertIn('→ 原始帧 ${frame.source_frame}', html)
        self.assertIn('最终优化停帧 ${frameIndex-first+1}/${total}', html)
        self.assertNotIn('输出帧 ${frameIndex} / 原始帧', html)

    @unittest.skipUnless(shutil.which('node'), 'Node is needed for the small viewer selection check')
    def test_html_selects_independent_marker_world_without_drawing_atlas_points(self):
        html = (Path(__file__).resolve().parents[1] / 'aruco_track/slam_replay.html').read_text()
        self.assertIn('id="map-overview"', html)
        self.assertIn('id="map-local"', html)
        self.assertIn('id="trajectory"', html)
        self.assertIn("view.mode==='local'", html)
        self.assertIn('主体 90% 自动适配', html)
        self.assertIn('view.width*.92/spanU', html)
        self.assertIn('targetCenter=f.camera.translation.slice()', html)
        self.assertIn("targetExtent=Math.max(m.metric?.20:.05", html)
        self.assertIn('trajectorySegments', html)
        self.assertIn("view.mode==='local'?view.extent*.04:0", html)
        self.assertIn("view.mode==='overview'&&!values.length", html)
        self.assertIn('grid-template-rows:minmax(0,1fr) minmax(0,1fr)', html)
        self.assertIn('html,body{width:100%;height:100%;overflow:hidden}', html)
        self.assertIn('.bar button,.bar label,.bar strong{white-space:nowrap', html)
        self.assertIn("rgba(20,20,20,.28)", html)
        self.assertIn('蓝线相机关键帧轨迹', html)
        self.assertIn("if(isOrigin)camera(view,k[2],'#e52424',size*2.4,3)", html)
        self.assertNotIn('蓝色其余关键帧', html)
        full_script = html.split('<script>')[1].split('</script>')[0]
        script = full_script.split("\nfetch('manifest.json')", 1)[0]
        # Execute the actual viewer functions with a minimal canvas/DOM, no web
        # server or video. A metric marker camera must not share the ORB cloud.
        harness = r'''
const assert=require('node:assert/strict');
let rectangles=[];
const context=new Proxy({fillRect(...r){rectangles.push(r)},
 measureText(text){return {width:String(text).length*7}}},{
 get(target,key){return key in target?target[key]:()=>{}}});
const elements=new Map();
global.document={body:{classList:{add(){}}},getElementById(id){
 if(!elements.has(id))elements.set(id,{value:'auto',checked:true,addEventListener(){},
 getContext(){return context},getBoundingClientRect(){return {width:640,height:480}},
 options:[],replaceChildren(){},add(){}});return elements.get(id)}};
global.location={search:'',pathname:'/test'};global.devicePixelRatio=1;
global.BroadcastChannel=class{postMessage(){}};global.window={};
'''
        checks = r'''
const metricView={id:'marker_world',metric:true,point_count:0,keyframes:[],markers:{}};
frames=[{map_id:'marker_world',metric:true,source:'marker',marker_world_view:metricView,camera:{translation:[0,0,-1],
 rotation:[[1,0,0],[0,1,0],[0,0,1]]},hands:{},trails:{right:[[0,0,0],[.1,0,0]]}}];
revision={active_map:7,maps:[{id:7,metric:false,point_count:1,keyframes:[],markers:{}}]};
points.set('7:1',[7,100,100,100]);
assert.equal(selectMap(),'marker_world');assert.equal(availableMaps().length,2);
draw();assert.equal(rectangles.filter(r=>r[2]===1.8).length,0);
$('maps').value='7';rectangles=[];draw();
assert.equal(selectMap(),7);assert.equal(rectangles.filter(r=>r[2]===1.8).length,2);
$('maps').value='marker_world';assert.equal(selectMap(),'marker_world');
frames[0].marker_world_view=null;frames[0].map_id='atlas_7';$('maps').value='auto';
assert.equal(selectMap(),7);assert.equal(availableMaps().length,1);
frames[0].camera={translation:[10,0,0],rotation:[[1,0,0],[0,1,0],[0,0,1]]};
frames[0].orb_map_point_ids=[1];points.set('7:1',[7,100,0,0]);
const smoothView=makeView($('map-local'),'local');smoothView.center=[0,0,0];smoothView.extent=1;
fit(smoothView,revision.maps[0],.9,true);
assert.deepStrictEqual(smoothView.center,[10,0,0]);assert.ok(smoothView.extent<=1.021);
points.clear();for(let i=0;i<10;i++)points.set('7:'+i,[7,i,0,0]);
assert.equal(displayedMapPointEntries(revision.maps[0]).length,9);
'''
        result = subprocess.run([shutil.which('node'), '-e',
                                 'new Function(' + json.dumps(full_script) + ');\n' + harness + script + checks],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)


class NativeCorrectionPublicationTests(unittest.TestCase):
    @staticmethod
    def rejection(sequence, kind='loop', reason='fixed_marker_geometry_requires_joint_optimization'):
        return {'sequence': sequence, 'timestamp': .125, 'keyframe_id': 8,
                'map_id': 1, 'other_map_id': 2, 'kind': kind, 'reason': reason}

    def test_rejected_corrections_are_never_described_as_committed_and_are_deduplicated(self):
        from aruco_track.slam_replay import _events
        previous, current = snapshot(), snapshot()
        previous['correction_rejections'] = [self.rejection(1)]
        current['correction_rejections'] = previous['correction_rejections'] + [
            self.rejection(2, 'merge'), self.rejection(3, 'global_ba'), self.rejection(3, 'global_ba')]
        events = _events(previous, current)
        self.assertEqual(len(events), 2)
        self.assertIn('视觉合并未提交', events[0])
        self.assertIn('固定 marker 几何尚不支持共同优化', events[0])
        self.assertIn('全局BA未提交', events[1])
        self.assertIn('优化期间 marker 坐标/尺度已变更', events[1])
        self.assertFalse(any('原生视觉回环：' in event or '原生地图合并：' in event for event in events))
        self.assertEqual(_events(current, current), [])
        first = copy.deepcopy(current)
        first['correction_rejections'] = [self.rejection(1)]
        self.assertIn('视觉回环未提交', _events(snapshot(), first)[0])

    def test_map_unavailable_is_reported_without_claiming_marker_rejection(self):
        from aruco_track.slam_replay import _events
        current = snapshot()
        current['correction_rejections'] = [self.rejection(1, 'merge', 'map_unavailable')]
        events = _events(snapshot(), current)
        self.assertEqual(len(events), 1)
        self.assertIn('目标地图不可用', events[0])
        self.assertNotIn('固定 marker', events[0])

    def test_rejection_appears_at_publication_not_at_earlier_candidate_keyframe_time(self):
        from aruco_track.slam_replay import pack_history
        history = [snapshot(), snapshot(), snapshot()]
        history[1]['timestamp'], history[2]['timestamp'] = 2., 3.
        history[0]['correction_rejections'] = []
        history[1]['correction_rejections'] = [self.rejection(1, 'global_ba')]
        history[2]['correction_rejections'] = [self.rejection(1, 'global_ba')]
        with tempfile.TemporaryDirectory() as temporary:
            rows = pack_history(history, Path(temporary), 30.)
            events = [json.loads(line) for line in (Path(temporary) / 'events.jsonl').read_text().splitlines()]
        rejections = [event for event in events if '未提交' in event['description']]
        self.assertEqual(len(rejections), 1)
        self.assertEqual((rejections[0]['timestamp_s'], rejections[0]['source_frame']), (2., 60))
        self.assertFalse(any('未提交' in text for text in rows[0]['events'] + rows[2]['events']))

    def test_pure_visual_committed_loops_and_merges_remain_visible(self):
        from aruco_track.slam_replay import _events
        current = snapshot()
        current['maps'][0].update(loops=[[1, 2]], merges=[[3, 4]])
        events = _events(snapshot(), current)
        self.assertIn('原生视觉回环：关键帧 1 ↔ 2', events)
        self.assertIn('原生地图合并：关键帧 3 ↔ 4', events)

    def test_resolved_culled_reference_is_already_in_current_units_only_relative_scales(self):
        from aruco_track.orbslam3_backend import camera_at_revision, select_camera_frame
        frame = snapshot(state=2)
        frame.update(reference=10, reference_scale=1., relative=[-2., 0., 1., 0., 0., 0., 1.])
        revision = snapshot(state=2, metric=True)
        revision['maps'][0]['id'] = 9
        revision['active_map'] = 9
        # This original KF has been culled; native GetReplayReference already
        # resolves its parent chain into map 9's current pose and cumulative scale.
        revision['references'] = [[10, 9, [5., 0., 0., 0., 0., 0., 1.], .4]]
        record = {'hands': {'right': {'wrist_camera_graph': {'translation_m': [.2, 0., 1.]}}}}
        for acquired_scale, relative in [(1., [-2., 0., 1., 0., 0., 0., 1.]),
                                          (.4, [-.8, 0., .4, 0., 0., 0., 1.])]:
            frame.update(reference_scale=acquired_scale, relative=relative)
            before = copy.deepcopy(frame)
            camera, map_id = camera_at_revision(frame, revision)
            self.assertEqual(map_id, 'atlas_9')
            np.testing.assert_allclose(camera.tvec.ravel(), [5.8, 0., -.4])
            trails, _, _ = trails_at_revision(0, [frame], [record], revision, 30.)
            np.testing.assert_allclose(trails['right'], [[6., 0., .6]])
            selected = select_camera_frame(frame, revision)
            self.assertTrue(selected.metric)
            self.assertTrue(selected.metric_recovered_later)
            self.assertEqual(frame, before)

    def test_unresolved_historical_reference_never_promotes_arbitrary_raw_pose_to_metres(self):
        from aruco_track.orbslam3_backend import camera_at_revision, select_camera_frame
        frame = snapshot(state=2)
        final = copy.deepcopy(frame)
        final['final'] = True
        final['maps'][0].update(metric=True, revision=8)
        for missing in ('references', 'relative'):
            incomplete_frame, incomplete_final = copy.deepcopy(frame), copy.deepcopy(final)
            if missing == 'references':
                incomplete_final['references'] = []
            else:
                incomplete_frame.pop('relative')
            with self.subTest(missing=missing):
                self.assertEqual(camera_at_revision(incomplete_frame, incomplete_final), (None, None))
                selected = select_camera_frame(incomplete_frame, incomplete_final)
                self.assertIsNone(selected.pose)
                self.assertFalse(selected.metric)
                self.assertFalse(selected.metric_recovered_later)
                self.assertEqual(selected.source, 'invalid')

    def test_legacy_current_measurement_stays_readable_but_unknown_future_transform_does_not(self):
        from aruco_track.orbslam3_backend import camera_at_revision
        frame = snapshot(state=2, metric=True)
        frame.pop('references')
        for current in (frame, copy.deepcopy(frame)):
            pose, map_id = camera_at_revision(frame, current)
            self.assertEqual(map_id, 'atlas_0')
            np.testing.assert_allclose(pose.tvec.ravel(), [2., 0., -1.])
        changed = copy.deepcopy(frame)
        changed['maps'][0]['revision'] += 1
        self.assertEqual(camera_at_revision(frame, changed), (None, None))
        changed = copy.deepcopy(frame)
        changed['final'] = True
        self.assertEqual(camera_at_revision(frame, changed), (None, None))

    def test_missing_reference_preserves_true_marker_seed_but_not_unregistered_raw_marker(self):
        from aruco_track.orbslam3_backend import camera_at_revision
        from aruco_track.slam_replay import replay_camera_frame
        seed = snapshot(state=6, metric=True)
        seed['tag_anchored'] = True
        seed.pop('references')
        final = copy.deepcopy(seed)
        final['final'] = True
        pose, map_id = camera_at_revision(seed, final)
        self.assertEqual(map_id, 'atlas_0')
        np.testing.assert_allclose(pose.tvec.ravel(), [2., 0., -1.])
        arbitrary = snapshot(state=2)
        final = copy.deepcopy(arbitrary)
        final.update(final=True, references=[])
        final['maps'][0]['metric'] = True
        selected = replay_camera_frame(arbitrary, final, action())
        self.assertIsNone(selected.pose)
        self.assertEqual(selected.source, 'invalid')


class MarkerGraphPublicationTests(unittest.TestCase):
    @staticmethod
    def event(sequence=1, kind='scale_reanchor', status='accepted'):
        return {'sequence': sequence, 'type': kind, 'status': status,
                'frame': 15, 'timestamp': .5, 'map_id': 0,
                'source_map_id': 1 if kind == 'marker_map_merge' else 0,
                'target_map_id': 0, 'marker_ids': [20, 21], 'scale': .9,
                'sigma': .005, 'affected_keyframes': [3, 4, 5],
                'residual_before': 1.2, 'residual_after': .4,
                'reason': '' if status == 'accepted' else 'anchor_inconsistent'}

    def test_marker_graph_commit_and_rejection_have_distinct_labels_and_deduplicate(self):
        from aruco_track.slam_replay import _events
        previous, current = snapshot(), snapshot()
        previous['marker_graph_events'] = [self.event()]
        current['marker_graph_events'] = previous['marker_graph_events'] + [
            self.event(2, 'marker_map_merge'), self.event(3, status='rejected'),
            self.event(3, status='rejected')]
        descriptions = _events(previous, current)
        self.assertEqual(len(descriptions), 2)
        self.assertIn('标记辅助合图已提交', descriptions[0])
        self.assertIn('1 → 0', descriptions[0])
        self.assertIn('尺度再锚定未提交', descriptions[1])
        self.assertIn('anchor_inconsistent', descriptions[1])
        self.assertFalse(any('原生视觉回环' in text for text in descriptions))
        self.assertEqual(_events(current, current), [])

    def test_interval_event_describes_committed_scale_and_affected_keyframes(self):
        from aruco_track.slam_replay import _events
        current = snapshot()
        current['marker_graph_events'] = [self.event()]
        descriptions = _events(snapshot(), current)
        self.assertEqual(len(descriptions), 1)
        self.assertIn('尺度再锚定已提交', descriptions[0])
        self.assertIn('0.9', descriptions[0])
        self.assertIn('3 个关键帧', descriptions[0])
        self.assertIn('20', descriptions[0])
        self.assertIn('21', descriptions[0])

    def test_marker_graph_event_is_published_once_at_committed_snapshot_not_candidate(self):
        from aruco_track.slam_replay import pack_history
        history = [snapshot(), snapshot(), snapshot()]
        history[1]['timestamp'], history[2]['timestamp'] = 2., 3.
        history[0]['marker_graph_events'] = []
        history[1]['marker_graph_events'] = [self.event()]
        history[2]['marker_graph_events'] = [self.event()]
        with tempfile.TemporaryDirectory() as temporary:
            rows = pack_history(history, Path(temporary), 30.)
            events = [json.loads(line) for line in (Path(temporary) / 'events.jsonl').read_text().splitlines()]
        commits = [event for event in events if '尺度再锚定' in event['description']]
        self.assertEqual(len(commits), 1)
        self.assertEqual((commits[0]['timestamp_s'], commits[0]['source_frame']), (2., 60))
        self.assertEqual(rows[0]['marker_graph_events'], [])
        self.assertEqual(rows[1]['marker_graph_events'], [self.event()])
        self.assertFalse(any('尺度再锚定' in text for text in rows[0]['events'] + rows[2]['events']))

    def test_marker_and_background_correction_sequences_have_separate_namespaces(self):
        from aruco_track.slam_replay import _events
        previous, current = snapshot(), snapshot()
        previous['correction_rejections'] = [NativeCorrectionPublicationTests.rejection(1)]
        current['correction_rejections'] = previous['correction_rejections']
        current['marker_graph_events'] = [self.event(1)]
        descriptions = _events(previous, current)
        self.assertEqual(len(descriptions), 1)
        self.assertIn('尺度再锚定已提交', descriptions[0])

    def test_reader_updates_marker_histories_to_merged_target_using_capture_graph_versions(self):
        from aruco_track.orbslam3_backend import camera_at_revision
        identity = [0, 1., 0., 0., 0., 0., 0., 0., 1.]
        after_scale = [1, .9, 1., 0., 0., 0., 0., 0., 1.]
        after_merge = [2, .9, 4., 0., 0., 0., 0., 0., 1.]
        first, second = snapshot(state=6, metric=True), snapshot(state=6, metric=True)
        for index, (frame, x, graph) in enumerate([(first, .2, identity),
                                                 (second, 1.18, after_scale)]):
            frame.update(timestamp=index/30., tag_anchored=True,
                         pose=[x, 0., 0., 0., 0., 0., 1.], reference_marker_graph=graph)
            frame['references'][0].append(graph)
        final = copy.deepcopy(second)
        final.update(final=True, active_map=2)
        final['maps'][0].update(id=2, revision=20)
        final['references'] = [[10, 2, [999., 0., 0., 0., 0., 0., 1.], .9, after_merge]]
        stream = io.StringIO('\n'.join(json.dumps(frame) for frame in [first, second, final]))
        with patch('aruco_track.orbslam3_backend.open', return_value=stream, create=True):
            result = read_native_result(Path('in_memory.jsonl'), [None, None], [0., 0.], [None, None], 30., {})
        for selected in result.frames:
            self.assertEqual((selected.map_id, selected.revision, selected.source), ('atlas_2', 20, 'marker'))
            np.testing.assert_allclose(selected.pose.tvec.ravel(), [4.18, 0., 0.])
        # Looking at the old publication never imports the final transform.
        for frame, expected in [(first, .2), (second, 1.18)]:
            camera, map_id = camera_at_revision(frame, frame)
            self.assertEqual(map_id, 'atlas_0')
            self.assertAlmostEqual(camera.tvec[0, 0], expected)

    def test_unrelated_marker_map_is_not_transformed_by_another_maps_commit(self):
        from aruco_track.orbslam3_backend import camera_at_revision
        frame = snapshot(state=6, metric=True)
        frame.update(tag_anchored=True, reference=11, active_map=1,
                     reference_marker_graph=[0, 1., 0., 0., 0., 0., 0., 0., 1.])
        frame['maps'][0]['id'] = 1
        revised = copy.deepcopy(frame)
        revised['timestamp'] = 2.
        revised['marker_graph_events'] = [self.event()]
        revised['references'] = [
            [10, 0, [100., 0., 0., 0., 0., 0., 1.], .9, [1, .9, 2., 0., 0., 0., 0., 0., 1.]],
            [11, 1, [200., 0., 0., 0., 0., 0., 1.], 1., frame['reference_marker_graph']]]
        camera, map_id = camera_at_revision(frame, revised)
        self.assertEqual(map_id, 'atlas_1')
        np.testing.assert_allclose(camera.tvec.ravel(), [2., 0., -1.])

    def test_capabilities_require_native_declaration_not_just_event_presence(self):
        from aruco_track.slam_replay import _marker_graph_capabilities
        current = snapshot()
        current['marker_graph_events'] = [self.event()]
        unsupported = {'interval_scale_reanchor': False, 'marker_map_merge': False,
                       'rigid_marker_pose_optimization': False,
                       'metric_tag_global_ba': False}
        self.assertEqual(_marker_graph_capabilities([]), unsupported)
        self.assertEqual(_marker_graph_capabilities([current]), unsupported)
        current['marker_graph_capabilities'] = {
            'interval_scale_reanchor': True, 'marker_map_merge': False,
            'rigid_marker_pose_optimization': False, 'metric_tag_global_ba': False}
        self.assertEqual(_marker_graph_capabilities([current]), current['marker_graph_capabilities'])
        current['marker_graph_capabilities'].update(
            marker_map_merge=True, rigid_marker_pose_optimization=True,
            metric_tag_global_ba=True)
        self.assertTrue(all(_marker_graph_capabilities([current, copy.deepcopy(current)]).values()))
        # Mixing old native records with new declarations must not claim the
        # whole process has correction history that older frames never saved.
        self.assertEqual(_marker_graph_capabilities([snapshot(), current]), unsupported)

    def test_final_labels_and_process_trails_share_rigid_gauge_without_future_scale(self):
        from aruco_track.orbslam3_backend import camera_at_revision
        identity = [0, 1., 0., 0., 0., 0., 0., 0., 1.]
        graph_scaled = [1, .85, .015, 0., 0., 0., 0., 0., 1.]
        first = snapshot(state=2, metric=True)
        first.update(pose=[.12, 0., -1., 0., 0., 0., 1.], tag_anchored=True,
                     reference_marker_graph=identity.copy(), reference_marker_gauge=identity.copy())
        first['references'][0].extend([identity.copy(), identity.copy()])
        second = copy.deepcopy(first)
        second.update(timestamp=1/30., pose=[.13, 0., -1., 0., 0., 0., 1.],
                      reference_marker_graph=graph_scaled, reference_scale=.85)
        second['references'][0][3:5] = [.85, graph_scaled]
        final = copy.deepcopy(second)
        final.update(final=True, active_map=2)
        final['maps'][0].update(id=2, revision=20)
        final['references'] = [[10, 2, [999., 0., 0., 0., 0., 0., 1.], .85,
                                [2, .85, 3.015, 0., 0., 0., 0., 0., 1.],
                                [2, 1., 3., 0., 0., 0., 0., 0., 1.]]]
        frames = [first, second]
        records = [action(.12), action(.13)]
        stream = io.StringIO('\n'.join(json.dumps(frame) for frame in [*frames, final]))
        with patch('aruco_track.orbslam3_backend.open', return_value=stream, create=True):
            result = read_native_result(Path('in_memory.jsonl'), [None, None], [0., 0.],
                                        [None, None], 30., {})
        for selected, expected in zip(result.frames, [3.12, 3.13]):
            self.assertEqual((selected.map_id, selected.revision, selected.source),
                             ('atlas_2', 20, 'marker+slam'))
            np.testing.assert_allclose(selected.pose.tvec.ravel(), [expected, 0., -1.])
        current, _, current_map = trails_at_revision(1, frames, records, second, 30.)
        finished, camera, final_map = trails_at_revision(1, frames, records, final, 30.)
        self.assertEqual((current_map, final_map), ('atlas_0', 'atlas_2'))
        np.testing.assert_allclose(current['right'], [[.3, 0., .2]]*2, atol=1e-12)
        np.testing.assert_allclose(finished['right'], [[3.3, 0., .2]]*2, atol=1e-12)
        np.testing.assert_allclose(camera.tvec, result.frames[1].pose.tvec)
        old_camera, old_map = camera_at_revision(first, second)
        self.assertEqual(old_map, 'atlas_0')
        np.testing.assert_allclose(old_camera.tvec.ravel(), [.12, 0., -1.])


class ReplayCalibrationTests(unittest.TestCase):
    @staticmethod
    def calibration():
        return Calibration(np.array([[1000., 0., 960.], [0., 1000., 540.], [0., 0., 1.]]),
                           np.array([.1, -.1, .01, 0., 0.]), (1920, 1080))

    def load_for_size(self, size, metadata_size=None):
        capture = MagicMock()
        capture.isOpened.return_value = True
        capture.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: size[0], cv2.CAP_PROP_FRAME_HEIGHT: size[1]}[prop]
        metadata = {'video': 'cached.mp4', 'calibration': 'calibration.json'}
        if metadata_size is not None:
            metadata['image_size'] = metadata_size
        with patch('render_slam_replay.cv2.VideoCapture', return_value=capture), \
             patch('render_slam_replay.Calibration.load', return_value=self.calibration()):
            try:
                return render_slam_replay.load_replay_calibration(metadata)
            finally:
                capture.release.assert_called_once()

    def test_rerender_scales_intrinsics_and_preserves_distortion(self):
        scaled = self.load_for_size((1280, 720), [1280, 720])
        expected = self.calibration().scaled_to((1280, 720))
        self.assertEqual(scaled.image_size, (1280, 720))
        np.testing.assert_allclose(scaled.camera_matrix, expected.camera_matrix)
        np.testing.assert_equal(scaled.dist_coeffs, expected.dist_coeffs)

    def test_legacy_metadata_uses_actual_video_size(self):
        self.assertEqual(self.load_for_size((1280, 720)).image_size, (1280, 720))

    def test_changed_aspect_ratio_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'aspect ratio'):
            self.load_for_size((640, 480), [640, 480])

    def test_cached_observation_size_must_match_actual_video(self):
        with self.assertRaisesRegex(ValueError, 'cached.*size|metadata.*size'):
            self.load_for_size((1280, 720), [1920, 1080])

    def test_existing_render_is_not_overwritten_or_opened_without_explicit_flag(self):
        with patch('sys.argv', ['render_slam_replay.py', 'actions.jsonl']), \
             patch('pathlib.Path.read_text', return_value=json.dumps({'replay': '/in_memory/index.html'})), \
             patch('pathlib.Path.exists', return_value=True), \
             patch('render_slam_replay.load_replay_calibration') as load, \
             patch('render_slam_replay.write_slam_replay') as render, \
             patch('sys.stderr', new_callable=io.StringIO) as errors:
            with self.assertRaises(SystemExit) as failure:
                render_slam_replay.main()
            self.assertEqual(failure.exception.code, 2)
            self.assertIn('--replace-render', errors.getvalue())
            load.assert_not_called()
            render.assert_not_called()

    def test_rerender_passes_already_loaded_actions_to_writer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            actions_path = root / 'actions.jsonl'
            rows = [action()]
            rows[0]['detected_marker_corners'] = {}
            actions_path.write_text(json.dumps(rows[0]) + '\n')
            replay = root / 'replay'
            replay.mkdir()
            metadata = {'replay': str(replay / 'index.html'), 'video': 'cached.mp4',
                        'calibration': 'calibration.json', 'fps': 30}
            actions_path.with_suffix('.meta.json').write_text(json.dumps(metadata))
            history = [snapshot(), snapshot()]
            history[-1]['final'] = True
            with gzip.open(replay / 'native_history.jsonl.gz', 'wt') as stream:
                stream.write('\n'.join(json.dumps(frame) for frame in history) + '\n')
            with patch('sys.argv', ['render_slam_replay.py', str(actions_path), '--replace-render']), \
                 patch('render_slam_replay.load_replay_calibration', return_value=self.calibration()), \
                 patch('render_slam_replay.write_slam_replay') as render:
                render_slam_replay.main()
            self.assertEqual(render.call_args.kwargs['actions'], rows)


if __name__ == '__main__':
    unittest.main()
