"""Run the real hybrid replay selectors and binary point caches in Node."""
import json
from pathlib import Path
import shutil
import subprocess
import unittest


@unittest.skipUnless(shutil.which('node'), 'Node is required')
class HybridReplayFrontendTests(unittest.TestCase):
    def run_viewer(self, checks):
        html = (Path(__file__).resolve().parents[1] / 'aruco_track/slam_replay.html').read_text()
        script = html.split('<script>', 1)[1].split('</script>', 1)[0]
        script = script.split("\nfetch('manifest.json')", 1)[0]
        harness = r'''
const assert=require('node:assert/strict'),elements=new Map();
const context=new Proxy({},{get(){return ()=>{}}});
global.document={body:{classList:{add(){}}},getElementById(id){
 if(!elements.has(id))elements.set(id,{value:id==='speed'?'2':'auto',checked:true,
 addEventListener(){},getContext(){return context},setPointerCapture(){},
 getBoundingClientRect(){return {width:640,height:480}},options:[]});return elements.get(id);}};
global.location={search:'',pathname:'/test'};global.devicePixelRatio=1;
global.BroadcastChannel=class{postMessage(){}};global.window={};
const I=[[1,0,0],[0,1,0],[0,0,1]],identity={translation:[0,0,0],rotation:I};
function snapshot(id,metric=true){return {id,metric,point_count:0,keyframes:[],markers:{},loops:[]};}
'''
        result = subprocess.run([shutil.which('node'), '-e',
                                 'new Function(' + json.dumps(script) + ');\n' + harness + script
                                 + '\n{\n' + checks + '\n}\n'],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_final_and_process_map_pose_sources_remain_separate(self):
        self.run_viewer(r'''
manifest={replay_mode:'hybrid'};
const early=snapshot(3,false),final=snapshot(7,true);
timeline=[{sequence:0,maps:[early],active_map:3},{sequence:1,maps:[final],active_map:7,final:true}];
revision=timeline[0];
const processPose={translation:[1,2,3],rotation:I};
frames=[{source_frame:0,camera:identity,map_id:'atlas_7',metric:true,source:'head-slam',
 hands:{left:[[0,0,1]]},trails:{left:[[0,0,1]]},orb_map_point_ids:[2],
 map_revision:99,process_camera:processPose,process_map_id:'atlas_3',process_map_revision:4,process_metric:false,
 process_source:'head-slam',process_orb_map_point_ids:[1]}];
assert.equal(mapRevision(overview),timeline[1]);assert.equal(mapRevision(localView),timeline[0]);
assert.equal(selectMap(overview),7);assert.equal(selectMap(localView),3);
assert.equal(mapFrame(overview),frames[0]);
assert.equal(mapFrame(localView).camera,processPose);
assert.equal(mapFrame(localView).map_revision,4);assert.equal(mapFrame(overview).map_revision,99);
assert.deepEqual(mapFrame(localView).hands,{});assert.deepEqual(mapFrame(localView).trails,{});
assert.deepEqual([...localPointIds(overview)],[2]);assert.deepEqual([...localPointIds(localView)],[1]);
assert.equal(handFrameState(frames[0]).valid,true,'final hands use final metric map, not arbitrary historical map');
''')

    def test_missing_historical_pose_does_not_borrow_recovered_pose(self):
        self.run_viewer(r'''
manifest={replay_mode:'hybrid'};
timeline=[{maps:[],active_map:null},{maps:[snapshot(0)],active_map:0,final:true}];
revision=timeline[0];
frames=[{camera:identity,map_id:'atlas_0',metric:true,source:'head-slam',
 hands:{left:[[0,0,1]]},trails:{left:[[0,0,1]]},orb_map_point_ids:[2]}];
assert.equal(mapFrame(localView).camera,null);assert.equal(mapFrame(localView).map_id,null);
assert.equal(mapFrame(localView).source,'invalid');assert.equal(mapFrame(localView).metric,false);
assert.equal(mapFrame(localView).map_revision,0);
assert.deepEqual([...localPointIds(localView)],[]);assert.equal(selectMap(localView),null);
assert.equal(selectMap(overview),0);assert.equal(handFrameState(frames[0]).valid,true);
frames[0].camera=null;frames[0].map_id=null;frames[0].source='invalid';
assert.equal(selectMap(overview),0,'unique final map can be shown without asserting frame localization');
assert.equal(handFrameState(frames[0]).valid,false);
timeline[1].maps.push(snapshot(1));assert.equal(selectMap(overview),null,'unrelated maps must not be guessed');
''')

    def test_final_point_cache_survives_history_updates_and_backward_seek(self):
        self.run_viewer(r'''
manifest={replay_mode:'hybrid'};
const buffer=new ArrayBuffer(4*28);binary=new DataView(buffer);
const records=[[0,1,1,0,0],[0,1,2,0,0],[0,2,3,0,0],[0,2,4,0,0]];
records.forEach((r,i)=>{const o=i*28;binary.setBigUint64(o,BigInt(r[0]),true);binary.setBigUint64(o+8,BigInt(r[1]),true);for(let j=0;j<3;j++)binary.setFloat32(o+16+j*4,r[j+2],true);});
const map=snapshot(0);
timeline=[{sequence:0,checkpoint:true,offset:0,count:1,deleted:[],maps:[map],active_map:0},
 {sequence:1,checkpoint:false,offset:28,count:2,deleted:[],maps:[map],active_map:0},
 {sequence:2,checkpoint:false,offset:84,count:1,deleted:[[0,1]],maps:[map],active_map:0,final:true}];
frames=[{map_id:'atlas_0',orb_map_point_ids:[2],process_map_id:'atlas_0',process_orb_map_point_ids:[1]}];
indexFinalPoints();assert.equal(finalPoints.size,1);assert.deepEqual(finalPoints.get('0:2'),[0,4,0,0]);
assert.equal(sequence,-1,'preloading final points must not seek causal history');
seek(0);assert.deepEqual(mapPointEntries(map,false,localView),[[1,0,0]]);
assert.deepEqual(mapPointEntries(map,false,overview),[[4,0,0]]);
assert.deepEqual(mapPointEntries(map,true,localView),[[1,0,0]]);
assert.deepEqual(mapPointEntries(map,true,overview),[[4,0,0]]);
seek(1);assert.equal(points.size,2);seek(2);assert.equal(points.size,1);
seek(0);assert.deepEqual([...points.values()],[[0,1,0,0]]);
assert.deepEqual([...finalPoints.values()],[[0,4,0,0]],'history rewind cannot mutate final cloud');
''')

    def test_historical_marker_only_view_is_not_final_marker_world(self):
        self.run_viewer(r'''
manifest={replay_mode:'hybrid'};
timeline=[{maps:[],active_map:null},{maps:[snapshot(0)],active_map:0,final:true}];revision=timeline[0];
const marker={...snapshot('marker_world'),geometry_source:'fixed'};
frames=[{camera:identity,map_id:'atlas_0',metric:true,source:'head-slam',
 process_camera:identity,process_map_id:'marker_world',process_metric:true,
 process_source:'marker',process_marker_world_view:marker}];
assert.equal(selectMap(overview),0);assert.equal(selectMap(localView),'marker_world');
assert.deepEqual(availableMaps(localView),[marker]);
assert.equal(availableMaps(overview).length,1);assert.equal(availableMaps(overview)[0].id,0);
''')

    def test_shared_initial_orientation_and_local_center_use_process_pose(self):
        self.run_viewer(r'''
manifest={replay_mode:'hybrid'};
const m=snapshot(0),finalPose={translation:[10,0,0],rotation:[[0,-1,0],[1,0,0],[0,0,1]]};
timeline=[{maps:[m],active_map:0},{maps:[m],active_map:0,final:true}];revision=timeline[0];
frames=[{camera:finalPose,map_id:'atlas_0',metric:true,source:'head-slam',
 process_camera:{translation:[1,2,3],rotation:I},process_map_id:'atlas_0',process_metric:true,process_source:'marker'}];
indexInitialMapOrientations(frames);syncInitialMapOrientation();
const top=viewBasis(overview),local=viewBasis(localView);
assert.deepEqual(top,local,
 'overview and local windows must share the same initial-Ego orientation; '
 + 'the local view changes center/scale only');
for(const view of views){view.width=640;view.height=480;}
fit(localView,m);assert.deepEqual(localView.center,[1,2,3]);
frames[0].camera.rotation=I;syncInitialMapOrientation();
assert.deepEqual(overview.baseBasis,[[0,1,0],[-1,0,0],[0,0,1]],'later final camera rotation must not spin Atlas');
assert.deepEqual(viewBasis(overview),top);
assert.deepEqual(viewBasis(localView),local);
''')

    def test_legacy_process_and_final_modes_still_share_their_own_state(self):
        self.run_viewer(r'''
const m=snapshot(0),r={maps:[m],active_map:0};timeline=[r];revision=r;
frames=[{camera:identity,map_id:'atlas_0',metric:true,source:'head-slam'}];
for(const mode of ['process','final-map']){
 manifest={replay_mode:mode};
 assert.equal(mapFrame(localView),frames[0]);assert.equal(mapFrame(overview),frames[0]);
 assert.equal(mapRevision(localView),r);assert.equal(mapRevision(overview),r);
 assert.equal(mapPoints(localView),points);assert.equal(mapPoints(overview),points);
}
''')

    def test_actual_draw_shows_final_cloud_at_zero_but_not_future_local_map(self):
        self.run_viewer(r'''
manifest={replay_mode:'hybrid',source_fps:90,fps:30};
const final={...snapshot(0),point_count:1};
timeline=[{maps:[],active_map:null},{maps:[final],active_map:0,final:true}];revision=timeline[0];
frames=[{source_frame:0,sequence:0,camera:null,map_id:null,metric:false,source:'invalid',hands:{},trails:{}}];
finalPoints.set('0:1',[0,1,0,1]);
draw();
assert.match($('overview-status').textContent,/显示点 1\/1/);
assert.match($('overview-status').textContent,/此帧未成功离线定位/);
assert.match($('local-status').textContent,/当时尚无可用地图/);
assert.match($('local-status').textContent,/当时定位无效/);
assert.match($('hand-frame-status').textContent,/当前相机失定位/);
assert.equal(points.size,0);assert.equal(finalPoints.size,1);
''')


if __name__ == '__main__':
    unittest.main()
