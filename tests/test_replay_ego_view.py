"""Exercise the actual replay hand-camera/projection code without a browser."""
import json
from pathlib import Path
import shutil
import subprocess
import unittest


@unittest.skipUnless(shutil.which('node'), 'Node is required')
class ReplayEgoViewTests(unittest.TestCase):
    def run_viewer(self, checks):
        html = (Path(__file__).resolve().parents[1] / 'aruco_track/slam_replay.html').read_text()
        script = html.split('<script>', 1)[1].split('</script>', 1)[0]
        script = script.split("\nfetch('manifest.json')", 1)[0]
        harness = r'''
const assert=require('node:assert/strict'),elements=new Map(),drawCalls=[];
const context=new Proxy({measureText(text){return {width:String(text).length*7}}},{
 get(target,key){if(key in target)return target[key];
  return (...args)=>drawCalls.push([key,...args]);}});
global.document={body:{classList:{add(){}}},getElementById(id){
 if(!elements.has(id))elements.set(id,{value:id==='speed'?'2':'auto',checked:true,
 addEventListener(){},getContext(){return context},setPointerCapture(){},
 getBoundingClientRect(){return {width:640,height:480}},options:[]});return elements.get(id);}};
global.location={search:'',pathname:'/test'};global.devicePixelRatio=1;
global.BroadcastChannel=class{postMessage(){}};global.window={};
const I=[[1,0,0],[0,1,0],[0,0,1]],identity={translation:[0,0,0],rotation:I};
function close(actual,expected,message='values differ'){
 assert.equal(actual.length,expected.length,message);
 actual.forEach((v,i)=>assert.ok(Math.abs(v-expected[i])<1e-10,
  `${message}: ${actual} != ${expected}`));
}
function freeze(value){if(value&&typeof value==='object'){
 Object.values(value).forEach(freeze);Object.freeze(value);}return value;}
'''
        result = subprocess.run([shutil.which('node'), '-e',
                                 'new Function(' + json.dumps(script) + ');\n' + harness + script
                                 + '\n{\n' + checks + '\n}\n'],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_world_to_camera_inverts_rotation_and_cancels_translation(self):
        self.run_viewer(r'''
const angle=.4,c=Math.cos(angle),s=Math.sin(angle);
const camera=freeze({translation:[3,-4,5],rotation:[[0,-c,s],[1,0,0],[0,s,c]]});
const expected=[.2,-.1,1.5];
const world=freeze(camera.translation.map((t,i)=>
 t+camera.rotation[i].reduce((sum,v,j)=>sum+v*expected[j],0)));
close(worldToHandCamera(world,camera),expected,'must use R_wc transpose, not R_wc');
close(worldToHandCamera(camera.translation,camera),[0,0,0]);
const offset=[-8,2,10],shifted={rotation:camera.rotation,
 translation:camera.translation.map((v,i)=>v+offset[i])};
close(worldToHandCamera(world.map((v,i)=>v+offset[i]),shifted),expected,
 'a shared world translation must cancel');
assert.notEqual(worldToHandCamera(world,camera),world,'return a new camera-space point');
''')

    def test_camera_roll_and_opencv_right_down_forward_axes(self):
        self.run_viewer(r'''
const camera={translation:[3,4,5],rotation:[[0,-1,0],[1,0,0],[0,0,1]]};
const toWorld=p=>camera.translation.map((t,i)=>
 t+camera.rotation[i].reduce((sum,v,j)=>sum+v*p[j],0));
const screen=p=>projectHandPoint(worldToHandCamera(p,camera),640,480);
const center=screen(toWorld([0,0,1]));close(center,[320,240]);
const right=screen(toWorld([.1,0,1])),down=screen(toWorld([0,.1,1]));
assert.ok(right[0]>center[0]);close([right[1]],[center[1]]);
assert.ok(down[1]>center[1]);close([down[0]],[center[0]]);
close(screen(toWorld([0,0,2])),center,'camera forward only changes depth');
const fixedWorldPoint=[3.1,4,6],rolled=screen(fixedWorldPoint);
const unrolled=projectHandPoint(worldToHandCamera(fixedWorldPoint,
 {translation:camera.translation,rotation:I}),640,480);
assert.ok(unrolled[0]>320&&rolled[1]<240,'camera roll must rotate the hand image');
close([rolled[0]],[320]);close([unrolled[1]],[240]);
''')

    def test_perspective_depth_default_fov_zoom_and_near_plane(self):
        self.run_viewer(r'''
const w=640,h=480,focal=h*.5/Math.tan(35*Math.PI/180);
const near=projectHandPoint([.1,.2,1],w,h),far=projectHandPoint([.1,.2,2],w,h);
close(near,[w/2+.1*focal,h/2+.2*focal]);
close([near[0]-w/2,near[1]-h/2],far.map((v,i)=>2*(v-[w/2,h/2][i])),
 'equal geometry twice as deep must appear half as large');
handView.zoom=2;
close(projectHandPoint([.1,.2,1],w,h),[w/2+.2*focal,h/2+.4*focal]);
close(projectHandPoint([0,0,1],w,h),[w/2,h/2],'zoom must not move the viewpoint');
handView.zoom=1;
close(projectHandPoint([0,0,.03],w,h),[w/2,h/2]);
for(const z of [.029,0,-.1])assert.equal(projectHandPoint([0,0,z],w,h),null,
 'near-plane and behind-camera points must not project');
''')

    def test_invalid_points_and_camera_fields_are_rejected(self):
        self.run_viewer(r'''
for(const point of [null,undefined,[],[0,1],[0,0,NaN],[Infinity,0,1],[0,-Infinity,1]]){
 assert.equal(worldToHandCamera(point,identity),null);
 assert.equal(projectHandPoint(point,640,480),null);
}
for(const camera of [null,{}, {rotation:I}, {translation:[0,0,0]},
 {translation:[0,NaN,0],rotation:I}, {translation:[0,0,0],rotation:[[1,0],[0,1],[0,0]]},
 {translation:[0,0,0],rotation:[[1,0,0],[0,Infinity,0],[0,0,1]]}]){
 assert.equal(worldToHandCamera([0,0,1],camera),null);
}
''')

    def test_trail_segments_break_at_missing_and_near_clipped_points(self):
        self.run_viewer(r'''
const trail=freeze([[0,0,1],[.1,0,1],null,[.2,0,1],[.3,0,1],
 [.4,0,.02],[.5,0,1],[.6,0,1],[0,0,-1],[.7,0,1],[.8,0,1],null]);
const before=JSON.stringify(trail),segments=handViewSegments(trail,identity,640,480);
assert.deepEqual(segments.map(segment=>segment.start),[0,3,6,9]);
assert.deepEqual(segments.map(segment=>segment.points.length),[2,2,2,2]);
for(const segment of segments)segment.points.forEach((p,j)=>
 close(p,projectHandPoint(trail[segment.start+j],640,480)));
assert.deepEqual(handViewSegments([null,[0,0,-1],null],identity,640,480),[]);
assert.deepEqual(handViewSegments(trail,null,640,480),[],'never reuse an earlier camera');
assert.equal(JSON.stringify(trail),before);
''')

    def test_hand_frame_requires_current_metric_pose_and_matching_map(self):
        self.run_viewer(r'''
revision={active_map:7,maps:[{id:0,metric:true},{id:7,metric:false}]};
const good={map_id:'atlas_0',metric:true,source:'slam',camera:identity,hands:{},trails:{}};
function state(frame,valid){const result=handFrameState(frame);
 assert.equal(result.valid,valid);assert.equal(typeof result.reason,'string');
 if(!valid)assert.ok(result.reason.length,'invalid frames need an explanation');}
frames=[good];state(good,true);
$('maps').value='7';state(good,true);
for(const patch of [{camera:null},{source:'invalid'},{metric:false},{metric:undefined},
 {map_id:null},{map_id:'atlas_99'},{map_id:'atlas_7'},
 {camera:{translation:[NaN,0,0],rotation:I}}])state({...good,...patch},false);
state(null,false);
revision.maps[0].metric=false;state(good,false);
const marker={...good,map_id:'marker_world',source:'marker',
 marker_world_view:{id:'marker_world',metric:true}};
state(marker,true);
state({...marker,marker_world_view:null},false);
state({...marker,marker_world_view:{id:'marker_world',metric:false}},false);
state({...marker,marker_world_view:{id:'atlas_0',metric:true}},false);
state({...marker,map_id:'atlas_0'},false);
state({...marker,source:'invalid'},false);
''')

    def test_renderer_uses_one_current_camera_and_never_mutates_world_data(self):
        self.run_viewer(r'''
revision={active_map:0,maps:[{id:0,metric:true}]};
const hands=freeze({left:[[3,4,6],[3.1,4,6]],right:[[3.2,4,6],[3.3,4,6]]});
const trails=freeze({left:[[3,4,6],null,[3.1,4,6],[3.2,4,6]],
 right:[[3.2,4,6],[3.3,4,6]]});
const currentCamera=freeze({translation:[3,4,5],rotation:I});
let handReads=0,trailReads=0;
const frame={map_id:'atlas_0',metric:true,source:'slam',camera:currentCamera,
 get hands(){handReads++;return hands},get trails(){trailReads++;return trails}};
const before=JSON.stringify({hands,trails,currentCamera}),calls=[];
const transform=worldToHandCamera;
worldToHandCamera=(p,camera)=>{calls.push({p,camera});return transform(p,camera)};
frames=[frame];drawEgoHands();
assert.equal(handReads,1);assert.equal(trailReads,1);
assert.ok(calls.length>0,'renderer must use the world-to-current-camera transform');
assert.ok(calls.every(call=>call.camera===currentCamera));
assert.ok(drawCalls.some(call=>call[0]==='arc'),'valid hand joints must be drawn');
for(const point of Object.values(hands).flat())assert.ok(calls.some(call=>call.p===point),
 'both current hand skeletons must be transformed');
assert.equal(JSON.stringify({hands,trails,currentCamera}),before);
const nextCamera=freeze({translation:[3.1,4,5],rotation:[[0,-1,0],[1,0,0],[0,0,1]]});
frames=[{...frame,camera:nextCamera}];calls.length=0;drawEgoHands();
assert.ok(calls.length>0&&calls.every(call=>call.camera===nextCamera));
for(const patch of [{camera:null},{source:'invalid'},{metric:false},{map_id:'atlas_99'}]){
 frames=[{...frame,...patch}];calls.length=0;drawCalls.length=0;drawEgoHands();
 assert.equal(calls.length,0,'invalid frames must not reuse the last valid hand camera');
 assert.ok(!drawCalls.some(call=>call[0]==='arc'),'invalid frames must not show old joints');
 assert.ok($('hand-frame-status').textContent.length,'invalid frames need visible status');
}
assert.equal(JSON.stringify({hands,trails,currentCamera}),before);
''')

    def test_map_basis_starts_at_first_valid_ego_rotation_including_roll(self):
        self.run_viewer(r'''
assert.equal(overview.yaw,-.65);assert.equal(overview.pitch,.5);
assert.deepEqual(viewBasis(localView),viewBasis(overview));
revision={active_map:0,maps:[{id:0,metric:true,point_count:0,keyframes:[],markers:{}}]};
const angle=.4,c=Math.cos(angle),s=Math.sin(angle);
const rotation=[[0,-c,s],[1,0,0],[0,s,c]],expected=[[0,1,0],[-c,0,s],[s,0,c]];
const good={map_id:'atlas_0',metric:true,source:'slam',
 camera:{translation:[3,-4,5],rotation},hands:{},trails:{}};
frames=freeze([{...good,source:'invalid',camera:identity},{...good,camera:null},
 {...good,camera:{translation:[0,NaN,0],rotation:I}},
 {...good,tail:true,camera:identity},good,{...good,camera:identity}]);
const before=JSON.stringify(frames);
indexInitialMapOrientations(frames);frameIndex=4;draw();
for(const view of views){
 assert.equal(view.yaw,0);assert.equal(view.pitch,0);
 viewBasis(view).forEach((row,i)=>close(row,expected[i],
  'the first valid camera must seed the full R_wc transpose, including roll'));
 const center=project(view,good.camera.translation);
 const right=project(view,good.camera.translation.map((v,i)=>v+rotation[i][0]));
 const down=project(view,good.camera.translation.map((v,i)=>v+rotation[i][1]));
 assert.ok(right[0]>center[0]);close([right[1]],[center[1]]);
 assert.ok(down[1]>center[1]);close([down[0]],[center[0]]);
}
assert.equal(JSON.stringify(frames),before,'indexing and drawing must not alter input poses');
''')

    def test_map_basis_stays_fixed_across_camera_rotation_and_seeking(self):
        self.run_viewer(r'''
const rotation=[[0,-1,0],[1,0,0],[0,0,1]],expected=[[0,1,0],[-1,0,0],[0,0,1]];
const map={id:0,metric:true,point_count:0,keyframes:[],markers:{}};
const good={map_id:'atlas_0',metric:true,source:'slam',hands:{},trails:{}};
frames=freeze([rotation,I,[[1,0,0],[0,0,-1],[0,1,0]],null].map((r,i)=>
 ({...good,sequence:i,camera:r?{translation:[i,2,3],rotation:r}:null})));
timeline=freeze(frames.map((f,i)=>({sequence:i,checkpoint:true,count:0,deleted:[],
 active_map:0,maps:[map]})));
const before=JSON.stringify({frames,timeline});
indexInitialMapOrientations(frames);
for(const index of [2,3,0,1,2,0]){
 frameIndex=index;seek(frames[index].sequence);draw();
 for(const view of views)viewBasis(view).forEach((row,i)=>close(row,expected[i],
  'map orientation must not follow current camera rotation'));
}
assert.equal(JSON.stringify({frames,timeline}),before);
''')

    def test_maps_and_metric_coordinate_domains_have_independent_initial_bases(self):
        self.run_viewer(r'''
const rotations=[[[0,-1,0],[1,0,0],[0,0,1]],
 [[1,0,0],[0,0,-1],[0,1,0]],[[0,0,1],[0,1,0],[-1,0,0]]];
const metadata=(id,metric)=>({id,metric,point_count:0,keyframes:[],markers:{}});
const frame=(map_id,metric,rotation)=>({map_id,metric,source:'slam',
 camera:{translation:[0,0,0],rotation},hands:{},trails:{}});
frames=freeze([frame('atlas_0',false,rotations[0]),frame('atlas_1',true,rotations[1]),
 frame('atlas_0',true,rotations[2]),frame('atlas_0',true,I)]);
const before=JSON.stringify(frames);indexInitialMapOrientations(frames);
for(const [index,id,metric,rotation] of [[0,0,false,rotations[0]],
 [1,1,true,rotations[1]],[2,0,true,rotations[2]],[3,0,true,rotations[2]],
 [0,0,false,rotations[0]],[1,1,true,rotations[1]]]){
 frameIndex=index;revision={active_map:id,maps:[metadata(0,metric),metadata(1,true)]};
 $('maps').value=String(id);draw();
 for(const view of views)viewBasis(view).forEach((row,i)=>
  close(row,rotation.map(axis=>axis[i]),'each map and metric domain needs its own seed'));
}
// An explicit map selection must not use the unrelated current frame's camera.
frameIndex=1;revision={active_map:1,maps:[metadata(0,true),metadata(1,true)]};
$('maps').value='0';draw();
viewBasis(overview).forEach((row,i)=>close(row,rotations[2].map(axis=>axis[i])));
assert.equal(JSON.stringify(frames),before);
''')

    def test_manual_map_rotation_is_synced_and_preserved_when_revisiting_map(self):
        self.run_viewer(r'''
const rotation=[[0,-1,0],[1,0,0],[0,0,1]];
revision={active_map:0,maps:[0,1].map(id=>
 ({id,metric:true,point_count:0,keyframes:[],markers:{}}))};
frames=freeze([0,1].map(id=>({map_id:'atlas_'+id,metric:true,source:'slam',
 camera:{translation:[id,2,3],rotation:id?I:rotation},hands:{},trails:{}})));
indexInitialMapOrientations(frames);draw();
const initial=viewBasis(overview).map(row=>row.slice());
overview.canvas.onpointerdown({clientX:10,clientY:20,button:0,shiftKey:false,pointerId:1});
overview.canvas.onpointermove({clientX:30,clientY:25});
overview.canvas.onpointerup();
assert.notDeepEqual(viewBasis(overview),initial,'map drag must still rotate the fixed view');
assert.deepEqual(viewBasis(localView),viewBasis(overview),'both maps share the user orientation');
close([overview.yaw,overview.pitch],[.16,.04]);
const cy=Math.cos(.16),sy=Math.sin(.16),cp=Math.cos(.04),sp=Math.sin(.04);
const orbit=[[cy,0,sy],[sp*sy,cp,-sp*cy],[-cp*sy,sp,cp*cy]];
const expected=orbit.map(row=>[0,1,2].map(i=>
 row.reduce((sum,v,j)=>sum+v*initial[j][i],0)));
viewBasis(overview).forEach((row,i)=>close(row,expected[i],
 'manual orbit must compose with the initial ego basis'));
const dragged=viewBasis(overview).map(row=>row.slice());
$('maps').value='1';frameIndex=1;draw();
viewBasis(overview).forEach((row,i)=>close(row,I[i],
 'a different map must not inherit the previous map drag'));
localView.canvas.onpointerdown({clientX:0,clientY:0,button:0,shiftKey:false,pointerId:2});
localView.canvas.onpointermove({clientX:-10,clientY:20});localView.canvas.onpointerup();
const secondDragged=viewBasis(localView).map(row=>row.slice());
assert.deepEqual(viewBasis(overview),secondDragged,'local-view drag must sync the overview');
$('maps').value='0';frameIndex=0;draw();
for(const view of views)assert.deepEqual(viewBasis(view),dragged,
 'returning to a map must preserve its manual rotation');
$('maps').value='1';frameIndex=1;draw();
for(const view of views)assert.deepEqual(viewBasis(view),secondDragged);
''')

    def test_map_without_initial_pose_uses_default_until_seed_is_available(self):
        self.run_viewer(r'''
revision={active_map:0,maps:[{id:0,metric:true,point_count:0,keyframes:[],markers:{}}]};
const good={map_id:'atlas_0',metric:true,source:'slam',hands:{},trails:{}};
frames=freeze([{...good,camera:null}]);indexInitialMapOrientations(frames);
const fallback=viewBasis(overview).map(row=>row.slice());draw();
for(const view of views)assert.deepEqual(viewBasis(view),fallback);
const rotation=[[0,-1,0],[1,0,0],[0,0,1]];
frames=freeze([...frames,{...good,camera:{translation:[0,0,0],rotation}}]);
indexInitialMapOrientations(frames);draw();
const seeded=viewBasis(overview).map(row=>row.slice());
seeded.forEach((row,i)=>close(row,rotation.map(axis=>axis[i])));
// Reindexing data must not overwrite an already initialized map's view.
frames=freeze([{...good,camera:identity}]);indexInitialMapOrientations(frames);draw();
for(const view of views)assert.deepEqual(viewBasis(view),seeded);
''')


if __name__ == '__main__':
    unittest.main()
