import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import zlib from 'node:zlib';
const context=vm.createContext({console});
vm.runInContext(fs.readFileSync(new URL('../aruco_track/replay_event_overlay.js',import.meta.url),'utf8'),context);
const {buildEvents,diffGeometry,eventsAt,focusEvent,eventTitle,fitBev95,smoothExtent,rotatePlane}=context.MonoTagReplayEventModel;
const scaleEvent={map:0,type:'scale_reanchor',correction:true,scale:1.137,title:'尺度重锚定'};
assert.equal(focusEvent([scaleEvent,{map:0,type:'status'},{map:0,correction:true,type:'marker_loop'}],0),scaleEvent);
assert.ok(eventTitle(scaleEvent).includes('13.70%'));
assert.equal(focusEvent([scaleEvent],3),undefined,'independent maps must not inherit highlights');
assert.equal(smoothExtent(1,1,1/30),1);
assert.equal(smoothExtent(1,10,0),1,'paused redraw must not change scale');
assert.ok(smoothExtent(1,100,1/30)<=Math.exp(.20/30)+1e-12,'zoom out rate limited');
assert.ok(smoothExtent(1,.001,1/30)>=Math.exp(-.12/30)-1e-12,'zoom in rate limited');
assert.ok(Math.abs(smoothExtent(1,1.12001,1/30)-smoothExtent(1,1.11999,1/30))<1e-6,'no abrupt 12-percent threshold');
let slow=1,fast=1;for(let i=0;i<30;i++)slow=smoothExtent(slow,2,1/30);for(let i=0;i<60;i++)fast=smoothExtent(fast,2,1/60);
assert.ok(Math.abs(slow-fast)<.003,'smoothing depends on elapsed time, not frame count');
const turned=rotatePlane([[1,0,0],[0,1,0],[0,0,1]],Math.PI/2);
assert.ok(Math.abs(turned[0][0])<1e-12&&Math.abs(turned[0][1]-1)<1e-12);
assert.deepEqual([...turned[2]],[0,0,1],'sharing heading does not change BEV vertical');
const cloud=Array.from({length:95},(_,i)=>[Math.cos(.7)*i,Math.sin(.7)*i,0]);
cloud.push(...Array.from({length:5},(_,i)=>[1e6+i,1e6,0]));
const original=JSON.stringify(cloud),bev=fitBev95(cloud,[[1,0,0],[0,1,0],[0,0,1]],640,300);
assert.equal(bev.selected.length,95);assert.ok(bev.selected.every(p=>p[0]<1000));
assert.equal(JSON.stringify(cloud),original,'display cannot mutate cloud');
assert.deepEqual([...bev.basis[2]],[0,0,1],'fit must stay in the BEV plane');
for(const p of bev.selected){const q=p.map((v,i)=>v-bev.center[i]);
 assert.ok(Math.abs(q.reduce((s,v,i)=>s+v*bev.basis[0][i],0))*bev.scale<=640*.451);
 assert.ok(Math.abs(q.reduce((s,v,i)=>s+v*bev.basis[1][i],0))*bev.scale<=300*.451);}
assert.ok(bev.scale>5,'far outliers must not shrink the map');
const map=(loops=[])=>({id:0,metric:true,keyframes:[[1,0,[0,0,0,0,0,0,1]],[2,2,[1,0,0,0,0,0,1]]],loops});
const marker={sequence:1,map_id:0,type:'marker_loop',status:'accepted',affected_keyframes:[1,2]};
const rows=[{sequence:0,timestamp:0,maps:[map()],events:['新地图']},
 {sequence:1,timestamp:1,maps:[map()],marker_graph_events:[marker]},
 {sequence:2,timestamp:2,maps:[map()],marker_graph_events:[marker]},
 {sequence:3,timestamp:3,maps:[map([[1,2]])],final:true,marker_graph_events:[marker]}];
const frames=Array.from({length:120},(_,i)=>({sequence:3,observation_sequence:Math.min(2,Math.floor(i/30)),source_frame:Math.min(29,Math.floor(i/3)),tail:i>=90}));
const events=buildEvents(rows,frames,{fps:30,source_fps:10});
assert.equal(events.filter(e=>e.type==='marker_loop').length,1,'cumulative event duplicated');
assert.equal(events.find(e=>e.type==='marker_loop').frame,30,'used final-label sequence instead of publication');
assert.equal(events.find(e=>e.type==='visual_loop'&&e.correction).frame,90);
// More delayed final publication: candidate notice must not move final geometry.
rows[3].timestamp=4;
const delayed=buildEvents(rows,frames,{fps:30,source_fps:10});
const notice=delayed.find(e=>e.status==='retrospective');
assert.equal(notice.frame,60);assert.equal(notice.correction,false);
assert.equal(delayed.find(e=>e.type==='visual_loop'&&e.correction).frame,90);
const tailFrames=frames.concat(Array.from({length:90},()=>({...frames.at(-1)})));
assert.ok(eventsAt(delayed,90,tailFrames,rows,30).length,'tail starts with correction');
assert.ok(eventsAt(delayed,180,tailFrames,rows,30).length,'tail correction remains visible in last second');
assert.ok(eventsAt(delayed,209,tailFrames,rows,30).length,'paused final frame retains correction');
assert.ok(eventsAt(delayed,90,tailFrames,rows,30).length,'rewind restores correction');
const before=map(),after=map();after.keyframes[1][2][0]+=.0003;
const g=diffGeometry(before,after,new Map([[1,[0,0,0]],[2,[1,0,0]],[4,[0,0,0]]]),new Map([[1,[.0005,0,0]],[2,[1,0,0]],[3,[0,0,0]]]));
assert.deepEqual([...g.changedKF],[2]);assert.deepEqual([...g.changedPoints],[1]);
assert.ok(Math.abs(g.maxPoint-.0005)<1e-10,'display exaggerated true displacement');
const source={id:3,metric:true,keyframes:[[99,3,[100,0,0,0,0,0,1]]]};
const merged={...after,keyframes:[...after.keyframes,[99,3,[2,0,0,0,0,0,1]],[100,4,[3,0,0,0,0,0,1]]]};
const mg=diffGeometry(before,merged,new Map([[1,[0,0,0]]]),new Map([[1,[.1,0,0]],[99,[2,0,0]],[100,[3,0,0]]]),source,new Map([[99,[100,0,0]],[101,[101,0,0]]]));
assert.deepEqual([...mg.importedKF],[99]);assert.deepEqual([...mg.importedPoints],[99]);
assert.ok(mg.highlightedKF.has(99));assert.ok(mg.highlightedPoints.has(99));
assert.ok(!mg.changedKF.has(99),'different source gauge is not a target-frame displacement');
assert.ok(!mg.highlightedKF.has(100));assert.ok(!mg.highlightedPoints.has(100),'unrelated new points are not merge evidence');
assert.equal(mg.beforeMap,before,'source geometry must not be concatenated into target ghost');
const mergeRows=[{maps:[before,source]},{maps:[merged],final:true,marker_graph_events:[{...marker,type:'marker_map_merge',source_map_id:3,target_map_id:0}]}];
const mergeEvents=buildEvents(mergeRows,[{sequence:0},{sequence:1,tail:true}],{fps:30});
assert.equal(mergeEvents[0].sourceMap,3);assert.equal(mergeEvents[0].targetMap,0);
assert.equal(eventsAt(mergeEvents,0,[{},{tail:true}],mergeRows,30).length,0,'no premature merge highlight');
if(process.argv[2]){
 const path=process.argv[2],read=name=>JSON.parse(zlib.gunzipSync(fs.readFileSync(path+'/'+name)));
 const actual=buildEvents(read('timeline.json.gz'),read('video_frames.json.gz'),JSON.parse(fs.readFileSync(path+'/manifest.json')));
 const visual=actual.find(e=>e.type==='visual_loop'&&e.status==='retrospective');
 assert.ok(visual.time>126&&visual.time<127);assert.equal(visual.correction,false);
 const marker=actual.find(e=>e.type==='marker_loop'&&e.status==='accepted');assert.ok(marker.time>149&&marker.time<150);
 console.log(actual.filter(e=>e.type!=='status').map(e=>({type:e.type,time:e.time,title:e.title,correction:e.correction})));
}
console.log('PASS event clocks / cumulative dedup / deferred snapshot honesty / actual geometry changes');
