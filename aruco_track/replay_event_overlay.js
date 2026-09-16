/* Display-only event clock and correction overlays. Never edits SLAM data. */
(() => {
  'use strict';
  const names = {marker_loop:'Marker 回环',scale_reanchor:'尺度重锚定',
    marker_global_ba:'全局 BA',map_merge:'地图合并',marker_merge:'地图合并',
    marker_map_merge:'共同 Marker 合图'};
  const distance = (a,b) => Math.hypot(...a.map((v,i)=>v-b[i]));
  function buildEvents(rows, videoFrames, meta) {
    const fps=Number(meta.fps)||30, sourceFps=Number(meta.source_fps)||fps;
    const firstFrames=new Map();
    videoFrames.forEach((f,i)=>{const s=f.tail?f.sequence:(f.observation_sequence??f.sequence);if(!firstFrames.has(s))firstFrames.set(s,i);});
    function publicationFrame(s){for(let i=s;i<rows.length;i++)if(firstFrames.has(i))return firstFrames.get(i);return videoFrames.length-1;}
    // Native graph events carry two clocks. `frame` is the first publication
    // containing the committed result; the candidate clock records when the
    // keyframe pair/interval was observed. Keep both so an offline final map
    // can reveal an already-computed correction at its candidate frame while
    // the causal process view still waits for the real publication.
    function candidateClock(event){
      const timestamp=Number(event.candidate_timestamp),sourceFrame=Number(event.candidate_frame);
      // Native's legacy default is candidate_timestamp=0 with frame=-1;
      // treat that pair as “no candidate metadata” rather than backdating an
      // unrelated event to the first video frame. A genuine t=0 candidate
      // also carries candidate_frame=0 and remains accepted below.
      const hasTimestamp=Number.isFinite(timestamp)&&timestamp>1e-9;
      const hasSourceFrame=Number.isFinite(sourceFrame)&&sourceFrame>=0;
      if(!hasTimestamp&&!hasSourceFrame)return null;
      const targetSource=hasTimestamp?Math.ceil(timestamp*sourceFps-1e-5):Math.round(sourceFrame);
      let frame=videoFrames.findIndex(f=>!f.tail&&Number(f.source_frame)>=targetSource);
      if(frame<0)frame=Math.max(0,videoFrames.length-1);
      return {frame,time:hasTimestamp?timestamp:frame/fps,
        sourceFrame:hasSourceFrame?Math.round(sourceFrame):targetSource};
    }
    const events=[],seen=new Set(),edges=new Set();
    const add=e=>{if(!seen.has(e.id)){seen.add(e.id);events.push(e);}};
    rows.forEach((row,index)=>{
      const frame=publicationFrame(index),base={frame,time:frame/fps,publication:index,before:index-1,
        publicationFrame:frame,publicationTime:frame/fps};
      for(const e of row.marker_graph_events||[]) {
        const id=`graph:${e.map_id}:${e.sequence}:${e.type}`;
        const accepted=e.status==='accepted';
        const candidate=candidateClock(e);
        add({...base,id,map:e.map_id,type:e.type,status:e.status,
          title:`${names[e.type]||e.type} · ${accepted?'已提交':'已拒绝'}`,
          detail:e.type==='marker_map_merge'
            ?`图 ${e.source_map_id} → 图 ${e.target_map_id} · ${e.reason||''}`
            :e.reason||'',scale:e.type==='scale_reanchor'?Number(e.scale):null,
          correction:accepted,affected:e.affected_keyframes||[],pair:null,
          sourceMap:e.source_map_id,targetMap:e.target_map_id,
          candidateFrame:candidate?.frame??null,candidateTime:candidate?.time??null,
          candidateSourceFrame:candidate?.sourceFrame??null,
          retroactive:Boolean(candidate&&candidate.frame<frame)});
      }
      for(const map of row.maps||[]) {
        const byId=new Map((map.keyframes||[]).map(k=>[k[0],k]));
        for(const pair of map.loops||[]) {
          const key=`visual:${map.id}:${[...pair].sort((a,b)=>a-b).join(':')}`;
          if(edges.has(key))continue;edges.add(key);
          const candidate=Math.max(...pair.map(id=>Number(byId.get(id)?.[1])||0));
          const delayed=Boolean(row.final)&&candidate+1<Number(row.timestamp);
          if(delayed) {
            // This old archive contains only the combined final publication.
            // Do NOT backdate its future points/BA to the candidate frame.
            const targetSource=Math.ceil(candidate*sourceFps-1e-5);
            let first=videoFrames.findIndex(f=>!f.tail&&Number(f.source_frame)>=targetSource);
            if(first<0)first=Math.max(0,Math.ceil(candidate*fps));
            add({...base,id:key+':notice',map:map.id,type:'visual_loop',status:'retrospective',
              frame:first,time:first/fps,pair,correction:false,
              title:'视觉回环 · 离线回溯确认',
              detail:'本旧包缺少该次独立修正快照；不提前借用最终地图。收尾展示已记录的综合修正。'});
          }
          add({...base,id:key,map:map.id,type:'visual_loop',status:'accepted',pair,correction:true,
            title:delayed?'收尾综合修正 · 含离线视觉回环':'视觉回环 · 已提交',
            detail:delayed?`候选 ${candidate.toFixed(2)} s；此快照也包含收尾其他优化，不归因于单次回环。`:'',
            candidate,delayed});
        }
      }
      for(const entry of row.events||[]) {
        const description=typeof entry==='string'?entry:entry.description;
        if(!description)continue;
        // Structured graph entries above give clearer reasons and geometry.
        if((row.marker_graph_events||[]).some(e=>Number(e.sequence)>=0)&&/回环|重锚定|再锚定|全局.?BA/.test(description))continue;
        add({...base,id:`status:${index}:${description}`,type:'status',status:'info',title:description,detail:'',correction:false});
      }
    });
    for(const e of meta.marker_loop_events||[]) {
      const id=`graph:${e.map_id}:${e.sequence}:marker_loop`;
      const existing=events.find(x=>x.id===id);
      const pair=[e.keyframe_a,e.keyframe_b];
      if(existing){existing.pair=pair;continue;}
      const index=e.commit_sequence,frame=publicationFrame(index);
      add({id,map:e.map_id,type:'marker_loop',status:'accepted',frame,time:frame/fps,
        publication:index,before:e.before_sequence,correction:true,pair,
        title:`Marker 回环 · ID ${e.marker_id} · 已提交`,detail:'',affected:e.affected_keyframes||[]});
    }
    return events.sort((a,b)=>a.frame-b.frame||a.id.localeCompare(b.id));
  }
  function diffGeometry(beforeMap,afterMap,beforePoints,afterPoints,sourceMap=null,sourcePoints=new Map()) {
    const before=new Map((beforeMap?.keyframes||[]).map(k=>[k[0],k]));
    const changedKF=new Set(),changedPoints=new Set();let maxKF=0,maxPoint=0;
    for(const k of afterMap?.keyframes||[]) {
      const old=before.get(k[0]);if(!old)continue;
      const d=distance(old[2].slice(0,3),k[2].slice(0,3));
      if(d>1e-7){changedKF.add(k[0]);maxKF=Math.max(maxKF,d);}
    }
    for(const [id,p] of afterPoints) {
      const old=beforePoints.get(id);if(!old)continue;
      const d=distance(old,p);if(d>1e-7){changedPoints.add(id);maxPoint=Math.max(maxPoint,d);}
    }
    // A merge moves identities across map IDs. They are not ordinary newly
    // triangulated points, and comparing only the target silently misses them.
    // Source coordinates remain in their own gauge: never draw them as a
    // pre-merge ghost in the target without a recorded alignment transform.
    const afterKF=new Set((afterMap?.keyframes||[]).map(k=>k[0]));
    const importedKF=new Set((sourceMap?.keyframes||[]).map(k=>k[0]).filter(id=>afterKF.has(id)&&!before.has(id)));
    const importedPoints=new Set([...sourcePoints.keys()].filter(id=>afterPoints.has(id)&&!beforePoints.has(id)));
    const highlightedKF=new Set([...changedKF,...importedKF]);
    const highlightedPoints=new Set([...changedPoints,...importedPoints]);
    return {changedKF,changedPoints,importedKF,importedPoints,highlightedKF,highlightedPoints,maxKF,maxPoint,beforeMap,afterMap};
  }
  function eventsAt(catalog,frameIndex,frames,rows,fps){
    // Keep shutdown corrections visible when paused on the last frame, too.
    const span=Math.ceil(fps*6);
    return catalog.filter(e=>e.frame<=frameIndex&&(frameIndex<e.frame+span||(frames[frameIndex]?.tail&&rows[e.publication]?.final)));
  }
  function eventFrame(event,offline){
    return offline&&Number.isFinite(Number(event.candidateFrame))
      ?Number(event.candidateFrame):Number(event.frame);
  }
  function focusEvent(active,mapId){
    const relevant=active.filter(e=>e.map===mapId||e.map===undefined);
    // Routine status/rejection messages must not immediately cover a committed
    // correction. Show scale changes for their full highlight interval.
    return relevant.filter(e=>e.correction&&e.type==='scale_reanchor').at(-1)
      ||relevant.filter(e=>e.correction).at(-1)||relevant.at(-1);
  }
  function eventTitle(event){
    return event.title+(event.type==='scale_reanchor'&&event.correction&&Number.isFinite(event.scale)
      ?` · ×${event.scale.toFixed(4)} (${(100*(event.scale-1)).toFixed(2)}%)`: '');
  }
  const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
  function rotatePlane(basis,angle){const c=Math.cos(angle),s=Math.sin(angle);return[
    basis[0].map((v,i)=>c*v+s*basis[1][i]),
    basis[0].map((v,i)=>-s*v+c*basis[1][i]),basis[2]];}
  function fitBev95(values,basis,width,height){
    if(!values.length)return null;
    const median=a=>[...a].sort((a,b)=>a-b)[Math.floor(a.length/2)];
    const middle=[0,1,2].map(i=>median(values.map(p=>p[i])));
    // Spatial trimming, not ID sampling: distant points cannot set the zoom.
    const selected=[...values].sort((a,b)=>distance(a,middle)-distance(b,middle)).slice(0,Math.max(1,Math.ceil(values.length*.95)));
    const projected=selected.map(p=>basis.map(axis=>dot(axis,p)));
    let best=null;
    // Search only in-plane rotations: remain BEV, never tilt to fit outliers.
    for(let degree=-90;degree<90;degree+=5){
      const angle=degree*Math.PI/180,c=Math.cos(angle),s=Math.sin(angle);
      let minU=Infinity,maxU=-Infinity,minV=Infinity,maxV=-Infinity;
      for(const p of projected){const u=c*p[0]+s*p[1],v=-s*p[0]+c*p[1];minU=Math.min(minU,u);maxU=Math.max(maxU,u);minV=Math.min(minV,v);maxV=Math.max(maxV,v);}
      const scale=Math.min(width*.9/Math.max(1e-6,maxU-minU),height*.9/Math.max(1e-6,maxV-minV));
      if(!best||scale>best.scale*1.000001||(Math.abs(scale-best.scale)<1e-6&&Math.abs(angle)<Math.abs(best.angle)))best={angle,scale,u:(minU+maxU)/2,v:(minV+maxV)/2};
    }
    const rotated=rotatePlane(basis,best.angle),center=[best.u,best.v,median(projected.map(p=>p[2]))];
    return {...best,selected,basis:rotated,center:[0,1,2].map(i=>rotated.reduce((s,axis,j)=>s+axis[i]*center[j],0))};
  }
  function smoothExtent(current,target,dt){
    if(!(current>0)||!(target>0)||dt<=0)return current;
    const error=Math.log(target/current),soft=Math.sign(error)*Math.max(0,Math.abs(error)-Math.log(1.04));
    return current*Math.exp(Math.max(-.12*dt,Math.min(.20*dt,soft*(1-Math.exp(-dt/1.6)))));
  }
  globalThis.MonoTagReplayEventModel={buildEvents,diffGeometry,eventsAt,focusEvent,eventTitle,fitBev95,smoothExtent,rotatePlane};
  if(typeof document==='undefined'||typeof draw!=='function')return;
  const baseBasis=viewBasis,baseFit=fit,baseDisplayed=displayedMapPointEntries;
  viewBasis=function(view){
    const basis=baseBasis(view);
    return view.mode==='overview'&&overview.bevFit?rotatePlane(basis,overview.bevFit.angle):basis;
  };
  fit=function(view,m,percentile=.9,smooth=false){
    if(view.mode!=='overview'){
      const previous=view.extent,zoom=view.zoom,dt=(frameIndex-view.lastFitFrame)/(Number(manifest?.fps)||30);
      // Ask either legacy or current player for the same raw fit target.
      baseFit(view,m,percentile,false);
      if(smooth&&dt>=0&&dt<=.5&&previous>0){view.extent=smoothExtent(previous,view.extent,dt);view.zoom=zoom;}
      else view.zoomEpoch=(view.zoomEpoch||0)+1;
      return;
    }
    const basis=baseBasis(view),key=JSON.stringify([m.id,m.metric,m.revision,m.point_count,view.width,view.height,basis]);
    if(view.bevKey!==key){
      view.bevFit=fitBev95(mapPointEntries(m,false,view),basis,view.width,view.height);view.bevKey=key;
      view.bevPointKeys=new Set((view.bevFit?.selected||[]).map(p=>p.join(',')));
    }
    if(!view.bevFit)return baseFit(view,m,.95,smooth);
    view.center=view.bevFit.center.slice();view.extent=Math.max(1e-6,Math.min(view.width,view.height)*.36/view.bevFit.scale);
    view.zoom=1;view.pan=[0,0];view.projectionBasis=viewBasis(view);
  };
  displayedMapPointEntries=function(m,view=overview){return view.mode==='overview'&&view.bevFit?view.bevFit.selected:baseDisplayed(m,view);};
  const viewHint=document.querySelector('header .toolbar .hint');
  if(viewHint)viewHint.textContent='全局：最适配 BEV / 主体 95% · 局部：Ego BEV / 90% · 平滑缩放';
  // Brief wheel easing is display-only. Stop when a seek/map switch refits.
  let wheelAnimation=null;
  localView.canvas.onwheel=e=>{
    e.preventDefault();
    const delta=e.deltaY*(e.deltaMode===1?16:e.deltaMode===2?localView.height:1);
    const epoch=localView.zoomEpoch||0;
    if(wheelAnimation&&wheelAnimation.epoch!==epoch)wheelAnimation=null;
    if(!wheelAnimation){
      wheelAnimation={target:localView.zoom,time:performance.now(),epoch};
      const animation=wheelAnimation;
      const step=now=>{
        if(wheelAnimation!==animation)return;
        if((localView.zoomEpoch||0)!==animation.epoch){wheelAnimation=null;return;}
        const dt=Math.min(.05,Math.max(0,(now-animation.time)/1000));animation.time=now;
        const error=Math.log(animation.target/localView.zoom);
        localView.zoom*=Math.exp(error*(1-Math.exp(-dt/.12)));
        if(Math.abs(error)<.001){localView.zoom=animation.target;wheelAnimation=null;}
        draw();if(wheelAnimation===animation)requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    }
    wheelAnimation.target=Math.max(.02,Math.min(100,wheelAnimation.target*Math.exp(-Math.max(-250,Math.min(250,delta))*.001)));
  };
  let indexedRows=null,catalog=[],cache=new Map(),historyPanel;
  const style=document.createElement('style');
  style.textContent=`#replay-event-history{position:fixed;right:12px;top:50px;bottom:12px;width:min(500px,95vw);background:#fff;border:1px solid #bbb;z-index:20;padding:12px;overflow:auto;display:none}#replay-event-history button{display:block;width:100%;text-align:left;margin:4px 0;padding:8px;font-size:12px}`;
  document.head.append(style);
  historyPanel=document.createElement('div');historyPanel.id='replay-event-history';document.body.append(historyPanel);
  const historyButton=document.createElement('button');historyButton.textContent='事件记录';historyButton.id='replay-event-history-button';
  document.querySelector('header .toolbar').prepend(historyButton);
  historyButton.onclick=()=>{historyPanel.style.display=historyPanel.style.display==='block'?'none':'block';};
  // Supersede the older split overlays (some compare process to FINAL camera,
  // which is not a per-event correction). The base map renderer stays intact.
  if(typeof drawMarkerLoops==='function')drawMarkerLoops=()=>{};
  if(typeof drawLoopTransition==='function')drawLoopTransition=()=>{};
  if(typeof updateVideoLoopBanner==='function')updateVideoLoopBanner=()=>{};
  if(typeof drawReanchorHighlight==='function')drawReanchorHighlight=()=>{};
  if(typeof reanchorHighlight==='function')reanchorHighlight=()=>null;
  const oldBanner=$('video-loop-banner');if(oldBanner)oldBanner.style.display='none';
  function clockLabel(seconds){const m=Math.floor(seconds/60),s=(seconds-m*60).toFixed(2).padStart(5,'0');return `${m}:${s}`;}
  function ensureIndex(){
    if(!manifest||!timeline.length||!frames.length)return false;
    if(indexedRows===timeline)return true;
    indexedRows=timeline;catalog=buildEvents(timeline,frames,manifest);cache.clear();historyPanel.replaceChildren();
    const title=document.createElement('strong');title.textContent='事件记录 · 点击跳转（视频时间）';historyPanel.append(title);
    for(const e of catalog){const b=document.createElement('button');
      const displayFrame=(finalMapReplay()||hybridReplay())&&Number.isFinite(Number(e.candidateFrame))?Number(e.candidateFrame):e.frame;
      const displayTime=displayFrame/(Number(manifest.fps)||30);
      b.textContent=`${clockLabel(displayTime)}  ${eventTitle(e)}${e.retroactive?' · 候选帧':''}`;
      b.onclick=()=>{video.pause();video.currentTime=(displayFrame+.1)/(Number(manifest.fps)||30);update(displayFrame);historyPanel.style.display='none';};historyPanel.append(b);}
    return true;
  }
  function activeEvents(view){
    const offline=typeof offlineView==='function'&&offlineView(view),fps=Number(manifest.fps)||30;
    const span=Math.ceil(fps*6);
    return catalog.filter(e=>{const start=eventFrame(e,offline);
      return start<=frameIndex&&(frameIndex<start+span||
        (frames[frameIndex]?.tail&&timeline[e.publication]?.final));});
  }
  function geometry(e){
    const key=`${e.publication}:${e.map}:${e.sourceMap??''}`;if(cache.has(key))return cache.get(key);
    const before=timeline[e.before]?.maps.find(m=>m.id===e.map),after=timeline[e.publication]?.maps.find(m=>m.id===e.map);
    if(!before||!after||before.metric!==after.metric)return null;
    const source=e.sourceMap!=null&&e.sourceMap!==e.map
      ?timeline[e.before]?.maps.find(m=>m.id===e.sourceMap):null;
    const result=diffGeometry(before,after,pointStateAt(e.before,e.map),pointStateAt(e.publication,e.map),
      source,source?pointStateAt(e.before,e.sourceMap):new Map());
    cache.set(key,result);if(cache.size>6)cache.delete(cache.keys().next().value);return result;
  }
  function overlay(view,active){
    const mapId=selectMap(view),m=availableMaps(view).find(x=>x.id===mapId);if(!m)return;
    // Every edge is revealed at its event time, never from final Atlas alone.
    const offline=typeof offlineView==='function'&&offlineView(view);
    const reached=catalog.filter(e=>e.map===mapId&&e.pair&&eventFrame(e,offline)<=frameIndex);
    const pairKey=e=>[...e.pair].sort((a,b)=>a-b).join(':');
    const count=new Set(reached.map(pairKey)).size,total=new Set(catalog.filter(e=>e.map===mapId&&e.pair).map(pairKey)).size;
    const status=view.mode==='overview'?$('overview-status'):$('local-status');
    if(view.mode==='overview')status.textContent=status.textContent.replace('（90%）','（主体 95%）').replace('主体 90% 自动适配','BEV · 主体 95% 自动适配').replace('初始 Ego 基底 · 固定观察朝向','初始 Ego 上方向 · 最适配 BEV');
    status.textContent=status.textContent.replace(/回环边 \d+(?:\/\d+)?(?:（后续回环尚未生效）)?/,`回环边 ${count}/${total}${count<total?'（后续回环尚未生效）':''}`);
    const kfs=new Map((m.keyframes||[]).map(k=>[k[0],k]));
    for(const e of reached){const [a,b]=e.pair.map(id=>kfs.get(id));if(a&&b)line(view,a[2].slice(0,3),b[2].slice(0,3),'#b84955',.9,3);}
    const event=focusEvent(active,mapId);if(!event)return;
    if(!event.correction){const c=view.ctx;c.save();c.fillStyle='#fff9ed';c.fillRect(8,8,Math.min(480,view.width-16),48);c.fillStyle='#694920';c.font='bold 13px system-ui';c.fillText(`${clockLabel(offline&&event.candidateTime!=null?event.candidateTime:event.time)} · ${event.title}`,17,26);c.font='12px system-ui';c.fillText(event.status==='retrospective'?'此旧包未记录独立修正快照；收尾展示综合结果':event.detail,17,44);c.restore();return;}
    const g=geometry(event);if(!g){
      // A valid event can lack a comparable before-map (e.g. metricization).
      // Report the commit, but never invent a geometry-change animation.
      const c=view.ctx;c.save();c.fillStyle='#fff9ed';c.fillRect(8,8,Math.min(560,view.width-16),48);
      c.fillStyle='#694920';c.font='bold 13px system-ui';c.fillText(`${clockLabel(offline&&event.candidateTime!=null?event.candidateTime:event.time)} · ${eventTitle(event)}`,17,26);
      c.font='12px system-ui';c.fillText('已提交；缺少同坐标系前后快照，不绘制虚构位移',17,44);c.restore();return;
    }
    const c=view.ctx;c.save();c.globalAlpha=1;c.fillStyle='rgba(210,128,24,.9)';
    const live=mapPoints(view);
    for(const id of g.highlightedPoints){const p=live.get(mapId+':'+id);if(!p)continue;const xyz=p.slice(1);if(view.mode==='overview'&&view.bevFit&&!view.bevPointKeys.has(xyz.join(',')))continue;const [x,y]=project(view,xyz);if(x>=0&&x<view.width&&y>=0&&y<view.height)c.fillRect(x-1.5,y-1.5,3,3);}
    // Keep the established ORB green-current-point convention over highlights.
    c.fillStyle='#00b84a';for(const p of mapPointEntries(m,true,view)){const[x,y]=project(view,p);if(x>=0&&x<view.width&&y>=0&&y<view.height)c.fillRect(x-1,y-1,3.8,3.8);}
    const ordered=[...(m.keyframes||[])].sort((a,b)=>a[1]-b[1]);
    if($('kfs').checked)for(let i=1;i<ordered.length;i++){
      const a=ordered[i-1],b=ordered[i];
      // Do not bridge two formerly disconnected trajectories at the map seam.
      if(g.importedKF.has(a[0])!==g.importedKF.has(b[0]))continue;
      if(typeof sameTrajectoryRun==='function'&&!sameTrajectoryRun(a,b))continue;
      if(g.highlightedKF.has(a[0])||g.highlightedKF.has(b[0]))
        line(view,a[2].slice(0,3),b[2].slice(0,3),'#a338a6',1,4);
    }
    // Process view: red ghost of actual pre-publication geometry. Final Atlas
    // uses affected IDs at final positions, not geometry from an older gauge.
    if(!offlineView(view)&&$('kfs').checked){const before=[...(g.beforeMap.keyframes||[])].sort((a,b)=>a[1]-b[1]);
      c.setLineDash([5,4]);for(let i=1;i<before.length;i++)if((typeof sameTrajectoryRun!=='function'||sameTrajectoryRun(before[i-1],before[i]))&&(g.changedKF.has(before[i-1][0])||g.changedKF.has(before[i][0])))
        line(view,before[i-1][2].slice(0,3),before[i][2].slice(0,3),'#a65f5b',.7,2);c.setLineDash([]);}
    c.globalAlpha=.97;c.fillStyle='#fff9ed';c.fillRect(8,8,Math.min(560,view.width-16),47);c.fillStyle='#694920';c.font='bold 13px system-ui';
    c.fillText(`${clockLabel(offline&&event.candidateTime!=null?event.candidateTime:event.time)} · ${eventTitle(event)}`,17,26);c.font='12px system-ui';
    c.fillText(`紫红轨迹：${g.highlightedKF.size} KF · 橙色点：${g.highlightedPoints.size}${g.importedKF.size?' · 含并入 '+g.importedKF.size+' KF':''}${offlineView(view)?'（最终坐标）':' · 红虚线：原目标图'}`,17,44);c.restore();
  }
  const baseDraw=draw;
  draw=function(){
    if(!ensureIndex()){baseDraw();return;}
    // Filter final-map visual edges for the base renderer without mutating
    // timeline objects. Restore selectors even if an unrelated draw fails.
    const baseMaps=availableMaps;
    availableMaps=function(view=overview){return baseMaps(view).map(m=>({...m,loops:[]}));};
    try{baseDraw();}finally{availableMaps=baseMaps;}
    for(const view of views)overlay(view,activeEvents(view));
  };
})();
