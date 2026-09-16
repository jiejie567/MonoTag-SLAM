#!/usr/bin/env python3
"""Build a self-contained Lieflat-style Odin accuracy explorer."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("accuracy", type=Path)
    parser.add_argument("samples", type=Path)
    parser.add_argument("actions", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--video", default="../odin_actions_replay/process.mp4")
    args = parser.parse_args()

    accuracy = json.loads(args.accuracy.read_text())
    samples = []
    for row in csv.DictReader(args.samples.open()):
        samples.append({
            "frame": int(row["frame"]),
            "t": float(row["video_time_s"]),
            "source": row["source"],
            "confidence": float(row["confidence"]),
            "estimate": [float(row[f"estimate_{axis}_m"]) for axis in "xyz"],
            "reference": [float(row[f"reference_{axis}_m"]) for axis in "xyz"],
            "translationMm": 1000.0 * float(row["translation_error_m"]),
            "rotationDeg": float(row["rotation_error_deg"]),
            "heldout": bool(int(row["heldout"])),
        })
    actions = [json.loads(line) for line in args.actions.open()]
    states = [{
        "frame": int(row["frame"]),
        "t": float(row["timestamp_s"]),
        "source": row.get("camera_world_source", "invalid"),
        "accepted": row.get("accepted_marker_ids", []),
        "mapPoints": int(row.get("head_slam_map_points", 0)),
        "keyframes": int(row.get("head_slam_keyframes", 0)),
    } for row in actions]
    payload = json.dumps({
        "accuracy": accuracy,
        "samples": samples,
        "states": states,
        "video": args.video,
    }, separators=(",", ":"))

    # Derived from Lieflat Basics F2 (hairline line), F8 (plumb scatter),
    # and Lupi L11 (event lineage), using the PORCELAIN color system.
    template = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Odin × Marker–ORB 精度交互结果</title>
<style>
:root{--paper:#F7F2EB;--ink:#081F5C;--royal:#334EAC;--mid:#7096D1;--pale:#BAD6EB;--wash:#D0E3FF;--muted:rgba(8,31,92,.64);--faint:rgba(8,31,92,.30);--grid:rgba(8,31,92,.15);--white:#fffdfa}
*{box-sizing:border-box} body{margin:0;background:var(--paper);color:var(--ink);font-family:Inter,"Noto Sans SC","PingFang SC",system-ui,sans-serif;-webkit-font-smoothing:antialiased}
main{width:min(1180px,calc(100% - 32px));margin:0 auto;padding:38px 0 60px}
.eyebrow{font-size:11px;font-weight:800;letter-spacing:.17em;text-transform:uppercase;color:var(--royal)}
h1{font-size:clamp(27px,4vw,48px);line-height:1.04;letter-spacing:-.045em;margin:9px 0 12px;max-width:900px}
.dek{font-size:14px;line-height:1.75;color:var(--muted);max-width:880px;margin:0}
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--grid);border:1px solid var(--grid);margin:28px 0 18px}
.metric{background:var(--paper);padding:18px}.metric strong{display:block;font-size:29px;letter-spacing:-.05em}.metric span{display:block;font-size:11px;color:var(--muted);line-height:1.45;margin-top:4px}.metric small{font-size:10px;color:var(--faint)}
.insight{border-left:2px solid var(--royal);padding:2px 0 2px 14px;margin:18px 0 28px;max-width:950px;font-size:13px;line-height:1.7}
.layout{display:grid;grid-template-columns:minmax(0,1.28fr) minmax(300px,.72fr);gap:18px;align-items:start}
.panel{border-top:1px solid var(--ink);padding-top:12px;margin-top:12px}.panel h2{font-size:15px;margin:0 0 3px;letter-spacing:-.01em}.sub{font-size:11px;color:var(--muted);line-height:1.55;margin-bottom:10px}
video{display:block;width:100%;background:#071332;aspect-ratio:16/9;object-fit:contain}.now{display:grid;grid-template-columns:auto 1fr auto;gap:10px;align-items:center;margin-top:9px;font-size:11px;color:var(--muted)}
input[type=range]{width:100%;accent-color:var(--royal)}button{border:1px solid var(--faint);background:transparent;color:var(--ink);font:inherit;padding:6px 10px;cursor:pointer}button.active{background:var(--ink);color:var(--paper)}
.controls{display:flex;gap:6px;justify-content:flex-end;margin:-30px 0 7px}.chart{width:100%;display:block;overflow:visible}.chart text{font-family:inherit}.tooltip{position:fixed;pointer-events:none;z-index:9;background:var(--ink);color:var(--paper);padding:8px 10px;font-size:11px;line-height:1.45;box-shadow:0 7px 24px rgba(8,31,92,.18);opacity:0;transform:translate(10px,-12px);max-width:220px}.tooltip.show{opacity:1}.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:10px;color:var(--muted);margin-top:5px}.key:before{content:"";display:inline-block;width:14px;height:2px;background:var(--c);vertical-align:middle;margin-right:5px}.key.dot:before{height:7px;width:7px;border-radius:50%}
.stategrid{display:grid;grid-template-columns:repeat(4,1fr);gap:7px;margin-top:10px}.state{border:1px solid var(--grid);padding:9px}.state b{display:block;font-size:16px}.state span{font-size:9px;color:var(--muted)}
.methods{margin-top:25px;display:grid;grid-template-columns:1fr 1fr;gap:24px;border-top:1px solid var(--ink);padding-top:16px}.methods h3{font-size:12px;margin:0 0 7px}.methods p,.methods li{font-size:11px;line-height:1.65;color:var(--muted)}.methods ul{padding-left:17px;margin:0}.foot{font-size:10px;line-height:1.65;color:var(--faint);border-top:1px solid var(--grid);padding-top:12px;margin-top:22px}
@media(max-width:820px){.metrics{grid-template-columns:1fr 1fr}.layout{grid-template-columns:1fr}.methods{grid-template-columns:1fr}.controls{margin:0 0 7px}.stategrid{grid-template-columns:1fr 1fr}}
</style>
</head>
<body><main>
<div class="eyebrow">Held-out trajectory agreement · 2026-09-04</div>
<h1>Marker–ORB 与 Odin 达到毫米级一致，主要短板是初始化前覆盖率</h1>
<p class="dek">同一 ROS 2 包中的 331 帧原始图像经过当前正式 Marker–ORB 后端处理。相机—IMU 刚体外参仅在前 55% 有效重叠段估计，以下主要数字来自后 45% 留出段；全程只允许米制 SE(3)，未使用 Sim(3) 缩放。</p>
<section class="metrics" id="metrics"></section>
<p class="insight" id="insight"></p>

<section class="layout">
  <div>
    <div class="panel"><h2>同步诊断视频</h2><div class="sub">拖动视频或下方滑块；误差曲线、轨迹点和定位状态使用同一个视频时钟。</div>
      <video id="video" controls preload="metadata"></video>
      <div class="now"><button id="play">播放</button><input id="scrub" type="range" min="0" max="22.82" step="0.01" value="0"><span id="clock">0.00 s</span></div>
    </div>
    <div class="panel"><h2>逐帧轨迹差异</h2><div class="sub">F2 Hairline Line：一枚点代表一个真实有效帧。浅底为外参标定段，深底为留出评估段；悬停查看，点击固定。</div>
      <div class="controls"><button class="mode active" data-mode="translationMm">平移 mm</button><button class="mode" data-mode="rotationDeg">旋转 °</button></div>
      <svg id="errorChart" class="chart" viewBox="0 0 760 285" role="img"></svg>
      <div class="legend"><span class="key dot" style="--c:#7096D1">marker</span><span class="key dot" style="--c:#334EAC">marker + SLAM</span><span class="key dot" style="--c:#081F5C">head SLAM</span><span class="key" style="--c:#334EAC">留出段起点</span></div>
    </div>
  </div>
  <div>
    <div class="panel"><h2>顶视轨迹对照</h2><div class="sub">F8 Plumb Scatter 的轨迹化：实线为 Marker–ORB，相同时间的虚线为 Odin 相机参考轨迹。坐标单位为米，未做尺度缩放。</div>
      <svg id="trajectory" class="chart" viewBox="0 0 430 410" role="img"></svg>
      <div class="legend"><span class="key" style="--c:#081F5C">Marker–ORB</span><span class="key" style="--c:#7096D1">Odin（经相机外参）</span></div>
    </div>
    <div class="panel"><h2>真实定位状态</h2><div class="sub">L11 Trend Lineage：状态来自逐帧正式输出，不由最终地图反推。</div>
      <svg id="states" class="chart" viewBox="0 0 430 92" role="img"></svg>
      <div class="stategrid" id="stateStats"></div>
    </div>
  </div>
</section>

<section class="methods">
 <div><h3>如何读这些数字</h3><ul><li>ATE 衡量对齐后绝对轨迹差异；RPE 衡量相隔 1 秒的局部运动差异。</li><li>外参和坐标系只在前 114 个有效帧拟合；后 93 帧完全留出。</li><li>时间偏移受限于 ±150 ms，结果接近 0，符合图像和里程计共享 ROS 时钟。</li></ul></div>
 <div><h3>证据边界</h3><ul><li>Odin 发布的是 <code>imu</code> 位姿，包内没有 camera→imu 外参，故本页估计一个固定 SE(3) 外参。</li><li>Odin 自报位置标准差约 7 mm；因此 5.55 mm 应解释为两系统一致度，不是更高精度的绝对真值。</li><li>开头未见可靠锚点/未建图时不输出世界位姿，这是正确的无效状态而非零误差。</li></ul></div>
</section>
<div class="foot">数据源：odin_20260904_120508 ROS 2 MCAP；图像 1600×1296，14.46 Hz，22.89 s。算法：当前 official ORB-SLAM3 monocular + independent metric markers（IDs 28–31）+ marker corner global BA。交互样式基于 Lieflat PORCELAIN 系统；全部点均为真实逐帧记录。</div>
</main><div class="tooltip" id="tip"></div>
<script>const DATA=__DATA__;
const A=DATA.accuracy,S=DATA.samples,ST=DATA.states,$=s=>document.querySelector(s),NS='http://www.w3.org/2000/svg';
const C={invalid:'#BAD6EB',marker:'#7096D1','marker+slam':'#334EAC','head-slam':'#081F5C'};
const fmt=(v,n=2)=>Number(v).toFixed(n), mm=v=>fmt(v*1000,2);
const held=A.fused_heldout, marker=A.marker_only_heldout, cov=A.reference.reported_median_position_sigma_m_xyz;
const improve=(marker.ate_translation_m_rmse-held.ate_translation_m_rmse)/marker.ate_translation_m_rmse*100;
$('#metrics').innerHTML=[
 ['5.55 mm','平移 ATE RMSE','后 93 个留出帧'],
 [fmt(held.ate_rotation_deg_rmse,2)+'°','旋转 ATE RMSE','留出段'],
 [mm(held.rpe_1s_translation_m_rmse)+' mm','1 秒平移 RPE','留出段'],
 [fmt(A.coverage.overall_fraction*100,1)+'%','全视频世界位姿覆盖率','首次有效后 '+fmt(A.coverage.valid_after_first_fraction*100,1)+'%']
].map(x=>`<div class="metric"><strong>${x[0]}</strong><span>${x[1]}</span><small>${x[2]}</small></div>`).join('');
$('#insight').innerHTML=`融合相对 marker-only 将留出段平移 ATE 从 <b>${mm(marker.ate_translation_m_rmse)} mm</b> 降至 <b>${mm(held.ate_translation_m_rmse)} mm</b>（${fmt(improve,1)}%）；1 秒尺度比例中位数为 <b>${fmt(held.scale_ratio_1s_median,3)}</b>。但 Odin 自报位置 σ 的三轴中位数为 ${cov.map(x=>mm(x)).join(' / ')} mm，当前应称为毫米级一致性。`;

function el(svg,name,attrs,text){const n=document.createElementNS(NS,name);for(const[k,v]of Object.entries(attrs||{}))n.setAttribute(k,v);if(text!=null)n.textContent=text;svg.appendChild(n);return n}
const margin={l:50,r:18,t:22,b:34};let mode='translationMm',pinned=null,currentTime=0;
function nearest(t){let best=0;for(let i=1;i<S.length;i++)if(Math.abs(S[i].t-t)<Math.abs(S[best].t-t))best=i;return best}
function showTip(event,d,pin=false){const tip=$('#tip');tip.innerHTML=`<b>frame ${d.frame} · ${fmt(d.t)} s</b><br>${d.source}<br>平移 ${fmt(d.translationMm)} mm · 旋转 ${fmt(d.rotationDeg,3)}°<br>置信度 ${fmt(d.confidence,3)}${d.heldout?' · 留出段':' · 标定段'}`;tip.style.left=event.clientX+'px';tip.style.top=event.clientY+'px';tip.classList.add('show');if(pin)pinned=d.frame}
function hideTip(){if(pinned==null)$('#tip').classList.remove('show')}
function errorChart(){const svg=$('#errorChart');svg.innerHTML='';const W=760,H=285,x0=margin.l,x1=W-margin.r,y0=H-margin.b,y1=margin.t,tmax=ST.at(-1).t;const vals=S.map(d=>d[mode]);const ymax=Math.max(mode==='translationMm'?10:.5,Math.ceil(Math.max(...vals)*1.1*(mode==='translationMm'?1:10))/(mode==='translationMm'?1:10));const X=t=>x0+t/tmax*(x1-x0),Y=v=>y0-v/ymax*(y0-y1);const split=S.find(d=>d.heldout).t;
 el(svg,'rect',{x:X(split),y:y1,width:x1-X(split),height:y0-y1,fill:'#D0E3FF',opacity:.38});
 for(let k=0;k<=4;k++){let v=ymax*k/4;el(svg,'line',{x1:x0,y1:Y(v),x2:x1,y2:Y(v),stroke:'rgba(8,31,92,.15)','stroke-width':.7});el(svg,'text',{x:x0-8,y:Y(v)+3,'text-anchor':'end','font-size':9,fill:'rgba(8,31,92,.60)'},fmt(v,mode==='translationMm'?0:2))}
 for(let t=0;t<=tmax;t+=5){el(svg,'line',{x1:X(t),y1:y0,x2:X(t),y2:y0+5,stroke:'#081F5C','stroke-width':.7});el(svg,'text',{x:X(t),y:y0+18,'text-anchor':'middle','font-size':9,fill:'rgba(8,31,92,.60)'},t+' s')}
 el(svg,'path',{d:S.map((d,i)=>(i?'L':'M')+X(d.t)+' '+Y(d[mode])).join(' '),fill:'none',stroke:'#334EAC','stroke-width':1.15});
 S.forEach(d=>{const c=el(svg,'circle',{cx:X(d.t),cy:Y(d[mode]),r:2.4,fill:C[d.source]||C.invalid,'data-frame':d.frame});c.addEventListener('pointerenter',e=>showTip(e,d));c.addEventListener('pointermove',e=>{if(pinned==null)showTip(e,d)});c.addEventListener('pointerleave',hideTip);c.addEventListener('click',e=>{pinned=pinned===d.frame?null:d.frame;showTip(e,d,pinned!=null)})});
 el(svg,'line',{id:'errorCursor',x1:X(currentTime),y1:y1,x2:X(currentTime),y2:y0,stroke:'#081F5C','stroke-width':1,'stroke-dasharray':'3 3'});el(svg,'text',{x:X(split)+5,y:y1+11,'font-size':8,'font-weight':700,fill:'#334EAC'},'HELD-OUT');
}
function trajectory(){const svg=$('#trajectory');svg.innerHTML='';const W=430,H=410,p=35,all=S.flatMap(d=>[d.estimate,d.reference]);const xs=all.map(v=>v[0]),ys=all.map(v=>v[1]);let xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);const span=Math.max(xmax-xmin,ymax-ymin)*1.12,cx=(xmin+xmax)/2,cy=(ymin+ymax)/2;xmin=cx-span/2;xmax=cx+span/2;ymin=cy-span/2;ymax=cy+span/2;const X=x=>p+(x-xmin)/(xmax-xmin)*(W-2*p),Y=y=>H-p-(y-ymin)/(ymax-ymin)*(H-2*p);
 for(let k=0;k<=4;k++){const x=xmin+(xmax-xmin)*k/4,y=ymin+(ymax-ymin)*k/4;el(svg,'line',{x1:X(x),y1:p,x2:X(x),y2:H-p,stroke:'rgba(8,31,92,.13)','stroke-width':.7});el(svg,'line',{x1:p,y1:Y(y),x2:W-p,y2:Y(y),stroke:'rgba(8,31,92,.13)','stroke-width':.7});el(svg,'text',{x:X(x),y:H-12,'text-anchor':'middle','font-size':8,fill:'rgba(8,31,92,.55)'},fmt(x,2));el(svg,'text',{x:5,y:Y(y)+3,'font-size':8,fill:'rgba(8,31,92,.55)'},fmt(y,2))}
 const path=(key)=>S.map((d,i)=>(i?'L':'M')+X(d[key][0])+' '+Y(d[key][1])).join(' ');el(svg,'path',{d:path('reference'),fill:'none',stroke:'#7096D1','stroke-width':1.2,'stroke-dasharray':'4 3'});el(svg,'path',{d:path('estimate'),fill:'none',stroke:'#081F5C','stroke-width':1.5});
 const i=nearest(currentTime),d=S[i];el(svg,'line',{id:'joinNow',x1:X(d.reference[0]),y1:Y(d.reference[1]),x2:X(d.estimate[0]),y2:Y(d.estimate[1]),stroke:'#334EAC','stroke-width':.8});el(svg,'circle',{id:'refNow',cx:X(d.reference[0]),cy:Y(d.reference[1]),r:4,fill:'#F7F2EB',stroke:'#7096D1','stroke-width':2});el(svg,'circle',{id:'estNow',cx:X(d.estimate[0]),cy:Y(d.estimate[1]),r:4,fill:'#081F5C'});svg._map={X,Y};
}
function stateChart(){const svg=$('#states');svg.innerHTML='';const W=430,x0=4,x1=426,tmax=ST.at(-1).t,X=t=>x0+t/tmax*(x1-x0),y=35;let start=0;for(let i=1;i<=ST.length;i++)if(i===ST.length||ST[i].source!==ST[start].source){const a=ST[start],b=ST[i-1];el(svg,'rect',{x:X(a.t),y,width:Math.max(1,X(b.t+1/14.46)-X(a.t)),height:18,fill:C[a.source]||C.invalid});start=i}el(svg,'line',{id:'stateCursor',x1:X(currentTime),y1:18,x2:X(currentTime),y2:68,stroke:'#081F5C','stroke-width':1});for(let t=0;t<=tmax;t+=5)el(svg,'text',{x:X(t),y:78,'text-anchor':'middle','font-size':8,fill:'rgba(8,31,92,.55)'},t+' s');svg._X=X;
 const counts={};ST.forEach(d=>counts[d.source]=(counts[d.source]||0)+1);$('#stateStats').innerHTML=Object.entries(counts).map(([k,v])=>`<div class="state"><b>${v}</b><span>${k}</span></div>`).join('')}
function update(t){currentTime=t;$('#scrub').value=t;$('#clock').textContent=fmt(t)+' s';const ec=$('#errorChart'),x=margin.l+t/ST.at(-1).t*(760-margin.l-margin.r);const line=ec.querySelector('#errorCursor');if(line){line.setAttribute('x1',x);line.setAttribute('x2',x)}const st=$('#states'),sx=st._X?st._X(t):0;const sl=st.querySelector('#stateCursor');if(sl){sl.setAttribute('x1',sx);sl.setAttribute('x2',sx)}const tr=$('#trajectory'),i=nearest(t),d=S[i];if(tr._map){const {X,Y}=tr._map,ref=tr.querySelector('#refNow'),est=tr.querySelector('#estNow'),join=tr.querySelector('#joinNow');ref.setAttribute('cx',X(d.reference[0]));ref.setAttribute('cy',Y(d.reference[1]));est.setAttribute('cx',X(d.estimate[0]));est.setAttribute('cy',Y(d.estimate[1]));join.setAttribute('x1',X(d.reference[0]));join.setAttribute('y1',Y(d.reference[1]));join.setAttribute('x2',X(d.estimate[0]));join.setAttribute('y2',Y(d.estimate[1]))}}
const video=$('#video');video.src=DATA.video;video.addEventListener('timeupdate',()=>update(video.currentTime));video.addEventListener('loadedmetadata',()=>{$('#scrub').max=Math.min(video.duration,ST.at(-1).t)});$('#scrub').addEventListener('input',e=>{video.currentTime=+e.target.value;update(+e.target.value)});$('#play').onclick=()=>video.paused?video.play():video.pause();video.addEventListener('play',()=>$('#play').textContent='暂停');video.addEventListener('pause',()=>$('#play').textContent='播放');document.querySelectorAll('.mode').forEach(b=>b.onclick=()=>{document.querySelectorAll('.mode').forEach(x=>x.classList.remove('active'));b.classList.add('active');mode=b.dataset.mode;errorChart()});document.body.addEventListener('click',e=>{if(!e.target.closest('circle')){pinned=null;hideTip()}});
errorChart();trajectory();stateChart();update(0);
</script></body></html>'''
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(template.replace("__DATA__", payload))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
