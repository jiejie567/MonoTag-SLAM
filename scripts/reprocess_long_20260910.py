"""Latest local production pipeline, six distinct >60s videos, original calibrations."""
import collections
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/long_latest_20260910'
SOURCES = [
    ('odin_0909', 'output/long_latest_20260909/odin_0909/actions.meta.json'),
    ('cross_room_0903', 'output/long_latest_20260909/cross_room_0903/actions.meta.json'),
    ('uvc90_0904', 'output/long_latest_20260909/uvc90_0904/actions.meta.json'),
    ('uvc90_0907', 'output/long_latest_20260909/uvc90_0907/actions.meta.json'),
    ('uvc90_0908_rot180', 'output/long_latest_20260909/uvc90_0908_rot180/actions.meta.json'),
    ('uvc90_0909_newlens', 'output/wrist_recalibrated_full_20260910/actions.meta.json'),
]
FLAGS = dict(FLOW_RECOVERY=1, OFFLINE_LOOP_SEARCH=1, INCREMENTAL_LOOP_SEARCH=1,
             MARKER_SIM3_LOOP=1, PARALLEL_DESCRIPTORS=1, PREFETCH_IMAGES=1,
             TEMPORAL_FLOW=0, DYNAMIC_GEOMETRY=0)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def publish(records):
    save(OUT/'results.json', records)
    rows = []
    for r in records:
        name = html.escape(r['name'])
        link = f'<a href="{r["name"]}/actions_replay/index.html">{name}</a>' if r['status']=='completed' else name
        rows.append(f'<tr><td>{link}</td><td>{r["seconds"]:.1f} s</td><td>{r["fps"]:.3f}</td>'
                    f'<td>{html.escape(Path(r["calibration"]).name)}</td><td>{r["status"]}</td>'
                    f'<td>{r.get("wall_seconds",0)/60:.1f} min</td></tr>')
    text = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta http-equiv="refresh" content="30">'
            '<title>长序列 · 最新算法重跑</title><style>body{font:16px system-ui;max-width:1250px;margin:40px auto;padding:20px;background:#fafafa}'
            'table{border-collapse:collapse;width:100%;background:white}td,th{text-align:left;padding:12px;border-bottom:1px solid #ddd}'
            'a{color:#3159a6}</style><h1>最新本地算法 · 超过60秒的序列</h1>'
            '<p>2026-09-10：每段使用对应内参与腕带布局；新跑检测、原生SLAM、双腕标签及完整回放，不跑手指关节。'
            '旧结果保留。≤0.5秒腕带缺口只在显示中连接，不改变标签。</p>'
            '<p>此页每30秒更新。completed 才提供回放链接；尺度重锚定、回环和拒绝原因见各序列结果。</p>'
            '<table><tr><th>序列</th><th>时长</th><th>输入FPS</th><th>相机内参</th><th>状态</th><th>处理耗时</th></tr>'
            + ''.join(rows)+'</table><p>Odin odometry仅用于运行后的参考评估，不参与SLAM。</p></html>')
    (OUT/'index.html.tmp').write_text(text)
    (OUT/'index.html.tmp').replace(OUT/'index.html')


def main():
    OUT.mkdir(exist_ok=False)
    records = []
    for name, metadata_path in SOURCES:
        meta = json.loads((ROOT/metadata_path).read_text())
        video, calibration = Path(meta['video']), Path(meta['calibration'])
        camera = json.loads(calibration.read_text())
        capture = cv2.VideoCapture(str(video))
        fps, frames = capture.get(cv2.CAP_PROP_FPS), int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        size = [int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))]
        capture.release()
        if (fps<=0 or frames/fps<=60 or frames!=meta['frames'] or abs(fps-meta['fps'])>1e-6
                or size!=meta['image_size'] or size!=camera['image_size']):
            raise ValueError(f'Video/calibration metadata mismatch: {name}')
        K=np.asarray(camera['camera_matrix']); D=np.asarray(camera['dist_coeffs'])
        if not np.isfinite(K).all() or not np.isfinite(D).all() or K[0,0]<=0 or K[1,1]<=0:
            raise ValueError(f'Invalid intrinsics: {name}')
        files = [calibration, *map(Path,meta['bands'])]
        auto = meta['auto_marker_map']
        directory = OUT/name
        command = [sys.executable, 'export_action_labels.py', str(video), '--calib', str(calibration),
                   '--head-slam', '--slam-init', 'auto', '--auto-marker-map',
                   '--static-marker-ids', ','.join(map(str,auto['static_marker_ids'])),
                   '--static-marker-size-mm', str(auto['marker_size_mm']),
                   '--no-hand-joints', '--no-slam-dynamic-filter', '--slam-replay',
                   '--output', str(directory/'actions.jsonl')]
        for band in meta['bands']:
            command += ['--band',band]
        records.append(dict(name=name, metadata_source=str(ROOT/metadata_path), video=str(video),
            calibration=str(calibration), bands=meta['bands'], seconds=frames/fps, fps=fps, frames=frames,
            image_size=size, intrinsics=K.tolist(), distortion=D.tolist(),
            input_stat=dict(size=video.stat().st_size,mtime_ns=video.stat().st_mtime_ns),
            calibration_and_layout_sha256={str(p):digest(p) for p in files},
            command=command, environment=FLAGS, status='queued', observation_cache_reused=False))
    files = sorted((ROOT/'aruco_track').glob('*.py')) + [ROOT/'aruco_track/slam_replay.html', ROOT/'export_action_labels.py']
    native = ROOT/'third_party/ORB_SLAM3'
    files += sorted((native/'src').glob('*.cc')) + sorted((native/'include').glob('*.h'))
    files += [native/'Examples/Monocular/mono_tum_headless',native/'lib/libORB_SLAM3.dylib']
    versions = {str(p):digest(p) for p in files}
    save(OUT/'version_and_scope.json',dict(files_sha256=versions,
        exclusions={'calibration_video':'band_layout_calibration_R_20260910_085155.avi is not an operation sequence',
                    'duplicate_orientation':'UVC90_20260908_211715.mp4 represented once by its lossless rotated copy and transformed calibration'},
        no_new_algorithm_trials=True, native_rebuilt_before_start=True))
    publish(records)
    env = {k:v for k,v in os.environ.items() if not k.startswith('ORB_SLAM3_')}
    env.update({'ORB_SLAM3_'+k:str(v) for k,v in FLAGS.items()})
    env['PYTHONUNBUFFERED']='1'
    for record in records:
        checked = versions | record['calibration_and_layout_sha256']
        changed = [path for path,sha in checked.items() if digest(path)!=sha]
        video=Path(record['video'])
        if dict(size=video.stat().st_size,mtime_ns=video.stat().st_mtime_ns)!=record['input_stat']:
            changed.append(str(video))
        if changed:
            record.update(status='blocked_input_changed',changed=changed)
            publish(records)
            raise RuntimeError('Inputs/code changed during batch; do not mix versions')
        directory=OUT/record['name'];directory.mkdir()
        start=time.monotonic();record.update(status='running',started_at=time.time());publish(records)
        print('START',record['name'],flush=True)
        with (directory/'run.log').open('w') as log:
            code=subprocess.call(record['command'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        record.update(returncode=code,wall_seconds=time.monotonic()-start,status='failed' if code else 'completed')
        if code==0:
            data=json.loads((directory/'actions.meta.json').read_text())
            record['camera_sources']=data['camera_world_sources']
            events=data.get('head_slam',{}).get('marker_graph_events',[])
            record['marker_events']=dict(collections.Counter(e['type']+':'+e['status'] for e in events))
            record['scale_events']=[e for e in events if e['type']=='scale_reanchor']
            with (directory/'verification.log').open('w') as log:
                record['verification_returncode']=subprocess.call([sys.executable,'verify_slam_replay.py',str(directory/'actions_replay')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if record['name']=='odin_0909':
                with (directory/'evaluation.log').open('w') as log:
                    record['evaluation_returncode']=subprocess.call([sys.executable,'scripts/evaluate_odin_raw_20260909.py',
                        '--output-dir',str(directory)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        save(directory/'run.json',record);publish(records)
        print('DONE',record['name'],record['status'],round(record['wall_seconds'],1),flush=True)
    print('BATCH FINISHED',flush=True)


if __name__=='__main__':
    main()
