"""Re-run five previously processed long recordings without replacing results."""
import collections
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/long_latest_20260909'
SOURCES = [
    ('uvc90_0904', 'output/current_best_global_shutter_20260905/uvc90_20260904_201356/actions.meta.json'),
    ('cross_room_0903', 'output/current_best_global_shutter_20260905/cross_room_20260903_183009/actions.meta.json'),
    ('uvc90_0907', 'output/samsung_uvc90_20260907_184339_scalegate_20260908/actions.meta.json'),
    ('uvc90_0908_rot180', 'output/samsung_uvc90_211715_rot180_latest_20260908/actions.meta.json'),
    ('odin_0909', 'output/odin_order_determinism_20260909/run_b/actions.meta.json'),
]


def main():
    OUT.mkdir(exist_ok=False)
    env = os.environ.copy()
    flags = dict(FLOW_RECOVERY=1, OFFLINE_LOOP_SEARCH=1,
                 INCREMENTAL_LOOP_SEARCH=1, MARKER_SIM3_LOOP=1,
                 PARALLEL_DESCRIPTORS=1, PREFETCH_IMAGES=1,
                 TEMPORAL_FLOW=0, DYNAMIC_GEOMETRY=0)
    env.update({'ORB_SLAM3_'+k: str(v) for k, v in flags.items()})
    env['PYTHONUNBUFFERED'] = '1'
    results = []
    for name, metadata in SOURCES:
        meta = json.loads((ROOT/metadata).read_text())
        directory = OUT/name
        directory.mkdir()
        cmd = [sys.executable, 'export_action_labels.py', meta['video'],
               '--calib', meta['calibration'], '--head-slam', '--slam-init', 'auto',
               '--auto-marker-map', '--static-marker-ids', '20-49',
               '--static-marker-size-mm', str(meta['auto_marker_map']['marker_size_mm']),
               '--no-hand-joints', '--no-slam-dynamic-filter',
               '--slam-debug-video', '--slam-replay', '--output', str(directory/'actions.jsonl')]
        for band in meta['bands']:
            cmd += ['--band', band]
        record = dict(name=name, video=meta['video'], baseline=metadata,
                      command=cmd, environment=flags, status='running')
        (directory/'run.json').write_text(json.dumps(record, indent=2))
        print('START', name, flush=True)
        start = time.monotonic()
        with (directory/'run.log').open('w') as log:
            code = subprocess.call(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        record.update(returncode=code, wall_seconds=time.monotonic()-start,
                      status='completed' if code == 0 else 'failed')
        if code == 0:
            data = json.loads((directory/'actions.meta.json').read_text())
            record.update(frames=data['frames'], fps=data['fps'],
                          camera_sources=data['camera_world_sources'])
            events = data['head_slam'].get('marker_graph_events', [])
            record['marker_events'] = dict(collections.Counter(
                e['type']+':'+e['status'] for e in events))
            record['scale_events'] = [e for e in events if e['type']=='scale_reanchor']
            if name == 'odin_0909':
                with (directory/'evaluation.log').open('w') as log:
                    record['evaluation_returncode'] = subprocess.call(
                        [sys.executable, 'scripts/evaluate_odin_raw_20260909.py',
                         '--output-dir', str(directory)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        (directory/'run.json').write_text(json.dumps(record, indent=2))
        results.append(record)
        (OUT/'results.json').write_text(json.dumps(results, indent=2))
        print('DONE', name, record['status'], round(record['wall_seconds'], 1), flush=True)
        rows = ''.join('<li><a href="'+r['name']+'/actions_replay/index.html">'+
                       html.escape(r['name'])+'</a> — '+r['status']+'</li>' for r in results)
        (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Long sequence reruns</title>'
            '<h1>当前算法 · 长序列回放</h1><p>2026-09-09，本地重跑；无手指关节推理。旧结果保留。</p><ul>'+rows+'</ul>')
    print('BATCH FINISHED', flush=True)


if __name__ == '__main__':
    main()
