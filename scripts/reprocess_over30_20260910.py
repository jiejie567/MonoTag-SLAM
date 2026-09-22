"""Isolated production batch; completed and verified packages only are published."""
import functools
import html
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from http.server import ThreadingHTTPServer

import cv2

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/all_over30_latest_20260910'
RUNTIME = OUT / 'runtime'


def save(path, obj):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def publish(records):
    save(OUT / 'results.json', records)
    rows = []
    for r in records:
        name = html.escape(r['name'])
        if r['status'] == 'verified':
            name = f'<a href="{r["name"]}/actions_replay/index.html">{name}</a>'
        rows.append(f'<tr><td>{name}</td><td>{r["seconds"]:.2f} s</td><td>{r["fps"]:.2f}</td>'
                    f'<td>{html.escape(r["status"])}</td><td>{r.get("wall_seconds",0)/60:.1f} min</td></tr>')
    page = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
            '<meta http-equiv="refresh" content="30"><title>超过30秒 · 最新算法</title>'
            '<style>body{font:16px system-ui;margin:32px auto;padding:16px;max-width:1200px;background:#faf9f6;color:#272727}'
            'td,th{padding:12px;text-align:left;border-bottom:1px solid #ddd}table{width:100%;border-collapse:collapse}a{color:#484878}</style>'
            '<h1>超过30秒 · 当前本地算法</h1><p>SLAM、双腕与默认增强手部检测；各序列使用自己的内参及布局。'
            '全局/视频/手部采用离线结果，局部窗口展示建图过程；默认2倍速。</p>'
            '<p>本页30秒刷新。仅 verified 提供回放；其他状态不表示完成。未重新加入154017、标定录像或重复旋转源。</p>'
            '<table><tr><th>交互回放</th><th>原视频时长</th><th>FPS</th><th>状态</th><th>耗时</th></tr>'
            + ''.join(rows) + '</table><p>使用HTTP入口，勿直接打开file://。无效定位不伪装为有效；BA拒绝保留诊断。</p></html>')
    temp = OUT / 'index.html.tmp'
    temp.write_text(page)
    temp.replace(OUT / 'index.html')


def inventory():
    best = {}
    candidates = list(ROOT.glob('output/**/actions.meta.json')) + list(ROOT.glob('recordings/*.meta.json'))
    for path in candidates:
        if OUT in path.parents:
            continue
        try:
            m = json.loads(path.read_text())
            video = Path(m['video']).resolve()
            if '154017' in video.name or 'calibration' in video.name or m['frames']/m['fps'] <= 30:
                continue
            if not video.is_file() or not Path(m['calibration']).is_file():
                continue
            old = best.get(video)
            if old is None or path.stat().st_mtime > old[0]:
                best[video] = (path.stat().st_mtime, path, m)
        except (KeyError, ValueError, OSError, ZeroDivisionError):
            continue
    records = []
    for video, (_, path, m) in best.items():
        c = cv2.VideoCapture(str(video))
        fps, n = c.get(5), int(c.get(7))
        size = [int(c.get(3)), int(c.get(4))]
        c.release()
        assert n == m['frames'] and abs(fps-m['fps']) < .001 and size == m['image_size'], video
        calib = json.loads(Path(m['calibration']).read_text())
        assert size == calib['image_size'], (video, m['calibration'])
        assert all(Path(p).is_file() for p in m['bands'])
        name = video.stem if video.stem != 'rectified_lossless' else 'odin_0909'
        records.append(dict(name=name, video=str(video), metadata_source=str(path), frames=n, fps=fps,
                            seconds=n/fps, calibration=m['calibration'], bands=m['bands'],
                            marker_config=m['auto_marker_map'], status='queued'))
    assert len({r['name'] for r in records}) == len(records)
    return sorted(records, key=lambda r:r['seconds'])


def main():
    OUT.mkdir(exist_ok=False)
    records = inventory()
    if not records:
        raise RuntimeError('No calibrated sequences')
    publish(records)
    spec = importlib.util.spec_from_file_location('snapshot', ROOT/'output/phone_latest4_20260910_1437/run_batch.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.OUT, helper.RUNTIME = OUT, RUNTIME
    helper.prepare()
    # The readonly prefix localizer is required by current offline backtracking.
    rel = Path('third_party/ORB_SLAM3/Examples/Monocular/relocalize_prefix_readonly')
    shutil.copy2(ROOT/rel, RUNTIME/rel)
    for r in records:
        config = RUNTIME/'config'/r['name']
        config.mkdir()
        for key in ['calibration', 'bands']:
            values = [r[key]] if key == 'calibration' else r[key]
            copies = []
            for i, value in enumerate(values):
                target = config/f'{key}_{i}.json'
                shutil.copy2(value, target)
                copies.append(str(target))
            r['frozen_'+key] = copies[0] if key == 'calibration' else copies
    snapshot = json.loads((OUT/'runtime_manifest.json').read_text())
    snapshot['files_sha256'][str(rel)] = helper.digest(RUNTIME/rel)
    save(OUT/'runtime_manifest.json', snapshot)
    sys.path.insert(0, str(RUNTIME))
    from tools.replay_orb_slam import ReplayHandler
    server = ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(ReplayHandler, directory=str(OUT)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    save(OUT/'server.json', dict(url=f'http://127.0.0.1:{server.server_port}/index.html', pid=os.getpid()))
    flags = dict(helper.FLAGS, OFFLINE_MARKER_BOOTSTRAP=1, PREFIX_RELOCALIZATION=1)
    env = {k:v for k,v in os.environ.items() if not k.startswith('ORB_SLAM3_')}
    env.update({'ORB_SLAM3_'+k:str(v) for k,v in flags.items()})
    native = RUNTIME/'third_party/ORB_SLAM3'
    env.update(PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1',
               SLAM_SEQUENCE_CACHE_DIR=str(OUT/'sequence_cache'),
               DYLD_LIBRARY_PATH=':'.join(map(str,[native/'lib', native/'Thirdparty/DBoW2/lib', native/'Thirdparty/g2o/lib',
                   RUNTIME/'third_party/opencv-4.10-install/lib',RUNTIME/'third_party/Pangolin/install/lib'])) )
    publish(records)
    print('URL', json.loads((OUT/'server.json').read_text())['url'], flush=True)
    for r in records:
        d = OUT/r['name']; d.mkdir()
        auto = r['marker_config']
        command = [sys.executable, '-B', str(RUNTIME/'tools/export_action_labels.py'), r['video'], '--calib',r['frozen_calibration'],
                   '--head-slam','--slam-init','auto','--auto-marker-map','--static-marker-ids',','.join(map(str,auto['static_marker_ids'])),
                   '--static-marker-size-mm',str(auto['marker_size_mm']), '--hand-joints','--hand-model',str(RUNTIME/'models/hand_landmarker.task'),
                   '--no-slam-dynamic-filter','--slam-replay','--output',str(d/'actions.jsonl')]
        for band in r['frozen_bands']:
            command += ['--band',band]
        r.update(status='analyzing', command=command, environment=flags)
        publish(records); start = time.monotonic()
        try:
            if shutil.disk_usage(OUT).free < 25*1024**3:
                raise RuntimeError('Less than 25GB free; batch stopped safely')
            with (d/'run.log').open('w') as log:
                subprocess.run(command,cwd=RUNTIME,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            r['status']='verifying';publish(records)
            with (d/'verification.log').open('w') as log:
                subprocess.run([sys.executable,'-B',str(RUNTIME/'tools/verify_slam_replay.py'),str(d/'actions_replay')],
                               cwd=RUNTIME,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            from urllib.request import Request, urlopen
            url=f'http://127.0.0.1:{server.server_port}/{r["name"]}/actions_replay/'
            with urlopen(url+'manifest.json') as response:
                json.load(response)
            with urlopen(Request(url+'process.mp4', headers={'Range':'bytes=1024-2047'})) as response:
                assert response.status == 206 and len(response.read()) == 1024
            r['status']='verified'
        except Exception as error:
            r.update(status='failed', error=str(error))
        r['wall_seconds']=time.monotonic()-start
        save(d/'run.json',r);publish(records)
        print('DONE',r['name'],r['status'],r['wall_seconds'],flush=True)
        if r['status']=='failed':
            print('Stopped before remaining jobs: inspect failure rather than repeat it.',flush=True)
            break
    print('BATCH STOPPED; replay server remains active',flush=True)
    threading.Event().wait()


if __name__ == '__main__':
    main()
