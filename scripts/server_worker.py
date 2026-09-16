"""Private, headless Linux worker using the same exporter and native sources."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from aruco_track.server_pipeline import digest, verify_release

def verify_build():
    build=ROOT/'.local/build.json'
    info=json.loads(build.read_text())
    for path,expected in info['files'].items():
        if digest(path)!=expected:raise ValueError(f'Native binary/resource changed: {path}')
    return info

def arguments(values):
    result=[str(values['video'])]
    for key,value in values.items():
        if key=='video' or value is None: continue
        flag='--'+key.replace('_','-')
        if isinstance(value,bool):
            if value: result.append(flag)
            elif key in {'hand_joints','slam_dynamic_filter','slam_weak_marker_corners','slam_replay'}:
                result.append('--no-'+key.replace('_','-'))
        elif isinstance(value,list):
            # The parser exposes static_marker_ids as a list but its CLI expects CSV.
            if key=='static_marker_ids': result.extend([flag,','.join(map(str,value))])
            else:
                for item in value: result.extend([flag,str(item)])
        else: result.extend([flag,str(value)])
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('request',nargs='?',type=Path);p.add_argument('--verify',type=Path)
    args=p.parse_args()
    if args.verify:
        data=json.loads(args.verify.read_text());verify_release(ROOT,data);verify_build();print(data['release']);return
    req=json.loads(args.request.read_text());job=Path(req['remote_job']);output=job/'output'
    status=dict(status='running',release=req['release'],started=time.time(),job=req['job'])
    def save():
        temp=job/'status.pending.json';temp.write_text(json.dumps(status,indent=2)+'\n');temp.replace(job/'status.json')
    try:
        data=json.loads(Path(req['manifest']).read_text());verify_release(ROOT,data)
        verify_build()
        if data['release']!=req['release']: raise ValueError('Requested release changed')
        output.mkdir(exist_ok=False);save()
        for entry in req['inputs'].values():
            if digest(entry['remote'])!=entry['sha256']:raise ValueError('Job input changed')
        env={k:v for k,v in os.environ.items() if not k.startswith('ORB_SLAM3_')}
        env.update(json.loads((ROOT/'config/production.json').read_text())['environment'])
        native=ROOT/'third_party/ORB_SLAM3';deps=Path(req['dependencies'])
        env.update(PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=req['cuda_device'],
                   SLAM_SEQUENCE_CACHE_DIR='/tmp/monotag-production-slam-cache',
                   LD_LIBRARY_PATH=':'.join(map(str,[native/'lib',native/'Thirdparty/DBoW2/lib',
                      native/'Thirdparty/g2o/lib',deps/'opencv-4.10/lib',deps/'pangolin/lib'])))
        command=[sys.executable,'-B',str(ROOT/'export_action_labels.py'),*arguments(req['arguments'])]
        status.update(command=command,environment={k:v for k,v in env.items() if k.startswith('ORB_SLAM3_')});save()
        print('SERVER: observation analysis / native SLAM / offline optimization / replay',flush=True)
        subprocess.run(command,cwd=ROOT,env=env,check=True)
        replay=output/'actions_replay'
        if req['arguments']['slam_replay']:
            print('SERVER: verifying replay',flush=True)
            subprocess.run([sys.executable,'-B',str(ROOT/'verify_slam_replay.py'),str(replay)],cwd=ROOT,env=env,check=True)
        meta=json.loads((output/'actions.meta.json').read_text())
        with (output/'actions.jsonl').open() as f: count=sum(1 for _ in f)
        if count!=meta['frames']:raise ValueError('Incomplete labels')
        # Regenerable decoded-frame caches live outside this output; preserve final Atlas and diagnostics.
        archive=job/'result.tar.gz'
        with tarfile.open(archive,'w:gz',compresslevel=1,dereference=True) as tar:
            for path in output.iterdir():tar.add(path,arcname=path.name)
        status.update(status='complete',frames=count,ended=time.time(),archive_sha256=digest(archive));save()
        print('SERVER COMPLETE: '+json.dumps(status),flush=True)
    except Exception:
        status.update(status='failed',error=traceback.format_exc(),ended=time.time());save();raise

if __name__=='__main__':main()
