"""Bind separately provisioned, licensed resources to a built source release."""
import argparse
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from aruco_track.server_pipeline import digest

def main():
    p=argparse.ArgumentParser();p.add_argument('resources',type=Path);args=p.parse_args()
    missing=[name for name in ('ffmpeg','ffprobe') if shutil.which(name) is None]
    if missing:
        p.error('Missing required video tools on PATH: '+', '.join(missing)+
                '. Ubuntu/Debian: sudo apt-get install ffmpeg (provides ffmpeg and ffprobe). No resources were modified.')
    cfg=json.loads(args.resources.read_text());tracked={}
    def resource(target,source,expected):
        source=Path(source).resolve()
        if digest(source)!=expected:raise ValueError(f'Resource hash differs: {source}')
        target=ROOT/target;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if digest(target)!=expected:raise ValueError(f'Existing resource differs: {target}')
        else:target.symlink_to(source)
        tracked[str(target)]=expected
    for key,target in [('vocabulary','third_party/ORB_SLAM3/Vocabulary/ORBvoc.txt'),
                       ('mediapipe','models/hand_landmarker.task')]:
        item=cfg[key];resource(target,item['path'],item['sha256'])
    hawor=cfg['hawor'];runtime=ROOT/'models/hawor/runtime.json';runtime.parent.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(hawor,indent=2)+'\n'
    if runtime.exists() and runtime.read_text()!=payload:raise ValueError('Existing HaWoR runtime differs')
    if not runtime.exists():runtime.write_text(payload)
    tracked[str(runtime)]=digest(runtime)
    for rel in ['lib/libORB_SLAM3.so','Thirdparty/DBoW2/lib/libDBoW2.so','Thirdparty/g2o/lib/libg2o.so',
                'Examples/Monocular/mono_tum_headless','Examples/Monocular/relocalize_prefix_readonly']:
        path=ROOT/'third_party/ORB_SLAM3'/rel;tracked[str(path)]=digest(path)
    for path,expected in cfg.get('extra_resources',{}).items():
        if digest(path)!=expected:raise ValueError(f'Private resource mismatch: {path}')
        tracked[path]=expected
    build=ROOT/'.local/build.json';build.parent.mkdir(exist_ok=True)
    build.write_text(json.dumps(dict(files=tracked,resources=cfg,python=sys.version),indent=2)+'\n')
    print(build)

if __name__=='__main__':main()
