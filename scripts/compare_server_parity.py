"""Compare matched source-frame outputs; never align away metric scale differences."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

def compare(a,b):
    rows=[];poses={};wrists={};ids={};counts={}
    for label,path in [('local',a),('server',b)]:
        with Path(path).open() as stream:data=[json.loads(x) for x in stream]
        rows.append(data);poses[label]={};wrists[label]={};ids[label]=set()
        counts[label]=dict(frames=len(data),sources=dict(Counter(r.get('camera_world_source','invalid') for r in data)))
        for r in data:
            p=r.get('camera_world_pose_fused')
            if p and r.get('camera_world_source')!='invalid':
                poses[label][r['frame']]=p;ids[label].add(r.get('camera_submap_id'))
            for name,hand in r.get('hands',{}).items():
                p=hand.get('wrist_world_graph')
                if p:wrists[label].setdefault(name,{})[r['frame']]=p
        counts[label].update(valid_camera=len(poses[label]),maps=sorted(map(str,ids[label])))
    if [(r['frame'],r['timestamp_s']) for r in rows[0]]!=[(r['frame'],r['timestamp_s']) for r in rows[1]]:
        raise ValueError('Source frames or timestamps differ')
    def statistics(x):
        return dict(count=len(x),rms=float(np.sqrt(np.mean(np.square(x)))),
                    median=float(np.median(x)),p95=float(np.percentile(x,95)),max=float(max(x))) if x else None
    common=sorted(poses['local'].keys()&poses['server'].keys());distance=[];angle=[]
    for f in common:
        x,y=poses['local'][f],poses['server'][f]
        distance.append(float(np.linalg.norm(np.array(x['translation_m'])-y['translation_m'])*1000))
        rot=lambda p:Rotation.from_quat(np.roll(p['quaternion_wxyz'],-1))
        angle.append(float((rot(x).inv()*rot(y)).magnitude()*180/np.pi))
    bands={}
    for name in wrists['local'].keys()|wrists['server'].keys():
        left=wrists['local'].get(name,{});right=wrists['server'].get(name,{})
        differences=[]
        for f in left.keys()&right.keys():
            get=lambda p:np.asarray(p.get('translation_m',p.get('tvec'))).reshape(3)
            differences.append(float(np.linalg.norm(get(left[f])-get(right[f]))*1000))
        bands[name]=statistics(differences)
    return dict(protocol='same-frame raw world poses; no SE3/Sim3 fit, no time offset fitting',
                runs=counts,camera_translation_difference_mm=statistics(distance),
                camera_rotation_difference_deg=statistics(angle),wrist_translation_difference_mm=bands,
                note='Cross-platform agreement is not external trajectory accuracy')

def main():
    p=argparse.ArgumentParser();p.add_argument('local',type=Path);p.add_argument('server',type=Path)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    value=compare(args.local,args.server);args.output.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value,indent=2))
if __name__=='__main__':main()
