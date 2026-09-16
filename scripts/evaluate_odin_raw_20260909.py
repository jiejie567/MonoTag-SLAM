"""Fixed extrinsic and header timestamps; odometry never enters SLAM."""
import csv
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input-dir',type=Path,default=Path(__file__).resolve().parents[1]/'output/odin_raw_20260909_094004_local')
parser.add_argument('--output-dir',type=Path)
args=parser.parse_args()
INPUT=args.input_dir
OUT=args.output_dir or INPUT
def stats(x):
    x=np.asarray(x)
    return {'rmse':float(np.sqrt(np.mean(x*x))),'median':float(np.median(x)),
            'p95':float(np.percentile(x,95)),'max':float(np.max(x))} if len(x) else None
def align(x,y,scaled):
    a,b=x.mean(0),y.mean(0);xc,yc=x-a,y-b
    assert np.isfinite(x).all() and np.isfinite(y).all()
    u,s,v=np.linalg.svd(np.einsum('ni,nj->ij',yc,xc)/len(x));D=np.eye(3);D[2,2]=np.linalg.det(u@v)
    R=u@D@v;c=float(np.sum(s*np.diag(D))/np.mean(np.sum(xc*xc,axis=1))) if scaled else 1.
    return c,R,b-c*R@a

odom=list(csv.DictReader((INPUT/'odometry.csv').open()))
ot=np.array([float(r['timestamp_s']) for r in odom]);assert np.all(np.diff(ot)>0)
assert {r['child_frame_id'] for r in odom}=={'imu'}
op=np.array([[float(r[k]) for k in ['x_m','y_m','z_m']] for r in odom])
oq=Rotation.from_quat([[float(r[k]) for k in ['qx','qy','qz','qw']] for r in odom])
ts=np.array([float(r['timestamp_s']) for r in csv.DictReader((INPUT/'image_timestamps.csv').open())])
Tic=np.array(json.loads((INPUT/'camera_rectified.json').read_text())['T_imu_camera'])
Tic[:3,:3]=Rotation.from_matrix(Tic[:3,:3]).as_matrix()
actions=[json.loads(l) for l in (OUT/'actions.jsonl').open()]
groups=defaultdict(list)
for row in actions:
    i=row['frame']
    if row.get('camera_world_pose_fused') and ot[0]<=ts[i]<=ot[-1] and row.get('camera_world_source')!='invalid':
        groups[row['camera_submap_id']].append(row)
result={'reference':'Odin odometry (not independent ground truth)','fixed_extrinsic':Tic.tolist(),
        'time_offset_fitted':False,'extrinsic_fitted':False,'maps':{},'frames':len(actions)}
samples=[]
for name,rows in groups.items():
    times=ts[[r['frame'] for r in rows]]
    ir=Slerp(ot,oq)(times).as_matrix()
    ip=np.column_stack([np.interp(times,ot,op[:,a]) for a in range(3)])
    ref=ip+np.einsum('nij,j->ni',ir,Tic[:3,3]);refR=ir@Tic[:3,:3]
    ep=np.array([r['camera_world_pose_fused']['translation_m'] for r in rows])
    eq=np.array([r['camera_world_pose_fused']['quaternion_wxyz'] for r in rows])
    er=Rotation.from_quat(eq[:,[1,2,3,0]]).as_matrix()
    metric=all(r['scale_status']=='metric' for r in rows)
    modes=['SE3','Sim3_diagnostic'] if metric else ['Sim3']
    entries={'frames':len(rows),'metric':metric,'start_s':float(times[0]-ts[0]),'end_s':float(times[-1]-ts[0])}
    for mode in modes:
        c,R,t=align(ep,ref,mode!='SE3');pred=c*np.einsum('ij,nj->ni',R,ep)+t;predR=R@er
        pe=np.linalg.norm(pred-ref,axis=1)
        re=np.degrees(Rotation.from_matrix(np.transpose(refR,(0,2,1))@predR).magnitude())
        rpe=[]
        for i,now in enumerate(times):
            j=int(np.argmin(np.abs(times-(now+1))))
            if abs(times[j]-now-1)>.04:continue
            if rows[j]['frame']-rows[i]['frame']!=j-i:continue
            a=predR[i].T@(pred[j]-pred[i]);b=refR[i].T@(ref[j]-ref[i])
            rpe.append(np.linalg.norm(a-b))
        entries[mode]={'alignment_scale':c,'ATE_m':stats(pe),'orientation_deg':stats(re),'RPE_1s_translation_m':stats(rpe)}
        if mode==modes[0]:
            for row,p,q,e,angle in zip(rows,pred,ref,pe,re):
                samples.append([row['frame'],ts[row['frame']]-ts[0],name,mode,*p,*q,e,angle])
    result['maps'][name]=entries
result['evaluated_frames']=sum(len(g) for g in groups.values())
result['coverage']=result['evaluated_frames']/len(actions)
(OUT/'accuracy.json').write_text(json.dumps(result,indent=2))
with (OUT/'accuracy_samples.csv').open('w') as f:
    w=csv.writer(f);w.writerow(['frame','time_s','map','alignment','estimate_x','estimate_y','estimate_z','reference_x','reference_y','reference_z','error_m','rotation_deg']);w.writerows(samples)
print(json.dumps(result,indent=2))
