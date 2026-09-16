"""Dataset adapter: manufacturer FishPoly rectification, no odometry feedback."""
import csv
import json
import re
import subprocess
from pathlib import Path
import cv2
import numpy as np
from mcap_ros2.reader import read_ros2_messages

ROOT = Path(__file__).resolve().parents[1]
BAG = ROOT / 'input/odin_raw_20260909_094004/odin_raw_20260909_094004'
OUT = ROOT / 'output/odin_raw_20260909_094004_local'
OUT.mkdir(exist_ok=True)
text = (BAG / 'calib.yaml').read_text()
def value(key):
    return float(re.search(r'^\s*'+key+r':\s*([-+\deE.]+)',text,re.M)[1])
assert 'cam_model: FishPoly' in text
assert value('p1') == value('p2') == 0
W,H=int(value('image_width')),int(value('image_height'))
# Same optical axes; pinhole target chosen from calibration, no fitted intrinsics.
K=np.array([[value('A11'),0,(W-1)/2],[0,value('A22'),(H-1)/2],[0,0,1.]])
y,x=np.indices((H,W),dtype=float)
xn,yn=(x-K[0,2])/K[0,0],(y-K[1,2])/K[1,1]
r=np.hypot(xn,yn);theta=np.arctan(r)
td=theta.copy()
for n in range(2,8):td+=value('k'+str(n))*theta**n
scale=np.divide(td,r,out=np.ones_like(r),where=r>1e-12)
mx=(value('A11')*xn*scale+value('A12')*yn*scale+value('u0')).astype('float32')
my=(value('A22')*yn*scale+value('v0')).astype('float32')
assert mx.min()>=0 and mx.max()<W-1 and my.min()>=0 and my.max()<H-1
rows=[];odom=[]
raw=OUT/'raw_jpeg';raw.mkdir(exist_ok=True)
for record in read_ros2_messages(next(BAG.glob('*.mcap'))):
    m=record.ros_msg;s=m.header.stamp;t=s.sec+s.nanosec*1e-9
    if record.channel.topic=='/odin1/image/compressed':
        name=f'{len(rows):06d}.jpg';(raw/name).write_bytes(bytes(m.data));rows.append([len(rows),t,name])
    elif record.channel.topic=='/odin1/odometry':
        p=m.pose.pose
        odom.append([t,m.header.frame_id,m.child_frame_id,p.position.x,p.position.y,p.position.z,
                     p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w,*m.pose.covariance])
times=np.array([r[1] for r in rows]);assert np.all(np.diff(times)>0)
fps=(len(times)-1)/(times[-1]-times[0]);linear=times[0]+np.arange(len(times))/fps
timing_error=float(np.max(np.abs(linear-times)))
assert timing_error<.001, 'Use a timestamp-aware video adapter for nonuniform input'
with (OUT/'image_timestamps.csv').open('w') as f:
    writer=csv.writer(f);writer.writerow(['frame','timestamp_s','file']);writer.writerows(rows)
with (OUT/'odometry.csv').open('w') as f:
    writer=csv.writer(f);writer.writerow(['timestamp_s','frame_id','child_frame_id','x_m','y_m','z_m','qx','qy','qz','qw']+[f'pose_covariance_{i}' for i in range(36)]);writer.writerows(odom)
Tcl=np.array([float(v) for v in re.search(r'Tcl_0:\s*\[([^]]+)\]',text,re.S)[1].replace('\n','').split(',')]).reshape(4,4)
Til=np.eye(4);Til[:3,3]=[-.02663,.03447,.02174]
Tic=Til@np.linalg.inv(Tcl)
cal={'image_size':[W,H],'camera_matrix':K.tolist(),'dist_coeffs':[0]*5,'source':str(BAG/'calib.yaml'),
     'input_is_undistorted':True,'original_model':'FishPoly','T_imu_camera':Tic.tolist()}
(OUT/'camera_rectified.json').write_text(json.dumps(cal,indent=2))
summary={'image_frames':len(rows),'duration_s':float(times[-1]-times[0]),'fps':fps,
         'uniform_time_max_error_s':timing_error,'odometry_frame_id':odom[0][1],
         'odometry_child_frame_id':odom[0][2],'reference_only':True,
         'extrinsic_source':'calib.yaml Tcl_0 + manufacturer fixed T_imu_lidar',
         'rectification_source':'https://github.com/ManifoldTechLtd/wiki/blob/master/docs/odin_series/odin1/5.%20Data%20output_.md'}
(OUT/'input_summary.json').write_text(json.dumps(summary,indent=2));print(summary,flush=True)
video=OUT/'rectified_lossless.mp4'
command=['ffmpeg','-nostdin','-v','error','-n','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}',
         '-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','ultrafast','-crf','0','-movflags','+faststart',str(video)]
p=subprocess.Popen(command,stdin=subprocess.PIPE)
try:
    for i,_,name in rows:
        im=cv2.imread(str(raw/name));assert im.shape[:2]==(H,W)
        rect=cv2.remap(im,mx,my,cv2.INTER_LINEAR)
        if i==30:cv2.imwrite(str(OUT/'rectified_preview.png'),rect)
        p.stdin.write(rect.tobytes())
finally:p.stdin.close()
assert p.wait()==0
print('Ready:',video,flush=True)
