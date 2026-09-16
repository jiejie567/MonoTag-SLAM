import copy,sys,unittest
from pathlib import Path
from dataclasses import replace
import numpy as np
import cv2
from aruco_track.camera_state import FusedCameraFrame
from aruco_track.models import Pose
from aruco_track.orbslam3_backend import MetricOrbSlamResult
from aruco_track.offline_gaps import gap_requests,apply_gap_candidates

def fixture():
    p=Pose(np.zeros(3),np.array([0.,0.,1.]),1.)
    good=FusedCameraFrame(p,'head-slam',1.,50,1.,'atlas_0',42,True,'marker',True)
    bad=FusedCameraFrame(None,'invalid',0.,0,None)
    frames=[good,bad,good];mapping=dict(id=0,metric=True,background=True,revision=42,
        points=[[i,(u-320)/500,(v-240)/500,2.] for i,(u,v,_,_) in enumerate(features())])
    history=[dict(timestamp=i*.1,state=3 if i==1 else 2,active_map=0,pose=None if i==1 else [0,0,1,0,0,0,1]) for i in range(3)]
    history.append(dict(final=True,timestamp=.2))
    return MetricOrbSlamResult(frames,np.empty((0,3)),(),[None]*3,0,1.,0,None,None,{},history,{'atlas_0':mapping})

def features():
    return [[float(u),float(v),i,0.] for i,(u,v) in enumerate(( (u,v) for u in np.linspace(80,560,8) for v in np.linspace(70,410,5)))]

def request(r):
    q=gap_requests(r,10)[0]
    q.update(image_width=640,image_height=480,camera_matrix=[[500,0,320],[0,500,240],[0,0,1]],dist_coeffs=[0]*5)
    return q

def rows():
    f=features();hull=cv2.contourArea(cv2.convexHull(np.asarray(f,np.float32)[:,:2].copy()))/(640*480)
    return [dict(type='metadata',schema='readonly-short-gap/v1',atlas_modified=False,controls_valid=True,chain_valid=True,map_id=0,map_revision=42),
            dict(type='frame',frame=1,accepted=True,source='offline-short-gap-relocalization',support_only=False,
                 map_id=0,map_revision=42,pose=[0,0,1,0,0,0,1],inliers=40,rms_px=0.,inlier_fraction=.8,
                 matches=50,matched_feature_count=40,matched_features=f,
                 T_world_camera=[[1,0,0,0],[0,1,0,0],[0,0,1,1],[0,0,0,1]],
                 validated_after_final_atlas=True,validation_effective_time_s=.2,gauge='final_metric_atlas',connected=True,candidate=True,
                 occupied_cells=12,hull_fraction=hull,timestamp_s=.1)]

class Tests(unittest.TestCase):
    def test_accept_without_history_or_valid_pose_change(self):
        r=fixture();h=copy.deepcopy(r.history);q=request(r);s=apply_gap_candidates(r,q,rows())
        self.assertIs(s.frames[0],r.frames[0]);self.assertIs(s.frames[2],r.frames[2]);self.assertEqual(s.history,h)
        self.assertIsNone(r.frames[1].pose);self.assertIsNotNone(s.frames[1].pose)
        self.assertFalse(s.frames[1].localization_recovery['original_tracking_valid'])
    def test_reject_long_cross_map_unmetric_prefix_and_tail(self):
        for change in ('long','map','revision','metric','prefix','tail','recovered'):
            r=fixture()
            if change=='long':r.history[2]['timestamp']=.7
            if change=='map':r.history[1]['active_map']=1
            if change=='revision':r.frames[2]=replace(r.frames[2],revision=43)
            if change=='metric':r.frames[0]=replace(r.frames[0],metric=False)
            if change=='prefix':r.history[1]['state']=1
            if change=='tail':r.frames[2]=r.frames[1]
            if change=='recovered':r.frames[0]=replace(r.frames[0],localization_recovery={'accepted':True})
            self.assertEqual(gap_requests(r,10),[],change)
    def test_bad_evidence_never_publishes(self):
        for change in ('controls','chain','map','revision','nan','quaternion','count','ratio','cells','hull','time','duplicate',
                       'nan_time','schema','modified','pixel','point_id','effective','transform','feature_count','geometry'):
            r=fixture();q=request(r);rs=rows()
            if change in ('controls','chain'):rs[0][change+'_valid']=False
            if change=='map':rs[1]['map_id']=1
            if change=='revision':rs[1]['map_revision']=43
            if change=='nan':rs[1]['pose'][0]=float('nan')
            if change=='quaternion':rs[1]['pose'][-1]=2
            if change=='count':rs[1]['inliers']=17
            if change=='ratio':rs[1]['inlier_fraction']=.2
            if change=='cells':rs[1]['occupied_cells']=2
            if change=='hull':rs[1]['hull_fraction']=.01
            if change=='time':rs[1]['timestamp_s']=1
            if change=='duplicate':rs.append(copy.deepcopy(rs[1]))
            if change=='nan_time':rs[1]['timestamp_s']=float('nan')
            if change=='schema':rs[0]['schema']='wrong'
            if change=='modified':rs[0]['atlas_modified']=True
            if change=='pixel':rs[1]['matched_features'][0][0]=-1
            if change=='point_id':rs[1]['matched_features'][0][2]=1000
            if change=='effective':rs[1]['validation_effective_time_s']=.1
            if change=='transform':rs[1]['T_world_camera'][0][3]=1
            if change=='feature_count':rs[1]['matched_feature_count']=39
            if change=='geometry':r.maps['atlas_0']['points'][0][1]+=1
            self.assertIsNone(apply_gap_candidates(r,q,rs).frames[1].pose,change)

if __name__=='__main__':unittest.main()
