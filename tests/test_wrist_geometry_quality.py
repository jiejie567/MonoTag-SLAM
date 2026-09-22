import unittest
import numpy as np,cv2
from aruco_track.models import Calibration,Pose
from tools.make_band import band_layout
try:
    from aruco_track.wrist_geometry_quality import wrist_geometry_quality
except ModuleNotFoundError:
    from wrist_geometry_quality import wrist_geometry_quality

class GeometryQualityTests(unittest.TestCase):
    def setUp(self):
        self.layout=band_layout('left',0,69.,55.,56.)
        self.cal=Calibration(np.array([[750.,0,800],[0,750.,640],[0,0,1.]]),np.zeros(5),(1600,1296))
        self.seed=Pose(np.array([[.2],[-.3],[.1]]),np.array([[.02],[.01],[.7]]),0.)
    def detections(self,p,ids=(0,1)):
        return {i:cv2.projectPoints(self.layout.markers[i],p.rvec,p.tvec,self.cal.camera_matrix,self.cal.dist_coeffs)[0].reshape(4,2) for i in ids}
    def test_does_not_mutate_inputs(self):
        det=self.detections(self.seed);r=self.seed.rvec.copy();t=self.seed.tvec.copy();saved={k:v.copy() for k,v in det.items()}
        q=wrist_geometry_quality(det,(0,1),self.layout,self.cal,self.seed)
        self.assertEqual(q['status'],'ok');np.testing.assert_array_equal(r,self.seed.rvec);np.testing.assert_array_equal(t,self.seed.tvec)
        for k in det:np.testing.assert_array_equal(det[k],saved[k])
    def test_ideal_faces_agree_without_prior(self):
        q=wrist_geometry_quality(self.detections(self.seed),(0,1),self.layout,self.cal,self.seed)
        self.assertLess(q['pairs'][0]['translation_disagreement_m'],1e-5)
    def test_sigma_scales_linearly(self):
        a=wrist_geometry_quality(self.detections(self.seed),(0,1),self.layout,self.cal,self.seed,.5)
        b=wrist_geometry_quality(self.detections(self.seed),(0,1),self.layout,self.cal,self.seed,1.)
        self.assertAlmostEqual(b['joint']['translation_sigma_max_m']/a['joint']['translation_sigma_max_m'],2.,places=5)
    def test_near_geometry_has_lower_position_uncertainty(self):
        near=Pose(self.seed.rvec.copy(),np.array([[.02],[.01],[.25]]),0.)
        a=wrist_geometry_quality(self.detections(self.seed),(0,1),self.layout,self.cal,self.seed)
        b=wrist_geometry_quality(self.detections(near),(0,1),self.layout,self.cal,near)
        self.assertLess(b['joint']['translation_sigma_max_m'],a['joint']['translation_sigma_max_m']/2)
    def test_retained_ids_only(self):
        q=wrist_geometry_quality(self.detections(self.seed),(0,),self.layout,self.cal,self.seed)
        self.assertEqual(set(q['faces']),{'0'});self.assertEqual(q['pairs'],[])
    def test_missing_is_unavailable(self):
        q=wrist_geometry_quality({},(),self.layout,self.cal,None)
        self.assertEqual(q['status'],'unavailable')
    def test_motion_is_not_suppressed(self):
        for x in (-.08,0.,.08):
            p=Pose(self.seed.rvec.copy(),np.array([[x],[.01],[.7]]),0.)
            q=wrist_geometry_quality(self.detections(p),(0,1),self.layout,self.cal,p)
            np.testing.assert_allclose(q['joint']['translation_m'],p.tvec.ravel(),atol=1e-5)
    def test_projection_jacobian(self):
        obj=np.concatenate(list(self.layout.markers.values()))
        v=np.r_[self.seed.rvec.ravel(),self.seed.tvec.ravel()]
        _,j=cv2.projectPoints(obj,v[:3],v[3:],self.cal.camera_matrix,self.cal.dist_coeffs)
        def project(x):return cv2.projectPoints(obj,x[:3],x[3:],self.cal.camera_matrix,self.cal.dist_coeffs)[0].ravel()
        numeric=[]
        for i in range(6):
            d=np.zeros(6);d[i]=1e-6;numeric.append((project(v+d)-project(v-d))/2e-6)
        np.testing.assert_allclose(j[:,:6],np.array(numeric).T,atol=1e-4,rtol=1e-5)
if __name__=='__main__':unittest.main()
