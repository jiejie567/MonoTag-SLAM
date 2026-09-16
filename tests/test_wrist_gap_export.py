import copy
import unittest
from aruco_track.wrist_gap_export import complete_wrists, CompletedReplayTrails

def rows():
    result=[]
    for i in range(3):
        result.append(dict(frame=i,timestamp_s=i*.1,world_frame_id='atlas_0',map_revision=4,
            scale_status='metric',camera_world_source='head-slam',camera_world_pose_fused={},
            hands={'left':dict(world_submap_id='atlas_0',wrist_world_graph=None if i==1 else
                dict(translation_m=[i,0,0],quaternion_wxyz=[1,0,0,0]))}))
    return result

class CompletionTests(unittest.TestCase):
    def test_replay_matches_export_and_preserves_joints(self):
        completed=complete_wrists(rows());cache=CompletedReplayTrails(completed)
        frame=dict(source_frame=2,trail_timestamps_s=[0,.1,.2],camera={},metric=True,
                   map_id='atlas_0',map_revision=4,hands={'left':[[9,8,7]]})
        cache.apply(frame)
        self.assertEqual(frame['trails']['left'][1],completed[1]['wrists']['left']['pose']['translation_m'])
        self.assertEqual(frame['trail_display_interpolated']['left'],[1])
        self.assertEqual(frame['hands'],{'left':[[9,8,7]]})
        frame['map_revision']=5;cache.apply(frame)
        self.assertEqual(frame['trails']['left'],[None]*3)
    def test_short_gap_and_raw_preserved(self):
        r=rows();before=copy.deepcopy(r);out=complete_wrists(r)
        self.assertEqual(out[1]['wrists']['left']['pose']['translation_m'],[1,0,0])
        self.assertEqual(out[1]['wrists']['left']['source'],'interpolated')
        self.assertFalse(out[1]['wrists']['left']['measurement_valid'])
        self.assertEqual(r,before)
    def test_breaks(self):
        for field,value in [('camera_world_source','invalid'),('scale_status','arbitrary-scale'),('world_frame_id','atlas_1'),('map_revision',5)]:
            with self.subTest(field=field):
                r=rows();r[1][field]=value
                self.assertIsNone(complete_wrists(r)[1]['wrists']['left']['pose'])
    def test_long_gap(self):
        r=rows();r[2]['timestamp_s']=.6
        self.assertIsNone(complete_wrists(r)[1]['wrists']['left']['pose'])
    def test_unbracketed(self):
        r=rows()[1:]
        self.assertIsNone(complete_wrists(r)[0]['wrists']['left']['pose'])
    def test_quaternion_sign(self):
        r=rows();r[2]['hands']['left']['wrist_world_graph']['quaternion_wxyz']=[-1,0,0,0]
        q=complete_wrists(r)[1]['wrists']['left']['pose']['quaternion_wxyz']
        self.assertAlmostEqual(abs(q[0]),1)

if __name__=='__main__':unittest.main()
