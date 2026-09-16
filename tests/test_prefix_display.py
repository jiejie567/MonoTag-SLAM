import copy
import unittest
from aruco_track.offline_prefix_display import validate_display_rows
from tests import test_prefix_replay_features as fixtures


class PrefixDisplayTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.PrefixReplayFeatureTests()
        fixture.setUp()
        self.request = fixture.prefix
        self.final = fixture.history[-1]
        transform = fixture.request['queries'][0]['T_world_camera']
        self.rows = [fixture.metadata,
            {**fixture.row, 'T_world_camera': transform},
            {**fixture.other, 'T_world_camera': transform}]

    def test_display_validation_preserves_inputs_and_marks_display_only(self):
        original = copy.deepcopy((self.rows, self.request, self.final))
        output = validate_display_rows(self.rows, self.request, self.final)
        self.assertIn(0, output)
        self.assertTrue(output[0]['display_only'])
        self.assertFalse(output[0]['labels_modified'])
        self.assertEqual(output[0]['source'], 'offline-prefix-display-correspondence')
        self.assertEqual((self.rows, self.request, self.final), original)

    def test_bad_pose_is_not_displayed(self):
        for transform in ([[0]*4]*4, [[float('nan')]*4]*4):
            rows=copy.deepcopy(self.rows)
            rows[1]['T_world_camera']=transform
            self.assertNotIn(0, validate_display_rows(rows,self.request,self.final))

    def test_geometry_failure_and_wrong_revision_remain_rejected(self):
        for key,value in [('accepted',False),('connected',False),('map_revision',999),('inliers',4)]:
            rows=copy.deepcopy(self.rows)
            rows[1][key]=value
            self.assertNotIn(0,validate_display_rows(rows,self.request,self.final))

    def test_failed_anchor_does_not_display_anything(self):
        self.rows[0]['anchor_valid']=False
        self.assertFalse(validate_display_rows(self.rows,self.request,self.final))

    def test_guided_matches_need_independent_bracketing_frames(self):
        base=self.rows[1]
        before={**base,'frame':0,'timestamp_s':0.}
        middle={**base,'frame':1,'timestamp_s':1/60.,'guided_matching':True,
                'guide_before':0,'guide_after':2}
        after={**base,'frame':2,'timestamp_s':2/60.}
        rows=[self.rows[0],before,middle,after]
        self.assertIn(1,validate_display_rows(rows,self.request,self.final))
        for field,value in [('connected',False),('guided_matching',True),
                             ('map_id',9),('timestamp_s',-.3)]:
            changed=copy.deepcopy(rows);changed[1][field]=value
            self.assertNotIn(1,validate_display_rows(changed,self.request,self.final))


if __name__=='__main__':unittest.main()
