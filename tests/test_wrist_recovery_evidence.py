"""Temporal evidence limits must not be renewed by recovered single tags."""
import unittest
from unittest.mock import patch
import numpy as np
from aruco_track.models import Pose
from aruco_track.tag_graph import TagPoseResult, refine_wrist_pose_sequence


class RecoveryEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.camera=Pose(np.zeros((3,1)),np.zeros((3,1)),0.)
        self.missing=TagPoseResult(None,(),(),{},None,0.)
        self.layout=type('Layout',(),{'markers':{1:np.zeros((4,3))}})()

    def measured(self,x=0.,ambiguous=True):
        pose=Pose(np.zeros((3,1)),np.array([[x],[0.],[1.]]),0.,
                  marker_ids=(1,),inlier_count=4,ambiguous=ambiguous)
        return TagPoseResult(pose,(1,),(),{1:0.},0.,1.)

    def run_sequence(self,initial,times,empty=(),maps=None):
        # A genuinely moving wrist, not a zero-motion assumption.
        detections=[{} if i in empty else {1:np.full((4,2),.2*t)}
                    for i,t in enumerate(times)]
        def solve(visible,*args,**kwargs):
            return self.measured(float(visible[1][0,0]))
        with patch('aruco_track.tag_graph.optimize_tag_pose',side_effect=solve):
            return refine_wrist_pose_sequence(
                [self.camera]*len(times),[True]*len(times),
                maps or ['world']*len(times),times,detections,initial,
                self.layout,None,maximum_prediction_gap_s=.15)

    def test_backward_recovered_pose_does_not_extend_horizon(self):
        times=np.arange(9)*.04
        initial=[self.missing]*8+[self.measured(.2*times[-1])]
        result=self.run_sequence(initial,times)
        for i,r in enumerate(result):
            self.assertEqual(r.pose is not None,times[-1]-times[i]<=.15)

    def test_forward_recovered_pose_does_not_extend_horizon(self):
        times=np.arange(9)*.04
        result=self.run_sequence([self.measured()]+[self.missing]*8,times)
        for i,r in enumerate(result):
            self.assertEqual(r.pose is not None,times[i]<=.15)
            if r.pose is not None:
                self.assertAlmostEqual(r.pose.tvec[0,0],.2*times[i])

    def test_explicit_veto_cannot_be_revived_from_future(self):
        veto=TagPoseResult(None,(),(1,),{},None,0.,True)
        result=self.run_sequence([veto,self.measured(.01)],[0.,.05])
        self.assertIsNone(result[0].pose)

    def test_missing_corners_and_other_world_cannot_be_recovered(self):
        result=self.run_sequence([self.missing,self.missing,self.measured()],
                                 [0.,.04,.08],empty=(1,),maps=['other','world','world'])
        self.assertIsNone(result[0].pose)
        self.assertIsNone(result[1].pose)

    def test_explicit_veto_cannot_be_revived_from_past(self):
        veto=TagPoseResult(None,(),(1,),{},None,0.,True)
        result=self.run_sequence([self.measured(),veto],[0.,.05])
        self.assertIsNone(result[1].pose)

    def test_new_measurement_renews_forward_support(self):
        times=np.arange(9)*.04
        initial=[self.missing]*9
        initial[0]=self.measured()
        initial[4]=self.measured(.2*times[4])
        result=self.run_sequence(initial,times)
        self.assertTrue(all(r.pose is not None for r in result[:8]))
        self.assertIsNone(result[8].pose)


if __name__=='__main__':unittest.main()
