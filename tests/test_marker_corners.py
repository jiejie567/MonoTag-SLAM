import tempfile
from pathlib import Path
import unittest

import cv2
import numpy as np

from aruco_track.marker_corners import MarkerCornerTracker, TrackedMarkerObservation
from aruco_track.models import BandLayout, Calibration, Pose
from aruco_track.orbslam3_backend import write_tag_observation_hints


class MarkerCornerTests(unittest.TestCase):
    def setUp(self):
        self.calibration = Calibration(np.array([[500.,0,320],[0,500.,240],[0,0,1]]), np.zeros(5), (640,480))
        self.world = np.array([[-.06,-.06,0],[.06,-.06,0],[.06,.06,0],[-.06,.06,0]])
        self.layout = BandLayout('world','DICT_4X4_50',{20:self.world})
        self.tracker = MarkerCornerTracker(self.calibration,self.layout)
        self.rvec = np.array([.35,-.2,.04])
        self.marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),20,160)

    def frame(self, x=0., hidden=()):
        tvec=np.array([-x,0,.6])
        rotation=cv2.Rodrigues(self.rvec)[0]
        pose=Pose(cv2.Rodrigues(rotation.T)[0],(-rotation.T@tvec).reshape(3,1),0.)
        corners=cv2.projectPoints(self.world,self.rvec,tvec,self.calibration.camera_matrix,np.zeros(5))[0].reshape(4,2)
        transform=cv2.getPerspectiveTransform(np.array([[0,0],[159,0],[159,159],[0,159]],np.float32),corners.astype(np.float32))
        image=cv2.warpPerspective(self.marker,transform,(640,480),borderValue=210)
        for index in hidden:
            cv2.circle(image,tuple(np.rint(corners[index]).astype(int)),13,210,-1)
        return image,corners,pose

    def seed(self):
        frame,corners,pose=self.frame()
        self.tracker.update(frame,0,{20:corners},(20,),pose,.9)

    def test_one_occluded_corner_keeps_only_three_real_corners(self):
        for hidden in range(4):
            for translation in (0.,.001,.003,.005):
                with self.subTest(hidden=hidden,translation=translation):
                    self.tracker=MarkerCornerTracker(self.calibration,self.layout)
                    self.seed()
                    frame,_,truth=self.frame(translation,(hidden,))
                    result=self.tracker.update(frame,1/60,{},(),None,0.)
                    self.assertEqual(result.corner_indices,[i for i in range(4) if i!=hidden],result)
                    self.assertIsNotNone(result.pose,result.reason)
                    self.assertTrue(result.partial_only)
                    self.assertLess(np.linalg.norm(result.pose.tvec-truth.tvec),.01)

    def test_no_decoded_identity_cannot_start_tracking(self):
        frame,_,_=self.frame(.001)
        result=self.tracker.update(frame,0,{},(),None,0.)
        self.assertIsNone(result.pose)
        self.assertFalse(result.world_points)

    def test_three_exact_corners_do_not_justify_a_large_pose_step(self):
        self.seed()
        _, corners, _ = self.frame(.0188)
        pose, reason, _ = self.tracker._partial_pose(
            self.world[1:], corners[1:], [20]*3, 1/60)
        self.assertIsNone(pose)
        self.assertEqual(reason, 'three_corner_motion_gate')

    def test_three_corner_pose_expires_before_accumulated_flow_drift(self):
        self.seed()
        for index in range(1,10):
            frame,_,_=self.frame(0.,(0,))
            result=self.tracker.update(frame,index/60,{},(),None,0.)
            if index<=6:
                self.assertIsNotNone(result.pose,result.reason)
            else:
                self.assertIsNone(result.pose)
                self.assertEqual(result.reason,'three_corner_age_gate')

    def test_two_visible_corners_and_full_occlusion_cannot_supply_pose(self):
        for hidden in [(0,1),(0,1,2,3)]:
            self.tracker=MarkerCornerTracker(self.calibration,self.layout)
            self.seed()
            frame,_,_=self.frame(.001,hidden)
            result=self.tracker.update(frame,1/60,{},(),None,0.)
            self.assertIsNone(result.pose,result)

    def test_decode_gap_expires_even_if_flow_stays_good(self):
        self.seed()
        for index in range(1,25):
            frame,_,_=self.frame()
            result=self.tracker.update(frame,index/60,{},(),None,0.)
            if index>18:
                self.assertIsNone(result.pose)
                self.assertFalse(result.world_points)

    def test_weak_detection_does_not_seed_corner_identities(self):
        frame,corners,pose=self.frame()
        self.tracker.update(frame,0,{20:corners},(20,),pose,.9,{20:.25})
        result=self.tracker.update(frame,1/60,{},(),None,0.)
        self.assertIsNone(result.pose)
        self.assertFalse(result.world_points)

    def test_pose_consistent_weak_corners_require_three_consecutive_frames(self):
        weak_world = self.world + np.array([.15, 0., 0.])
        self.tracker = MarkerCornerTracker(
            self.calibration,
            BandLayout('world', 'DICT_4X4_50', {20: self.world, 21: weak_world}),
        )
        frame, corners, pose = self.frame()
        weak_pixels, depth = self.tracker._project(weak_world, pose)
        self.assertTrue(np.all(depth > 0))
        results = []
        for index in range(3):
            results.append(self.tracker.update(
                frame, index / 60, {20: corners}, (20,), pose, .9,
                weak_detections={21: weak_pixels},
                weak_corner_weights={21: (.05,) * 4},
            ))
        self.assertFalse(results[0].world_points)
        self.assertFalse(results[1].world_points)
        self.assertEqual(results[2].marker_ids, [21] * 4)
        self.assertEqual(results[2].corner_indices, [0, 1, 2, 3])
        self.assertEqual(results[2].point_weights, [.05] * 4)

    def test_prediction_inconsistent_weak_corners_never_become_factors(self):
        weak_world = self.world + np.array([.15, 0., 0.])
        self.tracker = MarkerCornerTracker(
            self.calibration,
            BandLayout('world', 'DICT_4X4_50', {20: self.world, 21: weak_world}),
        )
        frame, corners, pose = self.frame()
        weak_pixels, _ = self.tracker._project(weak_world, pose)
        weak_pixels += np.array([3.0, 0.0])
        for index in range(5):
            result = self.tracker.update(
                frame, index / 60, {20: corners}, (20,), pose, .9,
                weak_detections={21: weak_pixels},
                weak_corner_weights={21: (.05,) * 4},
            )
            self.assertFalse(result.world_points)

    def test_hint_marks_three_corners_partial_not_a_complete_detection(self):
        _,corners,pose=self.frame()
        tracked=TrackedMarkerObservation(pose,.3,True,self.world[1:].tolist(),corners[1:].tolist(),
                                          [20]*3,[1,2,3],'tracked',1/60)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'tags.txt'
            write_tag_observation_hints(path,[None],[0.],[{}],[()],self.layout,self.calibration,60,0,
                                        tracked_observations=[tracked])
            line=path.read_text().splitlines()[-1]
        self.assertIn('weights 0.25 0.25 0.25 ids 20 20 20 tracked 3 partial 1 age',line)

    def test_hint_preserves_confirmed_weak_corner_information(self):
        _,corners,pose=self.frame()
        tracked=TrackedMarkerObservation(
            pose, .9, False, self.world.tolist(), corners.tolist(),
            [20]*4, [0,1,2,3], 'with_decoded_marker', 0., [.05]*4,
        )
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'tags.txt'
            write_tag_observation_hints(
                path,[pose],[.9],[{20:corners}],[(20,)],self.layout,
                self.calibration,60,0,tracked_observations=[tracked],
            )
            line=path.read_text().splitlines()[-1]
        self.assertIn('0.05 0.05 0.05 0.05 ids', line)


if __name__=='__main__':
    unittest.main()
