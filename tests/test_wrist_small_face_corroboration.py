"""A temporal planar branch must not discard good measured second faces."""
import json
from pathlib import Path
import unittest
import numpy as np
from aruco_track.models import BandLayout, Calibration, Pose
from aruco_track.tag_graph import (
    optimize_tag_pose, _prefer_dominant_face_when_small_face_disagrees,
)

class SmallFaceCorroborationTests(unittest.TestCase):
    def test_measured_second_face_overrules_stale_planar_prediction(self):
        data=json.loads((Path(__file__).parent/'fixtures'/'wrist_small_face_corroboration.json').read_text())
        layout=BandLayout('regression','DICT_4X4_50',
                          {int(k):np.array(v) for k,v in data['points'].items()})
        calibration=Calibration(np.array(data['camera_matrix']),
                                np.array(data['dist_coeffs']),(1920,1080))
        detections={int(k):np.array(v) for k,v in data['corners'].items()}
        predicted=Pose(np.array(data['predicted_rvec']).reshape(3,1),
                       np.array(data['predicted_tvec']).reshape(3,1),0.)
        joint=optimize_tag_pose(detections,layout,calibration,predicted,
                                validate_planar_ambiguity=True)
        self.assertEqual(len(joint.accepted_marker_ids),2)
        self.assertLess(max(joint.marker_errors_px.values()),2.)
        kept=_prefer_dominant_face_when_small_face_disagrees(
            joint,predicted,detections,layout,calibration,{})
        self.assertIs(kept,joint)
        self.assertFalse(kept.pose.ambiguous)

if __name__=='__main__': unittest.main()
