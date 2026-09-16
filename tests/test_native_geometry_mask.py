"""Native image/geometry regression; no learned model or camera required."""
import json
import os
from pathlib import Path
import subprocess
import unittest


@unittest.skipUnless(os.environ.get('RUN_NATIVE_SLAM_TESTS') == '1', 'explicit native integration run')
class NativeGeometryMaskTests(unittest.TestCase):
    def test_moving_patch_and_static_camera_motion_controls(self):
        root=Path(__file__).resolve().parents[1]
        binary=root/'third_party/ORB_SLAM3/Examples/Monocular/geometry_mask_regression'
        r=subprocess.run([str(binary)],text=True,capture_output=True,check=True)
        report=json.loads(r.stdout.splitlines()[-1])
        self.assertGreater(report['dynamic_hits']/report['dynamic_eligible'],.5)
        self.assertEqual(report['static_false_hits'],0)
        self.assertEqual(report['camera_only_masked'],0)
        self.assertEqual(report['bad_pose_masked'],0)
        self.assertEqual(report['lost_next_masked'],0)
        self.assertEqual(report['map_next_masked'],0)
        self.assertEqual(report['gap_next_masked'],0)
        self.assertEqual(report['excluded_features'],0)
        self.assertEqual(report['fully_masked_features'],0)
        self.assertGreater(report['allowed_feature_count'],1500)


if __name__=='__main__': unittest.main()
