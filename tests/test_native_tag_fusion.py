"""Numeric native optimizer test; explicitly build tag_pose_regression first."""
import json
import os
from pathlib import Path
import subprocess
import unittest


@unittest.skipUnless(os.environ.get('RUN_NATIVE_SLAM_TESTS') == '1', 'explicit native integration run')
class NativeTagFusionTests(unittest.TestCase):
    def run_numeric(self):
        root = Path(__file__).resolve().parents[1]
        binary = root/'third_party/ORB_SLAM3/Examples/Monocular/tag_pose_regression'
        self.assertTrue(binary.is_file(), 'build CMake target tag_pose_regression first')
        result = subprocess.run([str(binary)], capture_output=True, text=True, check=True)
        return json.loads(result.stdout.splitlines()[-1])

    def test_fixed_tag_resists_biased_background_and_is_robust_to_one_bad_corner(self):
        report = self.run_numeric()
        self.assertAlmostEqual(report['orb_position_m'], .008, places=5)
        self.assertLess(report['strong_position_m'], report['orb_position_m'])
        self.assertLess(report['strong_tag_rms_px'], report['weak_tag_rms_px'])
        self.assertLess(report['strong_tag_rms_px'], .25)
        self.assertLess(report['biased_tag_rms_px'], 1.)
        self.assertLess(report['corrupt_tag_rms_px'], 1.)

    def test_metric_tag_observations_anchor_local_ba_without_covisible_seed(self):
        report = self.run_numeric()
        self.assertEqual(report['tag_anchored_lba_keyframes'], 2)
        self.assertEqual(report['single_tag_lba_keyframes'], 0)
        self.assertEqual(report['zero_baseline_lba_keyframes'], 0)
        self.assertEqual(report['weak_tag_lba_keyframes'], 0)
        self.assertLess(report['lba_point_error_m'], .0002)

    def test_atlas_save_safely_removes_stale_point_observations(self):
        self.assertEqual(self.run_numeric()['stale_points_after_save'], 0)


if __name__ == '__main__':
    unittest.main()
