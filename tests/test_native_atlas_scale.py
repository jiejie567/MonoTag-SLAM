"""Exercise actual native graph guards and culled-reference scale propagation."""
import os
from pathlib import Path
import subprocess
import unittest


@unittest.skipUnless(os.environ.get("RUN_NATIVE_SLAM_TESTS") == "1", "explicit native integration run")
class NativeAtlasScaleTests(unittest.TestCase):
    def test_guards_culled_reference_chain_and_repeated_sim3_corrections(self):
        binary = (Path(__file__).resolve().parents[1] / "third_party/ORB_SLAM3"
                  / "Examples/Monocular/atlas_scale_regression")
        self.assertTrue(binary.is_file(), "build target atlas_scale_regression first")
        result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("ATLAS_SCALE_REGRESSION_OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
