"""Run the actual native cache/rollback/known-motion regression without SLAM.

Requires an up-to-date ORB-SLAM3 CMake build. Only the test executable is compiled,
inside a temporary directory; native implementations and libraries are not edited.
"""
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "third_party/ORB_SLAM3"
BUILD = NATIVE / "build"


class NativeReliableRecoveryTests(unittest.TestCase):
    def test_cache_lifecycle_rollback_and_measured_older_frame_recovery(self):
        compiler = shutil.which("c++")
        flags_path = BUILD / "CMakeFiles/ORB_SLAM3.dir/flags.make"
        link_path = BUILD / "CMakeFiles/mono_tum_headless.dir/link.txt"
        if compiler is None or not flags_path.is_file() or not link_path.is_file():
            self.skipTest("Configure/build native ORB-SLAM3 and install a C++14 compiler first")
        flags = {}
        for line in flags_path.read_text().splitlines():
            if " = " in line:
                key, value = line.split(" = ", 1)
                flags[key] = shlex.split(value)
        with tempfile.TemporaryDirectory(prefix="reliable_recovery_regression_") as temporary:
            obj = Path(temporary) / "regression.o"
            binary = Path(temporary) / "regression"
            compile_command = [
                compiler, *flags["CXX_DEFINES"], *flags["CXX_INCLUDES"],
                *flags["CXX_FLAGS"], "-Wno-deprecated-declarations", "-Wno-reorder-ctor",
                "-c", str(ROOT / "tests/native_reliable_recovery_regression.cc"), "-o", str(obj),
            ]
            compiled = subprocess.run(compile_command, capture_output=True, text=True, timeout=60)
            self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
            link = shlex.split(link_path.read_text())
            objects = [item for item in link if item.endswith("/mono_tum_headless.cc.o")]
            self.assertEqual(len(objects), 1, "Expected exactly one native runner object")
            link = [str(obj) if token == objects[0] else token for token in link]
            link[link.index("-o") + 1] = str(binary)
            linked = subprocess.run(link, cwd=BUILD, capture_output=True, text=True, timeout=60)
            self.assertEqual(linked.returncode, 0, linked.stdout + linked.stderr)
            result = subprocess.run(
                [str(binary), str(ROOT / "tests/fixtures/reliable_recovery_camera.yaml")],
                capture_output=True, text=True, timeout=30,
            )
            output = result.stdout + result.stderr
            self.assertEqual(result.returncode, 0, output)
            self.assertIn("RELIABLE_RECOVERY_SAFETY_OK:", result.stdout)
            self.assertIn("VISUAL_SUPPORT_FLOOR_OK:", result.stdout)
            self.assertIn("reason=opencv_exception", result.stdout)
            for weak_seeds in (0, 15):
                self.assertIn(
                    f"CONTROLLED_CACHE_AB weak_seeds={weak_seeds} cache_off=0 cache_on=1 "
                    "source_frame=3000 predecessor_frame=3001 current_frame=3002",
                    result.stdout,
                )


if __name__ == "__main__":
    unittest.main()
