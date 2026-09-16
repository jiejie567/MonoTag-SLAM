"""Link the real LoopClosing queue regression against an up-to-date native build."""
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "third_party/ORB_SLAM3/build"


class NativeLoopQueueTests(unittest.TestCase):
    def test_disabled_queue_stays_idle_and_enabled_filters_are_unchanged(self):
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
        with tempfile.TemporaryDirectory(prefix="native_loop_queue_") as temporary:
            obj = Path(temporary) / "queue.o"
            binary = Path(temporary) / "queue"
            compiled = subprocess.run(
                [compiler, *flags["CXX_DEFINES"], *flags["CXX_INCLUDES"],
                 *flags["CXX_FLAGS"], "-Wno-deprecated-declarations", "-Wno-reorder-ctor",
                 "-c", str(ROOT / "tests/native_loop_queue_regression.cc"), "-o", str(obj)],
                capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
            link = shlex.split(link_path.read_text())
            objects = [item for item in link if item.endswith("/mono_tum_headless.cc.o")]
            self.assertEqual(len(objects), 1, "Expected exactly one native runner object")
            link = [str(obj) if token == objects[0] else token for token in link]
            link[link.index("-o") + 1] = str(binary)
            linked = subprocess.run(link, cwd=BUILD, capture_output=True, text=True, timeout=60)
            self.assertEqual(linked.returncode, 0, linked.stdout + linked.stderr)
            result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("LOOP_QUEUE_SAFETY_OK: disabled 382 inserts stay idle", result.stdout)


if __name__ == "__main__":
    unittest.main()
