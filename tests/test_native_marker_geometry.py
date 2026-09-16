"""Source contracts plus an actual production-linked marker BA fixture.

Build the updated ORB-SLAM3 library before running NativeMarkerGeometryTests.
Only the fixture executable is compiled, in a temporary directory.
"""
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / 'third_party/ORB_SLAM3/build'
SOURCE = ROOT / 'third_party/ORB_SLAM3/src/MarkerGraphOptimizer.cc'


class MarkerGeometrySourceContractTests(unittest.TestCase):
    def test_strong_geometry_gate_precedes_complete_weak_resolution(self):
        source = SOURCE.read_text()
        prepare = source[source.index('bool prepare('):source.index('MGO::ResidualSummary residuals(')]
        self.assertLess(prepare.index('if(!strong)'), prepare.index('proposal.staticTags.emplace'))
        self.assertIn('if(disagreement > 6.1e-3f)', prepare)
        self.assertIn('tagCorners(keyframe, input)', prepare)
        self.assertNotIn('map->mStaticTags', prepare)
        self.assertIn('std::copy(canonical->second.begin(),canonical->second.end(),corners.begin()+start)', prepare)
        self.assertIn('std::max(.35f, keyframe->mTagObservationConfidence) * weight', prepare)
        self.assertIn('if(completeStrongMarkers.count(keyframe->mvTagIds[i]))', prepare)
        self.assertIn('best<=std::min(6.1e-3f,.2f*side)', prepare)


class NativeMarkerGeometryTests(unittest.TestCase):
    def test_actual_proposal_weak_groups_geometry_gates_and_input_immutability(self):
        compiler = shutil.which('c++')
        flags_path = BUILD / 'CMakeFiles/ORB_SLAM3.dir/flags.make'
        link_path = BUILD / 'CMakeFiles/mono_tum_headless.dir/link.txt'
        if compiler is None or not flags_path.is_file() or not link_path.is_file():
            self.skipTest('Configure/build native ORB-SLAM3 and install a C++14 compiler first')
        flags = {}
        for line in flags_path.read_text().splitlines():
            if ' = ' in line:
                key, value = line.split(' = ', 1)
                flags[key] = shlex.split(value)
        with tempfile.TemporaryDirectory(prefix='native_marker_geometry_') as temporary:
            obj, binary = Path(temporary) / 'regression.o', Path(temporary) / 'regression'
            compiled = subprocess.run(
                [compiler, *flags['CXX_DEFINES'], *flags['CXX_INCLUDES'], *flags['CXX_FLAGS'],
                 '-Wno-deprecated-declarations', '-Wno-reorder-ctor', '-c',
                 str(ROOT / 'tests/native_marker_geometry_regression.cc'), '-o', str(obj)],
                capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
            link = shlex.split(link_path.read_text())
            objects = [item for item in link if item.endswith('/mono_tum_headless.cc.o')]
            self.assertEqual(len(objects), 1)
            link = [str(obj) if token == objects[0] else token for token in link]
            link[link.index('-o') + 1] = str(binary)
            linked = subprocess.run(link, cwd=BUILD, capture_output=True, text=True, timeout=60)
            self.assertEqual(linked.returncode, 0, linked.stdout + linked.stderr)
            result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('MARKER_GEOMETRY_SAFETY_OK:', result.stdout)


if __name__ == '__main__':
    unittest.main()
