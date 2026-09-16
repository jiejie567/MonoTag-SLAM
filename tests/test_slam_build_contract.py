"""Keep the native dependency build coupled to the selected C++ standard."""
from pathlib import Path
import unittest


class SlamBuildContractTests(unittest.TestCase):
    def test_standard_override_is_preserved(self):
        source = Path(__file__).resolve().parents[1] / 'third_party/ORB_SLAM3/CMakeLists.txt'
        cmake = source.read_text()
        self.assertIn('if(NOT DEFINED CMAKE_CXX_STANDARD)\n  set(CMAKE_CXX_STANDARD 14)\nendif()', cmake)

    def test_g2o_is_a_build_dependency_not_a_stale_library_file(self):
        source = Path(__file__).resolve().parents[1] / 'third_party/ORB_SLAM3/CMakeLists.txt'
        cmake = source.read_text()
        self.assertIn('add_subdirectory(Thirdparty/g2o)', cmake)
        self.assertIn('${ORB_SLAM3_DBOW2_LIBRARY}\ng2o\n', cmake)
        self.assertNotIn('${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o', cmake)


if __name__ == '__main__':
    unittest.main()
