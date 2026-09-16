"""Corner-candidate lifetime is separate from continuous dense scale evidence.

Source guards run without native compilation; opt-in native cases exercise
the production selector and OnFrameEnd, stopping before any BA or commit.
"""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / 'third_party/ORB_SLAM3'


def body(source, signature):
    start = source.index('{', source.index(signature))
    depth = 1
    for end in range(start + 1, len(source)):
        depth += (source[end] == '{') - (source[end] == '}')
        if depth == 0:
            return source[start + 1:end]
    raise AssertionError(f'unclosed function: {signature}')


class CoordinatorRevisitSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (NATIVE / 'src/MarkerGraphCoordinator.cc').read_text()
        cls.header = (NATIVE / 'include/MarkerGraphCoordinator.h').read_text()

    def test_raw_window_contains_no_copied_pose_or_scale_sample(self):
        window = body(self.header, 'struct CornerRevisitWindow')
        self.assertNotIn('Sophus', window)
        self.assertNotIn('ScaleSample', window)
        self.assertNotIn('visualTwc', window)
        self.assertIn('std::set<int> markerIds', window)

    def test_dense_gap_and_map_epoch_invalidation_remain(self):
        frame = body(self.source, 'void MarkerGraphCoordinator::OnFrameEnd')
        self.assertIn('constexpr double kDenseMarkerGapS = .30;', self.source)
        # A revisit window may retain sparse marker evidence across a short
        # observation gap; the dense window must still clear when that
        # retention predicate is false.  Check the semantic guard rather than
        # a pre-refactor one-line spelling of the condition.
        self.assertIn('gap>kDenseMarkerGapS && !retain', frame)
        self.assertIn('samples_.clear();', frame)
        self.assertIn('if(observedMap_!=map)', frame)
        self.assertIn('if(observedCorrectionEpoch_!=correctionEpoch)', frame)
        self.assertGreaterEqual(frame.count('samples_.clear(); cornerRevisit_=CornerRevisitWindow();'), 2)
        estimate = body(self.source, 'MarkerGraphCoordinator::ScaleEvidence MarkerGraphCoordinator::EstimateScale')
        self.assertIn('mixed_map_correction_epochs', estimate)

    def test_revisit_admission_retains_current_observation_and_baseline_gates(self):
        selected = body(self.source, 'std::vector<unsigned long> MarkerGraphCoordinator::SelectCornerRevisit')
        self.assertIn('evidence.observations<8', selected)
        self.assertIn('evidence.baselineM<.04', selected)
        self.assertIn('window.correctionEpoch!=window.map->GetLastBigChangeIdx()', selected)
        self.assertIn('if(!anchorIds.count(id)) continue;', selected)
        self.assertIn('StrongIds(k).count(id)', selected)
        self.assertIn('if(result.size()>=2) return result;', selected)

    def test_revisit_fallback_only_selects_candidates_with_weak_unit_prior(self):
        frame = body(self.source, 'void MarkerGraphCoordinator::OnFrameEnd')
        fallback = body(frame, 'if(b.size()<2)')
        self.assertIn('SelectCornerRevisit(evidence,cornerRevisit_,anchorA,strong)', fallback)
        self.assertIn('revisitDriven=true', fallback)
        weak_prior = body(frame, 'if(cornerDriven)')
        self.assertIn('evidence.metricPerVisual=1.; evidence.sigma=.1;', weak_prior)
        self.assertNotIn('SetPose', frame)
        self.assertNotIn('CommitScale', frame)


@unittest.skipUnless(os.environ.get('RUN_NATIVE_SLAM_TESTS') == '1', 'explicit rebuilt native run')
class CoordinatorRevisitNativeTests(unittest.TestCase):
    def run_native(self, *arguments):
        binary = NATIVE / 'Examples/Monocular/marker_graph_coordinator_regression'
        self.assertTrue(binary.is_file(), 'build marker_graph_coordinator_regression first')
        result = subprocess.run([str(binary), *map(str, arguments)],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def test_same_id_map_epoch_motion_freshness_guards(self):
        output = self.run_native('--corner-revisit-only')
        self.assertIn('"corner_revisit_window_guards":true', output)

    def test_actual_runtime_scheduler_does_not_run_ba_or_replace_pose(self):
        import numpy as np
        from aruco_track.models import Calibration
        from aruco_track.orbslam3_backend import write_orbslam3_settings

        with tempfile.TemporaryDirectory() as temporary:
            settings = Path(temporary) / 'camera.yaml'
            calibration = Calibration(np.array([[500., 0, 320], [0, 500., 240], [0, 0, 1]]),
                                      np.zeros(5), (640, 480))
            write_orbslam3_settings(settings, calibration, 30)
            output = self.run_native('--corner-revisit-runtime', settings)
        self.assertIn('"runtime_corner_revisit_cases":8,"ba_runs":0,"commits":0', output)


if __name__ == '__main__':
    unittest.main()
