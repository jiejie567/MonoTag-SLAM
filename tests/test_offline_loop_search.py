"""Lifecycle contracts for the opt-in native post-tracking candidate pass."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1] / 'third_party/ORB_SLAM3'


class OfflineLoopSearchTests(unittest.TestCase):
    def test_staged_loop_validates_before_fusing(self):
        code = (ROOT / 'src/LoopClosing.cc').read_text()
        body=code[code.index('bool LoopClosing::CorrectLoop()'):code.index('bool LoopClosing::MergeLocal()')]
        self.assertLess(body.index('ProposeVisualLoop'),body.index('CommitRefine'))
        self.assertLess(body.index('CommitRefine'),body.index('pair.first->Replace'))
        self.assertLess(body.index('pair.first->Replace'),body.index('mpCurrentKF->AddLoopEdge'))
        optimizer=(ROOT / 'src/MarkerGraphOptimizer.cc').read_text()
        body=optimizer[optimizer.index('MarkerGraphOptimizer::Proposal MarkerGraphOptimizer::ProposeVisualLoop('):
                       optimizer.index('MarkerGraphOptimizer::Proposal MarkerGraphOptimizer::Reanchor(')]
        self.assertIn('useCommittedAdmission=true',body)
        self.assertIn('validate(result,originalObservations',body)
        self.assertNotIn('excludedBackgroundKeyframes.insert',body)

    def test_metric_free_scale_cannot_mutate_live_map(self):
        code = (ROOT / 'src/LoopClosing.cc').read_text()
        experiment = code[code.index('bool MetricLoopScaleExperiment('):
                          code.index('struct OfflineCandidateDiagnostic')]
        self.assertIn('ProposeVisualLoop', experiment)
        self.assertIn('return (offline || incrementalOffline)', experiment)
        guard = code[code.index('bool LoopClosing::RejectUnsupportedCorrection('):
                     code.index('void LoopClosing::Run()')]
        self.assertIn('std::isfinite(mg2oLoopScw.scale())', guard)
        self.assertIn('std::abs(mg2oLoopScw.scale()-1.0)<=1e-6', guard)
        self.assertIn('metric_sim3_requires_staged_validation', guard)
        correction = code[code.index('bool LoopClosing::CorrectLoop()'):
                          code.index('bool LoopClosing::MergeLocal()')]
        self.assertLess(correction.index('RejectUnsupportedCorrection'),
                        correction.index('mpCurrentKF->SetPose'))
        self.assertLess(correction.index('RejectUnsupportedCorrection'),
                        correction.index('SearchAndFuse'))

    def test_refined_projection_retains_optimized_sim3(self):
        code = (ROOT / 'src/LoopClosing.cc').read_text()
        body = code[code.index('bool LoopClosing::DetectAndReffineSim3FromLastKF('):
                    code.index('bool LoopClosing::DetectCommonRegionsFromBoW(')]
        self.assertIn('gScw_estimation = gScm * gSwm.inverse()', body)
        self.assertNotIn('gScw.rotation(), gScw.translation(),1.0', body)

    def test_search_runs_after_workers_before_final_marker_ba(self):
        code = (ROOT / 'src/System.cc').read_text()
        shutdown = code[code.index('void System::Shutdown()'):code.index('bool System::isShutDown()')]
        self.assertLess(shutdown.index('mpLoopCloser->isFinished()'),
                        shutdown.index('RunOfflineLoopSearch()'))
        self.assertLess(shutdown.index('RunOfflineLoopSearch()'),
                        shutdown.index('ProcessMarkerGraph(true)'))

    def test_opt_in_audit_and_separate_index(self):
        code = (ROOT / 'src/LoopClosing.cc').read_text()
        body = code[code.index('void LoopClosing::RunOfflineLoopSearch()'):
                    code.index('bool LoopClosing::NewDetectCommonRegions()')]
        self.assertIn('if(mode!="1" && mode!="audit") return;', body)
        self.assertIn('KeyFrameDatabase database(*mpORBVocabulary)', body)
        self.assertNotIn('mpKeyFrameDB->add', body)
        self.assertLess(body.index('database.DetectNBestCandidates'), body.index('database.add(query)'))
        self.assertIn('if(mode=="1")', body)
        self.assertIn('committed=CorrectLoop()', body)
        self.assertIn('while(isRunningGBA())', body)

    def test_native_thresholds_remain_unchanged(self):
        code = (ROOT / 'src/LoopClosing.cc').read_text()
        for text in ['nBoWMatches = 20', 'nBoWInliers = 15', 'nSim3Inliers = 20',
                     'nProjMatches = 50', 'nProjOptMatches = 80',
                     'return nNumCoincidences >= 3;']:
            self.assertIn(text, code)


if __name__ == '__main__':
    unittest.main()
