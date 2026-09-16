"""Integration contracts; numerical matching tests live in marker_graph_regression."""
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]/'third_party/ORB_SLAM3'

class PostLoopRematchTests(unittest.TestCase):
    def test_after_graph_before_bundle_adjustment(self):
        source=(ROOT/'src/LoopClosing.cc').read_text()
        body=source[source.index('bool LoopClosing::CorrectLoop()'):
                    source.index('bool LoopClosing::MergeLocal()')]
        self.assertLess(body.index('Optimizer::OptimizeEssentialGraph('),body.index('PostLoopRematch::Run('))
        self.assertLess(body.index('AddLoopEdge('),body.index('PostLoopRematch::Run('))
        self.assertLess(body.index('PostLoopRematch::Run('),body.index('LaunchGlobalBundleAdjustment('))
        self.assertIn('ORB_SLAM3_POST_LOOP_REMATCH',body)

    def test_audit_has_no_native_fuse_side_effect(self):
        body=(ROOT/'src/PostLoopRematch.cc').read_text()
        self.assertNotIn('matcher.Fuse(',body)
        self.assertIn('intoB[j]!=x',body)
        self.assertIn('proposals.size()<15',body)
        self.assertIn('cellsA.size()<4 || cellsB.size()<4',body)
        self.assertLess(body.index('if(apply)'),body.index('drop->Replace(keep)'))

    def test_saved_experiment_requires_loaded_atlas_and_loop_edge(self):
        body=(ROOT/'src/System.cc').read_text()
        self.assertIn('!mStrLoadAtlasFromFile.empty()',body)
        self.assertIn('for(KeyFrame* other:kf->GetLoopEdges())',body)

if __name__=='__main__':
    unittest.main()
