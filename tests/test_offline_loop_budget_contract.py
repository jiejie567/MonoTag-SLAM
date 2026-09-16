"""Keep the measured longer solve scoped to verified offline metric loops."""
from pathlib import Path
import unittest


PROJECT = Path(__file__).resolve().parents[1]


class OfflineLoopBudgetContract(unittest.TestCase):
    def test_offline_only_continuous_budget_does_not_change_local_defaults(self):
        source = (PROJECT / 'third_party/ORB_SLAM3/src/LoopClosing.cc').read_text()
        options = (PROJECT / 'third_party/ORB_SLAM3/include/MarkerGraphOptimizer.h').read_text()
        loop = source[source.index('bool LoopClosing::CorrectLoop()'):]
        scoped = loop[loop.index('MarkerGraphOptimizer::Options loopOptions;'):
                      loop.index('reason=proposal.reason;')]
        self.assertIn('if(MetricLoopScaleExperiment(mbOfflineLoopSearch,metricMap,metricMap))', scoped)
        self.assertIn('loopOptions.baIterations=60;', scoped)
        self.assertIn('mvpLoopMatchedMPs,loopOptions)', scoped)
        self.assertNotIn('convergeOffline', scoped)
        self.assertIn('int baIterations = 15;', options)


if __name__ == '__main__':
    unittest.main()
