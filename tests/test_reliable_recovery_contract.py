"""Supplementary source contracts; the executable probe is the behavioral test."""
from pathlib import Path
import unittest

SOURCE = (Path(__file__).resolve().parents[1] / "third_party/ORB_SLAM3/src/Tracking.cc").read_text()


def body(signature):
    start = SOURCE.index("{", SOURCE.index(signature))
    depth = 1
    for index in range(start + 1, len(SOURCE)):
        depth += (SOURCE[index] == "{") - (SOURCE[index] == "}")
        if depth == 0:
            return SOURCE[start + 1:index]
    raise AssertionError("unterminated function: " + signature)


class ReliableRecoveryContracts(unittest.TestCase):
    def test_reset_boundaries_clear_before_retiring_map(self):
        for signature, mutation in (
            ("void Tracking::CreateMapInAtlas()", "mpAtlas->CreateNewMap()"),
            ("void Tracking::Reset(bool", "mpLocalMapper->RequestReset()"),
            ("void Tracking::ResetActiveMap(bool", "mpLocalMapper->RequestResetActiveMap("),
        ):
            with self.subTest(signature=signature):
                function = body(signature)
                self.assertLess(function.index("ClearReliableFlowFrame();"),
                                function.index(mutation))

    def test_cache_identity_checked_under_active_map_lock(self):
        function = body("void Tracking::Track()")
        self.assertLess(function.index("lock(pCurrentMap->mMutexMapUpdate)"),
                        function.index("if(mpReliableFlowMap"))
        self.assertLess(function.index("ClearReliableFlowFrame();"),
                        function.index("TrackMarkerSeed()"))

    def test_cached_frame_has_ids_not_correspondence_pointers(self):
        function = body("void Tracking::UpdateReliableFlowFrame()")
        self.assertIn("point->mnId", function)
        self.assertIn("mReliableFlowFrame.mvpMapPoints.end(),nullptr", function)
        self.assertIn("mReliableFlowFrame.mpReferenceKF=nullptr", function)
        self.assertIn("mReliableFlowImage=mImGray.clone()", function)

    def test_recovery_checks_gauge_and_resolves_live_ids(self):
        function = body("bool Tracking::TryTemporalFlowRecovery(")
        for gate in ("mnReliableFlowMapId", "mnReliableFlowBigChange",
                     "mnReliableFlowGraphSequence", "mbReliableFlowMetric", "mReliableFlowScale"):
            self.assertLess(function.index(gate), function.index("Frame source=mReliableFlowFrame"))
        self.assertIn("map->GetAllMapPoints()", function)
        self.assertIn("live.find(mvReliableFlowPointIds[i])", function)
        self.assertNotIn("mLastFrame=source", function)
        self.assertNotIn("mLastFrame=mReliableFlowFrame", function)

    def test_trial_does_not_update_local_map_or_evidence_statistics(self):
        function = body("bool Tracking::TrackLocalMap(bool")
        self.assertIn("if(!recoveryTrial) UpdateLocalMap();", function)
        self.assertIn("if(!recoveryTrial) mCurrentFrame.mvpMapPoints[i]->IncreaseFound();", function)
        self.assertIn("if(!recoveryTrial) mpLocalMapper->mnMatchesInliers=mnMatchesInliers;", function)
        search = body("void Tracking::SearchLocalPoints(bool")
        self.assertEqual(search.count("if(!recoveryTrial) pMP->IncreaseVisible();"), 2)

    def test_recovery_has_exception_and_scalar_scratch_guards(self):
        function = body("bool Tracking::TryTemporalFlowRecovery(")
        self.assertIn("catch(const cv::Exception&", function)
        self.assertIn("std::unique_ptr<TemporalRecoveryPointScratch>", function)
        self.assertNotIn("end-begin", function)
        self.assertIn("~TemporalRecoveryPointScratch() noexcept", SOURCE)
        for field in ("mTrackProjX", "mTrackProjY", "mTrackDepth", "mTrackDepthR",
                      "mTrackProjXR", "mTrackProjYR", "mbTrackInView", "mbTrackInViewR",
                      "mnTrackScaleLevel", "mnTrackScaleLevelR", "mTrackViewCos",
                      "mTrackViewCosR", "mnTrackReferenceForFrame", "mnLastFrameSeen"):
            self.assertIn("visitor(point->" + field + ");", SOURCE)


if __name__ == "__main__":
    unittest.main()
