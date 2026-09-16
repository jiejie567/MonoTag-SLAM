"""Keep reanchor acceptance tied to unchanged measurement membership."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ReanchorObservationContractTests(unittest.TestCase):
    def test_admission_is_frozen_for_baseline_and_seed(self):
        code = (ROOT / 'third_party/ORB_SLAM3/src/MarkerGraphOptimizer.cc').read_text()
        body = code[code.index('MarkerGraphOptimizer::Proposal MarkerGraphOptimizer::Reanchor('):]
        for setting in ('raw.useCommittedAdmission = true',
                        'staged.useCommittedAdmission = true'):
            self.assertIn(setting, body)
            self.assertLess(body.index(setting), body.index('prepare(raw, options, before, observations)'))
        self.assertIn('validate(proposal, observations, options, fixedPoses)', body)
        self.assertNotIn('excludedBackgroundKeyframes.insert', body)

    def test_singletons_follow_retained_observer_before_and_after_ba(self):
        code = (ROOT / 'third_party/ORB_SLAM3/src/MarkerGraphOptimizer.cc').read_text()
        body = code[code.index('MarkerGraphOptimizer::Proposal MarkerGraphOptimizer::Reanchor('):]
        self.assertIn('observations.pointCounts.at(observation.point)!=1', body)
        self.assertIn('KeyFrame* observer=observation.keyframe', body)
        self.assertIn('reference=retained->second.front()->keyframe', body)
        self.assertIn('originalRay*proposal.replayScaleMultipliers.at(reference)', body)

    def test_safety_limits_not_relaxed(self):
        code = (ROOT / 'third_party/ORB_SLAM3/include/MarkerGraphOptimizer.h').read_text()
        for setting in ('maximumBackgroundRmsPx = 3.0',
                        'maximumBackgroundRmsIncreasePx = 0.5',
                        'maximumMarkerCornerDisplacementM = 0.10'):
            self.assertIn(setting, code)

    def test_independent_markers_do_not_inherit_boundary_camera_locks(self):
        code = (ROOT / 'third_party/ORB_SLAM3/src/MarkerGraphOptimizer.cc').read_text()
        self.assertIn('std::make_pair(observer->mnId,marker.first)', code)
        self.assertIn('marker.first==gaugeMarker.second', code)
        self.assertIn('input.rigidMarkerLayout || options.fixAllObservedGaugeMarkers', code)

    def test_pixel_retry_requires_independent_support_and_preserves_population(self):
        code = (ROOT / 'third_party/ORB_SLAM3/src/MarkerGraphOptimizer.cc').read_text()
        body = code[code.index('MarkerGraphOptimizer::Proposal MarkerGraphOptimizer::Reanchor('):]
        for contract in ('supportedMarkers<2', 'raw.fixedKeyframes.count(k) || protectedB.count(k)',
                         'ra.normalized().dot(rb.normalized())<.9998',
                         'local.size()*4<=frameCounts.at(k)',
                         'rejectedPixels.size()*20<=observations.background.size()',
                         'raw.excludedBackgroundObservations=rejectedPixels',
                         'staged.excludedBackgroundObservations=rejectedPixels'):
            self.assertIn(contract, body)

    def test_new_station_stays_provisional_until_a_committed_anchor(self):
        code = (ROOT / 'third_party/ORB_SLAM3/src/MarkerGraphOptimizer.cc').read_text()
        helper = code[code.index('std::map<int,KeyFrame*> provisionalMarkerReferences('):
                      code.index('bool prepare(')]
        for contract in ('map->mbRigidMarkerLayout', 'map->GetMarkerScaleAnchorKFId()',
                         'map->GetAllKeyFrames()', 'k->mnFrameId<first->second->mnFrameId',
                         'k->mnFrameId>acceptedAnchor->mnFrameId',
                         'for(int id:k->mvTagIds) provisional.erase(id)',
                         '!acceptedAnchor || acceptedAnchor->isBad()'):
            self.assertIn(contract, helper)
        self.assertNotIn('cornerScale', helper)
        refine = code[code.index('MarkerGraphOptimizer::Proposal MarkerGraphOptimizer::RefineMetricMap('):
                      code.index('MarkerGraphOptimizer::Proposal MarkerGraphOptimizer::InitializeMetric(',
                                 code.index('MarkerGraphOptimizer::Proposal MarkerGraphOptimizer::RefineMetricMap('))]
        loop = code[code.index('MarkerGraphOptimizer::Proposal MarkerGraphOptimizer::ProposeVisualLoop('):
                    code.index('MarkerGraphOptimizer::Proposal MarkerGraphOptimizer::Reanchor(')]
        reanchor = code[code.index('MarkerGraphOptimizer::Proposal MarkerGraphOptimizer::Reanchor('):]
        for body, call, assignment in (
                (refine, 'provisionalMarkerReferences(map,staged.keyframes)',
                 'staged.provisionalMarkerIds.insert(marker.first)'),
                (loop, 'provisionalMarkerReferences(map,raw.keyframes)',
                 'raw.provisionalMarkerIds.insert(marker.first)'),
                (reanchor, 'provisionalMarkerReferences(map,raw.keyframes,anchorA)',
                 'staged.provisionalMarkerIds.insert(marker.first)')):
            self.assertIn(call, body)
            self.assertIn(assignment, body)
            self.assertLess(body.index(assignment), body.index('RefineAndValidate('))
        self.assertNotIn('if(cornerScale.valid && !map->mbRigidMarkerLayout)', reanchor)

    def test_visual_registered_component_declares_atlas_coordinates(self):
        runner = (ROOT / 'third_party/ORB_SLAM3/Examples/Monocular/mono_tum_headless.cc').read_text()
        coordinator = (ROOT / 'third_party/ORB_SLAM3/src/MarkerGraphCoordinator.cc').read_text()
        self.assertIn('componentsRegisteredInAtlasWorld.insert(ComponentKey(outputMapId,rawTag.component))', runner)
        self.assertIn('tag.trackAgeS, inputInAtlasWorld', runner)
        self.assertIn('inputInAtlasWorld ? Sophus::SE3f() : map->mMarkerInputToWorld', coordinator)


if __name__ == '__main__':
    unittest.main()
