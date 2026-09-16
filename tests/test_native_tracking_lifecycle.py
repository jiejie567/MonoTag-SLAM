"""Focused native pause/timestamp regressions; no camera or vocabulary file."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / 'third_party/ORB_SLAM3'


def function_body(source, signature):
    start = source.index('{', source.index(signature))
    depth = 1
    end = start + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start + 1:end - 1]


class NativeTrackingLifecycleContracts(unittest.TestCase):
    """Also run without a native build, checking the cancellation call sites."""

    def test_flow_recovery_can_seed_unset_pose_after_graph_commit(self):
        source = (NATIVE / 'src/Tracking.cc').read_text()
        body = function_body(source, 'void Tracking::Track()')
        start = body.index('if(mbFlowRecoveryEnabled &&')
        recovery = body[start:body.index('if(!bOK)\n                cout', start)]
        self.assertIn('!bOK && mLastFrame.HasPose()', recovery)
        self.assertIn('TryTemporalFlowRecovery(mnMatchesInliers)', recovery)
        trial = function_body(source, 'bool Tracking::TryTemporalFlowRecovery(')
        self.assertLess(trial.index('const Frame original=mCurrentFrame'),
                        trial.index('UpdateLastFrame();'))
        self.assertLess(trial.index('mCurrentFrame.SetPose(mLastFrame.GetPose())'),
                        trial.index('TrackWithTemporalFlow(true,source,image)'))
        self.assertIn('TrackLocalMap(true)', trial)
        self.assertIn('mCurrentFrame=original; mnMatchesInliers=originalInliers;', trial)

    def test_incremental_offline_loop_search_is_past_only(self):
        source = (NATIVE / 'src/LoopClosing.cc').read_text()
        body = function_body(source, 'bool LoopClosing::NewDetectCommonRegions()')
        self.assertIn('ORB_SLAM3_INCREMENTAL_LOOP_SEARCH', body)
        self.assertIn('enhancedRecall ? 12 : 3', body)
        self.assertIn('incremental && IsStagedMetricLoop(currentMap,currentMap)', body)
        self.assertIn('!currentMap->IsInertial()', body)
        self.assertIn('if(vpMergeBowCand.size()>3) vpMergeBowCand.resize(3);', body)
        self.assertIn('past->mnId>=mpCurrentKF->mnId', body)
        self.assertIn('if(++extra>=32) break', body)
        self.assertIn('DetectCommonRegionsFromBoW(vpLoopBowCand', body)
        self.assertIn('(mbOfflineLoopSearch || incremental)', source)
        runner = (NATIVE / 'Examples/Monocular/mono_tum_headless.cc').read_text()
        self.assertNotIn('setenv("ORB_SLAM3_INCREMENTAL_LOOP_SEARCH","1",0)', runner)
        self.assertIn('incrementalLoopSearch && !slam.WaitForLoopClosingIdle(30000)', runner)
        self.assertLess(runner.index('slam.WaitForLoopClosingIdle(30000)'),
                        runner.index('slam.SaveReplaySnapshot(history,timestamps[index]'))

    def test_map_change_and_resets_cancel_alignment_before_waiting(self):
        source = (NATIVE / 'src/Tracking.cc').read_text()
        for signature, boundary in (
            ('void Tracking::CreateMapInAtlas()', 'mpAtlas->CreateNewMap()'),
            ('void Tracking::Reset(bool', 'mpLocalMapper->RequestReset()'),
            ('void Tracking::ResetActiveMap(bool', 'mpLocalMapper->RequestResetActiveMap('),
        ):
            with self.subTest(signature=signature):
                body = function_body(source, signature)
                self.assertIn('CancelPendingTagAlignment();', body)
                self.assertLess(body.index('CancelPendingTagAlignment();'), body.index(boundary))

    def test_alignment_uses_its_own_non_destructive_pause(self):
        source = (NATIVE / 'src/Tracking.cc').read_text()
        body = function_body(source, 'bool Tracking::TryAlignMapToTagWorld()')
        self.assertIn('RequestTagAlignmentStop()', body)
        self.assertIn('isStoppedForTagAlignment()', body)
        self.assertIn('CancelPendingTagAlignment();', body)
        self.assertNotIn('mpLocalMapper->Release()', body)

    def test_alignment_is_rejected_before_commit_when_marker_poses_disagree(self):
        source = (NATIVE / 'src/Tracking.cc').read_text()
        body = function_body(source, 'bool Tracking::TryAlignMapToTagWorld()')
        self.assertIn('positionResiduals', body)
        self.assertIn('rotationResiduals', body)
        self.assertIn('mTagMaxAlignmentPositionResidualM', body)
        self.assertIn('mTagMaxAlignmentRotationResidualDeg', body)
        self.assertLess(body.index('positionResiduals'), body.index('RequestTagAlignmentStop()'))
        self.assertLess(body.index('Tag metric alignment rejected'),
                        body.index('MarkerGraphOptimizer::InitializeMetric'))
        self.assertLess(body.index('MarkerGraphOptimizer::InitializeMetric'),
                        body.index('item.first->SetPose(item.second)'))

    def test_metric_alignment_requires_scale_invariant_visual_parallax(self):
        source = (NATIVE / 'src/Tracking.cc').read_text()
        body = function_body(source, 'bool Tracking::TryAlignMapToTagWorld()')
        self.assertIn('visualBaseline / medianSceneDepth', body)
        self.assertIn('mTagMinimumVisualBaselineDepthRatio', body)
        self.assertIn('evidenceSpan < 0.25', body)
        self.assertIn('mvTagScaleSamples.push_back(newest)', body)
        self.assertIn('Tag metric alignment waiting for visual parallax', body)
        self.assertLess(body.index('visualBaseline / medianSceneDepth'),
                        body.index('mPendingTagMetricScale = scale'))

    def test_local_ba_validates_metric_marker_residual_before_commit(self):
        source = (NATIVE / 'src/Optimizer.cc').read_text()
        self.assertIn('!pKF->GetMap()->mbMetric', source)
        body = function_body(source, 'void Optimizer::LocalBundleAdjustment(')
        self.assertIn('AddFixedTagProjectionEdges(optimizer, pKFi, thHuberMono, true)', body)
        self.assertIn('priorResidual>100.0', source)
        self.assertIn('FixedTagReprojectionRms', body)
        self.assertIn('LM-LBA rejected: metric marker reprojection', body)
        self.assertLess(body.index('FixedTagReprojectionRms'),
                        body.index('// Get Map Mutex'))

    def test_previous_frame_gate_precedes_marker_map_selection(self):
        source = (NATIVE / 'src/Tracking.cc').read_text()
        body = function_body(source, 'void Tracking::Track()')
        self.assertIn('std::isfinite(mLastFrame.mTimeStamp)', body)
        self.assertLess(body.index('std::isfinite(mLastFrame.mTimeStamp)'),
                        body.index('mState=MARKER_TRACKING'))
        self.assertIn('if(hasPreviousFrame)', body)

    def test_loss_cancels_before_marker_fallback(self):
        source = (NATIVE / 'src/Tracking.cc').read_text()
        body = function_body(source, 'void Tracking::Track()')
        self.assertIn('if(!bOK)\n            CancelPendingTagAlignment();', body)

    def test_metric_coverage_keyframe_remains_bounded(self):
        source = (NATIVE / 'src/Tracking.cc').read_text()
        body = function_body(source, 'bool Tracking::NeedNewKeyFrame()')
        self.assertIn('measuredInliers < 80', body)

    def test_unscaled_high_rate_map_has_bounded_keyframe_proposals(self):
        source = (NATIVE / 'src/Tracking.cc').read_text()
        body = function_body(source, 'bool Tracking::NeedNewKeyFrame()')
        self.assertIn('arbitraryScaleMinimumGap', body)
        self.assertIn('std::lround(0.10f * mMaxFrames)', body)
        self.assertLess(body.index('if(markerEvent)'),
                        body.index('arbitraryScaleMinimumGap'))

    def test_metric_motion_model_has_one_bounded_wide_search(self):
        source = (NATIVE / 'src/Tracking.cc').read_text()
        body = function_body(source, 'bool Tracking::TrackWithMotionModel()')
        self.assertIn('mSensor==System::MONOCULAR', body)
        self.assertIn('mbMetric', body)
        self.assertIn('4*th', body)
        self.assertIn('? 15 : 20', body)

    def test_metric_scale_does_not_weaken_visual_support(self):
        body = function_body((NATIVE / 'src/Tracking.cc').read_text(),
                             'bool Tracking::TrackLocalMap(bool')
        self.assertNotIn('mbMetric ? 15 : 30', body)
        self.assertIn('const int minimumInliers = 30;', body)

    def test_local_ba_uses_stable_id_order(self):
        body = function_body((NATIVE / 'src/Optimizer.cc').read_text(),
                             'void Optimizer::LocalBundleAdjustment(')
        self.assertIn('lLocalKeyFrames.sort(KeyFrame::lId)', body)
        self.assertIn('lLocalMapPoints.sort(', body)
        self.assertIn('lFixedCameras.sort(KeyFrame::lId)', body)
        self.assertIn('orderedObservations', body)

    def test_offline_mapper_wait_is_deterministic_and_bounded(self):
        source = (NATIVE / 'Examples/Monocular/mono_tum_headless.cc').read_text()
        self.assertNotIn('if(slam.GetTrackingState()==2) {', source)
        self.assertIn('slam.WaitForLocalMappingIdle(1000)', source)
        self.assertIn('local_mapping_wait_timeouts', source)
        self.assertNotIn('slam.GetTrackingState()==2 && tracked < 120', source)
        system = (NATIVE / 'src/System.cc').read_text()
        body = function_body(system, 'bool System::WaitForLocalMappingIdle(')
        self.assertIn('isStoppedForTagAlignment()', body)
        self.assertIn('isFinished()', body)
        self.assertIn('WaitUntilKeyFramesProcessed(maxMilliseconds)', body)
        local = (NATIVE / 'src/LocalMapping.cc').read_text()
        self.assertIn('mnPendingKeyFrames.fetch_add(1', local)
        self.assertIn('MarkKeyFrameProcessed();', local)
        self.assertIn('mKeyFrameCompletion.wait_for(', local)

    def test_mapper_ready_is_published_before_completion_notification(self):
        body = function_body((NATIVE / 'src/LocalMapping.cc').read_text(),
                             'void LocalMapping::Run(')
        self.assertLess(body.index('if(CheckNewKeyFrames()'),
                        body.index('SetAcceptKeyFrames(false)'))
        self.assertLess(body.index('SetAcceptKeyFrames(true)'),
                        body.index('if(processedKeyFrame) MarkKeyFrameProcessed()'))
        self.assertEqual(body.count('MarkKeyFrameProcessed()'), 1)

    def test_tracking_inputs_do_not_depend_on_pointer_addresses(self):
        map_source = (NATIVE / 'src/Map.cc').read_text()
        self.assertIn('sort(keyframes.begin(),keyframes.end(),KeyFrame::lId)', map_source)
        self.assertIn('return first->mnId<second->mnId', map_source)
        keyframe = (NATIVE / 'src/KeyFrame.cc').read_text()
        self.assertGreaterEqual(keyframe.count('first.second->mnId>second.second->mnId'), 2)
        tracking = function_body((NATIVE / 'src/Tracking.cc').read_text(),
                                 'void Tracking::UpdateLocalKeyFrames()')
        self.assertIn('orderedKeyframeVotes', tracking)
        self.assertIn('orderedChildren', tracking)
        map_point = function_body((NATIVE / 'src/MapPoint.cc').read_text(),
                                  'void MapPoint::ComputeDistinctiveDescriptors()')
        self.assertIn('orderedObservations', map_point)
        self.assertIn('first.first->mnId<second.first->mnId', map_point)
        normal = function_body((NATIVE / 'src/MapPoint.cc').read_text(),
                               'void MapPoint::UpdateNormalAndDepth()')
        self.assertIn('orderedObservations', normal)

    def test_offline_shutdown_drains_loop_queue_before_publishing_atlas(self):
        system = (NATIVE / 'src/System.cc').read_text()
        body = function_body(system, 'void System::Shutdown()')
        self.assertLess(body.index('mpLocalMapper->RequestFinish()'),
                        body.index('mpLoopCloser->RequestFinish()'))
        self.assertIn('while(!mpLocalMapper->isFinished())', body)
        loop = (NATIVE / 'src/LoopClosing.cc').read_text()
        run = function_body(loop, 'void LoopClosing::Run()')
        self.assertIn('!CheckNewKeyFrames() && !isRunningGBA()', run)
        runner = (NATIVE / 'Examples/Monocular/mono_tum_headless.cc').read_text()
        self.assertIn('offline_finalization_converged', runner)
        self.assertIn('ORB_SLAM3_FINALIZE_ONLY', runner)

    def test_final_marker_ba_ignores_only_preexisting_background_outliers(self):
        source = (NATIVE / 'src/MarkerGraphOptimizer.cc').read_text()
        overload = source.index('Map* map, const Options& options)',
                                source.index('MarkerGraphOptimizer::RefineMetricMap('))
        filter_gate = source.index('staged.filterInitialBackgroundOutliers = true;',
                                   overload)
        initialize_metric = source.index('MarkerGraphOptimizer::InitializeMetric(',
                                         overload)
        self.assertLess(filter_gate, initialize_metric)
        refine_end = initialize_metric
        refine = source[overload:refine_end]
        self.assertIn('excludedBackgroundKeyframes', refine)
        self.assertIn('maximumExcluded', refine)
        self.assertIn('!finalBackgroundPolicy && retry<3', refine)
        self.assertIn('staged.excludedBackgroundObservations.insert(rejected.begin(),rejected.end())', refine)
        self.assertIn('supportedMarkers<2', refine)
        self.assertIn('rejected.size()*20<=retained.background.size()', refine)
        self.assertIn('MARKER_FINAL_FEATURE_RETRY', refine)

    def test_final_marker_ba_preserves_gauge_and_grandfathers_only_existing_bad_frames(self):
        source = (NATIVE / 'src/MarkerGraphOptimizer.cc').read_text()
        self.assertIn('if(input.fixedKeyframes.count(observer))', source)
        self.assertIn('vertex->setFixed(true);', source)
        gate = (NATIVE / 'include/BackgroundResidualGate.h').read_text()
        self.assertIn('backgroundNormalizedRmsByKeyframe', source)
        self.assertIn('BackgroundResidualFrameConsistent(before->second,value.second)', source)
        self.assertIn('!(before<=3.0 && after>3.0)', gate)
        self.assertIn('after<=before+.5', gate)
        self.assertIn('FinalBackgroundResidualFrameConsistent', source)
        self.assertIn('after<=3.0 || BackgroundResidualFrameConsistent(before,after)', gate)
        self.assertIn(
            '1e-3f*std::max(1.0f, referenceRay.norm())', source
        )

    def test_visual_merge_metricizes_only_into_existing_marker_world(self):
        source = (NATIVE / 'src/LoopClosing.cc').read_text()
        self.assertIn('const bool arbitraryIntoMetric=!current->mbMetric', source)
        self.assertIn('!current->HasMetricTagGeometry()', source)
        self.assertIn('ShareConsistentMetricMarker(current,other)', source)
        self.assertIn('std::abs(std::log(mSold_new.scale()))<=0.03', source)
        self.assertIn('mSold_new.translation().norm()<=0.10', source)
        self.assertIn('mpAtlas->mMarkerMapAliases[retiredMapId]=metricMapId', source)
        self.assertIn('if(metricTargetMerge)', source)

    def test_metric_visual_loop_cannot_rescale_the_committed_gauge(self):
        source = (ROOT / 'third_party/ORB_SLAM3/src/LoopClosing.cc').read_text()
        self.assertIn('(!moderateMode || std::string(moderateMode)!="0")', source)
        self.assertIn('moderateSupport=cells.size()>=4', source)
        self.assertIn('return nNumCoincidences >= 3', source)
        self.assertIn('bool bFixedScale = mbFixScale || markerGauge;', source)
        self.assertGreaterEqual(
            source.count('bool bFixedScale = mbFixScale || metricGauge;'), 3
        )
        self.assertIn(
            'Optimizer::OptimizeSim3(mpCurrentKF, pMostBoWMatchesKF, vpMatchedMP, gScm, 10, bFixedScale',
            source,
        )
        self.assertNotIn(
            'Optimizer::OptimizeSim3(mpCurrentKF, pKFi, vpMatchedMP, gScm, 10, mbFixScale',
            source,
        )
        self.assertIn("Metric scale changes belong to", source)

    def test_offline_runner_fixes_robust_estimator_randomness(self):
        source = (NATIVE / 'Examples/Monocular/mono_tum_headless.cc').read_text()
        self.assertIn('std::srand(0);', source)
        self.assertIn('cv::setRNGSeed(0);', source)
        self.assertIn('cv::setNumThreads(1);', source)

    def test_no_replay_batch_uses_compact_native_history(self):
        runner = (NATIVE / 'Examples/Monocular/mono_tum_headless.cc').read_text()
        backend = (ROOT / 'aruco_track/orbslam3_backend.py').read_text()
        exporter = (ROOT / 'export_action_labels.py').read_text()
        self.assertIn('ORB_SLAM3_COMPACT_HISTORY', runner)
        self.assertIn('compact_history: bool = False', backend)
        self.assertIn('compact_history=not slam_replay', exporter)

    def test_replay_snapshot_uses_atomic_archived_frame_reference_pair(self):
        source = (NATIVE / 'src/System.cc').read_text()
        body = function_body(source, 'void System::SaveReplaySnapshot(')
        self.assertIn('mlRelativeFramePoses.back()', body)
        self.assertIn('mlpReferences.back()', body)
        self.assertIn('mlReferenceUnitScales.back()', body)
        self.assertIn('mlFrameTimes.back()-timestamp', body)
        self.assertIn('effectiveReference*correctedRelative.inverse()', body)
        self.assertIn('hasArchivedRelative ? archivedRelative', body)
        self.assertLess(body.index('mlRelativeFramePoses.back()'),
                        body.index('publishedWorldFromCamera'))

    def test_tracking_resolves_culled_reference_and_preserves_scale_stamp(self):
        source = (NATIVE / 'src/Tracking.cc').read_text()
        body = function_body(source, 'void Tracking::UpdateLastFrame()')
        self.assertIn('GetReplayReference', body)
        self.assertIn('currentReferenceScale/mlReferenceUnitScales.back()', body)
        self.assertIn('worldFromReference.inverse()', body)
        coordinator = (NATIVE / 'src/MarkerGraphCoordinator.cc').read_text()
        self.assertIn('*referenceScale=newScale;', coordinator)

    def test_offline_runner_registers_non_covisible_marker_components(self):
        source = (NATIVE / 'Examples/Monocular/mono_tum_headless.cc').read_text()
        self.assertIn('tagComponentToWorld', source)
        self.assertIn('ComponentKey componentKey(inputMapId,tag.component)', source)
        self.assertIn('outputMapId==inputMapId', source)
        self.assertIn('Tcw.inverse()*rawTag.Twc.inverse()', source)
        self.assertIn('pendingTagComponentTransforms.size()>=3', source)
        self.assertIn('through continuous metric SLAM trajectory', source)
        self.assertIn('MarkerComponentTransformsConsistent(', source)
        self.assertNotIn('candidate*pendingTagComponentTransforms.front().inverse()', source)

    def test_offline_runner_metricizes_a_healthy_unscaled_map_before_starting_a_new_map(self):
        source = (NATIVE / 'Examples/Monocular/mono_tum_headless.cc').read_text()
        metricize = source.index(
            'slam.GetTrackingState()==2 && !slam.IsTagMetricAligned()'
        )
        new_map = source.index('slam.StartNewMapForMarkerComponent()', metricize)
        self.assertLess(metricize, new_map)
        self.assertIn(
            'by metricizing the continuously tracked map', source
        )

    def test_unobservable_metricization_falls_back_to_a_separate_metric_map(self):
        source = (NATIVE / 'Examples/Monocular/mono_tum_headless.cc').read_text()
        self.assertIn('metricizingTagObservations>=3', source)
        self.assertIn('timestamps[index]-metricizingTagStartTime>=2.0', source)
        self.assertIn('Marker metricization timed out without observable visual scale', source)
        self.assertIn('preserving arbitrary map', source)
        self.assertIn('slam.StartNewMapForMarkerComponent();', source)

    def test_local_mapping_resets_account_for_discarded_pending_keyframes(self):
        source = (NATIVE / 'src/LocalMapping.cc').read_text()
        reset = source[source.index('void LocalMapping::ResetIfRequested()'):
                       source.index('void LocalMapping::RequestFinish()',
                                    source.index('void LocalMapping::ResetIfRequested()'))]
        self.assertGreaterEqual(reset.count('ClearQueuedKeyFrames();'), 2)
        self.assertGreaterEqual(source.count('MarkKeyFrameProcessed(clearedKeyFrames);'), 3)

    def test_marker_graph_never_dereferences_retired_history_keyframes(self):
        source = (NATIVE / 'src/MarkerGraphCoordinator.cc').read_text()
        self.assertIn('std::set<KeyFrame*> liveKeyFrames;', source)
        self.assertIn('!liveKeyFrames.count(kf)', source)
        self.assertIn('Eigen::aligned_allocator<ReferenceEntry>', source)


@unittest.skipUnless(os.environ.get('RUN_NATIVE_SLAM_TESTS') == '1', 'explicit native integration run')
class NativeTrackingLifecycleTests(unittest.TestCase):
    def test_owned_pause_cancellation_commit_and_first_frame_recovery(self):
        binary = NATIVE / 'Examples/Monocular/tracking_lifecycle_regression'
        self.assertTrue(binary.is_file(), 'build target tracking_lifecycle_regression first')
        with tempfile.TemporaryDirectory() as temporary:
            settings = Path(temporary) / 'camera.yaml'
            settings.write_text('''%YAML:1.0
File.version: "1.0"
Camera.type: "PinHole"
Camera1.fx: 500.0
Camera1.fy: 500.0
Camera1.cx: 320.0
Camera1.cy: 240.0
Camera1.k1: 0.0
Camera1.k2: 0.0
Camera1.p1: 0.0
Camera1.p2: 0.0
Camera1.k3: 0.0
Camera.fx: 500.0
Camera.fy: 500.0
Camera.cx: 320.0
Camera.cy: 240.0
Camera.k1: 0.0
Camera.k2: 0.0
Camera.p1: 0.0
Camera.p2: 0.0
Camera.k3: 0.0
Camera.width: 640
Camera.height: 480
Camera.fps: 30.0
Camera.RGB: 0
ORBextractor.nFeatures: 200
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7
TagFusion.enabled: 1
''')
            result = subprocess.run([str(binary), str(settings)], capture_output=True,
                                    text=True, check=True, timeout=20)
        report = json.loads(result.stdout.splitlines()[-1])
        for key in (
            'default_frame_timestamp_invalid', 'tag_release_keeps_queue',
            'tag_release_keeps_external_stop', 'external_release_keeps_tag_stop',
            'cancel_before_stop_keeps_queue', 'finish_exits_tag_stop',
            'reset_and_finish_work_while_externally_stopped',
            'map_change_cancels_tag_stop', 'alignment_commit_releases_tag_stop',
            'external_stop_defers_alignment_commit', 'marker_recovery_ignores_absent_previous_frame',
            'real_backwards_timestamp_still_rejected',
            'single_marker_small_residual_is_graph_only',
            'multi_marker_requires_three_consistent_frames',
            'strong_fixed_anchor_reobserved_requests_one_information_keyframe',
        ):
            with self.subTest(check=key):
                self.assertTrue(report[key], report)


if __name__ == '__main__':
    unittest.main()
