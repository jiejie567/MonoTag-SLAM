// Exercise real native pause/Tracking transitions without a vocabulary file.
#include "Atlas.h"
#include "Frame.h"
#include "KeyFrame.h"
#include "KeyFrameDatabase.h"
#include "LocalMapping.h"
#include "Tracking.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <new>
#include <stdexcept>
#include <thread>

using namespace ORB_SLAM3;


class MapperProbe : public LocalMapping {
public:
    explicit MapperProbe(Atlas* atlas=nullptr) : LocalMapping(nullptr,atlas,true,false) {
        // Model a running worker at its safe point; do not launch any thread.
        mbFinished=false;
    }
    void QueuePlaceholder() { mlNewKeyFrames.push_back(nullptr); }
    void Finish() { RequestFinish(); SetFinish(); }
};

class TrackerProbe : public Tracking {
public:
    TrackerProbe(Atlas* atlas, ORBVocabulary* vocabulary, KeyFrameDatabase* database,
                 MapperProbe* mapper, const std::string& settings)
        : Tracking(nullptr,vocabulary,nullptr,nullptr,atlas,database,settings,
                   System::MONOCULAR,nullptr) {
        SetLocalMapper(mapper);
        SetViewer(nullptr);
        mpReferenceKF=nullptr;
        mbf=0;
        mThDepth=1;
        // Pose-gate probes deliberately have no background observations.
        // Default Frame leaves these fields unset; do not inherit a prior
        // fixture's feature count or camera pointers from reused stack bytes.
        for(Frame* frame:{&mCurrentFrame,&mLastFrame}) {
            frame->N=0;
            frame->Nleft=frame->Nright=-1;
            frame->mpCamera=mpCamera;
            frame->mpCamera2=nullptr;
        }
    }
    void BeginAlignment(float scale=2.0f) {
        mbTagAlignmentPending=true;
        mpTagAlignmentMap=mpAtlas->GetCurrentMap();
        mPendingTagWorldFromSlamWorld=Sophus::SE3f();
        mPendingTagMetricScale=scale;
        mCurrentFrame.mnId=10;
        mCurrentFrame.SetPose(Sophus::SE3f());
        mpLocalMapper->RequestTagAlignmentStop();
    }
    bool CommitAlignment() { return TryAlignMapToTagWorld(); }
    bool AlignmentPending() const { return mbTagAlignmentPending; }
    bool RejectInstantTagJump(const Sophus::SE3f& visualTwc,
                              const Sophus::SE3f& markerTwc,
                              const std::vector<Eigen::Vector3f>& world,
                              const std::vector<cv::Point2f>& pixels,
                              const std::vector<float>& weights,
                              const std::vector<int>& ids) {
        mpAtlas->GetCurrentMap()->mbMetric=true;
        mbTagMetricAligned=true;
        mState=OK;
        mCurrentFrame.mnId=42;
        mCurrentFrame.SetPose(visualTwc.inverse());
        SetExternalTagObservation(markerTwc,1.0f,world,pixels,true,weights,ids);
        const unsigned int before=mnTagPoseConstraintsApplied;
        ApplyExternalTagPoseConstraint();
        const Sophus::SE3f published=mCurrentFrame.GetPose().inverse();
        return mnTagPoseConstraintsApplied==before &&
            !CurrentPoseHasTagConstraint() &&
            mMarkerTrackingStatus.accepted &&
            !mMarkerTrackingStatus.poseConstraintApplied &&
            mMarkerTrackingStatus.poseConstraintReason=="deferred_to_marker_graph" &&
            (published.translation()-visualTwc.translation()).norm()<1e-6f;
    }
    bool RejectMarkerOnlySingleTagSpike(
            const Sophus::SE3f& previousTwc,
            const Sophus::SE3f& markerTwc,
            const std::vector<Eigen::Vector3f>& world,
            const std::vector<cv::Point2f>& pixels,
            const std::vector<float>& weights,
            const std::vector<int>& ids) {
        mpAtlas->GetCurrentMap()->mbMetric=true;
        mbTagMetricAligned=true;
        mState=MARKER_TRACKING;
        mLastFrame.SetPose(previousTwc.inverse());
        mvExternalTagIds=std::vector<int>(4,20);
        mvExternalTagPointWeights=std::vector<float>(4,1.0f);
        mvExternalTagWorldPoints=world;
        SetExternalTagObservation(markerTwc,.45f,world,pixels,true,weights,ids);
        return !mbHasExternalTagObservation && !mMarkerTrackingStatus.accepted &&
            mMarkerTrackingStatus.reason=="single_marker_motion_gate";
    }
    bool DeferSmallSingleMarkerConstraint(
            const Sophus::SE3f& visualTwc,
            const Sophus::SE3f& markerTwc,
            const std::vector<Eigen::Vector3f>& world,
            const std::vector<cv::Point2f>& pixels,
            const std::vector<float>& weights,
            const std::vector<int>& ids) {
        mpAtlas->GetCurrentMap()->mbMetric=true;
        mbTagMetricAligned=true;
        mState=OK;
        mCurrentFrame.mnId=60;
        mCurrentFrame.SetPose(visualTwc.inverse());
        SetExternalTagObservation(markerTwc,1.0f,world,pixels,true,weights,ids);
        const unsigned int before=mnTagPoseConstraintsApplied;
        ApplyExternalTagPoseConstraint();
        return mnTagPoseConstraintsApplied==before &&
            mMarkerTrackingStatus.accepted &&
            !mMarkerTrackingStatus.poseConstraintApplied &&
            mMarkerTrackingStatus.poseConstraintReason=="single_marker_graph_only";
    }
    bool ConfirmStableMultiMarkerConstraint(
            const Sophus::SE3f& Twc,
            const std::vector<Eigen::Vector3f>& world,
            const std::vector<cv::Point2f>& pixels,
            const std::vector<float>& weights,
            const std::vector<int>& ids) {
        mpAtlas->GetCurrentMap()->mbMetric=true;
        mbTagMetricAligned=true;
        mState=OK;
        mnMatchesInliers=100;
        const unsigned int before=mnTagPoseConstraintsApplied;
        for(int frame=0;frame<3;++frame) {
            mCurrentFrame.mnId=70+frame;
            mCurrentFrame.SetPose(Twc.inverse());
            SetExternalTagObservation(Twc,1.0f,world,pixels,true,weights,ids);
            ApplyExternalTagPoseConstraint();
            if(frame<2 && mMarkerTrackingStatus.poseConstraintReason!="marker_set_unconfirmed")
                return false;
        }
        return mnTagPoseConstraintsApplied==before+1 &&
            mMarkerTrackingStatus.poseConstraintApplied &&
            mMarkerTrackingStatus.poseConstraintReason=="fused";
    }
    bool FuseMultiMarkerWhenVisualSupportIsWeak(
            const Sophus::SE3f& Twc,
            const std::vector<Eigen::Vector3f>& world,
            const std::vector<cv::Point2f>& pixels,
            const std::vector<float>& weights,
            const std::vector<int>& ids) {
        mpAtlas->GetCurrentMap()->mbMetric=true;
        mbTagMetricAligned=true;
        mState=OK;
        mnMatchesInliers=31;
        mCurrentFrame.mnId=80;
        mCurrentFrame.SetPose(Twc.inverse());
        SetExternalTagObservation(Twc,1.0f,world,pixels,true,weights,ids);
        const unsigned int before=mnTagPoseConstraintsApplied;
        ApplyExternalTagPoseConstraint();
        return mnTagPoseConstraintsApplied==before+1 &&
            mMarkerTrackingStatus.poseConstraintApplied &&
            mMarkerTrackingStatus.poseConstraintReason=="fused_low_visual_support";
    }
    bool FuseSingleMarkerOnlyDuringBackgroundRecovery(
            const Sophus::SE3f& Twc,
            const std::vector<Eigen::Vector3f>& world,
            const std::vector<cv::Point2f>& pixels,
            const std::vector<float>& weights,
            const std::vector<int>& ids) {
        mpAtlas->GetCurrentMap()->mbMetric=true;
        mbTagMetricAligned=true;
        mState=MARKER_TRACKING;
        mnMatchesInliers=200;
        mCurrentFrame.mnId=81;
        const Sophus::SE3f visualTwc(
            Eigen::Matrix3f::Identity(),Eigen::Vector3f(.04f,0,0));
        mCurrentFrame.SetPose(visualTwc.inverse());
        SetExternalTagObservation(Twc,1.0f,world,pixels,true,weights,ids);
        const unsigned int before=mnTagPoseConstraintsApplied;
        ApplyExternalTagPoseConstraint();
        return mnTagPoseConstraintsApplied==before+1 &&
            mMarkerTrackingStatus.poseConstraintApplied &&
            mMarkerTrackingStatus.poseConstraintReason=="fused_marker_recovery";
    }
    bool RejectDistantThreeMarkerOverride(
            const Sophus::SE3f& Twc,
            const std::vector<Eigen::Vector3f>& world,
            const std::vector<cv::Point2f>& pixels,
            const std::vector<float>& weights,
            const std::vector<int>& ids) {
        mpAtlas->GetCurrentMap()->mbMetric=true;
        mbTagMetricAligned=true;
        mState=OK;
        mnMatchesInliers=300;
        mCurrentFrame.mnId=82;
        const Sophus::SE3f visualTwc(
            Eigen::Matrix3f::Identity(),Eigen::Vector3f(.08f,0,0));
        mCurrentFrame.SetPose(visualTwc.inverse());
        SetExternalTagObservation(Twc,1.0f,world,pixels,true,weights,ids);
        const unsigned int before=mnTagPoseConstraintsApplied;
        ApplyExternalTagPoseConstraint();
        const Sophus::SE3f published=mCurrentFrame.GetPose().inverse();
        return mnTagPoseConstraintsApplied==before &&
            !mMarkerTrackingStatus.poseConstraintApplied &&
            mMarkerTrackingStatus.poseConstraintReason=="deferred_to_marker_graph" &&
            (published.translation()-visualTwc.translation()).norm()<1e-6f;
    }
    bool FuseThreeMarkersDuringVisualSupportCollapse(
            const Sophus::SE3f& Twc,
            const std::vector<Eigen::Vector3f>& world,
            const std::vector<cv::Point2f>& pixels,
            const std::vector<float>& weights,
            const std::vector<int>& ids) {
        mpAtlas->GetCurrentMap()->mbMetric=true;
        mbTagMetricAligned=true;
        mState=OK;
        mnMatchesInliers=72;
        mCurrentFrame.mnId=83;
        const Sophus::SE3f visualTwc(
            Eigen::Matrix3f::Identity(),Eigen::Vector3f(.04f,0,0));
        mCurrentFrame.SetPose(visualTwc.inverse());
        SetExternalTagObservation(Twc,1.0f,world,pixels,true,weights,ids);
        const unsigned int before=mnTagPoseConstraintsApplied;
        ApplyExternalTagPoseConstraint();
        return mnTagPoseConstraintsApplied==before+1 &&
            mMarkerTrackingStatus.poseConstraintApplied &&
            mMarkerTrackingStatus.poseConstraintReason==
                "fused_strong_marker_visual_rescue";
    }
    bool CheckAnchorReobservedKeyFrameEvent() {
        Map* map=mpAtlas->GetCurrentMap();
        map->mbMetric=map->mbMarkerSeed=map->mbBackgroundReady=true;
        mbTagMetricAligned=mbTagFusionEnabled=true;
        mState=OK;
        cv::Mat image(480,640,CV_8UC1),distortion=cv::Mat::zeros(4,1,CV_32F);
        cv::RNG rng(812);rng.fill(image,cv::RNG::UNIFORM,0,256);
        Frame base(image,0,mpORBextractorLeft,mpORBVocabulary,mpCamera,distortion,0,1);
        base.SetPose(Sophus::SE3f());
        std::map<int,std::vector<Eigen::Vector3f>> corners;
        std::map<int,std::vector<cv::Point2f>> pixels;
        for(int id:{49,50,51})
            for(const Eigen::Vector2f& offset:std::vector<Eigen::Vector2f>{
                    {-.024f,-.024f},{.024f,-.024f},{.024f,.024f},{-.024f,.024f}}) {
                const Eigen::Vector3f point(offset.x()+.15f*(id-49),offset.y(),.5f);
                corners[id].push_back(point);
                const Eigen::Vector2f pixel=mpCamera->project(point);
                pixels[id].emplace_back(pixel.x(),pixel.y());
                if(id!=51)
                    for(int axis=0;axis<3;++axis) map->mStaticTags[id].push_back(point[axis]);
            }
        std::vector<std::unique_ptr<KeyFrame>> keyframes;
        const auto recordGeometry=[&](Frame& frame,int id) {
            keyframes.emplace_back(new KeyFrame(frame,map,mpKeyFrameDB));
            KeyFrame* keyframe=keyframes.back().get();
            keyframe->mbHasTagObservation=keyframe->mbTagObservationActive=true;
            keyframe->mTagObservationConfidence=1.f;
            keyframe->mvTagIds.assign(4,id);keyframe->mvTagPointWeights.assign(4,1.f);
            keyframe->mvTagWorldPoints=corners.at(id);keyframe->mvTagImagePoints=pixels.at(id);
            map->AddKeyFrame(keyframe);
            return keyframe;
        };
        KeyFrame* origin=recordGeometry(base,49);
        map->mvpKeyFrameOrigins.push_back(origin);
        Frame later(base);later.mnId=base.mnId+1;later.mTimeStamp=.01;
        recordGeometry(later,50); // Registered/keyframed, but not the origin marker.
        recordGeometry(later,51); // Keyframed but not registered in the metric map.
        mpReferenceKF=origin;
        mLastFrame=base;
        bool passed=true;
        const auto check=[&](bool ok,const char* message) {
            if(!ok) std::cerr << "anchor_reobserved regression: " << message << std::endl;
            passed=passed && ok;
        };
        const auto observe=[&](double time,int id,bool valid=true,bool partial=false,bool weak=false) {
            mCurrentFrame=base;
            mCurrentFrame.mnId=100+unsigned(time*1024);
            mCurrentFrame.mTimeStamp=time;
            SetExternalTagObservation(Sophus::SE3f(),1.f,corners.at(id),pixels.at(id),valid,
                std::vector<float>(4,(partial||weak)?.25f:1.f),std::vector<int>(4,id),
                partial,partial?4:0,partial?.05f:0.f,true);
            UpdateMarkerKeyFrameEvents();
            check(mState==OK,"event altered successful ORB tracking state");
        };
        observe(0,50); // Prime previously keyframed IDs for long-gap checks below.
        check(!HasMarkerKeyFrameEvent(),"registered non-origin produced a first-seen event");
        observe(.0625,51);
        check(!HasMarkerKeyFrameEvent(),"unregistered keyframed ID produced a first-seen event");
        observe(.125,49);observe(.25,49);
        check(!HasMarkerKeyFrameEvent(),"continuously decoded anchor requested another KF");
        observe(.375,49,false);observe(.5,49);
        check(!HasMarkerKeyFrameEvent(),"short decoding flicker requested a KF");
        observe(.75,49,false);observe(1.,49); // Exactly 0.5 s since the last full decode.
        check(HasMarkerKeyFrameEvent() && mPendingMarkerEvents.count(49) &&
              mPendingMarkerEvents.at(49)=="anchor_reobserved","fixed anchor did not request a reobserved KF");
        check(HasMarkerKeyFrameEvent() && mPendingMarkerEvents.count(49),"query consumed the pending event");
        observe(1.0625,49,false);
        check(mPendingMarkerEvents.count(49),"a missing decode discarded an unrecorded event");
        observe(1.125,49);
        check(HasMarkerKeyFrameEvent() && mPendingMarkerEvents.count(49),"a later decode consumed pending evidence");
        KeyFrame* recorded=recordGeometry(mCurrentFrame,49);
        RecordMarkerKeyFrame(recorded);
        check(!mPendingMarkerEvents.count(49) && !HasMarkerKeyFrameEvent() &&
              mMarkerKeyFrameEvent=="anchor_reobserved:49" &&
              mnMarkerEventKeyFrameId==long(recorded->mnId),"actual KF record did not consume/journal the event once");
        observe(1.25,49);observe(1.375,49);
        check(!HasMarkerKeyFrameEvent(),"continuous visibility duplicated a consumed event");
        observe(1.5,49,false);observe(2.,49,true,true);
        check(!HasMarkerKeyFrameEvent() && !mPendingMarkerEvents.count(49),"partial corners requested an anchor KF");
        observe(2.125,49,true,false,true);
        check(!HasMarkerKeyFrameEvent() && !mPendingMarkerEvents.count(49),"weak corners requested an anchor KF");
        observe(2.25,49);
        check(HasMarkerKeyFrameEvent() && mPendingMarkerEvents.count(49),
              "partial/weak observations incorrectly refreshed the strong-decode clock");
        RecordMarkerKeyFrame(recordGeometry(mCurrentFrame,49));
        observe(2.375,50);
        check(HasMarkerKeyFrameEvent() && mPendingMarkerEvents.count(50) &&
              mPendingMarkerEvents.at(50)=="anchor_reobserved",
              "registered non-origin marker lost its reobserved information event");
        RecordMarkerKeyFrame(recordGeometry(mCurrentFrame,50));
        observe(2.5,51);
        check(!HasMarkerKeyFrameEvent() && !mPendingMarkerEvents.count(51),
              "an unregistered keyframed ID requested a metric-anchor event");
        observe(3.,50);
        check(mPendingMarkerEvents.count(50),"map-switch fixture did not retain old pending evidence");
        const auto registered=map->mStaticTags;
        mpAtlas->CreateNewMap();
        map=mpAtlas->GetCurrentMap();
        map->mbMetric=map->mbMarkerSeed=map->mbBackgroundReady=true;
        map->mStaticTags=registered;
        mpReferenceKF=recordGeometry(base,49);
        map->mvpKeyFrameOrigins.push_back(mpReferenceKF);
        recordGeometry(later,50);
        observe(3.5,49);
        check(!HasMarkerKeyFrameEvent() && mPendingMarkerEvents.empty() &&
              mLastDecodedMarkerTime.size()==1 && !mLastDecodedMarkerTime.count(50),
              "map switch retained an old decoded clock or pending event");
        observe(3.625,50);
        check(!HasMarkerKeyFrameEvent(),"another map's decode gap requested an anchor KF");
        // The probe owns these KFs; detach all Atlas references before deletion.
        mpReferenceKF=nullptr;
        for(const auto& keyframe:keyframes) {
            keyframe->GetMap()->mvpKeyFrameOrigins.clear();
            keyframe->GetMap()->EraseKeyFrame(keyframe.get());
        }
        return passed;
    }
};

struct Fixture {
    Atlas atlas;
    ORBVocabulary vocabulary;
    KeyFrameDatabase database;
    MapperProbe mapper;
    TrackerProbe tracker;
    explicit Fixture(const std::string& settings)
        : atlas(0), database(vocabulary), mapper(&atlas),
          tracker(&atlas,&vocabulary,&database,&mapper,settings) {}
};

int main(int argc, char** argv) {
    if(argc!=2) return 2;
    std::map<std::string,bool> checks;
    // A pre-filled allocation must not accidentally provide a previous time.
    alignas(Frame) unsigned char storage[sizeof(Frame)];
    std::memset(storage,0,sizeof(storage));
    Frame* frame=new(storage) Frame();
    checks["default_frame_timestamp_invalid"]=std::isnan(frame->mTimeStamp);
    frame->~Frame();

    {
        MapperProbe mapper;
        mapper.RequestTagAlignmentStop();
        mapper.Stop();
        mapper.QueuePlaceholder();
        mapper.ReleaseTagAlignmentStop();
        checks["tag_release_keeps_queue"]=mapper.KeyframesInQueue()==1 &&
            !mapper.stopRequested() && !mapper.isStopped();
    }
    {
        MapperProbe mapper;
        mapper.RequestTagAlignmentStop();
        mapper.RequestStop();
        mapper.Stop();
        mapper.QueuePlaceholder();
        mapper.ReleaseTagAlignmentStop();
        checks["tag_release_keeps_external_stop"]=mapper.KeyframesInQueue()==1 &&
            mapper.stopRequested() && mapper.isStopped() && !mapper.isStoppedForTagAlignment();
    }
    {
        MapperProbe mapper;
        mapper.RequestStop();
        mapper.RequestTagAlignmentStop();
        mapper.Stop();
        mapper.Release();
        checks["external_release_keeps_tag_stop"]=mapper.stopRequested() &&
            mapper.isStopped() && mapper.isStoppedForTagAlignment();
        mapper.ReleaseTagAlignmentStop();
    }
    {
        MapperProbe mapper;
        mapper.RequestTagAlignmentStop();
        mapper.QueuePlaceholder();
        mapper.ReleaseTagAlignmentStop();
        checks["cancel_before_stop_keeps_queue"]=!mapper.Stop() &&
            !mapper.isStopped() && mapper.KeyframesInQueue()==1;
    }
    {
        MapperProbe mapper;
        mapper.RequestTagAlignmentStop();
        mapper.Stop();
        mapper.Finish();
        mapper.ReleaseTagAlignmentStop();
        checks["finish_exits_tag_stop"]=mapper.isFinished() && mapper.isStopped() &&
            !mapper.stopRequested();
    }
    {
        // Exercise the real worker's stopped wait, including a concurrent
        // external stop. Resets must be serviced without releasing that stop.
        MapperProbe mapper;
        mapper.RequestTagAlignmentStop();
        mapper.RequestStop();
        std::thread worker([&mapper] { mapper.Run(); });
        while(!mapper.isStopped()) std::this_thread::yield();
        mapper.RequestReset();
        mapper.RequestResetActiveMap(nullptr);
        mapper.ReleaseTagAlignmentStop();
        const bool stillStopped=mapper.isStopped() && mapper.stopRequested();
        mapper.RequestFinish();
        worker.join();
        checks["reset_and_finish_work_while_externally_stopped"]=stillStopped && mapper.isFinished();
    }
    {
        Fixture fixture(argv[1]);
        Map* original=fixture.atlas.GetCurrentMap();
        fixture.tracker.BeginAlignment();
        fixture.mapper.Stop();
        fixture.mapper.QueuePlaceholder();
        fixture.tracker.CreateMapInAtlas();
        checks["map_change_cancels_tag_stop"]=fixture.atlas.GetCurrentMap()!=original &&
            !fixture.tracker.AlignmentPending() && !fixture.mapper.isStopped() &&
            !fixture.mapper.stopRequested() && fixture.mapper.KeyframesInQueue()==1;
    }
    {
        Fixture fixture(argv[1]);
        fixture.tracker.BeginAlignment();
        fixture.mapper.Stop();
        fixture.mapper.QueuePlaceholder();
        // This fixture deliberately has no keyframe/tag gauge, so the metric
        // proposal must fail closed. The lifecycle contract under test is
        // that a settled (rejected) attempt still releases only its own stop
        // and leaves unrelated queued work intact.
        checks["alignment_commit_releases_tag_stop"]=!fixture.tracker.CommitAlignment() &&
            !fixture.atlas.GetCurrentMap()->mbMetric && !fixture.tracker.AlignmentPending() &&
            !fixture.mapper.isStopped() && !fixture.mapper.stopRequested() &&
            fixture.mapper.KeyframesInQueue()==1;
    }
    {
        Fixture fixture(argv[1]);
        fixture.tracker.BeginAlignment();
        fixture.mapper.RequestStop();
        fixture.mapper.Stop();
        const bool deferred=!fixture.tracker.CommitAlignment() &&
            fixture.tracker.AlignmentPending() && !fixture.atlas.GetCurrentMap()->mbMetric;
        fixture.tracker.CreateMapInAtlas();
        checks["external_stop_defers_alignment_commit"]=deferred &&
            fixture.mapper.isStopped() && fixture.mapper.stopRequested();
    }
    {
        Fixture fixture(argv[1]);
        checks["strong_fixed_anchor_reobserved_requests_one_information_keyframe"]=
            fixture.tracker.CheckAnchorReobservedKeyFrameEvent();
    }
    {
        Fixture fixture(argv[1]);
        const Sophus::SE3f markerTwc;
        const Sophus::SE3f visualTwc(
            Eigen::Matrix3f::Identity(),Eigen::Vector3f(.07f,0,0));
        const std::vector<Eigen::Vector3f> world{
            {-.11f,-.05f,1},{-.01f,-.05f,1},{-.01f,.05f,1},{-.11f,.05f,1},
            {.01f,-.05f,1},{.11f,-.05f,1},{.11f,.05f,1},{.01f,.05f,1}};
        const std::vector<cv::Point2f> pixels{
            {265,215},{315,215},{315,265},{265,265},
            {325,215},{375,215},{375,265},{325,265}};
        std::vector<int> ids(4,20); ids.insert(ids.end(),4,21);
        const std::vector<float> weights(8,1.0f);
        checks["multi_marker_cannot_instantly_jump_localized_atlas"]=
            fixture.tracker.RejectInstantTagJump(
                visualTwc,markerTwc,world,pixels,weights,ids);
    }
    {
        Fixture fixture(argv[1]);
        const Sophus::SE3f markerTwc;
        const Sophus::SE3f visualTwc(
            Eigen::Matrix3f::Identity(),Eigen::Vector3f(.005f,0,0));
        const std::vector<Eigen::Vector3f> world{{-.05f,-.05f,1},{.05f,-.05f,1},
                                               {.05f,.05f,1},{-.05f,.05f,1}};
        const std::vector<cv::Point2f> pixels{{295,215},{345,215},{345,265},{295,265}};
        checks["single_marker_small_residual_is_graph_only"]=
            fixture.tracker.DeferSmallSingleMarkerConstraint(
                visualTwc,markerTwc,world,pixels,
                std::vector<float>(4,1.0f),std::vector<int>(4,20));
    }
    {
        Fixture fixture(argv[1]);
        const Sophus::SE3f Twc;
        const std::vector<Eigen::Vector3f> world{
            {-.11f,-.05f,1},{-.01f,-.05f,1},{-.01f,.05f,1},{-.11f,.05f,1},
            {.01f,-.05f,1},{.11f,-.05f,1},{.11f,.05f,1},{.01f,.05f,1}};
        const std::vector<cv::Point2f> pixels{
            {265,215},{315,215},{315,265},{265,265},
            {325,215},{375,215},{375,265},{325,265}};
        std::vector<int> ids(4,20); ids.insert(ids.end(),4,21);
        checks["multi_marker_requires_three_consistent_frames"]=
            fixture.tracker.ConfirmStableMultiMarkerConstraint(
                Twc,world,pixels,std::vector<float>(8,1.0f),ids);
    }
    {
        Fixture fixture(argv[1]);
        const Sophus::SE3f Twc;
        const std::vector<Eigen::Vector3f> world{
            {-.11f,-.05f,1},{-.01f,-.05f,1},{-.01f,.05f,1},{-.11f,.05f,1},
            {.01f,-.05f,1},{.11f,-.05f,1},{.11f,.05f,1},{.01f,.05f,1}};
        const std::vector<cv::Point2f> pixels{
            {265,215},{315,215},{315,265},{265,265},
            {325,215},{375,215},{375,265},{325,265}};
        std::vector<int> ids(4,20); ids.insert(ids.end(),4,21);
        checks["multi_marker_immediately_assists_weak_visual_frame"]=
            fixture.tracker.FuseMultiMarkerWhenVisualSupportIsWeak(
                Twc,world,pixels,std::vector<float>(8,1.0f),ids);
    }
    {
        Fixture fixture(argv[1]);
        const Sophus::SE3f Twc;
        const std::vector<Eigen::Vector3f> world{{-.05f,-.05f,1},{.05f,-.05f,1},
                                               {.05f,.05f,1},{-.05f,.05f,1}};
        const std::vector<cv::Point2f> pixels{{295,215},{345,215},{345,265},{295,265}};
        checks["single_marker_assists_only_marker_to_background_transition"]=
            fixture.tracker.FuseSingleMarkerOnlyDuringBackgroundRecovery(
                Twc,world,pixels,std::vector<float>(4,1.0f),std::vector<int>(4,20));
    }
    {
        Fixture fixture(argv[1]);
        const Sophus::SE3f Twc;
        const std::vector<Eigen::Vector3f> world{
            {-.17f,-.05f,1},{-.07f,-.05f,1},{-.07f,.05f,1},{-.17f,.05f,1},
            {-.05f,-.05f,1},{.05f,-.05f,1},{.05f,.05f,1},{-.05f,.05f,1},
            {.07f,-.05f,1},{.17f,-.05f,1},{.17f,.05f,1},{.07f,.05f,1}};
        const std::vector<cv::Point2f> pixels{
            {235,215},{285,215},{285,265},{235,265},
            {295,215},{345,215},{345,265},{295,265},
            {355,215},{405,215},{405,265},{355,265}};
        std::vector<int> ids(4,20); ids.insert(ids.end(),4,21); ids.insert(ids.end(),4,22);
        checks["three_markers_do_not_override_distant_strong_visual_pose"]=
            fixture.tracker.RejectDistantThreeMarkerOverride(
                Twc,world,pixels,std::vector<float>(12,1.0f),ids);
    }
    {
        Fixture fixture(argv[1]);
        const Sophus::SE3f Twc;
        const std::vector<Eigen::Vector3f> world{
            {-.17f,-.05f,1},{-.07f,-.05f,1},{-.07f,.05f,1},{-.17f,.05f,1},
            {-.05f,-.05f,1},{.05f,-.05f,1},{.05f,.05f,1},{-.05f,.05f,1},
            {.07f,-.05f,1},{.17f,-.05f,1},{.17f,.05f,1},{.07f,.05f,1}};
        const std::vector<cv::Point2f> pixels{
            {235,215},{285,215},{285,265},{235,265},
            {295,215},{345,215},{345,265},{295,265},
            {355,215},{405,215},{405,265},{355,265}};
        std::vector<int> ids(4,20); ids.insert(ids.end(),4,21); ids.insert(ids.end(),4,22);
        checks["three_markers_rescue_bounded_weak_visual_pose"]=
            fixture.tracker.FuseThreeMarkersDuringVisualSupportCollapse(
                Twc,world,pixels,std::vector<float>(12,1.0f),ids);
    }
    {
        Fixture fixture(argv[1]);
        const Sophus::SE3f previousTwc;
        const Sophus::SE3f markerTwc(
            Eigen::Matrix3f::Identity(),Eigen::Vector3f(.06f,0,0));
        const std::vector<Eigen::Vector3f> world{{-.05f,-.05f,1},{.05f,-.05f,1},
                                               {.05f,.05f,1},{-.05f,.05f,1}};
        // Exact projection under the 6 cm translated camera pose.
        const std::vector<cv::Point2f> pixels{{265,215},{315,215},{315,265},{265,265}};
        const std::vector<int> ids(4,27);
        const std::vector<float> weights(4,1.0f);
        checks["marker_only_single_tag_spike_is_invalid_not_stale"]=
            fixture.tracker.RejectMarkerOnlySingleTagSpike(
                previousTwc,markerTwc,world,pixels,weights,ids);
    }
    {
        Fixture fixture(argv[1]);
        const Sophus::SE3f Twc(Eigen::Matrix3f::Identity(),Eigen::Vector3f(0,0,-.5f));
        const std::vector<Eigen::Vector3f> world{{-.024f,-.024f,0},{.024f,-.024f,0},
                                               {.024f,.024f,0},{-.024f,.024f,0}};
        const std::vector<cv::Point2f> pixels{{296,216},{344,216},{344,264},{296,264}};
        const std::vector<int> ids(4,20);
        const std::vector<float> weights(4,1.0f);
        cv::Mat blank=cv::Mat::zeros(480,640,CV_8UC1);
        fixture.tracker.SetExternalTagObservation(Twc,1,world,pixels,true,weights,ids);
        fixture.tracker.GrabImageMonocular(blank,0,"");
        Map* original=fixture.atlas.GetCurrentMap();
        fixture.tracker.CreateMapInAtlas();
        // Model arbitrary old bytes. NO_IMAGES_YET must be checked before the
        // marker-selection branch turns the state into MARKER_TRACKING.
        fixture.tracker.mLastFrame.mTimeStamp=12345.0;
        fixture.tracker.SetExternalTagObservation(Twc,1,world,pixels,true,weights,ids);
        fixture.tracker.GrabImageMonocular(blank,0,"");
        checks["marker_recovery_ignores_absent_previous_frame"]=
            fixture.atlas.GetCurrentMap()==original &&
            fixture.tracker.mState==Tracking::MARKER_TRACKING &&
            fixture.tracker.mCurrentFrame.HasPose();
        fixture.tracker.SetExternalTagObservation(Twc,1,world,pixels,true,weights,ids);
        fixture.tracker.GrabImageMonocular(blank,-0.1,"");
        checks["real_backwards_timestamp_still_rejected"]=
            fixture.atlas.GetCurrentMap()!=original &&
            fixture.tracker.mState==Tracking::NO_IMAGES_YET &&
            !fixture.tracker.mCurrentFrame.HasPose();
    }
    bool passed=true, first=true;
    std::cout << "{";
    for(const auto& check:checks) {
        if(!first) std::cout << ",";
        first=false;
        std::cout << "\"" << check.first << "\":" << (check.second?"true":"false");
        passed=passed && check.second;
    }
    std::cout << "}" << std::endl;
    return passed?0:1;
}
