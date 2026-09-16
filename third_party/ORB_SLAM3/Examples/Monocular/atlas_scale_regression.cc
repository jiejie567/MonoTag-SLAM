// Small deterministic regressions; no images, vocabulary loading, or SLAM run.
#include "Atlas.h"
#include "KeyFrame.h"
#include "KeyFrameDatabase.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "Map.h"
#include "Optimizer.h"
#include "Settings.h"
#include "System.h"
#include "Tracking.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace ORB_SLAM3;

static void require(bool ok, const char* message) {
    if(!ok) throw std::runtime_error(message);
}
static bool close(float a, float b) {return std::abs(a-b)<1e-5f;}
static Sophus::SE3f cameraPose(float x) {
    return Sophus::SE3f(Eigen::Matrix3f::Identity(),Eigen::Vector3f(-x,0,0));
}

struct TestKeyFrame : KeyFrame {
    TestKeyFrame(Map* map, KeyFrameDatabase* db, unsigned long id, float x, float scale=1) {
        mnId=id; mpMap=map; mpKeyFrameDB=db; mReplayUnitScale=scale; bImu=false;
        mpCamera=nullptr; mpCamera2=nullptr; mpImuPreintegrated=nullptr;
        SetPose(cameraPose(x));
        SetVelocity(Eigen::Vector3f::Zero());
        map->AddKeyFrame(this);
    }
    void breakParent() {mpParent=nullptr;}
};

struct TestLoopClosing : LoopClosing {
    TestLoopClosing(Atlas* atlas):LoopClosing(atlas,nullptr,nullptr,false,true) {}
    void candidates(KeyFrame* current, KeyFrame* other) {
        mpCurrentKF=current; mpLoopMatchedKF=other; mpMergeMatchedKF=other;
    }
    bool runRejectedMerge() {return MergeLocal();}
    bool runRejectedLoop() {return CorrectLoop();}
    unsigned long registerWorker() {
        std::unique_lock<std::mutex> lock(mMutexGBA);
        return RegisterGBAWorkerLocked();
    }
    void cancelWorker() {CancelGlobalBundleAdjustment();}
    void attachEmptyThreadHandle() {mpThreadGBA=new std::thread;}
    bool hasThreadHandle() {return mpThreadGBA!=nullptr;}
    void runWorker(Map* map, unsigned long generation) {
        RunGlobalBundleAdjustment(map,1234,5.0,generation);
    }
    void runRejectedGBA(Map* map) {
        runWorker(map,registerWorker());
        require(!mbRunningGBA && mbFinishedGBA,"rejected GBA left running state stuck");
    }
    void returnAfterImuChange(Map* map, unsigned long generation) {
        GBAWorkerCompletion completion{this,generation};
        std::unique_lock<std::mutex> gate(mpAtlas->mMutexPoseGraphCorrection);
        std::unique_lock<std::mutex> lock(mMutexGBA);
        // Use the exact production eligibility check and return scope, without
        // running an inertial optimizer or introducing a timing-based race.
        if(!IsGBACommitCurrent(map,false,generation)) return;
        throw std::runtime_error("IMU state change should invalidate GBA");
    }
    bool mergeWithScale(double scale) {
        const auto pose=mpCurrentKF->GetPose().cast<double>();
        mg2oMergeScw=g2o::Sim3(pose.unit_quaternion(),pose.translation(),scale);
        return MergeLocal();
    }
    bool allowedKind(const std::string& kind, Map* current, Map* other) {
        std::unique_lock<std::mutex> gate(mpAtlas->mMutexPoseGraphCorrection);
        return !RejectUnsupportedCorrection(kind,current,other,
                                            mpCurrentKF->mnId,mpCurrentKF->mTimeStamp);
    }
    bool allowed(Map* current, Map* other) {return allowedKind("merge",current,other);}
};

struct StoppedLocalMapping : LocalMapping {
    StoppedLocalMapping(Atlas* atlas):LocalMapping(nullptr,atlas,1.0f,false) {
        // No mapping thread or queued frames: keep the real stop/queue API in
        // its finished-and-stopped state through both merge commit stages.
        mbStopped=true;
    }
};

static void testGuard() {
    Atlas atlas(0);
    Map* first=atlas.GetCurrentMap();
    atlas.CreateNewMap();
    Map* second=atlas.GetCurrentMap();
    ORBVocabulary vocabulary;
    KeyFrameDatabase database(vocabulary);
    TestKeyFrame a(first,&database,0,1), b(second,&database,1,2);
    TestLoopClosing loop(&atlas);
    loop.candidates(&b,&a);
    require(loop.allowed(first,second),"pure monocular merge was disabled");
    first->mbMetric=true;
    first->mStaticTags[20]={0,0,0,.05f,0,0,.05f,.05f,0,0,.05f,0};
    // An arbitrary visual map is now intentionally allowed to merge into a
    // metric destination.  Exercise the actual unsafe case instead: two
    // already-metric maps with unrelated marker gauges must not be joined by
    // a background-only Sim(3).
    second->mbMetric=true;
    second->mStaticTags[21]={0,0,0,.05f,0,0,.05f,.05f,0,0,.05f,0};
    const auto original=first->mStaticTags;
    require(!loop.runRejectedMerge(),"metric merge was not rejected before mutation");
    require(atlas.CountMaps()==2 && atlas.GetCurrentMap()==second,"rejected merge changed Atlas");
    require(first->mStaticTags==original && close(b.GetCameraCenter().x(),2),"rejected merge changed geometry");
    loop.candidates(&a,&a);
    require(loop.allowedKind("loop",first,first),"gauge-preserving metric visual loop was disabled");
    a.mbHasTagObservation=true; a.mbTagObservationActive=true;
    require(loop.allowedKind("loop",first,first),"current marker factor was excluded from joint visual-marker loop");
    first->mbMetric=false; first->mStaticTags.clear();
    second->mbMetric=false; second->mStaticTags.clear();
    a.mbTagObservationActive=false;
    require(!loop.allowed(first,second),"stored tag factors lost their gauge guard");
    loop.runRejectedGBA(first);
    require(close(a.GetCameraCenter().x(),1),"late GBA changed metric map geometry");
    const auto events=loop.GetCorrectionRejections();
    require(events.size()==3 && events[0].kind=="merge" && events[1].kind=="merge" &&
            events[2].kind=="global_ba" && events[2].keyframeId==1234 && events[2].timestamp==5.0,
            "rejections missing or mislabelled");
    require(events[0].reason=="fixed_marker_geometry_requires_joint_optimization",
            "rejection reason missing");
}

static void testCulledReferences() {
    ORBVocabulary vocabulary;
    KeyFrameDatabase database(vocabulary);
    Map map(0);
    // Root predates a metric alignment; middle is inserted after it. They now
    // share metre coordinates, despite different scale stamps (.1 and 1).
    TestKeyFrame root(&map,&database,0,0,.1f);
    TestKeyFrame middle(&map,&database,1,1,1);
    TestKeyFrame leaf(&map,&database,2,2,1);
    middle.ChangeParent(&root); leaf.ChangeParent(&middle);
    leaf.SetBadFlag();
    require(close(leaf.mReplayParentUnitScale,1),"leaf cull stamp incorrect");
    middle.SetBadFlag();
    require(close(middle.mReplayParentUnitScale,.1f),"new KF stamped with its own rather than parent's units");
    Sophus::SE3f world; float scale=0; Map* surviving=nullptr;
    require(leaf.GetReplayReference(world,scale,surviving),"two-edge cull chain failed");
    require(close(world.translation().x(),2) && close(scale,1),"same-unit bad references were scaled twice");
    // Only the root remains in the map. Apply a later true 0.5 scale change.
    map.ApplyScaledRotation(Sophus::SE3f(),.5f,false);
    require(leaf.GetReplayReference(world,scale,surviving),"scaled cull chain failed");
    require(close(world.translation().x(),1) && close(scale,.5f),"two-edge corrections did not propagate");
    require(map.GetAllKeyFrames().size()==1,"culled references should not enter Atlas");
    std::set<GeometricCamera*> cameras;
    map.PreSave(cameras);
    require(leaf.GetReplayReference(world,scale,surviving) && close(world.translation().x(),1),
            "PreSave invalidated live historical references");
    leaf.breakParent();
    require(!leaf.GetReplayReference(world,scale,surviving),"broken chain was treated as valid");
}

static void testEssentialGraphScale() {
    ORBVocabulary vocabulary;
    KeyFrameDatabase database(vocabulary);
    Map map(0);
    TestKeyFrame root(&map,&database,0,0), child(&map,&database,1,2);
    child.ChangeParent(&root);
    for(int iteration=0;iteration<2;++iteration) {
        const float oldX=child.GetCameraCenter().x();
        LoopClosing::KeyFrameAndPose original, corrected;
        for(KeyFrame* kf:map.GetAllKeyFrames()) {
            const auto old=kf->GetPose().cast<double>();
            original[kf]=g2o::Sim3(old.unit_quaternion(),old.translation(),1.0);
            corrected[kf]=g2o::Sim3(old.unit_quaternion(),old.translation(),2.0);
            // CorrectLoop publishes this provisional SE3 first, but must not
            // advance the replay stamp until EssentialGraph commits once.
            kf->SetPose(Sophus::SE3f(old.rotationMatrix().cast<float>(),old.translation().cast<float>()/2));
        }
        std::map<KeyFrame*,std::set<KeyFrame*>> connections;
        Optimizer::OptimizeEssentialGraph(&map,&root,&child,original,corrected,connections,false);
        require(close(child.GetCameraCenter().x(),oldX*.5f),"graph Sim3 pose correction wrong");
        require(close(child.mReplayUnitScale,std::pow(.5f,iteration+1)),"loop scale was missing or applied twice");
    }
}

static void testMetricEssentialGraphKeepsMarkerGauge() {
    ORBVocabulary vocabulary;
    KeyFrameDatabase database(vocabulary);
    Map map(0);
    map.mbMetric=true;
    map.mStaticTags[20]={-.02f,-.02f,1,.02f,-.02f,1,.02f,.02f,1,-.02f,.02f,1};
    TestKeyFrame root(&map,&database,0,0), marker(&map,&database,1,1), current(&map,&database,2,2);
    marker.ChangeParent(&root); current.ChangeParent(&marker);
    Pinhole camera(std::vector<float>{500,500,320,240});
    marker.mpCamera=&camera;
    marker.mbHasTagObservation=true;
    marker.mbTagObservationActive=true;
    marker.mTagObservationConfidence=1.f;
    for(std::size_t index=0;index<map.mStaticTags.at(20).size();index+=3) {
        const Eigen::Vector3f point(map.mStaticTags.at(20)[index],
                                    map.mStaticTags.at(20)[index+1],
                                    map.mStaticTags.at(20)[index+2]);
        const Eigen::Vector2f pixel=camera.project(marker.GetPose()*point);
        marker.mvTagIds.push_back(20);
        marker.mvTagWorldPoints.push_back(point);
        marker.mvTagImagePoints.emplace_back(pixel.x(),pixel.y());
        marker.mvTagPointWeights.push_back(1.f);
    }
    map.mnMarkerScaleAnchorKFId=long(root.mnId);

    LoopClosing::KeyFrameAndPose original, corrected;
    for(KeyFrame* keyframe:map.GetAllKeyFrames()) {
        const auto pose=keyframe->GetPose().cast<double>();
        original[keyframe]=g2o::Sim3(pose.unit_quaternion(),pose.translation(),1.0);
        corrected[keyframe]=original[keyframe];
    }
    // A visual loop proposes a correction only for the untagged branch. The
    // root is the sole exact gauge; the marker keyframe is constrained by its
    // original corner pixels rather than frozen.
    const Sophus::SE3f proposed=cameraPose(2.4f);
    corrected[&current]=g2o::Sim3(proposed.unit_quaternion().cast<double>(),
                                  proposed.translation().cast<double>(),1.0);
    current.SetPose(proposed);
    const auto rootBefore=root.GetPose();
    const auto tagBefore=map.mStaticTags;
    std::map<KeyFrame*,std::set<KeyFrame*>> connections;
    Optimizer::OptimizeEssentialGraph(&map,&root,&current,original,corrected,connections,false);
    require((root.GetPose().matrix()-rootBefore.matrix()).norm()<1e-7f,
            "metric visual loop moved its single gauge keyframe");
    double markerSquaredError=0.;
    for(std::size_t index=0;index<marker.mvTagWorldPoints.size();++index) {
        const Eigen::Vector2f pixel=camera.project(marker.GetPose()*marker.mvTagWorldPoints[index]);
        markerSquaredError+=(pixel-Eigen::Vector2f(marker.mvTagImagePoints[index].x,
                                                    marker.mvTagImagePoints[index].y)).squaredNorm();
    }
    require(std::sqrt(markerSquaredError/marker.mvTagWorldPoints.size())<.1,
            "metric visual loop violated marker corner projection constraints");
    require(map.mStaticTags==tagBefore,"metric visual loop changed physical marker geometry");
    require(close(marker.mReplayUnitScale,1.f),"metric visual loop rescaled marker gauge history");
}

static void testGbaWorkerLifecycle() {
    Atlas atlas(0);
    ORBVocabulary vocabulary;
    KeyFrameDatabase database(vocabulary);
    Map* map=atlas.GetCurrentMap();
    TestKeyFrame root(map,&database,0,1);
    TestLoopClosing loop(&atlas);
    loop.candidates(&root,&root);

    auto old=loop.registerWorker();
    loop.attachEmptyThreadHandle();
    loop.cancelWorker();
    require(!loop.hasThreadHandle(),"cancel left a dangling thread handle");
    loop.cancelWorker(); // Cancelling again must not touch a freed handle.
    require(loop.isRunningGBA(),"cancel reported a still-alive worker as finished");
    loop.runWorker(map,old); // Actual generation-mismatch return.
    require(!loop.isRunningGBA() && loop.isFinishedGBA(),"cancel without replacement stayed running");

    map->mbMetric=true; // Current workers return at the real metric commit guard.
    old=loop.registerWorker(); loop.cancelWorker();
    auto newer=loop.registerWorker();
    loop.runWorker(map,old);
    require(loop.isRunningGBA() && !loop.isFinishedGBA(),"old return cleared newer GBA");
    loop.runWorker(map,newer);
    require(!loop.isRunningGBA(),"last new worker did not finish");

    old=loop.registerWorker(); loop.cancelWorker(); newer=loop.registerWorker();
    loop.runWorker(map,newer);
    require(loop.isRunningGBA() && !loop.isFinishedGBA(),"new return hid older detached worker");
    loop.runWorker(map,old);
    require(!loop.isRunningGBA(),"last old worker did not finish");

    auto imuWorker=loop.registerWorker();
    map->SetImuInitialized();
    loop.returnAfterImuChange(map,imuWorker);
    require(!loop.isRunningGBA() && loop.isFinishedGBA(),"IMU-change early return leaked worker");
}

static void testPureVisualMergeScale() {
    Atlas atlas(0);
    ORBVocabulary vocabulary;
    KeyFrameDatabase database(vocabulary);
    Map* target=atlas.GetCurrentMap();
    TestKeyFrame fixed(target,&database,0,9);
    atlas.CreateNewMap();
    Map* source=atlas.GetCurrentMap();
    TestKeyFrame outside(source,&database,1,90), local(source,&database,2,100);
    local.ChangeParent(&outside);

    // Reuse the bundled camera settings; construct only the tracker, never
    // process an image or load a vocabulary. MergeLocal only needs its sensor.
    const std::string sourceFile=__FILE__;
    const std::string config=sourceFile.substr(0,sourceFile.find_last_of('/'))+"/TUM1.yaml";
    Settings settings(config,System::MONOCULAR);
    Tracking tracker(nullptr,&vocabulary,nullptr,nullptr,&atlas,&database,
                     config,System::MONOCULAR,&settings);
    StoppedLocalMapping mapper(&atlas);
    TestLoopClosing loop(&atlas);
    loop.SetTracker(&tracker); loop.SetLocalMapper(&mapper);
    loop.candidates(&local,&fixed);
    require(loop.mergeWithScale(10.0),"pure visual merge was rejected");
    require(close(local.GetCameraCenter().x(),10) && close(outside.GetCameraCenter().x(),9),
            "merge did not correct both local and outside keyframes");
    require(close(local.mReplayUnitScale,.1f) && close(outside.mReplayUnitScale,.1f),
            "merge replay scale missing or doubled in one commit stage");
    require(close(fixed.GetCameraCenter().x(),9) && close(fixed.mReplayUnitScale,1),
            "merge rescaled the receiving map");
    require(local.GetMap()==target && outside.GetMap()==target &&
            atlas.GetCurrentMap()==target && atlas.CountMaps()==1,
            "merge did not preserve final map ownership");
}

int main() {
    try {
        testGuard();
        testCulledReferences();
        testEssentialGraphScale();
        testMetricEssentialGraphKeepsMarkerGauge();
        testGbaWorkerLifecycle();
        testPureVisualMergeScale();
        std::cout << "ATLAS_SCALE_REGRESSION_OK: guards, culled references, loop/merge scale, GBA lifecycles" << std::endl;
        return 0;
    } catch(const std::exception& error) {
        std::cerr << "ATLAS_SCALE_REGRESSION_FAILED: " << error.what() << std::endl;
        return 1;
    }
}
