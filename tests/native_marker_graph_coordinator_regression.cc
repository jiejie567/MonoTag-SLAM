// Coordinator transaction regressions: no camera, vocabulary file, or SLAM run.
#include "MarkerGraphCoordinator.h"
#include "Optimizer.h"
#include "Atlas.h"
#include "CameraModels/Pinhole.h"
#include "Frame.h"
#include "KeyFrameDatabase.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "MapDrawer.h"
#include "MapPoint.h"
#include "ORBextractor.h"
#include "ORBmatcher.h"
#include <chrono>
#include "Settings.h"
#include "Tracking.h"
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <thread>

using namespace ORB_SLAM3;

static void require(bool ok,const std::string& message) {
    if(!ok) throw std::runtime_error(message);
}
static bool samePose(const Sophus::SE3f& a,const Sophus::SE3f& b,float eps=1e-6f) {
    return (a.matrix()-b.matrix()).norm()<eps;
}
static bool sameGraph(const MarkerGraphTransform& a,const MarkerGraphTransform& b) {
    return a.sequence==b.sequence && std::abs(a.scale-b.scale)<1e-10 &&
        (a.translation-b.translation).norm()<1e-9 &&
        (a.rotation.toRotationMatrix()-b.rotation.toRotationMatrix()).norm()<1e-9;
}
static std::vector<float> square(const Eigen::Vector3f& center,float side) {
    std::vector<float> result;
    for(const auto& offset:std::vector<Eigen::Vector2f>{{-1,-1},{1,-1},{1,1},{-1,1}}) {
        const Eigen::Vector3f p=center+Eigen::Vector3f(offset.x()*side/2,offset.y()*side/2,0);
        for(int axis=0;axis<3;++axis) result.push_back(p(axis));
    }
    return result;
}
static std::vector<float> transformCorners(const std::vector<float>& corners,const Sophus::SE3f& T) {
    std::vector<float> result;
    for(std::size_t i=0;i<corners.size();i+=3) {
        const Eigen::Vector3f p=T*Eigen::Vector3f(corners[i],corners[i+1],corners[i+2]);
        for(int axis=0;axis<3;++axis) result.push_back(p(axis));
    }
    return result;
}
static float side(const std::vector<float>& corners) {
    return (Eigen::Vector3f(corners[3],corners[4],corners[5])-
            Eigen::Vector3f(corners[0],corners[1],corners[2])).norm();
}

struct Fixture {
    Atlas atlas{int(KeyFrame::nNextId)};
    ORBVocabulary vocabulary;
    KeyFrameDatabase database{vocabulary};
    Pinhole camera{std::vector<float>{500,500,320,240}};
    ORBextractor extractor{180,1.2f,8,20,7};
    Map *target=nullptr,*source=nullptr,*unrelated=nullptr;
    Sophus::SE3f sourceToTarget{Sophus::SO3f::exp(Eigen::Vector3f(.13f,-.21f,.07f)),
                              Eigen::Vector3f(.4f,-.2f,.08f)};
    std::unique_ptr<Frame> base;
    std::vector<std::unique_ptr<KeyFrame>> keyframes;
    std::vector<std::unique_ptr<MapPoint>> points;

    bool uniformFeatures=false;
    Fixture(bool uniform=false) : uniformFeatures(uniform) {
        target=atlas.GetCurrentMap();
        atlas.CreateNewMap(); source=atlas.GetCurrentMap();
        atlas.CreateNewMap(); unrelated=atlas.GetCurrentMap();
        for(Map* map:{target,source,unrelated}) {
            map->mbMetric=map->mbMarkerSeed=map->mbBackgroundReady=true;
            map->mMetricScale=1.f;
        }
        target->mStaticTags[20]=square(Eigen::Vector3f(0,0,1),.08f);
        target->mStaticTags[21]=square(Eigen::Vector3f(.2f,.03f,1.05f),.08f);
        for(const auto& tag:target->mStaticTags)
            source->mStaticTags[tag.first]=transformCorners(tag.second,sourceToTarget.inverse());
        source->mStaticTags[22]=transformCorners(square(Eigen::Vector3f(-.2f,.04f,1.1f),.06f),sourceToTarget.inverse());
        unrelated->mStaticTags[99]=square(Eigen::Vector3f(.1f,0,1),.05f);
        cv::Mat image(480,640,CV_8UC1),distortion=cv::Mat::zeros(4,1,CV_32F);
        cv::RNG random(9876); random.fill(image,cv::RNG::UNIFORM,0,256);
        base.reset(new Frame(image,0,&extractor,&vocabulary,&camera,distortion,0,1));
        base->SetPose(Sophus::SE3f());
        require(base->N>=40,"fixture descriptor storage insufficient");
        populate(target,Sophus::SE3f()); populate(source,sourceToTarget);
        populate(unrelated,Sophus::SE3f());
        atlas.AddCamera(&camera);
        atlas.ChangeMap(source);
    }
    ~Fixture() {
        // Atlas deliberately does not own maps it already removed as bad.
        if(source && source->IsBad()) delete source;
        if(unrelated && unrelated->IsBad()) delete unrelated;
    }
    KeyFrame* addEmpty(Map* map,const Sophus::SE3f& Tcw,
                       const std::vector<Eigen::Vector3f>& world={},double timestamp=-1.) {
        Frame frame(*base);
        if(uniformFeatures) for(auto& key:frame.mvKeysUn) {key.octave=0;key.angle=0;}
        frame.mnId=Frame::nNextId++;
        frame.mTimeStamp=timestamp<0?double(frame.mnId)*.1:timestamp;
        frame.SetPose(Tcw);
        for(std::size_t i=0;i<world.size();++i) {
            const auto p=camera.project(Tcw*world[i]);
            frame.mvKeysUn[i]=cv::KeyPoint(cv::Point2f(p.x(),p.y()),1);
        }
        keyframes.emplace_back(new KeyFrame(frame,map,&database));
        auto* kf=keyframes.back().get(); map->AddKeyFrame(kf);
        return kf;
    }
    void populate(Map* map,const Sophus::SE3f& toTarget) {
        std::vector<Eigen::Vector3f> world;
        for(int i=0;i<40;++i)
            world.push_back(toTarget.inverse()*Eigen::Vector3f((i%8-3.5f)*.045f,(i/8-2.f)*.04f,1.2f+.03f*(i%3)));
        std::vector<KeyFrame*> views;
        for(int view=0;view<3;++view) {
            const Sophus::SE3f Tcw=Sophus::SE3f(Eigen::Matrix3f::Identity(),Eigen::Vector3f(-.02f*view,0,0))*toTarget;
            KeyFrame* kf=addEmpty(map,Tcw,world);
            kf->mbHasTagObservation=kf->mbTagObservationActive=true;
            kf->mTagObservationConfidence=1.f;
            for(const auto& tag:map->mStaticTags) for(std::size_t j=0;j<4;++j) {
                const Eigen::Vector3f p(tag.second[j*3],tag.second[j*3+1],tag.second[j*3+2]);
                const auto pixel=camera.project(Tcw*p);
                kf->mvTagIds.push_back(tag.first); kf->mvTagWorldPoints.push_back(p);
                kf->mvTagImagePoints.emplace_back(pixel.x(),pixel.y()); kf->mvTagPointWeights.push_back(1.f);
            }
            if(!views.empty()) kf->ChangeParent(views.back());
            // The fixture has explicitly established this tree. Its three
            // views will receive all shared points at once, so first-connection
            // parent selection would otherwise replace it using pointer ties.
            kf->SetFirstConnection(false);
            views.push_back(kf);
        }
        map->mvpKeyFrameOrigins.push_back(views.front());
        map->mnMarkerScaleAnchorKFId=long(views.front()->mnId);
        for(std::size_t i=0;i<world.size();++i) {
            points.emplace_back(new MapPoint(world[i],views.front(),map));
            auto* point=points.back().get();
            for(auto* kf:views) {point->AddObservation(kf,i); kf->AddMapPoint(point,i);}
            map->AddMapPoint(point);
        }
        for(auto* kf:views) kf->UpdateConnections();
        require(map->GetInitKFid()==views.front()->mnId && views.front()->GetParent()==nullptr,
                "fixture root does not match its map origin");
        for(std::size_t i=1;i<views.size();++i)
            require(views[i]->GetParent()==views[i-1],"fixture covisibility update replaced the explicit parent tree");
    }
};

static double tagReprojectionRms(KeyFrame* keyframe) {
    double squared=0.; std::size_t count=0;
    const Sophus::SE3f Tcw=keyframe->GetPose();
    for(std::size_t index=0;index<keyframe->mvTagWorldPoints.size();++index) {
        const Eigen::Vector3f cameraPoint=Tcw*keyframe->mvTagWorldPoints[index];
        if(cameraPoint.z()<=0) continue;
        const Eigen::Vector3d cameraPointDouble=cameraPoint.cast<double>();
        const Eigen::Vector2d projected=keyframe->mpCamera->project(cameraPointDouble);
        const Eigen::Vector2d measured(keyframe->mvTagImagePoints[index].x,
                                       keyframe->mvTagImagePoints[index].y);
        squared+=(projected-measured).squaredNorm(); ++count;
    }
    return count?std::sqrt(squared/double(count)):std::numeric_limits<double>::infinity();
}

static void testEssentialGraphUsesMarkerCornersWithoutFreezingMarkerKeyframes() {
    for(bool fixedScale : {true,false}) {
    Fixture f; f.atlas.ChangeMap(f.target);
    const auto markerGeometry=f.target->mStaticTags;
    auto keyframes=f.target->GetAllKeyFrames();
    std::sort(keyframes.begin(),keyframes.end(),KeyFrame::lId);
    KeyFrame* gauge=keyframes.front();
    KeyFrame* perturbed=keyframes[1];
    const Sophus::SE3f gaugeBefore=gauge->GetPose();
    const Sophus::SE3f expected=perturbed->GetPose();
    Sophus::SE3f wrong=expected;
    wrong.translation().x()+=.04f;
    perturbed->SetPose(wrong);
    const double before=tagReprojectionRms(perturbed);
    require(before>10.,"essential-graph marker fixture did not create a visible pose error");

    LoopClosing::KeyFrameAndPose emptyPoses;
    std::map<KeyFrame*,std::set<KeyFrame*>> emptyConnections;
    Optimizer::OptimizeEssentialGraph(f.target,gauge,keyframes.back(),emptyPoses,
                                      emptyPoses,emptyConnections,fixedScale);
    const double after=tagReprojectionRms(perturbed);
    require(after<before*.25 && after<2.,
            "marker corner factors did not correct an optimizable marker keyframe: "+
            std::to_string(before)+" -> "+std::to_string(after));
    require(samePose(gauge->GetPose(),gaugeBefore,1e-6f),
            "essential graph moved the single marker gauge keyframe");
    require((perturbed->GetPose().translation()-wrong.translation()).norm()>.02f,
            "marker keyframe remained frozen instead of participating in joint optimization");
    require(f.target->mStaticTags==markerGeometry,"loop changed physical marker corners");
    require(std::abs(gauge->mReplayUnitScale-1.f)<1e-6,"loop rescaled the fixed gauge");
    }
}

// Geometric state, independent of diagnostics. Used before every rejection
// and around the third map during successful commits.
struct MapState {
    Map* map;
    unsigned long revision,graphSequence;
    long anchor;
    bool metric,seed,background,bad;
    float metricScale;
    Sophus::SE3f inputGauge;
    std::map<int,std::vector<float>> tags;
    std::vector<KeyFrame*> keyframes,origins;
    std::vector<MapPoint*> points;
    MarkerGraphOptimizer::PoseMap poses;
    MarkerGraphOptimizer::PointMap positions;
    MarkerGraphOptimizer::TagCornerMap tagCorners;
    std::map<KeyFrame*,float> unitScales;
    using MarkerTransformEntry=std::pair<KeyFrame* const,MarkerGraphTransform>;
    using MarkerTransformMap=std::map<KeyFrame*,MarkerGraphTransform,
        std::less<KeyFrame*>,Eigen::aligned_allocator<MarkerTransformEntry>>;
    MarkerTransformMap graphs,gauges,parentGauges;
    explicit MapState(Map* m):map(m),revision(m->mnRevision),graphSequence(m->mnMarkerGraphSequence),
        anchor(m->mnMarkerScaleAnchorKFId),metric(m->mbMetric),seed(m->mbMarkerSeed),
        background(m->mbBackgroundReady),bad(m->IsBad()),metricScale(m->mMetricScale),
        inputGauge(m->mMarkerInputToWorld),tags(m->mStaticTags),keyframes(m->GetAllKeyFrames()),
        origins(m->mvpKeyFrameOrigins),points(m->GetAllMapPoints()) {
        for(auto* kf:keyframes) {
            poses[kf]=kf->GetPose(); unitScales[kf]=kf->mReplayUnitScale;
            graphs[kf]=kf->mReplayMarkerGraph; tagCorners[kf]=kf->mvTagWorldPoints;
            gauges[kf]=kf->mReplayMarkerGauge; parentGauges[kf]=kf->mReplayParentMarkerGauge;
        }
        for(auto* point:points) positions[point]=point->GetWorldPos();
    }
    void unchanged(const std::string& label) const {
        require(map->mnRevision==revision && map->mnMarkerGraphSequence==graphSequence &&
                map->mnMarkerScaleAnchorKFId==anchor && map->mbMetric==metric &&
                map->mbMarkerSeed==seed && map->mbBackgroundReady==background &&
                map->IsBad()==bad && map->mMetricScale==metricScale,label+": flags/revision changed");
        require(map->mStaticTags==tags && samePose(map->mMarkerInputToWorld,inputGauge),label+": marker geometry changed");
        require(map->GetAllKeyFrames()==keyframes && map->GetAllMapPoints()==points &&
                map->mvpKeyFrameOrigins==origins,label+": ownership set changed");
        for(auto* kf:keyframes) {
            require(kf->GetMap()==map && samePose(kf->GetPose(),poses.at(kf)) &&
                    kf->mReplayUnitScale==unitScales.at(kf) && sameGraph(kf->mReplayMarkerGraph,graphs.at(kf)),
                    label+": keyframe pose/units/graph changed");
            require(sameGraph(kf->mReplayMarkerGauge,gauges.at(kf)) &&
                    sameGraph(kf->mReplayParentMarkerGauge,parentGauges.at(kf)),label+": rigid marker gauge changed");
            require(kf->mvTagWorldPoints.size()==tagCorners.at(kf).size(),label+": corner count changed");
            for(std::size_t i=0;i<kf->mvTagWorldPoints.size();++i)
                require((kf->mvTagWorldPoints[i]-tagCorners.at(kf)[i]).norm()<1e-8f,label+": corner moved");
        }
        for(auto* point:points)
            require(point->GetMap()==map && (point->GetWorldPos()-positions.at(point)).norm()<1e-8f,
                    label+": point position/owner changed");
    }
};

static MarkerGraphOptimizer::Proposal scaleProposal(Map* map,float ratio=.9f) {
    // Test the coordinator's transaction, not the optimizer's independent
    // geometric validator. This is an explicitly already-accepted proposal.
    MarkerGraphOptimizer::Proposal p; p.accepted=true;
    auto kfs=map->GetAllKeyFrames(); std::sort(kfs.begin(),kfs.end(),KeyFrame::lId);
    for(std::size_t i=0;i<kfs.size();++i) {
        auto* kf=kfs[i]; const float local=i==0?1.f:(i==1?(1.f+ratio)/2:ratio);
        auto Twc=kf->GetPoseInverse(); Twc.translation()*=local;
        p.keyframePoses[kf]=Twc.inverse(); p.replayScaleMultipliers[kf]=local;
        p.tagWorldCorners[kf]=kf->mvTagWorldPoints; p.affectedKeyFrameIds.push_back(kf->mnId);
    }
    for(auto* point:map->GetAllMapPoints())
        p.pointPositions[point]=point->GetWorldPos()+Eigen::Vector3f(.001f,0,0);
    p.before.tagRmsPx=1.2; p.after.tagRmsPx=.4;
    p.before.backgroundRmsPx=1.1; p.after.backgroundRmsPx=.3;
    return p;
}

static MarkerGraphCoordinator::ScaleSampleVector scaleSamples(double scale,double baseline=.14) {
    MarkerGraphCoordinator::ScaleSampleVector result;
    for(unsigned long i=0;i<8;++i) {
        MarkerGraphCoordinator::ScaleSample s;
        s.frameId=100+i; s.timestamp=i*.1;
        const float x=float(baseline*i/7);
        s.markerTwc=Sophus::SE3f(Eigen::Matrix3f::Identity(),Eigen::Vector3f(x,0,0));
        s.visualTwc=Sophus::SE3f(Eigen::Matrix3f::Identity(),Eigen::Vector3f(float(x/scale),0,0));
        result.push_back(s);
    }
    return result;
}

static void testScaleEvidence() {
    using C=MarkerGraphCoordinator;
    require(C::CanRetryScale(3,5,1,.6),"new independent geometry cannot retry a rejection");
    require(!C::CanRetryScale(3,3,1,2.),"same rejected graph retried");
    require(!C::CanRetryScale(3,5,1,.1),"retry ignored cooldown");
    require(!C::CanRetryScale(3,5,3,2.),"retry budget ignored");
    C::ScaleEvidence corner;
    corner.reason="inconsistent_or_extreme_scale_observations";
    corner.observations=8; corner.baselineM=.05;
    require(C::CanTryCornerInterval(corner,true,3),"multi-view corner interval blocked by motion-ratio noise");
    require(!C::CanTryCornerInterval(corner,false,3),"non-revisit triggered corner interval");
    require(!C::CanTryCornerInterval(corner,true,2),"two-keyframe corner fallback admitted");
    corner.baselineM=.039;
    require(!C::CanTryCornerInterval(corner,true,3),"insufficient corner baseline admitted");
    corner.baselineM=.05; corner.reason="mixed_map_correction_epochs";
    require(!C::CanTryCornerInterval(corner,true,3),"mixed revisions admitted to corner fallback");
    auto samples=scaleSamples(.85);
    auto e=C::EstimateScale(samples);
    require(e.reliable && e.observations==8 && std::abs(e.metricPerVisual-.85)<1e-6 &&
            e.baselineM>.139 && e.sigma<=.006,"eight independent scale observations were not accepted");
    auto seven=samples; seven.pop_back();
    auto mixed=samples; mixed.back().correctionEpoch=1;
    const auto mixedEvidence=C::EstimateScale(mixed);
    require(!mixedEvidence.geometricallyValid && mixedEvidence.reason=="mixed_map_correction_epochs",
            "pre-loop scale observations mixed with corrected geometry");
    for(auto& sample:mixed) sample.correctionEpoch=1;
    require(C::EstimateScale(mixed).reliable,"fresh post-loop scale observations rejected");
    require(!C::EstimateScale(seven).reliable,"seven images passed the eight-frame gate");
    for(auto& s:samples) s.frameId=100;
    e=C::EstimateScale(samples);
    require(!e.reliable && e.observations==1,"duplicate image IDs counted as independent observations");
    require(!C::EstimateScale(scaleSamples(.85,.039)).reliable,"sub-4cm baseline triggered a correction");
    require(C::EstimateScale(scaleSamples(.85,.041)).reliable,"valid >4cm baseline was discarded");
    e=C::EstimateScale(scaleSamples(.98));
    require(e.geometricallyValid && !e.reliable,
            "a stable near-unit metric anchor was discarded or triggered a drift correction");
    require(C::ShouldScheduleScale(e,true),
            "a new or revisited metric anchor did not request interval refinement");
    require(!C::ShouldScheduleScale(e,false),
            "a continuously visible near-unit anchor requested redundant interval refinement");
    e=C::EstimateScale(scaleSamples(1.));
    require(e.geometricallyValid && !e.reliable && C::ShouldScheduleScale(e,true),
            "a new unit-scale anchor did not remain usable for interval refinement");
    samples=scaleSamples(.85);
    const float noisy[]={0,.01f,.10f,.03f,.12f,.05f,.14f,.07f};
    for(std::size_t i=0;i<samples.size();++i) samples[i].visualTwc.translation().x()=noisy[i];
    e=C::EstimateScale(samples);
    require(!e.reliable && e.sigma>.1,"high-uncertainty scale evidence was accepted");
    samples=scaleSamples(.85);
    samples.back().markerTwc.translation().x()=std::numeric_limits<float>::quiet_NaN();
    require(!C::EstimateScale(samples).reliable,"nonfinite observation passed the independent-frame gate");
}

static void testCornerRevisitWindow() {
    using C=MarkerGraphCoordinator;
    Fixture f;
    auto kfs=f.target->GetAllKeyFrames(); std::sort(kfs.begin(),kfs.end(),KeyFrame::lId);
    KeyFrame* a=kfs[0]; KeyFrame* b=nullptr; KeyFrame* c=nullptr;
    const auto evidence=C::EstimateScale(scaleSamples(1.));
    const auto windowFor=[&](double gap) {
        C::CornerRevisitWindow window;
        b=f.addEmpty(f.target,a->GetPose(),{},1.);
        c=f.addEmpty(f.target,a->GetPose(),{},1.+gap);
        for(KeyFrame* k:{b,c}) {
            k->mbHasTagObservation=k->mbTagObservationActive=true;
            k->mTagObservationConfidence=1.f;
            k->mvTagIds=a->mvTagIds; k->mvTagWorldPoints=a->mvTagWorldPoints;
            k->mvTagImagePoints=a->mvTagImagePoints; k->mvTagPointWeights=a->mvTagPointWeights;
        }
        kfs={a,b,c};
        window.Observe(f.target,b->mnFrameId,b->mTimeStamp,{20});
        window.Observe(f.target,c->mnFrameId,c->mTimeStamp,{20});
        return window;
    };
    for(double gap:{.31,.4,.5,1.}) {
        const auto window=windowFor(gap);
        require(C::SelectCornerRevisit(evidence,window,a,kfs)==
                std::vector<unsigned long>{b->mnId,c->mnId},
                "same-marker raw KFs lost through bounded decode gap "+std::to_string(gap));
    }
    require(C::SelectCornerRevisit(evidence,windowFor(.2),a,kfs).empty(),
            "ordinary continuous scale path was replaced by revisit fallback");
    require(C::SelectCornerRevisit(evidence,windowFor(1.001),a,kfs).empty(),
            "stale raw corner visit crossed maximum gap");
    auto window=windowFor(.4);
    auto invalid=evidence; invalid.observations=7;
    require(C::SelectCornerRevisit(invalid,window,a,kfs).empty(),"seven current frames admitted revisit");
    invalid=evidence; invalid.baselineM=.039;
    require(C::SelectCornerRevisit(invalid,window,a,kfs).empty(),"stationary/sub-4cm visit admitted");
    invalid=evidence; invalid.baselineM=std::numeric_limits<double>::quiet_NaN();
    require(C::SelectCornerRevisit(invalid,window,a,kfs).empty(),"nonfinite motion admitted");
    invalid=evidence; invalid.reason="mixed_map_correction_epochs";
    require(C::SelectCornerRevisit(invalid,window,a,kfs).empty(),"mixed dense epochs admitted");
    require(C::SelectCornerRevisit(evidence,window,a,{b,b}).empty(),"duplicate KF counted twice");
    auto weights=c->mvTagPointWeights; c->mvTagPointWeights.assign(weights.size(),.25f);
    require(C::SelectCornerRevisit(evidence,window,a,kfs).empty(),"weak KF admitted as complete strong revisit");
    c->mvTagPointWeights=weights;
    window.Observe(f.source,c->mnFrameId+1,1.5,{20});
    require(C::SelectCornerRevisit(evidence,window,a,kfs).empty(),"raw visit crossed map switch");
    window.Observe(f.target,c->mnFrameId+2,1.6,{20});
    require(C::SelectCornerRevisit(evidence,window,a,kfs).empty(),"returning map reused pre-switch window");
    window=windowFor(.4); f.target->InformNewBigChange();
    require(C::SelectCornerRevisit(evidence,window,a,kfs).empty(),"stale window survived correction epoch");
    window.Observe(f.target,c->mnFrameId+1,1.5,{20});
    require(!window.fragmented && window.firstFrame==c->mnFrameId+1,"post-correction visit did not restart");
    window=windowFor(.4); window.Observe(f.target,c->mnFrameId+1,1.5,{21});
    require(C::SelectCornerRevisit(evidence,window,a,kfs).empty(),"different marker IDs bridged a visit");
    window=windowFor(.4); window.Observe(f.target,c->mnFrameId+1,1.5,{99});
    require(window.map==nullptr && C::SelectCornerRevisit(evidence,window,a,kfs).empty(),
            "unregistered marker supplied revisit evidence");
    // A newly registered ID is still not an independently anchored origin.
    f.target->mStaticTags[30]=f.target->mStaticTags.at(20);
    for(KeyFrame* k:{b,c}) for(int& id:k->mvTagIds) if(id==20) id=30;
    C::CornerRevisitWindow newId;
    newId.Observe(f.target,b->mnFrameId,1.,{30}); newId.Observe(f.target,c->mnFrameId,1.4,{30});
    require(C::SelectCornerRevisit(evidence,newId,a,kfs).empty(),"new marker borrowed origin authority");
    for(KeyFrame* k:{b,c}) for(int& id:k->mvTagIds) if(id==30) id=20;
    window=windowFor(.4);
    for(int i=1;i<=4;++i) window.Observe(f.target,c->mnFrameId+i,1.4+.7*i,{20});
    require(!window.fragmented && window.firstTime>4.,"raw window exceeded bounded three-second lifetime");
    window=windowFor(.4); window.Observe(f.target,c->mnFrameId+1,
                                      std::numeric_limits<double>::quiet_NaN(),{20});
    require(window.map==nullptr,"nonfinite timestamp retained visit");
    std::cout << "{\"corner_revisit_window_guards\":true,\"dense_pose_storage_added\":false}" << std::endl;
}

static void testScaleCommit() {
    Fixture f; f.atlas.ChangeMap(f.target);
    MapState source(f.source),unrelated(f.unrelated);
    const auto tags=f.target->mStaticTags;
    auto kfs=f.target->GetAllKeyFrames(); std::sort(kfs.begin(),kfs.end(),KeyFrame::lId);
    KeyFrame* a=kfs.front(); const auto anchored=a->GetPose();
    f.atlas.mnMarkerGraphSequence=3;
    kfs.back()->mReplayMarkerGraph.sequence=3;
    kfs.back()->mReplayMarkerGraph.scale=.8;
    kfs.back()->mReplayMarkerGraph.translation=Eigen::Vector3d(.2,-.1,.05);
    kfs.back()->mReplayUnitScale=.8f;
    kfs.back()->mReplayMarkerGauge.sequence=2;
    kfs.back()->mReplayMarkerGauge.translation=Eigen::Vector3d(.03,-.02,.01);
    for(int step=0;step<2;++step) {
        auto p=scaleProposal(f.target,step==0?.9f:.95f);
        const auto oldSequence=f.atlas.mnMarkerGraphSequence;
        MapState before(f.target);
        MarkerGraphEvent event; event.frameId=50+step; event.timestamp=5.+step;
        event.scale=step==0?.9:.95; event.sigma=.005; event.markerIds={20,21};
        std::unique_lock<std::mutex> gate(f.atlas.mMutexPoseGraphCorrection);
        std::unique_lock<std::mutex> lock(f.target->mMutexMapUpdate);
        require(MarkerGraphCoordinator::CommitScale(f.atlas,f.target,p,kfs.back(),event),"accepted scale transaction failed");
        require(event.status=="accepted" && event.sequence==oldSequence+1 && event.type=="scale_reanchor" &&
                f.atlas.mMarkerGraphEvents.back().sequence==event.sequence,"scale commit journal not atomic");
        for(auto* kf:kfs) {
            require(samePose(kf->GetPose(),p.keyframePoses.at(kf)),"scale KF pose was not committed");
            require(std::abs(kf->mReplayUnitScale-before.unitScales.at(kf)*p.replayScaleMultipliers.at(kf))<1e-6,
                    "per-KF historical units applied zero or two times");
            const auto delta=kf->mReplayMarkerGraph*before.graphs.at(kf).inverse();
            require(samePose(delta.apply(before.poses.at(kf).inverse()),kf->GetPoseInverse()),
                    "graph-only accumulator does not map the old camera to the committed camera");
            require(sameGraph(kf->mReplayMarkerGauge,before.gauges.at(kf)) &&
                    sameGraph(kf->mReplayParentMarkerGauge,before.parentGauges.at(kf)),
                    "local scale commit changed a rigid marker-world gauge");
        }
        for(const auto& point:p.pointPositions)
            require((point.first->GetWorldPos()-point.second).norm()<1e-8f,"staged point was not committed");
        require(samePose(a->GetPose(),anchored) && f.target->mStaticTags==tags && f.target->mMetricScale==1.f,
                "scale commit moved A, resized a physical marker, or globally rescaled the metric map");
        require(f.target->mnMarkerScaleAnchorKFId==long(kfs.back()->mnId) &&
                f.target->mnMarkerGraphSequence==event.sequence && event.affectedKeyframes==p.affectedKeyFrameIds &&
                event.afterTagRms==p.after.tagRmsPx,"next anchor/residual/affected-KF publication missing");
    }
    source.unchanged("unmerged source"); unrelated.unchanged("unrelated scale map");
}

static void testRejectedScaleIsAtomic() {
    Fixture f; f.atlas.ChangeMap(f.target);
    const auto original=scaleProposal(f.target);
    auto kfs=f.target->GetAllKeyFrames(); std::sort(kfs.begin(),kfs.end(),KeyFrame::lId);
    Frame frame(*f.base); frame.SetPose(Sophus::SE3f());
    std::unique_ptr<KeyFrame> orphanKF(new KeyFrame(frame,f.target,&f.database));
    orphanKF->UpdateMap(nullptr);  // Map::clear() can leave this state without marking a KF bad.
    std::unique_ptr<MapPoint> orphanMP(new MapPoint(Eigen::Vector3f(0,0,1),kfs.front(),f.target));
    orphanMP->UpdateMap(nullptr);
    for(int variant=0;variant<7;++variant) {
        auto p=original; KeyFrame* next=kfs.back();
        if(variant==0) p.accepted=false;
        if(variant==1) p.keyframePoses[f.unrelated->GetOriginKF()]=Sophus::SE3f();
        if(variant==2) p.pointPositions.begin()->second.x()=std::numeric_limits<float>::quiet_NaN();
        if(variant==3) p.replayScaleMultipliers[kfs.back()]=-.5f;
        if(variant==4) next=f.source->GetOriginKF();
        if(variant==5) p.keyframePoses[orphanKF.get()]=Sophus::SE3f();
        if(variant==6) p.pointPositions[orphanMP.get()]=Eigen::Vector3f(0,0,1);
        MapState target(f.target),source(f.source),unrelated(f.unrelated);
        const auto sequence=f.atlas.mnMarkerGraphSequence; const auto events=f.atlas.mMarkerGraphEvents.size();
        MarkerGraphEvent event;
        std::unique_lock<std::mutex> gate(f.atlas.mMutexPoseGraphCorrection);
        std::unique_lock<std::mutex> lock(f.target->mMutexMapUpdate);
        require(!MarkerGraphCoordinator::CommitScale(f.atlas,f.target,p,next,event),"invalid scale transaction accepted");
        target.unchanged("rejected scale target"); source.unchanged("rejected scale source"); unrelated.unchanged("rejected scale third map");
        require(f.atlas.mnMarkerGraphSequence==sequence && f.atlas.mMarkerGraphEvents.size()==events &&
                f.atlas.GetCurrentMap()==f.target && f.atlas.mMarkerMapAliases.empty(),"rejected scale changed Atlas state");
    }
}

static MarkerMapMerge::Proposal mergeProposal(Fixture& f) {
    auto p=MarkerMapMerge::Propose(f.target,f.source);
    require(p.accepted,"real raw-corner merge fixture rejected: "+p.reason);
    return p;
}
static MarkerGraphEvent commitMerge(Fixture& f,const MarkerMapMerge::Proposal& p) {
    MarkerGraphEvent event; event.frameId=80; event.timestamp=8.;
    std::unique_lock<std::mutex> gate(f.atlas.mMutexPoseGraphCorrection);
    std::unique_lock<std::mutex> targetLock(f.target->mMutexMapUpdate);
    std::unique_lock<std::mutex> sourceLock(f.source->mMutexMapUpdate);
    require(MarkerGraphCoordinator::CommitMerge(f.atlas,f.target,f.source,p,event),"accepted merge transaction failed");
    return event;
}

static void testMarkerGlobalBACommit() {
    Fixture f;
    f.atlas.ChangeMap(f.target);
    auto proposal=MarkerGraphOptimizer::RefineMetricMap(f.target);
    require(proposal.accepted,"marker global BA proposal rejected: "+proposal.reason);
    require(!proposal.optimizedMarkerIds.empty(),"marker global BA exposed no marker variables");
    MapState before(f.target);
    MarkerGraphEvent event; event.frameId=90; event.timestamp=9.;
    {
        std::unique_lock<std::mutex> gate(f.atlas.mMutexPoseGraphCorrection);
        std::unique_lock<std::mutex> lock(f.target->mMutexMapUpdate);
        require(MarkerGraphCoordinator::CommitRefine(f.atlas,f.target,proposal,event),
                "accepted marker global BA transaction failed");
    }
    require(event.status=="accepted" && event.type=="marker_global_ba" &&
            event.markerIds==proposal.optimizedMarkerIds &&
            f.atlas.mMarkerGraphEvents.back().sequence==event.sequence,
            "marker global BA journal incorrect");
    MarkerGraphTransform commonGauge;
    bool haveGauge=false;
    for(KeyFrame* kf:before.keyframes) {
        const auto delta=kf->mReplayMarkerGauge*before.gauges.at(kf).inverse();
        require(delta.scale==1. && delta.sequence==event.sequence,
                "marker global BA history correction is not rigid");
        if(!haveGauge) { commonGauge=delta; haveGauge=true; }
        else require((delta.translation-commonGauge.translation).norm()<1e-9 &&
                     std::abs(std::abs(delta.rotation.dot(commonGauge.rotation))-1.)<1e-9,
                     "marker global BA used a reference-KF-dependent dense gauge");
    }
    require(haveGauge,"marker global BA exposed no dense marker gauge");
    for(const auto& marker:proposal.staticTags) {
        const auto& committed=f.target->mStaticTags.at(marker.first);
        require(committed.size()==12,"marker global BA registry lost corners");
        for(int corner=0;corner<4;++corner) for(int axis=0;axis<3;++axis)
            require(committed[3*corner+axis]==marker.second[corner](axis),
                    "marker global BA registry did not commit proposal");
    }

    auto invalid=proposal;
    invalid.staticTags.begin()->second.front().x()=
        std::numeric_limits<float>::quiet_NaN();
    MapState committedState(f.target);
    const auto sequence=f.atlas.mnMarkerGraphSequence;
    MarkerGraphEvent rejected;
    {
        std::unique_lock<std::mutex> gate(f.atlas.mMutexPoseGraphCorrection);
        std::unique_lock<std::mutex> lock(f.target->mMutexMapUpdate);
        require(!MarkerGraphCoordinator::CommitRefine(f.atlas,f.target,invalid,rejected),
                "invalid marker global BA transaction committed");
    }
    committedState.unchanged("rejected marker global BA");
    require(f.atlas.mnMarkerGraphSequence==sequence,
            "rejected marker global BA consumed a sequence");

    Fixture shifted;
    shifted.atlas.ChangeMap(shifted.target);
    auto gaugeJump=MarkerGraphOptimizer::RefineMetricMap(shifted.target);
    require(gaugeJump.accepted,"gauge-shift fixture did not produce a valid proposal");
    for(auto& marker:gaugeJump.staticTags)
        for(auto& corner:marker.second) corner.x()+=.05f;
    MapState beforeGaugeJump(shifted.target);
    MarkerGraphEvent gaugeRejected;
    {
        std::unique_lock<std::mutex> gate(shifted.atlas.mMutexPoseGraphCorrection);
        std::unique_lock<std::mutex> lock(shifted.target->mMutexMapUpdate);
        require(!MarkerGraphCoordinator::CommitRefine(
                    shifted.atlas,shifted.target,gaugeJump,gaugeRejected),
                "marker global BA redefined the fixed metric world gauge");
    }
    require(gaugeRejected.reason=="marker_gauge_shift_validation_failed",
            "large marker gauge shift reported the wrong rejection reason");
    beforeGaugeJump.unchanged("large marker gauge shift rejection");

    Fixture board;
    board.target->mbRigidMarkerLayout=true;
    auto boardJump=MarkerGraphOptimizer::RefineMetricMap(board.target);
    require(boardJump.accepted,"fixed-board gauge fixture did not produce a valid proposal");
    // Keeping the lowest-ID origin marker alone is not sufficient for a
    // calibrated board: all its markers belong to the same fixed vertex.
    for(auto& corner:boardJump.staticTags.at(21)) corner.x()+=.01f;
    MapState beforeBoardJump(board.target);
    MarkerGraphEvent boardRejected;
    {
        std::unique_lock<std::mutex> gate(board.atlas.mMutexPoseGraphCorrection);
        std::unique_lock<std::mutex> lock(board.target->mMutexMapUpdate);
        require(!MarkerGraphCoordinator::CommitRefine(board.atlas,board.target,boardJump,boardRejected),
                "a displaced non-origin board marker escaped the fixed-board check");
    }
    require(boardRejected.reason=="marker_gauge_shift_validation_failed",
            "fixed-board movement reported the wrong rejection reason");
    beforeBoardJump.unchanged("fixed-board gauge rejection");
}

static void testUnsurveyedOriginMarker() {
    for(float offset:{.012f,.12f,.30f}) {
        Fixture f;
        f.atlas.ChangeMap(f.target);
        const auto truth=f.target->mStaticTags.at(20);
        KeyFrame* origin=f.target->GetOriginKF();
        const auto originPose=origin->GetPose();
        const Sophus::SE3f error(Sophus::SO3f::exp(Eigen::Vector3f(0,.08f,0)),
                                Eigen::Vector3f(offset,0,0));
        for(KeyFrame* k:f.target->GetAllKeyFrames())
            for(std::size_t i=0;i<k->mvTagIds.size();++i)
                if(k->mvTagIds[i]==20) k->mvTagWorldPoints[i]=error*k->mvTagWorldPoints[i];
        for(int j=0;j<4;++j) {
            const Eigen::Vector3f p=error*Eigen::Vector3f(truth[3*j],truth[3*j+1],truth[3*j+2]);
            for(int axis=0;axis<3;++axis) f.target->mStaticTags[20][3*j+axis]=p(axis);
        }
        MapState before(f.target);
        MarkerGraphOptimizer::Options legacy;
        legacy.legacyIndependentMarkerWorldPrior=true;
        const auto locked=MarkerGraphOptimizer::RefineMetricMap(f.target,legacy);
        require(!locked.accepted,"legacy fixture failed to expose a pinned noisy origin marker");
        const auto p=MarkerGraphOptimizer::RefineMetricMap(f.target);
        require(p.accepted,"unsurveyed origin-marker BA rejected: "+p.reason);
        require(p.after.tagRmsPx<.05 && p.after.backgroundRmsPx<.05,
                "free origin marker did not recover the raw geometry");
        double maxError=0;
        for(int j=0;j<4;++j) maxError=std::max(maxError,double((p.staticTags.at(20)[j]-
            Eigen::Vector3f(truth[3*j],truth[3*j+1],truth[3*j+2])).norm()));
        require(maxError<.001,"free origin marker failed to recover its true rigid pose");
        before.unchanged("uncommitted free-marker proposal");
        MarkerGraphEvent event;
        require(MarkerGraphCoordinator::CommitRefine(f.atlas,f.target,p,event),
                "commit incorrectly pinned the origin marker: "+event.reason);
        require(samePose(origin->GetPose(),originPose),"origin-camera gauge moved");
        std::cout << "{\"unsurveyed_offset_m\":" << offset << ",\"max_corner_error_m\":" << maxError
                  << ",\"tag_rms_px\":" << p.after.tagRmsPx << ",\"background_rms_px\":"
                  << p.after.backgroundRmsPx << "}" << std::endl;
    }
    Fixture corrupt;
    corrupt.target->GetOriginKF()->mvTagImagePoints[0].x+=100;
    const auto bad=MarkerGraphOptimizer::RefineMetricMap(corrupt.target);
    require(!bad.accepted,"free marker pose hid a corrupted origin observation");
}

static void testWeakCornersCannotCreateMarkerVariables() {
    Fixture f;
    f.atlas.ChangeMap(f.target);
    for(KeyFrame* keyframe:f.target->GetAllKeyFrames())
        for(std::size_t index=0;index<keyframe->mvTagIds.size();++index)
            if(keyframe->mvTagIds[index]==21) keyframe->mvTagPointWeights[index]=.25f;
    const auto proposal=MarkerGraphOptimizer::RefineMetricMap(f.target);
    require(proposal.accepted,"strong marker plus weak auxiliary factors rejected: "+proposal.reason);
    require(proposal.optimizedMarkerIds==std::vector<int>{20},
            "weak-only observations created a free marker-pose variable");
}

static void testFreeMarkerRefineDoesNotMoveWorldGauge() {
    Fixture f;
    f.atlas.ChangeMap(f.target);
    KeyFrame* origin=f.target->GetOriginKF();
    const auto correctMarker=f.target->mStaticTags.at(21);
    // Marker 20 fixes the origin. Marker 21 was first registered later with
    // an unsurveyed 120 mm world-placement error, not a changed physical size.
    for(std::size_t i=origin->mvTagIds.size();i-- >0;)
        if(origin->mvTagIds[i]==21) {
            origin->mvTagIds.erase(origin->mvTagIds.begin()+i);
            origin->mvTagWorldPoints.erase(origin->mvTagWorldPoints.begin()+i);
            origin->mvTagImagePoints.erase(origin->mvTagImagePoints.begin()+i);
            origin->mvTagPointWeights.erase(origin->mvTagPointWeights.begin()+i);
        }
    for(KeyFrame* k:f.target->GetAllKeyFrames())
        for(std::size_t i=0;i<k->mvTagIds.size();++i)
            if(k->mvTagIds[i]==21) k->mvTagWorldPoints[i].x()+=.12f;
    for(int j=0;j<4;++j) f.target->mStaticTags[21][3*j]+=.12f;
    const auto proposal=MarkerGraphOptimizer::RefineMetricMap(f.target);
    require(proposal.accepted,"free-marker gauge fixture BA failed: "+proposal.reason);
    for(int j=0;j<4;++j)
        require((proposal.staticTags.at(21)[j]-Eigen::Vector3f(correctMarker[3*j],
                    correctMarker[3*j+1],correctMarker[3*j+2])).norm()<.001f,
                "free-marker fixture did not recover the 120 mm position error");
    MapState before(f.target);
    MarkerGraphEvent event;
    {
        std::unique_lock<std::mutex> gate(f.atlas.mMutexPoseGraphCorrection);
        std::unique_lock<std::mutex> lock(f.target->mMutexMapUpdate);
        require(MarkerGraphCoordinator::CommitRefine(f.atlas,f.target,proposal,event),
                "legitimate free-marker movement was mistaken for a world-gauge change: "+event.reason);
    }
    require(samePose(origin->GetPose(),before.poses.at(origin)),"free-marker refinement moved the fixed origin camera");
    for(std::size_t i=0;i<before.tags.at(20).size();++i)
        require(std::abs(f.target->mStaticTags.at(20)[i]-before.tags.at(20)[i])<1e-5,
                "free-marker refinement damaged the already-correct marker");
    const Sophus::SE3f wrist(Sophus::SO3f(),Eigen::Vector3f(.02f,-.03f,.45f));
    for(KeyFrame* k:before.keyframes) {
        const auto delta=k->mReplayMarkerGauge*before.gauges.at(k).inverse();
        require(delta.sequence==event.sequence && delta.scale==1. && delta.translation.norm()<1e-9 &&
                    Eigen::AngleAxisd(delta.rotation).angle()<1e-9,
                "free-marker movement manufactured a global dense-history transform");
        require(samePose(k->mReplayMarkerGauge.apply(wrist),before.gauges.at(k).apply(wrist)),
                "a free marker update moved or resized a raw marker-anchored wrist pose");
        require(k->mReplayUnitScale==before.unitScales.at(k),"free marker pose update changed historical metric units");
    }
    // An actually moved origin is still invalid, even if a caller labels the
    // proposal accepted. Reject before publishing any geometry or sequence.
    auto invalid=proposal;
    invalid.keyframePoses.at(origin).translation().x()+=.05f;
    MapState committed(f.target);
    const auto sequence=f.atlas.mnMarkerGraphSequence;
    MarkerGraphEvent rejected;
    {
        std::unique_lock<std::mutex> gate(f.atlas.mMutexPoseGraphCorrection);
        std::unique_lock<std::mutex> lock(f.target->mMutexMapUpdate);
        require(!MarkerGraphCoordinator::CommitRefine(f.atlas,f.target,invalid,rejected),
                "an altered fixed origin passed the refine commit gauge check");
    }
    require(rejected.reason=="marker_gauge_shift_validation_failed" &&
                f.atlas.mnMarkerGraphSequence==sequence,"origin rejection published a sequence or wrong reason");
    committed.unchanged("altered fixed-origin rejection");

    // CommitRefine is also the transaction for an already-validated visual
    // loop. An identity world gauge must not discard its local Sim3 units.
    // As in scaleProposal(), this isolates publication, not pixel validation.
    Fixture loop;
    auto local=MarkerGraphOptimizer::RefineMetricMap(loop.target);
    require(local.accepted,"local-unit transaction fixture BA failed: "+local.reason);
    KeyFrame* loopOrigin=loop.target->GetOriginKF();
    for(auto& item:local.keyframePoses) {
        const float ratio=item.first==loopOrigin ? 1.f : 8.f;
        auto world=item.second.inverse();
        world.translation()=loopOrigin->GetCameraCenter()+
            ratio*(world.translation()-loopOrigin->GetCameraCenter());
        item.second=world.inverse();
        local.replayScaleMultipliers[item.first]=ratio;
    }
    MapState oldLoop(loop.target);
    MarkerGraphEvent loopEvent;
    {
        std::unique_lock<std::mutex> gate(loop.atlas.mMutexPoseGraphCorrection);
        std::unique_lock<std::mutex> lock(loop.target->mMutexMapUpdate);
        require(MarkerGraphCoordinator::CommitRefine(loop.atlas,loop.target,local,loopEvent),
                "fixed-world local-unit transaction rejected: "+loopEvent.reason);
    }
    for(KeyFrame* k:oldLoop.keyframes) {
        const float ratio=local.replayScaleMultipliers.at(k);
        require(k->mReplayUnitScale==oldLoop.unitScales.at(k)*ratio,
                "refine commit discarded an accepted local-unit multiplier");
        const auto expected=MarkerGraphTransform::between(oldLoop.poses.at(k).inverse(),
            local.keyframePoses.at(k).inverse(),ratio,loopEvent.sequence)*oldLoop.graphs.at(k);
        require(sameGraph(k->mReplayMarkerGraph,expected),
                "refine commit lost the accepted local camera/scale correction");
        require(samePose(k->mReplayMarkerGauge.apply(wrist),oldLoop.gauges.at(k).apply(wrist)),
                "local visual-unit correction scaled a raw marker-anchored wrist");
    }
    std::cout << "{\"free_marker_120mm_refine_committed\":true,\"fixed_world_and_dense_history_unchanged\":true}" << std::endl;
}

static void testIsolatedMarkerRetryGuards() {
    for(int mode=0;mode<4;++mode) {
        Fixture f;
        auto views=f.source->GetAllKeyFrames();
        std::sort(views.begin(),views.end(),KeyFrame::lId);
        KeyFrame* damaged=mode==3 ? views.front() : views.back();
        for(std::size_t i=0;i<damaged->mvTagIds.size();++i) {
            if(mode && damaged->mvTagIds[i]==22)
                damaged->mvTagImagePoints[i].x+=(i%2 ? 20.f : -20.f);
            if(mode==2 && damaged->mvTagIds[i]==21) damaged->mvTagPointWeights[i]=.25f;
        }
        MarkerGraphOptimizer::Options off;
        off.retryIsolatedMarkerGroups=false;
        MapState unchanged(f.source);
        const auto start=std::chrono::steady_clock::now();
        const auto baseline=MarkerGraphOptimizer::RefineMetricMap(f.source,off);
        const auto middle=std::chrono::steady_clock::now();
        const auto candidate=MarkerGraphOptimizer::RefineMetricMap(f.source);
        const auto end=std::chrono::steady_clock::now();
        unchanged.unchanged("marker group proposal mutated live map");
        std::cout << "FAULT_GROUP mode=" << mode << " baseline=" << baseline.accepted
                  << " candidate=" << candidate.accepted << " corners=" << candidate.after.tagCorners
                  << " baseline_ms=" << std::chrono::duration<double,std::milli>(middle-start).count()
                  << " candidate_ms=" << std::chrono::duration<double,std::milli>(end-middle).count() << std::endl;
        if(mode==1) {
            require(!baseline.accepted && candidate.accepted,"damaged group not rescued");
            require(candidate.after.tagCorners==32,"healthy corner groups were discarded");
        } else {
            require(candidate.accepted==baseline.accepted &&
                    candidate.after.tagCorners==baseline.after.tagCorners,"retry guard changed control result");
            if(mode==0) require(candidate.accepted,"clean control rejected");
        }
    }
}

static void testSoleRevisitedMarkerCannotBeExcluded() {
    Fixture f;
    f.atlas.ChangeMap(f.target);
    auto oldViews=f.target->GetAllKeyFrames();
    std::sort(oldViews.begin(),oldViews.end(),KeyFrame::lId);
    f.target->mStaticTags.erase(21);
    for(KeyFrame* k:oldViews) {
        // Retain one complete, already-established physical world marker.
        k->mvTagIds.resize(4);k->mvTagWorldPoints.resize(4);
        k->mvTagImagePoints.resize(4);k->mvTagPointWeights.resize(4);
        require(k->mvTagIds==std::vector<int>(4,20),"unexpected origin marker in revisit fixture");
    }
    auto points=f.target->GetAllMapPoints();
    std::sort(points.begin(),points.end(),[](MapPoint* a,MapPoint* b){return a->mnId<b->mnId;});
    std::vector<Eigen::Vector3f> world;
    for(MapPoint* point:points) world.push_back(point->GetWorldPos());
    const Sophus::SE3f pose(Eigen::Matrix3f::Identity(),Eigen::Vector3f(-.08f,0,0));
    KeyFrame* revisit=f.addEmpty(f.target,pose,world);
    require(revisit->mTimeStamp-oldViews.back()->mTimeStamp>.5,
            "revisit fixture did not separate the marker episodes");
    revisit->ChangeParent(oldViews.back());revisit->SetFirstConnection(false);
    revisit->mbHasTagObservation=revisit->mbTagObservationActive=true;
    revisit->mTagObservationConfidence=1.f;
    revisit->mvTagIds.assign(4,20);revisit->mvTagPointWeights.assign(4,1.f);
    revisit->mvTagWorldPoints=oldViews.front()->mvTagWorldPoints;
    for(std::size_t i=0;i<4;++i) {
        auto pixel=f.camera.project(pose*revisit->mvTagWorldPoints[i]);
        pixel.x()+=i%2?20.f:-20.f; // Contradictory raw evidence, not permission to erase the visit.
        revisit->mvTagImagePoints.emplace_back(pixel.x(),pixel.y());
    }
    for(std::size_t i=0;i<points.size();++i) {
        points[i]->AddObservation(revisit,i);revisit->AddMapPoint(points[i],i);
    }
    for(KeyFrame* k:f.target->GetAllKeyFrames()) k->UpdateConnections();
    const auto rawPixels=revisit->mvTagImagePoints;
    const auto rawMatches=revisit->GetMapPointMatches();
    MapState unchanged(f.target);
    MarkerGraphOptimizer::Options legacy;
    legacy.retryIsolatedMarkerGroups=false;
    const auto old=MarkerGraphOptimizer::RefineMetricMap(f.target,legacy);
    require(old.accepted && old.excludedTagKeyFrameIds==std::vector<unsigned long>{revisit->mnId} &&
            old.after.tagCorners==12,"legacy fixture did not erase the sole returning marker evidence");
    const auto guarded=MarkerGraphOptimizer::RefineMetricMap(f.target);
    require(!guarded.accepted && guarded.reason=="tag_reprojection_validation_failed" &&
            guarded.excludedTagKeyFrameIds.empty() && guarded.excludedTagGroupIds.empty() &&
            guarded.after.tagCorners==16,"default refinement accepted by deleting the sole marker revisit");
    require(guarded.before.backgroundObservations==old.before.backgroundObservations &&
            guarded.after.backgroundObservations==guarded.before.backgroundObservations,
            "marker-retry guard changed the background population");
    require(revisit->mvTagImagePoints==rawPixels && revisit->GetMapPointMatches()==rawMatches,
            "marker-retry proposal changed raw pixels or feature matches");
    unchanged.unchanged("sole-marker revisit proposal changed live geometry");
    std::cout << "{\"sole_revisited_marker_preserved\":true,\"legacy_false_acceptance\":true,"
                 "\"raw_tag_corners_retained\":16,\"live_geometry_unchanged\":true}" << std::endl;
}

static void testWiderProjectionSearchFaults() {
    Fixture f(true);
    auto refs=f.target->GetAllKeyFrames();
    std::sort(refs.begin(),refs.end(),KeyFrame::lId);
    for(MapPoint* point:f.target->GetAllMapPoints()) {
        point->ComputeDistinctiveDescriptors(); point->UpdateNormalAndDepth();
    }
    for(int mode=0;mode<3;++mode) {
        Frame frame(*f.base);
        frame.mvKeysUn=refs.front()->mvKeysUn;
        frame.mvpMapPoints.assign(frame.N,nullptr);
        for(int x=0;x<FRAME_GRID_COLS;++x) for(int y=0;y<FRAME_GRID_ROWS;++y) frame.mGrid[x][y].clear();
        for(int i=0;i<40;++i) {
            auto key=frame.mvKeysUn[i];
            const int x=std::round((key.pt.x-frame.mnMinX)*frame.mfGridElementWidthInv);
            const int y=std::round((key.pt.y-frame.mnMinY)*frame.mfGridElementHeightInv);
            if(x>=0&&x<FRAME_GRID_COLS&&y>=0&&y<FRAME_GRID_ROWS) frame.mGrid[x][y].push_back(i);
        }
        // 18 mm / 36 mm camera-seed offsets create about 7.5 / 15 px
        // prediction errors at the fixture's real depths. No threshold changes.
        frame.SetPose(Sophus::SE3f(Eigen::Matrix3f::Identity(),Eigen::Vector3f(mode==0?.018f:.036f,0,0)));
        if(mode==2) {
            frame.mDescriptors=frame.mDescriptors.clone();
            cv::RNG random(736);random.fill(frame.mDescriptors,cv::RNG::UNIFORM,0,256);
        }
        ORBmatcher matcher(.8,true);
        std::set<MapPoint*> found;
        int first=matcher.SearchByProjection(frame,refs.front(),found,10,100);
        for(MapPoint* point:frame.mvpMapPoints) if(point) found.insert(point);
        int extra=0;
        if(first<15) extra=matcher.SearchByProjection(frame,refs.front(),found,20,64);
        const int inliers=first+extra>=15 ? Optimizer::PoseOptimization(&frame) : 0;
        std::cout << "FAULT_SEARCH mode=" << mode << " first=" << first << " extra=" << extra
                  << " inliers=" << inliers << " position_error=" << frame.GetPose().translation().norm() << std::endl;
        if(mode==0) require(first>=15 && extra==0,"normal search unexpectedly needed retry");
        if(mode==1) require(first<15 && inliers>=30 && frame.GetPose().translation().norm()<.001f,
                           "wide search failed to recover correct measured pose");
        if(mode==2) require(inliers<15,"unrelated descriptors produced false recovery");
    }
}

static void testCalibratedBoardUsesOneRigidMarkerVariable() {
    Fixture f;
    f.atlas.ChangeMap(f.target);
    f.target->mbRigidMarkerLayout=true;
    const auto before=f.target->mStaticTags;
    for(KeyFrame* keyframe:f.target->GetAllKeyFrames())
        for(std::size_t index=0;index<keyframe->mvTagIds.size();++index)
            if(keyframe->mvTagIds[index]==21) keyframe->mvTagImagePoints[index].x+=.35f;
    const auto proposal=MarkerGraphOptimizer::RefineMetricMap(f.target);
    require(proposal.accepted,"calibrated rigid-board BA rejected: "+proposal.reason);
    require(proposal.optimizedMarkerIds==std::vector<int>({20,21}),
            "calibrated board did not expose both strong markers");
    const auto center=[](const std::vector<float>& corners) -> Eigen::Vector3f {
        Eigen::Vector3f result=Eigen::Vector3f::Zero();
        for(std::size_t i=0;i<corners.size();i+=3)
            result+=Eigen::Vector3f(corners[i],corners[i+1],corners[i+2]);
        return Eigen::Vector3f(result/4.f);
    };
    const float initialDistance=(center(before.at(20))-center(before.at(21))).norm();
    std::map<int,std::vector<float>> after;
    for(const auto& marker:proposal.staticTags)
        for(const auto& corner:marker.second) for(int axis=0;axis<3;++axis)
            after[marker.first].push_back(corner(axis));
    const float distanceError=std::abs(
        (center(after.at(20))-center(after.at(21))).norm()-initialDistance);
    require(distanceError<5e-6f,
            "calibrated board marker spacing deformed during BA: "+std::to_string(distanceError));
}

static void testMergeCommit() {
    Fixture f;
    f.atlas.mnMarkerGraphSequence=3;
    for(auto* kf:f.source->GetAllKeyFrames()) {
        kf->mReplayMarkerGauge.sequence=2;
        kf->mReplayMarkerGauge.rotation=Eigen::AngleAxisd(.2,Eigen::Vector3d::UnitZ());
        kf->mReplayMarkerGauge.translation=Eigen::Vector3d(-.1,.03,.02);
    }
    for(auto* kf:f.target->GetAllKeyFrames()) {
        kf->mReplayMarkerGauge.sequence=3;
        kf->mReplayMarkerGauge.translation=Eigen::Vector3d(.1,-.05,.02);
    }
    MapState unrelated(f.unrelated),before(f.source),targetBefore(f.target);
    const auto targetTags=f.target->mStaticTags,sourceTags=f.source->mStaticTags;
    const auto targetCount=f.target->KeyFramesInMap(),pointCount=f.target->MapPointsInMap();
    const unsigned long legacy=1000000+f.source->GetId();
    f.atlas.mMarkerMapAliases[legacy]=f.source->GetId();
    auto p=mergeProposal(f); const auto event=commitMerge(f,p);
    require(event.status=="accepted" && event.type=="marker_map_merge" && event.scale==1. &&
            event.markerIds==p.verifiedMarkerIds && f.atlas.mMarkerGraphEvents.back().sequence==event.sequence,
            "merge commit journal incorrect");
    require(f.source->IsBad() && f.source->KeyFramesInMap()==0 && f.source->MapPointsInMap()==0 &&
            f.atlas.CountMaps()==2 && f.atlas.GetCurrentMap()==f.target,"merge did not remove source/activate target");
    require(f.target->KeyFramesInMap()==targetCount+before.keyframes.size() &&
            f.target->MapPointsInMap()==pointCount+before.points.size(),"merge lost or duplicated native objects");
    require(f.atlas.mMarkerMapAliases.at(f.source->GetId())==f.target->GetId() &&
            f.atlas.mMarkerMapAliases.at(legacy)==f.target->GetId(),"historical map aliases not redirected");
    for(auto* kf:before.keyframes) {
        require(kf->GetMap()==f.target && samePose(kf->GetPose(),p.graph.keyframePoses.at(kf)),"source KF has wrong owner/pose");
        require(kf->mReplayUnitScale==before.unitScales.at(kf),"metric merge scaled camera-relative history twice");
        require(samePose((kf->mReplayMarkerGraph*before.graphs.at(kf).inverse()).apply(before.poses.at(kf).inverse()),
                         kf->GetPoseInverse()),"source marker history did not follow the merge");
        const auto rigidDelta=kf->mReplayMarkerGauge*before.gauges.at(kf).inverse();
        require(samePose(rigidDelta.apply(before.poses.at(kf).inverse()),
                         p.graph.keyframePoses.at(kf).inverse()) && rigidDelta.scale==1. &&
                rigidDelta.sequence==event.sequence,"source marker history omitted BA motion or included scale");
        for(std::size_t i=0;i<kf->mvTagWorldPoints.size();++i)
            require((kf->mvTagWorldPoints[i]-p.graph.tagWorldCorners.at(kf)[i]).norm()<1e-6f,"source tag factor stayed in old gauge");
    }
    for(auto* kf:targetBefore.keyframes) {
        const auto rigidDelta=kf->mReplayMarkerGauge*targetBefore.gauges.at(kf).inverse();
        require(rigidDelta.scale==1. && rigidDelta.sequence==event.sequence &&
                samePose(rigidDelta.apply(targetBefore.poses.at(kf).inverse()),
                         p.graph.keyframePoses.at(kf).inverse()),
                "target marker history did not follow joint marker BA");
    }
    for(auto* point:before.points)
        require(point->GetMap()==f.target && (point->GetWorldPos()-p.graph.pointPositions.at(point)).norm()<1e-6f,
                "source point has wrong owner/position");
    for(const auto& tag:targetTags) {
        const auto& refined=f.target->mStaticTags.at(tag.first);
        require(std::abs(side(refined)-side(tag.second))<1e-6f,
                "target marker physical size changed");
        require(refined==std::vector<float>({
                    p.graph.staticTags.at(tag.first)[0].x(),p.graph.staticTags.at(tag.first)[0].y(),p.graph.staticTags.at(tag.first)[0].z(),
                    p.graph.staticTags.at(tag.first)[1].x(),p.graph.staticTags.at(tag.first)[1].y(),p.graph.staticTags.at(tag.first)[1].z(),
                    p.graph.staticTags.at(tag.first)[2].x(),p.graph.staticTags.at(tag.first)[2].y(),p.graph.staticTags.at(tag.first)[2].z(),
                    p.graph.staticTags.at(tag.first)[3].x(),p.graph.staticTags.at(tag.first)[3].y(),p.graph.staticTags.at(tag.first)[3].z()}),
                "optimized target marker proposal was not committed");
    }
    require(std::abs(side(f.target->mStaticTags.at(22))-side(sourceTags.at(22)))<1e-6f,
            "source-only marker physical size changed");
    require(samePose(f.target->mMarkerInputToWorld,f.sourceToTarget),"incoming marker gauge not preserved after active-source merge");
    unrelated.unchanged("unrelated merge map");
}

static void testRejectedMergeIsAtomic() {
    Fixture f; const auto original=mergeProposal(f);
    for(int variant=0;variant<6;++variant) {
        auto p=original;
        if(variant==0) p.accepted=false;
        if(variant==1) ++p.targetRevision;
        if(variant==2) p.graph.keyframePoses.erase(f.source->GetAllKeyFrames().back());
        if(variant==3) p.graph.keyframePoses[f.unrelated->GetOriginKF()]=Sophus::SE3f();
        if(variant==4) p.graph.pointPositions.erase(f.source->GetAllMapPoints().back());
        if(variant==5) p.graph.pointPositions.begin()->second.x()=std::numeric_limits<float>::quiet_NaN();
        MapState target(f.target),source(f.source),unrelated(f.unrelated);
        const auto sequence=f.atlas.mnMarkerGraphSequence; MarkerGraphEvent event;
        std::unique_lock<std::mutex> gate(f.atlas.mMutexPoseGraphCorrection);
        std::unique_lock<std::mutex> targetLock(f.target->mMutexMapUpdate),sourceLock(f.source->mMutexMapUpdate);
        require(!MarkerGraphCoordinator::CommitMerge(f.atlas,f.target,f.source,p,event),"invalid/stale merge transaction accepted");
        target.unchanged("rejected merge target"); source.unchanged("rejected merge source"); unrelated.unchanged("rejected merge third map");
        require(f.atlas.mnMarkerGraphSequence==sequence && f.atlas.mMarkerGraphEvents.empty() &&
                f.atlas.mMarkerMapAliases.empty() && f.atlas.CountMaps()==3 && f.atlas.GetCurrentMap()==f.source,
                "rejected merge changed Atlas/journal");
    }
}

static void testCulledReferenceGraphChain() {
    Fixture f; f.atlas.ChangeMap(f.target);
    auto kfs=f.target->GetAllKeyFrames(); std::sort(kfs.begin(),kfs.end(),KeyFrame::lId);
    auto* parent=kfs.back();
    auto* middle=f.addEmpty(f.target,Sophus::SE3f(Eigen::Matrix3f::Identity(),Eigen::Vector3f(-.06f,0,0)));
    auto* leaf=f.addEmpty(f.target,Sophus::SE3f(Eigen::Matrix3f::Identity(),Eigen::Vector3f(-.08f,0,0)));
    middle->ChangeParent(parent); leaf->ChangeParent(middle);
    middle->mReplayMarkerGraph.sequence=2; middle->mReplayMarkerGraph.scale=1.1;
    middle->mReplayMarkerGraph.translation=Eigen::Vector3d(.2,0,0); middle->mReplayUnitScale=1.1f;
    leaf->mReplayMarkerGraph.sequence=3; leaf->mReplayMarkerGraph.scale=.7;
    leaf->mReplayMarkerGraph.translation=Eigen::Vector3d(-.1,.05,0); leaf->mReplayUnitScale=.7f;
    parent->mReplayMarkerGauge.translation=Eigen::Vector3d(.1,0,0);
    middle->mReplayMarkerGauge.sequence=2;
    middle->mReplayMarkerGauge.rotation=Eigen::AngleAxisd(.2,Eigen::Vector3d::UnitZ());
    middle->mReplayMarkerGauge.translation=Eigen::Vector3d(-.2,.1,0);
    leaf->mReplayMarkerGauge.sequence=3;
    leaf->mReplayMarkerGauge.translation=Eigen::Vector3d(.05,.1,.2);
    const auto leafGraph=leaf->mReplayMarkerGraph;
    const auto leafGauge=leaf->mReplayMarkerGauge;
    const auto oldWorld=leaf->GetPoseInverse();
    leaf->SetBadFlag(); middle->SetBadFlag();
    require(leaf->isBad() && middle->isBad(),"fixture cull chain was not created");
    f.atlas.mnMarkerGraphSequence=3;
    auto p=scaleProposal(f.target); MarkerGraphEvent event;
    {
        std::unique_lock<std::mutex> gate(f.atlas.mMutexPoseGraphCorrection);
        std::unique_lock<std::mutex> lock(f.target->mMutexMapUpdate);
        require(MarkerGraphCoordinator::CommitScale(f.atlas,f.target,p,parent,event),"scale commit with culled refs failed");
    }
    MarkerGraphTransform corrected;
    require(leaf->GetReplayMarkerGraph(corrected),"two-edge culled graph chain did not resolve");
    const auto moved=(corrected*leafGraph.inverse()).apply(oldWorld);
    require(std::abs(moved.translation().x()-.072f)<1e-6f && corrected.sequence==event.sequence,
            "culled graph correction lost parent delta or applied historical delta twice");
    Sophus::SE3f world; float units=0; Map* owner=nullptr;
    require(leaf->GetReplayReference(world,units,owner) && owner==f.target &&
            std::abs(world.translation().x()-.072f)<1e-6f && std::abs(units-.63f)<1e-6f,
            "culled relative-history units disagree with marker-graph history");
    MarkerGraphTransform correctedGauge;
    require(leaf->GetReplayMarkerGauge(correctedGauge) && sameGraph(correctedGauge,leafGauge),
            "scale correction leaked through a two-edge culled marker gauge");

    // A separate consistent fixture tests a genuine merge after culling;
    // culled references are no longer in the source map's live KF set.
    Fixture merged;
    auto sourceKFs=merged.source->GetAllKeyFrames();
    std::sort(sourceKFs.begin(),sourceKFs.end(),KeyFrame::lId);
    auto* surviving=sourceKFs.back();
    const auto oldSurvivingWorld=surviving->GetPoseInverse();
    auto* retiredParent=merged.addEmpty(merged.source,Sophus::SE3f());
    auto* retired=merged.addEmpty(merged.source,Sophus::SE3f());
    retiredParent->ChangeParent(surviving); retired->ChangeParent(retiredParent);
    retiredParent->mReplayMarkerGauge.translation=Eigen::Vector3d(-.2,.1,0);
    retired->mReplayMarkerGauge.rotation=Eigen::AngleAxisd(-.3,Eigen::Vector3d::UnitY());
    retired->mReplayMarkerGauge.translation=Eigen::Vector3d(.05,.1,.2);
    const auto captured=retired->mReplayMarkerGauge;
    retired->SetBadFlag(); retiredParent->SetBadFlag();
    require(retired->isBad() && retiredParent->isBad(),"merge fixture did not cull both references");
    const auto proposal=mergeProposal(merged); const auto mergeEvent=commitMerge(merged,proposal);
    require(retired->GetReplayMarkerGauge(correctedGauge),"culled marker gauge did not resolve through merged parent");
    const auto delta=correctedGauge*captured.inverse();
    const auto expectedDelta=MarkerGraphTransform::between(
        oldSurvivingWorld,proposal.graph.keyframePoses.at(surviving).inverse(),1.,mergeEvent.sequence);
    const Sophus::SE3f markerCamera(Eigen::Matrix3f::Identity(),Eigen::Vector3f(.12,.03,-.2));
    require(delta.scale==1. && delta.sequence==mergeEvent.sequence &&
            samePose(delta.apply(markerCamera),expectedDelta.apply(markerCamera)),
            "culled marker history lost the committed merge/BA correction or applied it twice");
}

static void testVersionedAtlasRoundTrip() {
    require(boost::serialization::version<Atlas>::value>=1 &&
            boost::serialization::version<Map>::value>=1 &&
            boost::serialization::version<KeyFrame>::value>=2,"marker graph/gauge archive class versions missing");
    Fixture f; const auto p=mergeProposal(f); const auto event=commitMerge(f,p);
    auto* sample=f.target->GetAllKeyFrames().back();
    sample->mReplayParentMarkerGraph.sequence=9; sample->mReplayParentMarkerGraph.scale=.8;
    sample->mReplayParentMarkerGraph.translation=Eigen::Vector3d(.1,-.2,.3);
    sample->mReplayMarkerGauge.sequence=event.sequence;
    sample->mReplayMarkerGauge.rotation=Eigen::AngleAxisd(.2,Eigen::Vector3d::UnitY());
    sample->mReplayMarkerGauge.translation=Eigen::Vector3d(.14,-.25,.36);
    sample->mReplayParentMarkerGauge.translation=Eigen::Vector3d(-.3,.2,-.1);
    const auto expectedGraph=sample->mReplayMarkerGraph,expectedParent=sample->mReplayParentMarkerGraph;
    const auto expectedGauge=sample->mReplayMarkerGauge,expectedParentGauge=sample->mReplayParentMarkerGauge;
    const auto sampleId=sample->mnId,targetId=f.target->GetId();
    f.atlas.PreSave(); std::stringstream bytes;
    {boost::archive::binary_oarchive archive(bytes); archive << f.atlas;}
    Atlas restored;
    {boost::archive::binary_iarchive archive(bytes); archive >> restored;}
    restored.SetKeyFrameDababase(&f.database); restored.SetORBVocabulary(&f.vocabulary); restored.PostLoad();
    require(restored.CountMaps()==2 && restored.mnMarkerGraphSequence==event.sequence &&
            restored.mMarkerMapAliases==f.atlas.mMarkerMapAliases && restored.mMarkerGraphEvents.size()==1 &&
            restored.mMarkerGraphEvents.front().affectedKeyframes==event.affectedKeyframes,
            "Atlas v1 graph sequence/events/aliases did not round-trip");
    Map* target=nullptr; KeyFrame* restoredSample=nullptr;
    std::vector<std::unique_ptr<KeyFrame>> ownedKeyframes;
    std::vector<std::unique_ptr<MapPoint>> ownedPoints;
    for(auto* map:restored.GetAllMaps()) {
        if(map->GetId()==targetId) target=map;
        for(auto* kf:map->GetAllKeyFrames()) {ownedKeyframes.emplace_back(kf); if(kf->mnId==sampleId) restoredSample=kf;}
        for(auto* point:map->GetAllMapPoints()) ownedPoints.emplace_back(point);
    }
    for(auto* camera:restored.GetAllCameras()) {
        auto* pinhole=dynamic_cast<Pinhole*>(camera);
        require(pinhole!=nullptr,"unexpected camera type in the fixture archive");
    }
    require(target && target->mStaticTags==f.target->mStaticTags &&
            target->mnMarkerScaleAnchorKFId==f.target->mnMarkerScaleAnchorKFId &&
            target->mnMarkerGraphSequence==f.target->mnMarkerGraphSequence &&
            samePose(target->mMarkerInputToWorld,f.target->mMarkerInputToWorld),"Map v1 marker gauge/anchor state lost");
    require(restoredSample && restoredSample->GetMap()==target &&
            sameGraph(restoredSample->mReplayMarkerGraph,expectedGraph) &&
            sameGraph(restoredSample->mReplayParentMarkerGraph,expectedParent),"KF v1 marker history/cull snapshot lost");
    MarkerGraphTransform restoredGauge;
    require(sameGraph(restoredSample->mReplayMarkerGauge,expectedGauge) &&
            sameGraph(restoredSample->mReplayParentMarkerGauge,expectedParentGauge) &&
            restoredSample->GetReplayMarkerGauge(restoredGauge) && sameGraph(restoredGauge,expectedGauge),
            "KF v2 rigid marker gauge/cull snapshot did not round-trip");
}

class RuntimeMapper : public LocalMapping {
public:
    explicit RuntimeMapper(Atlas* atlas):LocalMapping(nullptr,atlas,true,false) {mbFinished=false;}
    void finishForTest() {SetFinish();}
};

class RuntimeTracker : public Tracking {
public:
    RuntimeTracker(Fixture& f,RuntimeMapper& mapper,LoopClosing& loop,MapDrawer& drawer,
                   const std::string& settingsPath,Settings& settings)
        :Tracking(nullptr,&f.vocabulary,nullptr,&drawer,&f.atlas,&f.database,settingsPath,System::MONOCULAR,&settings) {
        SetLocalMapper(&mapper); SetLoopClosing(&loop); SetViewer(nullptr);
        mapper.SetTracker(this); loop.SetTracker(this); loop.SetLocalMapper(&mapper);
        mState=OK; mbTagMetricAligned=true; mRecoveredTagMetricScale=1.f;
    }
    void feed(const Frame& frame,const Sophus::SE3f& markerTwc,
              const std::vector<Eigen::Vector3f>& world,const std::vector<cv::Point2f>& pixels,
              const std::vector<int>& ids,bool metricTagHistory=false) {
        if(mCurrentFrame.HasPose()) mLastFrame=mCurrentFrame;
        mCurrentFrame=frame; mpReferenceKF=frame.mpReferenceKF;
        SetExternalTagObservation(markerTwc,1.f,world,pixels,true,std::vector<float>(world.size(),1.f),ids);
        require(GetMarkerTrackingStatus().accepted,"runtime raw marker observation failed its production geometry gate");
        // Feed the actual pre-tag visual estimate through its production
        // capture entry; the test does not inject ScaleEvidence or a Proposal.
        CaptureMarkerGraphVisualPose();
        if(metricTagHistory) {
            // This is an actual raw marker pose, not a visual pose relabelled
            // as metric. Exercise the production marker-history store after
            // separately capturing the pre-tag visual scale observation.
            mState=MARKER_TRACKING;
            mCurrentFrame.SetPose(markerTwc.inverse());
            StoreMarkerFrame();
            mState=OK;
        } else {
            mlpReferences.push_back(frame.mpReferenceKF);
            mlRelativeFramePoses.push_back(frame.GetPose()*frame.mpReferenceKF->GetPoseInverse());
        }
        ProcessMarkerGraph();
    }
    Sophus::SE3f externalMarkerWorld() const {return mExternalTagTwc;}
    void armVisualMotionForTest() {
        mVelocity=mCurrentFrame.GetPose()*mLastFrame.GetPose().inverse(); mbVelocity=true;
    }
    bool motionValidForTest() const {return mbVelocity;}
    Sophus::SE3f motionForTest() const {return mVelocity;}
    void checkInstantMarkerTransaction(Fixture& f) {
        auto* reference=f.target->GetAllKeyFrames().front();
        for(bool conflict:{true,false}) {
            mCurrentFrame=*f.base;
            mCurrentFrame.SetPose(reference->GetPose());
            mCurrentFrame.mvpMapPoints.assign(mCurrentFrame.N,nullptr);
            mCurrentFrame.mvbOutlier.assign(mCurrentFrame.N,false);
            int count=0;
            for(auto* point:f.target->GetAllMapPoints()) {
                const auto uv=f.camera.project(reference->GetPose()*point->GetWorldPos());
                mCurrentFrame.mvpMapPoints[count]=point;
                mCurrentFrame.mvKeysUn[count]=cv::KeyPoint(cv::Point2f(uv.x(),uv.y()),1);
                ++count;
            }
            mState=OK; mnMatchesInliers=count; mbTagMetricAligned=true;
            mbHasExternalTagObservation=true; mbHasTrackedTagObservation=false;
            mExternalTagConfidence=1.f;
            mExternalTagTwc=reference->GetPoseInverse();
            if(conflict) mExternalTagTwc.translation().x()+=.02f;
            mvExternalTagWorldPoints.clear(); mvExternalTagImagePoints.clear();
            mvExternalTagIds.clear(); mvExternalTagPointWeights.clear();
            for(int id=20;id<23;++id) {
                const auto corners=square(Eigen::Vector3f(.15f*(id-21),0,1.f),.08f);
                for(int j=0;j<4;++j) {
                    const Eigen::Vector3f p(corners[3*j],corners[3*j+1],corners[3*j+2]);
                    const auto uv=f.camera.project(mExternalTagTwc.inverse()*p);
                    mvExternalTagWorldPoints.push_back(p);
                    mvExternalTagImagePoints.emplace_back(uv.x(),uv.y());
                    mvExternalTagIds.push_back(id); mvExternalTagPointWeights.push_back(1.f);
                }
            }
            mMarkerTrackingStatus=MarkerTrackingStatus();
            const auto before=mCurrentFrame.GetPose();
            const auto outliers=mCurrentFrame.mvbOutlier;
            ApplyExternalTagPoseConstraint();
            if(conflict) {
                require(!mMarkerTrackingStatus.poseConstraintApplied &&
                        mMarkerTrackingStatus.poseConstraintReason=="deferred_to_marker_graph_background_conflict",
                        "conflicting marker pose overwrote healthy background tracking");
                require(samePose(before,mCurrentFrame.GetPose()) &&
                        outliers==mCurrentFrame.mvbOutlier,
                        "rejected marker pose did not restore pose and outlier flags atomically");
                require(mbHasExternalTagObservation && mvExternalTagWorldPoints.size()==12,
                        "background gate discarded marker evidence needed by graph BA");
            } else require(mMarkerTrackingStatus.poseConstraintApplied,
                           "consistent marker refinement was disabled");
        }
    }
    Sophus::SE3f initialMarkerWorld() const {return mInitialTagTwc;}
    void prepareInitialScale() {
        CancelPendingTagAlignment();
        mbTagMetricAligned=false; mbHasTagScaleReference=false;
        mRecoveredTagMetricScale=0;
    }
    void feedInitialScale(const Frame& frame,const Sophus::SE3f& markerTwc,
                          const std::vector<Eigen::Vector3f>& world,
                          const std::vector<cv::Point2f>& pixels,
                          const std::vector<int>& ids) {
        mCurrentFrame=frame; mpReferenceKF=frame.mpReferenceKF; mState=OK;
        SetExternalTagObservation(markerTwc,1.f,world,pixels,true,
                                  std::vector<float>(world.size(),1.f),ids);
        require(GetMarkerTrackingStatus().accepted,
                "initial-scale raw marker observation failed its production geometry gate: "+
                GetMarkerTrackingStatus().reason+" rms="+
                std::to_string(GetMarkerTrackingStatus().reprojectionPx));
        TryAlignMapToTagWorld();
    }
    void loseInitialScaleTracking() {CancelPendingTagAlignment();}
    void checkRecoveryGate(const Frame& base) {
        for(int mode=0;mode<7;++mode) {
            mCurrentFrame=base;
            mCurrentFrame.SetPose(Sophus::SE3f());
            mExternalTagTwc=Sophus::SE3f();
            mvExternalTagWorldPoints={{-.1f,-.1f,2.f},{.1f,-.1f,2.f},
                                      {.1f,.1f,2.f},{-.1f,.1f,2.f}};
            mvExternalTagPointWeights.assign(4,1.f);
            mvExternalTagImagePoints.clear();
            for(const auto& p:mvExternalTagWorldPoints) {
                const Eigen::Vector2f uv=mpCamera->project(p);
                mvExternalTagImagePoints.emplace_back(uv.x(),uv.y());
            }
            if(mode==1) mExternalTagTwc.translation().x()=.04f;
            if(mode==2) mCurrentFrame.SetPose(Sophus::SE3f(
                Eigen::AngleAxisf(.3f,Eigen::Vector3f::UnitY()).toRotationMatrix(),Eigen::Vector3f::Zero()));
            if(mode==3) for(auto& uv:mvExternalTagImagePoints) uv.x+=5.f;
            if(mode==4) mvExternalTagWorldPoints.front().z()=-2.f;
            if(mode==5) mvExternalTagPointWeights.assign(4,.25f);
            if(mode==6) mvExternalTagImagePoints.front().x=std::numeric_limits<float>::quiet_NaN();
            const bool accepted=MarkerRecoveryPoseConsistent();
            require(accepted==(mode==0),"production recovery marker consistency gate failed mode "+std::to_string(mode));
        }
        std::cout << "RECOVERY_MARKER_GATE cases=7 passed=1" << std::endl;
    }
};

static void testRecoveryMarkerGate(const std::string& settingsPath) {
    Fixture f;
    Settings parameters(settingsPath,System::MONOCULAR);
    RuntimeMapper mapper(&f.atlas);
    LoopClosing loop(&f.atlas,&f.database,&f.vocabulary,false,true);
    MapDrawer drawer(&f.atlas,settingsPath,&parameters);
    RuntimeTracker tracker(f,mapper,loop,drawer,settingsPath,parameters);
    tracker.checkRecoveryGate(*f.base);
    f.atlas.ChangeMap(f.target);
    tracker.checkInstantMarkerTransaction(f);
}

static void markerMeasurement(Fixture& f,Map* map,const Sophus::SE3f& markerTwc,
                              std::vector<Eigen::Vector3f>& world,
                              std::vector<cv::Point2f>& pixels,std::vector<int>& ids) {
    world.clear(); pixels.clear(); ids.clear();
    for(const auto& tag:map->mStaticTags) for(int corner=0;corner<4;++corner) {
        const Eigen::Vector3f point(tag.second[corner*3],tag.second[corner*3+1],tag.second[corner*3+2]);
        const Eigen::Vector2f pixel=f.camera.project(markerTwc.inverse()*point);
        world.push_back(point); pixels.emplace_back(pixel.x(),pixel.y()); ids.push_back(tag.first);
    }
}

static Frame markerFrame(Fixture& f,Map* map,double timestamp,const Sophus::SE3f& visualTwc) {
    Frame frame(*f.base); frame.mnId=Frame::nNextId++; frame.mTimeStamp=timestamp;
    frame.SetPose(visualTwc.inverse());
    auto keyframes=map->GetAllKeyFrames(); std::sort(keyframes.begin(),keyframes.end(),KeyFrame::lId);
    frame.mpReferenceKF=keyframes.back();
    return frame;
}

static void replaceMarkerRegistry(Fixture& f,Map* map,
                                  const std::map<int,std::vector<float>>& tags) {
    map->mStaticTags=tags;
    for(KeyFrame* keyframe:map->GetAllKeyFrames()) {
        keyframe->mbHasTagObservation=keyframe->mbTagObservationActive=true;
        keyframe->mTagObservationConfidence=1.f;
        keyframe->mvTagIds.clear(); keyframe->mvTagWorldPoints.clear();
        keyframe->mvTagImagePoints.clear(); keyframe->mvTagPointWeights.clear();
        for(const auto& tag:tags) for(int corner=0;corner<4;++corner) {
            const Eigen::Vector3f point(tag.second[corner*3],tag.second[corner*3+1],tag.second[corner*3+2]);
            const Eigen::Vector2f pixel=f.camera.project(keyframe->GetPose()*point);
            keyframe->mvTagIds.push_back(tag.first); keyframe->mvTagWorldPoints.push_back(point);
            keyframe->mvTagImagePoints.emplace_back(pixel.x(),pixel.y());
            keyframe->mvTagPointWeights.push_back(1.f);
        }
    }
}

static void testConcurrentAtlasRetirement() {
    Fixture f; f.atlas.ChangeMap(f.target);
    std::atomic<bool> started{false},done{false};
    std::thread reader([&]() {
        started.store(true,std::memory_order_release);
        while(!done.load(std::memory_order_acquire)) {
            const auto maps=f.atlas.GetAllMaps();
            (void)f.atlas.CountMaps();
            for(Map* map:maps) (void)map->IsBad();
        }
    });
    while(!started.load(std::memory_order_acquire)) std::this_thread::yield();
    f.atlas.SetMapBad(f.unrelated);
    f.atlas.RemoveBadMaps();
    done.store(true,std::memory_order_release);
    reader.join();
    require(f.unrelated->IsBad() && f.atlas.CountMaps()==2,
            "Atlas retirement did not atomically publish a bad map outside the live set");
}

static void testInitialScaleWindowBoundaries(const std::string& settings) {
    Fixture f; f.atlas.ChangeMap(f.target);
    f.target->mbMetric=false; f.source->mbMetric=false;
    Settings parameters(settings,System::MONOCULAR);
    RuntimeMapper mapper(&f.atlas);
    LoopClosing loop(&f.atlas,&f.database,&f.vocabulary,false,true);
    MapDrawer drawer(&f.atlas,settings,&parameters);
    RuntimeTracker tracker(f,mapper,loop,drawer,settings,parameters);
    tracker.prepareInitialScale();
    const auto feed=[&](Map* map,double timestamp,float x) {
        const Sophus::SE3f markerTwc(Eigen::Matrix3f::Identity(),Eigen::Vector3f(x,0,0));
        auto visualTwc=markerTwc; visualTwc.translation().x()=x/.85f;
        Frame frame=markerFrame(f,map,timestamp,visualTwc);
        std::vector<Eigen::Vector3f> world; std::vector<cv::Point2f> pixels; std::vector<int> ids;
        markerMeasurement(f,map,markerTwc,world,pixels,ids);
        tracker.feedInitialScale(frame,markerTwc,world,pixels,ids);
    };
    for(int i=0;i<4;++i) feed(f.target,1.+i*.1f,.01f*i);
    for(int i=0;i<4;++i) feed(f.target,2.+i*.1f,.04f+.01f*i);
    require(!mapper.stopRequested(),"initial metric scale reused observations across a long marker gap");

    tracker.loseInitialScaleTracking();
    for(int i=0;i<4;++i) feed(f.target,2.5+i*.1f,.01f*i);
    for(int i=0;i<4;++i) feed(f.target,2.+i*.1f,.04f+.01f*i);
    require(!mapper.stopRequested(),"initial metric scale reused observations across reversed timestamps");

    tracker.loseInitialScaleTracking();
    for(int i=0;i<4;++i) feed(f.target,3.+i*.1f,.01f*i);
    f.atlas.ChangeMap(f.source);
    for(int i=0;i<4;++i) feed(f.source,3.4+i*.1f,.04f+.01f*i);
    require(!mapper.stopRequested(),"initial metric scale reused observations from another map epoch");

    tracker.loseInitialScaleTracking();
    for(int i=0;i<4;++i) feed(f.source,4.+i*.1f,.01f*i);
    tracker.loseInitialScaleTracking();
    for(int i=0;i<4;++i) feed(f.source,4.4+i*.1f,.04f+.01f*i);
    require(!mapper.stopRequested(),"initial metric scale retained observations through tracking loss");

    tracker.loseInitialScaleTracking();
    for(int i=0;i<8;++i) feed(f.source,5.+i*.04f,.01f*i);
    require(mapper.stopRequested(),"eight fresh continuous initial-scale observations did not schedule alignment");
    tracker.loseInitialScaleTracking();
    require(!mapper.stopRequested(),"initial-scale test cleanup retained the owned mapper stop");
}

static void testRuntimeCornerRevisitScheduling(const std::string& settings) {
    struct Case { double gap; bool epoch, differentId, stationary, expected; };
    const std::vector<Case> cases{{.31,false,false,false,true},{.4,false,false,false,true},
        {.5,false,false,false,true},{1.,false,false,false,true},
        {.4,true,false,false,false},{.4,false,true,false,false},
        {.4,false,false,true,false},{1.01,false,false,false,false}};
    for(const auto& test:cases) {
        Fixture f; f.atlas.ChangeMap(f.target);
        // No cross-map merge is relevant to this scheduling-only fixture.
        f.source->mStaticTags.clear(); f.unrelated->mStaticTags.clear();
        Settings parameters(settings,System::MONOCULAR);
        RuntimeMapper mapper(&f.atlas);
        LoopClosing loop(&f.atlas,&f.database,&f.vocabulary,false,true);
        MapDrawer drawer(&f.atlas,settings,&parameters);
        RuntimeTracker tracker(f,mapper,loop,drawer,settings,parameters);
        auto kfs=f.target->GetAllKeyFrames(); std::sort(kfs.begin(),kfs.end(),KeyFrame::lId);
        KeyFrame* reference=kfs.back();
        std::vector<unsigned long> added;
        for(int index=0;index<16;++index) {
            if(index==8 && test.epoch) f.target->InformNewBigChange();
            const int segment=index/8, local=index%8, id=segment && test.differentId?21:20;
            const double timestamp=1.+segment*(.14+test.gap)+local*.02;
            const float x=(test.stationary?.001f:.01f)*index;
            const Sophus::SE3f markerTwc(Eigen::Matrix3f::Identity(),Eigen::Vector3f(x,0,0));
            auto visualTwc=markerTwc;
            // Deliberately incompatible segment offsets: copied dense poses
            // must not be differenced across the gap to manufacture a scale.
            visualTwc.translation().x()+=segment?-2.f:4.f;
            Frame frame=markerFrame(f,f.target,timestamp,visualTwc);
            std::vector<Eigen::Vector3f> world; std::vector<cv::Point2f> pixels; std::vector<int> ids;
            const auto& corners=f.target->mStaticTags.at(id);
            for(int j=0;j<4;++j) {
                const Eigen::Vector3f p(corners[3*j],corners[3*j+1],corners[3*j+2]);
                const auto pixel=f.camera.project(markerTwc.inverse()*p);
                world.push_back(p); pixels.emplace_back(pixel.x(),pixel.y()); ids.push_back(id);
            }
            if(local==0) {
                f.keyframes.emplace_back(new KeyFrame(frame,f.target,&f.database));
                KeyFrame* current=f.keyframes.back().get();
                current->mbHasTagObservation=current->mbTagObservationActive=true;
                current->mTagObservationConfidence=1.f;
                current->mvTagWorldPoints=world; current->mvTagImagePoints=pixels;
                current->mvTagIds=ids; current->mvTagPointWeights.assign(4,1.f);
                f.target->AddKeyFrame(current); current->ChangeParent(reference); current->SetFirstConnection(false);
                reference=current; added.push_back(current->mnId);
            }
            frame.mpReferenceKF=reference;
            {
                std::unique_lock<std::mutex> gate(f.atlas.mMutexPoseGraphCorrection);
                tracker.feed(frame,markerTwc,world,pixels,ids);
            }
            require(f.atlas.mMarkerGraphEvents.empty(),"corner revisit committed before mapper acknowledgement");
            require(samePose(frame.GetPose(),tracker.mCurrentFrame.GetPose()),
                    "corner revisit directly replaced camera with single-marker pose");
            if(index<15) require(!mapper.stopRequested(),"revisit scheduled before eight CURRENT observations");
        }
        require(added.size()==2 && mapper.stopRequested()==test.expected,
                "runtime bounded corner revisit schedule mismatch, gap="+std::to_string(test.gap)+
                " epoch="+std::to_string(test.epoch)+" different_id="+std::to_string(test.differentId)+
                " stationary="+std::to_string(test.stationary));
        tracker.CancelMarkerGraph();
        require(!mapper.stopRequested(),"cancelling candidate retained owned mapper stop");
        require(f.atlas.mMarkerGraphEvents.empty() && f.target->mnMarkerGraphSequence==0,
                "candidate-only regression changed committed graph");
    }
    std::cout << "{\"runtime_corner_revisit_cases\":" << cases.size()
              << ",\"ba_runs\":0,\"commits\":0}" << std::endl;
}

static void testRuntimeScaleScheduling(const std::string& settings,bool metricTagHistory=false) {
    Fixture f; f.atlas.ChangeMap(f.target);
    // This regression isolates re-anchoring: a common-ID inactive map should
    // otherwise legitimately schedule a merge before scale estimation.
    std::map<int,std::vector<float>> unrelatedTags;
    for(const auto& tag:f.source->mStaticTags) unrelatedTags[tag.first+10]=tag.second;
    f.source->mStaticTags=unrelatedTags;
    for(auto* kf:f.source->GetAllKeyFrames()) for(int& id:kf->mvTagIds) id+=10;
    Settings parameters(settings,System::MONOCULAR);
    RuntimeMapper mapper(&f.atlas);
    LoopClosing loop(&f.atlas,&f.database,&f.vocabulary,false,true);
    MapDrawer drawer(&f.atlas,settings,&parameters);
    RuntimeTracker tracker(f,mapper,loop,drawer,settings,parameters);
    MapState source(f.source),unrelated(f.unrelated);
    MapState truth(f.target);
    MarkerGraphOptimizer::PoseMap trueWorld;
    for(const auto& value:truth.poses) trueWorld[value.first]=value.second.inverse();
    auto kfs=f.target->GetAllKeyFrames(); std::sort(kfs.begin(),kfs.end(),KeyFrame::lId);
    KeyFrame* anchor=kfs.front(); KeyFrame* reference=kfs.back();
    const auto anchorPose=anchor->GetPose(); const auto tags=f.target->mStaticTags;
    const Eigen::Vector3f origin=anchor->GetCameraCenter();
    const float drift=1.15f;
    // A real internally consistent monocular scale error: every background
    // point and non-A camera moves in world space, while original image
    // pixels and physical marker dimensions stay unchanged.
    for(auto* kf:kfs) {
        auto Twc=kf->GetPoseInverse();
        Twc.translation()=origin+drift*(Twc.translation()-origin); kf->SetPose(Twc.inverse());
    }
    for(const auto& point:truth.positions)
        point.first->SetWorldPos(origin+drift*(point.second-origin));
    const unsigned long firstFrame=Frame::nNextId;
    std::vector<unsigned long> bIds;
    for(int index=0;index<8;++index) {
        const Sophus::SE3f markerTwc(Eigen::Matrix3f::Identity(),Eigen::Vector3f(.02f+.01f*index,0,0));
        auto visualTwc=markerTwc;
        visualTwc.translation()=origin+drift*(visualTwc.translation()-origin);
        Frame frame(*f.base); frame.mnId=Frame::nNextId++; frame.mTimeStamp=1.+index*.1;
        frame.SetPose(visualTwc.inverse());
        std::vector<Eigen::Vector3f> tagWorld; std::vector<cv::Point2f> tagPixels; std::vector<int> ids;
        for(const auto& tag:tags) for(int j=0;j<4;++j) {
            const Eigen::Vector3f p(tag.second[j*3],tag.second[j*3+1],tag.second[j*3+2]);
            const auto pixel=f.camera.project(markerTwc.inverse()*p);
            tagWorld.push_back(p); tagPixels.emplace_back(pixel.x(),pixel.y()); ids.push_back(tag.first);
        }
        for(std::size_t i=0;i<40;++i) {
            auto* point=anchor->GetMapPoint(i);
            const auto pixel=f.camera.project(markerTwc.inverse()*truth.positions.at(point));
            frame.mvKeysUn[i]=cv::KeyPoint(cv::Point2f(pixel.x(),pixel.y()),1);
            frame.mvpMapPoints[i]=point;
        }
        if(index==0 || index==7) {
            f.keyframes.emplace_back(new KeyFrame(frame,f.target,&f.database));
            KeyFrame* current=f.keyframes.back().get();
            current->mbHasTagObservation=current->mbTagObservationActive=true;
            current->mTagObservationConfidence=1.f;
            current->mvTagWorldPoints=tagWorld; current->mvTagImagePoints=tagPixels;
            current->mvTagIds=ids; current->mvTagPointWeights.assign(ids.size(),1.f);
            f.target->AddKeyFrame(current); current->ChangeParent(reference); current->SetFirstConnection(false);
            for(std::size_t i=0;i<40;++i) frame.mvpMapPoints[i]->AddObservation(current,i);
            for(auto* kf:f.target->GetAllKeyFrames()) kf->UpdateConnections();
            trueWorld[current]=markerTwc; reference=current; bIds.push_back(current->mnId);
        }
        frame.mpReferenceKF=reference;
        {
            std::unique_lock<std::mutex> gate(f.atlas.mMutexPoseGraphCorrection);
            tracker.feed(frame,markerTwc,tagWorld,tagPixels,ids,metricTagHistory);
        }
        require(f.atlas.mMarkerGraphEvents.empty(),"scale committed before mapper stop acknowledgement");
        if(index<7) require(!mapper.stopRequested(),"scale scheduled before eight independent images");
    }
    require(bIds.size()==2 && Frame::nNextId-firstFrame==8 && mapper.stopRequested() && !mapper.isStopped(),
            "eight independent images with A+two B keyframes did not request safe mapper stop");
    auto runtimeKFs=f.target->GetAllKeyFrames();
    std::sort(runtimeKFs.begin(),runtimeKFs.end(),KeyFrame::lId);
    for(auto* kf:runtimeKFs) {
        std::set<KeyFrame*> seen;
        std::ostringstream chain;
        KeyFrame* current=kf;
        while(current && seen.insert(current).second) {
            if(seen.size()>1) chain << ",";
            chain << current->mnId;
            require(current->GetMap()==f.target && !current->isBad(),"runtime fixture parent crosses an invalid map");
            if(current==anchor) break;
            current=current->GetParent();
        }
        require(current==anchor,"runtime fixture parent chain does not reach A: ["+chain.str()+"]");
        std::cout << "{\"runtime_parent_chain\":[" << chain.str() << "],\"anchor_kf\":" << anchor->mnId << "}" << std::endl;
    }
    MapState pending(f.target);
    {
        std::unique_lock<std::mutex> gate(f.atlas.mMutexPoseGraphCorrection);
        tracker.ProcessMarkerGraph();
    }
    require(f.atlas.mMarkerGraphEvents.empty(),"pending graph ran without mapper acknowledgement");
    pending.unchanged("runtime pending transaction");
    require(tracker.mMarkerMetricFrames.size()==(metricTagHistory?8u:0u),
            "runtime marker store omitted a measured frame or relabelled visual history");
    const auto currentBefore=tracker.mCurrentFrame.GetPose();
    const auto lastBefore=tracker.mLastFrame.GetPose();
    tracker.armVisualMotionForTest();
    const auto motionBefore=tracker.motionForTest();
    const float unitBefore=tracker.mCurrentFrame.mpReferenceKF->mReplayUnitScale;
    double cameraBefore=0,pointsBefore=0;
    for(const auto& value:trueWorld) cameraBefore+=(value.first->GetCameraCenter()-value.second.translation()).squaredNorm();
    for(const auto& value:truth.positions) pointsBefore+=(value.first->GetWorldPos()-value.second).squaredNorm();
    require(cameraBefore>1e-5 && pointsBefore>.1,"runtime fixture has no real geometric scale drift");
    require(mapper.Stop() && mapper.isStoppedForTagAlignment(),"mapper safe point did not acknowledge its owned pause");
    {
        std::unique_lock<std::mutex> gate(f.atlas.mMutexPoseGraphCorrection);
        tracker.ProcessMarkerGraph();
    }
    require(f.atlas.mMarkerGraphEvents.size()==1,"runtime coordinator did not publish exactly one result");
    const auto& event=f.atlas.mMarkerGraphEvents.front();
    require(event.type=="scale_reanchor" && event.status=="accepted", "runtime scale candidate failed: "+event.reason);
    require(tracker.motionValidForTest()!=metricTagHistory,
            "graph commit lost valid visual prediction or retained marker-only prediction");
    if(!metricTagHistory) {
        auto expectedMotion=motionBefore;
        expectedMotion.translation()*=tracker.mCurrentFrame.mpReferenceKF->mReplayUnitScale/unitBefore;
        require(samePose(tracker.motionForTest(),expectedMotion),
                "camera-relative motion did not retain rotation and correct translation units once");
    }
    require(event.candidateFrameId==long(firstFrame+7) && event.frameId==long(firstFrame+7) &&
            std::abs(event.scale-1./drift)<1e-5,"runtime event lost measured scale or real triggering frame");
    for(auto id:bIds) require(std::find(event.affectedKeyframes.begin(),event.affectedKeyframes.end(),id)!=event.affectedKeyframes.end(),
                            "runtime scale commit omitted a verified B keyframe");
    double cameraAfter=0,pointsAfter=0;
    for(const auto& value:trueWorld) cameraAfter+=(value.first->GetCameraCenter()-value.second.translation()).squaredNorm();
    for(const auto& value:truth.positions) pointsAfter+=(value.first->GetWorldPos()-value.second).squaredNorm();
    require(std::sqrt(cameraAfter/cameraBefore)<.3 && std::sqrt(pointsAfter/pointsBefore)<.3,
            "scheduled scale optimization did not correct the injected camera/point drift");
    require(tracker.mlRelativeFramePoses.size()==8 && tracker.mlpReferences.size()==8,
            "runtime history lost one of the eight captured frames");
    double historySquaredError=0,historyMaxError=0;
    auto historicalPose=tracker.mlRelativeFramePoses.begin();
    auto historicalReference=tracker.mlpReferences.begin();
    for(int index=0;index<8;++index,++historicalPose,++historicalReference) {
        const auto Twc=(*historicalReference)->GetPoseInverse()*historicalPose->inverse();
        const Eigen::Vector3f expected(.02f+.01f*index,0,0);
        const double error=(Twc.translation()-expected).norm();
        historySquaredError+=error*error; historyMaxError=std::max(historyMaxError,error);
        require(Twc.so3().log().norm()<1e-3f,"runtime reconstructed history has spurious rotation");
        if(metricTagHistory) {
            const auto& metric=tracker.mMarkerMetricFrames.at(firstFrame+index);
            require(metric.historyIndex==std::size_t(index) &&
                    (metric.worldFromCamera.translation()-expected).norm()<1e-7f &&
                    samePose(metric.worldFromCamera,Twc),
                    "native metric marker history changed or no longer agrees with its updated reference");
        }
    }
    require(historyMaxError<(metricTagHistory?1e-6:5e-4),
            "runtime historical frame units disagree with final metric geometry: max error="+
            std::to_string(historyMaxError)+" m");
    if(metricTagHistory)
        require(samePose(tracker.mCurrentFrame.GetPose(),currentBefore) &&
                samePose(tracker.mLastFrame.GetPose(),lastBefore),
                "runtime current/last metric marker pose was corrected a second time");
    for(const auto& value:pending.gauges)
        require(sameGraph(value.first->mReplayMarkerGauge,value.second),
                "runtime scale re-anchor modified a fixed marker-world gauge");
    bool rigidMarkers=f.target->mStaticTags.size()==tags.size();
    for(const auto& marker:tags)
        rigidMarkers=rigidMarkers && f.target->mStaticTags.count(marker.first) &&
            std::abs(side(f.target->mStaticTags.at(marker.first))-side(marker.second))<1e-6f;
    require(samePose(anchor->GetPose(),anchorPose) && rigidMarkers && f.target->mMetricScale==1.f,
            "runtime re-anchor moved A or changed physical marker dimensions");
    require(!mapper.stopRequested() && !mapper.isStopped(),"runtime completion did not release its owned mapper pause");
    source.unchanged("runtime unrelated source"); unrelated.unchanged("runtime unrelated map");
    std::cout << "{\"runtime_scale_scheduling\":true,\"marker_metric_history\":"
              << (metricTagHistory?"true":"false") << ",\"scale\":" << event.scale
              << ",\"camera_error_ratio\":" << std::sqrt(cameraAfter/cameraBefore)
              << ",\"point_error_ratio\":" << std::sqrt(pointsAfter/pointsBefore)
              << ",\"history_rms_m\":" << std::sqrt(historySquaredError/8)
              << ",\"history_max_m\":" << historyMaxError << "}" << std::endl;
}

static void testRuntimeInactiveSourceMerge(const std::string& settings) {
    Fixture f; f.atlas.ChangeMap(f.target);
    Settings parameters(settings,System::MONOCULAR);
    RuntimeMapper mapper(&f.atlas);
    LoopClosing loop(&f.atlas,&f.database,&f.vocabulary,false,true);
    MapDrawer drawer(&f.atlas,settings,&parameters);
    RuntimeTracker tracker(f,mapper,loop,drawer,settings,parameters);
    MapState target(f.target),source(f.source),unrelated(f.unrelated);
    auto kfs=f.target->GetAllKeyFrames(); std::sort(kfs.begin(),kfs.end(),KeyFrame::lId);
    const Sophus::SE3f markerTwc(Eigen::Matrix3f::Identity(),Eigen::Vector3f(.03,0,0));
    Frame frame(*f.base); frame.mnId=Frame::nNextId++; frame.mTimeStamp=2.;
    frame.SetPose(markerTwc.inverse()); frame.mpReferenceKF=kfs.back();
    std::vector<Eigen::Vector3f> tagWorld;
    std::vector<cv::Point2f> tagPixels; std::vector<int> ids;
    for(const auto& tag:target.tags) for(int j=0;j<4;++j) {
        const Eigen::Vector3f p(tag.second[j*3],tag.second[j*3+1],tag.second[j*3+2]);
        const auto pixel=f.camera.project(markerTwc.inverse()*p);
        tagWorld.push_back(p); tagPixels.emplace_back(pixel.x(),pixel.y()); ids.push_back(tag.first);
    }
    for(std::size_t i=0;i<40;++i) {
        auto* point=kfs.front()->GetMapPoint(i);
        const auto pixel=f.camera.project(markerTwc.inverse()*point->GetWorldPos());
        frame.mvKeysUn[i]=cv::KeyPoint(cv::Point2f(pixel.x(),pixel.y()),1);
        frame.mvpMapPoints[i]=point;
    }
    const auto initialBefore=tracker.initialMarkerWorld();
    {
        std::unique_lock<std::mutex> gate(f.atlas.mMutexPoseGraphCorrection);
        tracker.feed(frame,markerTwc,tagWorld,tagPixels,ids,true);
    }
    require(f.atlas.GetCurrentMap()==f.target && mapper.stopRequested() && !mapper.isStopped() &&
            f.atlas.mMarkerGraphEvents.empty(),"inactive common-marker source did not schedule a safe merge into the active target");
    const auto externalBefore=tracker.externalMarkerWorld();
    {
        std::unique_lock<std::mutex> gate(f.atlas.mMutexPoseGraphCorrection);
        tracker.ProcessMarkerGraph();
    }
    require(f.atlas.mMarkerGraphEvents.empty(),"inactive-source merge committed without mapper acknowledgement");
    source.unchanged("pending inactive source");
    require(mapper.Stop() && mapper.isStoppedForTagAlignment(),"inactive-source mapper pause not acknowledged");
    {
        std::unique_lock<std::mutex> gate(f.atlas.mMutexPoseGraphCorrection);
        tracker.ProcessMarkerGraph();
    }
    require(f.atlas.mMarkerGraphEvents.size()==1,"inactive-source scheduling published an unexpected result count");
    const auto& event=f.atlas.mMarkerGraphEvents.front();
    require(event.type=="marker_map_merge" && event.status=="accepted" &&
            event.sourceMapId==long(f.source->GetId()) && event.targetMapId==long(f.target->GetId()) &&
            event.frameId==long(frame.mnId),"inactive-source merge failed or swapped the retained world: "+event.reason);
    require(f.atlas.GetCurrentMap()==f.target && f.source->IsBad() && f.atlas.CountMaps()==2 &&
            f.atlas.mMarkerMapAliases.at(f.source->GetId())==f.target->GetId(),
            "inactive-source merge changed the active target or lost the source alias");
    require(samePose(tracker.externalMarkerWorld(),externalBefore) &&
            samePose(tracker.initialMarkerWorld(),initialBefore) &&
            samePose(f.target->mMarkerInputToWorld,target.inputGauge),
            "inactive-source merge transformed the already-active marker input gauge");
    require(tracker.mMarkerMetricFrames.size()==1 &&
            samePose(tracker.mMarkerMetricFrames.at(frame.mnId).worldFromCamera,markerTwc) &&
            samePose(tracker.mCurrentFrame.GetPose().inverse(),markerTwc) &&
            samePose(tracker.mLastFrame.GetPose().inverse(),markerTwc),
            "inactive-source merge moved the target's current/last metric marker history");
    const auto historical=tracker.mlpReferences.front()->GetPoseInverse()*tracker.mlRelativeFramePoses.front().inverse();
    require(samePose(historical,markerTwc),"active-target marker relative history does not resolve to the preserved metric pose");
    for(auto* kf:target.keyframes) {
        const auto delta=kf->mReplayMarkerGauge*target.gauges.at(kf).inverse();
        require(delta.scale==1. && delta.sequence==event.sequence &&
                samePose(delta.apply(target.poses.at(kf).inverse()),kf->GetPoseInverse()),
                "active-target marker history missed the committed joint BA");
    }
    for(auto* kf:source.keyframes) {
        MarkerGraphTransform resolved;
        require(kf->GetMap()==f.target && kf->GetReplayMarkerGauge(resolved) && resolved.scale==1. &&
                samePose((resolved*source.gauges.at(kf).inverse()).apply(source.poses.at(kf).inverse()),
                         kf->GetPoseInverse()),
                "inactive-source keyframe ownership or rigid marker correction was not transferred");
    }
    require(!mapper.stopRequested() && !mapper.isStopped(),"inactive-source merge retained its mapper pause");
    unrelated.unchanged("inactive-source unrelated map");
    std::cout << "{\"runtime_inactive_source_merge\":true,\"active_target_preserved\":true}" << std::endl;
}

static void testCancelledMergeCanReschedule(const std::string& settings) {
    Fixture f; f.atlas.ChangeMap(f.target);
    Settings parameters(settings,System::MONOCULAR);
    RuntimeMapper mapper(&f.atlas);
    LoopClosing loop(&f.atlas,&f.database,&f.vocabulary,false,true);
    MapDrawer drawer(&f.atlas,settings,&parameters);
    RuntimeTracker tracker(f,mapper,loop,drawer,settings,parameters);
    const Sophus::SE3f markerTwc(Eigen::Matrix3f::Identity(),Eigen::Vector3f(.03f,0,0));
    std::vector<Eigen::Vector3f> world; std::vector<cv::Point2f> pixels; std::vector<int> ids;
    markerMeasurement(f,f.target,markerTwc,world,pixels,ids);
    Frame first=markerFrame(f,f.target,2.,markerTwc);
    {
        std::unique_lock<std::mutex> gate(f.atlas.mMutexPoseGraphCorrection);
        tracker.feed(first,markerTwc,world,pixels,ids,true);
    }
    require(mapper.stopRequested(),"merge cancellation fixture did not schedule its first attempt");
    f.atlas.ChangeMap(f.unrelated);
    {
        std::unique_lock<std::mutex> gate(f.atlas.mMutexPoseGraphCorrection);
        tracker.ProcessMarkerGraph();
    }
    require(!mapper.stopRequested() && f.atlas.mMarkerGraphEvents.empty(),
            "early merge cancellation did not release its owned stop cleanly");
    f.atlas.ChangeMap(f.target);
    Frame second=markerFrame(f,f.target,2.1,markerTwc);
    {
        std::unique_lock<std::mutex> gate(f.atlas.mMutexPoseGraphCorrection);
        tracker.feed(second,markerTwc,world,pixels,ids,true);
    }
    require(mapper.stopRequested(),"an unexecuted merge attempt suppressed a later identical retry");
    tracker.CancelMarkerGraph();
    require(!mapper.stopRequested(),"cancelled-merge test cleanup retained the owned mapper stop");
}

static void testLocalizationFinalProcessesPending(const std::string& settings) {
    Fixture f; f.atlas.ChangeMap(f.target);
    Settings parameters(settings,System::MONOCULAR);
    RuntimeMapper mapper(&f.atlas);
    LoopClosing loop(&f.atlas,&f.database,&f.vocabulary,false,true);
    MapDrawer drawer(&f.atlas,settings,&parameters);
    RuntimeTracker tracker(f,mapper,loop,drawer,settings,parameters);
    const Sophus::SE3f markerTwc(Eigen::Matrix3f::Identity(),Eigen::Vector3f(.03f,0,0));
    std::vector<Eigen::Vector3f> world; std::vector<cv::Point2f> pixels; std::vector<int> ids;
    markerMeasurement(f,f.target,markerTwc,world,pixels,ids);
    Frame frame=markerFrame(f,f.target,2.,markerTwc);
    {
        std::unique_lock<std::mutex> gate(f.atlas.mMutexPoseGraphCorrection);
        tracker.feed(frame,markerTwc,world,pixels,ids,true);
    }
    require(mapper.stopRequested(),"localization-final fixture did not schedule a merge");
    mapper.finishForTest(); tracker.InformOnlyTracking(true);
    {
        std::unique_lock<std::mutex> gate(f.atlas.mMutexPoseGraphCorrection);
        tracker.ProcessMarkerGraph(true);
    }
    const auto& events=f.atlas.mMarkerGraphEvents;
    require(events.size()==3 && events.front().type=="marker_map_merge" &&
            events.front().status=="accepted" && f.source->IsBad() &&
            std::count_if(events.begin(),events.end(),[](const MarkerGraphEvent& event) {
                return event.type=="marker_global_ba" && event.status=="accepted";
            })==2,
            "localization-only final flush dropped its merge or per-map final BA");
}

static void testFinalDrainsRecursiveMergeChain(const std::string& settings) {
    Fixture f;
    const auto target20=f.target->mStaticTags.at(20);
    const auto source22=f.source->mStaticTags.at(22);
    const auto target22=transformCorners(source22,f.sourceToTarget);
    replaceMarkerRegistry(f,f.target,{{20,target20}});
    replaceMarkerRegistry(f,f.source,{{22,source22}});
    replaceMarkerRegistry(f,f.unrelated,{{20,target20},{22,target22}});
    f.atlas.ChangeMap(f.unrelated);
    Settings parameters(settings,System::MONOCULAR);
    RuntimeMapper mapper(&f.atlas);
    LoopClosing loop(&f.atlas,&f.database,&f.vocabulary,false,true);
    MapDrawer drawer(&f.atlas,settings,&parameters);
    RuntimeTracker tracker(f,mapper,loop,drawer,settings,parameters);
    const Sophus::SE3f markerTwc(Eigen::Matrix3f::Identity(),Eigen::Vector3f(.03f,0,0));
    std::vector<Eigen::Vector3f> world; std::vector<cv::Point2f> pixels; std::vector<int> ids;
    markerMeasurement(f,f.unrelated,markerTwc,world,pixels,ids);
    Frame frame=markerFrame(f,f.unrelated,2.,markerTwc);
    {
        std::unique_lock<std::mutex> gate(f.atlas.mMutexPoseGraphCorrection);
        tracker.feed(frame,markerTwc,world,pixels,ids,true);
    }
    require(mapper.stopRequested(),"recursive-final fixture did not schedule bridge-map merge");
    mapper.finishForTest();
    {
        std::unique_lock<std::mutex> gate(f.atlas.mMutexPoseGraphCorrection);
        tracker.ProcessMarkerGraph(true);
    }
    require(f.atlas.CountMaps()==1 && f.atlas.GetCurrentMap()==f.target &&
            f.source->IsBad() && f.unrelated->IsBad(),
            "one final drain did not recursively merge the three-map marker chain");
    require(f.atlas.mMarkerGraphEvents.size()==3 &&
            f.atlas.mMarkerGraphEvents[0].status=="accepted" &&
            f.atlas.mMarkerGraphEvents[1].status=="accepted" &&
            f.atlas.mMarkerGraphEvents[0].type=="marker_map_merge" &&
            f.atlas.mMarkerGraphEvents[1].type=="marker_map_merge" &&
            f.atlas.mMarkerGraphEvents[2].type=="marker_global_ba" &&
            f.atlas.mMarkerGraphEvents[2].status=="accepted" &&
            f.atlas.mMarkerMapAliases.at(f.source->GetId())==f.target->GetId() &&
            f.atlas.mMarkerMapAliases.at(f.unrelated->GetId())==f.target->GetId(),
            "recursive final drain lost an event or did not flatten both map aliases");
}

static void testIndependentMarkerInputAfterBA() {
    Fixture f;
    std::vector<Eigen::Vector3f> cached;
    std::vector<int> ids;
    for(const auto& tag:f.target->mStaticTags)
        for(int j=0;j<4;++j) {
            cached.emplace_back(tag.second[j*3],tag.second[j*3+1],tag.second[j*3+2]);
            ids.push_back(tag.first);
        }
    // Keep the physical square unchanged while BA adjusts one marker pose.
    auto& moved=f.target->mStaticTags.at(21);
    for(int j=0;j<4;++j) moved[j*3+2]+=.025f;
    f.target->mnMarkerGraphSequence=1;
    const Sophus::SE3f truth(Sophus::SO3f::exp(Eigen::Vector3f(.02f,-.04f,.01f)),
                            Eigen::Vector3f(.015f,-.01f,.02f));
    std::vector<cv::Point2f> pixels;
    std::vector<Eigen::Vector3f> expected;
    for(int i=0;i<int(ids.size());++i) {
        const auto& tag=f.target->mStaticTags.at(ids[i]); const int j=i%4;
        expected.emplace_back(tag[j*3],tag[j*3+1],tag[j*3+2]);
        const Eigen::Vector3f cameraPoint=truth.inverse()*expected.back();
        const Eigen::Vector2f uv=f.camera.project(cameraPoint);
        pixels.emplace_back(uv.x(),uv.y());
    }
    std::string reason;
    auto pose=truth; auto corners=cached;
    require(!MarkerGraphCoordinator::AlignMarkerInput(f.target,pose,corners,ids,{},false,reason),
            "rigid-only admission unexpectedly accepted independent layout changes");
    require(samePose(pose,truth) && corners==cached,"failed input alignment mutated observations");
    pose=truth; corners=cached;
    require(MarkerGraphCoordinator::AlignMarkerInput(f.target,pose,corners,ids,{},false,reason,
                                                     &f.camera,pixels),
            "committed independent corners were not used to refine the cached pose");
    require(samePose(pose,truth,1e-4f),"refined marker input did not recover the measured pose");
    require(corners==expected,"front-end factors still contain obsolete marker layout");
    // The same registered ID with a different physical size remains invalid.
    pose=truth; corners=cached; corners[1].x()+=.01f;
    const auto badCorners=corners; const auto oldGauge=f.target->mMarkerInputToWorld;
    require(!MarkerGraphCoordinator::AlignMarkerInput(f.target,pose,corners,ids,{},false,reason,
                                                      &f.camera,pixels) &&
            reason=="registered_marker_size_conflict" && samePose(pose,truth) &&
            corners==badCorners && samePose(oldGauge,f.target->mMarkerInputToWorld),
            "size conflict bypassed validation or changed live state");
    // Inconsistent pixels cannot be turned into a pose by accepting BA geometry.
    pose=truth; corners=cached; pixels[0].x+=50;
    require(!MarkerGraphCoordinator::AlignMarkerInput(f.target,pose,corners,ids,{},false,reason,
                                                      &f.camera,pixels) && samePose(pose,truth) &&
            corners==cached && samePose(oldGauge,f.target->mMarkerInputToWorld),
            "inconsistent pixels were accepted or changed live state");
}

static void testRegisteredWeakMarkerInputAfterBA() {
    Fixture f;
    // Reproduce KF212: marker 26 is soft-grid (4 x .25), while marker 45
    // supplies the strong pose. BA has changed their relative layout.
    const std::map<int,std::vector<float>> cachedTags{
        {26,f.target->mStaticTags.at(20)},{45,f.target->mStaticTags.at(21)}};
    f.target->mStaticTags=cachedTags;
    const Sophus::SE3f move26(Eigen::Matrix3f::Identity(),Eigen::Vector3f(0,0,.00637829f));
    f.target->mStaticTags[26]=transformCorners(cachedTags.at(26),move26);
    f.target->mnMarkerGraphSequence=1;
    const auto committed=f.target->mStaticTags;
    std::vector<Eigen::Vector3f> cached,expected;
    std::vector<int> ids;
    for(const auto& tag:cachedTags) for(int j=0;j<4;++j) {
        cached.emplace_back(tag.second[j*3],tag.second[j*3+1],tag.second[j*3+2]);
        const auto& current=committed.at(tag.first);
        expected.emplace_back(current[j*3],current[j*3+1],current[j*3+2]);
        ids.push_back(tag.first);
    }
    const Sophus::SE3f truth(Sophus::SO3f::exp(Eigen::Vector3f(.02f,-.04f,.01f)),
                            Eigen::Vector3f(.015f,-.01f,.02f));
    std::vector<cv::Point2f> pixels;
    for(const auto& point:expected) {
        const Eigen::Vector2f uv=f.camera.project(truth.inverse()*point);
        pixels.emplace_back(uv.x(),uv.y());
    }
    const auto originalPixels=pixels; const auto originalIds=ids;
    std::vector<float> weights{.25f,.25f,.25f,.25f,1.f,1.f,1.f,1.f};
    const auto originalWeights=weights;
    std::string reason; auto pose=truth; auto corners=cached;
    require(MarkerGraphCoordinator::AlignMarkerInput(f.target,pose,corners,ids,weights,false,
                                                     reason,&f.camera,pixels),
            "registered soft-grid marker rejected beside a strong marker");
    require(corners==expected && samePose(pose,truth,1e-5f),
            "weak 26 retained its obsolete layout or biased strong 45's pose");
    require(ids==originalIds && pixels==originalPixels && weights==originalWeights &&
            f.target->mStaticTags==committed,
            "canonical input synchronization changed pixels, weights, IDs or the map");
    const auto strongGauge=f.target->mMarkerInputToWorld;

    // A weak marker's old pose and even its pixels must not affect the strong
    // SVD/PnP seed. (Tracking separately checks the final combined residual.)
    const Sophus::SE3f staleWeak(Sophus::SO3f::exp(Eigen::Vector3f(.1f,.2f,-.3f)),
                               Eigen::Vector3f(.4f,-.2f,.3f));
    auto unrelatedWeak=cached;
    for(int j=0;j<4;++j) {
        unrelatedWeak[j]=staleWeak*unrelatedWeak[j];
        pixels[j]+=cv::Point2f(100.f,-80.f);
    }
    pose=truth; corners=unrelatedWeak;
    require(MarkerGraphCoordinator::AlignMarkerInput(f.target,pose,corners,ids,weights,false,
                                                     reason,&f.camera,pixels) &&
            samePose(pose,truth,1e-5f) && samePose(f.target->mMarkerInputToWorld,strongGauge) &&
            corners==expected,"weak geometry or pixels entered the strong pose alignment");
    pixels=originalPixels;

    // Switching the same ID back to strong must retain exactly the same
    // committed geometry, while the existing strong multi-marker PnP runs.
    weights.assign(8,1.f); pose=truth; corners=cached;
    require(MarkerGraphCoordinator::AlignMarkerInput(f.target,pose,corners,ids,weights,false,
                                                     reason,&f.camera,pixels) &&
            corners==expected && samePose(pose,truth,1e-4f),
            "weak-to-strong transition changed canonical corners or bypassed PnP");
    weights={1.f,1.f,1.f,1.f,.25f,.25f,.25f,.25f};
    pose=move26.inverse()*truth; corners=cached;
    require(MarkerGraphCoordinator::AlignMarkerInput(f.target,pose,corners,ids,weights,false,
                                                     reason,&f.camera,pixels) &&
            corners==expected && samePose(pose,truth,1e-5f),
            "strong 26/weak 45 transition retained the previous relative layout");

    // All-weak input may refresh references, but cannot establish a new pose
    // seed or input gauge. Native Tracking still requires a full strong group.
    const auto inheritedGauge=f.target->mMarkerInputToWorld;
    weights.assign(8,.25f); pose=truth; corners=cached;
    require(MarkerGraphCoordinator::AlignMarkerInput(f.target,pose,corners,ids,weights,false,
                                                     reason,&f.camera,pixels) &&
            corners==expected && samePose(pose,inheritedGauge*truth) &&
            samePose(f.target->mMarkerInputToWorld,inheritedGauge),
            "all-weak input was used to estimate a pose or a gauge");
    require(f.target->mStaticTags==committed,"input alignment changed the committed marker map");
}

static void testRegisteredWeakMarkerInputGuards() {
    Fixture f; f.target->mnMarkerGraphSequence=1;
    std::vector<Eigen::Vector3f> cached;
    std::vector<int> ids;
    for(const auto& tag:f.target->mStaticTags) for(int j=0;j<4;++j) {
        cached.emplace_back(tag.second[j*3],tag.second[j*3+1],tag.second[j*3+2]);
        ids.push_back(tag.first);
    }
    for(int j=0;j<4;++j) f.target->mStaticTags.at(20)[j*3+2]+=.02f;
    const Sophus::SE3f truth;
    std::vector<cv::Point2f> pixels;
    for(std::size_t i=0;i<ids.size();++i) {
        const auto& tag=f.target->mStaticTags.at(ids[i]); const int j=int(i%4);
        const Eigen::Vector2f uv=f.camera.project(Eigen::Vector3f(tag[j*3],tag[j*3+1],tag[j*3+2]));
        pixels.emplace_back(uv.x(),uv.y());
    }
    const std::vector<float> weights{.25f,.25f,.25f,.25f,1.f,1.f,1.f,1.f};
    const auto originalPixels=pixels; const auto originalWeights=weights;
    const MapState before(f.target);
    const auto rejected=[&](std::vector<Eigen::Vector3f> invalid,const std::string& label) {
        const auto original=invalid; auto pose=truth; std::string reason;
        require(!MarkerGraphCoordinator::AlignMarkerInput(f.target,pose,invalid,ids,weights,false,
                                                          reason,&f.camera,pixels),label+": admitted");
        require(samePose(pose,truth) && invalid==original && pixels==originalPixels &&
                weights==originalWeights,label+": failed alignment mutated an input");
        before.unchanged(label);
    };
    auto invalid=cached;
    const Eigen::Vector3f center=.25f*(cached[0]+cached[1]+cached[2]+cached[3]);
    for(int j=0;j<4;++j) invalid[j]=center+1.05f*(invalid[j]-center);
    rejected(invalid,"weak physical size mismatch");
    invalid=cached; std::swap(invalid[2],invalid[3]);
    rejected(invalid,"weak crossed corner order");
    invalid=cached; invalid[2]=invalid[1];
    rejected(invalid,"weak repeated corner");
    invalid=cached; invalid[2].z()+=.01f;
    rejected(invalid,"weak nonplanar square");

    // A strong PnP failure after staging canonical geometry must also be atomic.
    pixels[4].x+=80.f;
    auto corners=cached; auto pose=truth; std::string reason;
    auto strongWeights=weights; strongWeights[0]=strongWeights[1]=strongWeights[2]=strongWeights[3]=1.f;
    require(!MarkerGraphCoordinator::AlignMarkerInput(f.target,pose,corners,ids,strongWeights,false,
                                                      reason,&f.camera,pixels) &&
            samePose(pose,truth) && corners==cached,
            "failed strong PnP committed its staged canonical geometry");
    before.unchanged("strong PnP failure");
    pixels=originalPixels;

    // Partial corner sets do not carry their ordered indices. Even four
    // partial corners, interleaved IDs, or a duplicate group must not be
    // guessed into an ordered reference by their current 3D proximity.
    const auto notGuessed=[&](std::vector<Eigen::Vector3f> points,std::vector<int> pointIds,
                              bool partial,const std::string& label) {
        const auto original=points; auto inputPose=truth;
        const std::vector<float> weak(points.size(),.25f);
        std::string why;
        require(MarkerGraphCoordinator::AlignMarkerInput(f.target,inputPose,points,pointIds,weak,
                                                         partial,why) && points==original &&
                samePose(inputPose,truth),label+": guessed an unknown corner index");
        before.unchanged(label);
    };
    notGuessed({cached[1],cached[2],cached[3]},{20,20,20},true,"three partial corners");
    notGuessed({cached[1],cached[2],cached[3],cached[0]},{20,20,20,20},true,"four partial corners");
    notGuessed({cached[0],cached[1],cached[2]},{20,20,20},false,"incomplete full group");
    auto interleaved=ids; std::swap(interleaved[2],interleaved[5]);
    notGuessed(cached,interleaved,false,"interleaved marker IDs");
    auto duplicate=cached; duplicate.insert(duplicate.end(),cached.begin(),cached.begin()+4);
    auto duplicateIds=ids; duplicateIds.insert(duplicateIds.end(),4,20);
    // Marker 21 is already canonical; ambiguous marker 20 remains untouched.
    notGuessed(duplicate,duplicateIds,false,"duplicate marker group");
}

static void testRegisteredMarkerInputAtLargeWorldCoordinates() {
    Fixture f; f.target->mnMarkerGraphSequence=1;
    // Actual 48 mm Atlas corners from the 72.188888889 s / KF212 conflict.
    // Float storage at ~29 m makes one canonical diagonal differ from the
    // first-edge-derived square by 1.108e-4 relatively: this is not a bad ID.
    f.target->mStaticTags={
        {26,{10.3193826675f,-1.48772287369f,-28.8936195374f,
             10.2713899612f,-1.48776578903f,-28.8929538727f,
             10.2707252502f,-1.48923587799f,-28.9409294128f,
             10.3187217712f,-1.48919248581f,-28.9415931702f}},
        {45,{10.0183000565f,-1.48037779331f,-28.8951187134f,
             9.97089290619f,-1.48172175884f,-28.9025154114f,
             9.97828197479f,-1.48128831387f,-28.9499435425f,
             10.0256891251f,-1.4799476862f,-28.9425430298f}}};
    const std::vector<Eigen::Vector3f> cached{
        {10.3211078644f,-1.48561608791f,-28.8878517151f},
        {10.2731151581f,-1.48639142513f,-28.887462616f},
        {10.2727460861f,-1.48733413219f,-28.9354534149f},
        {10.3207387924f,-1.48655879498f,-28.9358386993f},
        {10.0183010101f,-1.48037874699f,-28.895116806f},
        {9.97089290619f,-1.48172104359f,-28.9025154114f},
        {9.97828197479f,-1.48128926754f,-28.9499435425f},
        {10.0256900787f,-1.47994697094f,-28.9425430298f}};
    const std::vector<int> ids{26,26,26,26,45,45,45,45};
    std::vector<Eigen::Vector3f> expected;
    for(const auto& tag:f.target->mStaticTags) for(int j=0;j<4;++j)
        expected.emplace_back(tag.second[j*3],tag.second[j*3+1],tag.second[j*3+2]);
    const double firstSide=(expected[1]-expected[0]).cast<double>().norm();
    double quantization=0;
    for(int j=0;j<4;++j) for(int k=j+1;k<4;++k) {
        const double ideal=firstSide*((k-j==2)?std::sqrt(2.):1.);
        quantization=std::max(quantization,std::abs((expected[k]-expected[j]).cast<double>().norm()/ideal-1.));
    }
    require(quantization>1e-4 && (cached[0]-expected[0]).norm()>.006f,
            "large-world regression lost the actual quantization/stale-layout evidence");
    const Sophus::SE3f truth(Eigen::Matrix3f::Identity(),Eigen::Vector3f(10.f,-1.f,-30.f));
    std::vector<cv::Point2f> pixels;
    for(const auto& point:expected) {
        const Eigen::Vector2f uv=f.camera.project(truth.inverse()*point);
        pixels.emplace_back(uv.x(),uv.y());
    }
    for(const float weight:{.25f,1.f}) {
        std::vector<float> weights{weight,weight,weight,weight,1.f,1.f,1.f,1.f};
        auto pose=truth; auto corners=cached; std::string reason;
        require(MarkerGraphCoordinator::AlignMarkerInput(f.target,pose,corners,ids,weights,false,
                                                         reason,&f.camera,pixels),
                "float-quantized real-world marker geometry rejected: "+reason);
        require(corners==expected,"large-world strong/weak input did not use exact canonical corners");
    }
}

static void testNewComponentAlreadyInAtlasWorld() {
    Fixture f;
    f.target->mMarkerInputToWorld=Sophus::SE3f(
        Sophus::SO3f::exp(Eigen::Vector3f(0,0,.015f)),Eigen::Vector3f(.002f,0,0));
    const auto oldGauge=f.target->mMarkerInputToWorld;
    const Sophus::SE3f measured(Eigen::Matrix3f::Identity(),Eigen::Vector3f(30.f,-.2f,0));
    std::vector<Eigen::Vector3f> points{{29.976f,-.024f,1},{30.024f,-.024f,1},
                                      {30.024f,.024f,1},{29.976f,.024f,1}};
    const std::vector<int> ids(4,49);
    require(!f.target->mStaticTags.count(49),"new-component test ID already registered");
    auto pose=measured; auto legacyPoints=points; std::string reason;
    require(MarkerGraphCoordinator::AlignMarkerInput(f.target,pose,legacyPoints,ids,{},false,reason),
            "legacy alignment failed before exposing double transform");
    require((pose.translation()-measured.translation()).norm()>.4f,
            "fixture did not expose distant-component double transformation");
    f.target->mMarkerInputToWorld=oldGauge;
    pose=measured; auto aligned=points;
    require(MarkerGraphCoordinator::AlignMarkerInput(f.target,pose,aligned,ids,{},false,reason,
                                                     nullptr,{},true),"Atlas-world input rejected");
    require(samePose(pose,measured) && aligned==points,
            "new Atlas-aligned component inherited the previous component's gauge");
}

static void testInitialCornerScaleUnits() {
    Fixture f;
    auto views=f.target->GetAllKeyFrames();
    std::sort(views.begin(),views.end(),KeyFrame::lId);
    for(auto* k:views) {
        // One physical marker suffices for observable arbitrary-unit scale;
        // unlike a very large correction to an already metric map.
        for(std::size_t i=k->mvTagIds.size();i-- >0;) if(k->mvTagIds[i]!=20) {
            k->mvTagIds.erase(k->mvTagIds.begin()+i);
            k->mvTagWorldPoints.erase(k->mvTagWorldPoints.begin()+i);
            k->mvTagImagePoints.erase(k->mvTagImagePoints.begin()+i);
            k->mvTagPointWeights.erase(k->mvTagPointWeights.begin()+i);
        }
        k->mbTagObservationActive=false;
        // Stay away from the floating-point 4 cm boundary in this unit test.
        auto physicalPose=k->GetPose();physicalPose.translation()*=1.5f;k->SetPose(physicalPose);
        for(std::size_t i=0;i<k->mvTagImagePoints.size();++i) {
            const auto uv=f.camera.project(physicalPose*k->mvTagWorldPoints[i]);
            k->mvTagImagePoints[i]=cv::Point2f(uv.x(),uv.y());
        }
    }
    std::vector<Sophus::SE3f> saved;
    for(auto* k:views)saved.push_back(k->GetPose());
    for(float visualPerMetric:{.01f,.1f,1.f,10.f}) {
        for(std::size_t i=0;i<views.size();++i) {
            auto pose=saved[i];pose.translation()*=visualPerMetric;views[i]->SetPose(pose);
        }
        for(std::size_t count:{1,2,3}) {
            const std::vector<KeyFrame*> subset(views.begin(),views.begin()+count);
            const auto cue=MarkerGraphOptimizer::EstimateInitialCornerScale(subset);
            std::cout<<"INITIAL_CUE_TEST unit="<<visualPerMetric<<" count="<<count
                     <<" valid="<<cue.valid<<" scale="<<cue.scale<<" sigma="<<cue.sigma
                     <<" markers="<<cue.markers<<" rms="<<cue.rms<<std::endl;
            require(cue.valid==(count==3),"initial corner scale miscounted multi-view support");
            if(count==3)require(std::abs(cue.scale*visualPerMetric-1.)<.005,"arbitrary unit changed recovered metric scale");
        }
    }
    for(auto* k:views)k->SetPose(saved.front());
    require(!MarkerGraphOptimizer::EstimateInitialCornerScale(views).valid,"zero parallax accepted");
    for(std::size_t i=0;i<views.size();++i) {
        views[i]->SetPose(saved[i]);
        std::fill(views[i]->mvTagPointWeights.begin(),views[i]->mvTagPointWeights.end(),.25f);
    }
    require(!MarkerGraphOptimizer::EstimateInitialCornerScale(views).valid,"weak marker established scale");
    for(std::size_t i=0;i<views.size();++i) {
        std::fill(views[i]->mvTagPointWeights.begin(),views[i]->mvTagPointWeights.end(),1.f);
        for(auto& id:views[i]->mvTagIds)id=20+int(i)*100;
    }
    require(!MarkerGraphOptimizer::EstimateInitialCornerScale(views).valid,"unrelated one-view markers established scale");
    for(auto* k:views)for(auto& id:k->mvTagIds)id=20;
    views[1]->mvTagImagePoints[0].x+=60;
    require(!MarkerGraphOptimizer::EstimateInitialCornerScale(views).valid,"damaged corner established scale");
    views[1]->mvTagImagePoints[0].x-=60;
    for(std::size_t i=0;i<views.size();++i) {
        auto pose=saved[i];pose.translation()*=.1f;views[i]->SetPose(pose);
        views[i]->mbTagObservationActive=true;
    }
    require(!MarkerGraphOptimizer::EstimateCornerScale(views).valid,"ordinary extreme reanchor protection changed");
    require(MarkerGraphOptimizer::EstimateInitialCornerScale(views).valid,"valid arbitrary-unit initialization lost");
    std::cout<<"INITIAL_CORNER_SCALE_UNITS_OK\n";
}

int main(int argc,char** argv) {
    if(argc>3) return 2;
    try {
        if(argc==2 && std::string(argv[1])=="--initial-corner-scale-only") {
            testInitialCornerScaleUnits();return 0;
        }
        if(argc==3 && std::string(argv[1])=="--corner-revisit-runtime") {
            testRuntimeCornerRevisitScheduling(argv[2]); return 0;
        }
        if(argc>2) return 2;
        if(argc==2 && std::string(argv[1])=="--unsurveyed-markers-only") {
            testUnsurveyedOriginMarker(); return 0;
        }
        if(argc==2 && std::string(argv[1])=="--corner-revisit-only") {
            testScaleEvidence(); testCornerRevisitWindow(); return 0;
        }
        if(argc==2 && std::string(argv[1])=="--refine-gauge-only") {
            testMarkerGlobalBACommit(); testFreeMarkerRefineDoesNotMoveWorldGauge();
            std::cout << "{\"refine_gauge_regressions\":true}" << std::endl;
            return 0;
        }
        if(argc==2 && std::string(argv[1])=="--marker-retry-only") {
            testIsolatedMarkerRetryGuards();testSoleRevisitedMarkerCannotBeExcluded();
            return 0;
        }
        if(argc==2 && std::string(argv[1])=="--marker-input-only") {
            testIndependentMarkerInputAfterBA(); testRegisteredWeakMarkerInputAfterBA();
            testRegisteredWeakMarkerInputGuards(); testRegisteredMarkerInputAtLargeWorldCoordinates();
            testNewComponentAlreadyInAtlasWorld();
            std::cout << "{\"marker_input_regressions\":5,\"passed\":true}" << std::endl;
            return 0;
        }
        testEssentialGraphUsesMarkerCornersWithoutFreezingMarkerKeyframes();
        testIndependentMarkerInputAfterBA();
        testRegisteredWeakMarkerInputAfterBA();
        testRegisteredWeakMarkerInputGuards();
        testRegisteredMarkerInputAtLargeWorldCoordinates();
        testNewComponentAlreadyInAtlasWorld();
        testScaleEvidence(); testCornerRevisitWindow(); testScaleCommit(); testRejectedScaleIsAtomic();
        testMergeCommit(); testRejectedMergeIsAtomic(); testMarkerGlobalBACommit();
        testFreeMarkerRefineDoesNotMoveWorldGauge();
        testWeakCornersCannotCreateMarkerVariables();
        testIsolatedMarkerRetryGuards();
        testSoleRevisitedMarkerCannotBeExcluded();
        testWiderProjectionSearchFaults();
        testCalibratedBoardUsesOneRigidMarkerVariable();
        testCulledReferenceGraphChain();
        testVersionedAtlasRoundTrip(); testConcurrentAtlasRetirement();
        if(argc==2) {
            testRecoveryMarkerGate(argv[1]);
            testInitialScaleWindowBoundaries(argv[1]);
            testRuntimeScaleScheduling(argv[1]);
            testRuntimeScaleScheduling(argv[1],true);
            testRuntimeInactiveSourceMerge(argv[1]);
            testCancelledMergeCanReschedule(argv[1]);
            testLocalizationFinalProcessesPending(argv[1]);
            testFinalDrainsRecursiveMergeChain(argv[1]);
        }
        std::cout << "{\"marker_graph_coordinator_regressions\":" << (argc==2?22:14)
                  << ",\"passed\":true}" << std::endl;
        return 0;
    } catch(const std::exception& error) {
        std::cerr << "marker_graph_coordinator_regression: " << error.what() << std::endl;
        return 1;
    }
}
