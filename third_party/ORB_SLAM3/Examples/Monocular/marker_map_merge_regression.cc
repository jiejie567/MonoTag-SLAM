// Deterministic common-marker merge proposals. No camera or vocabulary load.
#include "MarkerMapMerge.h"
#include "CameraModels/Pinhole.h"
#include "Frame.h"
#include "KeyFrame.h"
#include "Map.h"
#include "MapPoint.h"
#include "ORBextractor.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>

using namespace ORB_SLAM3;

static void require(bool condition,const std::string& message)
{
    if(!condition) throw std::runtime_error(message);
}

static std::vector<float> square(const Eigen::Vector3f& center,float side)
{
    std::vector<float> values;
    for(const Eigen::Vector2f& offset:std::vector<Eigen::Vector2f>{{-1,-1},{1,-1},{1,1},{-1,1}}) {
        const Eigen::Vector3f point=center+Eigen::Vector3f(offset.x()*side/2,offset.y()*side/2,0);
        for(int axis=0;axis<3;++axis) values.push_back(point(axis));
    }
    return values;
}

static std::vector<float> transformCorners(const std::vector<float>& values,const Sophus::SE3f& transform)
{
    std::vector<float> result;
    for(std::size_t i=0;i<values.size();i+=3) {
        const Eigen::Vector3f point=transform*Eigen::Vector3f(values[i],values[i+1],values[i+2]);
        for(int axis=0;axis<3;++axis) result.push_back(point(axis));
    }
    return result;
}

struct Fixture {
    Pinhole camera{std::vector<float>{500,500,320,240}};
    ORBextractor extractor{180,1.2f,8,20,7};
    Map target{static_cast<int>(KeyFrame::nNextId)},source{static_cast<int>(KeyFrame::nNextId)};
    Sophus::SE3f sourceToTarget{Sophus::SO3f::exp(Eigen::Vector3f(.13f,-.21f,.07f)),
                              Eigen::Vector3f(.4f,-.2f,.08f)};
    std::unique_ptr<Frame> base;
    std::vector<std::unique_ptr<KeyFrame>> keyframes;
    std::vector<std::unique_ptr<MapPoint>> points;
    MarkerGraphOptimizer::PoseMap originalPoses;
    MarkerGraphOptimizer::PointMap originalPoints;

    explicit Fixture(bool duplicateFrameIds=false,int corruptBackground=0)
    {
        target.mbMetric=source.mbMetric=true;
        target.mbBackgroundReady=source.mbBackgroundReady=true;
        target.mMetricScale=source.mMetricScale=1;
        target.mStaticTags[20]=square(Eigen::Vector3f(0,0,1),.08f);
        target.mStaticTags[21]=square(Eigen::Vector3f(.2f,.03f,1.05f),.08f);
        for(const auto& tag:target.mStaticTags)
            source.mStaticTags[tag.first]=transformCorners(tag.second,sourceToTarget.inverse());
        source.mStaticTags[22]=transformCorners(square(Eigen::Vector3f(-.2f,.04f,1.1f),.06f),sourceToTarget.inverse());
        cv::Mat image(480,640,CV_8UC1),distortion=cv::Mat::zeros(4,1,CV_32F);
        cv::RNG random(9876); random.fill(image,cv::RNG::UNIFORM,0,256);
        base.reset(new Frame(image,0,&extractor,nullptr,&camera,distortion,0,1));
        base->SetPose(Sophus::SE3f());
        require(base->N>=40,"fixture did not obtain descriptor storage");
        populate(target,Sophus::SE3f(),duplicateFrameIds,false);
        populate(source,sourceToTarget,duplicateFrameIds,corruptBackground);
    }

    void populate(Map& map,const Sophus::SE3f& toTarget,bool duplicateFrameIds,int corruptBackground)
    {
        std::vector<Eigen::Vector3f> world;
        for(int i=0;i<40;++i)
            world.push_back(toTarget.inverse()*Eigen::Vector3f((i%8-3.5f)*.045f,(i/8-2.f)*.04f,1.2f+.03f*(i%3)));
        std::vector<KeyFrame*> views;
        const unsigned long firstFrame=Frame::nNextId;
        for(int view=0;view<3;++view) {
            Frame frame(*base);
            frame.mnId=duplicateFrameIds?firstFrame:Frame::nNextId++;
            frame.mTimeStamp=(map.GetId()+1)*10.0+view*.15;
            const Sophus::SE3f truth= Sophus::SE3f(Eigen::Matrix3f::Identity(),
                                      Eigen::Vector3f(-.02f*view,0,0))*toTarget;
            frame.SetPose(truth);
            for(std::size_t i=0;i<world.size();++i) {
                const Eigen::Vector3f point=truth*world[i];
                const auto pixel=camera.project(point);
                cv::Point2f observed(pixel.x(),pixel.y());
                if(view==1 && corruptBackground==1) observed.y+=30;
                if(view==1 && corruptBackground==2) {
                    observed.x+=i%2?50:-50;
                    observed.y+=(i/2)%2?45:-45;
                }
                frame.mvKeysUn[i]=cv::KeyPoint(observed,1);
            }
            keyframes.emplace_back(new KeyFrame(frame,&map,nullptr));
            KeyFrame* keyframe=keyframes.back().get();
            keyframe->mbHasTagObservation=keyframe->mbTagObservationActive=true;
            keyframe->mTagObservationConfidence=1;
            for(const auto& tag:map.mStaticTags) for(std::size_t j=0;j<4;++j) {
                const Eigen::Vector3f point(tag.second[j*3],tag.second[j*3+1],tag.second[j*3+2]);
                const Eigen::Vector3f cameraPoint=truth*point;
                const auto pixel=camera.project(cameraPoint);
                keyframe->mvTagIds.push_back(tag.first);
                keyframe->mvTagWorldPoints.push_back(point);
                keyframe->mvTagImagePoints.emplace_back(pixel.x(),pixel.y());
                keyframe->mvTagPointWeights.push_back(1);
            }
            map.AddKeyFrame(keyframe);
            if(!views.empty()) keyframe->ChangeParent(views.back());
            views.push_back(keyframe);
            originalPoses[keyframe]=keyframe->GetPose();
        }
        if(duplicateFrameIds) ++Frame::nNextId;
        for(std::size_t i=0;i<world.size();++i) {
            points.emplace_back(new MapPoint(world[i],views.front(),&map));
            MapPoint* point=points.back().get();
            for(KeyFrame* keyframe:views) {point->AddObservation(keyframe,i);keyframe->AddMapPoint(point,i);}
            map.AddMapPoint(point);
            originalPoints[point]=point->GetWorldPos();
        }
        for(KeyFrame* keyframe:views) keyframe->UpdateConnections();
    }

    void requireUnchanged() const
    {
        for(const auto& original:originalPoses)
            require((original.first->GetPose().matrix()-original.second.matrix()).norm()<1e-7f,
                    "proposal changed a live keyframe");
        for(const auto& original:originalPoints)
            require((original.first->GetWorldPos()-original.second).norm()<1e-7f,
                    "proposal changed a live map point");
    }

    void changeSourceMarker(int id,const std::vector<float>& values)
    {
        source.mStaticTags[id]=values;
        for(KeyFrame* keyframe:source.GetAllKeyFrames())
            for(std::size_t i=0;i<keyframe->mvTagIds.size();i+=4) if(keyframe->mvTagIds[i]==id)
                for(std::size_t j=0;j<4;++j) {
                    const Eigen::Vector3f point(values[j*3],values[j*3+1],values[j*3+2]);
                    keyframe->mvTagWorldPoints[i+j]=point;
                    const Eigen::Vector3f cameraPoint=keyframe->GetPose()*point;
                    const auto pixel=camera.project(cameraPoint);
                    keyframe->mvTagImagePoints[i+j]=cv::Point2f(pixel.x(),pixel.y());
                }
    }
};

static void testRigidMerge()
{
    Fixture data;
    const auto targetTags=data.target.mStaticTags,sourceTags=data.source.mStaticTags;
    auto proposal=MarkerMapMerge::Propose(&data.target,&data.source);
    require(proposal.accepted,"correct metric merge rejected: "+proposal.reason);
    require((proposal.sourceToTarget.matrix()-data.sourceToTarget.matrix()).norm()<1e-5f,
            "source-to-target SE3 direction/rotation incorrect");
    require(proposal.verifiedMarkerIds.size()==2 && proposal.evidence.at(20).sourceFrames==3 &&
            proposal.evidence.at(20).targetFrames==3,"independent raw corner evidence missing");
    require(proposal.graph.after.tagCorners==60 && proposal.graph.after.backgroundObservations==240,
            "real tag and background reprojection factors were not used");
    for(const auto& original:data.originalPoses) {
        const Sophus::SE3f expected=original.first->GetMap()==&data.source
            ?original.second*data.sourceToTarget.inverse():original.second;
        require((proposal.graph.keyframePoses.at(original.first).matrix()-expected.matrix()).norm()<2e-4f,
                "camera was not transformed with the marker gauge");
        require(proposal.graph.replayScaleMultipliers.at(original.first)==1,
                "metric SE3 merge applied scale again");
    }
    for(const auto& original:data.originalPoints) {
        const Eigen::Vector3f expected=original.first->GetMap()==&data.source
            ?data.sourceToTarget*original.second:original.second;
        require((proposal.graph.pointPositions.at(original.first)-expected).norm()<2e-4f,
                "background point did not share the camera/marker transformation");
    }
    for(const auto& tag:targetTags) require(proposal.staticTags.at(tag.first)==tag.second,"target marker moved");
    const auto expectedExclusive=transformCorners(sourceTags.at(22),data.sourceToTarget);
    for(std::size_t i=0;i<12;++i)
        require(std::abs(expectedExclusive[i]-proposal.staticTags.at(22)[i])<1e-5,
                "source-only marker lost its physical geometry");
    require((proposal.graph.keyframePoses.at(data.target.GetOriginKF()).matrix()-
             data.target.GetOriginKF()->GetPose().matrix()).norm()<1e-7f,"target A root moved");
    require(data.target.mStaticTags==targetTags && data.source.mStaticTags==sourceTags,"live marker registry changed");
    data.requireUnchanged();
}

static void testScaleAndIdentity()
{
    Fixture data;
    data.source.mbMetric=false;
    auto result=MarkerMapMerge::Propose(&data.target,&data.source);
    require(!result.accepted && result.reason=="source_scale_unknown","unscaled source was merged as metres");
    data.source.mbMetric=true;
    MarkerMapMerge::StaticTags renamed;
    for(const auto& tag:data.source.mStaticTags) renamed[tag.first+100]=tag.second;
    data.source.mStaticTags=renamed;
    for(KeyFrame* keyframe:data.source.GetAllKeyFrames()) for(int& id:keyframe->mvTagIds) id+=100;
    result=MarkerMapMerge::Propose(&data.target,&data.source);
    require(!result.accepted && result.reason=="no_common_marker","same size but different IDs were merged");
    data.requireUnchanged();
}

static void testSingleCommonMarker()
{
    Fixture data;
    // One physical square is sufficient to define SE3 once both maps are
    // already metric; do not require two IDs or fit an unobservable scale.
    data.source.mStaticTags.erase(21);
    for(KeyFrame* keyframe:data.source.GetAllKeyFrames()) {
        for(std::size_t i=keyframe->mvTagIds.size();i>0;) {
            i-=4;
            if(keyframe->mvTagIds[i]!=21) continue;
            keyframe->mvTagIds.erase(keyframe->mvTagIds.begin()+i,keyframe->mvTagIds.begin()+i+4);
            keyframe->mvTagWorldPoints.erase(keyframe->mvTagWorldPoints.begin()+i,keyframe->mvTagWorldPoints.begin()+i+4);
            keyframe->mvTagImagePoints.erase(keyframe->mvTagImagePoints.begin()+i,keyframe->mvTagImagePoints.begin()+i+4);
            keyframe->mvTagPointWeights.erase(keyframe->mvTagPointWeights.begin()+i,keyframe->mvTagPointWeights.begin()+i+4);
        }
    }
    const auto result=MarkerMapMerge::Propose(&data.target,&data.source);
    require(result.accepted && result.verifiedMarkerIds==std::vector<int>{20},
            "one common planar marker with repeated observations was rejected: "+result.reason);
    require((result.sourceToTarget.matrix()-data.sourceToTarget.matrix()).norm()<1e-5f,
            "planar common-marker SE3 alignment is incorrect");
    data.requireUnchanged();
}

static void testAsymmetricShortSubmapEvidence()
{
    Fixture data;
    auto sourceFrames=data.source.GetAllKeyFrames();
    std::sort(sourceFrames.begin(),sourceFrames.end(),[](KeyFrame* a,KeyFrame* b) {
        return a->mTimeStamp<b->mTimeStamp;
    });
    require(sourceFrames.size()==3,"fixture no longer has three source keyframes");
    sourceFrames[1]->mbTagObservationActive=false;
    auto result=MarkerMapMerge::Propose(&data.target,&data.source);
    require(result.accepted && result.evidence.at(20).targetFrames==3 &&
            result.evidence.at(20).sourceFrames==2,
            "two-view short submap was not supported by the established map: "+result.reason);
    sourceFrames[0]->mbTagObservationActive=false;
    result=MarkerMapMerge::Propose(&data.target,&data.source);
    require(!result.accepted && result.reason=="insufficient_common_marker_evidence",
            "one-frame marker evidence was allowed to merge maps");
    data.requireUnchanged();
}

static void testJointOptimizationAndRejection()
{
    Fixture data;
    const auto truth=data.originalPoints;
    for(MapPoint* point:data.source.GetAllMapPoints()) {
        point->SetWorldPos(point->GetWorldPos()+Eigen::Vector3f(.003f,-.002f,.01f));
        data.originalPoints[point]=point->GetWorldPos();
    }
    auto result=MarkerMapMerge::Propose(&data.target,&data.source);
    require(result.accepted,"small background errors did not refine: "+result.reason);
    require(result.graph.before.backgroundRmsPx>.1 && result.graph.after.backgroundRmsPx<.002,
            "joint BA did not actually optimize original background pixel constraints");
    for(MapPoint* point:data.source.GetAllMapPoints())
        require((result.graph.pointPositions.at(point)-data.sourceToTarget*truth.at(point)).norm()<2e-4f,
                "joint BA did not recover metric background depth");
    data.requireUnchanged();

    Fixture corrupted(false,1);
    result=MarkerMapMerge::Propose(&corrupted.target,&corrupted.source);
    {
        double maximumCameraShift=0,maximumPointShift=0;
        for(const auto& original:corrupted.originalPoses) {
            const Sophus::SE3f expected=original.first->GetMap()==&corrupted.source
                ?original.second*corrupted.sourceToTarget.inverse():original.second;
            maximumCameraShift=std::max(maximumCameraShift,double((result.graph.keyframePoses.at(original.first).inverse().translation()-expected.inverse().translation()).norm()));
        }
        for(const auto& original:corrupted.originalPoints) {
            const Eigen::Vector3f expected=original.first->GetMap()==&corrupted.source
                ?corrupted.sourceToTarget*original.second:original.second;
            maximumPointShift=std::max(maximumPointShift,double((result.graph.pointPositions.at(original.first)-expected).norm()));
        }
        std::cerr << "uniform-shift diagnostics (accepted=" << result.accepted << "): tag " << result.graph.before.tagRmsPx << " -> "
                  << result.graph.after.tagRmsPx << ", background " << result.graph.before.backgroundRmsPx
                  << " -> " << result.graph.after.backgroundRmsPx << ", max camera shift "
                  << maximumCameraShift << "m, max point shift " << maximumPointShift << "m" << std::endl;
    }
    // A uniform displacement with a very short straight baseline can admit
    // another positive-depth reconstruction. Its known synthetic GT error is
    // diagnostic, not proof that raw pixel constraints must be inconsistent.
    if(result.accepted) {
        require(result.graph.after.tagRmsPx<=2.5 && result.graph.after.backgroundRmsPx<=3,
                "accepted uniform shift did not satisfy reported pixel gates");
        for(const auto& residual:result.graph.after.backgroundRmsByKeyframe)
            require(residual.second<=3,"a bad keyframe was hidden in the global background mean");
    } else require(result.reason.find("joint_ba_")==0,"uniform shift was not checked by joint BA");
    corrupted.requireUnchanged();

    Fixture mismatched(false,2);
    result=MarkerMapMerge::Propose(&mismatched.target,&mismatched.source);
    require(!result.accepted && result.reason.find("joint_ba_")==0,
            "alternating spatially inconsistent ORB matches bypassed joint BA: "+result.reason);
    mismatched.requireUnchanged();
}

static void testGeometricCounterEvidence()
{
    Fixture data;
    auto values=data.source.mStaticTags.at(21);
    for(std::size_t i=0;i<values.size();i+=3) values[i]+=.08f;
    data.changeSourceMarker(21,values);
    auto result=MarkerMapMerge::Propose(&data.target,&data.source);
    require(!result.accepted && result.reason=="common_marker_layout_conflict",
            "a contradictory second common marker was ignored: "+result.reason);

    Fixture differentSize;
    values=differentSize.source.mStaticTags.at(20);
    Eigen::Vector3f center=Eigen::Vector3f::Zero();
    for(std::size_t j=0;j<4;++j) center+=Eigen::Vector3f(values[j*3],values[j*3+1],values[j*3+2])*.25f;
    for(std::size_t j=0;j<4;++j) for(int axis=0;axis<3;++axis)
        values[j*3+axis]=center(axis)+1.2f*(values[j*3+axis]-center(axis));
    differentSize.changeSourceMarker(20,values);
    result=MarkerMapMerge::Propose(&differentSize.target,&differentSize.source);
    require(!result.accepted && result.reason=="marker_size_mismatch","physical marker size was silently rescaled");
    data.requireUnchanged(); differentSize.requireUnchanged();
}

static void testRawEvidence()
{
    Fixture data;
    KeyFrame* first=data.source.GetAllKeyFrames().front();
    first->mvTagImagePoints[0].x+=20;
    auto result=MarkerMapMerge::Propose(&data.target,&data.source);
    require(result.accepted && result.evidence.at(20).sourceFrames==2,
            "a damaged raw-corner observation was not excluded from merge evidence");
    first->mvTagImagePoints[0].x-=20;
    for(KeyFrame* keyframe:data.source.GetAllKeyFrames())
        for(float& weight:keyframe->mvTagPointWeights) weight=.25f;
    result=MarkerMapMerge::Propose(&data.target,&data.source);
    require(!result.accepted && result.reason=="insufficient_common_marker_evidence","weak corners triggered a merge");
    Fixture duplicates(true);
    result=MarkerMapMerge::Propose(&duplicates.target,&duplicates.source);
    require(!result.accepted && result.reason=="insufficient_common_marker_evidence",
            "multiple keyframes of one image counted as independent evidence");
    data.requireUnchanged(); duplicates.requireUnchanged();
}

static void testFullMarkerWithPartialAuxiliaryCorners()
{
    Fixture data;
    for(KeyFrame* keyframe:data.source.GetAllKeyFrames()) {
        // Production appends arbitrary LK corners after the complete marker:
        // four decoded ID20 + three weak ID21 + two weak ID22 = nine corners.
        for(std::size_t i=keyframe->mvTagIds.size();i>0;) {
            --i;
            const int id=keyframe->mvTagIds[i];
            const bool remove=(id==21 && i%4==3) || (id==22 && i%4>=2);
            if(remove) {
                keyframe->mvTagIds.erase(keyframe->mvTagIds.begin()+i);
                keyframe->mvTagWorldPoints.erase(keyframe->mvTagWorldPoints.begin()+i);
                keyframe->mvTagImagePoints.erase(keyframe->mvTagImagePoints.begin()+i);
                keyframe->mvTagPointWeights.erase(keyframe->mvTagPointWeights.begin()+i);
            } else if(id!=20) keyframe->mvTagPointWeights[i]=.25f;
        }
    }
    auto result=MarkerMapMerge::Propose(&data.target,&data.source);
    require(result.accepted && result.verifiedMarkerIds==std::vector<int>{20},
            "legitimate full+partial corner stream rejected: "+result.reason);
    for(KeyFrame* keyframe:data.source.GetAllKeyFrames()) {
        const auto& staged=result.graph.tagWorldCorners.at(keyframe);
        require(staged.size()==9 && staged.size()==keyframe->mvTagImagePoints.size(),
                "partial corner factors were discarded or regrouped");
        for(std::size_t i=0;i<staged.size();++i)
            require((staged[i]-data.sourceToTarget*keyframe->mvTagWorldPoints[i]).norm()<1e-5f,
                    "partial marker override lost its original pixel-corner correspondence");
        keyframe->mvTagPointWeights[0]=.25f;
    }
    result=MarkerMapMerge::Propose(&data.target,&data.source);
    require(!result.accepted && result.reason=="insufficient_common_marker_evidence",
            "a partial/weak fourth corner completed a strong merge anchor");
    data.requireUnchanged();
}

static void testInvalidInput()
{
    Fixture data;
    KeyFrame* first=data.source.GetAllKeyFrames().front();
    first->mvTagImagePoints[0].x=std::numeric_limits<float>::quiet_NaN();
    auto result=MarkerMapMerge::Propose(&data.target,&data.source);
    require(!result.accepted && result.reason=="source_invalid_marker_observation","nonfinite pixels accepted");
    const auto pixel=data.camera.project(Eigen::Vector3f(first->GetPose()*first->mvTagWorldPoints[0]));
    first->mvTagImagePoints[0].x=pixel.x();
    first->mvTagWorldPoints[0].x()+=.01f;
    result=MarkerMapMerge::Propose(&data.target,&data.source);
    require(!result.accepted && result.reason=="source_marker_observation_layout_mismatch",
            "raw factor and registered marker disagreement accepted");
    first->mvTagWorldPoints[0].x()-=.01f;
    first->mvTagIds.pop_back();
    result=MarkerMapMerge::Propose(&data.target,&data.source);
    require(!result.accepted && result.reason=="source_invalid_marker_observation","truncated IDs accepted");
    data.requireUnchanged();
}

static void testForeignPointObserver()
{
    Fixture data;
    Map unrelated(static_cast<int>(KeyFrame::nNextId));
    Frame frame(*data.base);
    frame.mnId=Frame::nNextId++; frame.mTimeStamp=999;
    std::unique_ptr<KeyFrame> foreign(new KeyFrame(frame,&unrelated,nullptr));
    unrelated.AddKeyFrame(foreign.get());
    MapPoint* point=data.source.GetAllMapPoints().front();
    point->AddObservation(foreign.get(),0); foreign->AddMapPoint(point,0);
    const auto original=foreign->GetPose();
    const auto result=MarkerMapMerge::Propose(&data.target,&data.source);
    require(!result.accepted && result.reason=="invalid_point_observation_membership",
            "source point observed in a third map was moved without that observer");
    require((foreign->GetPose().matrix()-original.matrix()).norm()<1e-7f,
            "unrelated map changed during rejected proposal");
    point->EraseObservation(foreign.get()); foreign->EraseMapPointMatch(point);
    data.requireUnchanged();
}

int main()
{
    try {
        testRigidMerge();
        testScaleAndIdentity();
        testSingleCommonMarker();
        testAsymmetricShortSubmapEvidence();
        testJointOptimizationAndRejection();
        testGeometricCounterEvidence();
        testRawEvidence();
        testFullMarkerWithPartialAuxiliaryCorners();
        testInvalidInput();
        testForeignPointObserver();
        std::cout << "{\"marker_map_merge_regressions\":10,\"passed\":true}" << std::endl;
        return 0;
    } catch(const std::exception& error) {
        std::cerr << "marker_map_merge_regression: " << error.what() << std::endl;
        return 1;
    }
}
