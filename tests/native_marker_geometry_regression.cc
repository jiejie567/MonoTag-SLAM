// Actual proposal-only BA regressions; never start SLAM, workers or commit.
#include "MarkerGraphOptimizer.h"
#include "CameraModels/Pinhole.h"
#include "Frame.h"
#include "KeyFrame.h"
#include "KeyFrameDatabase.h"
#include "Map.h"
#include "ORBextractor.h"
#include <algorithm>
#include <iostream>
#include <memory>
#include <stdexcept>

using namespace ORB_SLAM3;
using MGO=MarkerGraphOptimizer;
static void require(bool ok,const std::string& why) {if(!ok) throw std::runtime_error(why);}
static bool sameCorners(const std::vector<Eigen::Vector3f>& a,const std::vector<Eigen::Vector3f>& b) {
    if(a.size()!=b.size()) return false;
    for(std::size_t i=0;i<a.size();++i) if((a[i]-b[i]).norm()!=0) return false;
    return true;
}

struct Fixture {
    Map map{int(KeyFrame::nNextId)};
    ORBVocabulary vocabulary;
    KeyFrameDatabase database{vocabulary};
    Pinhole camera{std::vector<float>{500,500,320,240}};
    ORBextractor extractor{100,1.2f,8,20,7};
    std::vector<std::unique_ptr<KeyFrame>> owned;
    std::vector<Eigen::Vector3f> canonical;
    int weakIndex;
    Fixture(int weak=-1):weakIndex(weak) {
        map.mbMetric=map.mbBackgroundReady=true;
        map.mMetricScale=1;
        for(const auto& xy:std::vector<Eigen::Vector2f>{{-1,-1},{1,-1},{1,1},{-1,1}})
            canonical.push_back(Eigen::Vector3f(.024f*xy.x(),.024f*xy.y(),1.2f));
        for(const auto& p:canonical) for(int axis=0;axis<3;++axis) map.mStaticTags[26].push_back(p[axis]);
        cv::Mat image(480,640,CV_8UC1), distortion=cv::Mat::zeros(4,1,CV_32F);
        cv::RNG rng(98123);rng.fill(image,cv::RNG::UNIFORM,0,256);
        Frame base(image,0,&extractor,&vocabulary,&camera,distortion,0,1);
        base.SetPose(Sophus::SE3f());
        for(int i=0;i<3;++i) {
            Frame frame(base);
            frame.mnId=Frame::nNextId++;
            frame.mTimeStamp=.1*i;
            frame.SetPose(Sophus::SE3f(Eigen::Matrix3f::Identity(),Eigen::Vector3f(-.02f*i,0,0)));
            owned.emplace_back(new KeyFrame(frame,&map,&database));
            map.AddKeyFrame(owned.back().get());
            addTag(i,26,i==weak?.25f:1.f,i==weak?Eigen::Vector3f(.007f,0,0):Eigen::Vector3f::Zero());
        }
        map.mvpKeyFrameOrigins.push_back(owned.front().get());
        map.mnMarkerScaleAnchorKFId=long(owned.front()->mnId);
    }
    void addTag(int index,int id,float weight,const Eigen::Vector3f& staleOffset=Eigen::Vector3f::Zero()) {
        auto* k=owned[index].get();
        k->mbHasTagObservation=k->mbTagObservationActive=true;
        k->mTagObservationConfidence=1;
        for(const auto& p:canonical) {
            const Eigen::Vector3f world=p+(id==26?Eigen::Vector3f::Zero():Eigen::Vector3f(.2f,0,0));
            const Eigen::Vector2f pixel=camera.project(k->GetPose()*world);
            k->mvTagIds.push_back(id);
            k->mvTagWorldPoints.push_back(world+staleOffset);
            k->mvTagImagePoints.emplace_back(pixel.x(),pixel.y());
            k->mvTagPointWeights.push_back(weight);
        }
    }
    MGO::StagedBAInput input() const {
        MGO::StagedBAInput result;
        for(std::size_t i=0;i<owned.size();++i) {
            result.keyframes.push_back(owned[i].get());
            // Keep the weak camera and a strong gauge fixed, with one healthy
            // free camera so g2o has an active variable rather than an empty graph.
            if(int(i)!=(weakIndex==2?1:2)) result.fixedKeyframes.insert(owned[i].get());
        }
        return result;
    }
    MGO::Proposal run(const MGO::StagedBAInput& input,const MGO::Options& options=MGO::Options(),bool fullMap=false) {
        const auto oldMap=map.mStaticTags;
        const auto oldInputCorners=input.tagWorldCorners;
        const auto oldInputPoses=input.keyframePoses;
        std::vector<std::vector<Eigen::Vector3f>> oldCorners;
        std::vector<std::vector<cv::Point2f>> oldPixels;
        std::vector<std::vector<float>> oldWeights;
        std::vector<std::vector<int>> oldIds;
        std::vector<Sophus::SE3f> oldPoses;
        for(const auto& k:owned) {
            oldCorners.push_back(k->mvTagWorldPoints);oldPixels.push_back(k->mvTagImagePoints);
            oldWeights.push_back(k->mvTagPointWeights);oldIds.push_back(k->mvTagIds);
            oldPoses.push_back(k->GetPose());
        }
        auto result=fullMap ? MGO::RefineMetricMap(&map,options) : MGO::RefineAndValidate(input,options);
        require(map.mStaticTags==oldMap,"proposal changed live registered geometry");
        for(std::size_t i=0;i<owned.size();++i) {
            const auto* k=owned[i].get();
            require(sameCorners(k->mvTagWorldPoints,oldCorners[i]),"proposal changed live XYZ");
            require(k->mvTagImagePoints==oldPixels[i] && k->mvTagPointWeights==oldWeights[i] &&
                    k->mvTagIds==oldIds[i],"proposal changed raw pixels/weights/IDs");
            require((owned[i]->GetPose().matrix()-oldPoses[i].matrix()).norm()==0,"proposal changed live camera");
            require(k->mbHasTagObservation && k->mbTagObservationActive,"proposal deactivated live observations");
        }
        require(input.tagWorldCorners.size()==oldInputCorners.size(),"proposal changed input overrides");
        for(const auto& item:oldInputCorners)
            require(sameCorners(input.tagWorldCorners.at(item.first),item.second),"proposal wrote staged input XYZ");
        for(const auto& item:oldInputPoses)
            require((input.keyframePoses.at(item.first).matrix()-item.second.matrix()).norm()==0,"proposal wrote staged input pose");
        return result;
    }
};

static void orderedWeakCases() {
    for(int weak:{0,1,2}) {
        Fixture f(weak);
        auto input=f.input();
        const auto result=f.run(input);
        require(result.accepted,"ordered weak group rejected: "+result.reason);
        require(result.after.tagCorners==12 && result.after.tagRmsPx<1e-3,"raw weak pixels were omitted or changed");
        require(sameCorners(result.staticTags.at(26),f.canonical),"weak keyframe chose the canonical gauge");
        require(sameCorners(result.tagWorldCorners.at(f.owned[weak].get()),f.canonical),"full weak corners not indexed to canonical");
        std::reverse(input.keyframes.begin(),input.keyframes.end());
        const auto reordered=f.run(input);
        require(reordered.accepted && sameCorners(reordered.staticTags.at(26),result.staticTags.at(26)),
                "input/weak-first order changed canonical selection");
        MGO::Options legacy;legacy.canonicalizeWeakMarkerCorners=false;
        const auto stale=f.run(input,legacy);
        require(!stale.accepted && stale.reason=="tag_reprojection_validation_failed",
                "stale 7mm world offset bypassed pixel validation");
    }
}

static void mixedAndUnknownCases() {
    Fixture f(1);
    f.addTag(1,45,1);
    f.addTag(1,88,.25f,Eigen::Vector3f(.02f,0,0));
    auto result=f.run(f.input());
    require(result.accepted,"mixed strong/weak group rejected: "+result.reason);
    require(result.staticTags.count(26) && result.staticTags.count(45) && !result.staticTags.count(88),
            "unknown weak marker created a static landmark");
    require(result.after.tagCorners==16,"unknown weak-only geometry became an absolute factor");
    require(std::find(result.optimizedMarkerIds.begin(),result.optimizedMarkerIds.end(),88)==result.optimizedMarkerIds.end(),
            "weak-only marker created a free pose vertex");
    Fixture onlyWeak;
    for(auto& k:onlyWeak.owned) k->mvTagPointWeights.assign(4,.25f);
    result=onlyWeak.run(onlyWeak.input());
    require(!result.accepted && result.reason=="insufficient_tag_constraints" && result.staticTags.empty(),
            "weak-only input established a landmark/scale from the old map gauge");
}

static void geometryAndResidualRejections() {
    for(const std::string kind:{"shape","size","order","strong","pixels","depth","ids"}) {
        Fixture f(1);
        auto input=f.input();
        auto* weak=f.owned[1].get();
        std::string expected;
        if(kind=="shape") {weak->mvTagWorldPoints[2].x()+=.01f;expected="inconsistent_weak_marker_shape";}
        if(kind=="size") {
            for(auto& p:weak->mvTagWorldPoints) p=f.canonical[0]+1.1f*(p-f.canonical[0]);
            expected="inconsistent_weak_marker_shape";
        }
        if(kind=="order") {std::swap(weak->mvTagWorldPoints[1],weak->mvTagWorldPoints[2]);expected="inconsistent_weak_marker_shape";}
        if(kind=="strong") {
            for(auto& p:f.owned[2]->mvTagWorldPoints) p.x()+=.0062f;
            expected="inconsistent_static_marker_geometry";
        }
        if(kind=="pixels") {weak->mvTagImagePoints[0].x+=40;expected="tag_reprojection_validation_failed";}
        if(kind=="depth") {
            input.keyframePoses[weak]=Sophus::SE3f(Eigen::Matrix3f::Identity(),Eigen::Vector3f(0,0,-2.4f));
            expected="positive_depth_validation_failed";
        }
        if(kind=="ids") {weak->mvTagIds.pop_back();expected="invalid_tag_observation";}
        const auto result=f.run(input);
        require(!result.accepted && result.reason==expected,kind+" rejection changed: "+result.reason);
    }
    Fixture within;
    for(auto& p:within.owned[2]->mvTagWorldPoints) p.x()+=.006f;
    require(within.run(within.input()).accepted,"existing strong geometry tolerance was tightened");
}

static void partialAndStagedGaugeCases() {
    Fixture fullMap(1);
    const auto refined=fullMap.run(fullMap.input(),MGO::Options(),true);
    require(refined.accepted && refined.after.tagRmsPx<1e-3,
            "final metric-map BA failed to use strong canonical corners: "+refined.reason);
    Fixture partial(1);
    auto* weak=partial.owned[1].get();
    weak->mvTagIds.resize(1);weak->mvTagWorldPoints.resize(1);
    weak->mvTagImagePoints.resize(1);weak->mvTagPointWeights.resize(1);
    auto result=partial.run(partial.input());
    require(!result.accepted && result.reason=="tag_reprojection_validation_failed",
            "unknown-index partial corner was snapped across the 6.1mm gate");
    weak->mvTagWorldPoints[0]=partial.canonical[0]+Eigen::Vector3f(.005f,0,0);
    result=partial.run(partial.input());
    require(result.accepted && result.after.tagCorners==9,"existing bounded partial-corner rule changed");

    Fixture staged(0);
    auto input=staged.input();
    const Sophus::SE3f transformed(Sophus::SO3f::exp(Eigen::Vector3f(.1f,-.2f,.15f)),Eigen::Vector3f(.5f,-.3f,.2f));
    for(auto& k:staged.owned) {
        auto corners=k->mvTagWorldPoints;
        for(auto& p:corners) p=transformed*p;
        input.tagWorldCorners[k.get()]=corners;
        input.keyframePoses[k.get()]=k->GetPose()*transformed.inverse();
    }
    result=staged.run(input);
    require(result.accepted,"transformed staged gauge rejected: "+result.reason);
    for(std::size_t i=0;i<4;++i)
        require((result.staticTags.at(26)[i]-transformed*staged.canonical[i]).norm()<1e-6f,
                "live old-map canonical geometry overwrote the staged gauge");
}

int main() {
    try {
        orderedWeakCases();mixedAndUnknownCases();geometryAndResidualRejections();partialAndStagedGaugeCases();
        std::cout << "MARKER_GEOMETRY_SAFETY_OK: strong-first canonical, weak-first ordering, mixed groups, "
                     "6.1mm strong gate, physical shape/order, raw pixels/weights, depth, partial gate, "
                     "staged gauge and input immutability\n";
    } catch(const std::exception& error) {std::cerr << error.what() << '\n'; return 1;}
}
