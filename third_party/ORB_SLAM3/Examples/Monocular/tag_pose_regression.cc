// Numeric regression for fixed-tag/ORB pose fusion; no camera or vocabulary.
#include "Optimizer.h"
#include "Frame.h"
#include "MapPoint.h"
#include "Map.h"
#include "KeyFrame.h"
#include "ORBextractor.h"
#include "CameraModels/Pinhole.h"
#include <iostream>
#include <memory>

using namespace ORB_SLAM3;

struct Result { double position, tagRms; };

int cleanupBeforeSave()
{
    Map map(KeyFrame::nNextId),other(KeyFrame::nNextId);
    Pinhole camera(std::vector<float>{800,800,320,240});
    ORBextractor extractor(250,1.2f,8,20,7);
    cv::Mat image(480,640,CV_8UC1),distortion=cv::Mat::zeros(4,1,CV_32F);
    cv::RNG rng(123); rng.fill(image,cv::RNG::UNIFORM,0,256);
    Frame frame(image,0,&extractor,nullptr,&camera,distortion,0,1);
    frame.SetPose(Sophus::SE3f());
    auto* first=new KeyFrame(frame,&map,nullptr);
    auto* stale=new KeyFrame(frame,&other,nullptr);
    map.AddKeyFrame(first);
    // A removed/merged keyframe may leave observations to be cleaned up.
    // Removing these observations also removes the points from the map set.
    for(int i=0;i<40;++i) {
        auto* point=new MapPoint(Eigen::Vector3f(0,0,1),stale,&map);
        point->AddObservation(stale,i); stale->AddMapPoint(point,i);
        if(i%2) {point->AddObservation(first,i);first->AddMapPoint(point,i);}
        map.AddMapPoint(point);
    }
    std::set<GeometricCamera*> cameras;
    map.PreSave(cameras);
    return map.MapPointsInMap();
}

int localBundle(bool secondTag, float baseline, float weight, double& pointError)
{
    Map map(KeyFrame::nNextId);
    map.mbMetric=true;
    Pinhole camera(std::vector<float>{800,800,320,240});
    ORBextractor extractor(250,1.2f,8,20,7);
    cv::Mat image(480,640,CV_8UC1),distortion=cv::Mat::zeros(4,1,CV_32F);
    cv::RNG rng(123); rng.fill(image,cv::RNG::UNIFORM,0,256);
    Frame base(image,0,&extractor,nullptr,&camera,distortion,0,1);
    base.SetPose(Sophus::SE3f());
    auto* root=new KeyFrame(base,&map,nullptr);
    map.AddKeyFrame(root);
    std::vector<KeyFrame*> keyframes;
    std::vector<Eigen::Vector3f> truth;
    for(int i=0;i<80;++i) truth.emplace_back((i%10-4.5f)*.03f,(i/10-3.5f)*.03f,1.f+.03f*(i%3));
    for(int view=0;view<2;++view) {
        Frame frame(base);
        frame.mTimeStamp=view+1;
        const Sophus::SE3f Tcw(Eigen::Matrix3f::Identity(),Eigen::Vector3f(-view*baseline,0,0));
        frame.SetPose(Tcw);
        for(size_t i=0;i<truth.size();++i) {
            const auto uv=camera.project(Tcw*truth[i]);
            frame.mvKeysUn[i]=cv::KeyPoint(cv::Point2f(uv.x(),uv.y()),1.f);
        }
        auto* keyframe=new KeyFrame(frame,&map,nullptr);
        keyframe->mbTagObservationActive=view==0 || secondTag;
        keyframe->mTagObservationConfidence=1;
        for(auto corner:std::vector<Eigen::Vector3f>{{-.06f,-.06f,.5f},{.06f,-.06f,.5f},
                                                   {.06f,.06f,.5f},{-.06f,.06f,.5f}}) {
            keyframe->mvTagWorldPoints.push_back(corner);
            const auto uv=camera.project(Tcw*corner);
            keyframe->mvTagImagePoints.emplace_back(uv.x(),uv.y());
            keyframe->mvTagPointWeights.push_back(weight);
        }
        map.AddKeyFrame(keyframe);
        keyframes.push_back(keyframe);
    }
    for(size_t i=0;i<truth.size();++i) {
        auto* point=new MapPoint(truth[i]+Eigen::Vector3f(.004f,0,0),keyframes[0],&map);
        for(auto* keyframe:keyframes) {point->AddObservation(keyframe,i);keyframe->AddMapPoint(point,i);}
        map.AddMapPoint(point);
    }
    for(auto* keyframe:keyframes) keyframe->UpdateConnections();
    int fixed=0,optimized=0,points=0,edges=0;
    bool stop=false;
    Optimizer::LocalBundleAdjustment(keyframes[1],&stop,&map,fixed,optimized,points,edges);
    pointError=0;
    for(size_t i=0;i<truth.size();++i) {
        auto* point=keyframes[0]->GetMapPoint(i);
        pointError+=point?(point->GetWorldPos()-truth[i]).norm():1.0;
    }
    pointError/=truth.size();
    return optimized;
}

Result solve(float tagWeight, bool corruptCorner, float bias=.008f)
{
    Pinhole camera(std::vector<float>{800, 800, 320, 240});
    Frame frame;
    frame.mpCamera=&camera;
    frame.mpCamera2=nullptr;
    frame.N=100;
    frame.mvuRight.assign(frame.N,-1);
    frame.mvbOutlier.assign(frame.N,false);
    frame.mvInvLevelSigma2={1.0f};
    frame.SetPose(Sophus::SE3f(Eigen::Matrix3f::Identity(), Eigen::Vector3f(-bias,0,0)));
    std::vector<std::unique_ptr<MapPoint>> points;
    for(int i=0;i<frame.N;++i) {
        const Eigen::Vector3f point((i%10-4.5f)*.06f,(i/10-4.5f)*.04f,1.0f+.05f*(i%3));
        const auto uv=camera.project(point);
        frame.mvKeysUn.emplace_back(cv::Point2f(uv.x(),uv.y()),1.f);
        points.emplace_back(new MapPoint());
        // A slightly distorted background competes with the correct tag.
        points.back()->SetWorldPos(point+Eigen::Vector3f(bias,0,0));
        frame.mvpMapPoints.push_back(points.back().get());
    }
    std::vector<Eigen::Vector3f> world;
    std::vector<cv::Point2f> pixels;
    std::vector<float> weights;
    if(tagWeight>0) {
        for(float center:{-.075f,.075f})
            for(auto corner:std::vector<Eigen::Vector2f>{{-.024f,-.024f},{.024f,-.024f},
                                                        {.024f,.024f},{-.024f,.024f}}) {
                world.emplace_back(center+corner.x(),corner.y(),.5f);
                const auto uv=camera.project(world.back());
                pixels.emplace_back(uv.x(),uv.y());
                weights.push_back(tagWeight);
            }
        if(corruptCorner) pixels[0].x+=40;
    }
    Optimizer::PoseOptimization(&frame,world,pixels,weights);
    double squared=0;
    for(const auto& p:world)
        squared+=(camera.project(frame.GetPose()*p)-camera.project(p)).squaredNorm();
    return {frame.GetCameraCenter().norm(),world.empty()?0:std::sqrt(squared/world.size())};
}

int main()
{
    const auto orb=solve(0,false),strong=solve(1,false),weak=solve(.25f,false),bad=solve(1,true);
    const auto biased=solve(1,false,.03f);
    double corrected,unused;
    const int local=localBundle(true,.04f,1,corrected);
    const int one=localBundle(false,.04f,1,unused);
    const int stationary=localBundle(true,0,1,unused);
    const int weakLocal=localBundle(true,.04f,.25f,unused);
    const int savedPoints=cleanupBeforeSave();
    std::cout << "{\"orb_position_m\":" << orb.position
              << ",\"strong_position_m\":" << strong.position
              << ",\"strong_tag_rms_px\":" << strong.tagRms
              << ",\"weak_tag_rms_px\":" << weak.tagRms
              << ",\"corrupt_tag_rms_px\":" << bad.tagRms
              << ",\"biased_tag_rms_px\":" << biased.tagRms
              << ",\"tag_anchored_lba_keyframes\":" << local
              << ",\"single_tag_lba_keyframes\":" << one
              << ",\"zero_baseline_lba_keyframes\":" << stationary
              << ",\"weak_tag_lba_keyframes\":" << weakLocal
              << ",\"lba_point_error_m\":" << corrected
              << ",\"stale_points_after_save\":" << savedPoints << "}" << std::endl;
    if(!std::isfinite(strong.tagRms) || strong.tagRms>=weak.tagRms ||
       strong.position>=orb.position || bad.tagRms>5.0 ||
       local!=2 || one!=0 || stationary!=0 || weakLocal!=0 || corrected>=.0002 || savedPoints!=0)
        return 1;
    return 0;
}
