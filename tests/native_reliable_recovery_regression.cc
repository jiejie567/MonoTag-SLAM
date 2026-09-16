#include "Tracking.h"
#include "System.h"
#include "Settings.h"
#include "Atlas.h"
#include "Map.h"
#include "MapPoint.h"
#include "KeyFrameDatabase.h"
#include "LocalMapping.h"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <limits>
#include <set>
#include <stdexcept>
#include <tuple>

using namespace ORB_SLAM3;

static void require(bool condition, const char* message) {
    if(!condition) throw std::runtime_error(message);
}

static auto scratchState(const MapPoint& point) {
    return std::make_tuple(point.mTrackProjX,point.mTrackProjY,point.mTrackDepth,
        point.mTrackDepthR,point.mTrackProjXR,point.mTrackProjYR,
        point.mbTrackInView,point.mbTrackInViewR,point.mnTrackScaleLevel,
        point.mnTrackScaleLevelR,point.mTrackViewCos,point.mTrackViewCosR,
        point.mnTrackReferenceForFrame,point.mnLastFrameSeen);
}

// This harness calls the real native cache/recovery methods, without starting
// mapping workers or processing a SLAM sequence. Synthetic correspondences test
// cache policy and stale-object rejection, not geometric recovery quality.
class ProbeTracker : public Tracking {
public:
    using Tracking::Tracking;
    using Tracking::ClearReliableFlowFrame;
    using Tracking::UpdateReliableFlowFrame;
    using Tracking::TryTemporalFlowRecovery;
    using Tracking::TrackWithTemporalFlow;
    using Tracking::TrackLocalMap;
    using Tracking::mReliableFlowFrame;
    using Tracking::mReliableFlowImage;
    using Tracking::mpReliableFlowMap;
    using Tracking::mvReliableFlowPointIds;
    using Tracking::mnMatchesInliers;
    using Tracking::mbFlowRecoveryEnabled;
    using Tracking::mbReliableFrameRecoveryEnabled;
    using Tracking::mbReliableFrameCacheEnabled;
    using Tracking::mTemporalPreviousImage;
    using Tracking::mTemporalPreviousTime;
    using Tracking::mpReferenceKF;
    using Tracking::mvpLocalMapPoints;
    using Tracking::mvpLocalKeyFrames;
    using Tracking::mbVelocity;
    using Tracking::mVelocity;

    Frame syntheticFrame(const cv::Mat& image, double timestamp, unsigned long id) {
        Frame frame(image,timestamp,mpORBextractorLeft,mpORBVocabulary,
                    mpCamera,mDistCoef,mbf,mThDepth);
        frame.mnId=id;
        frame.SetPose(Sophus::SE3f());
        return frame;
    }
};

int main(int argc, char** argv) {
    try {
        require(argc==2,"pass a camera settings file");
        setenv("ORB_SLAM3_FLOW_RECOVERY","1",1);
        setenv("ORB_SLAM3_RELIABLE_FRAME_RECOVERY","1",1);
        setenv("ORB_SLAM3_TEMPORAL_FLOW","0",1);
        ORBVocabulary vocabulary;
        Atlas atlas(0);
        KeyFrameDatabase database(vocabulary);
        Settings settings(argv[1],System::MONOCULAR);
        ProbeTracker tracker(nullptr,&vocabulary,nullptr,nullptr,&atlas,&database,
                             argv[1],System::MONOCULAR,&settings);
        Map* map=atlas.GetCurrentMap();
        cv::Mat image(480,640,CV_8UC1);
        cv::RNG random(72615);
        random.fill(image,cv::RNG::UNIFORM,0,256);
        Frame healthy=tracker.syntheticFrame(image,100.,1000);
        require(healthy.N>=100,"synthetic ORB extraction insufficient");
        // Exercise the actual local-map pose solver, not a duplicate policy
        // function. Known scale cannot rescue insufficient visual support.
        for(bool metric : {false,true}) {
            map->mbMetric=metric;
            for(int count : {14,15,28,29,30,40}) {
                tracker.mCurrentFrame=healthy;
                tracker.mState=Tracking::OK;
                std::vector<std::unique_ptr<MapPoint>> support;
                for(int i=0;i<count;++i) {
                    std::unique_ptr<MapPoint> point(new MapPoint());
                    point->nObs=2;
                    point->UpdateMap(map);
                    const auto uv=healthy.mvKeysUn[i].pt;
                    point->SetWorldPos(Eigen::Vector3f(
                        (uv.x-healthy.cx)*healthy.invfx*3.f,
                        (uv.y-healthy.cy)*healthy.invfy*3.f,3.f));
                    tracker.mCurrentFrame.mvpMapPoints[i]=point.get();
                    support.push_back(std::move(point));
                }
                const bool accepted=tracker.TrackLocalMap(true);
                require(tracker.mnMatchesInliers==count,"exact visual correspondences rejected");
                require(accepted==(count>=30),"metric flag weakened pure-visual support floor");
            }
        }
        tracker.mCurrentFrame=healthy;
        map->mbMetric=false;
        std::cout<<"VISUAL_SUPPORT_FLOOR_OK: metric/unscaled 14,15,28,29 rejected; 30,40 accepted"<<std::endl;
        std::vector<std::unique_ptr<MapPoint>> owned;
        for(int index=0;index<60;++index) {
            std::unique_ptr<MapPoint> point(new MapPoint());
            point->mnId=100000+index;
            point->nObs=2;
            point->UpdateMap(map);
            point->SetWorldPos(Eigen::Vector3f(float(index%8)*.1f,float(index/8)*.1f,3.f));
            map->AddMapPoint(point.get());
            healthy.mvpMapPoints[index]=point.get();
            // Spread the cache-admission test over the complete 8x6 grid.
            healthy.mvKeys[index].pt=cv::Point2f(40.f+80.f*(index%8),40.f+80.f*((index/8)%6));
            healthy.mvbOutlier[index]=false;
            owned.push_back(std::move(point));
        }
        tracker.mCurrentFrame=healthy;
        tracker.mImGray=image;
        tracker.mState=Tracking::OK;
        tracker.mnMatchesInliers=59;
        tracker.UpdateReliableFlowFrame();
        require(!tracker.mpReliableFlowMap,"59 inliers were admitted");
        tracker.mnMatchesInliers=60;
        tracker.UpdateReliableFlowFrame();
        require(tracker.mpReliableFlowMap==map,"60 good inliers were not admitted");
        require(tracker.mReliableFlowFrame.mnId==1000,"cache source frame wrong");
        require(std::all_of(tracker.mReliableFlowFrame.mvpMapPoints.begin(),
                           tracker.mReliableFlowFrame.mvpMapPoints.end(),
                           [](MapPoint* point){return point==nullptr;}),
                "cache owns stale raw map-point pointers");
        require(tracker.mReliableFlowImage.data!=image.data,"cache image aliases mutable input");

        tracker.mCurrentFrame.mnId=1001;
        tracker.mCurrentFrame.mTimeStamp=100.07;
        tracker.mnMatchesInliers=15;
        tracker.UpdateReliableFlowFrame();
        require(tracker.mReliableFlowFrame.mnId==1000,"weak frame replaced reliable source");
        tracker.mState=Tracking::RECENTLY_LOST;
        tracker.mnMatchesInliers=60;
        tracker.UpdateReliableFlowFrame();
        require(tracker.mReliableFlowFrame.mnId==1000,"lost frame replaced reliable source");

        tracker.ClearReliableFlowFrame();
        require(!tracker.mpReliableFlowMap && tracker.mReliableFlowImage.empty() &&
                tracker.mvReliableFlowPointIds.empty(),"clear left cached resources");
        tracker.mState=Tracking::OK;
        tracker.mCurrentFrame=healthy;
        tracker.mbFlowRecoveryEnabled=false;
        tracker.UpdateReliableFlowFrame();
        require(!tracker.mpReliableFlowMap,"FLOW_RECOVERY=0 admitted a cache");
        tracker.mbFlowRecoveryEnabled=true;
        tracker.mbReliableFrameRecoveryEnabled=false;
        tracker.UpdateReliableFlowFrame();
        require(!tracker.mpReliableFlowMap,"RELIABLE_FRAME_RECOVERY=0 admitted a cache");
        tracker.mbReliableFrameRecoveryEnabled=true;
        tracker.mCurrentFrame.mTimeStamp=std::numeric_limits<double>::quiet_NaN();
        tracker.UpdateReliableFlowFrame();
        require(!tracker.mpReliableFlowMap,"non-finite timestamp admitted a cache");
        tracker.mCurrentFrame=healthy;
        for(auto& point:owned) point->nObs=1;
        tracker.UpdateReliableFlowFrame();
        require(!tracker.mpReliableFlowMap,"single-observation points admitted");
        for(auto& point:owned) point->nObs=2;
        std::fill(tracker.mCurrentFrame.mvpMapPoints.begin(),
                  tracker.mCurrentFrame.mvpMapPoints.end(),owned.front().get());
        tracker.UpdateReliableFlowFrame();
        require(!tracker.mpReliableFlowMap,"duplicate point identities admitted");
        tracker.mCurrentFrame=healthy;
        for(auto& key:tracker.mCurrentFrame.mvKeys) key.pt=cv::Point2f(40,40);
        tracker.UpdateReliableFlowFrame();
        require(!tracker.mpReliableFlowMap,"one-cell correspondence set admitted");
        tracker.mCurrentFrame=healthy;
        tracker.UpdateReliableFlowFrame();
        require(tracker.mpReliableFlowMap==map,"cache failed to re-admit healthy source");

        // Erase AND free all cached points. The subsequent attempt must resolve
        // IDs from current map membership, not dereference cached addresses.
        for(auto& point:owned) map->EraseMapPoint(point.get());
        owned.clear();
        Frame weak=tracker.syntheticFrame(image,100.07,1001);
        Frame current=tracker.syntheticFrame(image,100.14,1002);
        tracker.mLastFrame=weak;
        tracker.mCurrentFrame=current;
        tracker.mTemporalPreviousImage=image.clone();
        tracker.mTemporalPreviousTime=weak.mTimeStamp;
        tracker.mState=Tracking::RECENTLY_LOST;
        tracker.mnMatchesInliers=7;
        tracker.mbVelocity=true;
        tracker.mVelocity=Sophus::SE3f();
        tracker.mpReferenceKF=nullptr;
        const auto previousId=tracker.mLastFrame.mnId;
        const auto beforePose=tracker.mCurrentFrame.GetPose().matrix();
        const auto beforeHistory=tracker.mlRelativeFramePoses.size();
        require(!tracker.TryTemporalFlowRecovery(7),"deleted-point cache falsely recovered");
        require(tracker.mnMatchesInliers==7 && tracker.mCurrentFrame.mnId==1002 &&
                tracker.mCurrentFrame.GetPose().matrix().isApprox(beforePose),
                "failed attempt changed current-frame estimate");
        require(tracker.mLastFrame.mnId==previousId && tracker.mbVelocity &&
                tracker.mlRelativeFramePoses.size()==beforeHistory && !tracker.mpReferenceKF &&
                tracker.mvpLocalMapPoints.empty() && tracker.mvpLocalKeyFrames.empty(),
                "failed attempt polluted tracking history/reference state");
        tracker.mCurrentFrame.mTimeStamp=100.30;
        require(!tracker.TryTemporalFlowRecovery(7),"expired cache falsely recovered");
        atlas.CreateNewMap();
        tracker.mCurrentFrame.mTimeStamp=100.14;
        require(!tracker.TryTemporalFlowRecovery(7),"cross-map cache falsely recovered");
        tracker.ClearReliableFlowFrame();

        // Now exercise real LK, descriptor association, PnP and local-map pose
        // optimization using identical images and non-coplanar synthetic 3-D.
        map=atlas.GetCurrentMap();
        Frame measured=tracker.syntheticFrame(image,200.,2000);
        const float fx=measured.mK.at<float>(0,0),fy=measured.mK.at<float>(1,1);
        const float cx=measured.mK.at<float>(0,2),cy=measured.mK.at<float>(1,2);
        for(int index=0;index<measured.N;++index) {
            std::unique_ptr<MapPoint> point(new MapPoint());
            point->mnId=200000+index;
            point->nObs=2;
            point->UpdateMap(map);
            const float depth=2.f+.13f*(index%11);
            const auto pixel=measured.mvKeysUn[index].pt;
            point->SetWorldPos(Eigen::Vector3f((pixel.x-cx)*depth/fx,
                                             (pixel.y-cy)*depth/fy,depth));
            point->mTrackProjX=17.f;
            point->mTrackProjY=18.f;
            point->mTrackDepth=19.f;
            point->mTrackDepthR=20.f;
            point->mTrackProjXR=21.f;
            point->mTrackProjYR=22.f;
            point->mbTrackInView=false;
            point->mbTrackInViewR=false;
            point->mnTrackScaleLevel=0;
            point->mnTrackScaleLevelR=0;
            point->mTrackViewCos=.2f;
            point->mTrackViewCosR=.3f;
            point->mnTrackReferenceForFrame=122;
            point->mnLastFrameSeen=123;
            map->AddMapPoint(point.get());
            measured.mvpMapPoints[index]=point.get();
            measured.mvbOutlier[index]=false;
            owned.push_back(std::move(point));
        }
        tracker.mCurrentFrame=measured;
        tracker.mState=Tracking::OK;
        tracker.mnMatchesInliers=measured.N;
        tracker.UpdateReliableFlowFrame();
        require(tracker.mReliableFlowFrame.mnId==2000,"geometric source not cached");
        weak=measured;
        weak.mnId=2001; weak.mTimeStamp=200.07;
        std::fill(weak.mvpMapPoints.begin(),weak.mvpMapPoints.end(),nullptr);
        current=weak;
        current.mnId=2002; current.mTimeStamp=200.14;
        tracker.mLastFrame=weak;
        tracker.mCurrentFrame=current;
        tracker.mTemporalPreviousTime=weak.mTimeStamp;
        tracker.mState=Tracking::RECENTLY_LOST;
        tracker.mnMatchesInliers=7;
        LocalMapping mapper(nullptr,&atlas,1.f,false);
        tracker.SetLocalMapper(&mapper);
        mapper.mnMatchesInliers=11;
        const auto beforeScratch=scratchState(*owned.front());
        require(tracker.TrackWithTemporalFlow(true,&measured,&image),
                "synthetic geometric flow positive control failed");
        tracker.mCurrentFrame=current;
        tracker.mCurrentFrame.mTimeStamp=std::numeric_limits<double>::quiet_NaN();
        require(!tracker.TryTemporalFlowRecovery(7),"non-finite recovery age accepted");
        tracker.mCurrentFrame=current;
        const auto oldGraph=map->mnMarkerGraphSequence;
        ++map->mnMarkerGraphSequence;
        require(!tracker.TryTemporalFlowRecovery(7),"marker-gauge change retained cache");
        map->mnMarkerGraphSequence=oldGraph;
        const bool oldMetric=map->mbMetric;
        map->mbMetric=!oldMetric;
        require(!tracker.TryTemporalFlowRecovery(7),"metric-unit transition retained cache");
        map->mbMetric=oldMetric;
        const float oldScale=map->mMetricScale;
        map->mMetricScale=oldScale+1.f;
        require(!tracker.TryTemporalFlowRecovery(7),"scale transition retained cache");
        map->mMetricScale=oldScale;
        // An ordinary map revision must NOT discard otherwise valid evidence.
        MapPoint extra;
        extra.mnId=999999;
        extra.UpdateMap(map);
        map->AddMapPoint(&extra);
        map->EraseMapPoint(&extra);
        tracker.mImGray=cv::Mat(480,640,CV_16UC1,cv::Scalar(0));
        require(!tracker.TryTemporalFlowRecovery(7),
                "invalid-image OpenCV trial did not become a rejected candidate");
        require(tracker.mnMatchesInliers==7 && mapper.mnMatchesInliers==11 &&
                tracker.mCurrentFrame.mnId==2002 && tracker.mLastFrame.mnId==2001,
                "OpenCV exception trial did not restore frame/counter state");
        tracker.mImGray=image;
        require(!tracker.TryTemporalFlowRecovery(1000000),
                "improvement gate failed to reject geometric candidate");
        require(tracker.mnMatchesInliers==7 && mapper.mnMatchesInliers==11,
                "rejected geometric candidate changed inlier counters");
        for(const auto& point:owned) {
            require(point->GetFound()==1 && point->GetFoundRatio()==1.f,
                    "rejected geometric candidate changed point statistics");
            require(scratchState(*point)==beforeScratch,
                    "rejected geometric candidate changed projection scratch");
        }
        require(tracker.TryTemporalFlowRecovery(7),
                "valid measured reliable-source recovery was rejected");
        require(tracker.mnMatchesInliers>=30 && mapper.mnMatchesInliers==tracker.mnMatchesInliers,
                "accepted candidate counters did not commit");
        for(const auto& point:owned) {
            require(scratchState(*point)==beforeScratch,
                    "accepted geometric candidate changed restored scratch");
            require(point->GetFoundRatio()<=1.f,
                    "accepted recovery increments Found without matching Visible");
        }

        // Controlled older-source A/B, with real feature extraction in a fresh
        // moved image. A fronto-parallel plane at Z=3 m translated by (+12,+6)
        // pixels is exactly the image of Tcw.translation=(.072,.036,0) m with
        // this pinhole camera. The prediction is identity, so a frozen pose
        // cannot pass the known-motion checks below.
        tracker.ClearReliableFlowFrame();
        for(auto& point:owned) map->EraseMapPoint(point.get());
        owned.clear();
        const double sourceTime=300.;
        const float planeDepth=3.f,shiftX=12.f,shiftY=6.f;
        cv::Mat planeImage;
        cv::GaussianBlur(image,planeImage,cv::Size(5,5),1.0);
        Frame source=tracker.syntheticFrame(planeImage,sourceTime,3000);
        for(int index=0;index<source.N;++index) {
            std::unique_ptr<MapPoint> point(new MapPoint());
            point->mnId=300000+index;
            point->nObs=2;
            point->UpdateMap(map);
            const auto pixel=source.mvKeysUn[index].pt;
            point->SetWorldPos(Eigen::Vector3f((pixel.x-cx)*planeDepth/fx,
                                             (pixel.y-cy)*planeDepth/fy,planeDepth));
            map->AddMapPoint(point.get());
            source.mvpMapPoints[index]=point.get();
            source.mvbOutlier[index]=false;
            owned.push_back(std::move(point));
        }
        tracker.mCurrentFrame=source;
        tracker.mImGray=planeImage;
        tracker.mState=Tracking::OK;
        tracker.mnMatchesInliers=source.N;
        tracker.UpdateReliableFlowFrame();
        require(tracker.mReliableFlowFrame.mnId==3000,"controlled reliable source not cached");
        cv::Mat movedImage;
        const cv::Mat affine=(cv::Mat_<double>(2,3)<<1,0,shiftX,0,1,shiftY);
        cv::warpAffine(planeImage,movedImage,affine,planeImage.size(),cv::INTER_NEAREST,
                       cv::BORDER_CONSTANT,cv::Scalar(0));
        const Frame moved=tracker.syntheticFrame(movedImage,sourceTime+.14,3002);
        const Eigen::Vector3f expectedTranslation(shiftX*planeDepth/fx,
                                                shiftY*planeDepth/fy,0.f);
        for(const int weakSeeds:{0,15}) {
            weak=source;
            weak.mnId=3001; weak.mTimeStamp=sourceTime+.07;
            for(int index=weakSeeds;index<weak.N;++index) weak.mvpMapPoints[index]=nullptr;
            tracker.mCurrentFrame=weak;
            tracker.mImGray=planeImage;
            tracker.mState=Tracking::OK;
            tracker.mnMatchesInliers=weakSeeds;
            tracker.UpdateReliableFlowFrame();
            require(tracker.mReliableFlowFrame.mnId==3000,
                    "controlled weak predecessor overwrote reliable source");
            tracker.mLastFrame=weak;
            tracker.mCurrentFrame=moved;
            tracker.mImGray=movedImage;
            tracker.mTemporalPreviousImage=planeImage.clone();
            tracker.mTemporalPreviousTime=weak.mTimeStamp;
            tracker.mState=Tracking::RECENTLY_LOST;
            tracker.mnMatchesInliers=0;
            tracker.mbReliableFrameCacheEnabled=false;
            require(!tracker.TryTemporalFlowRecovery(0),
                    "controlled cache-off trial unexpectedly recovered");
            require(tracker.mnMatchesInliers==0 &&
                    tracker.mCurrentFrame.GetPose().translation().norm()<1e-7f,
                    "cache-off failure published a moved or frozen fallback");
            tracker.mbReliableFrameCacheEnabled=true;
            require(tracker.TryTemporalFlowRecovery(0),
                    "controlled cache-on older-source trial failed");
            require(tracker.mnMatchesInliers>=30 && tracker.mLastFrame.mnId==3001 &&
                    tracker.mReliableFlowFrame.mnId==3000,
                    "controlled recovery did not use the older reliable source");
            std::set<int> coverage;
            for(int index=0;index<tracker.mCurrentFrame.N;++index)
                if(tracker.mCurrentFrame.mvpMapPoints[index] && !tracker.mCurrentFrame.mvbOutlier[index]) {
                    const auto pixel=tracker.mCurrentFrame.mvKeys[index].pt;
                    coverage.insert(int(pixel.y*6/image.rows)*8+int(pixel.x*8/image.cols));
                }
            const auto recovered=tracker.mCurrentFrame.GetPose();
            const float translationError=(recovered.translation()-expectedTranslation).norm();
            const float rotationDegrees=recovered.so3().log().norm()*57.2957795f;
            require(coverage.size()>=16,"controlled recovery lacks broad image coverage");
            require(recovered.translation().norm()>.03f && translationError<.02f &&
                    rotationDegrees<1.f,"controlled recovery froze or missed known camera motion");
            std::cout<<"CONTROLLED_CACHE_AB weak_seeds="<<weakSeeds
                     <<" cache_off=0 cache_on=1 source_frame=3000 predecessor_frame=3001"
                     <<" current_frame=3002 age_s=0.14 inliers="<<tracker.mnMatchesInliers
                     <<" cells="<<coverage.size()<<" translation_error_m="<<translationError
                     <<" rotation_error_deg="<<rotationDegrees
                     <<" recovered_t="<<recovered.translation().transpose()<<std::endl;
            tracker.mCurrentFrame=moved;
            tracker.mCurrentFrame.mTimeStamp=sourceTime+.251;
            tracker.mnMatchesInliers=0;
            require(!tracker.TryTemporalFlowRecovery(0),
                    "controlled expired older-source trial recovered");
        }
        std::cout<<"RELIABLE_RECOVERY_SAFETY_OK: thresholds, reliability, IDs, image ownership, both flags, observations, uniqueness, coverage, freed points, rollback, finite time, expiry, map/gauge/scale changes, ordinary revision, geometric positive/rejection, point statistics, controlled moved-image cache A/B with 0/15 weak seeds"<<std::endl;
        return 0;
    } catch(const std::exception& error) {
        std::cerr<<"RELIABLE_RECOVERY_SAFETY_FAILED: "<<error.what()<<std::endl;
        return 1;
    }
}
