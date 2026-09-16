#include "GeometricDynamicMask.h"
#include "ORBextractor.h"
#include <iostream>

struct Score {int masked=0, dynamicHits=0, staticHits=0, eligible=0, afterReset=0;};

Score movingPatch(bool independentMotion,bool badPose=false,int reset=0) {
    cv::Mat K=(cv::Mat_<double>(3,3)<<500,0,320,0,500,240,0,0,1),D=cv::Mat::zeros(5,1,CV_64F);
    GeometricDynamicMask gate(K,D);
    cv::Mat texture(480,640,CV_8UC1);cv::RNG rng(42);rng.fill(texture,cv::RNG::UNIFORM,0,256);
    const cv::Rect patch(180,160,160,160);
    std::vector<cv::Point2f> base;
    std::vector<bool> moving;
    for(int y=40;y<440;y+=30) for(int x=40;x<600;x+=30) {
        const bool inside=patch.contains(cv::Point(x,y));
        // Keep test landmarks safely away from an occluding patch boundary.
        if(inside && !(x>200 && x<300 && y>180 && y<280)) continue;
        if(!inside && x>140 && x<440 && y>120 && y<360) continue;
        base.emplace_back(x,y);moving.push_back(inside);
    }
    Score score;
    for(int f=0;f<15;++f) {
        const int camera=f, object=independentMotion?f*4:0;
        cv::Mat image;
        cv::Mat warp=(cv::Mat_<double>(2,3)<<1,0,camera,0,1,0);
        cv::warpAffine(texture,image,warp,texture.size(),cv::INTER_LINEAR,cv::BORDER_REFLECT);
        texture(patch).copyTo(image(cv::Rect(patch.x+camera+object,patch.y,patch.width,patch.height)));
        auto mask=gate.prepare(image,cv::Mat(),f/60.+(reset==3 && f>=9?1.:0.));
        if(f==10) score.afterReset=gate.masked;
        std::vector<GeometricDynamicMask::Seed> seeds;
        for(size_t i=0;i<base.size();++i) {
            cv::Point2f xy=base[i]+cv::Point2f(camera+(moving[i]?object:0),0);
            seeds.push_back({i,xy,cv::Point3f((base[i].x-320)/500.f,(base[i].y-240)/500.f,1)});
            if(f>=6) {
                if(moving[i]) ++score.eligible;
                if(mask.at<unsigned char>(cvRound(xy.y),cvRound(xy.x))==0) {
                    if(moving[i]) ++score.dynamicHits;else ++score.staticHits;
                }
            }
        }
        score.masked+=gate.masked;
        cv::Mat rvec=cv::Mat::zeros(3,1,CV_64F),tvec=(cv::Mat_<double>(3,1)<<(camera+(badPose?10:0))/500.,0,0);
        gate.observe(rvec,tvec,seeds,!(reset==1 && f==9),reset==2 && f>=9?1:0,0);
    }
    return score;
}

int main() {
    auto moving=movingPatch(true), stationary=movingPatch(false), bad=movingPatch(false,true);
    auto lost=movingPatch(true,false,1), map=movingPatch(true,false,2), gap=movingPatch(true,false,3);
    cv::Mat image(480,640,CV_8UC1);cv::RNG rng(123);rng.fill(image,cv::RNG::UNIFORM,0,256);
    cv::Mat allowed(image.size(),CV_8UC1,cv::Scalar(255));allowed.colRange(0,320).setTo(0);
    ORB_SLAM3::ORBextractor extractor(2000,1.2f,8,20,7);
    extractor.SetAllowedMask(allowed);
    std::vector<cv::KeyPoint> keys;cv::Mat descriptors;std::vector<int> overlap{0,0};
    extractor(image,cv::Mat(),keys,descriptors,overlap);
    int inside=0;for(auto& k:keys) if(allowed.at<unsigned char>(cvRound(k.pt.y),cvRound(k.pt.x))==0) ++inside;
    const int count=keys.size();
    allowed.setTo(0);extractor.SetAllowedMask(allowed);extractor(image,cv::Mat(),keys,descriptors,overlap);
    std::cout<<"{\"moving_masked\":"<<moving.masked<<",\"dynamic_hits\":"<<moving.dynamicHits
        <<",\"dynamic_eligible\":"<<moving.eligible<<",\"static_false_hits\":"<<moving.staticHits
        <<",\"camera_only_masked\":"<<stationary.masked<<",\"bad_pose_masked\":"<<bad.masked
        <<",\"lost_next_masked\":"<<lost.afterReset<<",\"map_next_masked\":"<<map.afterReset
        <<",\"gap_next_masked\":"<<gap.afterReset
        <<",\"allowed_feature_count\":"<<count<<",\"excluded_features\":"<<inside
        <<",\"fully_masked_features\":"<<keys.size()<<"}"<<std::endl;
}
