#pragma once
// Optional classical motion-consistency gate. No new camera-motion backend:
// only actual native map landmarks and measured native camera poses are used.
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <algorithm>
#include <cmath>
#include <map>
#include <vector>

class GeometricDynamicMask {
public:
    struct Seed { unsigned long id; cv::Point2f pixel; cv::Point3f world; };
    struct Track { Seed seed; int streak=0; int missing=0; };
    cv::Mat K,D;
    int tested=0, confirmed=0, masked=0;
    int probeFrames=0, reusedFrames=0;
    std::vector<cv::Point2f> maskedPixels;
    const char* reason="waiting";

    GeometricDynamicMask(const cv::Mat& camera, const cv::Mat& distortion):K(camera),D(distortion) {}

    cv::Mat prepare(const cv::Mat& image,const cv::Mat& allowed,double timestamp) {
        cv::Mat gray;
        if(image.channels()==3) cv::cvtColor(image,gray,cv::COLOR_BGR2GRAY);
        else gray=image;
        cv::resize(gray,gray,cv::Size(),.5,.5,cv::INTER_AREA);
        if(timestamp-lastTime>.1 || timestamp<=lastTime) tracks.clear();
        lastTime=timestamp;
        masked=0;
        maskedPixels.clear();
        // LK coordinates are defined in `previous`. Advancing `previous`
        // without advancing every track silently changes that coordinate
        // system and can create a one-frame false camera correction. Keep the
        // inexpensive half-resolution flow frame-synchronous.
        flowAdvanced=!previous.empty() && !tracks.empty();
        if(flowAdvanced) {
            ++probeFrames;
            std::vector<cv::Point2f> before,after,back;
            std::vector<unsigned char> status,backStatus;
            std::vector<float> error;
            for(auto& t:tracks) before.push_back(t.seed.pixel*.5f);
            cv::calcOpticalFlowPyrLK(previous,gray,before,after,status,error,cv::Size(21,21),3);
            cv::calcOpticalFlowPyrLK(gray,previous,after,back,backStatus,error,cv::Size(21,21),3);
            std::vector<Track> kept;
            for(size_t i=0;i<tracks.size();++i) {
                auto t=tracks[i]; const auto p=after[i]*2.f;
                if(!status[i] || !backStatus[i] || cv::norm(back[i]-before[i])>.75 ||
                   !std::isfinite(p.x) || !std::isfinite(p.y) || p.x<0 || p.y<0 ||
                   p.x>=image.cols || p.y>=image.rows || ++t.missing>12) continue;
                t.seed.pixel=p;kept.push_back(t);
            }
            tracks.swap(kept);
        }
        previous=gray;
        cv::Mat result=allowed.empty()?cv::Mat(image.size(),CV_8UC1,cv::Scalar(255)):allowed.clone();
        // This gate is deliberately conservative.  A large fraction of
        // apparently inconsistent map points usually means that the pose or
        // map revision is still settling, not that most of the scene moved.
        // Masking them all starves the next ORB frame and can turn a transient
        // geometric disagreement into a tracking loss.  Only apply a mask
        // when a small, already-confirmed subset is coherent; otherwise leave
        // the extractor unconstrained for this frame.
        size_t eligible=0;
        for(const auto& t:tracks) if(t.streak>=3) ++eligible;
        const size_t maxEligible=std::max<size_t>(8,tracks.size()/5);
        const bool stable=(reason==std::string("checked") && confirmed>=3 &&
                           eligible>0 && eligible<=maxEligible);
        if(eligible>0 && !stable) reason=eligible>maxEligible?
            "too_many_dynamic_candidates":"waiting_for_confirmed_geometry";
        if(stable) for(const auto& t:tracks) if(t.streak>=3) {
            cv::circle(result,t.seed.pixel,16,cv::Scalar(0),-1);++masked;maskedPixels.push_back(t.seed.pixel);
        }
        return result;
    }

    // Separate numeric gate so loss/reset/false-positive behavior is testable.
    std::vector<bool> classify(const std::vector<cv::Point2f>& observed,
                               const std::vector<cv::Point2f>& predicted,
                               bool reliable) {
        tested=observed.size();confirmed=0;
        std::vector<bool> coherent(tested,false);
        if(!reliable || tested<40) {reason="no_reliable_geometry";return coherent;}
        std::vector<double> errors, deviations;
        std::vector<cv::Point2f> residuals;
        for(int i=0;i<tested;++i) {
            residuals.push_back(observed[i]-predicted[i]);
            const double error=cv::norm(residuals.back());
            if(!std::isfinite(error)) {reason="invalid_projection";return coherent;}
            errors.push_back(error);
        }
        const double center=median(errors);
        for(double e:errors) deviations.push_back(std::abs(e-center));
        const double threshold=std::max(4.,center+4.*1.4826*median(deviations));
        std::vector<int> bad;
        for(int i=0;i<tested;++i) if(errors[i]>threshold) bad.push_back(i);
        if(center>2.5 || bad.size()>.35*tested) {reason="global_disagreement";return coherent;}
        reason="checked";
        for(int i:bad) {
            int neighbors=0;
            for(int j:bad) if(cv::norm(observed[i]-observed[j])<80. &&
                residuals[i].dot(residuals[j])>.5*errors[i]*errors[j]) ++neighbors;
            coherent[i]=neighbors>=3;
        }
        return coherent;
    }

    void observe(const cv::Mat& rvec,const cv::Mat& tvec,const std::vector<Seed>& seeds,
                 bool reliable,long mapId,int bigChange) {
        if(mapId!=scope || bigChange!=revision) {
            tracks.clear();scope=mapId;revision=bigChange;
        }
        tested=confirmed=0;
        if(!reliable || seeds.size()<40) {
            tracks.clear();reason="no_reliable_geometry";return;
        }
        if(!tracks.empty() && flowAdvanced) {
            std::vector<cv::Point3f> world;
            std::vector<cv::Point2f> observed,predicted;
            cv::Mat rotation;cv::Rodrigues(rvec,rotation);
            std::vector<Track> front;
            for(const auto& t:tracks) {
                const auto& p=t.seed.world;
                double z=rotation.at<double>(2,0)*p.x+rotation.at<double>(2,1)*p.y+
                         rotation.at<double>(2,2)*p.z+tvec.at<double>(2);
                if(z>.01) front.push_back(t);
            }
            tracks.swap(front);
            for(const auto& t:tracks) {world.push_back(t.seed.world);observed.push_back(t.seed.pixel);}
            if(!world.empty()) cv::projectPoints(world,rvec,tvec,K,D,predicted);
            auto candidates=classify(observed,predicted,true);
            for(size_t i=0;i<tracks.size();++i) {
                tracks[i].streak=candidates[i]?tracks[i].streak+1:0;
                if(tracks[i].streak>=3) ++confirmed;
            }
        }
        std::map<unsigned long,size_t> existing;
        for(size_t i=0;i<tracks.size();++i) existing[tracks[i].seed.id]=i;
        for(const auto& s:seeds) {
            auto found=existing.find(s.id);
            if(found!=existing.end()) {
                auto& t=tracks[found->second];
                if(cv::norm(t.seed.pixel-s.pixel)>2.) t.streak=0;
                // Do not let a moved object's re-estimated depth erase the
                // independent-motion evidence against its old static position.
                if(!t.streak) t.seed.world=s.world;
                t.seed.pixel=s.pixel;t.missing=0;
            } else if(tracks.size()<400) {
                existing[s.id]=tracks.size();Track t;t.seed=s;tracks.push_back(t);
            }
        }
    }

private:
    std::vector<Track> tracks;
    cv::Mat previous;
    double lastTime=-1.;long scope=-1;int revision=-1;
    bool flowAdvanced=false;
    static double median(std::vector<double> values) {
        if(values.empty()) return 0.;
        const size_t middle=values.size()/2;
        std::nth_element(values.begin(),values.begin()+middle,values.end());
        double result=values[middle];
        if(values.size()%2==0) result=(result+*std::max_element(values.begin(),values.begin()+middle))*.5;
        return result;
    }
};
