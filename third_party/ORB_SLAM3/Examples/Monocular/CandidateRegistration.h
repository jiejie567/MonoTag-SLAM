#pragma once
#include <deque>

// Registration proposals only: these samples must never become pose hints.
struct CandidateRegistrationSample {
    TagObservation tag;
    Sophus::SE3f Tcw, Twm;
    double time;
};
struct CandidateRegistrationWindow {
    std::deque<CandidateRegistrationSample> samples;
    unsigned long mapId=0;
    int revision=-1;
    std::string component;

    bool add(const TagObservation& tag, const Sophus::SE3f& Tcw, double time,
             unsigned long map, int rev, const cv::Mat& K, float imageScale,
             Sophus::SE3f& result, float& worstRms) {
        if(component!=tag.component || mapId!=map || revision!=rev ||
           (!samples.empty() && (time-samples.back().time>0.5 || time<=samples.back().time)))
            samples.clear();
        component=tag.component;mapId=map;revision=rev;
        while(!samples.empty() && time-samples.front().time>1.0) samples.pop_front();
        if(tag.partial || tag.worldPoints.size()!=4 || tag.imagePoints.size()!=4 ||
           tag.tagIds.size()!=4 || tag.pointWeights.size()!=4 || !std::isfinite(tag.confidence) ||
           tag.confidence<0.15f || !Tcw.matrix().allFinite()) return false;
        for(int id:tag.tagIds) if(id!=tag.tagIds.front()) return false;
        for(float weight:tag.pointWeights) if(weight<0.99f) return false;
        Sophus::SE3f Twm=Tcw.inverse()*tag.Twc.inverse();
        if(!Twm.matrix().allFinite()) return false;
        if(!samples.empty() && !ORB_SLAM3::MarkerComponentTransformsConsistent(samples.front().Twm,Twm))
            samples.clear();
        samples.push_back({tag,Tcw,Twm,time});
        while(samples.size()>32) samples.pop_front();
        bool strong=false,candidate=false;
        for(const auto& s:samples) {
            strong|=!s.tag.candidate && s.tag.confidence>=0.35f;
            candidate|=s.tag.candidate;
        }
        if(samples.size()<3 || !strong || !candidate) return false;
        Eigen::Vector3f translation=Eigen::Vector3f::Zero();
        Eigen::Vector4f sum=Eigen::Vector4f::Zero();
        Eigen::Quaternionf reference(samples.front().Twm.rotationMatrix());
        for(const auto& s:samples) {
            translation+=s.Twm.translation();
            Eigen::Quaternionf q(s.Twm.rotationMatrix());
            if(reference.dot(q)<0) q.coeffs()*=-1;
            sum+=q.coeffs();
        }
        Eigen::Quaternionf q;q.coeffs()=sum.normalized();
        result=Sophus::SE3f(q.normalized(),translation/float(samples.size()));
        worstRms=0;
        for(const auto& s:samples) {
            float squared=0;
            for(size_t i=0;i<4;++i) {
                auto p=s.Tcw*(result*s.tag.worldPoints[i]);
                if(!p.allFinite() || p.z()<=0) return false;
                float u=K.at<double>(0,0)*p.x()/p.z()+K.at<double>(0,2);
                float v=K.at<double>(1,1)*p.y()/p.z()+K.at<double>(1,2);
                float du=u-s.tag.imagePoints[i].x*imageScale;
                float dv=v-s.tag.imagePoints[i].y*imageScale;
                squared+=du*du+dv*dv;
            }
            if(!std::isfinite(squared)) return false;
            worstRms=std::max(worstRms,std::sqrt(squared/4)/imageScale);
        }
        return std::isfinite(worstRms) && worstRms<=2.5f;
    }
};
