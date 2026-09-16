#pragma once
#include <algorithm>
#include <cmath>
#include <map>
#include <vector>

namespace ORB_SLAM3 {
struct LoopMarkerEvidenceView {
    unsigned long keyframe;
    int marker;
    double timestamp;
    double maximumErrorPx;
    bool strong;
};
inline std::map<int,std::map<unsigned long,double>> ReliableLoopMarkerViews(
    const std::vector<LoopMarkerEvidenceView>& views,double center) {
    std::map<int,std::map<unsigned long,double>> result;
    for(const auto& v:views)
        if(v.strong && std::isfinite(v.timestamp) && std::abs(v.timestamp-center)<=2.0 &&
           std::isfinite(v.maximumErrorPx) && v.maximumErrorPx>=0 && v.maximumErrorPx<=3.0)
            result[v.marker][v.keyframe]=v.timestamp;
    return result;
}
inline bool HasCommonLoopMarkerEvidence(const std::vector<LoopMarkerEvidenceView>& first,
    const std::vector<LoopMarkerEvidenceView>& second,double firstTime,double secondTime) {
    if(!std::isfinite(firstTime) || !std::isfinite(secondTime) ||
       std::abs(firstTime-secondTime)<=4.0) return false;
    const auto a=ReliableLoopMarkerViews(first,firstTime),b=ReliableLoopMarkerViews(second,secondTime);
    const auto enough=[](const std::map<unsigned long,double>& views) {
        if(views.size()<2) return false;
        double lo=views.begin()->second,hi=lo;
        for(const auto& v:views) {lo=std::min(lo,v.second);hi=std::max(hi,v.second);}
        return hi-lo>=0.1;
    };
    for(const auto& entry:a) {
        auto other=b.find(entry.first);
        if(other==b.end() || !enough(entry.second) || !enough(other->second)) continue;
        bool overlap=false;
        for(const auto& v:entry.second) overlap=overlap || other->second.count(v.first);
        if(!overlap) return true;
    }
    return false;
}
}
