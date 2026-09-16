#pragma once

#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <cmath>

namespace ORB_SLAM3 {

// Half-resolution, prediction-guided LK. Outputs remain in full-image pixels.
// A flow observation is only a matching proposal, never a pose or map point.
inline std::vector<unsigned char> TemporalFlow(
        const cv::Mat& previous, const cv::Mat& current,
        const std::vector<cv::Point2f>& source, std::vector<cv::Point2f>& target,
        const cv::Mat& allowedMask)
{
    CV_Assert(previous.type()==CV_8UC1 && current.type()==CV_8UC1);
    CV_Assert(previous.size()==current.size() && source.size()==target.size());
    CV_Assert(allowedMask.empty() || (allowedMask.type()==CV_8UC1 &&
                                    allowedMask.size()==current.size()));
    std::vector<unsigned char> valid(source.size(),0);
    if(source.empty()) return valid;
    cv::Mat a,b;
    cv::resize(previous,a,cv::Size(),.5,.5,cv::INTER_AREA);
    cv::resize(current,b,a.size(),0,0,cv::INTER_AREA);
    std::vector<cv::Point2f> p=source,q=target,back;
    for(auto& x:p)x*=.5f;
    for(auto& x:q)x*=.5f;
    std::vector<unsigned char> forward,reverse;
    std::vector<float> error,backError;
    const cv::TermCriteria criteria(cv::TermCriteria::COUNT|cv::TermCriteria::EPS,20,.01);
    cv::calcOpticalFlowPyrLK(a,b,p,q,forward,error,cv::Size(15,15),2,criteria,
                           cv::OPTFLOW_USE_INITIAL_FLOW);
    // Only pass finite forward results into the reverse solver.
    back=p;
    for(size_t i=0;i<q.size();++i)
        if(!std::isfinite(q[i].x) || !std::isfinite(q[i].y)) {
            forward[i]=0;q[i]=p[i];
        }
    cv::calcOpticalFlowPyrLK(b,a,q,back,reverse,backError,cv::Size(15,15),2,criteria,
                           cv::OPTFLOW_USE_INITIAL_FLOW);
    for(size_t i=0;i<q.size();++i) {
        target[i]=q[i]*2.f;
        const auto& x=target[i];
        if(!forward[i] || !reverse[i] || !std::isfinite(error[i]) || error[i]>20 ||
           !std::isfinite(back[i].x) || !std::isfinite(back[i].y) ||
           cv::norm(back[i]-p[i])>.5 || x.x<8 || x.y<8 ||
           x.x>=current.cols-8 || x.y>=current.rows-8) continue;
        if(!allowedMask.empty() && !allowedMask.at<unsigned char>(cvRound(x.y),cvRound(x.x)))
            continue;
        valid[i]=1;
    }
    return valid;
}
}
