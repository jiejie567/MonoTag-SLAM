#include "TemporalFlow.h"
#include <iostream>
#include <stdexcept>

static void require(bool ok,const char* message) {
    if(!ok) throw std::runtime_error(message);
}
int main() {
    try {
        cv::setNumThreads(1);
        cv::Mat a(480,640,CV_8UC1),b;
        cv::RNG rng(7201);rng.fill(a,cv::RNG::UNIFORM,0,256);
        cv::GaussianBlur(a,a,cv::Size(3,3),.7);
        const cv::Mat transform=(cv::Mat_<double>(2,3)<<1,0,4,0,1,-2);
        cv::warpAffine(a,b,transform,a.size());
        std::vector<cv::Point2f> points;
        for(int y=40;y<440;y+=30)for(int x=40;x<600;x+=30)points.emplace_back(x,y);
        auto q=points;
        for(auto& p:q)p+=cv::Point2f(3,-1); // deliberately imperfect prior
        auto valid=ORB_SLAM3::TemporalFlow(a,b,points,q,cv::Mat());
        int count=0;
        for(size_t i=0;i<q.size();++i)if(valid[i]) {
            ++count;require(cv::norm(q[i]-points[i]-cv::Point2f(4,-2))<.4,"bad translation");
        }
        require(count>int(points.size()*.9),"insufficient translation coverage");
        cv::Mat blocked=cv::Mat::zeros(a.size(),CV_8UC1);
        q=points;valid=ORB_SLAM3::TemporalFlow(a,b,points,q,blocked);
        require(cv::countNonZero(valid)==0,"excluded foreground survived");
        cv::Mat blank=cv::Mat::zeros(a.size(),CV_8UC1);
        q=points;valid=ORB_SLAM3::TemporalFlow(blank,blank,points,q,cv::Mat());
        require(cv::countNonZero(valid)==0,"textureless flow accepted");
        q.clear();valid=ORB_SLAM3::TemporalFlow(a,b,{},q,cv::Mat());
        require(valid.empty(),"empty input failed");
        std::cout<<"temporal flow translation/mask/texture/empty tests passed\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
