#include "ORBextractor.h"
#include <opencv2/imgcodecs.hpp>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

int main(int argc,char** argv) {
    try {
        if(argc<2) throw std::runtime_error("supply grayscale-readable images");
        cv::setNumThreads(1);
        double serialMs=0,parallelMs=0;
        int comparisons=0;
        for(int file=1;file<argc;++file) {
            const cv::Mat image=cv::imread(argv[file],cv::IMREAD_GRAYSCALE);
            if(image.empty())throw std::runtime_error("image unreadable");
            for(int featureCount:{2000,10000}) for(bool masked:{false,true}) {
                ORB_SLAM3::ORBextractor extractor(featureCount,1.2,8,20,7);
                if(masked) {
                    cv::Mat mask(image.size(),CV_8UC1,cv::Scalar(255));
                    mask(cv::Rect(image.cols/4,image.rows/4,image.cols/2,image.rows/2))=0;
                    extractor.SetAllowedMask(mask);
                }
                for(int repeat=0;repeat<10;++repeat) {
                    std::vector<cv::KeyPoint> keys[2];cv::Mat descriptors[2];int mono[2];
                    // Alternate order; warm-up pair excluded from timing.
                    for(int order=0;order<2;++order) {
                        const int mode=(order+repeat)%2;
                        setenv("ORB_SLAM3_PARALLEL_DESCRIPTORS",mode?"1":"0",1);
                        std::vector<int> lapping{0,1000};
                        const auto begin=std::chrono::steady_clock::now();
                        mono[mode]=extractor(image,cv::Mat(),keys[mode],descriptors[mode],lapping);
                        const double ms=std::chrono::duration<double,std::milli>(
                            std::chrono::steady_clock::now()-begin).count();
                        if(repeat) (mode?parallelMs:serialMs)+=ms;
                    }
                    if(mono[0]!=mono[1] || keys[0].size()!=keys[1].size() ||
                       descriptors[0].size()!=descriptors[1].size())
                        throw std::runtime_error("feature count/order mismatch");
                    if(!descriptors[0].empty() && cv::norm(descriptors[0],descriptors[1],cv::NORM_INF)!=0)
                        throw std::runtime_error("descriptor bytes changed");
                    for(size_t i=0;i<keys[0].size();++i) {
                        const auto& a=keys[0][i];const auto& b=keys[1][i];
                        if(a.pt!=b.pt || a.angle!=b.angle || a.octave!=b.octave ||
                           a.size!=b.size || a.response!=b.response)
                            throw std::runtime_error("keypoint changed");
                    }
                    ++comparisons;
                }
            }
        }
        std::cout<<"identical_pairs="<<comparisons<<" serial_ms="<<serialMs
                 <<" parallel_ms="<<parallelMs<<" speedup="<<serialMs/parallelMs<<'\n';
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
