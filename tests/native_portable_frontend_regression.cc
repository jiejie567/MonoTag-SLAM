// Focused ORBextractor regression; no SLAM, datasets, vocabulary or RNG state.
// ORB_SLAM3_PORTABLE_FRONTEND=1 selects exact resize and canonical feature order.
// Unset, 0 and other values keep the legacy frontend. Configure before extraction.
#include "ORBextractor.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition, const char* message)
{
    if(!condition) throw std::runtime_error(message);
}

class ExtractorProbe : public ORB_SLAM3::ORBextractor {
public:
    ExtractorProbe() : ORBextractor(600, 1.2f, 8, 20, 7) {}
    using ORBextractor::ComputePyramid;
    using ORBextractor::DistributeOctTree;
};

bool sameKey(const cv::KeyPoint& a, const cv::KeyPoint& b)
{
    return a.pt==b.pt && a.size==b.size && a.angle==b.angle &&
           a.response==b.response && a.octave==b.octave && a.class_id==b.class_id;
}

cv::Mat texture()
{
    cv::Mat image(479, 641, CV_8UC1);
    uint32_t state=123456789u;
    for(int y=0; y<image.rows; ++y) for(int x=0; x<image.cols; ++x) {
        state ^= state << 13; state ^= state >> 17; state ^= state << 5;
        image.at<unsigned char>(y,x)=static_cast<unsigned char>(state >> 24);
    }
    return image;
}

void testPyramid(const cv::Mat& image)
{
    bool modesDiffer=false;
    std::vector<cv::Mat> legacy;
    for(const char* option : {"0", "1"}) {
        setenv("ORB_SLAM3_PORTABLE_FRONTEND", option, 1);
        ExtractorProbe extractor;
        extractor.ComputePyramid(image);
        cv::Mat expected=image.clone();
        const auto scales=extractor.GetInverseScaleFactors();
        for(int level=0; level<extractor.GetLevels(); ++level) {
            if(level) {
                const cv::Size size(cvRound(float(image.cols)*scales[level]),
                                    cvRound(float(image.rows)*scales[level]));
                cv::resize(expected, expected, size, 0, 0,
                           option[0]=='1' ? cv::INTER_LINEAR_EXACT : cv::INTER_LINEAR);
            }
            const auto& actual=extractor.mvImagePyramid[level];
            require(actual.size()==expected.size() && cv::norm(actual,expected,cv::NORM_INF)==0,
                    "pyramid differs from requested interpolation");
            if(option[0]=='0') legacy.push_back(actual.clone());
            else modesDiffer |= cv::norm(actual,legacy[level],cv::NORM_INF)!=0;
            // Include the halo used by orientation, not only visible pixels.
            cv::Mat padded;
            cv::copyMakeBorder(expected,padded,19,19,19,19,cv::BORDER_REFLECT_101);
            cv::Mat actualPadded=actual;
            actualPadded.adjustROI(19,19,19,19);
            require(actualPadded.size()==padded.size() && cv::norm(actualPadded,padded,cv::NORM_INF)==0,
                    "pyramid reflection halo changed");
        }
    }
    require(modesDiffer,"fixture must distinguish exact and legacy interpolation");
}

void testOctreeTies()
{
    setenv("ORB_SLAM3_PORTABLE_FRONTEND","1",1);
    ExtractorProbe extractor;
    std::vector<cv::KeyPoint> candidates;
    for(int y : {10,30,60,80}) for(int x : {10,30,60,80})
        candidates.emplace_back(float(x),float(y),7.f,-1.f,20.f,0,-1);
    // Four equal-population nodes: the canonical largest x/y node expands.
    // Each unexpanded node chooses its first image-coordinate response tie.
    std::vector<cv::Point2f> expected{{10,10},{60,10},{10,60},
                                     {60,60},{80,60},{60,80},{80,80}};
    for(int weighted=0; weighted<2; ++weighted) {
        if(weighted) {
            for(auto& key : candidates) if(key.pt==cv::Point2f(30,30)) key.response=21.f;
            expected[0]=cv::Point2f(30,30);
            // Canonical image order now places the upper-right representative first.
            std::swap(expected[0],expected[1]);
        }
        std::vector<cv::Point2f> referenceOrder;
        for(int repeat=0; repeat<20; ++repeat) {
            std::reverse(candidates.begin(),candidates.end());
            std::rotate(candidates.begin(),candidates.begin()+repeat%candidates.size(),candidates.end());
            const auto keys=extractor.DistributeOctTree(candidates,0,100,0,100,5,0);
            require(keys.size()==expected.size(),"octree tie fixture feature count changed");
            std::vector<cv::Point2f> actual;
            for(const auto& key : keys) actual.push_back(key.pt);
            if(repeat==0) referenceOrder=actual;
            require(actual==referenceOrder,"octree traversal order depends on candidate input order");
            const auto imageOrder=[](const cv::Point2f& a,const cv::Point2f& b) {
                return a.y==b.y ? a.x<b.x : a.y<b.y;
            };
            std::sort(actual.begin(),actual.end(),imageOrder);
            auto expectedSet=expected;
            std::sort(expectedSet.begin(),expectedSet.end(),imageOrder);
            require(actual==expectedSet,"octree node/response tie selected a different point");
        }
    }
}

struct Features {
    std::vector<cv::KeyPoint> keys;
    cv::Mat descriptors;
    int mono;
};

Features extract(ExtractorProbe& extractor, const cv::Mat& image, bool parallel,
                 std::vector<int> overlap)
{
    setenv("ORB_SLAM3_PARALLEL_DESCRIPTORS",parallel ? "1" : "0",1);
    Features result;
    result.mono=extractor(image,cv::Mat(),result.keys,result.descriptors,overlap);
    require(!result.keys.empty(),"empty feature fixture");
    require(result.descriptors.rows==int(result.keys.size()),"key/descriptor count mismatch");
    return result;
}

void compare(const Features& a, const Features& b)
{
    require(a.mono==b.mono && a.keys.size()==b.keys.size(),"feature partition/count changed");
    require(cv::norm(a.descriptors,b.descriptors,cv::NORM_INF)==0,"descriptor bytes changed");
    for(size_t i=0; i<a.keys.size(); ++i)
        require(sameKey(a.keys[i],b.keys[i]),"keypoint order/metadata changed");
}

void testExtraction(const cv::Mat& image)
{
    for(bool portable : {false,true}) for(bool masked : {false,true}) {
        setenv("ORB_SLAM3_PORTABLE_FRONTEND",portable ? "1" : "0",1);
        ExtractorProbe extractor;
        cv::Mat mask(image.size(),CV_8UC1,cv::Scalar(255));
        if(masked) {
            mask(cv::Rect(image.cols/4,image.rows/4,image.cols/2,image.rows/2))=0;
            extractor.SetAllowedMask(mask);
        }
        const auto baseline=extract(extractor,image,false,{0,0});
        require(baseline.mono==int(baseline.keys.size()),"all-mono partition changed");
        compare(baseline,extract(extractor,image,false,{0,0}));
        compare(baseline,extract(extractor,image,true,{0,0}));
        for(size_t i=0; i<baseline.keys.size(); ++i) {
            const auto& key=baseline.keys[i];
            require(mask.at<unsigned char>(std::lround(key.pt.y),std::lround(key.pt.x))!=0,
                    "feature escaped allowed mask");
            if(portable && i) {
                const auto& prev=baseline.keys[i-1];
                require(prev.octave<=key.octave,"final features changed pyramid level order");
            }
        }
        const int left=image.cols/3, right=2*image.cols/3;
        const auto mixed=extract(extractor,image,false,{left,right});
        compare(mixed,extract(extractor,image,true,{left,right}));
        std::vector<size_t> mono,stereo;
        for(size_t i=0; i<baseline.keys.size(); ++i) {
            const float x=baseline.keys[i].pt.x;
            (x>=left && x<=right ? stereo : mono).push_back(i);
        }
        require(!mono.empty() && !stereo.empty(),"fixture must exercise both stereo partitions");
        require(mixed.mono==int(mono.size()),"mixed mono count changed");
        mono.insert(mono.end(),stereo.rbegin(),stereo.rend());
        require(mixed.keys.size()==mono.size(),"mixed partition lost features");
        for(size_t i=0; i<mono.size(); ++i) {
            require(sameKey(mixed.keys[i],baseline.keys[mono[i]]),"stereo key order changed");
            require(cv::norm(mixed.descriptors.row(int(i)),baseline.descriptors.row(int(mono[i])),
                             cv::NORM_INF)==0,"descriptor row detached from its keypoint");
        }
        if(!portable) {
            unsetenv("ORB_SLAM3_PORTABLE_FRONTEND");
            compare(baseline,extract(extractor,image,false,{0,0}));
            setenv("ORB_SLAM3_PORTABLE_FRONTEND","true",1);
            compare(baseline,extract(extractor,image,false,{0,0}));
        }
    }
}
} // namespace

int main()
{
    try {
        cv::setNumThreads(1);
        const cv::Mat image=texture();
        testPyramid(image);
        testOctreeTies();
        testExtraction(image);
        std::cout << "portable frontend: pyramid/halo, octree ties, mask, repeat, parallel and stereo pairing passed\n";
        return 0;
    } catch(const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
