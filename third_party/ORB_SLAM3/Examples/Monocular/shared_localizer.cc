// Offline pre-initialization recovery against an explicit, immutable final map.
// No System, tracking/mapping workers, triangulation, Atlas save or interpolation.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <sys/stat.h>
#include <boost/archive/binary_iarchive.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include "Atlas.h"
#include "KeyFrameDatabase.h"
#include "ORBextractor.h"

using namespace ORB_SLAM3;
namespace {
struct Query {
    int frame=-1; double time=0;
    bool supportOnly=false;
    std::vector<std::vector<cv::Point>> excluded;
    cv::Mat frozenTwc;
};
struct FeatureMatch {
    cv::Point2d pixel;
    unsigned long pointId;
    double error;
};
struct Result {
    int frame=-1,features=0,matches=0,inliers=0,cells=0;
    double time=0,rms=-1,p95=-1,hull=0,ratio=0;
    bool candidate=false,connected=false,accepted=false,supportOnly=false;
    std::string reason="not_attempted";
    cv::Mat Twc;
    cv::Mat independentTwc;
    std::vector<FeatureMatch> matchedFeatures;
    std::vector<cv::KeyPoint> queryKeys;
    cv::Mat queryDescriptors;
    cv::Mat grayImage;
    bool guided=false;
    int guideBefore=-1,guideAfter=-1;
};
cv::Mat matrix(const cv::FileNode& node,int rows,int cols) {
    if(node.size()!=size_t(rows)) throw std::runtime_error("invalid_matrix_shape");
    cv::Mat value(rows,cols,CV_64F);
    for(int r=0;r<rows;++r) {
        if(node[r].size()!=size_t(cols)) throw std::runtime_error("invalid_matrix_shape");
        for(int c=0;c<cols;++c) value.at<double>(r,c)=double(node[r][c]);
    }
    if(!cv::checkRange(value)) throw std::runtime_error("nonfinite_matrix");
    return value;
}
cv::Mat frozenPose(const cv::FileNode& node) {
    cv::Mat value=matrix(node,4,4),rotation=value(cv::Rect(0,0,3,3));
    const cv::Mat identity=cv::Mat::eye(3,3,CV_64F);
    if(cv::norm(rotation.t()*rotation-identity,cv::NORM_INF)>1e-6 ||
       std::abs(cv::determinant(rotation)-1.)>1e-6 ||
       std::abs(value.at<double>(3,0))+std::abs(value.at<double>(3,1))+
       std::abs(value.at<double>(3,2))+std::abs(value.at<double>(3,3)-1.)>1e-9)
        throw std::runtime_error("invalid_frozen_camera_se3");
    return value;
}
std::vector<std::vector<cv::Point>> polygons(const cv::FileNode& node) {
    std::vector<std::vector<cv::Point>> result;
    for(auto polygon:node) {
        std::vector<cv::Point> points;
        for(auto point:polygon) {
            if(point.size()!=2) throw std::runtime_error("invalid_mask_point");
            const double x=double(point[0]),y=double(point[1]);
            if(!std::isfinite(x)||!std::isfinite(y)||std::abs(x)>100000||std::abs(y)>100000)
                throw std::runtime_error("invalid_mask_coordinate");
            points.emplace_back(cvRound(x),cvRound(y));
        }
        if(points.size()>=3) {
            std::vector<cv::Point> hull;cv::convexHull(points,hull);result.push_back(std::move(hull));
        }
    }
    return result;
}
std::string shellQuote(const std::string& text) {
    std::string quoted="'";for(char c:text)quoted+=c=='\''?"'\\''":std::string(1,c);return quoted+"'";
}
void rejectOutputAliasing(const std::string& output,const std::string& input) {
    struct stat a,b;
    if(output==input || (stat(output.c_str(),&a)==0&&stat(input.c_str(),&b)==0&&
       a.st_dev==b.st_dev&&a.st_ino==b.st_ino)) throw std::runtime_error("output_aliases_immutable_input");
}
std::string readCommand(const std::string& command) {
    FILE* pipe=popen(command.c_str(),"r");if(!pipe)throw std::runtime_error("probe_start_failed");
    std::string text;char buffer[4096];size_t count;
    while((count=fread(buffer,1,sizeof(buffer),pipe))>0)text.append(buffer,count);
    if(pclose(pipe)!=0)throw std::runtime_error("probe_failed");return text;
}
class GrayVideoReader {
    FILE* pipe=nullptr;cv::Size size;int last,lastFrame=-1;std::string video,ffmpeg;
    std::vector<int> schedule;size_t cursor=0;cv::Mat cached;
public:
    double decodingSeconds=0.;
    GrayVideoReader(const std::string& v,const std::string& f,cv::Size s,int l,const std::vector<int>& frames={}):size(s),last(l),video(v),ffmpeg(f),schedule(frames) {}
    ~GrayVideoReader(){if(pipe)pclose(pipe);}
    cv::Mat imageAt(int frame) {
        if(frame==lastFrame)return cached;
        auto started=std::chrono::steady_clock::now();
        if(!pipe) {
            if(schedule.empty())for(int i=frame;i<=last;i++)schedule.push_back(i);
            std::string filter="select=";
            for(size_t i=0;i<schedule.size();) {
                size_t j=i;while(j+1<schedule.size()&&schedule[j+1]==schedule[j]+1)j++;
                if(i)filter+="+";
                filter+="between(n\\,"+std::to_string(schedule[i])+"\\,"+std::to_string(schedule[j])+")";i=j+1;
            }
            // Four codec threads preserve exact frames, with bounded memory.
            // Keep conversion/output at one thread and all frame selection intact.
            const auto command=shellQuote(ffmpeg)+" -v error -threads 4 -noautorotate -i "+shellQuote(video)+
                " -map 0:v:0 -vf "+shellQuote(filter)+" -vsync 0 -frames:v "+std::to_string(schedule.size())+
                " -threads 1 -f rawvideo -pix_fmt bgr24 -";
            pipe=popen(command.c_str(),"r");if(!pipe)throw std::runtime_error("decoder_start_failed");
        }
        if(frame<lastFrame||cursor>=schedule.size()||!std::binary_search(schedule.begin()+cursor,schedule.end(),frame))throw std::runtime_error("nonsequential_decode_request");
        cv::Mat image(size,CV_8UC3);
        while(cursor<schedule.size()&&schedule[cursor]<=frame) {
            size_t count=0,total=image.total()*image.elemSize();
            while(count<total){size_t n=fread(image.data+count,1,total-count,pipe);if(!n)throw std::runtime_error("decode_failed");count+=n;}
            lastFrame=schedule[cursor++];
        }
        cv::cvtColor(image,image,cv::COLOR_BGR2GRAY);cached=image;
        decodingSeconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();return image;
    }
};
double translationDistance(const cv::Mat& a,const cv::Mat& b) {
    return cv::norm(a(cv::Rect(3,0,1,3))-b(cv::Rect(3,0,1,3)));
}
double rotationDistance(const cv::Mat& a,const cv::Mat& b) {
    cv::Mat r;cv::Rodrigues(a(cv::Rect(0,0,3,3))*b(cv::Rect(0,0,3,3)).t(),r);
    return cv::norm(r)*180./CV_PI;
}
Result localize(const Query& query,GrayVideoReader& reader,ORBextractor& extractor,
                const std::vector<cv::Point3d>& points,const cv::Mat& descriptors,
                const std::vector<unsigned long>& pointIds,
                const cv::Mat& K,const cv::Mat& D,cv::Size size,
                const Result* cached=nullptr,const cv::Mat& searchPose=cv::Mat()) {
    Result result;result.frame=query.frame;result.time=query.time;result.supportOnly=query.supportOnly;
    cv::Mat image;
    if(!cached) {
        image=reader.imageAt(query.frame);
        if(image.size()!=size || image.type()!=CV_8UC1) throw std::runtime_error("calibration_image_mismatch");
    }
    cv::Mat mask(size,CV_8U,cv::Scalar(255));
    if(!query.excluded.empty()) cv::fillPoly(mask,query.excluded,cv::Scalar(0));
    cv::erode(mask,mask,cv::Mat::ones(17,17,CV_8U));
    extractor.SetAllowedMask(mask);
    std::vector<cv::KeyPoint> keys;cv::Mat queryDescriptors;std::vector<int> lapping{0,0};
    if(cached) {keys=cached->queryKeys;queryDescriptors=cached->queryDescriptors;}
    else extractor(image,mask,keys,queryDescriptors,lapping);
    result.grayImage=cached?cached->grayImage:image;
    result.features=int(keys.size());result.queryKeys=keys;result.queryDescriptors=queryDescriptors;
    if(keys.size()<30 || points.size()<30) {result.reason="insufficient_features";return result;}
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> forward;std::vector<cv::DMatch> reverse;
    cv::Mat searchMask;
    if(!searchPose.empty()) {
        // Search prior only: no seed pose is supplied to PnP or emitted as a label.
        cv::Mat R=searchPose(cv::Rect(0,0,3,3)).t();
        cv::Mat t=-R*searchPose(cv::Rect(3,0,1,3)),r;cv::Rodrigues(R,r);
        std::vector<cv::Point2d> projected;cv::projectPoints(points,r,t,K,D,projected);
        searchMask=cv::Mat::zeros(int(keys.size()),int(points.size()),CV_8U);
        for(size_t j=0;j<points.size();++j) {
            const auto& p=points[j];
            const double z=R.at<double>(2,0)*p.x+R.at<double>(2,1)*p.y+R.at<double>(2,2)*p.z+t.at<double>(2);
            if(z<=0||!std::isfinite(projected[j].x)||!std::isfinite(projected[j].y))continue;
            for(size_t i=0;i<keys.size();++i)
                if(cv::norm(cv::Point2d(keys[i].pt)-projected[j])<=32.)searchMask.at<unsigned char>(i,j)=255;
        }
    }
    matcher.knnMatch(queryDescriptors,descriptors,forward,2,searchMask);
    if(query.frozenTwc.empty()) matcher.match(descriptors,queryDescriptors,reverse,searchMask.empty()?cv::Mat():searchMask.t());
    else {
        // Display correspondences require an independent ratio gate in BOTH
        // descriptor directions, not a projected point masquerading as a match.
        std::vector<std::vector<cv::DMatch>> backward;
        matcher.knnMatch(descriptors,queryDescriptors,backward,2);
        for(const auto& pair:backward) if(pair.size()==2 &&
            pair[0].distance<55 && pair[0].distance<.75*pair[1].distance) reverse.push_back(pair[0]);
    }
    std::vector<int> mutual(points.size(),-1);
    for(auto match:reverse) mutual[match.queryIdx]=match.trainIdx;
    std::vector<cv::Point3d> world;std::vector<cv::Point2d> pixels;
    std::vector<unsigned long> matchedPointIds;
    for(const auto& pair:forward) if(pair.size()==2) {
        const auto& a=pair[0];const auto& b=pair[1];
        if(a.distance<55 && a.distance<.75*b.distance && mutual[a.trainIdx]==a.queryIdx) {
            world.push_back(points[a.trainIdx]);pixels.emplace_back(keys[a.queryIdx].pt);
            matchedPointIds.push_back(pointIds[a.trainIdx]);
        }
    }
    result.matches=int(world.size());
    if(world.size()<30) {result.reason="insufficient_matches";return result;}
    cv::Mat r,t,ids;cv::setRNGSeed(123);
    if(!cv::solvePnPRansac(world,pixels,K,D,r,t,false,1000,3.,.999,ids,cv::SOLVEPNP_EPNP)||ids.total()<30) {
        result.inliers=int(ids.total());result.reason="pnp_inliers";return result;
    }
    std::vector<cv::Point3d> fitWorld;std::vector<cv::Point2d> fitPixels;
    for(int i=0;i<ids.rows;++i) {const int j=ids.at<int>(i);fitWorld.push_back(world[j]);fitPixels.push_back(pixels[j]);}
    cv::solvePnPRefineLM(fitWorld,fitPixels,K,D,r,t);
    cv::Mat R;cv::Rodrigues(r,R);
    if(!cv::checkRange(R)||!cv::checkRange(t)) {result.reason="nonfinite_pose";return result;}
    std::vector<cv::Point2d> projected;cv::projectPoints(world,r,t,K,D,projected);
    std::vector<double> errors;std::vector<cv::Point2f> support;std::set<int> cells;
    std::vector<bool> independentInliers(world.size(),false);
    for(size_t i=0;i<world.size();++i) {
        const double error=cv::norm(projected[i]-pixels[i]);
        const double depth=R.at<double>(2,0)*world[i].x+R.at<double>(2,1)*world[i].y+R.at<double>(2,2)*world[i].z+t.at<double>(2);
        if(std::isfinite(error)&&(query.frozenTwc.empty()?error<3.:error<=3.)&&depth>0) {
            independentInliers[i]=true;
            errors.push_back(error);support.emplace_back(pixels[i]);
            const int x=int(pixels[i].x*4/size.width),y=int(pixels[i].y*3/size.height);
            if(x>=0&&x<4&&y>=0&&y<3) cells.insert(y*4+x);
        }
    }
    result.inliers=int(errors.size());result.ratio=double(errors.size())/world.size();result.cells=int(cells.size());
    if(errors.size()<30) {result.reason="refined_inliers";return result;}
    std::vector<cv::Point2f> hull;cv::convexHull(support,hull);
    result.hull=cv::contourArea(hull)/double(size.area());
    double squared=0;for(double e:errors)squared+=e*e;
    result.rms=std::sqrt(squared/errors.size());std::sort(errors.begin(),errors.end());
    result.p95=errors[size_t(.95*(errors.size()-1))];
    result.independentTwc=cv::Mat::eye(4,4,CV_64F);
    cv::Mat independentR=R.t(),independentT=-independentR*t;
    independentR.copyTo(result.independentTwc(cv::Rect(0,0,3,3)));
    independentT.copyTo(result.independentTwc(cv::Rect(3,0,1,3)));
    if(query.frozenTwc.empty()) {
        // Retain the actual matches used by prefix PnP. Publishing evidence is
        // independent of all pose/ratio/temporal acceptance gates below.
        std::map<std::pair<double,double>,FeatureMatch> uniquePixels;
        for(size_t i=0;i<world.size();++i) if(independentInliers[i]) {
            const double error=cv::norm(projected[i]-pixels[i]);
            const auto key=std::make_pair(pixels[i].x,pixels[i].y);
            const auto old=uniquePixels.find(key);
            if(old==uniquePixels.end()||error<old->second.error)
                uniquePixels[key]={pixels[i],matchedPointIds[i],error};
        }
        for(const auto& item:uniquePixels)result.matchedFeatures.push_back(item.second);
    }
    result.candidate=result.ratio>=.45&&result.cells>=5&&result.hull>=.06;
    result.reason=result.candidate?"geometric_candidate":"coverage_or_ratio";
    if(result.candidate) {
        result.Twc=cv::Mat::eye(4,4,CV_64F);cv::Mat Rt=R.t(),position=-Rt*t;
        Rt.copyTo(result.Twc(cv::Rect(0,0,3,3)));position.copyTo(result.Twc(cv::Rect(3,0,1,3)));
        if(!searchPose.empty() && (translationDistance(result.Twc,searchPose)>.05 ||
                                  rotationDistance(result.Twc,searchPose)>5.)) {
            result.candidate=false;result.reason="guided_pose_disagrees";
        }
    }
    if(!query.frozenTwc.empty() && result.candidate) {
        // PnP is only an independent correspondence check. Never publish it as
        // a camera correction: every emitted residual uses the saved camera.
        if(translationDistance(result.Twc,query.frozenTwc)>.05 ||
           rotationDistance(result.Twc,query.frozenTwc)>5.) {
            result.candidate=false;result.reason="frozen_pose_disagrees_with_pnp";return result;
        }
        R=query.frozenTwc(cv::Rect(0,0,3,3)).t();
        t=-R*query.frozenTwc(cv::Rect(3,0,1,3));cv::Rodrigues(R,r);
        cv::projectPoints(world,r,t,K,D,projected);
        errors.clear();support.clear();cells.clear();
        std::map<std::pair<double,double>,FeatureMatch> uniquePixels;
        for(size_t i=0;i<world.size();++i) {
            const double error=cv::norm(projected[i]-pixels[i]);
            const double depth=R.at<double>(2,0)*world[i].x+R.at<double>(2,1)*world[i].y+R.at<double>(2,2)*world[i].z+t.at<double>(2);
            const int u=cvRound(pixels[i].x),v=cvRound(pixels[i].y);
            if(independentInliers[i]&&std::isfinite(error)&&error<=3.&&depth>0.&&
               u>=0&&u<size.width&&v>=0&&v<size.height&&mask.at<unsigned char>(v,u)!=0) {
                const auto key=std::make_pair(pixels[i].x,pixels[i].y);
                const auto old=uniquePixels.find(key);
                if(old==uniquePixels.end()||error<old->second.error)
                    uniquePixels[key]={pixels[i],matchedPointIds[i],error};
            }
        }
        // Multi-octave descriptors can land on the same original pixel. Keep
        // only its lowest frozen-pose residual, then measure the emitted set.
        for(const auto& item:uniquePixels) {
            const auto& match=item.second;result.matchedFeatures.push_back(match);
            errors.push_back(match.error);support.emplace_back(match.pixel);
            cells.insert(int(match.pixel.y*3/size.height)*4+int(match.pixel.x*4/size.width));
        }
        result.inliers=int(errors.size());result.ratio=double(errors.size())/world.size();result.cells=int(cells.size());
        result.hull=0.;result.rms=result.p95=-1.;
        if(!errors.empty()) {
            cv::convexHull(support,hull);result.hull=cv::contourArea(hull)/double(size.area());
            double squared=0.;for(double error:errors)squared+=error*error;
            result.rms=std::sqrt(squared/errors.size());std::sort(errors.begin(),errors.end());
            result.p95=errors[size_t(.95*(errors.size()-1))];
        }
        // The independent PnP candidate already passed the full ratio gate.
        // This smaller dual-pose-consistent subset is display-only, not a new
        // camera estimate; require count/coverage but do not fit or ratio-gate it.
        result.candidate=result.inliers>=30&&result.cells>=5&&result.hull>=.06;
        result.reason=result.candidate?"frozen_pose_correspondence_candidate":"frozen_pose_coverage_or_reprojection";
        result.Twc.release();
    }
    return result;
}
void writeResult(std::ostream& out,const Result& result,long mapId,unsigned long revision,double effective,
                 bool correspondenceMode) {
    out<<"{\"type\":\""<<(result.supportOnly?"support":"frame")<<"\",\"frame\":"<<result.frame<<",\"timestamp_s\":"<<result.time
       <<",\"map_id\":"<<mapId<<",\"map_revision\":"<<revision
       <<",\"gauge\":\"final_metric_atlas\",\"source\":\""
       <<(correspondenceMode?"offline-final-map-correspondence":result.supportOnly?
           "offline-prefix-temporal-support":"offline-prefix-relocalization")<<"\""
       <<",\"validated_after_final_atlas\":true,\"validation_effective_time_s\":"<<effective
       <<",\"features\":"<<result.features<<",\"matches\":"<<result.matches<<",\"inliers\":"<<result.inliers
       <<",\"rms_px\":"<<result.rms<<",\"p95_px\":"<<result.p95<<",\"hull_fraction\":"<<result.hull
       <<",\"occupied_cells\":"<<result.cells<<",\"inlier_fraction\":"<<result.ratio
       <<",\"support_only\":"<<(result.supportOnly?"true":"false")
       <<",\"guided_matching\":"<<(result.guided?"true":"false")
       <<",\"guide_before\":"<<result.guideBefore<<",\"guide_after\":"<<result.guideAfter
       <<",\"connected\":"<<(result.connected?"true":"false")
       <<",\"candidate\":"<<(result.candidate?"true":"false")<<",\"accepted\":"<<(result.accepted?"true":"false")
       <<",\"reason\":\""<<result.reason<<"\"";
    out<<",\"matched_feature_count\":"<<(result.accepted?result.matchedFeatures.size():0)
       <<",\"matched_features\":[";
    if(result.accepted) for(size_t i=0;i<result.matchedFeatures.size();++i) {
        if(i)out<<',';const auto& match=result.matchedFeatures[i];
        out<<'['<<match.pixel.x<<','<<match.pixel.y<<','<<match.pointId<<','<<match.error<<']';
    }
    out<<']';
    if(correspondenceMode) {out<<"}\n";return;}
    out<<",\"T_world_camera\":";
    if(!result.accepted) out<<"null";
    else {out<<'[';for(int r=0;r<4;++r){if(r)out<<',';out<<'[';for(int c=0;c<4;++c){if(c)out<<',';out<<result.Twc.at<double>(r,c);}out<<']';}out<<']';}
    out<<",\"pose\":";
    if(!result.accepted)out<<"null";
    else {
        Eigen::Matrix3d rotation;for(int r=0;r<3;++r)for(int c=0;c<3;++c)rotation(r,c)=result.Twc.at<double>(r,c);
        Eigen::Quaterniond q(rotation);q.normalize();
        out<<'['<<result.Twc.at<double>(0,3)<<','<<result.Twc.at<double>(1,3)<<','<<result.Twc.at<double>(2,3)
           <<','<<q.x()<<','<<q.y()<<','<<q.z()<<','<<q.w()<<']';
    }
    out<<"}\n";
}
}

int main(int argc,char** argv) {
    try {
        if(argc!=5) throw std::runtime_error("usage: relocalize_prefix_readonly VOC ATLAS MANIFEST_JSON OUTPUT_JSONL");
        cv::setNumThreads(1);cv::setRNGSeed(123);
        cv::FileStorage input(argv[3],cv::FileStorage::READ);
        if(!input.isOpened()) throw std::runtime_error("manifest_unreadable");
        const bool correspondenceMode=std::string(input["purpose"])=="replay-feature-correspondence";
        for(int i=1;i<=3;++i)rejectOutputAliasing(argv[4],argv[i]);
        rejectOutputAliasing(argv[4],std::string(input["video"]));
        if(!input["camera_model"].empty()&&std::string(input["camera_model"])!="pinhole")
            throw std::runtime_error("unsupported_camera_model");
        const cv::Mat K=matrix(input["camera_matrix"],3,3);
        cv::Mat D(1,int(input["dist_coeffs"].size()),CV_64F);
        for(int i=0;i<D.cols;++i) D.at<double>(i)=double(input["dist_coeffs"][i]);
        if(!D.empty()&&!cv::checkRange(D)) throw std::runtime_error("nonfinite_distortion");
        const cv::Size size{int(input["image_width"]),int(input["image_height"])};
        const long mapId=int(input["map_id"]);const int boundaryFrame=int(input["boundary_frame"]);
        const double boundaryTime=double(input["boundary_time_s"]);
        if(size.width<=0||size.height<=0||boundaryFrame<0||!std::isfinite(boundaryTime)||boundaryTime<0)
            throw std::runtime_error("invalid_prefix_boundary");
        std::vector<Query> queries;std::map<int,std::vector<std::vector<cv::Point>>> masks;
        for(auto node:input["queries"]) {
            Query query;query.frame=int(node["frame"]);query.time=double(node["timestamp_s"]);
            query.excluded=polygons(node["excluded_polygons"]);
            if(query.frame<0||query.frame>=boundaryFrame||!std::isfinite(query.time)||query.time<0||
               query.time>=boundaryTime||boundaryTime-query.time>5.)
                throw std::runtime_error("query_outside_initial_prefix_5s");
            if(!correspondenceMode&&!node["original_pose_valid"].empty()&&int(node["original_pose_valid"])!=0)
                throw std::runtime_error("refusing_existing_pose");
            if(correspondenceMode) query.frozenTwc=frozenPose(node["T_world_camera"]);
            queries.push_back(std::move(query));
        }
        if(queries.empty()) throw std::runtime_error("no_prefix_queries");
        if(correspondenceMode&&!input["support_queries"].empty())
            throw std::runtime_error("support_queries_only_for_prefix_localization");
        for(auto node:input["support_queries"]) {
            Query query;query.frame=int(node["frame"]);query.time=double(node["timestamp_s"]);
            query.supportOnly=true;query.excluded=polygons(node["excluded_polygons"]);
            // Existing poses are permitted only as non-publishing image support.
            // Each support pose is independently solved from real image/Atlas
            // correspondences; no interpolation or saved pose is used as proof.
            if(query.frame<boundaryFrame||!std::isfinite(query.time)||query.time<boundaryTime||
               query.time>boundaryTime+2.) throw std::runtime_error("support_outside_anchor_window_2s");
            queries.push_back(std::move(query));
        }
        std::sort(queries.begin(),queries.end(),[](const Query& a,const Query& b){return a.frame<b.frame;});
        for(size_t i=1;i<queries.size();++i)
            if(queries[i].frame==queries[i-1].frame||queries[i].time<=queries[i-1].time)
                throw std::runtime_error("nonmonotonic_prefix_queries");
        for(auto node:input["mask_frames"]) masks[int(node["frame"])]=polygons(node["excluded_polygons"]);
        std::ifstream source(argv[2],std::ios::binary);if(!source) throw std::runtime_error("atlas_unreadable");
        boost::archive::binary_iarchive archive(source);std::string vocabularyName,checksum,format;Atlas* atlas=nullptr;
        archive>>vocabularyName>>checksum>>format;
        if(format!="marker-orb-atlas/v1"&&format!="marker-orb-atlas/v2") throw std::runtime_error("atlas_format");
        archive>>atlas;ORBVocabulary vocabulary;
        if(!vocabulary.loadFromTextFile(argv[1])) throw std::runtime_error("vocabulary_unreadable");
        KeyFrameDatabase database(vocabulary);atlas->SetORBVocabulary(&vocabulary);atlas->SetKeyFrameDababase(&database);atlas->PostLoad();
        Map* map=nullptr;for(Map* candidate:atlas->GetAllMaps()) if(candidate->GetId()==unsigned(mapId)) map=candidate;
        if(!map||map->IsBad()||!map->mbMetric) throw std::runtime_error("requested_map_not_metric");
        if(input["map_revision"].empty()||double(input["map_revision"])!=double(map->mnRevision))
            throw std::runtime_error("final_map_revision_mismatch");
        auto keyframes=map->GetAllKeyFrames();std::sort(keyframes.begin(),keyframes.end(),KeyFrame::lId);
        std::set<KeyFrame*> local;KeyFrame* anchor=nullptr;
        for(KeyFrame* kf:keyframes) if(kf&&!kf->isBad()&&kf->GetMap()==map&&
            kf->mTimeStamp>=boundaryTime-1e-6&&kf->mTimeStamp<=boundaryTime+2.) {
            int count=0;for(MapPoint* p:kf->GetMapPointMatches()) if(p&&!p->isBad()&&p->Observations()>=2)++count;
            if(count>=30) {local.insert(kf);if(!anchor||kf->mTimeStamp<anchor->mTimeStamp)anchor=kf;}
        }
        if(!anchor) throw std::runtime_error("no_background_anchor_near_initialization");
        const auto seeds=local;
        for(KeyFrame* kf:seeds) for(KeyFrame* neighbor:kf->GetBestCovisibilityKeyFrames(10))
            if(neighbor&&!neighbor->isBad()&&neighbor->GetMap()==map) local.insert(neighbor);
        std::map<unsigned long,MapPoint*> pool;
        for(KeyFrame* kf:local) for(MapPoint* p:kf->GetMapPointMatches())
            if(p&&!p->isBad()&&p->GetMap()==map&&p->Observations()>=2) pool[p->mnId]=p;
        std::vector<cv::Point3d> points;std::vector<unsigned long> pointIds;cv::Mat descriptors;
        for(const auto& item:pool) {
            MapPoint* p=item.second;int observations=0;
            for(const auto& observation:p->GetObservations())
                if(observation.first&&!observation.first->isBad()&&observation.first->GetMap()==map) ++observations;
            const auto xyz=p->GetWorldPos();const cv::Mat descriptor=p->GetDescriptor();
            if(observations<2||!xyz.allFinite()||descriptor.rows!=1||descriptor.cols!=32||descriptor.type()!=CV_8U)continue;
            points.emplace_back(xyz.x(),xyz.y(),xyz.z());pointIds.push_back(item.first);descriptors.push_back(descriptor);
        }
        if(points.size()<30) throw std::runtime_error("insufficient_static_map_points");
        const std::string video=input["video"],ffmpeg=input["ffmpeg"].empty()?"ffmpeg":std::string(input["ffmpeg"]),
                          ffprobe=input["ffprobe"].empty()?"ffprobe":std::string(input["ffprobe"]);
        const auto probe=readCommand(shellQuote(ffprobe)+" -v error -select_streams v:0 -show_entries stream=width,height:format=duration -of json "+shellQuote(video));
        cv::FileStorage metadata(probe,cv::FileStorage::READ|cv::FileStorage::MEMORY|cv::FileStorage::FORMAT_JSON);
        const auto stream=metadata["streams"][0];
        if(int(stream["width"])!=size.width||int(stream["height"])!=size.height)
            throw std::runtime_error("calibration_image_mismatch");
        const double duration=std::stod(std::string(metadata["format"]["duration"]));
        const double effective=input["validation_effective_time_s"].empty()?duration:double(input["validation_effective_time_s"]);
        if(!std::isfinite(effective)||effective<duration-1e-6) throw std::runtime_error("validation_precedes_final_video");
        Query control;control.frame=int(anchor->mnFrameId);control.time=anchor->mTimeStamp;
        if(correspondenceMode&&masks.find(control.frame)==masks.end())
            throw std::runtime_error("anchor_exclusion_mask_missing");
        control.excluded=masks[control.frame];
        if(control.frame<boundaryFrame)throw std::runtime_error("anchor_frame_precedes_boundary");
        // The actual background keyframe is selected from the loaded Atlas.
        // Later support frames are unnecessary; control is evaluated separately.
        queries.erase(std::remove_if(queries.begin(),queries.end(),[&](const Query& query){
            return query.supportOnly&&query.frame>=control.frame;
        }),queries.end());
        GrayVideoReader reader(video,ffmpeg,size,control.frame);
        ORBextractor extractor(2000,1.2,8,20,7);std::vector<Result> results;
        for(const auto& query:queries) results.push_back(localize(query,reader,extractor,points,descriptors,pointIds,K,D,size));
        Result check=localize(control,reader,extractor,points,descriptors,pointIds,K,D,size);
        cv::Mat anchorPose=cv::Mat::eye(4,4,CV_64F);const auto pose=anchor->GetPoseInverse().matrix();
        for(int r=0;r<4;++r)for(int c=0;c<4;++c)anchorPose.at<double>(r,c)=pose(r,c);
        const double anchorTranslation=check.candidate?translationDistance(check.Twc,anchorPose):-1;
        const double anchorRotation=check.candidate?rotationDistance(check.Twc,anchorPose):-1;
        const bool anchorValid=check.candidate&&anchorTranslation<.05&&anchorRotation<5.;
        Result last=check;int connected=0;
        if(anchorValid&&!correspondenceMode) for(auto it=results.rbegin();it!=results.rend();++it) {
            if(!it->candidate)continue;
            const double dt=last.time-it->time;
            if(dt>0&&dt<=.25&&translationDistance(it->Twc,last.Twc)<=.02+5.*dt&&
               rotationDistance(it->Twc,last.Twc)<=2.+360.*dt) {
                it->connected=true;if(!it->supportOnly)++connected;last=*it;
            }
        }
        const bool guidedDisplay=!input["guided_prefix_matching"].empty() && int(input["guided_prefix_matching"])!=0;
        if(anchorValid&&!correspondenceMode&&guidedDisplay) {
            // No cascading rescue: both bracketing guides must be independently
            // connected in the original pass. Keep failed trials transactional.
            const auto original=results;
            for(size_t i=0;i<original.size();++i) {
                if(original[i].candidate||original[i].supportOnly||original[i].independentTwc.empty()
                    ||original[i].inliers<30||original[i].cells<5||original[i].hull<.06)continue;
                const Result *before=nullptr,*after=nullptr;
                for(size_t j=0;j<i;++j)if(original[j].candidate&&original[j].connected)before=&original[j];
                for(size_t j=i+1;j<original.size();++j)if(original[j].candidate&&original[j].connected){after=&original[j];break;}
                if(!after)after=&check;
                if(!before||!after->candidate||original[i].time-before->time>.25||after->time-original[i].time>.25)continue;
                const double span=after->time-before->time;
                if(span<=0)continue;
                const double alpha=(original[i].time-before->time)/span;
                cv::Mat prior=cv::Mat::eye(4,4,CV_64F),r;
                cv::Mat R0=before->Twc(cv::Rect(0,0,3,3)),R1=after->Twc(cv::Rect(0,0,3,3));
                cv::Rodrigues(R0.t()*R1,r);r*=alpha;cv::Mat delta;cv::Rodrigues(r,delta);
                cv::Mat rotation=R0*delta,position=(1.-alpha)*before->Twc(cv::Rect(3,0,1,3))+alpha*after->Twc(cv::Rect(3,0,1,3));
                rotation.copyTo(prior(cv::Rect(0,0,3,3)));position.copyTo(prior(cv::Rect(3,0,1,3)));
                Result trial=localize(queries[i],reader,extractor,points,descriptors,pointIds,K,D,size,&original[i],prior);
                std::cerr<<"GUIDED_PREFIX frame="<<trial.frame<<" inliers="<<trial.inliers<<" ratio="<<trial.ratio<<" candidate="<<trial.candidate<<" reason="<<trial.reason<<'\n';
                if(trial.candidate && translationDistance(trial.Twc,original[i].independentTwc)<=.02
                    && rotationDistance(trial.Twc,original[i].independentTwc)<=2.) {
                    trial.guided=true;trial.guideBefore=before->frame;trial.guideAfter=after->frame;
                    results[i]=std::move(trial);
                }
            }
            last=check;connected=0;
            for(auto it=results.rbegin();it!=results.rend();++it) {
                it->connected=false;if(!it->candidate)continue;
                const double dt=last.time-it->time;
                if(dt>0&&dt<=.25&&translationDistance(it->Twc,last.Twc)<=.02+5.*dt&&rotationDistance(it->Twc,last.Twc)<=2.+360.*dt) {
                    it->connected=true;if(!it->supportOnly)++connected;last=*it;
                }
            }
        }
        for(auto& result:results) {
            if(correspondenceMode) {
                result.accepted=result.candidate&&anchorValid;
                if(result.candidate)result.reason=anchorValid?"validated_frozen_pose_correspondence":"anchor_control_failed";
            } else {
                result.accepted=!result.supportOnly&&result.candidate&&result.connected&&connected>=2;
                if(result.supportOnly) {
                    if(result.candidate)result.reason=result.connected?"validated_temporal_support":"unconnected_temporal_support";
                    continue;
                }
                if(result.candidate)result.reason=result.accepted?"validated_final_map_prefix":
                    (!anchorValid?"anchor_control_failed":"temporal_anchor_support");
            }
        }
        std::ofstream out(argv[4]);if(!out)throw std::runtime_error("output_unwritable");out<<std::setprecision(12);
        out<<"{\"type\":\"metadata\",\"schema\":\""
           <<(correspondenceMode?"readonly-final-map-correspondence/v1":"readonly-prefix-localization/v1")<<"\",\"map_id\":"<<mapId
           <<",\"map_revision\":"<<map->mnRevision<<",\"map_points\":"<<points.size()<<",\"local_keyframes\":"<<local.size()
           <<",\"anchor_keyframe\":"<<anchor->mnId<<",\"anchor_frame\":"<<control.frame
           <<",\"anchor_valid\":"<<(anchorValid?"true":"false")<<",\"anchor_translation_difference_m\":"<<anchorTranslation
           <<",\"anchor_rotation_difference_deg\":"<<anchorRotation<<",\"validation_effective_time_s\":"<<effective
           <<",\"atlas_modified\":false}\n";
        for(const auto& result:results)writeResult(out,result,mapId,map->mnRevision,effective,correspondenceMode);
        return 0;
    } catch(const std::exception& error) {std::cerr<<"PREFIX_LOCALIZATION_ERROR "<<error.what()<<std::endl;return 2;}
}
