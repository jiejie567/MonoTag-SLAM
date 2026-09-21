// Reuse the frozen read-only ORB extraction / matching / PnP implementation.
// No tracking workers, map writes, interpolated labels, or odometry input.
#define main original_prefix_main
#include <opencv2/video/tracking.hpp>
#include <chrono>
#include "shared_localizer.cc"
#undef main

Result flowLocalize(const Query& q,const Result& cached,const Result& before,const Result& after,
                    const std::map<unsigned long,MapPoint*>& pool,const cv::Mat& K,const cv::Mat& D,cv::Size size) {
    Result r=cached;r.candidate=false;r.reason="bidirectional_flow_support";
    struct Track {cv::Point2f pixel;int ends;bool bad=false;};std::map<unsigned long,Track> tracks;
    cv::Mat mask(size,CV_8U,cv::Scalar(255));if(!q.excluded.empty())cv::fillPoly(mask,q.excluded,cv::Scalar(0));
    cv::erode(mask,mask,cv::Mat::ones(17,17,CV_8U));
    for(int end=0;end<2;end++) {
        const Result& a=end==0?before:after;std::vector<cv::Point2f> p0,p1,p2;std::vector<unsigned long> ids;
        for(auto f:a.matchedFeatures) {p0.emplace_back(f.pixel);ids.push_back(f.pointId);}
        if(p0.empty())continue;
        std::vector<unsigned char> s1,s2;std::vector<float> e1,e2;
        cv::calcOpticalFlowPyrLK(a.grayImage,cached.grayImage,p0,p1,s1,e1,cv::Size(21,21),3);
        cv::calcOpticalFlowPyrLK(cached.grayImage,a.grayImage,p1,p2,s2,e2,cv::Size(21,21),3);
        for(size_t j=0;j<p0.size();j++) {
            if(!s1[j]||!s2[j]||cv::norm(p0[j]-p2[j])>1.5)continue;
            int u=cvRound(p1[j].x),v=cvRound(p1[j].y);
            if(u<0||v<0||u>=size.width||v>=size.height||!mask.at<unsigned char>(v,u))continue;
            auto it=tracks.find(ids[j]);
            if(it==tracks.end())tracks[ids[j]]={p1[j],1<<end,false};
            else {if(cv::norm(it->second.pixel-p1[j])>2.)it->second.bad=true;it->second.ends|=1<<end;}
        }
    }
    std::vector<cv::Point3d> world;std::vector<cv::Point2d> pixels;std::vector<unsigned long> ids;std::vector<int> ends;
    for(auto item:tracks)if(!item.second.bad&&pool.count(item.first)) {
        bool duplicate=false;for(auto p:pixels)if(cv::norm(p-cv::Point2d(item.second.pixel))<2.){duplicate=true;break;}
        if(duplicate)continue;auto x=pool.at(item.first)->GetWorldPos();
        world.emplace_back(x.x(),x.y(),x.z());pixels.emplace_back(item.second.pixel);ids.push_back(item.first);ends.push_back(item.second.ends);
    }
    r.matches=int(world.size());if(world.size()<30)return r;
    cv::Mat rv,tv,fit;
    if(!cv::solvePnPRansac(world,pixels,K,D,rv,tv,false,200,3.,.999,fit,cv::SOLVEPNP_EPNP)||fit.rows<30)return r;
    std::vector<cv::Point3d> fw;std::vector<cv::Point2d> fp;
    for(int j=0;j<fit.rows;j++){int k=fit.at<int>(j);fw.push_back(world[k]);fp.push_back(pixels[k]);}
    cv::solvePnPRefineLM(fw,fp,K,D,rv,tv);cv::Mat R;cv::Rodrigues(rv,R);
    std::vector<cv::Point2d> projected;cv::projectPoints(world,rv,tv,K,D,projected);
    std::vector<cv::Point2f> support;std::set<int> cells;std::vector<double> errors;int a=0,b=0,both=0;
    r.matchedFeatures.clear();
    for(size_t j=0;j<world.size();j++) {
        double z=R.at<double>(2,0)*world[j].x+R.at<double>(2,1)*world[j].y+R.at<double>(2,2)*world[j].z+tv.at<double>(2);
        double error=cv::norm(projected[j]-pixels[j]);if(!(z>0&&std::isfinite(error)&&error<3.))continue;
        errors.push_back(error);support.emplace_back(pixels[j]);cells.insert(int(pixels[j].y*3/size.height)*4+int(pixels[j].x*4/size.width));
        a+=(ends[j]&1)!=0;b+=(ends[j]&2)!=0;both+=ends[j]==3;r.matchedFeatures.push_back({pixels[j],ids[j],error});
    }
    r.inliers=int(errors.size());r.ratio=double(r.inliers)/world.size();r.cells=int(cells.size());
    if(errors.size()<30||a<10||b<10||both<10)return r;
    std::vector<cv::Point2f> hull;cv::convexHull(support,hull);r.hull=cv::contourArea(hull)/size.area();
    double sum=0;for(double e:errors)sum+=e*e;r.rms=std::sqrt(sum/errors.size());std::sort(errors.begin(),errors.end());r.p95=errors[size_t(.95*(errors.size()-1))];
    r.candidate=r.ratio>=.6&&r.cells>=5&&r.hull>=.06&&r.rms<=3.;
    if(r.candidate) {r.Twc=cv::Mat::eye(4,4,CV_64F);cv::Mat Rt=R.t(),t=-Rt*tv;Rt.copyTo(r.Twc(cv::Rect(0,0,3,3)));t.copyTo(r.Twc(cv::Rect(3,0,1,3)));}
    r.reason=r.candidate?"bidirectional_flow_pnp":"bidirectional_flow_geometry";
    std::cerr<<"GAP_FLOW frame="<<q.frame<<" pairs="<<world.size()<<" inliers="<<r.inliers<<" both="<<both<<" rms="<<r.rms<<" accepted="<<r.candidate<<std::endl;
    return r;
}

static std::vector<int> batchFrames;
static std::unique_ptr<GrayVideoReader> batchReader;
static std::string batchVideo;
int processRequest(int argc,char** argv) {
    try {
        auto tick=[](){return std::chrono::steady_clock::now();};
        auto seconds=[&](std::chrono::steady_clock::time_point t){return std::chrono::duration<double>(tick()-t).count();};
        double atlasSeconds=0,vocabularySeconds=0,postloadSeconds=0;
        if(argc!=5)throw std::runtime_error("usage: gap VOC ATLAS REQUEST OUTPUT");
        cv::setNumThreads(1);cv::setRNGSeed(123);
        cv::FileStorage input(argv[3],cv::FileStorage::READ);
        if(!input.isOpened())throw std::runtime_error("request_unreadable");
        for(int i=1;i<=3;i++)rejectOutputAliasing(argv[4],argv[i]);
        const std::string video=input["video"];
        rejectOutputAliasing(argv[4],video);
        const cv::Mat K=matrix(input["camera_matrix"],3,3);
        cv::Mat D(1,int(input["dist_coeffs"].size()),CV_64F);
        for(int i=0;i<D.cols;i++)D.at<double>(i)=double(input["dist_coeffs"][i]);
        if(!D.empty()&&!cv::checkRange(D))throw std::runtime_error("invalid_distortion");
        const cv::Size size{int(input["image_width"]),int(input["image_height"])};
        long mapId=int(input["map_id"]);double revision=double(input["map_revision"]);
        std::vector<Query> queries;std::vector<cv::Mat> controls;
        for(auto node:input["queries"]) {
            Query q;q.frame=int(node["frame"]);q.time=double(node["timestamp_s"]);
            q.supportOnly=int(node["support_only"])!=0;
            q.excluded=polygons(node["excluded_polygons"]);
            if(q.frame<0||!std::isfinite(q.time)||q.time<0)throw std::runtime_error("invalid_frame");
            if(!queries.empty()&&(q.frame!=queries.back().frame+1||q.time<=queries.back().time))
                throw std::runtime_error("noncontiguous_gap");
            if(q.supportOnly)controls.push_back(frozenPose(node["T_world_camera"]));
            else if(int(node["original_pose_valid"])!=0)throw std::runtime_error("refusing_existing_pose");
            queries.push_back(q);
        }
        if(queries.size()<3||controls.size()!=2||!queries.front().supportOnly||!queries.back().supportOnly
           ||queries.back().time-queries.front().time>.5+1e-6)throw std::runtime_error("gap_not_bracketed_or_too_long");
        for(size_t i=1;i+1<queries.size();i++)if(queries[i].supportOnly)throw std::runtime_error("interior_control");
        static Atlas* atlas=nullptr;
        static ORBVocabulary vocabulary;
        static std::unique_ptr<KeyFrameDatabase> database;
        if(!atlas) {
            auto stage=tick();
            std::ifstream source(argv[2],std::ios::binary);
            boost::archive::binary_iarchive archive(source);std::string name,checksum,format;Atlas* loaded=nullptr;
            archive>>name>>checksum>>format;
            if(format!="marker-orb-atlas/v1"&&format!="marker-orb-atlas/v2")throw std::runtime_error("atlas_format");
            archive>>loaded;
            atlasSeconds=seconds(stage);stage=tick();
            if(!vocabulary.loadFromTextFile(argv[1]))throw std::runtime_error("vocabulary");
            vocabularySeconds=seconds(stage);stage=tick();
            database.reset(new KeyFrameDatabase(vocabulary));loaded->SetORBVocabulary(&vocabulary);
            loaded->SetKeyFrameDababase(database.get());loaded->PostLoad();atlas=loaded;
            postloadSeconds=seconds(stage);
        }
        Map* map=nullptr;for(auto m:atlas->GetAllMaps())if(m->GetId()==unsigned(mapId))map=m;
        if(!map||map->IsBad()||!map->mbMetric||double(map->mnRevision)!=revision)throw std::runtime_error("map_or_revision_mismatch");
        std::set<KeyFrame*> local;
        for(auto k:map->GetAllKeyFrames())if(k&&!k->isBad()&&k->mTimeStamp>=queries.front().time-1.
            &&k->mTimeStamp<=queries.back().time+1.)local.insert(k);
        const auto seeds=local;
        for(auto k:seeds)for(auto n:k->GetBestCovisibilityKeyFrames(10))
            if(n&&!n->isBad()&&n->GetMap()==map)local.insert(n);
        std::map<unsigned long,MapPoint*> pool;
        for(auto k:local)for(auto p:k->GetMapPointMatches())if(p&&!p->isBad()&&p->GetMap()==map)pool[p->mnId]=p;
        std::vector<cv::Point3d> points;std::vector<unsigned long> ids;cv::Mat descriptors;
        for(auto item:pool) {
            auto p=item.second;int observations=0;
            for(auto o:p->GetObservations())if(o.first&&!o.first->isBad()&&o.first->GetMap()==map)observations++;
            auto xyz=p->GetWorldPos();cv::Mat d=p->GetDescriptor();
            if(observations<2||!xyz.allFinite()||d.rows!=1||d.cols!=32||d.type()!=CV_8U)continue;
            points.emplace_back(xyz.x(),xyz.y(),xyz.z());ids.push_back(item.first);descriptors.push_back(d);
        }
        if(points.size()<30)throw std::runtime_error("insufficient_map_points");
        const std::string ffmpeg=input["ffmpeg"].empty()?"ffmpeg":std::string(input["ffmpeg"]);
        const auto probe=readCommand("ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of json "+shellQuote(video));
        cv::FileStorage meta(probe,cv::FileStorage::READ|cv::FileStorage::MEMORY|cv::FileStorage::FORMAT_JSON);
        if(int(meta["streams"][0]["width"])!=size.width||int(meta["streams"][0]["height"])!=size.height)
            throw std::runtime_error("calibration_image_mismatch");
        if(batchReader && batchVideo!=video)throw std::runtime_error("batch_video_mismatch");
        if(!batchReader) {
            batchVideo=video;
            batchReader.reset(new GrayVideoReader(video,ffmpeg,size,queries.back().frame,batchFrames));
        }
        auto& reader=*batchReader;
        const double effective=double(input["validation_effective_time_s"]);
        if(!std::isfinite(effective)||effective<queries.back().time)throw std::runtime_error("invalid_effective_time");
        const int featureBudget=input["orb_features"].empty()?2000:int(input["orb_features"]);
        if(featureBudget<2000||featureBudget>4000)throw std::runtime_error("invalid_feature_budget");
        auto localizationStart=tick();double previousDecode=reader.decodingSeconds;
        ORBextractor extractor(featureBudget,1.2,8,20,7);std::vector<Result> results;
        for(auto q:queries)results.push_back(localize(q,reader,extractor,points,descriptors,ids,K,D,size));
        // Saved endpoint poses only restrict descriptor search. PnP remains
        // unseeded and must independently recover both control images.
        for(int j=0;j<2;j++) {
            const size_t i=j==0?0:results.size()-1;
            if(!results[i].candidate)results[i]=localize(queries[i],reader,extractor,points,descriptors,ids,K,D,size,&results[i],controls[j]);
        }
        bool validControls=true;
        for(int j=0;j<2;j++) {
            auto& r=j==0?results.front():results.back();
            validControls=validControls&&r.candidate&&r.rms<=3.&&
                translationDistance(r.Twc,controls[j])<=.05&&rotationDistance(r.Twc,controls[j])<=5.;
        }
        if(validControls)for(size_t i=1;i+1<results.size();i++)if(!results[i].candidate) {
            const double alpha=(queries[i].time-queries.front().time)/(queries.back().time-queries.front().time);
            cv::Mat prior=cv::Mat::eye(4,4,CV_64F),r,delta;
            cv::Mat R0=results.front().Twc(cv::Rect(0,0,3,3)),R1=results.back().Twc(cv::Rect(0,0,3,3));
            cv::Rodrigues(R0.t()*R1,r);r*=alpha;cv::Rodrigues(r,delta);
            cv::Mat R=R0*delta,t=(1.-alpha)*results.front().Twc(cv::Rect(3,0,1,3))+alpha*results.back().Twc(cv::Rect(3,0,1,3));
            R.copyTo(prior(cv::Rect(0,0,3,3)));t.copyTo(prior(cv::Rect(3,0,1,3)));
            auto trial=localize(queries[i],reader,extractor,points,descriptors,ids,K,D,size,&results[i],prior);
            trial.guided=true;trial.guideBefore=queries.front().frame;trial.guideAfter=queries.back().frame;
            results[i]=std::move(trial);
        }
        if(validControls && (input["disable_flow"].empty()||int(input["disable_flow"])==0))
            for(size_t i=1;i+1<results.size();i++)if(!results[i].candidate)
                results[i]=flowLocalize(queries[i],results[i],results.front(),results.back(),pool,K,D,size);
        // Each query is independently solved; neither successful neighbours
        // nor an interpolated prior is a substitute for image evidence.
        bool chain=validControls;
        for(size_t i=1;i<results.size();i++) {
            const auto& a=results[i-1];const auto& b=results[i];const double dt=b.time-a.time;
            chain=chain&&a.candidate&&b.candidate&&b.rms<=3.&&
                translationDistance(a.Twc,b.Twc)<=.02+5.*dt&&rotationDistance(a.Twc,b.Twc)<=2.+360.*dt;
        }
        std::cerr<<"GAP_TIMING atlas_s="<<atlasSeconds<<" vocabulary_s="<<vocabularySeconds
                 <<" postload_s="<<postloadSeconds<<" decode_s="<<reader.decodingSeconds-previousDecode
                 <<" localization_without_decode_s="<<seconds(localizationStart)-(reader.decodingSeconds-previousDecode)<<std::endl;
        std::ofstream out(argv[4]);if(!out)throw std::runtime_error("output_unwritable");out<<std::setprecision(12);
        out<<"{\"type\":\"metadata\",\"schema\":\"readonly-short-gap/v1\",\"map_id\":"<<mapId
            <<",\"map_revision\":"<<revision<<",\"controls_valid\":"<<(validControls?"true":"false")
            <<",\"chain_valid\":"<<(chain?"true":"false")<<",\"map_points\":"<<points.size()<<",\"atlas_modified\":false}\n";
        for(auto& r:results) {
            r.connected=chain;r.accepted=!r.supportOnly&&chain;
            if(r.candidate)r.reason=!validControls?"gap_control_failed":!chain?"gap_chain_failed":r.supportOnly?"gap_control_valid":"validated_final_map_short_gap";
            std::ostringstream row;row<<std::setprecision(12);writeResult(row,r,mapId,map->mnRevision,double(input["validation_effective_time_s"]),false);
            std::string s=row.str();const std::string old="offline-prefix-relocalization";
            const auto pos=s.find(old);if(pos!=std::string::npos)s.replace(pos,old.size(),"offline-short-gap-relocalization");
            out<<s;
        }
        return 0;
    } catch(const std::exception& e) {std::cerr<<"GAP_LOCALIZATION_ERROR "<<e.what()<<std::endl;return 2;}
}

int main(int argc,char** argv) {
    if(argc==5)return processRequest(argc,argv);
    if(argc<6||std::string(argv[3])!="--batch"||(argc-4)%2) return 2;
    // One Atlas, vocabulary and ordered decode stream for all disjoint gaps.
    // Each gap retains independent controls and atomic acceptance.
    try {
        int previous=-1;
        for(int i=4;i<argc;i+=2) {
            for(int j=1;j<argc;j++)if(j!=i+1&&j!=3)rejectOutputAliasing(argv[i+1],argv[j]);
            cv::FileStorage request(argv[i],cv::FileStorage::READ);
            const std::string video=request["video"];rejectOutputAliasing(argv[i+1],video);
            if(batchVideo.empty())batchVideo=video;
            if(video!=batchVideo)throw std::runtime_error("batch_video_mismatch");
            int first=int(request["queries"][0]["frame"]);
            if(first<previous)throw std::runtime_error("unordered_batch");
            for(auto q:request["queries"])batchFrames.push_back(int(q["frame"]));
            previous=batchFrames.back();
        }
        std::sort(batchFrames.begin(),batchFrames.end());batchFrames.erase(std::unique(batchFrames.begin(),batchFrames.end()),batchFrames.end());
        for(int i=4;i<argc;i+=2) {
            char* args[]={argv[0],argv[1],argv[2],argv[i],argv[i+1]};
            if(processRequest(5,args)) {
                std::ofstream error(argv[i+1]);error<<"{\"type\":\"error\",\"reason\":\"gap_adapter_failed\"}\n";
            }
        }
        return 0;
    } catch(const std::exception& e) {std::cerr<<"GAP_BATCH_ERROR "<<e.what()<<std::endl;return 2;}
}
