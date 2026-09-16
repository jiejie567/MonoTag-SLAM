// Production marker graph solver regression. No camera, vocabulary or SLAM run.
#include "MarkerGraphOptimizer.h"
#include "MarkerGraphCoordinator.h"
#include "LoopClosing.h"
#include "Thirdparty/DBoW2/DUtils/Random.h"
#include "PostLoopRematch.h"
#include "Sim3Solver.h"
#include "Optimizer.h"
#include "LocalMetricScaleGuard.h"
#include "MarkerComponentRegistration.h"
#include "BackgroundResidualGate.h"
#include "Atlas.h"
#include "KeyFrameDatabase.h"
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <fstream>
#include "Frame.h"
#include "KeyFrame.h"
#include "Map.h"
#include "MapPoint.h"
#include "ORBextractor.h"
#include "ORBmatcher.h"
#include "CameraModels/Pinhole.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>

using namespace ORB_SLAM3;

static void require(bool condition, const std::string& message)
{
    if(!condition) throw std::runtime_error(message);
}

static void dumpReanchorProposal(Map* map, KeyFrame* a, const std::vector<KeyFrame*>& b,
                                const MarkerGraphOptimizer::Proposal& proposal, bool visualLoop=false,
                                bool finalGlobal=false)
{
    const char* path=std::getenv("MARKER_REANCHOR_DUMP");
    if(!path) return;
    require(!std::ifstream(path).good(),"refusing to overwrite diagnostic proposal dump");
    std::ofstream out(path);
    require(out.good(),"cannot open diagnostic proposal dump");
    out << std::setprecision(12);
    // Reconstruct the production variable-selection rule solely for labeling
    // the dump. No map state or observations are changed by this diagnostic.
    std::set<KeyFrame*> selected;
    for(KeyFrame* endpoint:b) {
        std::vector<KeyFrame*> chain;
        std::set<KeyFrame*> ancestors;
        for(KeyFrame* k=a;k && ancestors.insert(k).second;k=k->GetParent()) chain.push_back(k);
        std::set<KeyFrame*> seen;
        KeyFrame* common=endpoint;
        while(common && !ancestors.count(common) && seen.insert(common).second) {
            selected.insert(common);common=common->GetParent();
        }
        require(common && ancestors.count(common),"diagnostic tree path disconnected");
        for(KeyFrame* k:chain) {selected.insert(k);if(k==common) break;}
    }
    unsigned long last=a->mnFrameId;
    for(KeyFrame* k:b) last=std::max(last,k->mnFrameId);
    for(KeyFrame* k:map->GetAllKeyFrames())
        if(k && !k->isBad() && k->mnFrameId>=a->mnFrameId && k->mnFrameId<=last) selected.insert(k);
    const auto pathKeyframes=selected;
    for(KeyFrame* k:pathKeyframes)
        for(KeyFrame* n:k->GetBestCovisibilityKeyFrames(MarkerGraphOptimizer::Options().covisibleNeighbors))
            if(n && !n->isBad() && n->GetMap()==map) selected.insert(n);
    const auto vector=[&](const Eigen::Vector3f& v) {
        out << '[' << v.x() << ',' << v.y() << ',' << v.z() << ']';
    };
    const auto pose=[&](const Sophus::SE3f& value) {
        const auto m=value.matrix3x4();out << '[';
        for(int i=0;i<12;++i) out << (i?",":"") << m(i/4,i%4);
        out << ']';
    };
    out << "{\"type\":\"header\",\"accepted\":" << (proposal.accepted?"true":"false")
        << ",\"reason\":\"" << proposal.reason << "\",\"anchor\":" << a->mnId
        << ",\"scope\":\"Frozen final Atlas, raw observations retained; not exact historical rewind.\""
        << ",\"fixed_selection\":\"" << (finalGlobal?"global_ba_origin_only":
            (visualLoop?"visual_loop_origin_only":"reanchor_path_and_boundary")) << "\"}\n";
    std::size_t fixed=0;double maximumTime=0;
    auto keyframes=map->GetAllKeyFrames();std::sort(keyframes.begin(),keyframes.end(),KeyFrame::lId);
    for(KeyFrame* k:keyframes) {
        if(!k || k->isBad()) continue;
        const auto after=proposal.keyframePoses.find(k);
        const bool inProposal=after!=proposal.keyframePoses.end();
        const bool isFixed=inProposal && (k==a || (!visualLoop && !selected.count(k)));
        fixed+=isFixed;
        if(inProposal) maximumTime=std::max(maximumTime,k->mTimeStamp);
        out << "{\"type\":\"keyframe\",\"id\":" << k->mnId << ",\"frame\":" << k->mnFrameId
            << ",\"time\":" << k->mTimeStamp << ",\"in_proposal\":" << (inProposal?"true":"false")
            << ",\"fixed_reconstructed\":" << (isFixed?"true":"false") << ",\"Tcw_before\":";
        pose(k->GetPose());out << ",\"Tcw_after\":";
        if(inProposal) pose(after->second);else out << "null";
        out << ",\"camera_parameters\":[";
        for(std::size_t i=0;i<k->mpCamera->size();++i) out << (i?",":"") << k->mpCamera->getParameter(i);
        out << "]}\n";
        for(std::size_t i=0;i<k->mvTagIds.size();++i) {
            out << "{\"type\":\"tag_observation\",\"kf\":" << k->mnId << ",\"id\":" << k->mvTagIds[i]
                << ",\"index\":" << i << ",\"active\":" << (k->mbTagObservationActive?"true":"false")
                << ",\"confidence\":" << k->mTagObservationConfidence
                << ",\"weight\":" << (k->mvTagPointWeights.empty()?1.f:k->mvTagPointWeights[i])
                << ",\"pixel\":[" << k->mvTagImagePoints[i].x << ',' << k->mvTagImagePoints[i].y
                << "],\"world_before\":";vector(k->mvTagWorldPoints[i]);
            const auto staged=proposal.tagWorldCorners.find(k);
            out << ",\"world_after\":";
            if(staged!=proposal.tagWorldCorners.end() && i<staged->second.size()) vector(staged->second[i]);
            else out << "null";
            out << "}\n";
        }
    }
    std::vector<MapPoint*> points;
    for(const auto& item:proposal.pointPositions) points.push_back(item.first);
    std::sort(points.begin(),points.end(),[](MapPoint* x,MapPoint* y){return x->mnId<y->mnId;});
    std::size_t observationCount=0;
    for(MapPoint* point:points) {
        KeyFrame* reference=point->GetReferenceKeyFrame();
        out << "{\"type\":\"point\",\"id\":" << point->mnId << ",\"reference\":"
            << (reference?long(reference->mnId):-1) << ",\"first_keyframe\":" << point->mnFirstKFid << ",\"before\":";vector(point->GetWorldPos());
        out << ",\"after\":";vector(proposal.pointPositions.at(point));out << "}\n";
        for(const auto& observation:point->GetObservations()) {
            KeyFrame* k=observation.first;const int index=std::get<0>(observation.second);
            if(!k || k->isBad() || index<0 || std::size_t(index)>=k->mvKeysUn.size()) continue;
            const auto& key=k->mvKeysUn[index];
            if(key.octave<0 || std::size_t(key.octave)>=k->mvInvLevelSigma2.size()) continue;
            out << "{\"type\":\"background_observation\",\"point\":" << point->mnId
                << ",\"kf\":" << k->mnId << ",\"feature_index\":" << index << ",\"octave\":" << key.octave
                << ",\"reciprocal_id\":" << (k->GetMapPoint(index)?long(k->GetMapPoint(index)->mnId):-1)
                << ",\"pixel\":[" << key.pt.x << ',' << key.pt.y << "],\"information\":"
                << k->mvInvLevelSigma2[key.octave] << "}\n";
            ++observationCount;
        }
    }
    out.close();require(out.good(),"diagnostic proposal dump failed");
    std::cout << "ATLAS_REANCHOR_DUMP path=" << path << " keyframes=" << proposal.keyframePoses.size()
              << " fixed_reconstructed=" << fixed << " maximum_time=" << maximumTime
              << " points=" << points.size() << " raw_observations=" << observationCount << std::endl;
}

// No copy of native matching logic: run its exact single-candidate path once,
// then serialize the returned seed and feature-index/MapPoint-ID matches.
class FrozenLoopDetector : public LoopClosing {
public:
    FrozenLoopDetector(Atlas* atlas, KeyFrameDatabase* database, ORBVocabulary* vocabulary)
        : LoopClosing(atlas,database,vocabulary,false,true) { mpTracker=nullptr; }
    bool detect(KeyFrame* query, KeyFrame* candidate, KeyFrame*& matched,
                g2o::Sim3& seed, std::vector<MapPoint*>& matches) {
        require(query && candidate && query->GetMap()==candidate->GetMap() &&
                !query->GetMap()->IsInertial(),"frozen detector requires one non-inertial map");
        for(KeyFrame* k:query->GetMap()->GetAllKeyFrames())
            require(k->NLeft==-1 && !k->mpCamera2,"frozen detector requires monocular cameras");
        mpCurrentKF=query;mbOfflineLoopSearch=true;
        std::vector<KeyFrame*> candidates{candidate};
        std::vector<MapPoint*> points;KeyFrame* last=nullptr;int neighbors=0;
        return DetectCommonRegionsFromBoW(candidates,matched,last,seed,neighbors,points,matches);
    }
};

static void frozenAtlasLoop(Map* map, Atlas* atlas, KeyFrameDatabase* database,
                            ORBVocabulary* vocabulary, int argc, char** argv)
{
    require(argc==7 && map->mbMetric && !map->IsInertial(),"invalid frozen-loop input");
    using MGO=MarkerGraphOptimizer;
    std::map<unsigned long,KeyFrame*> keyframes;
    std::map<unsigned long,MapPoint*> points;
    MGO::Proposal original;
    MGO::ScaleMap originalUnits;
    std::map<KeyFrame*,std::vector<MapPoint*>> originalMatches;
    std::map<MapPoint*,std::map<KeyFrame*,std::tuple<int,int>>> originalObservations;
    for(KeyFrame* k:map->GetAllKeyFrames()) if(k && !k->isBad()) {
        keyframes[k->mnId]=k;original.keyframePoses[k]=k->GetPose();
        original.tagWorldCorners[k]=k->mvTagWorldPoints;originalUnits[k]=k->mReplayUnitScale;
        originalMatches[k]=k->GetMapPointMatches();
    }
    for(MapPoint* p:map->GetAllMapPoints()) if(p && !p->isBad()) {
        points[p->mnId]=p;original.pointPositions[p]=p->GetWorldPos();
        originalObservations[p]=p->GetObservations();
    }
    const auto revision=map->mnRevision;
    const auto registry=map->mStaticTags;
    const auto anchor=map->GetMarkerScaleAnchorKFId();
    const auto unchanged=[&]() {
        require(revision==map->mnRevision && registry==map->mStaticTags &&
                anchor==map->GetMarkerScaleAnchorKFId(),"frozen loop changed map metadata");
        for(const auto& item:original.keyframePoses) {
            require((item.first->GetPose().matrix()-item.second.matrix()).norm()==0 &&
                    item.first->mReplayUnitScale==originalUnits.at(item.first),"frozen loop changed live camera/units");
            require(item.first->GetMapPointMatches()==originalMatches.at(item.first),"frozen loop changed feature slots");
            const auto& corners=original.tagWorldCorners.at(item.first);
            require(corners.size()==item.first->mvTagWorldPoints.size(),"frozen loop changed raw tag count");
            for(std::size_t i=0;i<corners.size();++i)
                require((corners[i]-item.first->mvTagWorldPoints[i]).norm()==0,"frozen loop changed raw tag geometry");
        }
        for(const auto& item:original.pointPositions) {
            require((item.first->GetWorldPos()-item.second).norm()==0,"frozen loop changed live point");
            require(item.first->GetObservations()==originalObservations.at(item.first),"frozen loop changed raw observation links");
        }
    };
    if(std::string(argv[1])=="--atlas-loop-freeze") {
        require(!std::ifstream(argv[6]).good(),"refusing to overwrite frozen candidate");
        KeyFrame* query=keyframes.at(std::stoul(argv[4]));
        KeyFrame* candidate=keyframes.at(std::stoul(argv[5]));
        require(map->GetOriginKF(),"missing frozen loop origin");
        original.reason="unmodified_atlas_baseline";
        dumpReanchorProposal(map,map->GetOriginKF(),{query},original,true);
        FrozenLoopDetector detector(atlas,database,vocabulary);
        KeyFrame* matched=nullptr;g2o::Sim3 seed;std::vector<MapPoint*> matches;
        DUtils::Random::SeedRand(0);
        require(detector.detect(query,candidate,matched,seed,matches),"native single candidate failed geometric validation");
        unchanged();
        std::ofstream output(argv[6]);require(output.good(),"cannot open frozen candidate");
        output << std::setprecision(17) << "marker-loop-seed/v1\n"
               << query->mnId << ' ' << matched->mnId << ' ' << map->GetId() << ' ' << revision
               << ' ' << keyframes.size() << ' ' << points.size() << '\n';
        const auto q=seed.rotation();const auto t=seed.translation();
        output << q.x() << ' ' << q.y() << ' ' << q.z() << ' ' << q.w() << ' '
               << t.x() << ' ' << t.y() << ' ' << t.z() << ' ' << seed.scale() << '\n';
        const auto source=query->GetMapPointMatches();
        output << matches.size() << '\n';
        for(std::size_t i=0;i<matches.size();++i)
            output << i << ' ' << (i<source.size() && source[i]?long(source[i]->mnId):-1)
                   << ' ' << (matches[i]?long(matches[i]->mnId):-1) << '\n';
        output.close();require(output.good(),"failed writing frozen candidate");
        std::cout << "FROZEN_LOOP_SAVED query=" << query->mnId << " requested_candidate=" << candidate->mnId
                  << " matched=" << matched->mnId << " seed_scale=" << seed.scale()
                  << " slots=" << matches.size() << " live_geometry_unchanged=1" << std::endl;
        return;
    }
    std::ifstream input(argv[4]);require(input.good(),"missing frozen candidate");
    std::string format;input>>format;require(format=="marker-loop-seed/v1","invalid frozen candidate format");
    unsigned long queryId,matchedId,mapId,storedRevision;std::size_t kCount,pCount;
    input>>queryId>>matchedId>>mapId>>storedRevision>>kCount>>pCount;
    require(mapId==map->GetId() && storedRevision==revision && kCount==keyframes.size() && pCount==points.size(),
            "frozen candidate Atlas metadata mismatch");
    double x,y,z,w,tx,ty,tz,scale;input>>x>>y>>z>>w>>tx>>ty>>tz>>scale;
    require(input.good() && std::isfinite(scale) && scale>0,"invalid frozen candidate Sim3");
    const g2o::Sim3 seed(Eigen::Quaterniond(w,x,y,z),Eigen::Vector3d(tx,ty,tz),scale);
    KeyFrame* query=keyframes.at(queryId);KeyFrame* matched=keyframes.at(matchedId);
    const auto source=query->GetMapPointMatches();std::size_t n;input>>n;
    require(n==source.size(),"frozen candidate feature count mismatch");
    std::vector<MapPoint*> matches(n,nullptr);
    for(std::size_t i=0;i<n;++i) {
        std::size_t index;long from,to;input>>index>>from>>to;
        require(input.good() && index==i && from==(source[i]?long(source[i]->mnId):-1),
                "frozen candidate source slot changed");
        if(to>=0) matches[i]=points.at(to);
    }
    std::istringstream budgets(argv[5]);std::string token;
    while(std::getline(budgets,token,',')) {
        std::size_t consumed=0;const int budget=std::stoi(token,&consumed);
        require(consumed==token.size() && budget>0 && budget<=240,"invalid loop BA diagnostic budget");
        const std::string trace=std::string(argv[6])+".ba"+token+".jsonl";
        require(!std::ifstream(trace).good(),"refusing to overwrite loop stage trace");
        setenv("MARKER_LOOP_STAGE_DUMP",trace.c_str(),1);
        MGO::Options options;options.baIterations=budget;
        const char* essential=std::getenv("MARKER_LOOP_ESSENTIAL_COVISIBILITY");
        if(essential) options.useEssentialGraphCovisibility=std::string(essential)=="1";
        const char* retryAliases=std::getenv("MARKER_LOOP_RETRY_MASKED_ALIASES");
        if(retryAliases) options.retryLoopAliasesAfterTagFailure=std::string(retryAliases)=="1";
        const auto proposal=MGO::ProposeVisualLoop(map,query,matched,seed,matches,options);
        unsetenv("MARKER_LOOP_STAGE_DUMP");unchanged();
        dumpReanchorProposal(map,map->GetOriginKF(),{query},proposal,true);
        double gauge=0;
        const auto pose=proposal.keyframePoses.find(map->GetOriginKF());
        if(pose!=proposal.keyframePoses.end())
            gauge=(pose->second.matrix()-map->GetOriginKF()->GetPose().matrix()).norm();
        std::cout << std::setprecision(12) << "FROZEN_LOOP_RESULT query=" << queryId << " matched=" << matchedId
                  << " ba_budget=" << budget << " accepted=" << proposal.accepted << " reason=" << proposal.reason
                  << " seed_scale=" << seed.scale() << " tag_before=" << proposal.before.tagRmsPx
                  << " tag_after=" << proposal.after.tagRmsPx << " background_before=" << proposal.before.backgroundRmsPx
                  << " background_after=" << proposal.after.backgroundRmsPx << " tag_count=" << proposal.before.tagCorners
                  << " background_count=" << proposal.before.backgroundObservations << " gauge_matrix_delta=" << gauge
                  << " live_geometry_unchanged=1 trace=" << trace << std::endl;
    }
}

static void componentOriginConsistencyCase()
{
    const Sophus::SE3f reference(Sophus::SO3f(),Eigen::Vector3f(12.f,26.f,.2f));
    const Sophus::SE3f candidate(Sophus::SO3f::exp(Eigen::Vector3f(0,0,.015f)),
                               reference.translation()+Eigen::Vector3f(.005f,0,0));
    require((candidate*reference.inverse()).translation().norm()>.03f,
            "fixture must expose the old world-origin lever arm");
    require(MarkerComponentTransformsConsistent(reference,candidate),
            "small actual marker-origin motion rejected far from world zero");
    const Sophus::SE3f gauges[] = {
        Sophus::SE3f(),
        Sophus::SE3f(Sophus::SO3f::exp(Eigen::Vector3f(.2f,-.3f,.5f)),Eigen::Vector3f(-120.f,42.f,3.f)),
        Sophus::SE3f(Sophus::SO3f(),Eigen::Vector3f(200.f,-300.f,5.f))
    };
    const Sophus::SE3f moved(candidate.so3(),reference.translation()+Eigen::Vector3f(.04f,0,0));
    const Sophus::SE3f rotated(Sophus::SO3f::exp(Eigen::Vector3f(0,0,.20f)),reference.translation());
    for(const auto& gauge:gauges) {
        require(MarkerComponentTransformsConsistent(gauge*reference,gauge*candidate),
                "component admission depends on the Atlas world gauge");
        require(!MarkerComponentTransformsConsistent(gauge*reference,gauge*moved),
                "real translation beyond 3 cm silently accepted");
        require(!MarkerComponentTransformsConsistent(gauge*reference,gauge*rotated),
                "real rotation beyond 10 degrees silently accepted");
    }
    Sophus::SE3f invalid=candidate;
    invalid.translation().x()=std::numeric_limits<float>::quiet_NaN();
    require(!MarkerComponentTransformsConsistent(reference,invalid),"nonfinite candidate accepted");
    std::cout << "{\"component_origin_gauge_invariant\":true,\"translation_rotation_limits_retained\":true}" << std::endl;
}

static Frame makeBase(ORBextractor& extractor, Pinhole& camera)
{
    cv::Mat image(480, 640, CV_8UC1), distortion = cv::Mat::zeros(4, 1, CV_32F);
    cv::RNG rng(914);
    rng.fill(image, cv::RNG::UNIFORM, 0, 256);
    Frame frame(image, 0, &extractor, nullptr, &camera, distortion, 0, 1);
    frame.SetPose(Sophus::SE3f());
    require(frame.N >= 80, "synthetic frame does not have enough feature slots");
    return frame;
}

static void triangulationUniqueTargetCase()
{
    Pinhole camera(std::vector<float>{600,600,320,240});
    ORBextractor extractor(250,1.2f,8,20,7);
    Frame base=makeBase(extractor,camera);
    for(int mode:{0,1,2}) {
        const bool distinctTargets=mode==0;
        const bool laterExactMatch=mode==2;
        Map map(KeyFrame::nNextId);
        Frame first(base),second(base);
        first.mnId=Frame::nNextId++;
        second.mnId=Frame::nNextId++;
        first.SetPose(Sophus::SE3f());
        second.SetPose(Sophus::SE3f(Sophus::SO3f(),Eigen::Vector3f(-.1f,0,-.02f)));
        // Populate mutable Frames before constructing immutable KF raw data.
        // No vocabulary is needed: only this shared feature node is searched.
        first.mFeatVec.clear();second.mFeatVec.clear();
        first.mFeatVec[17]={0,1};
        second.mFeatVec[17]=distinctTargets ? std::vector<unsigned int>{0,1}
                                           : std::vector<unsigned int>{0};
        first.mDescriptors.row(0).setTo(cv::Scalar(0));
        first.mDescriptors.row(1).setTo(cv::Scalar(distinctTargets?255:0));
        second.mDescriptors.row(0).setTo(cv::Scalar(0));
        second.mDescriptors.row(1).setTo(cv::Scalar(255));
        // A valid but worse candidate must not lock out the later exact one.
        if(laterExactMatch) first.mDescriptors.at<unsigned char>(0,0)=255;
        const Eigen::Vector3f a(0,0,2),b(.2f,0,2);
        const Eigen::Vector2f sourceA=camera.project(a),sourceB=camera.project(b);
        const Eigen::Vector2f targetA=camera.project(second.GetPose()*a);
        const Eigen::Vector2f targetB=camera.project(second.GetPose()*b);
        first.mvKeysUn[0]=cv::KeyPoint(cv::Point2f(sourceA.x(),sourceA.y()),1,0,0,0);
        first.mvKeysUn[1]=cv::KeyPoint(cv::Point2f(distinctTargets?sourceB.x():sourceA.x()+2,
                                                sourceA.y()),1,0,0,0);
        second.mvKeysUn[0]=cv::KeyPoint(cv::Point2f(targetA.x(),targetA.y()),1,0,0,0);
        second.mvKeysUn[1]=cv::KeyPoint(cv::Point2f(targetB.x(),targetB.y()),1,0,0,0);
        KeyFrame source(first,&map,nullptr),target(second,&map,nullptr);
        const Sophus::SE3f relative=source.GetPose()*target.GetPoseInverse();
        require(camera.epipolarConstrain(&camera,source.mvKeysUn[0],target.mvKeysUn[0],
                    relative.rotationMatrix(),relative.translation(),1,1) &&
                camera.epipolarConstrain(&camera,source.mvKeysUn[1],
                    target.mvKeysUn[distinctTargets?1:0],relative.rotationMatrix(),relative.translation(),1,1),
                "unique-target fixture did not satisfy the production epipolar gate");
        if(laterExactMatch)
            require(ORBmatcher::DescriptorDistance(source.mDescriptors.row(0),target.mDescriptors.row(0))==8 &&
                    ORBmatcher::DescriptorDistance(source.mDescriptors.row(1),target.mDescriptors.row(0))==0,
                    "later-exact fixture did not produce the intended 8-bit versus 0-bit conflict");
        ORBmatcher matcher(.6,true);
        std::vector<std::pair<size_t,size_t>> pairs;
        const int count=matcher.SearchForTriangulation(&source,&target,pairs,false,false);
        std::set<size_t> sourceSlots,targetSlots;
        for(const auto& pair:pairs) {sourceSlots.insert(pair.first);targetSlots.insert(pair.second);}
        std::cout << "{\"triangulation_unique_target\":true,\"distinct_targets\":" << distinctTargets
                  << ",\"later_exact_match\":" << laterExactMatch
                  << ",\"returned_pairs\":" << pairs.size() << ",\"unique_target_slots\":"
                  << targetSlots.size() << ",\"first_source\":"
                  << (pairs.empty()?-1:long(pairs.front().first)) << "}" << std::endl;
        require(count==int(pairs.size()),"triangulation count differs from returned pairs");
        require(sourceSlots.size()==pairs.size() && targetSlots.size()==pairs.size(),
                "triangulation returned multiple MapPoint candidates for one target feature slot");
        require(count==(distinctTargets?2:1),
                "unique-target triangulation removed an independent valid match");
        if(!distinctTargets)
            require(pairs.front()==std::make_pair(size_t(laterExactMatch?1:0),size_t(0)),
                    laterExactMatch ? "an earlier weak match blocked the later exact target match"
                                    : "equal-distance target conflicts did not stably retain the first source");
        require(source.GetMapPoints().empty() && target.GetMapPoints().empty(),
                "triangulation matcher mutated live feature associations");
    }
}

struct Fixture {
    static constexpr int views = 12;
    Map map;
    Pinhole camera;
    ORBextractor extractor;
    Frame base;
    std::vector<std::unique_ptr<KeyFrame>> ownedKeyframes;
    std::vector<std::unique_ptr<MapPoint>> ownedPoints;
    std::vector<KeyFrame*> keyframes;
    std::vector<Sophus::SE3f> truth;
    std::vector<Eigen::Vector3f> truePoints;
    std::set<MapPoint*> singlyObservedPoints;
    MarkerGraphOptimizer::PoseMap originalPoses;
    MarkerGraphOptimizer::PointMap originalPoints;
    std::map<int, std::vector<float>> originalMarkers;
    std::vector<MapPoint*> loopMatches;

    explicit Fixture(double drift, bool corruptSparseBackground = false,
                     float backgroundDepthScale = 1.0f, int markerOnlyView = -1,
                     bool foreignSingletonReference = false, bool loopPairs = false,
                     bool reverseAllocation = false, bool wideLoopPairs = false,
                     int finalPixelOutliers = 0)
        : map(KeyFrame::nNextId), camera(std::vector<float>{600, 600, 320, 240}),
          extractor(250, 1.2f, 8, 20, 7), base(makeBase(extractor, camera))
    {
        map.mbMetric = true;
        map.mbBackgroundReady = true;
        map.mMetricScale = 1;
        for(int id = 24; id <= 25; ++id) {
            const float center = .20f + .14f*(id-24);
            for(const auto& corner : std::vector<Eigen::Vector2f>{{-.035f,-.035f},{.035f,-.035f},
                                                               {.035f,.035f},{-.035f,.035f}}) {
                auto& values = map.mStaticTags[id];
                values.push_back(center+corner.x()); values.push_back(corner.y()); values.push_back(1.2f);
            }
        }
        originalMarkers = map.mStaticTags;
        std::vector<Frame> frames;
        std::vector<Eigen::Vector3f> oldCenters;
        std::vector<int> nextFeature(views, 0);
        for(int view = 0; view < views; ++view) {
            Frame frame(base);
            frame.mnId = Frame::nNextId++;
            frame.mTimeStamp = .1*view;
            const float u = float(view)/(views-1);
            const Eigen::Vector3f center(.04f*view, 0, 0);
            Eigen::Vector3f oldCenter = center*(1.0+0.5*drift*u);
            if(finalPixelOutliers && view==6) oldCenter.x()+=.06f;
            // Integrating a gradually increasing visual length bias gives a
            // spatially varying scale drift, not a globally rescaled map.
            truth.emplace_back(Eigen::Matrix3f::Identity(), -center);
            oldCenters.push_back(oldCenter);
            frame.SetPose(Sophus::SE3f(Eigen::Matrix3f::Identity(), -oldCenter));
            frames.push_back(frame);
        }
        struct PointSpec {
            Eigen::Vector3f truth, initial;
            int reference;
            std::vector<std::pair<int,int>> observations;
        };
        std::vector<PointSpec> specs;
        for(int first = 0; first+2 < views; ++first) {
            const int reference = first+1 == markerOnlyView ? first : first+1;
            const Eigen::Vector3f referenceCenter = truth[reference].inverse().translation();
            const float localScale = 1.0 + drift*reference/(views-1);
            for(int index = 0; index < 20; ++index) {
                const bool finalBad=finalPixelOutliers && first==5 && index<finalPixelOutliers;
                const Eigen::Vector3f point(referenceCenter.x()+(index%5-2)*.04f,
                                            (index/5-1.5f)*.045f,
                                            (finalPixelOutliers && !finalBad ? 15.f : 1.3f+.08f*(index%3))*backgroundDepthScale);
                PointSpec spec{point, oldCenters[reference]+localScale*(point-referenceCenter), reference, {}};
                if(finalPixelOutliers) spec.initial=point;
                for(int view = first; view <= first+2; ++view) {
                    if(view == markerOnlyView) continue;
                    const int feature = nextFeature[view]++;
                    Eigen::Vector2f pixel = camera.project(truth[view]*point);
                    if(corruptSparseBackground && first == 5 && index == 0 && view == 6) pixel.y() += 40;
                    if(finalBad && view==6) pixel.x()-=600.f*.06f/point.z();
                    frames[view].mvKeysUn[feature] = cv::KeyPoint(cv::Point2f(pixel.x(), pixel.y()), 1.0f);
                    spec.observations.emplace_back(view, feature);
                }
                specs.push_back(spec);
            }
        }
        for(int view = 0; view < views; ++view) {
            const Eigen::Vector3f center = truth[view].inverse().translation();
            const Eigen::Vector3f point = center+Eigen::Vector3f(.02f, .02f, 1.5f);
            const float scale = 1.0+drift*view/(views-1);
            const int feature = nextFeature[view]++;
            const auto pixel = camera.project(truth[view]*point);
            frames[view].mvKeysUn[feature] = cv::KeyPoint(cv::Point2f(pixel.x(), pixel.y()), 1.0f);
            specs.push_back({point, oldCenters[view]+scale*(point-center),
                foreignSingletonReference && view==views-1 ? 0 : view, {{view, feature}}});
        }
        std::vector<std::pair<int,int>> pairSpecs;
        if(loopPairs)for(int i=0;i<25;++i) {
            pairSpecs.emplace_back(specs.size(),specs.size()+1);
            Eigen::Vector3f point(.15f+.025f*(i%5),.025f*(i/5-2),1.4f+.02f*(i%3));
            if(wideLoopPairs) {
                const Eigen::Vector2f pixel(60+125*(i%5),45+90*(i/5));
                const Eigen::Vector3f camera((pixel.x()-320)/600*1.6f,(pixel.y()-240)/600*1.6f,1.6f);
                point=truth.back().inverse()*camera;
            }
            for(int view:{0,views-1}) {
                const int feature=nextFeature[view]++;
                const auto pixel=camera.project(truth[view]*point);
                frames[view].mvKeysUn[feature]=cv::KeyPoint(pixel.x(),pixel.y(),1.f);
                const Eigen::Vector3f center=truth[view].inverse().translation();
                specs.push_back({point,oldCenters[view]+float(1+drift*view/(views-1))*(point-center),view,{{view,feature}}});
            }
        }
        // Same logical IDs/observations, different allocation order. Assign
        // fixture IDs before registering any graph membership or connection.
        const auto firstKeyframeId=KeyFrame::nNextId;
        ownedKeyframes.resize(views); keyframes.resize(views);
        for(int allocation=0;allocation<views;++allocation) {
            const int view=reverseAllocation?views-1-allocation:allocation;
            ownedKeyframes[view].reset(new KeyFrame(frames[view], &map, nullptr));
            keyframes[view]=ownedKeyframes[view].get();
            keyframes[view]->mnId=firstKeyframeId+view;
        }
        for(int view = 0; view < views; ++view) {
            KeyFrame* keyframe = keyframes[view];
            map.AddKeyFrame(keyframe);
            if(view > 0) keyframe->ChangeParent(keyframes[view-1]);
            if(view == 0 || view >= views-3 || view == markerOnlyView) {
                keyframe->mbHasTagObservation = true;
                keyframe->mbTagObservationActive = true;
                keyframe->mTagObservationConfidence = 1;
                for(const auto& marker : map.mStaticTags) {
                    for(int corner = 0; corner < 4; ++corner) {
                        const Eigen::Vector3f point(marker.second[corner*3], marker.second[corner*3+1], marker.second[corner*3+2]);
                        keyframe->mvTagIds.push_back(marker.first);
                        keyframe->mvTagWorldPoints.push_back(point);
                        const auto pixel = camera.project(truth[view]*point);
                        keyframe->mvTagImagePoints.emplace_back(pixel.x(), pixel.y());
                        keyframe->mvTagPointWeights.push_back(1);
                    }
                }
            }
            originalPoses.emplace(keyframe, keyframe->GetPose());
        }
        const auto firstPointId=MapPoint::nNextId;
        ownedPoints.resize(specs.size());
        for(std::size_t allocation=0;allocation<specs.size();++allocation) {
            const auto index=reverseAllocation?specs.size()-1-allocation:allocation;
            const auto& spec=specs[index];
            ownedPoints[index].reset(new MapPoint(spec.initial,keyframes[spec.reference],&map));
            ownedPoints[index]->mnId=firstPointId+index;
        }
        for(std::size_t index=0;index<specs.size();++index) {
            const auto& spec=specs[index];
            MapPoint* point = ownedPoints[index].get();
            map.AddMapPoint(point);
            truePoints.push_back(spec.truth);
            originalPoints.emplace(point, spec.initial);
            if(spec.observations.size() == 1) singlyObservedPoints.insert(point);
            for(const auto& observation : spec.observations) {
                point->AddObservation(keyframes[observation.first], observation.second);
                keyframes[observation.first]->AddMapPoint(point, observation.second);
            }
        }
        loopMatches.resize(base.N,nullptr);
        for(const auto& pair:pairSpecs)
            loopMatches[specs[pair.second].observations.front().second]=ownedPoints[pair.first].get();
        for(KeyFrame* keyframe : keyframes) keyframe->UpdateConnections();
        // UpdateConnections may choose a stronger covisible parent. Retain a
        // genuine connected spanning chain; the graph also uses covisibility.
        for(int view = 1; view < views; ++view) keyframes[view]->ChangeParent(keyframes[view-1]);
    }

    std::vector<KeyFrame*> window() const
    {
        return std::vector<KeyFrame*>(keyframes.end()-3, keyframes.end());
    }

    double measuredScale() const
    {
        std::vector<double> ratios;
        for(int first = views-3; first < views; ++first) for(int second = first+1; second < views; ++second) {
            const double metric = (truth[first].inverse().translation()-truth[second].inverse().translation()).norm();
            const double visual = (keyframes[first]->GetCameraCenter()-keyframes[second]->GetCameraCenter()).norm();
            ratios.push_back(metric/visual);
        }
        std::sort(ratios.begin(), ratios.end());
        return ratios[ratios.size()/2];
    }

    double cameraRms(const MarkerGraphOptimizer::PoseMap& poses) const
    {
        double squared = 0;
        for(int view = 0; view < views; ++view)
            squared += (poses.at(keyframes[view]).inverse().translation()-truth[view].inverse().translation()).squaredNorm();
        return std::sqrt(squared/views);
    }

    void assertUnchanged() const
    {
        require(map.mStaticTags == originalMarkers, "solver wrote the fixed marker layout");
        for(const auto& pose : originalPoses) {
            require((pose.first->GetPose().matrix()-pose.second.matrix()).norm() == 0, "solver wrote a live keyframe pose");
            require(pose.first->mReplayUnitScale == 1, "solver wrote a replay scale");
            require(pose.first->mnBALocalForKF == 0, "solver wrote a live BA stamp");
        }
        for(const auto& point : originalPoints)
            require((point.first->GetWorldPos()-point.second).norm() == 0, "solver wrote a live map point");
    }
};

static void backgroundCheiralityRepairCase()
{
    for(int mode=0;mode<3;++mode) {
        const bool contradictory=mode==2;
        const bool lowParallax=mode==1;
        Fixture f(0,contradictory,lowParallax?100.f:1.f);
        MarkerGraphOptimizer::StagedBAInput raw;
        raw.keyframes=f.keyframes;
        raw.keyframePoses=f.originalPoses;
        raw.pointPositions=f.originalPoints;
        raw.fixedKeyframes.insert(f.keyframes.begin(),f.keyframes.end());
        MarkerGraphOptimizer::Options legacy;
        legacy.baIterations=1;
        legacy.repairBackgroundCheirality=false;
        const auto baseline=MarkerGraphOptimizer::RefineAndValidate(raw,legacy);
        require(baseline.before.backgroundObservations>0,"cheirality fixture has no raw observations");
        MapPoint* point=f.ownedPoints[contradictory?100:0].get();
        const auto observations=point->GetObservations();
        require(observations.size()==3,"cheirality fixture requires independent three-view pixels");
        MarkerGraphOptimizer::StagedBAInput staged=raw;
        const auto center=point->GetReferenceKeyFrame()->GetCameraCenter();
        staged.pointPositions.at(point)=center-1000.f*(point->GetWorldPos()-center);
        const auto old=MarkerGraphOptimizer::RefineAndValidate(staged,legacy,&baseline.before);
        require(old.pointPositions.count(point),"legacy cheirality solve lost point: "+old.reason);
        bool oldNegative=false;
        for(const auto& observation:observations)
            oldNegative|=(old.keyframePoses.at(observation.first)*old.pointPositions.at(point)).z()<=0;
        require(oldNegative,"cheirality fixture did not expose the negative-depth basin");
        MarkerGraphOptimizer::Options repairedOptions=legacy;
        repairedOptions.repairBackgroundCheirality=true;
        const auto repaired=MarkerGraphOptimizer::RefineAndValidate(staged,repairedOptions,&baseline.before);
        require(repaired.pointPositions.count(point),"cheirality repair lost point: "+repaired.reason);
        require(repaired.after.backgroundObservations==baseline.before.backgroundObservations,
                "cheirality repair removed raw background observations");
        require(point->GetObservations()==observations,"cheirality repair changed live observation indices");
        for(KeyFrame* k:f.keyframes)
            require((repaired.keyframePoses.at(k).matrix()-f.originalPoses.at(k).matrix()).norm()==0,
                    "cheirality repair moved a fixed camera");
        if(mode==0) {
            require(!old.accepted,"negative-depth legacy proposal unexpectedly passed the healthy baseline");
            require(repaired.accepted,"observable cheirality repair rejected: "+repaired.reason);
            require(repaired.after.backgroundRmsPx<.01,"cheirality repair did not restore raw pixels");
            for(const auto& observation:observations)
                require((repaired.keyframePoses.at(observation.first)*repaired.pointPositions.at(point)).z()>0,
                        "repaired point stayed behind an observing camera");
        } else if(lowParallax) {
            require((repaired.pointPositions.at(point)-old.pointPositions.at(point)).norm()==0,
                    "low-parallax point was assigned an unsupported repaired depth");
        } else {
            require(!repaired.accepted,"contradictory raw pixels bypassed the unchanged residual gates");
        }
        f.assertUnchanged();
        std::cout << "{\"background_cheirality_repair\":true,\"mode\":" << mode
                  << ",\"legacy_accepted\":" << old.accepted
                  << ",\"repaired_accepted\":" << repaired.accepted
                  << ",\"background_before\":" << baseline.before.backgroundRmsPx
                  << ",\"background_after\":" << repaired.after.backgroundRmsPx
                  << ",\"raw_observations_retained\":" << repaired.after.backgroundObservations << "}" << std::endl;
    }
}

static std::pair<double, std::size_t> depthScaleEvidence(const Fixture& fixture,
        const MarkerGraphOptimizer::Proposal& proposal, KeyFrame* keyframe)
{
    std::vector<double> ratios;
    for(const auto& original : fixture.originalPoints) {
        const auto observations = original.first->GetObservations();
        if(observations.size() < 2 || !observations.count(keyframe) ||
           !proposal.pointPositions.count(original.first)) continue;
        const Eigen::Vector3f before = fixture.originalPoses.at(keyframe)*original.second;
        const Eigen::Vector3f after = proposal.keyframePoses.at(keyframe)*proposal.pointPositions.at(original.first);
        if(before.z() > 1e-6f && after.z() > 1e-6f) ratios.push_back(after.z()/before.z());
    }
    std::sort(ratios.begin(), ratios.end());
    const double median = ratios.empty() ? -1.0 :
        .5*(ratios[(ratios.size()-1)/2]+ratios[ratios.size()/2]);
    return {median, ratios.size()};
}

static void printScaleEvidence(const Fixture& fixture,
                               const MarkerGraphOptimizer::Proposal& proposal)
{
    std::cout << "{\"selected_scale_evidence\":[";
    bool first = true;
    for(KeyFrame* keyframe : fixture.keyframes) {
        if(!proposal.keyframePoses.count(keyframe) || !proposal.replayScaleMultipliers.count(keyframe)) continue;
        const auto evidence = depthScaleEvidence(fixture, proposal, keyframe);
        if(!first) std::cout << ",";
        first = false;
        std::cout << "{\"keyframe\":" << keyframe->mnId
                  << ",\"replay_multiplier\":" << proposal.replayScaleMultipliers.at(keyframe)
                  << ",\"multiview_depth_ratio\":" << evidence.first
                  << ",\"multiview_points\":" << evidence.second << "}";
    }
    std::cout << "]}" << std::endl;
}

static void assertSingletonScaleContract(const Fixture& fixture,
                                        const MarkerGraphOptimizer::Proposal& proposal)
{
    for(MapPoint* point : fixture.singlyObservedPoints) {
        KeyFrame* reference = point->GetReferenceKeyFrame();
        const Eigen::Vector3f oldRay = fixture.originalPoses.at(reference)*fixture.originalPoints.at(point);
        const Eigen::Vector3f correctedRay = proposal.keyframePoses.at(reference)*proposal.pointPositions.at(point);
        const double rayError = (correctedRay-oldRay*proposal.replayScaleMultipliers.at(reference)).norm();
        if(rayError >= 1e-5)
            std::cerr << "single-view mismatch point=" << point->mnId << " reference=" << reference->mnId
                      << " observations=" << point->GetObservations().size()
                      << " multiplier=" << proposal.replayScaleMultipliers.at(reference)
                      << " old_ray=" << oldRay.transpose() << " corrected_ray=" << correctedRay.transpose()
                      << " error=" << rayError << std::endl;
        require(rayError < 1e-5,
                "single-view point did not follow its corrected BA reference");
    }
}

static void correctionCase(double drift)
{
    Fixture fixture(drift);
    const double ratio = fixture.measuredScale();
    const auto proposal = MarkerGraphOptimizer::Reanchor(&fixture.map, fixture.keyframes.front(), fixture.window(), ratio, .01);
    fixture.assertUnchanged();
    require(proposal.accepted, "reanchor rejected: "+proposal.reason);
    const double before = fixture.cameraRms(fixture.originalPoses);
    const double after = fixture.cameraRms(proposal.keyframePoses);
    require(after < .3*before, "path camera error did not improve by at least 70%");
    require((proposal.keyframePoses.at(fixture.keyframes.front()).matrix()-fixture.originalPoses.at(fixture.keyframes.front()).matrix()).norm() < 1e-6,
            "world anchor A moved");
    require(std::abs(proposal.replayScaleMultipliers.at(fixture.keyframes.front())-1.0) < 1e-6,
            "world anchor A was scaled");
    for(KeyFrame* keyframe : fixture.window())
        require(std::abs(proposal.replayScaleMultipliers.at(keyframe)/ratio-1.0) < .03,
                "B local scale anchor was not respected");
    for(KeyFrame* keyframe : fixture.keyframes) {
        if(keyframe == fixture.keyframes.front()) continue;
        const auto evidence = depthScaleEvidence(fixture, proposal, keyframe);
        require(evidence.second >= 3 &&
                std::abs(proposal.replayScaleMultipliers.at(keyframe)/evidence.first-1.0) < 1e-5,
                "path replay multiplier does not describe the final background geometry");
    }
    assertSingletonScaleContract(fixture, proposal);
    require(proposal.after.backgroundObservations >= 500, "joint BA omitted real background observations");
    require(proposal.after.tagRmsPx < 2.5 && proposal.after.backgroundRmsPx < 3,
            "accepted candidate violates reprojection gates");
    std::cout << "{\"drift\":" << drift << ",\"metric_per_visual\":" << ratio
              << ",\"before_camera_rms_m\":" << before << ",\"after_camera_rms_m\":" << after
              << ",\"improvement\":" << 1-after/before
              << ",\"tag_rms_px\":" << proposal.after.tagRmsPx
              << ",\"background_rms_px\":" << proposal.after.backgroundRmsPx << "}" << std::endl;
    printScaleEvidence(fixture, proposal);
}

static void initialMetricizationCase()
{
    // One stale 40 px ORB observation is intentionally retained in the live
    // map; initial metricization must use the healthy native inlier subgraph.
    Fixture fixture(0, true);
    constexpr float trueMetricPerVisual = .25f;
    constexpr float closedFormInitialScale = .27f;
    fixture.map.mbMetric = false;
    fixture.map.mMetricScale = 0;

    MarkerGraphOptimizer::PoseMap visualPoses;
    MarkerGraphOptimizer::PointMap visualPoints;
    for(int view = 0; view < Fixture::views; ++view) {
        Sophus::SE3f visualTwc = fixture.truth[view].inverse();
        visualTwc.translation() /= trueMetricPerVisual;
        fixture.keyframes[view]->SetPose(visualTwc.inverse());
        fixture.keyframes[view]->mbTagObservationActive = false;
        visualPoses.emplace(fixture.keyframes[view], fixture.keyframes[view]->GetPose());
    }
    for(std::size_t index = 0; index < fixture.ownedPoints.size(); ++index) {
        MapPoint* point = fixture.ownedPoints[index].get();
        point->SetWorldPos(fixture.truePoints[index]/trueMetricPerVisual);
        visualPoints.emplace(point, point->GetWorldPos());
    }
    // One decoded group contains a damaged corner. Initial metricization may
    // reject that whole keyframe, but may not weaken the global 2.5 px gate or
    // discard a majority of the marker evidence.
    fixture.keyframes[10]->mvTagImagePoints[0].x += 20.0f;

    const auto proposal = MarkerGraphOptimizer::InitializeMetric(
        &fixture.map, Sophus::SE3f(), closedFormInitialScale);
    require(proposal.accepted, "initial metric joint BA rejected: "+proposal.reason);
    double beforeSquared = 0, afterSquared = 0;
    for(int view = 0; view < Fixture::views; ++view) {
        const Eigen::Vector3f truth = fixture.truth[view].inverse().translation();
        const Eigen::Vector3f visual = visualPoses.at(fixture.keyframes[view]).inverse().translation();
        const Eigen::Vector3f initial = visual*closedFormInitialScale;
        const Eigen::Vector3f refined = proposal.keyframePoses.at(fixture.keyframes[view]).inverse().translation();
        beforeSquared += (initial-truth).squaredNorm();
        afterSquared += (refined-truth).squaredNorm();
    }
    const double before = std::sqrt(beforeSquared/Fixture::views);
    const double after = std::sqrt(afterSquared/Fixture::views);
    require(after < .3*before, "initial marker factors did not refine ORB map scale");
    require(proposal.after.tagRmsPx < 2.5 && proposal.after.backgroundRmsPx < 3,
            "initial metric candidate violates joint reprojection gates");
    require(!proposal.after.tagRmsByKeyframe.count(fixture.keyframes[10]),
            "damaged initial marker keyframe survived group rejection");
    require(!fixture.keyframes.back()->mbTagObservationActive,
            "initial metric proposal activated a live tag observation");
    require(std::abs(proposal.replayScaleMultipliers.at(fixture.keyframes.back())/
                     trueMetricPerVisual-1.0f) < .01f,
            "replay retained the superseded closed-form initial scale");
    for(const auto& pose : visualPoses)
        require((pose.first->GetPose().matrix()-pose.second.matrix()).norm() == 0,
                "initial metric proposal wrote a live keyframe pose");
    for(const auto& point : visualPoints)
        require((point.first->GetWorldPos()-point.second).norm() == 0,
                "initial metric proposal wrote a live map point");
    std::cout << "{\"initial_metric_before_camera_rms_m\":" << before
              << ",\"initial_metric_after_camera_rms_m\":" << after
              << ",\"initial_metric_tag_rms_px\":" << proposal.after.tagRmsPx
              << ",\"initial_metric_background_rms_px\":" << proposal.after.backgroundRmsPx
              << ",\"inactive_raw_corners_used\":true"
              << ",\"damaged_tag_keyframe_rejected\":true}" << std::endl;
}

static void rejectionCases()
{
    Fixture fixture(.2);
    const auto b = fixture.window();
    auto result = MarkerGraphOptimizer::Reanchor(&fixture.map, fixture.keyframes.front(), {b[0],b[0]}, fixture.measuredScale(), .01);
    require(!result.accepted, "duplicate B observations established scale");
    result = MarkerGraphOptimizer::Reanchor(&fixture.map, fixture.keyframes.front(), b, fixture.measuredScale(), 0);
    require(!result.accepted, "zero scale uncertainty accepted");
    for(KeyFrame* keyframe : b) keyframe->mvTagPointWeights.assign(keyframe->mvTagWorldPoints.size(), .25f);
    result = MarkerGraphOptimizer::Reanchor(&fixture.map, fixture.keyframes.front(), b, fixture.measuredScale(), .01);
    require(!result.accepted, "weak-only B observations established scale");
    for(KeyFrame* keyframe : b) keyframe->mvTagPointWeights.assign(keyframe->mvTagWorldPoints.size(), 1.0f);
    b.back()->mvTagImagePoints[0].x += 40;
    result = MarkerGraphOptimizer::Reanchor(&fixture.map, fixture.keyframes.front(), b, fixture.measuredScale(), .01);
    require(!result.accepted, "inconsistent marker pixels passed validation");
    fixture.assertUnchanged();
    std::cout << "{\"rejection_cases\":4,\"live_geometry_unchanged\":true}" << std::endl;
}

static KeyFrame* mixedWeakMarker(Fixture& fixture, int view,
        const Eigen::Vector3f& storedOffset = Eigen::Vector3f::Zero(), float pixelOffset = 0)
{
    KeyFrame* keyframe=fixture.keyframes[view];
    require(keyframe->mvTagWorldPoints.empty(),"mixed-marker fixture already has tag observations");
    keyframe->mbHasTagObservation=true;
    keyframe->mbTagObservationActive=true;
    keyframe->mTagObservationConfidence=1;
    for(int corner=0;corner<4;++corner) {
        const auto& values=fixture.map.mStaticTags.at(24);
        const Eigen::Vector3f point(values[3*corner],values[3*corner+1],values[3*corner+2]);
        const auto pixel=fixture.camera.project(fixture.truth[view]*point);
        keyframe->mvTagIds.push_back(24);
        keyframe->mvTagWorldPoints.push_back(point);
        keyframe->mvTagImagePoints.emplace_back(pixel.x(),pixel.y());
        keyframe->mvTagPointWeights.push_back(1);
    }
    const auto& values=fixture.map.mStaticTags.at(25);
    const Eigen::Vector3f point(values[0],values[1],values[2]);
    const auto pixel=fixture.camera.project(fixture.truth[view]*point);
    keyframe->mvTagIds.push_back(25);
    keyframe->mvTagWorldPoints.push_back(point+storedOffset);
    keyframe->mvTagImagePoints.emplace_back(pixel.x()+pixelOffset,pixel.y());
    keyframe->mvTagPointWeights.push_back(.25f);
    return keyframe;
}

static MarkerGraphOptimizer::StagedBAInput fixedMarkerInput(const Fixture& fixture)
{
    MarkerGraphOptimizer::StagedBAInput input;
    input.keyframes=fixture.keyframes;
    input.keyframePoses=fixture.originalPoses;
    input.pointPositions=fixture.originalPoints;
    input.fixedKeyframes.insert(fixture.keyframes.begin(),fixture.keyframes.end());
    // These association tests intentionally hold physical marker geometry
    // fixed; fixed cameras alone no longer imply a surveyed marker layout.
    input.rigidMarkerLayout=true;
    return input;
}

static void weakMarkerCanonicalizationCase()
{
    Fixture fixture(0);
    KeyFrame* mixed=mixedWeakMarker(fixture,4,Eigen::Vector3f(.0025f,0,0));
    const auto original=mixed->mvTagWorldPoints;
    auto input=fixedMarkerInput(fixture);
    // Weak observations can precede every complete observation of their ID.
    // Discovery order must not decide which physical corner they constrain.
    input.keyframes.erase(std::find(input.keyframes.begin(),input.keyframes.end(),mixed));
    input.keyframes.insert(input.keyframes.begin(),mixed);
    const auto proposal=MarkerGraphOptimizer::RefineAndValidate(input);
    require(proposal.accepted,"weak stored-corner offset rejected: "+proposal.reason);
    const auto& values=fixture.map.mStaticTags.at(25);
    const Eigen::Vector3f canonical(values[0],values[1],values[2]);
    require((proposal.tagWorldCorners.at(mixed).back()-canonical).norm()<1e-7f,
            "weak singleton was not associated with its unique registered physical corner");
    require(proposal.after.tagRmsByKeyframeMarker.at({mixed,25})<1e-3,
            "canonicalized weak singleton retained cached-layout reprojection error");
    require((mixed->mvTagWorldPoints.back()-original.back()).norm()==0,
            "canonicalization wrote a live weak observation");
    fixture.assertUnchanged();
    std::cout << "{\"weak_marker_canonicalization\":true,\"weak_first_order\":true}" << std::endl;
}

static void weakMarkerAssociationGuardCase()
{
    for(bool ambiguous:{false,true}) {
        Fixture fixture(0);
        KeyFrame* mixed=mixedWeakMarker(fixture,4,Eigen::Vector3f(ambiguous?.002f:.009f,0,0));
        if(ambiguous) {
            // Two registered corners are equally close in world space. Keep
            // their adjacent edges long enough that the side-relative gate
            // alone does not decide this deliberately ambiguous association.
            const auto& values=fixture.map.mStaticTags.at(25);
            const Eigen::Vector3f nearCorner(values[0]+.004f,values[1],values[2]);
            for(int view=0;view<Fixture::views;++view) {
                KeyFrame* keyframe=fixture.keyframes[view];
                if(keyframe==mixed) continue;
                for(std::size_t i=0;i+3<keyframe->mvTagIds.size();i+=4) {
                    if(keyframe->mvTagIds[i]!=25) continue;
                    keyframe->mvTagWorldPoints[i+2]=nearCorner;
                    const auto pixel=fixture.camera.project(fixture.truth[view]*nearCorner);
                    keyframe->mvTagImagePoints[i+2]=cv::Point2f(pixel.x(),pixel.y());
                }
            }
        }
        const Eigen::Vector3f stored=mixed->mvTagWorldPoints.back();
        const auto proposal=MarkerGraphOptimizer::RefineAndValidate(fixedMarkerInput(fixture));
        require(!proposal.accepted ||
                (proposal.tagWorldCorners.at(mixed).back()-stored).norm()<1e-7f,
                ambiguous?"ambiguous weak corner was silently snapped":"far weak corner was snapped by image proximity");
        require((mixed->mvTagWorldPoints.back()-stored).norm()==0,
                "guarded weak association changed the live observation");
        fixture.assertUnchanged();
    }
    Fixture unknown(0);
    KeyFrame* mixed=mixedWeakMarker(unknown,4);
    mixed->mvTagIds.back()=999;
    const auto proposal=MarkerGraphOptimizer::RefineAndValidate(fixedMarkerInput(unknown));
    require(std::find(proposal.optimizedMarkerIds.begin(),proposal.optimizedMarkerIds.end(),999)==
            proposal.optimizedMarkerIds.end(),"unknown weak singleton created a free marker-pose variable");
    require(!proposal.accepted ||
            (proposal.tagWorldCorners.at(mixed).back()-mixed->mvTagWorldPoints.back()).norm()<1e-7f,
            "unknown weak singleton borrowed another marker's canonical corner");
    unknown.assertUnchanged();
    std::cout << "{\"weak_marker_far_and_ambiguous_not_snapped\":true,\"unknown_weak_marker_not_variable\":true}" << std::endl;
}

static void weakMarkerReanchorRetryCase()
{
    Fixture fixture(0);
    KeyFrame* mixed=mixedWeakMarker(fixture,4,Eigen::Vector3f::Zero(),40);
    const auto originalPixel=mixed->mvTagImagePoints.back();
    const auto proposal=MarkerGraphOptimizer::Reanchor(&fixture.map,fixture.keyframes.front(),
        fixture.window(),1.,.01);
    require(proposal.accepted,"isolated weak marker vetoed interval BA: "+proposal.reason);
    require(proposal.after.tagRmsByKeyframeMarker.count({mixed,24})==1 &&
            proposal.after.tagRmsByKeyframeMarker.count({mixed,25})==0,
            "weak retry discarded the same keyframe's complete strong marker or retained its bad weak group");
    require(proposal.before.tagRmsByKeyframeMarker.count({mixed,25})==0,
            "weak retry compared baseline and candidate from different evidence populations");
    require(proposal.excludedTagKeyFrameIds.empty(),"weak retry excluded an entire tag keyframe");
    for(KeyFrame* keyframe:fixture.window())
        require(proposal.after.tagRmsByKeyframeMarker.count({keyframe,24})==1 &&
                proposal.after.tagRmsByKeyframeMarker.count({keyframe,25})==1,
                "weak retry removed strong endpoint evidence");
    require(mixed->mvTagImagePoints.back()==originalPixel,"weak retry edited the original tracked pixel");
    fixture.assertUnchanged();
    std::cout << "{\"weak_marker_reanchor_group_retry\":true,\"same_frame_strong_marker_retained\":true}" << std::endl;
}

static void weakMarkerRetrySafetyCase()
{
    Fixture strong(0,false,1.f,4);
    strong.keyframes[4]->mvTagImagePoints[4].x+=40;
    const auto contradiction=MarkerGraphOptimizer::Reanchor(&strong.map,strong.keyframes.front(),
        strong.window(),1.,.01);
    require(!contradiction.accepted,"weak-group retry erased a contradictory full strong marker");
    strong.assertUnchanged();
    Fixture excessive(0);
    for(int view:{3,4,5,6}) mixedWeakMarker(excessive,view,Eigen::Vector3f::Zero(),40);
    const auto bounded=MarkerGraphOptimizer::Reanchor(&excessive.map,excessive.keyframes.front(),
        excessive.window(),1.,.01);
    require(!bounded.accepted,"weak-group retries exceeded their three-group budget");
    excessive.assertUnchanged();
    std::cout << "{\"weak_marker_retry_preserves_strong_contradictions\":true,\"weak_retry_bounded\":true}" << std::endl;
}

static void backgroundDilutionCase()
{
    Fixture fixture(0, true);
    MarkerGraphOptimizer::StagedBAInput input;
    input.keyframes = fixture.keyframes;
    input.keyframePoses = fixture.originalPoses;
    input.pointPositions = fixture.originalPoints;
    // The damaged frame is fixed to an independently trusted pose. Retain one
    // healthy variable camera so the test exercises a real SE3/XYZ BA solve.
    input.fixedKeyframes.insert(fixture.keyframes.begin(), fixture.keyframes.end()-1);
    const auto proposal = MarkerGraphOptimizer::RefineAndValidate(input);
    require(proposal.accepted,
            "an existing damaged background frame vetoed an otherwise improving BA: "+proposal.reason);
    require(proposal.after.backgroundRmsPx < 3 &&
            proposal.before.backgroundRmsByKeyframe.at(fixture.keyframes[6]) > 3 &&
            proposal.after.backgroundRmsByKeyframe.at(fixture.keyframes[6]) > 3 &&
            proposal.after.backgroundRmsByKeyframe.at(fixture.keyframes[6]) <=
                proposal.before.backgroundRmsByKeyframe.at(fixture.keyframes[6])+.5,
            "grandfathered background frame was hidden globally or materially worsened");
    input.filterFixedBackgroundOutliers = true;
    const auto filtered = MarkerGraphOptimizer::RefineAndValidate(input);
    require(filtered.accepted &&
            filtered.after.backgroundRmsByKeyframe.at(fixture.keyframes[6]) < 3,
            "a stale observation on an unchanged boundary keyframe still vetoed interval BA: "+
            filtered.reason);
    fixture.assertUnchanged();
    std::cout << "{\"global_background_rms_px\":" << proposal.after.backgroundRmsPx
              << ",\"damaged_frame_rms_px\":" << proposal.after.backgroundRmsByKeyframe.at(fixture.keyframes[6])
              << ",\"existing_damage_not_worsened\":true,\"fixed_boundary_filter_accepted\":true}" << std::endl;
}

static void graphInputOrderCase()
{
    Fixture f(0);
    // These small, already-admitted layout differences exercise which raw
    // observation supplies canonical marker geometry, not a changed weight.
    for(KeyFrame* k:f.keyframes) if(k!=f.keyframes.front())
        for(auto& corner:k->mvTagWorldPoints) corner.x()+=.0002f;
    MarkerGraphOptimizer::StagedBAInput input;
    input.keyframes=f.keyframes; input.keyframePoses=f.originalPoses;
    input.pointPositions=f.originalPoints; input.fixedKeyframes.insert(f.keyframes.front());
    const auto baseline=MarkerGraphOptimizer::RefineAndValidate(input);
    require(baseline.accepted,"ordered BA fixture rejected: "+baseline.reason);
    for(int permutation=0;permutation<3;++permutation) {
        if(permutation==0) std::reverse(input.keyframes.begin(),input.keyframes.end());
        else std::rotate(input.keyframes.begin(),input.keyframes.begin()+3,input.keyframes.end());
        const auto reordered=MarkerGraphOptimizer::RefineAndValidate(input);
        require(reordered.accepted,"reordered BA rejected: "+reordered.reason);
        for(const auto& pose:baseline.keyframePoses)
            require((pose.second.matrix()-reordered.keyframePoses.at(pose.first).matrix()).norm()==0,
                    "BA pose depends on input keyframe order");
        for(const auto& point:baseline.pointPositions)
            require((point.second-reordered.pointPositions.at(point.first)).norm()==0,
                    "BA point depends on input keyframe order");
        for(const auto& marker:baseline.staticTags)
            for(std::size_t i=0;i<marker.second.size();++i)
                require((marker.second[i]-reordered.staticTags.at(marker.first)[i]).norm()==0,
                        "BA marker gauge depends on input keyframe order");
    }
    input.keyframes.push_back(nullptr);
    require(!MarkerGraphOptimizer::RefineAndValidate(input).accepted,
            "sorting accepted a null keyframe");
    f.assertUnchanged();
    std::cout << "{\"ba_input_order_invariant\":true,\"exact_float_outputs\":true}" << std::endl;
}

static void graphAllocationOrderCase()
{
    Fixture a(.10), b(.10,false,1.f,-1,false,false,true);
    auto windowB=b.window();std::reverse(windowB.begin(),windowB.end());
    const auto pa=MarkerGraphOptimizer::Reanchor(&a.map,a.keyframes.front(),a.window(),a.measuredScale(),.01);
    const auto pb=MarkerGraphOptimizer::Reanchor(&b.map,b.keyframes.front(),windowB,b.measuredScale(),.01);
    require(pa.accepted && pb.accepted,"allocation-order reanchor rejected: "+pa.reason+" / "+pb.reason);
    for(int i=0;i<Fixture::views;++i) {
        require((pa.keyframePoses.at(a.keyframes[i]).matrix()-pb.keyframePoses.at(b.keyframes[i]).matrix()).norm()==0,
                "reanchor pose depends on allocation order");
        require(pa.replayScaleMultipliers.at(a.keyframes[i])==pb.replayScaleMultipliers.at(b.keyframes[i]),
                "reanchor scale depends on allocation order");
    }
    for(std::size_t i=0;i<a.ownedPoints.size();++i)
        require((pa.pointPositions.at(a.ownedPoints[i].get())-pb.pointPositions.at(b.ownedPoints[i].get())).norm()==0,
                "reanchor point depends on allocation order");
    a.assertUnchanged();b.assertUnchanged();
    std::cout << "{\"sim3_ba_allocation_order_invariant\":true,\"exact_float_outputs\":true}" << std::endl;
}

static void erasedReferenceOrderCase()
{
    for(bool reverse:{false,true}) {
        Fixture f(0,false,1.f,-1,false,false,reverse);
        MapPoint* p=f.ownedPoints.front().get();
        p->AddObservation(f.keyframes[3],f.base.N-1);
        p->AddObservation(f.keyframes[4],f.base.N-1);
        require(p->GetReferenceKeyFrame()==f.keyframes[1],"incorrect erasure fixture reference");
        p->EraseObservation(f.keyframes[1]);
        require(!p->isBad(),"well-observed point was culled by reference replacement");
        require(p->GetReferenceKeyFrame()==f.keyframes[0],
                "replacement point reference depends on observer memory address");
    }
    std::cout << "{\"erased_point_reference_id_order\":true}" << std::endl;
}

static void independentGaugeMarkerCase()
{
    Fixture f(0);
    for(KeyFrame* k:f.keyframes)
        for(std::size_t i=0;i<k->mvTagIds.size();++i)
            if(k->mvTagIds[i]==25) k->mvTagWorldPoints[i].x()+=.008f;
    for(int j=0;j<4;++j) f.map.mStaticTags[25][3*j]+=.008f;
    f.originalMarkers=f.map.mStaticTags;
    MarkerGraphOptimizer::StagedBAInput input;
    input.keyframes=f.keyframes; input.keyframePoses=f.originalPoses;
    input.pointPositions=f.originalPoints; input.fixedKeyframes.insert(f.keyframes.front());
    MarkerGraphOptimizer::Options old; old.fixAllObservedGaugeMarkers=true;
    const auto locked=MarkerGraphOptimizer::RefineAndValidate(input,old);
    require(!locked.accepted,"regression did not expose overconstrained marker layout");
    const auto free=MarkerGraphOptimizer::RefineAndValidate(input);
    require(free.accepted && free.after.tagRmsPx<.01,"independent marker gauge correction failed: "+free.reason);
    require((free.keyframePoses.at(f.keyframes.front()).matrix()-f.originalPoses.at(f.keyframes.front()).matrix()).norm()<1e-7f,
            "independent marker refinement moved the fixed camera gauge");
    for(int j=0;j<4;++j) {
        const auto& fixed=f.originalMarkers.at(24);
        require((free.staticTags.at(24)[j]-Eigen::Vector3f(fixed[3*j],fixed[3*j+1],fixed[3*j+2])).norm()<1e-5f,
                "independent marker refinement damaged the correct marker geometry");
        require(std::abs(free.staticTags.at(25)[j].x()-f.originalMarkers.at(25)[3*j]+.008f)<.0002f,
                "independent marker relative placement was not recovered");
    }
    input.rigidMarkerLayout=true;
    require(!MarkerGraphOptimizer::RefineAndValidate(input).accepted,
            "calibrated rigid-board internal geometry was silently released");
    f.assertUnchanged();
    std::cout << "{\"independent_gauge_marker\":true,\"fixed_world_preserved\":true,\"rigid_board_preserved\":true}" << std::endl;
}

static void unsurveyedScaleObservabilityCase()
{
    for(bool distinctMarkers:{false,true}) {
        Fixture f(0);
        MarkerGraphOptimizer::StagedBAInput input;
        input.keyframes=f.keyframes; input.fixedKeyframes.insert(f.keyframes.front());
        input.pointPositions=f.originalPoints;
        for(std::size_t v=0;v<f.keyframes.size();++v) {
            KeyFrame* k=f.keyframes[v];
            if(distinctMarkers) for(int& id:k->mvTagIds) id+=int(v)*10;
            else input.keyframePoses[k]=f.keyframes.front()->GetPose();
        }
        const auto p=MarkerGraphOptimizer::RefineAndValidate(input);
        require(!p.accepted && p.reason=="insufficient_shared_marker_baseline",
                "private markers or zero translation invented metric scale: "+p.reason);
    }
    std::cout << "{\"private_markers_do_not_establish_scale\":true,\"zero_baseline_rejected\":true}" << std::endl;
}

static void verifiedBackgroundPixelRetryCase()
{
    for(bool twoMarkers:{false,true}) {
        Fixture f(0,true);
        KeyFrame* k=f.keyframes[6];
        k->mbHasTagObservation=k->mbTagObservationActive=true;
        k->mTagObservationConfidence=1;
        for(const auto& marker:f.map.mStaticTags) {
            if(!twoMarkers && marker.first==25) continue;
            for(int j=0;j<4;++j) {
                const Eigen::Vector3f p(marker.second[3*j],marker.second[3*j+1],marker.second[3*j+2]);
                const auto uv=f.camera.project(f.truth[6]*p);
                k->mvTagIds.push_back(marker.first); k->mvTagWorldPoints.push_back(p);
                k->mvTagImagePoints.emplace_back(uv.x(),uv.y()); k->mvTagPointWeights.push_back(1);
            }
        }
        MapPoint* damaged=nullptr;
        for(std::size_t i=0;i<f.ownedPoints.size();++i) {
            MapPoint* p=f.ownedPoints[i].get();
            const auto views=p->GetObservations(); const auto o=views.find(k);
            if(o==views.end() || views.size()<3) continue;
            const auto pixel=k->mvKeysUn[std::get<0>(o->second)].pt;
            const Eigen::Vector3f camera=f.truth[6]*f.truePoints[i];
            const auto expected=f.camera.project(camera);
            if(std::abs(pixel.y-expected.y())<30) continue;
            // Initially this wrong 3-D point agrees with the corrupt pixel.
            // Its two other views identify the disagreement during BA.
            Eigen::Vector3f biased=p->GetWorldPos();
            biased.y()+=(pixel.y-expected.y())*camera.z()/500.f;
            p->SetWorldPos(biased); f.originalPoints[p]=biased; damaged=p; break;
        }
        require(damaged,"missing corrupt multi-view background pixel");
        const auto result=MarkerGraphOptimizer::Reanchor(&f.map,f.keyframes.front(),f.window(),1.,.01);
        if(twoMarkers) {
            require(result.accepted,"independently verified pixel retry failed: "+result.reason);
            require(result.after.backgroundRmsByKeyframe.at(k)<3,
                    "verified retry still damaged its source keyframe");
            require(result.before.backgroundObservations==result.after.backgroundObservations,
                    "retry compared different raw pixel populations");
        } else require(!result.accepted,"single marker erased a conflicting ORB pixel");
        f.assertUnchanged();
        require(damaged->GetObservations().count(k)==1,"trial erased a live observation");
    }
    std::cout << "{\"verified_background_pixel_retry\":true,\"single_marker_cannot_prune\":true}" << std::endl;
}

static void finalBackgroundPolicyCase()
{
    require(!BackgroundResidualFrameConsistent(.4,1.5) &&
            FinalBackgroundResidualFrameConsistent(.4,1.5),"healthy redistribution not distinguished");
    require(!FinalBackgroundResidualFrameConsistent(2.9,3.1),"new over-noise damage accepted");
    require(!FinalBackgroundResidualFrameConsistent(4.,4.6),"old damage materially worsened");
    require(!FinalBackgroundResidualFrameConsistent(1.,std::numeric_limits<double>::quiet_NaN()),"NaN accepted");
    for(int mode=0;mode<4;++mode) {
        Fixture f(0,false,1.f,-1,false,false,false,false,mode==3?0:(mode==2?20:1));
        for(std::size_t v=0;v<f.keyframes.size();++v) {
            KeyFrame* k=f.keyframes[v];
            if(k->mbHasTagObservation && !k->mvTagIds.empty()) continue;
            k->mbHasTagObservation=k->mbTagObservationActive=true;k->mTagObservationConfidence=1;
            for(const auto& marker:f.map.mStaticTags) {
                if(mode==1 && v==6 && marker.first==25) continue;
                for(int j=0;j<4;++j) {
                    const Eigen::Vector3f p(marker.second[3*j],marker.second[3*j+1],marker.second[3*j+2]);
                    const auto uv=f.camera.project(f.truth[v]*p);
                    k->mvTagIds.push_back(marker.first);k->mvTagWorldPoints.push_back(p);
                    k->mvTagImagePoints.emplace_back(uv.x(),uv.y());k->mvTagPointWeights.push_back(1);
                }
            }
        }
        const auto legacy=MarkerGraphOptimizer::RefineMetricMap(&f.map,MarkerGraphOptimizer::Options(),true,false);
        const auto result=MarkerGraphOptimizer::RefineMetricMap(&f.map);
        std::cout << "FINAL_BACKGROUND_CASE mode=" << mode << " legacy=" << legacy.accepted
                  << " accepted=" << result.accepted << " reason=" << result.reason
                  << " bg_before=" << result.before.backgroundRmsPx << " bg_after=" << result.after.backgroundRmsPx
                  << " observations=" << result.after.backgroundObservations << std::endl;
        require(result.excludedBackgroundKeyFrameIds.empty(),"default final BA discarded a whole frame");
        if(mode==0) {
            require(result.accepted,"independently supported bad pixel not recovered: "+result.reason);
            MarkerGraphOptimizer::StagedBAInput seed;
            seed.keyframes=f.keyframes;seed.fixedKeyframes.insert(f.keyframes.front());
            seed.pointPositions=f.originalPoints;seed.filterInitialBackgroundOutliers=true;
            const auto unpruned=MarkerGraphOptimizer::RefineAndValidate(seed,MarkerGraphOptimizer::Options(),nullptr,true,true);
            require(!unpruned.accepted && unpruned.reason=="background_reprojection_validation_failed",
                    "fixture did not trigger actual background retry");
            require(result.after.backgroundObservations+1==unpruned.after.backgroundObservations,
                    "retry did not remove exactly the one contradicted observation");
            require(legacy.excludedBackgroundKeyFrameIds.size()==1 &&
                    legacy.after.backgroundObservations+61==unpruned.after.backgroundObservations,
                    "legacy A/B did not expose the whole-frame removal");
            require(result.before.backgroundObservations==result.after.backgroundObservations,
                    "retry before/after used different observation sets");
        } else if(mode<3) require(!result.accepted,"unsupported or excessive outliers bypassed validation");
        else require(result.accepted==legacy.accepted && result.reason==legacy.reason,
                     "clean graph changed acceptance");
        f.assertUnchanged();
    }
}

static void committedBoundaryAdmissionCase()
{
    Fixture fixture(0);
    MarkerGraphOptimizer::StagedBAInput raw;
    raw.keyframes=fixture.keyframes;
    raw.keyframePoses=fixture.originalPoses;
    raw.pointPositions=fixture.originalPoints;
    raw.fixedKeyframes.insert(fixture.keyframes.front());
    raw.filterFixedBackgroundOutliers=true;
    raw.useCommittedAdmission=true;
    const auto baseline=MarkerGraphOptimizer::RefineAndValidate(raw);
    require(baseline.accepted,"clean admission fixture failed");
    auto staged=raw;
    // A seed is not a measurement: moving a healthy multi-view point far
    // from its fixed observer must not silently remove that observer's pixel.
    MapPoint* displaced=nullptr;
    for(const auto& point:raw.pointPositions)
        if(point.first->GetObservations().count(fixture.keyframes.front()) &&
           point.first->GetObservations().size()>=2) {displaced=point.first;break;}
    require(displaced!=nullptr,"no boundary multi-view point");
    staged.pointPositions.at(displaced).x()+=.04f;
    const auto frozen=MarkerGraphOptimizer::RefineAndValidate(staged);
    require(frozen.before.backgroundObservations==baseline.before.backgroundObservations &&
            frozen.after.backgroundObservations==baseline.before.backgroundObservations,
            "Sim3 seed changed committed observation membership");
    staged.useCommittedAdmission=false;
    const auto changed=MarkerGraphOptimizer::RefineAndValidate(staged);
    require(changed.before.backgroundObservations<baseline.before.backgroundObservations,
            "regression did not expose seed-dependent boundary admission");
    fixture.assertUnchanged();
    std::cout << "{\"committed_boundary_admission\":true,\"original_observations\":"
              << baseline.before.backgroundObservations << ",\"frozen_observations\":"
              << frozen.before.backgroundObservations << ",\"seed_filtered_observations\":"
              << changed.before.backgroundObservations << "}" << std::endl;
}

static void contradictoryScaleCueCase()
{
    Fixture fixture(0);
    // The complete raw graph is already metric and geometrically correct.
    // A false scale cue must not survive only as a replay-unit multiplier
    // when the following raw-observation BA restores the original geometry.
    const auto proposal = MarkerGraphOptimizer::Reanchor(&fixture.map, fixture.keyframes.front(),
                                                         fixture.window(), .85, .01);
    fixture.assertUnchanged();
    std::cout << "{\"contradictory_scale_cue_accepted\":" << (proposal.accepted ? "true" : "false")
              << ",\"reason\":\"" << proposal.reason << "\""
              << ",\"tag_rms_px\":" << proposal.after.tagRmsPx
              << ",\"background_rms_px\":" << proposal.after.backgroundRmsPx;
    if(proposal.keyframePoses.size() == fixture.originalPoses.size())
        std::cout << ",\"after_camera_rms_m\":" << fixture.cameraRms(proposal.keyframePoses);
    std::cout << "}" << std::endl;
    printScaleEvidence(fixture, proposal);
    require(!proposal.accepted && proposal.reason == "scale_geometry_consistency_failed",
            "BA discarded a contradictory scale cue in geometry but retained it in replay units");
}

static void uncertainScaleCueCase()
{
    Fixture fixture(0);
    // This video's uncertain 1.124x ratio must not force a correct raw graph
    // to rescale. Raw corner BA can prefer 1x and publish matching depth units.
    const auto proposal = MarkerGraphOptimizer::Reanchor(&fixture.map, fixture.keyframes.front(),
                                                         fixture.window(), 1.12418362, .0967327696);
    fixture.assertUnchanged();
    require(proposal.accepted, "uncertain scale cue vetoed valid raw geometry: "+proposal.reason);
    require(fixture.cameraRms(proposal.keyframePoses)<1e-4,
            "uncertain scale cue distorted an already correct trajectory");
    for(KeyFrame* keyframe:fixture.window())
        require(std::abs(proposal.replayScaleMultipliers.at(keyframe)-1.0f)<1e-3,
                "uncertain provisional scale leaked into replay units");
    assertSingletonScaleContract(fixture, proposal);
    std::cout << "{\"uncertain_scale_cue_accepted\":true,\"final_scale_near_one\":true}" << std::endl;
}

static void revisitedAnchorIntervalCase()
{
    Fixture fixture(.20);
    // A reattached return keyframe shortcuts the spanning tree. The middle
    // of the physically travelled interval must remain optimizable.
    for(KeyFrame* keyframe:fixture.window()) keyframe->ChangeParent(fixture.keyframes.front());
    MarkerGraphOptimizer::Options options;
    options.covisibleNeighbors=0;
    const auto proposal=MarkerGraphOptimizer::Reanchor(&fixture.map,fixture.keyframes.front(),
        fixture.window(),fixture.measuredScale(),.01,options);
    require(proposal.accepted,"revisited interval rejected: "+proposal.reason);
    require(proposal.affectedKeyFrameIds.size()==fixture.keyframes.size()-1,
            "spanning-tree shortcut omitted travelled interval keyframes");
    require(fixture.cameraRms(proposal.keyframePoses)<.3*fixture.cameraRms(fixture.originalPoses),
            "full-interval scale correction did not improve geometry");
    fixture.assertUnchanged();
    assertSingletonScaleContract(fixture,proposal);
    std::cout << "{\"revisited_interval_full_coverage\":true}" << std::endl;
}

static void weakUnitSeedCase()
{
    for(double drift:{0.,.02,.10,.20}) {
        Fixture fixture(drift);
        const auto proposal=MarkerGraphOptimizer::Reanchor(&fixture.map,fixture.keyframes.front(),
                                                           fixture.window(),1.,.1);
        fixture.assertUnchanged();
        require(proposal.accepted,"corner-only committed seed rejected: "+proposal.reason);
        const double before=fixture.cameraRms(fixture.originalPoses);
        const double after=fixture.cameraRms(proposal.keyframePoses);
        std::cout << "WEAK_UNIT_BA drift=" << drift << " before=" << before
                  << " after=" << after << std::endl;
        require(after<(drift<.1?1e-4:.3*before),
                "corner-only BA failed to retain near-metric geometry or reduce large drift by 70%");
        require(proposal.affectedKeyFrameIds.size()==fixture.keyframes.size()-1,
                "corner-only seed omitted interval cameras");
        for(KeyFrame* keyframe:fixture.window()) {
            const auto evidence=depthScaleEvidence(fixture,proposal,keyframe);
            require(evidence.second>=3 &&
                    std::abs(proposal.replayScaleMultipliers.at(keyframe)-evidence.first)<1e-5,
                    "corner-only BA published the unit prior instead of measured local depth scale");
        }
        require(std::abs(proposal.replayScaleMultipliers.at(fixture.keyframes.back())/
                         fixture.measuredScale()-1)<(drift<.1?.001:.01),
                "corner-only BA failed to recover the endpoint scale");
        assertSingletonScaleContract(fixture,proposal);
    }
    std::cout << "{\"weak_unit_seed_retains_full_metric_validation\":true}" << std::endl;
}

static void unobservableScaleCase()
{
    Fixture fixture(0, false, 100.0f);
    const auto proposal = MarkerGraphOptimizer::Reanchor(&fixture.map, fixture.keyframes.front(),
                                                         fixture.window(), 1.0, .01);
    fixture.assertUnchanged();
    require(!proposal.accepted && proposal.reason == "insufficient_scale_geometry",
            "near-zero-parallax background supplied a metric scale");
    std::cout << "{\"unobservable_scale_rejected\":true}" << std::endl;
}

static void markerOnlyInteriorCase()
{
    Fixture fixture(.20, false, 1.0f, 5);
    const auto proposal = MarkerGraphOptimizer::Reanchor(&fixture.map, fixture.keyframes.front(),
                                                         fixture.window(), fixture.measuredScale(), .01);
    fixture.assertUnchanged();
    require(proposal.accepted, "marker-only interior node rejected: "+proposal.reason);
    require(proposal.replayScaleMultipliers.at(fixture.keyframes[5]) == 1.0f,
            "marker-only interior node invented a local visual scale");
    require(fixture.cameraRms(proposal.keyframePoses) < .3*fixture.cameraRms(fixture.originalPoses),
            "marker-only interior node prevented the measured path correction");
    assertSingletonScaleContract(fixture, proposal);
    std::cout << "{\"marker_only_interior_unit\":1,\"singleton_contract\":true}" << std::endl;
}

static void stagedVisualLoopCase()
{
    Fixture fixture(.10,false,1.f,-1,false,true);
    auto* current=fixture.keyframes.back();auto* matched=fixture.keyframes.front();
    const auto truth=fixture.truth.back().cast<double>();
    g2o::Sim3 seed(truth.unit_quaternion(),truth.translation(),1.0);
    auto result=MarkerGraphOptimizer::ProposeVisualLoop(&fixture.map,current,matched,seed,fixture.loopMatches);
    fixture.assertUnchanged();
    require(result.accepted,"staged loop rejected: "+result.reason);
    require(result.pointAliases.size()==25,"trial fusion lost pairs");
    require(fixture.cameraRms(result.keyframePoses)<.3*fixture.cameraRms(fixture.originalPoses),"loop failed to correct path");
    const auto loopBaseline=result.before;
    MarkerGraphOptimizer::Options loopOptions;
    require(loopOptions.useEssentialGraphCovisibility && loopOptions.retryLoopAliasesAfterTagFailure,
            "validated loop policy is not enabled by default");
    const g2o::Sim3 invalid(truth.unit_quaternion(),truth.translation(),.00325174);
    result=MarkerGraphOptimizer::ProposeVisualLoop(&fixture.map,current,matched,invalid,fixture.loopMatches);
    require(!result.accepted,"catastrophic scale accepted");fixture.assertUnchanged();
    auto wrong=fixture.loopMatches;std::vector<std::size_t> indices;
    for(std::size_t i=0;i<wrong.size();++i)if(wrong[i])indices.push_back(i);
    for(std::size_t i=0;i<indices.size();++i)wrong[indices[i]]=fixture.loopMatches[indices[indices.size()-1-i]];
    result=MarkerGraphOptimizer::ProposeVisualLoop(&fixture.map,current,matched,seed,wrong);
    require(!result.accepted,"incorrect point correspondences accepted");fixture.assertUnchanged();
    loopOptions.useEssentialGraphCovisibility=false;loopOptions.retryLoopAliasesAfterTagFailure=false;
    auto seeded=MarkerGraphOptimizer::ProposeVisualLoop(&fixture.map,current,matched,seed,fixture.loopMatches,loopOptions);
    require(seeded.accepted && seeded.pointAliases.size()==25,"legacy loop control rejected: "+seeded.reason);
    require(seeded.before.tagCorners==loopBaseline.tagCorners &&
            seeded.before.backgroundObservations==loopBaseline.backgroundObservations,
            "legacy loop control changed raw validation population");fixture.assertUnchanged();
    seeded=MarkerGraphOptimizer::ProposeVisualLoop(&fixture.map,current,matched,invalid,fixture.loopMatches,loopOptions);
    require(!seeded.accepted,"legacy loop control bypassed scale gate");fixture.assertUnchanged();
    seeded=MarkerGraphOptimizer::ProposeVisualLoop(&fixture.map,current,matched,seed,wrong,loopOptions);
    require(!seeded.accepted,"legacy loop control bypassed raw correspondence gates");fixture.assertUnchanged();
    loopOptions.useEssentialGraphCovisibility=true;
    seeded=MarkerGraphOptimizer::ProposeVisualLoop(&fixture.map,current,matched,seed,fixture.loopMatches,loopOptions);
    require(seeded.accepted && seeded.pointAliases.size()==25,"essential covisibility damaged valid loop: "+seeded.reason);
    require(seeded.before.tagCorners==loopBaseline.tagCorners &&
            seeded.before.backgroundObservations==loopBaseline.backgroundObservations,
            "essential covisibility changed raw validation population");fixture.assertUnchanged();
    seeded=MarkerGraphOptimizer::ProposeVisualLoop(&fixture.map,current,matched,invalid,fixture.loopMatches,loopOptions);
    require(!seeded.accepted,"essential covisibility bypassed scale gate");fixture.assertUnchanged();
    seeded=MarkerGraphOptimizer::ProposeVisualLoop(&fixture.map,current,matched,seed,wrong,loopOptions);
    require(!seeded.accepted,"essential covisibility bypassed raw correspondence gates");fixture.assertUnchanged();
    loopOptions.retryLoopAliasesAfterTagFailure=true;
    seeded=MarkerGraphOptimizer::ProposeVisualLoop(&fixture.map,current,matched,seed,fixture.loopMatches,loopOptions);
    require(seeded.accepted && seeded.pointAliases.size()==25,"masked-alias retry damaged valid loop: "+seeded.reason);
    require(seeded.before.tagCorners==loopBaseline.tagCorners &&
            seeded.before.backgroundObservations==loopBaseline.backgroundObservations,
            "masked-alias retry changed raw validation population");fixture.assertUnchanged();
    seeded=MarkerGraphOptimizer::ProposeVisualLoop(&fixture.map,current,matched,seed,wrong,loopOptions);
    require(!seeded.accepted,"masked-alias retry bypassed raw correspondence gates");fixture.assertUnchanged();
    // An inconsistent old raw marker group is never erased or grandfathered.
    fixture.window().front()->mvTagImagePoints[0].x+=80.f;
    seeded=MarkerGraphOptimizer::ProposeVisualLoop(&fixture.map,current,matched,seed,fixture.loopMatches,loopOptions);
    require(!seeded.accepted,"masked-alias retry accepted a contradictory historical marker");
    require(seeded.before.tagCorners==loopBaseline.tagCorners,"masked-alias retry removed a historical corner");
    fixture.assertUnchanged();
    std::cout << "{\"staged_loop_and_rejection_isolation\":true}" << std::endl;
}

static void knownMarkerLoopScaleCase()
{
    Fixture f(0,false,1.f,-1,false,true,false,true);
    auto* current=f.keyframes.back();auto* origin=f.keyframes.front();
    const double scale=.17;
    const auto truth=f.truth.back().cast<double>();
    const g2o::Sim3 seed(truth.unit_quaternion(),truth.translation()*scale,scale);
    for(auto& marker:f.map.mStaticTags)for(int i=0;i<4;++i)marker.second[3*i+2]=.6f;
    f.originalMarkers=f.map.mStaticTags;
    for(std::size_t view=0;view<f.keyframes.size();++view) {
        KeyFrame* k=f.keyframes[view];
        for(std::size_t i=0;i<k->mvTagWorldPoints.size();++i) {
            k->mvTagWorldPoints[i].z()=.6f;
            const auto pixel=f.camera.project(f.truth[view]*k->mvTagWorldPoints[i]);
            k->mvTagImagePoints[i]=cv::Point2f(pixel.x(),pixel.y());
        }
    }
    // A translated late window is in a different visual unit. Its raw marker
    // pixels remain independent, unchanged measurements in the physical world.
    for(KeyFrame* k:f.window()) {
        auto pose=k->GetPose();pose.translation()*=scale;k->SetPose(pose);
        f.originalPoses[k]=pose;
    }
    // Natural loop support spans the image and lies outside printed markers.
    for(std::size_t i=0;i<f.loopMatches.size();++i) {
        MapPoint* b=f.loopMatches[i];if(!b)continue;
        const Eigen::Vector3f world=b->GetWorldPos();
        MapPoint* a=current->GetMapPoint(i);a->SetWorldPos(float(scale)*world);
        f.originalPoints[a]=a->GetWorldPos();
    }
    MarkerGraphOptimizer::Options options;
    const auto check=[&](const g2o::Sim3& candidate) {
        return MarkerGraphOptimizer::ValidateKnownMarkerLoopScale(&f.map,current,candidate,f.loopMatches,options);
    };
    const auto evidence=check(seed);
    require(evidence.valid && evidence.baselineM>=.04 && evidence.selfRms<1e-3 && evidence.holdoutRms<1e-3,
            "independent known-marker support did not admit the true scale");
    require(evidence.holdoutKeyframeId<current->mnId,"future KF authorized online correction");
    require(!check(g2o::Sim3(seed.rotation(),seed.translation()*1.1,scale*1.1)).valid,
            "wrong scale passed by preserving a single-view PnP pose");
    require(!check(g2o::Sim3(seed.rotation(),seed.translation(),.003)).valid,
            "catastrophic unsupported scale passed");
    // A single translated strong view is insufficient: the origin is a gauge,
    // not an independent local held-out observation of this visual unit.
    for(KeyFrame* k:f.window())if(k!=current)k->mbTagObservationActive=false;
    require(!check(seed).valid,"one strong marker view authorized large loop scale");
    for(KeyFrame* k:f.window())if(k!=current)k->mbTagObservationActive=true;
    const auto originWeights=origin->mvTagPointWeights;
    std::fill(origin->mvTagPointWeights.begin(),origin->mvTagPointWeights.end(),.5f);
    require(!check(seed).valid,"unanchored/provisional marker authorized a large loop");
    origin->mvTagPointWeights=originWeights;
    const auto pixels=current->mvTagImagePoints;
    for(auto& p:current->mvTagImagePoints)p.y+=20;
    require(!check(seed).valid,"corrupted raw corners authorized a large loop");
    current->mvTagImagePoints=pixels;
    const auto weights=current->mvTagPointWeights;
    current->mvTagPointWeights[0]=.5f;current->mvTagPointWeights[4]=.5f;
    require(!check(seed).valid,"partial marker corners authorized a large loop");
    current->mvTagPointWeights=weights;
    for(KeyFrame* k:f.window())if(k!=current)k->SetPose(current->GetPose());
    require(!check(seed).valid,"zero-baseline marker views authorized a large loop");
    for(KeyFrame* k:f.window())k->SetPose(f.originalPoses.at(k));
    auto sparse=f.loopMatches;int retained=0;
    for(auto& p:sparse)if(p && ++retained>19)p=nullptr;
    require(!MarkerGraphOptimizer::ValidateKnownMarkerLoopScale(&f.map,current,seed,sparse,options).valid,
            "insufficient natural feature support authorized a large loop");
    options.allowLargeKnownMarkerLoopRepair=false;
    const auto old=MarkerGraphOptimizer::ProposeVisualLoop(&f.map,current,origin,seed,f.loopMatches,options);
    require(!old.accepted && old.reason=="metric_loop_seed_scale_out_of_bounds",
            "disabled diagnostic did not retain the ordinary scale guard");
    f.assertUnchanged();
    std::cout << "{\"known_marker_loop_scale_independent_holdout\":true,\"single_view_wrong_scale_and_weak_evidence_rejected\":true}" << std::endl;
}

static void provisionalMarkerProposalCase()
{
    const auto corner=[](const std::vector<float>& marker,int j) {
        return Eigen::Vector3f(marker[3*j],marker[3*j+1],marker[3*j+2]);
    };
    for(bool loop:{false,true}) {
        Fixture f(0,false,1.f,-1,false,loop);
        const auto truth=f.originalMarkers.at(25);
        constexpr float offset=.12f;
        KeyFrame* origin=f.keyframes.front();
        // Only marker 24 was observed at the accepted A. Marker 25 was
        // registered later in a drifted world, never validated by interval BA.
        for(std::size_t i=origin->mvTagIds.size();i-- >0;)
            if(origin->mvTagIds[i]==25) {
                origin->mvTagIds.erase(origin->mvTagIds.begin()+i);
                origin->mvTagWorldPoints.erase(origin->mvTagWorldPoints.begin()+i);
                origin->mvTagImagePoints.erase(origin->mvTagImagePoints.begin()+i);
                origin->mvTagPointWeights.erase(origin->mvTagPointWeights.begin()+i);
            }
        for(KeyFrame* k:f.keyframes)
            for(std::size_t i=0;i<k->mvTagIds.size();++i)
                if(k->mvTagIds[i]==25) k->mvTagWorldPoints[i].x()+=offset;
        for(int j=0;j<4;++j) f.map.mStaticTags[25][3*j]+=offset;
        f.originalMarkers=f.map.mStaticTags;
        f.map.SetMarkerScaleAnchorKFId(long(origin->mnId));

        MarkerGraphOptimizer::StagedBAInput legacy;
        legacy.keyframes=f.keyframes;legacy.fixedKeyframes.insert(origin);
        legacy.pointPositions=f.originalPoints;
        MarkerGraphOptimizer::Options oldPolicy;
        oldPolicy.legacyIndependentMarkerWorldPrior=true;
        const auto old=MarkerGraphOptimizer::RefineAndValidate(legacy,oldPolicy);
        require(!old.accepted && old.reason=="marker_pose_displacement_too_large",
                "fixture did not expose lost provisional marker metadata: "+old.reason);
        const auto end=f.truth.back().cast<double>();
        const g2o::Sim3 seed(end.unit_quaternion(),end.translation(),1.0);
        const auto propose=[&]() {
            return loop ? MarkerGraphOptimizer::ProposeVisualLoop(&f.map,f.keyframes.back(),
                origin,seed,f.loopMatches) : MarkerGraphOptimizer::RefineMetricMap(&f.map);
        };
        const auto result=propose();
        require(result.accepted,"unvalidated marker placement blocked proposal: "+result.reason);
        require(result.keyframePoses.at(origin).matrix().isApprox(origin->GetPose().matrix(),1e-7f),
                "provisional marker proposal moved the world-origin camera");
        for(int j=0;j<4;++j) {
            require((result.staticTags.at(24)[j]-corner(f.originalMarkers.at(24),j)).norm()<1e-5f,
                    "provisional marker proposal damaged the correct origin marker");
            require((result.staticTags.at(25)[j]-corner(truth,j)).norm()<.001f,
                    "provisional marker world placement was not recovered");
        }
        require(std::abs((result.staticTags.at(25)[1]-result.staticTags.at(25)[0]).norm()-.07f)<1e-6f,
                "provisional marker proposal changed a physical marker size");
        f.assertUnchanged();

        // Metricization does not survey a marker's world placement. Even an
        // old registered landmark remains free when raw evidence supports it.
        f.map.SetMarkerScaleAnchorKFId(long(f.window().front()->mnId));
        const auto validated=propose();
        require(validated.accepted,
                "an old unsurveyed marker was incorrectly pinned: "+validated.reason);
        f.map.SetMarkerScaleAnchorKFId(long(origin->mnId));
        f.map.mbRigidMarkerLayout=true;
        require(!propose().accepted,"calibrated board layout was released as provisional");
        f.map.mbRigidMarkerLayout=false;

        // Missing scale-anchor metadata is not a new world-position prior.
        f.map.SetMarkerScaleAnchorKFId(long(KeyFrame::nNextId+100));
        const auto missing=propose();
        require(missing.accepted,"missing metadata pinned a measured rigid landmark: "+missing.reason);
        f.assertUnchanged();
    }
    std::cout << "{\"provisional_marker_final_and_loop\":true,\"registered_marker_still_unsurveyed\":true,\"board_protected\":true}" << std::endl;
}

static void observerRelativeGlobalMarkerCase()
{
    for(int mode=0;mode<5;++mode) {
        Fixture f(0);
        // Two stations connected by raw background tracks. The distant
        // station and its observing cameras have accumulated a coherent drift.
        for(std::size_t v=0;v<f.keyframes.size();++v) {
            KeyFrame* k=f.keyframes[v];
            for(std::size_t i=k->mvTagIds.size();i-- >0;)
                if((k->mvTagIds[i]==24 && v>2) ||
                   (k->mvTagIds[i]==25 && v<std::size_t(mode==1?10:9))) {
                    k->mvTagIds.erase(k->mvTagIds.begin()+i);
                    k->mvTagWorldPoints.erase(k->mvTagWorldPoints.begin()+i);
                    k->mvTagImagePoints.erase(k->mvTagImagePoints.begin()+i);
                    k->mvTagPointWeights.erase(k->mvTagPointWeights.begin()+i);
                }
            auto pose=k->GetPose();
            pose.translation().x()-=.12f*v/(Fixture::views-1);
            k->SetPose(pose);
            for(std::size_t i=0;i<k->mvTagIds.size();++i)
                if(k->mvTagIds[i]==25) k->mvTagWorldPoints[i].x()+=.12f;
        }
        if(mode==2) f.keyframes.back()->mvTagImagePoints.back().x+=100;
        MarkerGraphOptimizer::StagedBAInput input;
        input.keyframes=f.keyframes;input.fixedKeyframes.insert(f.keyframes.front());
        input.rigidMarkerLayout=mode==3;
        for(const auto& point:f.originalPoints) {
            const double t=point.first->GetReferenceKeyFrame()->mTimeStamp;
            input.pointPositions[point.first]=point.second+Eigen::Vector3f(.12*t/1.1,0,0);
        }
        if(mode==4) {
            const Sophus::SE3f gauge(Sophus::SO3f::exp(Eigen::Vector3f(.2f,-.1f,.3f)),
                                    Eigen::Vector3f(120,-42,3));
            for(KeyFrame* k:f.keyframes) {
                input.keyframePoses[k]=k->GetPose()*gauge.inverse();
                auto& corners=input.tagWorldCorners[k];
                for(const auto& point:k->mvTagWorldPoints) corners.push_back(gauge*point);
            }
            for(auto& point:input.pointPositions) point.second=gauge*point.second;
        }
        MarkerGraphOptimizer::Options oldPolicy;
        oldPolicy.legacyIndependentMarkerWorldPrior=true;
        const auto legacy=MarkerGraphOptimizer::RefineAndValidate(input,oldPolicy);
        const auto result=MarkerGraphOptimizer::RefineAndValidate(input,oldPolicy,nullptr,true);
        if(mode==0 || mode==4) {
            require(!legacy.accepted && legacy.reason=="marker_pose_displacement_too_large",
                    "coherent-drift fixture did not trigger legacy gate: "+legacy.reason);
            require(result.accepted,"coherent global correction rejected: "+result.reason);
            const auto fixed=mode==4?input.keyframePoses.at(f.keyframes.front()):f.keyframes.front()->GetPose();
            require(result.keyframePoses.at(f.keyframes.front()).matrix().isApprox(
                    fixed.matrix(),1e-7f),"global correction moved origin");
            require(result.after.tagRmsPx<.1 && result.after.backgroundRmsPx<.1,
                    "global correction did not recover raw geometry");
            require(std::abs((result.staticTags.at(25)[0]-result.staticTags.at(25)[1]).norm()-.07)<2e-5,
                    "global correction changed physical marker size");
        } else if(mode==3) {
            require(result.accepted==legacy.accepted && result.reason==legacy.reason,
                    "observer-relative policy changed rigid-board acceptance");
            for(int id:{24,25}) for(int j=0;j<4;++j) {
                const auto& xyz=f.originalMarkers.at(id);
                const Eigen::Vector3f before(xyz[3*j]+(id==25?.12f:0),xyz[3*j+1],xyz[3*j+2]);
                require((result.staticTags.at(id)[j]-before).norm()<1e-7f,
                        "observer-relative policy moved a rigid-board marker");
            }
        } else require(!result.accepted,"insufficient views or bad pixels bypassed validation");
    }
    std::cout << "{\"observer_relative_global_gate\":true,\"two_views_bad_pixels_rejected\":true,\"rigid_board_preserved\":true,\"world_gauge_invariant\":true}" << std::endl;
}

static void retainedSingleObserverCase()
{
    Fixture fixture(.10,false,1.f,-1,true);
    KeyFrame* observer=fixture.keyframes.back();
    MapPoint* point=fixture.ownedPoints.back().get();
    const Eigen::Vector3f position=point->GetWorldPos();
    const Eigen::Vector3f ray=observer->GetPose()*position;
    MarkerGraphOptimizer::StagedBAInput input;
    input.keyframes=fixture.keyframes;
    input.fixedKeyframes.insert(fixture.keyframes.front());
    input.pointPositions=fixture.originalPoints;
    const auto result=MarkerGraphOptimizer::RefineAndValidate(input);
    require(result.accepted,"retained singleton rejected: "+result.reason);
    require((result.keyframePoses.at(observer)*result.pointPositions.at(point)-ray).norm()<1e-5,
            "single-view propagation followed a non-retained reference");
    require((result.keyframePoses.at(observer).translation()-observer->GetPose().translation()).norm()>.001,
            "singleton test did not move observing camera");
    const auto reanchored=MarkerGraphOptimizer::Reanchor(&fixture.map,fixture.keyframes.front(),
        fixture.window(),fixture.measuredScale(),.01);
    require(reanchored.accepted,"reanchor lost retained singleton: "+reanchored.reason);
    const Eigen::Vector3f correctedRay=reanchored.keyframePoses.at(observer)*
        reanchored.pointPositions.at(point);
    require((correctedRay-ray*reanchored.replayScaleMultipliers.at(observer)).norm()<1e-5,
            "final reanchor propagation used an obsolete singleton reference");
    fixture.assertUnchanged();
    std::cout << "{\"retained_single_observer\":true}" << std::endl;
}

static void fixedBoundaryCase()
{
    Fixture fixture(.20);
    Frame frame(fixture.base);
    frame.mnId = Frame::nNextId++;
    frame.mTimeStamp = 2.0;
    frame.SetPose(Sophus::SE3f(Eigen::Matrix3f::Identity(), Eigen::Vector3f(-.02f, -.06f, 0)));
    for(int index = 0; index < 20; ++index) {
        const auto pixel = fixture.camera.project(frame.GetPose()*fixture.truePoints[index]);
        frame.mvKeysUn[index] = cv::KeyPoint(cv::Point2f(pixel.x(), pixel.y()), 1.0f);
    }
    fixture.ownedKeyframes.emplace_back(new KeyFrame(frame, &fixture.map, nullptr));
    KeyFrame* boundary = fixture.ownedKeyframes.back().get();
    fixture.map.AddKeyFrame(boundary);
    boundary->ChangeParent(fixture.keyframes.front());
    fixture.originalPoses.emplace(boundary, boundary->GetPose());
    for(int index = 0; index < 20; ++index) {
        MapPoint* point = fixture.ownedPoints[index].get();
        point->AddObservation(boundary, index);
        boundary->AddMapPoint(point, index);
    }
    boundary->UpdateConnections();
    MarkerGraphOptimizer::Options options;
    options.covisibleNeighbors = 0;
    const auto proposal = MarkerGraphOptimizer::Reanchor(&fixture.map, fixture.keyframes.front(),
        fixture.window(), fixture.measuredScale(), .01, options);
    fixture.assertUnchanged();
    require(proposal.accepted, "fixed boundary correction rejected: "+proposal.reason);
    require(proposal.replayScaleMultipliers.at(boundary) == 1.0f &&
            (proposal.keyframePoses.at(boundary).matrix()-fixture.originalPoses.at(boundary).matrix()).norm() < 1e-6,
            "an outside fixed observer changed its pose or replay unit");
    assertSingletonScaleContract(fixture, proposal);
    std::cout << "{\"fixed_boundary_unit\":1,\"singleton_contract\":true}" << std::endl;
}

static Eigen::Vector3f markerCorner(const std::vector<float>& marker, int corner)
{
    return Eigen::Vector3f(marker[3*corner], marker[3*corner+1], marker[3*corner+2]);
}

static void jointMarkerPoseCase(bool singleView=false)
{
    Fixture fixture(0);
    const auto truth=fixture.originalMarkers.at(25);
    constexpr float injected=0.015f;
    for(int corner=0;corner<4;++corner)
        fixture.map.mStaticTags.at(25)[3*corner]+=injected;
    for(KeyFrame* keyframe:fixture.keyframes)
        for(std::size_t index=0;index<keyframe->mvTagIds.size();++index)
            if(keyframe->mvTagIds[index]==25) keyframe->mvTagWorldPoints[index].x()+=injected;

    // Marker 24, observed by the fixed origin keyframe, defines the gauge.
    // Marker 25 is independently placed and first appears only in later
    // keyframes, so its pose must remain an optimizable graph variable.
    KeyFrame* origin=fixture.keyframes.front();
    for(KeyFrame* view:fixture.keyframes) {
        if(view!=origin && (!singleView || view==fixture.keyframes.back())) continue;
        for(std::size_t index=view->mvTagIds.size();index-- > 0;)
            if(view->mvTagIds[index]==25) {
                view->mvTagIds.erase(view->mvTagIds.begin()+index);
                view->mvTagWorldPoints.erase(view->mvTagWorldPoints.begin()+index);
                view->mvTagImagePoints.erase(view->mvTagImagePoints.begin()+index);
                view->mvTagPointWeights.erase(view->mvTagPointWeights.begin()+index);
            }
    }

    const auto liveTags=fixture.map.mStaticTags;
    MarkerGraphOptimizer::StagedBAInput input;
    input.keyframes=fixture.keyframes;
    input.fixedKeyframes.insert(fixture.keyframes.front());
    input.pointPositions=fixture.originalPoints;
    input.provisionalMarkerIds.insert(24);
    const auto wrongGauge=MarkerGraphOptimizer::RefineAndValidate(input);
    require(!wrongGauge.accepted && wrongGauge.reason=="fixed_gauge_marked_provisional",
            "a fixed old anchor was allowed to become provisional");
    input.provisionalMarkerIds.clear();
    const auto proposal=MarkerGraphOptimizer::RefineAndValidate(input);

    input.convergeOffline=true;
    const auto converged=MarkerGraphOptimizer::RefineAndValidate(input);
    require(converged.accepted,"offline converged marker BA rejected: "+converged.reason);
    require(converged.after.tagRmsPx<=proposal.after.tagRmsPx+1e-5,
            "offline continuation worsened marker residuals");
    require((converged.keyframePoses.at(origin).matrix()-origin->GetPose().matrix()).norm()<1e-6,
            "offline continuation moved the fixed world origin");

    require(fixture.map.mStaticTags==liveTags,"joint marker proposal wrote the live registry");
    require(proposal.accepted,"joint marker pose BA rejected: "+proposal.reason);
    require(std::find(proposal.optimizedMarkerIds.begin(),proposal.optimizedMarkerIds.end(),25)!=
            proposal.optimizedMarkerIds.end(),"multi-view marker did not become a graph variable");
    const auto& refined=proposal.staticTags.at(25);
    double beforeSquared=0,afterSquared=0;
    for(int corner=0;corner<4;++corner) {
        const Eigen::Vector3f target=markerCorner(truth,corner);
        beforeSquared+=(markerCorner(liveTags.at(25),corner)-target).squaredNorm();
        afterSquared+=(refined[corner]-target).squaredNorm();
    }
    const double before=std::sqrt(beforeSquared/4),after=std::sqrt(afterSquared/4);
    const double sideBefore=(markerCorner(truth,1)-markerCorner(truth,0)).norm();
    const double sideAfter=(refined[1]-refined[0]).norm();
    require(after<.25*before,"joint BA did not correct the marker pose");
    require(std::abs(sideAfter-sideBefore)<1e-6,"joint BA changed marker size");
    require(proposal.after.tagRmsPx<proposal.before.tagRmsPx*.25,
            "marker projection residual did not improve");
    std::cout << "{\"marker_pose_before_rms_m\":" << before
              << ",\"marker_pose_after_rms_m\":" << after
              << ",\"rigid_side_error_m\":" << std::abs(sideAfter-sideBefore)
              << ",\"optimized_marker_count\":" << proposal.optimizedMarkerIds.size()
              << "}" << std::endl;
}

static void cornerScaleCase()
{
    for(float scale:{1.12f,8.f,.125f}) {
        Fixture f(0);
        auto views=f.window();
        for(KeyFrame* k:views) {
            auto pose=k->GetPoseInverse();pose.translation()/=scale;k->SetPose(pose.inverse());
        }
        const auto estimate=MarkerGraphOptimizer::EstimateCornerScale(views);
        require(estimate.valid && estimate.markers==2 && std::abs(estimate.scale/scale-1)<.01,
                "raw multi-view corner size did not recover local scale");
        require(!MarkerGraphOptimizer::EstimateCornerScale({views[0],views[1]}).valid,
                "two views passed held-out scale validation");
        std::cout << "CORNER_SCALE_TEST true=" << scale << " measured=" << estimate.scale
                  << " sigma=" << estimate.sigma << " rms=" << estimate.rms << std::endl;
        if(scale>2 || scale<.5) {
            const auto saved=views.back()->mvTagImagePoints;
            for(std::size_t i=0;i<views.back()->mvTagIds.size();++i)
                if(views.back()->mvTagIds[i]==25) views.back()->mvTagImagePoints[i].y+=(i%2?40:-40);
            require(!MarkerGraphOptimizer::EstimateCornerScale(views).valid,
                    "one good marker bypassed the extreme-scale independent confirmation");
            views.back()->mvTagImagePoints=saved;
        }
        for(KeyFrame* k:views) k->SetPose(views.front()->GetPose());
        require(!MarkerGraphOptimizer::EstimateCornerScale(views).valid,
                "stationary cameras claimed observable scale");
    }
    Fixture far(0);
    const auto farViews=far.window();
    for(KeyFrame* k:farViews) {
        const auto index=std::find(far.keyframes.begin(),far.keyframes.end(),k)-far.keyframes.begin();
        for(std::size_t i=0;i<k->mvTagWorldPoints.size();++i) {
            k->mvTagWorldPoints[i].z()=8.f;
            const auto pixel=far.camera.project(far.truth[index]*k->mvTagWorldPoints[i]);
            k->mvTagImagePoints[i]=cv::Point2f(pixel.x(),pixel.y());
        }
    }
    // Both markers have exact pixels and an 8 cm metric baseline, but at
    // eight metres their corner rays lack the depth evidence for a large
    // automatic repair. The moderate branch retains its previous behavior.
    for(float scale:{1.12f,8.f,.125f}) {
        for(KeyFrame* k:farViews) {
            const auto index=std::find(far.keyframes.begin(),far.keyframes.end(),k)-far.keyframes.begin();
            auto pose=far.truth[index].inverse();pose.translation()/=scale;k->SetPose(pose.inverse());
        }
        const auto estimate=MarkerGraphOptimizer::EstimateCornerScale(farViews);
        if(scale==1.12f)
            require(estimate.valid && estimate.markers==2 && std::abs(estimate.scale/scale-1)<.01,
                    "far-marker fixture failed the unchanged moderate-scale fit");
        else require(!estimate.valid,"two distant markers bypassed large-repair parallax validation");
    }
}

static void extremeNewStationCase(int iterations=15,float ratio=8.f)
{
    Fixture f(0);
    auto b=f.window();
    // A retains its observed metric marker. B contains two NEW physical IDs,
    // provisionally registered by the already shrunken visual camera.
    f.map.mStaticTags[26]=f.map.mStaticTags.at(24);
    for(int j=0;j<4;++j) f.map.mStaticTags[26][3*j+1]+=.15f;
    const Eigen::Vector3f delta=f.truth[Fixture::views-3].inverse().translation()*(1.f/ratio-1.f);
    for(KeyFrame* k:f.keyframes) {
        auto pose=k->GetPoseInverse();pose.translation()/=ratio;k->SetPose(pose.inverse());
        if(k==f.keyframes.front()) {
            for(std::size_t i=k->mvTagIds.size();i-- >0;)
                if(k->mvTagIds[i]==25) {
                    k->mvTagIds.erase(k->mvTagIds.begin()+i);
                    k->mvTagWorldPoints.erase(k->mvTagWorldPoints.begin()+i);
                    k->mvTagImagePoints.erase(k->mvTagImagePoints.begin()+i);
                    k->mvTagPointWeights.erase(k->mvTagPointWeights.begin()+i);
                }
            continue;
        }
        const auto index=std::find(f.keyframes.begin(),f.keyframes.end(),k)-f.keyframes.begin();
        for(std::size_t i=0;i<k->mvTagIds.size();++i) {
            if(k->mvTagIds[i]==24) {
                k->mvTagIds[i]=26;k->mvTagWorldPoints[i].y()+=.15f;
                const auto uv=f.camera.project(f.truth[index]*k->mvTagWorldPoints[i]);
                k->mvTagImagePoints[i]=cv::Point2f(uv.x(),uv.y());
            }
            k->mvTagWorldPoints[i]+=delta;
        }
    }
    for(int id:{25,26}) for(int j=0;j<4;++j) for(int axis=0;axis<3;++axis)
        f.map.mStaticTags[id][3*j+axis]+=delta[axis];
    for(auto& point:f.ownedPoints) point->SetWorldPos(point->GetWorldPos()/ratio);
    const auto oldTags=f.map.mStaticTags;
    MarkerGraphOptimizer::Options options;options.baIterations=iterations;
    if(ratio>2 || ratio<.5) {
        auto legacy=options;legacy.allowLargeCornerScaleRepair=false;
        const auto guarded=MarkerGraphOptimizer::Reanchor(&f.map,f.keyframes.front(),b,1.,.1,legacy);
        require(!guarded.accepted && guarded.reason=="corner_scale_outside_safe_range",
                "legacy envelope did not reproduce the rejected scale repair");
    }
    const auto oldPose=f.keyframes.front()->GetPose();
    const auto oldPoint=f.ownedPoints.front()->GetWorldPos();
    const auto p=MarkerGraphOptimizer::Reanchor(&f.map,f.keyframes.front(),b,1.,.1,options);
    std::cout << "EXTREME_TRIAL tag=" << p.after.tagRmsPx << " bg=" << p.after.backgroundRmsPx
              << " camera=" << f.cameraRms(p.keyframePoses) << std::endl;
    require(f.map.mStaticTags==oldTags,"extreme repair mutated the live marker registry");
    require(f.keyframes.front()->GetPose().matrix().isApprox(oldPose.matrix(),1e-7f) &&
            f.ownedPoints.front()->GetWorldPos().isApprox(oldPoint,1e-7f),
            "extreme proposal mutated live camera or point geometry");
    require(p.accepted,"extreme station reconstruction rejected: "+p.reason);
    require(std::abs(p.cornerScale/ratio-1)<.01,"station repair did not use raw corner scale");
    require(f.cameraRms(p.keyframePoses)<.01,"extreme repair left the old camera path shrunken");
    require(p.keyframePoses.at(f.keyframes.front()).matrix().isApprox(oldPose.matrix(),1e-7f),
            "extreme repair moved the old fixed anchor");
    for(int id:{24,25,26}) {
        const auto& marker=p.staticTags.at(id);
        require(std::abs((marker[0]-marker[1]).norm()-.07f)<1e-5,
                "extreme repair resized a physical marker");
    }
    std::cout << "EXTREME_REANCHOR rms_m=" << f.cameraRms(p.keyframePoses)
              << " tag_px=" << p.after.tagRmsPx << " background_px=" << p.after.backgroundRmsPx << std::endl;
}

static void covisibleSim3OwnerCase(float angle=.1f)
{
    Map map;
    Pinhole camera({500,500,320,240});
    ORBextractor extractor(800,1.2f,8,20,7);
    Frame base=makeBase(extractor,camera);
    std::vector<std::unique_ptr<KeyFrame>> frames;
    for(int i=0;i<3;++i) {
        frames.emplace_back(new KeyFrame(base,&map,nullptr));
        map.AddKeyFrame(frames.back().get());
    }
    std::vector<std::unique_ptr<MapPoint>> points;
    std::vector<MapPoint*> matches(base.N,nullptr);
    std::vector<KeyFrame*> owners(base.N,nullptr);
    for(int i=0;i<40;++i) {
        const Eigen::Vector3f position=camera.unprojectEig(base.mvKeysUn[i].pt)*(1.2f+.03f*i);
        for(int side: {0,2}) {
            const Eigen::Vector3f local = side==0 ? position :
                (Sophus::SO3f::exp(Eigen::Vector3f(0,angle,0))*position).eval();
            points.emplace_back(new MapPoint(local,frames[side].get(),&map));
            points.back()->AddObservation(frames[side].get(),i);
            frames[side]->AddMapPoint(points.back().get(),i);
        }
        matches[i]=points.back().get(); owners[i]=frames[2].get();
    }
    // Reference frame 1 does not itself observe any candidate points.
    Sim3Solver solver(frames[0].get(),frames[1].get(),matches,true,owners);
    solver.SetRansacParameters(.99,15,300);
    bool done=false,converged=false;int inliers=0;std::vector<bool> mask;
    while(!done && !converged) solver.iterate(20,done,mask,inliers,converged);
    require(converged && inliers==40,"covisible observation owners lost in Sim3 solver");
    require(std::abs(solver.GetEstimatedScale()-1.f)<1e-5,"owner fix changed metric scale");
    g2o::Sim3 estimate(solver.GetEstimatedRotation().cast<double>(),
        solver.GetEstimatedTranslation().cast<double>(),1.0);
    Eigen::Matrix<double,7,7> hessian;
    const int refined=Optimizer::OptimizeSim3(frames[0].get(),frames[1].get(),
        matches,estimate,10,true,hessian,true);
    require(refined==40,"covisible-only Sim3 pixel observations rejected correct geometry");
    std::cout << "{\"covisible_owner_inliers\":" << inliers << "}" << std::endl;
}

static void postLoopRematchCase()
{
    Map map;
    Pinhole camera({500,500,320,240});
    ORBextractor extractor(800,1.2f,8,20,7);
    Frame base=makeBase(extractor,camera);
    std::vector<std::unique_ptr<KeyFrame>> frames;
    for(int i=0;i<4;++i) {
        Frame frame=base;
        frame.mTimeStamp=i<2?i:8+i;
        frames.emplace_back(new KeyFrame(frame,&map,nullptr));
        map.AddKeyFrame(frames.back().get());
    }
    std::vector<std::unique_ptr<MapPoint>> points;
    constexpr int count=80;
    for(int pass=0;pass<2;++pass) for(int i=0;i<count;++i) {
        const auto pixel=base.mvKeysUn[i].pt;
        const Eigen::Vector3f position=camera.unprojectEig(pixel)*1.5f;
        points.emplace_back(new MapPoint(position,frames[2*pass].get(),&map));
        MapPoint* point=points.back().get();
        map.AddMapPoint(point);
        for(int j=2*pass;j<2*pass+2;++j) {
            point->AddObservation(frames[j].get(),i);
            frames[j]->AddMapPoint(point,i);
        }
        point->ComputeDistinctiveDescriptors();
        point->UpdateNormalAndDepth();
    }
    for(auto& kf:frames) kf->UpdateConnections();
    MapPoint* a=points[0].get(); MapPoint* b=points[count].get();
    require(PostLoopRematch::ConsistentDuplicate(a,b),"true multi-view duplicate rejected");
    const auto original=b->GetWorldPos();
    b->SetWorldPos(original+Eigen::Vector3f(.1f,0,0));
    require(!PostLoopRematch::ConsistentDuplicate(a,b),"wrong geometry accepted as duplicate");
    b->SetWorldPos(original);
    require(!PostLoopRematch::ConsistentDuplicate(a,points[1].get()),"shared image features fused");
    const auto before=map.MapPointsInMap();
    PostLoopRematch::Run(&map,frames.front().get(),frames.back().get(),false);
    require(map.MapPointsInMap()==before,"rematch audit mutated map");
    PostLoopRematch::Run(&map,frames.front().get(),frames.back().get(),true);
    const auto after=map.MapPointsInMap();
    require(after<before,"known duplicate constellation failed to fuse");
    require(after>=count,"distinct physical landmarks over-fused");
    for(auto& kf:frames)
        require(kf->GetPose().matrix().isApprox(Sophus::SE3f().matrix(),1e-7f),
                "matching changed the fixed camera geometry");
    std::cout << "{\"post_loop_points_before\":" << before
              << ",\"post_loop_points_after\":" << after << "}" << std::endl;
}

int main(int argc,char** argv)
{
    try {
        if(argc==2 && std::string(argv[1])=="--triangulation-unique-only") {
            triangulationUniqueTargetCase();return 0;
        }
        componentOriginConsistencyCase();
        require(BackgroundResidualFrameConsistent(1.35,1.436),"healthy high-octave residual rejected");
        require(!BackgroundResidualFrameConsistent(2.526,21.89),"large mismatch passed normalized gate");
        require(!BackgroundResidualFrameConsistent(2.9,3.1),"new normalized damage accepted");
        require(BackgroundResidualFrameConsistent(4.,4.1),"pre-existing unchanged damage rejected");
        require(!BackgroundResidualFrameConsistent(4.,4.6),"worsened existing damage accepted");
        require(!BackgroundResidualFrameConsistent(1.,std::numeric_limits<double>::quiet_NaN()),"NaN passed residual gate");
        if(argc==2 && std::string(argv[1])=="--component-origin-only") return 0;
        if(argc==2 && std::string(argv[1])=="--background-pixel-only") {
            verifiedBackgroundPixelRetryCase(); return 0;
        }
        if(argc==2 && std::string(argv[1])=="--final-background-only") {
            finalBackgroundPolicyCase();return 0;
        }
        if(argc==2 && std::string(argv[1])=="--marker-gauge-only") {
            independentGaugeMarkerCase(); unsurveyedScaleObservabilityCase(); return 0;
        }
        if(argc==2 && std::string(argv[1])=="--provisional-marker-only") {
            provisionalMarkerProposalCase(); return 0;
        }
        if(argc==2 && std::string(argv[1])=="--observer-relative-marker-only") {
            observerRelativeGlobalMarkerCase(); return 0;
        }
        if(argc==2 && std::string(argv[1])=="--known-marker-loop-only") {
            knownMarkerLoopScaleCase();stagedVisualLoopCase();return 0;
        }
        if(argc==2 && std::string(argv[1])=="--cheirality-repair-only") {
            backgroundCheiralityRepairCase();return 0;
        }
        if(argc==2 && std::string(argv[1])=="--corner-scale-only") {
            cornerScaleCase(); return 0;
        }
        if(argc==2 && std::string(argv[1])=="--graph-order-only") {
            graphInputOrderCase(); graphAllocationOrderCase(); erasedReferenceOrderCase(); return 0;
        }
        if((argc==2 || argc==3) && std::string(argv[1])=="--extreme-new-station") {
            extremeNewStationCase(argc==3?std::stoi(argv[2]):15);return 0;
        }
        if((argc==2 || argc==3) && std::string(argv[1])=="--new-station") {
            const int iterations=argc==3?std::stoi(argv[2]):15;
            extremeNewStationCase(iterations,1.2f);extremeNewStationCase(iterations,.8f);return 0;
        }
        double localDepthMedian=1.0;
        require(LocalMetricDepthChangeAccepted(std::vector<double>(100,1.01),localDepthMedian),
                "small local metric depth refinement rejected");
        require(!LocalMetricDepthChangeAccepted(std::vector<double>(100,.1),localDepthMedian),
                "same-frame tenfold depth collapse accepted");
        require(!LocalMetricDepthChangeAccepted(std::vector<double>(100,10.),localDepthMedian),
                "same-frame tenfold depth expansion accepted");
        std::vector<double> sparseOutliers(100,1.0);
        std::fill(sparseOutliers.begin(),sparseOutliers.begin()+10,.01);
        require(LocalMetricDepthChangeAccepted(sparseOutliers,localDepthMedian),
                "few unstable points vetoed an otherwise stable local BA");
        const bool compareWeak=argc==7 && std::string(argv[1])=="--atlas-weak-compare";
        const bool inspectAtlas=argc==4 && std::string(argv[1])=="--atlas-inspect";
        const bool landmarkCompare=argc==4 && std::string(argv[1])=="--atlas-landmark-compare";
        const bool globalCompare=argc==4 && std::string(argv[1])=="--atlas-global-compare";
        const bool backgroundCompare=argc==4 && std::string(argv[1])=="--atlas-background-compare";
        const bool frozenLoop=argc==7 && (std::string(argv[1])=="--atlas-loop-freeze" ||
                                         std::string(argv[1])=="--atlas-loop-compare");
        if(landmarkCompare || backgroundCompare || globalCompare || inspectAtlas || compareWeak || frozenLoop || ((argc==7 || argc==8) && std::string(argv[1])=="--atlas-reanchor")) {
            // Optional output saves a separately validated Atlas copy only.
            if(argc==8) require(!std::ifstream(argv[7]).good(),"refusing to overwrite output Atlas");
            ORBVocabulary vocabulary;
            require(vocabulary.loadFromTextFile(argv[3]),"vocabulary load failed");
            KeyFrameDatabase database(vocabulary);
            std::ifstream input(argv[2],std::ios::binary);
            require(input.good(),"atlas missing");
            Atlas* atlas=nullptr;
            std::string name,checksum,format;
            { boost::archive::binary_iarchive archive(input);
              archive >> name >> checksum >> format;
              require(format=="marker-orb-atlas/v1" || format=="marker-orb-atlas/v2","unsupported atlas");
              archive >> atlas; }
            atlas->SetKeyFrameDababase(&database); atlas->SetORBVocabulary(&vocabulary); atlas->PostLoad();
            Map* map=nullptr;
            for(Map* candidate:atlas->GetAllMaps())
                if(!map || candidate->KeyFramesInMap()>map->KeyFramesInMap()) map=candidate;
            require(map!=nullptr,"empty atlas");
            if(landmarkCompare || globalCompare || backgroundCompare) {
                for(bool relative:{false,true}) {
                    MarkerGraphOptimizer::Options policy;
                    policy.legacyIndependentMarkerWorldPrior=landmarkCompare && !relative;
                    const auto p=landmarkCompare ? MarkerGraphOptimizer::RefineMetricMap(map,policy) : backgroundCompare
                        ?MarkerGraphOptimizer::RefineMetricMap(map,MarkerGraphOptimizer::Options(),true,relative)
                        :MarkerGraphOptimizer::RefineMetricMap(map,MarkerGraphOptimizer::Options(),relative);
                    std::cout << std::setprecision(12) << (landmarkCompare?"LANDMARK_COMPARE free=":backgroundCompare?"BACKGROUND_COMPARE enabled=":"GLOBAL_COMPARE relative=") << relative
                              << " accepted=" << p.accepted << " reason=" << p.reason
                              << " tag_before=" << p.before.tagRmsPx << " tag_after=" << p.after.tagRmsPx
                              << " bg_before=" << p.before.backgroundRmsPx << " bg_after=" << p.after.backgroundRmsPx
                              << " background_observations=" << p.after.backgroundObservations
                              << " excluded_background_frames=" << p.excludedBackgroundKeyFrameIds.size()
                              << " positive_depth=" << p.after.positiveDepthFraction << std::endl;
                    if(relative) dumpReanchorProposal(map,map->GetOriginKF(),{},p,true,true);
                }
                // No commit, serialization or observation changes.
                return 0;
            }
            if(frozenLoop) {
                frozenAtlasLoop(map,atlas,&database,&vocabulary,argc,argv);
                return 0;
            }
            double unitMin=1e30,unitMax=0,tagSquared=0;std::size_t corners=0,behind=0;
            for(KeyFrame* k:map->GetAllKeyFrames()) {
                unitMin=std::min(unitMin,double(k->mReplayUnitScale));
                unitMax=std::max(unitMax,double(k->mReplayUnitScale));
                if(!k->mbTagObservationActive && !inspectAtlas) continue;
                for(std::size_t i=0;i<k->mvTagWorldPoints.size();++i) {
                    const Eigen::Vector3f p=k->GetPose()*k->mvTagWorldPoints[i];
                    if(p.z()<=0) {++behind;continue;}
                    const auto pixel=k->mpCamera->project(p);
                    const auto measured=k->mvTagImagePoints[i];
                    tagSquared+=(pixel-Eigen::Vector2f(measured.x,measured.y)).squaredNorm();++corners;
                    if(inspectAtlas)
                        std::cout << "ATLAS_CORNER kf=" << k->mnId << " active=" << k->mbTagObservationActive << " frame=" << k->mnFrameId
                                  << " time=" << k->mTimeStamp << " id=" << k->mvTagIds[i]
                                  << " weight=" << (k->mvTagPointWeights.empty()?1.f:k->mvTagPointWeights[i])
                                  << " xyz=" << k->mvTagWorldPoints[i].transpose()
                                  << " measured=" << measured.x << "," << measured.y
                                  << " projected=" << pixel.transpose() << std::endl;
                }
            }
            std::cout << "ATLAS_GEOMETRY unit_min=" << unitMin << " unit_max=" << unitMax
                      << " marker_rms=" << std::sqrt(tagSquared/std::max<std::size_t>(1,corners))
                      << " marker_corners=" << corners << " behind=" << behind << std::endl;
            if(inspectAtlas) return 0;
            KeyFrame* a=nullptr; std::vector<KeyFrame*> b;
            // Frozen-Atlas diagnostics may target a recorded historical B
            // window explicitly. This selects existing observations only;
            // it does not rewind the Atlas or remove later boundary factors.
            const char* explicitA=std::getenv("MARKER_REANCHOR_A_KF");
            const char* explicitB=std::getenv("MARKER_REANCHOR_B_KFS");
            require(bool(explicitA)==bool(explicitB),"provide both explicit A and B diagnostic IDs");
            std::set<unsigned long> requestedB;
            unsigned long requestedA=0;
            if(explicitA) {
                requestedA=std::stoul(explicitA);
                std::istringstream inputIds(explicitB);
                std::string id;
                while(std::getline(inputIds,id,',')) {
                    std::size_t consumed=0;
                    const auto value=std::stoul(id,&consumed);
                    require(consumed==id.size() && requestedB.insert(value).second,
                            "invalid or duplicate explicit B diagnostic ID");
                }
                require(requestedB.size()>=2 && !requestedB.count(requestedA),
                        "explicit B needs independent keyframes different from A");
            }
            for(KeyFrame* k:map->GetAllKeyFrames()) {
                if(explicitA ? k->mnId==requestedA : long(k->mnId)==map->GetMarkerScaleAnchorKFId()) a=k;
                if(!k->isBad() && k->mbTagObservationActive && k->mTagObservationConfidence>=.35f &&
                   (explicitB ? requestedB.count(k->mnId)>0 : k->mTimeStamp>std::stod(argv[6]))) b.push_back(k);
            }
            b.erase(std::remove(b.begin(),b.end(),a),b.end());
            std::sort(b.begin(),b.end(),KeyFrame::lId);
            if(explicitB) require(b.size()==requestedB.size(),"explicit B contains missing or inactive keyframes");
            else if(b.size()>3) b.erase(b.begin(),b.end()-3);
            require(a && b.size()>=2,"missing interval anchors");
            std::cout << "ATLAS_REANCHOR_INPUT anchor=" << a->mnId << " B=";
            for(std::size_t i=0;i<b.size();++i) std::cout << (i?",":"") << b[i]->mnId;
            std::cout << " explicit_ids=" << bool(explicitB)
                      << " later_boundary_observations_retained=1" << std::endl;
            if(compareWeak) {
                // Both proposals see exactly the same loaded graph. This path
                // never commits, saves, or feeds one solve's result to another.
                MarkerGraphOptimizer::PoseMap originalPoses;
                MarkerGraphOptimizer::PointMap originalPoints;
                MarkerGraphOptimizer::TagCornerMap originalCorners;
                MarkerGraphOptimizer::ScaleMap originalUnits;
                const auto originalMarkers=map->mStaticTags;
                const auto originalRevision=map->mnRevision;
                const auto originalAnchor=map->GetMarkerScaleAnchorKFId();
                const auto originalKeyframeCount=map->KeyFramesInMap();
                const auto originalPointCount=map->MapPointsInMap();
                KeyFrame* diagnostic=nullptr;
                for(KeyFrame* k:map->GetAllKeyFrames()) {
                    originalPoses.emplace(k,k->GetPose());
                    originalCorners.emplace(k,k->mvTagWorldPoints);
                    originalUnits.emplace(k,k->mReplayUnitScale);
                    if(k->mnId==33) diagnostic=k;
                }
                for(MapPoint* point:map->GetAllMapPoints())
                    originalPoints.emplace(point,point->GetWorldPos());
                const auto unchanged=[&]() {
                    require(map->mStaticTags==originalMarkers && map->mnRevision==originalRevision &&
                            map->GetMarkerScaleAnchorKFId()==originalAnchor &&
                            map->KeyFramesInMap()==originalKeyframeCount && map->MapPointsInMap()==originalPointCount,
                            "read-only comparison changed live map state");
                    for(const auto& pose:originalPoses) {
                        require((pose.first->GetPose().matrix()-pose.second.matrix()).norm()==0 &&
                                pose.first->mReplayUnitScale==originalUnits.at(pose.first),
                                "read-only comparison changed live keyframe pose/units");
                        const auto& old=originalCorners.at(pose.first);
                        require(pose.first->mvTagWorldPoints.size()==old.size(),"comparison changed tag corner count");
                        for(std::size_t i=0;i<old.size();++i)
                            require((pose.first->mvTagWorldPoints[i]-old[i]).norm()==0,
                                    "read-only comparison changed stored tag coordinates");
                    }
                    for(const auto& point:originalPoints)
                        require((point.first->GetWorldPos()-point.second).norm()==0,
                                "read-only comparison changed live map points");
                };
                if(diagnostic) for(std::size_t i=0;i<diagnostic->mvTagIds.size();++i) {
                    if(diagnostic->mvTagIds[i]!=45 || diagnostic->mvTagPointWeights.empty() ||
                       diagnostic->mvTagPointWeights[i]>=.99f) continue;
                    const auto marker=map->mStaticTags.find(45);
                    if(marker==map->mStaticTags.end() || marker->second.size()!=12) continue;
                    const Eigen::Vector3f stored=diagnostic->mvTagWorldPoints[i];
                    Eigen::Vector3f canonical=stored;
                    double closest=1e30; int cornerIndex=-1;
                    for(int corner=0;corner<4;++corner) {
                        const Eigen::Vector3f point(marker->second[3*corner],marker->second[3*corner+1],marker->second[3*corner+2]);
                        const double distance=(stored-point).norm();
                        if(distance<closest) {closest=distance; canonical=point; cornerIndex=corner;}
                    }
                    const auto measured=diagnostic->mvTagImagePoints[i];
                    const Eigen::Vector2f pixel(measured.x,measured.y);
                    const Eigen::Vector3f storedCamera=diagnostic->GetPose()*stored;
                    const Eigen::Vector3f canonicalCamera=diagnostic->GetPose()*canonical;
                    const double storedError=storedCamera.z()>0?
                        (diagnostic->mpCamera->project(storedCamera)-pixel).norm():-1;
                    const double canonicalError=canonicalCamera.z()>0?
                        (diagnostic->mpCamera->project(canonicalCamera)-pixel).norm():-1;
                    std::cout << "ATLAS_WEAK_STORED keyframe=33 marker=45 observation=" << i
                              << " nearest_corner=" << cornerIndex << " offset_m=" << closest
                              << " stored_pixel_error=" << storedError << " canonical_pixel_error=" << canonicalError
                              << " weight=" << diagnostic->mvTagPointWeights[i] << std::endl;
                }
                const auto report=[&](const std::string& mode,const MarkerGraphOptimizer::Proposal& proposal) {
                    const auto group=std::make_pair(diagnostic,45);
                    const auto groupBefore=proposal.before.tagRmsByKeyframeMarker.find(group);
                    const auto groupAfter=proposal.after.tagRmsByKeyframeMarker.find(group);
                    const auto firstBScale=proposal.replayScaleMultipliers.find(b.front());
                    const auto lastBScale=proposal.replayScaleMultipliers.find(b.back());
                    double low=1e30,high=0;
                    for(const auto& multiplier:proposal.replayScaleMultipliers) {
                        low=std::min(low,double(multiplier.second));
                        high=std::max(high,double(multiplier.second));
                    }
                    std::cout << "ATLAS_WEAK_COMPARE mode=" << mode << " accepted=" << proposal.accepted
                              << " reason=" << proposal.reason << " anchor=" << a->mnId
                              << " B=" << b.front()->mnId << ":" << b.back()->mnId
                              << " input_scale_cue=" << std::stod(argv[4])
                              << " group_33_45_before=" << (groupBefore==proposal.before.tagRmsByKeyframeMarker.end()?-1:groupBefore->second)
                              << " group_33_45_after=" << (groupAfter==proposal.after.tagRmsByKeyframeMarker.end()?-1:groupAfter->second)
                              << " tag_before=" << proposal.before.tagRmsPx << " tag_after=" << proposal.after.tagRmsPx
                              << " background_before=" << proposal.before.backgroundRmsPx
                              << " background_after=" << proposal.after.backgroundRmsPx
                              << " replay_scale_kind=" << (proposal.accepted?"validated_final":"unvalidated_staged")
                              << " replay_scale_min=" << (proposal.replayScaleMultipliers.empty()?-1:low)
                              << " replay_scale_max=" << (proposal.replayScaleMultipliers.empty()?-1:high)
                              << " B_first_replay_scale=" << (firstBScale==proposal.replayScaleMultipliers.end()?-1:firstBScale->second)
                              << " B_last_replay_scale=" << (lastBScale==proposal.replayScaleMultipliers.end()?-1:lastBScale->second)
                              << " excluded_groups=";
                    bool first=true;
                    for(const auto& excluded:proposal.excludedTagGroupIds) {
                        if(!first) std::cout << ',';
                        first=false;
                        std::cout << excluded.first << ':' << excluded.second;
                    }
                    std::cout << " live_geometry_unchanged=1" << std::endl;
                };
                MarkerGraphOptimizer::Options legacy;
                legacy.canonicalizeWeakMarkerCorners=false;
                legacy.retryIsolatedMarkerGroups=false;
                const auto oldProposal=MarkerGraphOptimizer::Reanchor(map,a,b,std::stod(argv[4]),std::stod(argv[5]),legacy);
                unchanged(); report("legacy",oldProposal);
                const auto fixedProposal=MarkerGraphOptimizer::Reanchor(map,a,b,std::stod(argv[4]),std::stod(argv[5]));
                unchanged(); report("fixed",fixedProposal);
                return 0;
            }
            MarkerGraphOptimizer::Options reanchorOptions;
            const char* disableRepair=std::getenv("MARKER_REANCHOR_DISABLE_CHEIRALITY_REPAIR");
            if(disableRepair && std::string(disableRepair)=="1") reanchorOptions.repairBackgroundCheirality=false;
            const auto p=MarkerGraphOptimizer::Reanchor(map,a,b,std::stod(argv[4]),std::stod(argv[5]),reanchorOptions);
            dumpReanchorProposal(map,a,b,p);
            std::cout << "ATLAS_REANCHOR accepted=" << p.accepted << " reason=" << p.reason
                      << " anchor=" << a->mnId << " B=" << b.front()->mnId << ":" << b.back()->mnId
                      << " affected=" << p.affectedKeyFrameIds.size()
                      << " background_before=" << p.before.backgroundRmsPx
                      << " background_after=" << p.after.backgroundRmsPx << std::endl;
            if(argc==8) {
                require(p.accepted,"rejected proposal is never saved");
                MarkerGraphEvent event;
                event.candidateFrameId=b.back()->mnFrameId;
                event.candidateTimestamp=b.back()->mTimeStamp;
                for(KeyFrame* k:map->GetAllKeyFrames()) if(k && !k->isBad()) {
                    event.timestamp=std::max(event.timestamp,k->mTimeStamp);
                    event.frameId=std::max(event.frameId,long(k->mnFrameId));
                }
                event.scale=std::stod(argv[4]); event.sigma=std::stod(argv[5]);
                {
                    std::unique_lock<std::mutex> gate(atlas->mMutexPoseGraphCorrection);
                    std::unique_lock<std::mutex> lock(map->mMutexMapUpdate);
                    require(MarkerGraphCoordinator::CommitScale(*atlas,map,p,b.back(),event),
                            "production CommitScale rejected proposal");
                }
                double low=1e30,high=0;
                for(const auto& multiplier:p.replayScaleMultipliers) {
                    low=std::min(low,double(multiplier.second));
                    high=std::max(high,double(multiplier.second));
                }
                std::cout << "ATLAS_REANCHOR_COMMIT accepted=1 revision=" << map->mnRevision
                    << " multiplier_min=" << low << " multiplier_max=" << high
                    << " marker_before=" << p.before.tagRmsPx
                    << " marker_after=" << p.after.tagRmsPx << std::endl;
                atlas->PreSave();
                std::ofstream output(argv[7],std::ios::binary);
                require(output.good(),"cannot open output Atlas");
                { boost::archive::binary_oarchive archive(output);
                  archive << name << checksum << format << atlas; }
            }
            return 0;
        }
        if(argc==2 && std::string(argv[1])=="--post-loop-only") {
            postLoopRematchCase();
            return 0;
        }
        if(argc==2 && std::string(argv[1])=="--weak-marker-only") {
            weakMarkerCanonicalizationCase();
            weakMarkerAssociationGuardCase();
            weakMarkerReanchorRetryCase();
            weakMarkerRetrySafetyCase();
            return 0;
        }
        triangulationUniqueTargetCase();
        correctionCase(.10);
        correctionCase(.20);
        initialMetricizationCase();
        rejectionCases();
        weakMarkerCanonicalizationCase();
        weakMarkerAssociationGuardCase();
        weakMarkerReanchorRetryCase();
        weakMarkerRetrySafetyCase();
        backgroundDilutionCase();
        backgroundCheiralityRepairCase();
        independentGaugeMarkerCase();
        unsurveyedScaleObservabilityCase();
        verifiedBackgroundPixelRetryCase();
        finalBackgroundPolicyCase();
        committedBoundaryAdmissionCase();
        contradictoryScaleCueCase();
        uncertainScaleCueCase();
        weakUnitSeedCase();
        revisitedAnchorIntervalCase();
        unobservableScaleCase();
        markerOnlyInteriorCase();
        stagedVisualLoopCase();
        knownMarkerLoopScaleCase();
        provisionalMarkerProposalCase();
        observerRelativeGlobalMarkerCase();
        retainedSingleObserverCase();
        fixedBoundaryCase();
        jointMarkerPoseCase();
        jointMarkerPoseCase(true);
        cornerScaleCase();
        postLoopRematchCase();
        covisibleSim3OwnerCase();
        covisibleSim3OwnerCase(0.f);
    } catch(const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
    return 0;
}
