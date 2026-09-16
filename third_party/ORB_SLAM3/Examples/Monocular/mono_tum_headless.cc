/** Headless TUM-sequence runner used by the local offline action exporter. */

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <System.h>
#include "GeometricDynamicMask.h"
#include "MarkerComponentRegistration.h"
#include "MarkerComponentWindow.h"

using namespace std;

struct TagObservation
{
    bool valid = false;
    bool candidate = false;
    float confidence = 0.0f;
    Sophus::SE3f Twc;
    vector<Eigen::Vector3f> worldPoints;
    vector<cv::Point2f> imagePoints;
    vector<float> pointWeights;
    vector<int> tagIds;
    bool partial=false;
    int trackedCorners=0;
    float trackAgeS=0;
    string component;
};

#include "CandidateRegistration.h"

static void LoadImages(const string &file, vector<string> &filenames,
                       vector<double> &timestamps)
{
    ifstream stream(file.c_str());
    string line;
    while(getline(stream, line))
    {
        if(line.empty() || line[0] == '#')
            continue;
        stringstream parser(line);
        double timestamp;
        string filename;
        if(parser >> timestamp >> filename)
        {
            timestamps.push_back(timestamp);
            filenames.push_back(filename);
        }
    }
}

static void LoadTagObservations(
    const string &file, vector<TagObservation> &observations)
{
    ifstream stream(file.c_str());
    string line;
    while(getline(stream, line))
    {
        if(line.empty() || line[0] == '#')
            continue;
        stringstream parser(line);
        double timestamp;
        int valid;
        TagObservation observation;
        if(!(parser >> timestamp >> valid))
            continue;
        if(!valid)
        {
            observations.push_back(observation);
            continue;
        }
        float tx, ty, tz, qx, qy, qz, qw;
        size_t count;
        if(!(parser >> observation.confidence >> tx >> ty >> tz
             >> qx >> qy >> qz >> qw >> count))
        {
            observations.push_back(TagObservation());
            continue;
        }
        Eigen::Quaternionf quaternion(qw, qx, qy, qz);
        if(quaternion.norm() < 1e-6f)
        {
            observations.push_back(TagObservation());
            continue;
        }
        observation.Twc = Sophus::SE3f(
            quaternion.normalized().toRotationMatrix(),
            Eigen::Vector3f(tx, ty, tz));
        observation.worldPoints.reserve(count);
        observation.imagePoints.reserve(count);
        for(size_t index = 0; index < count; ++index)
        {
            float x, y, z, u, v;
            if(!(parser >> x >> y >> z >> u >> v))
                break;
            observation.worldPoints.emplace_back(x, y, z);
            observation.imagePoints.emplace_back(u, v);
        }
        observation.valid =
            observation.worldPoints.size() == count && count >= 3;
        // Optional labelled suffixes keep legacy five-value corner files readable.
        string suffix;
        while(parser >> suffix) {
            if(suffix=="weights") {
                float weight;
                for(size_t i=0;i<count && parser >> weight;++i)
                    observation.pointWeights.push_back(weight);
                observation.valid = observation.valid && observation.pointWeights.size() == count;
            }
            else if(suffix=="ids") {
                for(size_t i=0;i<count;++i) {
                    int id;
                    if(!(parser>>id)) {observation.valid=false;break;}
                    observation.tagIds.push_back(id);
                }
            }
            else if(suffix=="tracked") {
                int partial=0;
                string partialLabel,ageLabel;
                if(!(parser >> observation.trackedCorners >> partialLabel >> partial >> ageLabel >> observation.trackAgeS)
                   || partialLabel!="partial" || ageLabel!="age" || (partial!=0 && partial!=1))
                    observation.valid=false;
                observation.partial=partial==1;
            }
            else if(suffix=="candidate") {
                int flag=0;if(!(parser>>flag) || flag!=1) observation.valid=false;
                observation.candidate=true;
            }
            else if(suffix=="component") {
                if(!(parser >> observation.component)) observation.valid=false;
            }
            else observation.valid=false;
        }
        observation.valid=observation.valid && (observation.partial || count>=4);
        observations.push_back(observation);
    }
}

int main(int argc, char **argv)
{
    if(argc != 9 && argc != 10)
    {
        cerr << "Usage: mono_tum_headless vocabulary settings sequence "
             << "frames.txt keyframes.txt points.xyz observations.txt timing.txt "
             << "[tag_observations.txt]" << endl;
        return 1;
    }

    // Offline processing must not inherit process-specific RANSAC seeds or
    // OpenCV worker scheduling. ORB-SLAM's own mapping threads remain intact;
    // this only makes the image/robust-estimation primitives reproducible.
    std::srand(0);
    cv::setRNGSeed(0);
    cv::setNumThreads(1);
    // Offline defaults only; respect explicit opt-outs and leave LK opt-in.
    // Set before System starts its workers, never while threads read the env.
    if(!std::getenv("ORB_SLAM3_PARALLEL_DESCRIPTORS"))
        setenv("ORB_SLAM3_PARALLEL_DESCRIPTORS","1",0);
    if(!std::getenv("ORB_SLAM3_PREFETCH_IMAGES"))
        setenv("ORB_SLAM3_PREFETCH_IMAGES","1",0);
    const char* compactHistoryEnv=std::getenv("ORB_SLAM3_COMPACT_HISTORY");
    const bool compactHistory=compactHistoryEnv && std::string(compactHistoryEnv)!="0";

    vector<string> filenames;
    vector<double> timestamps;
    LoadImages(string(argv[3]) + "/rgb.txt", filenames, timestamps);
    const char* finalizeOnlyEnv=std::getenv("ORB_SLAM3_FINALIZE_ONLY");
    const bool finalizeOnly=finalizeOnlyEnv && std::string(finalizeOnlyEnv)!="0";
    if(filenames.empty() && !finalizeOnly)
    {
        cerr << "No images in sequence " << argv[3] << endl;
        return 2;
    }

    vector<TagObservation> tagObservations;
    if(argc == 10)
        LoadTagObservations(argv[9], tagObservations);

    ORB_SLAM3::System slam(argv[1], argv[2], ORB_SLAM3::System::MONOCULAR, false);
    if(finalizeOnly) {
        slam.SelectLargestMapForOfflineFinalization();
        ofstream history(string(argv[4])+".history.jsonl");
        const auto start=chrono::steady_clock::now();
        slam.Shutdown();
        const double milliseconds=chrono::duration<double,milli>(
            chrono::steady_clock::now()-start).count();
        slam.SaveReplaySnapshot(history,0.0,true,compactHistory);
        history.close();
        slam.SaveKeyFrameTrajectoryTUM(argv[5]);
        slam.SaveMapPointsXYZ(argv[6]);
        ofstream timing(argv[8]);
        timing << fixed << setprecision(6)
               << "frames 0\n"
               << "offline_loop_queue_after_finalize " << slam.GetLoopClosingQueueSize() << "\n"
               << "offline_finalization_ms " << milliseconds << "\n"
               << "offline_finalization_converged " << (slam.OfflineFinalizationConverged()?1:0) << "\n";
        return 0;
    }
    const float imageScale = slam.GetImageScale();
    cv::FileStorage settings(argv[2],cv::FileStorage::READ);
    const bool dynamicEnabled=int(settings["ORBextractor.dynamicGeometry"])!=0;
    const bool synchronousMapping=int(settings["Offline.synchronousMapping"])!=0;
    // Experimental until end-to-end loop and finalization validation passes.
    // Normal processing retains its established schedule.
    const char* incrementalMode=std::getenv("ORB_SLAM3_INCREMENTAL_LOOP_SEARCH");
    const bool incrementalLoopSearch=incrementalMode && std::string(incrementalMode)=="1";
    cv::Mat K=(cv::Mat_<double>(3,3)<<double(settings["Camera1.fx"])*imageScale,0,double(settings["Camera1.cx"])*imageScale,
        0,double(settings["Camera1.fy"])*imageScale,double(settings["Camera1.cy"])*imageScale,0,0,1);
    cv::Mat D=(cv::Mat_<double>(5,1)<<double(settings["Camera1.k1"]),double(settings["Camera1.k2"]),
        double(settings["Camera1.p1"]),double(settings["Camera1.p2"]),double(settings["Camera1.k3"]));
    GeometricDynamicMask dynamic(K,D);
    ofstream dynamicLog;
    if(dynamicEnabled) dynamicLog.open(string(argv[4])+".dynamic.jsonl");
    double dynamicMilliseconds=0;
    size_t dynamicMaskedFrames=0,dynamicMaskedPoints=0;
    vector<double> trackingTimes;
    trackingTimes.reserve(filenames.size());
    vector<char> historyBuffer(1<<20);
    ofstream history;
    history.rdbuf()->pubsetbuf(historyBuffer.data(),historyBuffer.size());
    history.open(string(argv[4])+".history.jsonl");
    ofstream observations(argv[7]);
    observations << fixed;
    double replaySnapshotMilliseconds=0.0;
    double localMappingWaitMilliseconds=0.0;
    size_t localMappingWaitFrames=0;
    size_t localMappingWaitTimeouts=0;
    double loopClosingWaitMilliseconds=0.0;
    size_t loopClosingWaitFrames=0;
    size_t loopClosingWaitTimeouts=0;
    // An unsurveyed component transform is meaningful only in the Atlas-map
    // gauge in which it was estimated.  A process-global component transform
    // incorrectly treats independently seeded maps as one coordinate system:
    // when an old marker reappears in a new map, final BA can then pull a few
    // keyframes toward the old map's unrelated origin.  Register the same
    // component independently in each map; its repeated marker IDs provide
    // the evidence used by native common-anchor map merging.
    using ComponentKey = pair<unsigned long,string>;
    map<ComponentKey,Sophus::SE3f> tagComponentToWorld;
    set<ComponentKey> componentsRegisteredInAtlasWorld;
    CandidateRegistrationWindow candidateWindow;
    map<ComponentKey,map<int,vector<Eigen::Vector3f>>> componentReferenceCorners;
    string pendingTagComponent;
    string metricizingTagComponent;
    unsigned long metricizingTagMapId=numeric_limits<unsigned long>::max();
    double metricizingTagStartTime=-1.0;
    int metricizingTagObservations=0;
    int pendingTagComponentFrames=0;
    vector<Sophus::SE3f> pendingTagComponentTransforms;
    const char* windowEnv=std::getenv("ORB_SLAM3_COMPONENT_WINDOW");
    const bool componentWindow=windowEnv && std::string(windowEnv)=="1";
    double componentFirstTime=-1.,componentLastTime=-1.;
    unsigned long componentMapId=numeric_limits<unsigned long>::max();

    // Decode at most one future immutable input while native SLAM handles
    // the current frame. No Atlas access, frame dropping or timestamp changes.
    const char* prefetchOption=std::getenv("ORB_SLAM3_PREFETCH_IMAGES");
    const bool prefetchImages=prefetchOption && std::string(prefetchOption)=="1";
    using InputImages=std::pair<cv::Mat,cv::Mat>;
    const auto readImages=[&](size_t index) {
        const string path=string(argv[3])+"/"+filenames[index];
        InputImages data{cv::imread(path,cv::IMREAD_UNCHANGED),cv::Mat()};
        if(std::ifstream(path+".mask.png").good())
            data.second=cv::imread(path+".mask.png",cv::IMREAD_GRAYSCALE);
        return data;
    };
    std::future<InputImages> pendingImage;
    if(prefetchImages && !filenames.empty())
        pendingImage=std::async(std::launch::async,readImages,0);
    double inputWaitMilliseconds=0.;

    for(size_t index = 0; index < filenames.size(); ++index)
    {
        const auto inputStart=chrono::steady_clock::now();
        InputImages loaded=prefetchImages?pendingImage.get():readImages(index);
        inputWaitMilliseconds+=chrono::duration<double,std::milli>(chrono::steady_clock::now()-inputStart).count();
        if(prefetchImages && index+1<filenames.size())
            pendingImage=std::async(std::launch::async,readImages,index+1);
        cv::Mat image=loaded.first;
        if(image.empty())
        {
            cerr << "Cannot load image " << filenames[index] << endl;
            slam.Shutdown();
            return 3;
        }
        if(imageScale != 1.f)
            cv::resize(image, image, cv::Size(), imageScale, imageScale);

        bool estimateContinuousComponent=false;
        const unsigned long inputMapId=slam.GetCurrentMapId();
        // This buffer belongs only to metric visual-bridge registration.
        // Do not clear the separate three-observation disconnected-marker
        // bootstrap counter on every non-tracking frame.
        if(componentWindow && !pendingTagComponentTransforms.empty()) {
            const bool changed=slam.MapChanged();
            if(!ORB_SLAM3::RetainComponentMeasurements(timestamps[index],componentFirstTime,componentLastTime,
                slam.GetTrackingState()==2 && slam.IsTagMetricAligned(),inputMapId==componentMapId,!changed)) {
                pendingTagComponent.clear();pendingTagComponentTransforms.clear();pendingTagComponentFrames=0;
            }
        }
        if(index < tagObservations.size() && tagObservations[index].valid && !tagObservations[index].candidate)
        {
            const TagObservation &rawTag = tagObservations[index];
            TagObservation tag=rawTag;
            bool useTag=true;
            bool inputInAtlasWorld=false;
            if(!tag.component.empty()) {
                ComponentKey componentKey(inputMapId,tag.component);
                // Remember only past, complete strong observations. This cache
                // provides component-local geometry, never image evidence.
                if(!tag.partial) {
                    for(size_t i=0;i+3<tag.tagIds.size();i+=4) {
                        const int id=tag.tagIds[i];bool strong=i+3<tag.worldPoints.size();
                        for(size_t j=i;j<i+4;++j)strong=strong && tag.tagIds[j]==id &&
                            (tag.pointWeights.empty() || (j<tag.pointWeights.size() && tag.pointWeights[j]>=.99f));
                        if(strong)componentReferenceCorners[componentKey][id]=
                            vector<Eigen::Vector3f>(tag.worldPoints.begin()+i,tag.worldPoints.begin()+i+4);
                    }
                }
                auto registered=tagComponentToWorld.find(componentKey);
                if(registered==tagComponentToWorld.end() && tagComponentToWorld.empty()) {
                    tagComponentToWorld[componentKey]=Sophus::SE3f();
                    registered=tagComponentToWorld.find(componentKey);
                }
                if(registered==tagComponentToWorld.end() &&
                   tag.component==metricizingTagComponent &&
                   slam.GetTrackingState()==2 && slam.IsTagMetricAligned()) {
                    // TryAlignMapToTagWorld expresses the entire active map in
                    // this component's metric frame.  Register that gauge only
                    // after the native joint solve has committed.
                    tagComponentToWorld[componentKey]=Sophus::SE3f();
                    registered=tagComponentToWorld.find(componentKey);
                    cout << "Registered marker component " << tag.component
                         << " by metricizing the continuously tracked map" << endl;
                    metricizingTagComponent.clear();
                    metricizingTagMapId=numeric_limits<unsigned long>::max();
                    metricizingTagStartTime=-1.0;
                    metricizingTagObservations=0;
                }
                if(registered==tagComponentToWorld.end()) {
                    useTag=false;
                    if(slam.GetTrackingState()==2 && !slam.IsTagMetricAligned()) {
                        // A healthy arbitrary-scale map already supplies the
                        // visual connection.  Feed the new component to the
                        // native multi-view metricization path instead of
                        // discarding that map and starting a marker seed map.
                        if(metricizingTagComponent!=tag.component ||
                           metricizingTagMapId!=inputMapId) {
                            metricizingTagComponent=tag.component;
                            metricizingTagMapId=inputMapId;
                            metricizingTagStartTime=timestamps[index];
                            metricizingTagObservations=1;
                        }
                        else ++metricizingTagObservations;
                        useTag=true;
                        pendingTagComponent.clear();
                        pendingTagComponentFrames=0;
                        pendingTagComponentTransforms.clear();
                        // Pure rotation cannot reveal monocular translation
                        // scale.  Give the existing visual map a bounded
                        // interval to acquire real parallax; if it remains
                        // arbitrary while a reliable marker component keeps
                        // returning, preserve that map and start a metric
                        // marker-seed map for the task zone.  This publishes a
                        // measured metric pose without pretending the old
                        // transition map was successfully scaled.
                        if(metricizingTagObservations>=3 && metricizingTagStartTime>=0.0 &&
                           timestamps[index]-metricizingTagStartTime>=2.0) {
                            cout << "Marker metricization timed out without observable visual scale; "
                                    "preserving arbitrary map " << inputMapId <<
                                    " and starting a metric task-zone map" << endl;
                            slam.StartNewMapForMarkerComponent();
                            componentKey=ComponentKey(slam.GetCurrentMapId(),tag.component);
                            tagComponentToWorld[componentKey]=Sophus::SE3f();
                            registered=tagComponentToWorld.find(componentKey);
                            metricizingTagComponent.clear();
                            metricizingTagMapId=numeric_limits<unsigned long>::max();
                            metricizingTagStartTime=-1.0;
                            metricizingTagObservations=0;
                        }
                    }
                    // A valid metric visual track supplies the missing
                    // relationship between markers that never share a frame.
                    // Accumulate it after TrackMonocular has produced this
                    // frame's pose; until then the new marker cannot influence
                    // the camera estimate used to register itself.
                    else if(slam.GetTrackingState()==2 && slam.IsTagMetricAligned()) {
                        estimateContinuousComponent=true;
                        if(pendingTagComponent!=tag.component) {
                            pendingTagComponent=tag.component;
                            pendingTagComponentFrames=0;
                            pendingTagComponentTransforms.clear();
                        }
                    } else {
                        if(tag.component==pendingTagComponent) ++pendingTagComponentFrames;
                        else {
                            pendingTagComponent=tag.component;
                            pendingTagComponentFrames=1;
                            pendingTagComponentTransforms.clear();
                        }
                        useTag=pendingTagComponentFrames>=3;
                        if(useTag) {
                            cout << "Disconnected marker component " << tag.component
                                 << ": no metric visual bridge; preserving the old map and "
                                    "starting a new metric map" << endl;
                            slam.StartNewMapForMarkerComponent();
                            componentKey=ComponentKey(slam.GetCurrentMapId(),tag.component);
                            tagComponentToWorld[componentKey]=Sophus::SE3f();
                            registered=tagComponentToWorld.find(componentKey);
                            pendingTagComponent.clear(); pendingTagComponentFrames=0;
                            metricizingTagComponent.clear();
                            metricizingTagMapId=numeric_limits<unsigned long>::max();
                            metricizingTagStartTime=-1.0;
                            metricizingTagObservations=0;
                        }
                    }
                }
                if(registered!=tagComponentToWorld.end()) {
                    inputInAtlasWorld=!tag.partial && componentsRegisteredInAtlasWorld.count(componentKey);
                    // A previously registered component may have moved during
                    // BA. Before inserting a new ID, refresh from a known ID in
                    // THAT component, even when that reference is not visible.
                    // Never apply the last, unrelated component's global gauge.
                    if(inputInAtlasWorld && slam.IsTagMetricAligned()) {
                        ORB_SLAM3::Map* current=nullptr;
                        for(auto* point:slam.GetTrackedMapPoints())
                            if(point && !point->isBad() && point->GetMap() &&
                               point->GetMap()->GetId()==inputMapId) {current=point->GetMap();break;}
                        if(current) {
                            unique_lock<mutex> guard(current->mMutexMapUpdate);
                            bool newId=false;
                            for(int id:tag.tagIds)newId=newId || !current->mStaticTags.count(id);
                            if(newId)for(const auto& reference:componentReferenceCorners[componentKey]) {
                                auto committed=current->mStaticTags.find(reference.first);
                                if(committed!=current->mStaticTags.end() &&
                                   ORB_SLAM3::RefreshMarkerComponentTransform(reference.second,committed->second,registered->second)) {
                                    inputInAtlasWorld=true;
                                    componentsRegisteredInAtlasWorld.insert(componentKey);
                                    cout<<"MARKER_COMPONENT_REFRESH time="<<timestamps[index]
                                        <<" map="<<inputMapId<<" component="<<tag.component
                                        <<" reference="<<reference.first<<endl;
                                    break;
                                }
                            }
                        }
                    }
                    const Sophus::SE3f& worldFromComponent=registered->second;
                    tag.Twc=worldFromComponent*tag.Twc;
                    for(auto& point:tag.worldPoints) point=worldFromComponent*point;
                    pendingTagComponent.clear(); pendingTagComponentFrames=0;
                    pendingTagComponentTransforms.clear();
                }
            } else {
                pendingTagComponent.clear(); pendingTagComponentFrames=0;
                pendingTagComponentTransforms.clear();
            }
            if(useTag)
                slam.SetExternalTagObservation(
                    tag.Twc, tag.confidence, tag.worldPoints, tag.imagePoints, true, tag.pointWeights, tag.tagIds,
                    tag.partial, tag.trackedCorners, tag.trackAgeS, inputInAtlasWorld);
            else
                slam.SetExternalTagObservation(Sophus::SE3f(), 0.0f, {}, {}, false);
        }
        else
        {
            if(!componentWindow) {pendingTagComponent.clear(); pendingTagComponentFrames=0;}
            slam.SetExternalTagObservation(
                Sophus::SE3f(), 0.0f, {}, {}, false);
        }

        cv::Mat mask=loaded.second;
        if(!mask.empty() && imageScale!=1.f)
            cv::resize(mask,mask,image.size(),0,0,cv::INTER_NEAREST);
        const auto start = chrono::steady_clock::now();
        if(dynamicEnabled) mask=dynamic.prepare(image,mask,timestamps[index]);
        const auto prepared=chrono::steady_clock::now();
        slam.SetFeatureMask(mask);
        const Sophus::SE3f Tcw=slam.TrackMonocular(image, timestamps[index]);
        const auto stop = chrono::steady_clock::now();

        if(estimateContinuousComponent && index<tagObservations.size()) {
            const TagObservation& rawTag=tagObservations[index];
            const unsigned long outputMapId=slam.GetCurrentMapId();
            if(outputMapId==inputMapId && slam.GetTrackingState()==2 && slam.IsTagMetricAligned() &&
               Tcw.matrix().allFinite()) {
                if(componentWindow && slam.MapChanged()) {
                    pendingTagComponentTransforms.clear();
                }
                const Sophus::SE3f candidate=Tcw.inverse()*rawTag.Twc.inverse();
                bool consistent=candidate.matrix().allFinite();
                if(consistent && !pendingTagComponentTransforms.empty()) {
                    consistent=ORB_SLAM3::MarkerComponentTransformsConsistent(
                        pendingTagComponentTransforms.front(),candidate);
                }
                if(!consistent) pendingTagComponentTransforms.clear();
                if(candidate.matrix().allFinite()) {
                    if(pendingTagComponentTransforms.empty()) componentFirstTime=timestamps[index];
                    componentLastTime=timestamps[index];componentMapId=outputMapId;
                    pendingTagComponentTransforms.push_back(candidate);
                }
                if(componentWindow) cout << "MARKER_COMPONENT_WINDOW component=" << rawTag.component
                    << " frame=" << index << " time=" << timestamps[index]
                    << " consistent=" << consistent << " count=" << pendingTagComponentTransforms.size() << endl;
                if(pendingTagComponentTransforms.size()>=3) {
                    Eigen::Vector3f translation=Eigen::Vector3f::Zero();
                    Eigen::Vector4f quaternionSum=Eigen::Vector4f::Zero();
                    Eigen::Quaternionf reference(
                        pendingTagComponentTransforms.front().rotationMatrix());
                    for(const Sophus::SE3f& transform:pendingTagComponentTransforms) {
                        translation+=transform.translation();
                        Eigen::Quaternionf quaternion(transform.rotationMatrix());
                        if(reference.dot(quaternion)<0.f) quaternion.coeffs()*=-1.f;
                        quaternionSum+=quaternion.coeffs();
                    }
                    translation/=float(pendingTagComponentTransforms.size());
                    Eigen::Quaternionf rotation;
                    rotation.coeffs()=quaternionSum.normalized();
                    tagComponentToWorld[ComponentKey(outputMapId,rawTag.component)]=Sophus::SE3f(
                        rotation.normalized(),translation);
                    componentsRegisteredInAtlasWorld.insert(ComponentKey(outputMapId,rawTag.component));
                    cout << "Registered marker component " << rawTag.component
                         << " through continuous metric SLAM trajectory" << endl;
                    pendingTagComponent.clear(); pendingTagComponentFrames=0;
                    pendingTagComponentTransforms.clear();
                }
            } else {
                pendingTagComponentTransforms.clear();
            }
        }


        // Collect independent metric ORB evidence for an unregistered component.
        // Low-confidence candidates were explicitly withheld from TrackMonocular.
        if(slam.GetTrackingState()!=2 || !slam.IsTagMetricAligned() || slam.GetCurrentMapId()!=inputMapId)
            candidateWindow.samples.clear();
        else if(index<tagObservations.size()) {
            const auto& raw=tagObservations[index];
            const ComponentKey key(inputMapId,raw.component);
            if(raw.valid && !raw.component.empty() && !tagComponentToWorld.count(key)) {
                int revision=-1;
                for(auto* p:slam.GetTrackedMapPoints()) if(p && !p->isBad() && p->GetMap()->GetId()==inputMapId) {
                    revision=p->GetMap()->GetLastBigChangeIdx();break;
                }
                Sophus::SE3f transform;float rms=0;
                bool accepted=revision>=0 && candidateWindow.add(raw,Tcw,timestamps[index],inputMapId,revision,K,imageScale,transform,rms);
                // Offline short-visit rescue only. If the ordinary strong
                // stream has enough observations in this bounded visit, do
                // not pre-empt its existing registration with a weaker seed.
                int standardViews=0;
                for(const auto& sample:candidateWindow.samples)
                    standardViews+=!sample.tag.candidate && sample.tag.confidence>=.35f;
                double lastSeen=timestamps[index];
                const double firstSeen=candidateWindow.samples.empty()?lastSeen:candidateWindow.samples.front().time;
                for(size_t j=index+1;j<tagObservations.size() && j<timestamps.size();++j) {
                    if(timestamps[j]-firstSeen>1.0 || timestamps[j]-lastSeen>.5) break;
                    const auto& future=tagObservations[j];
                    if(future.valid && future.component==raw.component) {
                        lastSeen=timestamps[j];
                        standardViews+=!future.candidate && !future.partial && future.confidence>=.35f;
                    }
                }
                if(standardViews>=3) accepted=false;
                cout << "MARKER_CANDIDATE_WINDOW component=" << raw.component << " frame=" << index
                     << " candidate=" << raw.candidate << " revision=" << revision
                     << " views=" << candidateWindow.samples.size() << " max_rms_px=" << rms
                     << " standard_visit_views=" << standardViews << " accepted=" << accepted << endl;
                if(accepted) {
                    tagComponentToWorld[key]=transform;
                    componentsRegisteredInAtlasWorld.insert(key);
                    cout << "MARKER_CANDIDATE_REGISTERED component=" << raw.component << " frame=" << index
                         << " time=" << timestamps[index] << " views=" << candidateWindow.samples.size()
                         << " max_rms_px=" << rms << endl;
                    candidateWindow.samples.clear();
                    pendingTagComponent.clear();pendingTagComponentFrames=0;pendingTagComponentTransforms.clear();
                }
            }
        }
        const vector<ORB_SLAM3::MapPoint*> mapPoints = slam.GetTrackedMapPoints();
        const vector<cv::KeyPoint> keyPoints = slam.GetTrackedKeyPointsUn();
        const size_t count = min(mapPoints.size(), keyPoints.size());
        if(dynamicEnabled) {
            vector<GeometricDynamicMask::Seed> seeds;
            vector<cv::Point3f> rays;
            ORB_SLAM3::Map* map=nullptr;
            for(size_t j=0;j<count;++j) {
                auto* p=mapPoints[j];
                if(!p || p->isBad()) continue;
                if(!map) map=p->GetMap();
                if(p->GetMap()!=map) continue;
                const auto world=p->GetWorldPos();
                seeds.push_back({p->mnId,{},cv::Point3f(world.x(),world.y(),world.z())});
                rays.emplace_back((keyPoints[j].pt.x-K.at<double>(0,2))/K.at<double>(0,0),
                    (keyPoints[j].pt.y-K.at<double>(1,2))/K.at<double>(1,1),1.f);
            }
            vector<cv::Point2f> pixels;
            if(!rays.empty()) cv::projectPoints(rays,cv::Vec3d(),cv::Vec3d(),K,D,pixels);
            for(size_t j=0;j<seeds.size();++j) seeds[j].pixel=pixels[j];
            cv::Mat rotation(3,3,CV_64F),translation(3,1,CV_64F),rvec;
            for(int row=0;row<3;++row) {
                translation.at<double>(row)=Tcw.translation()(row);
                for(int col=0;col<3;++col) rotation.at<double>(row,col)=Tcw.rotationMatrix()(row,col);
            }
            cv::Rodrigues(rotation,rvec);
            dynamic.observe(rvec,translation,seeds,slam.GetTrackingState()==2 && map,
                map?long(map->GetId()):-1,map?map->GetLastBigChangeIdx():-1);
            dynamicMilliseconds+=chrono::duration<double,milli>(prepared-start).count()+
                chrono::duration<double,milli>(chrono::steady_clock::now()-stop).count();
            dynamicMaskedFrames+=dynamic.masked>0;dynamicMaskedPoints+=dynamic.masked;
            dynamicLog<<"{\"frame\":"<<index<<",\"tested\":"<<dynamic.tested
                <<",\"confirmed\":"<<dynamic.confirmed<<",\"masked\":"<<dynamic.masked
                <<",\"reason\":\""<<dynamic.reason<<"\",\"masked_pixels\":[";
            for(size_t j=0;j<dynamic.maskedPixels.size();++j) {
                if(j) dynamicLog<<",";
                dynamicLog<<"["<<dynamic.maskedPixels[j].x<<","<<dynamic.maskedPixels[j].y<<"]";
            }
            dynamicLog<<"]}"<<endl;
        }
        trackingTimes.push_back(chrono::duration<double,milli>(chrono::steady_clock::now()-start).count());
        size_t tracked = 0;
        for(size_t pointIndex = 0; pointIndex < count; ++pointIndex)
            if(mapPoints[pointIndex] != nullptr && !mapPoints[pointIndex]->isBad())
                ++tracked;
        observations << setprecision(9) << timestamps[index] << " "
                     << slam.GetTrackingState() << " " << tracked;
        for(size_t pointIndex = 0; pointIndex < count; ++pointIndex)
        {
            if(mapPoints[pointIndex] == nullptr || mapPoints[pointIndex]->isBad())
                continue;
            // Persist the association, not only a transient 2-D dot.  The
            // final offline pose-only pass can then reproject the same match
            // against the final (post-BA) Atlas point without extracting ORB
            // a second time.
            observations << " " << mapPoints[pointIndex]->mnId
                         << " " << setprecision(3) << keyPoints[pointIndex].pt.x
                         << " " << keyPoints[pointIndex].pt.y;
        }
        observations << endl;
        // Deterministic offline mode snapshots only after both mapping workers
        // commit the current keyframe. This prevents CPU scheduling from
        // changing which map revision the next frame observes.
        if(synchronousMapping) {
            const auto waitStart=chrono::steady_clock::now();
            if(!slam.WaitForLocalMappingIdle(1000)) {
                ++localMappingWaitTimeouts;
                if(incrementalLoopSearch && !slam.WaitForLocalMappingIdle(30000))
                    throw std::runtime_error("Offline mapping did not drain; refusing to advance replay time");
            }
            localMappingWaitMilliseconds+=chrono::duration<double,milli>(
                chrono::steady_clock::now()-waitStart).count();
            ++localMappingWaitFrames;
            // Drain the loop worker before advancing video time. A long solve
            // slows offline processing, not the timestamp at which it appears.
            const auto loopWaitStart=chrono::steady_clock::now();
            if(!slam.WaitForLoopClosingIdle(1000)) {
                ++loopClosingWaitTimeouts;
                if(incrementalLoopSearch && !slam.WaitForLoopClosingIdle(30000))
                    throw std::runtime_error("Offline loop worker did not drain; refusing to advance replay time");
            }
            loopClosingWaitMilliseconds+=chrono::duration<double,milli>(
                chrono::steady_clock::now()-loopWaitStart).count();
            ++loopClosingWaitFrames;
        }
        // A snapshot contains the latest atomically committed map revision;
        // later offline events capture subsequent mapping/loop corrections.
        const auto replayStart=chrono::steady_clock::now();
        slam.SaveReplaySnapshot(history,timestamps[index],false,compactHistory);
        replaySnapshotMilliseconds+=chrono::duration<double,milli>(
            chrono::steady_clock::now()-replayStart).count();

    }

    const size_t offlineLoopQueueAtShutdown=slam.GetLoopClosingQueueSize();
    const auto offlineFinalizationStart=chrono::steady_clock::now();
    slam.Shutdown();
    const double offlineFinalizationMilliseconds=chrono::duration<double,milli>(
        chrono::steady_clock::now()-offlineFinalizationStart).count();
    const bool offlineFinalizationConverged=slam.OfflineFinalizationConverged();
    const auto replayFinalStart=chrono::steady_clock::now();
    slam.SaveReplaySnapshot(history,timestamps.back(),true,compactHistory);
    const double replayFinalMilliseconds=chrono::duration<double,milli>(
        chrono::steady_clock::now()-replayFinalStart).count();
    history.close();
    slam.SaveFrameTrajectoryTUM(argv[4]);
    slam.SaveKeyFrameTrajectoryTUM(argv[5]);
    slam.SaveMapPointsXYZ(argv[6]);

    sort(trackingTimes.begin(), trackingTimes.end());
    double total = 0.0;
    for(double value : trackingTimes)
        total += value;
    ofstream timing(argv[8]);
    timing << fixed << setprecision(6)
           << "frames " << trackingTimes.size() << endl
           << "median_ms " << trackingTimes[trackingTimes.size() / 2] << endl
           << "mean_ms " << total / trackingTimes.size() << endl
           << "tag_metric_aligned " << (slam.IsTagMetricAligned() ? 1 : 0) << endl
           << "tag_metric_scale " << slam.GetRecoveredTagMetricScale() << endl
           << "tag_keyframes_accepted " << slam.GetTagKeyFramesAccepted() << endl
           << "tag_keyframes_rejected " << slam.GetTagKeyFramesRejected() << endl
           << "tag_pose_constraints " << slam.GetTagPoseConstraintsApplied() << endl
           << "compact_history " << (compactHistory ? 1 : 0) << endl
           << "replay_snapshot_mean_ms " << replaySnapshotMilliseconds/trackingTimes.size() << endl
           << "replay_final_ms " << replayFinalMilliseconds << endl
           << "local_mapping_wait_ms " << localMappingWaitMilliseconds << endl
           << "local_mapping_wait_frames " << localMappingWaitFrames << endl
           << "local_mapping_wait_timeouts " << localMappingWaitTimeouts << endl
           << "loop_closing_wait_ms " << loopClosingWaitMilliseconds << endl
           << "loop_closing_wait_frames " << loopClosingWaitFrames << endl
           << "loop_closing_wait_timeouts " << loopClosingWaitTimeouts << endl
           << "offline_loop_queue_at_shutdown " << offlineLoopQueueAtShutdown << endl
           << "offline_loop_queue_after_finalize " << slam.GetLoopClosingQueueSize() << endl
           << "offline_finalization_ms " << offlineFinalizationMilliseconds << endl
           << "offline_finalization_converged " << (offlineFinalizationConverged?1:0) << endl;
    timing << "offline_synchronous_mapping " << (synchronousMapping?1:0) << endl
           << "prefetch_images " << (prefetchImages?1:0) << endl
           << "input_wait_ms " << inputWaitMilliseconds << endl
           << "parallel_descriptors " << (std::getenv("ORB_SLAM3_PARALLEL_DESCRIPTORS") &&
                std::string(std::getenv("ORB_SLAM3_PARALLEL_DESCRIPTORS"))=="1"?1:0) << endl
           << "temporal_flow " << (std::getenv("ORB_SLAM3_TEMPORAL_FLOW") &&
                std::string(std::getenv("ORB_SLAM3_TEMPORAL_FLOW"))=="1"?1:0) << endl
           << "dynamic_geometry " << (dynamicEnabled?1:0) << endl
           << "dynamic_mean_ms " << dynamicMilliseconds/trackingTimes.size() << endl
           << "dynamic_masked_frames " << dynamicMaskedFrames << endl
           << "dynamic_masked_point_frames " << dynamicMaskedPoints << endl
           << "dynamic_probe_frames " << dynamic.probeFrames << endl
           << "dynamic_reused_frames " << dynamic.reusedFrames << endl;
    return 0;
}
