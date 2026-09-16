/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/


#include "LoopClosing.h"

#include "Sim3Solver.h"
#include "Converter.h"
#include "Optimizer.h"
#include "ORBmatcher.h"
#include "G2oTypes.h"
#include "PostLoopRematch.h"
#include "MarkerGraphOptimizer.h"
#include "MarkerGraphCoordinator.h"
#include "LoopMarkerEvidencePolicy.h"
#include "CameraModels/GeometricCamera.h"

#include<mutex>
#include<thread>
#include<cstdlib>
#include<limits>


namespace ORB_SLAM3
{

namespace {
bool IsStagedMetricLoop(Map* current, Map* matched)
{
    // HasMetricTagGeometry alone also includes not-yet-metric marker seeds.
    // Enhanced recall/support requires the staged metric commit in CorrectLoop.
    return current && current==matched && current->mbMetric && current->HasMetricTagGeometry();
}
// Privilege for 50--79 visual matches must be supported by this candidate,
// not merely by a marker registered somewhere in the same metric map.
bool CandidateMarkerEvidence(KeyFrame* current,KeyFrame* matched,const g2o::Sim3& Scw)
{
    if(!current || !matched || !IsStagedMetricLoop(current->GetMap(),matched->GetMap())) return false;
    Map* map=current->GetMap();
    std::vector<LoopMarkerEvidenceView> sides[2];
    const auto currentPose=current->GetPose().cast<double>();
    const g2o::Sim3 original(currentPose.unit_quaternion(),currentPose.translation(),1.0);
    const g2o::Sim3 forward=original.inverse()*Scw;
    const g2o::Sim3 backward=Scw.inverse()*original;
    for(KeyFrame* k:map->GetAllKeyFrames()) {
        if(!k || k->isBad() || k->GetMap()!=map || !k->mpCamera ||
           !k->mbHasTagObservation || !k->mbTagObservationActive ||
           !std::isfinite(k->mTagObservationConfidence) || k->mTagObservationConfidence<.35f ||
           k->mvTagIds.size()!=k->mvTagImagePoints.size() ||
           k->mvTagIds.size()!=k->mvTagWorldPoints.size()) continue;
        const bool near[2]={std::abs(k->mTimeStamp-current->mTimeStamp)<=2.0,
                            std::abs(k->mTimeStamp-matched->mTimeStamp)<=2.0};
        if(!near[0] && !near[1]) continue;
        std::map<int,std::vector<std::size_t>> groups;
        for(std::size_t i=0;i<k->mvTagIds.size();++i) groups[k->mvTagIds[i]].push_back(i);
        const auto pose=k->GetPose().cast<double>();
        const g2o::Sim3 camera(pose.unit_quaternion(),pose.translation(),1.0);
        for(const auto& group:groups) {
            auto landmark=map->mStaticTags.find(group.first);
            if(group.second.size()!=4 || landmark==map->mStaticTags.end() || landmark->second.size()!=12) continue;
            for(int side=0;side<2;++side) if(near[side]) {
                bool strong=true;double maximum=0;
                const g2o::Sim3 prediction=camera*(side==0?forward:backward);
                for(std::size_t corner=0;corner<4;++corner) {
                    const auto i=group.second[corner];
                    if(!k->mvTagPointWeights.empty() && (i>=k->mvTagPointWeights.size() ||
                       !std::isfinite(k->mvTagPointWeights[i]) || k->mvTagPointWeights[i]<.99f)) strong=false;
                    const auto& xyz=landmark->second;
                    const Eigen::Vector3d world(xyz[3*corner],xyz[3*corner+1],xyz[3*corner+2]);
                    const Eigen::Vector3d projected=prediction.map(world);
                    if(!projected.allFinite() || projected.z()<=1e-6) {strong=false;break;}
                    const Eigen::Vector2d pixel=k->mpCamera->project(projected);
                    const auto observed=k->mvTagImagePoints[i];
                    const double error=(pixel-Eigen::Vector2d(observed.x,observed.y)).norm();
                    if(!std::isfinite(error)) {strong=false;break;}
                    maximum=std::max(maximum,error);
                }
                sides[side].push_back({k->mnId,group.first,k->mTimeStamp,maximum,strong});
            }
        }
    }
    return HasCommonLoopMarkerEvidence(sides[0],sides[1],current->mTimeStamp,matched->mTimeStamp);
}
bool MetricLoopScaleExperiment(bool offline, Map* current, Map* matched)
{
    const char* enabled=std::getenv("ORB_SLAM3_MARKER_SIM3_LOOP");
    const char* incremental=std::getenv("ORB_SLAM3_INCREMENTAL_LOOP_SEARCH");
    const bool incrementalOffline=incremental && std::string(incremental)=="1";
    // This only releases candidate estimation; all metric commits now go
    // through ProposeVisualLoop, including incremental offline processing.
    return (offline || incrementalOffline) && IsStagedMetricLoop(current,matched) &&
        enabled && std::string(enabled)=="1";
}
struct OfflineCandidateDiagnostic {
    bool enabled;
    unsigned long query, candidate;
    const char* stage = "invalid_candidate";
    int bow=0, ransac=0, projection=0, optimized=0, refined=0, neighbors=0;
    ~OfflineCandidateDiagnostic() {
        if(enabled) std::cout << "OFFLINE_LOOP_CANDIDATE {\"query\":" << query
            << ",\"candidate\":" << candidate << ",\"stage\":\"" << stage
            << "\",\"bow\":" << bow << ",\"ransac\":" << ransac
            << ",\"projection\":" << projection << ",\"optimized\":" << optimized
            << ",\"refined\":" << refined << ",\"neighbors\":" << neighbors << "}" << std::endl;
    }
};
bool ShareConsistentMetricMarker(Map* first, Map* second)
{
    if(!first || !second || !first->mbMetric || !second->mbMetric) return false;
    for(const auto& marker:first->mStaticTags) {
        const auto other=second->mStaticTags.find(marker.first);
        if(other==second->mStaticTags.end() || marker.second.size()!=12 ||
           other->second.size()!=12) continue;
        bool consistent=true;
        for(std::size_t corner=0;corner<4;++corner) {
            const Eigen::Vector3f a(marker.second[3*corner],marker.second[3*corner+1],
                                    marker.second[3*corner+2]);
            const Eigen::Vector3f b(other->second[3*corner],other->second[3*corner+1],
                                    other->second[3*corner+2]);
            consistent=consistent && a.allFinite() && b.allFinite() &&
                (a-b).norm()<=0.006f;
        }
        if(consistent) return true;
    }
    return false;
}
}

LoopClosing::LoopClosing(Atlas *pAtlas, KeyFrameDatabase *pDB, ORBVocabulary *pVoc, const bool bFixScale, const bool bActiveLC):
    mbResetRequested(false), mbResetActiveMapRequested(false), mbFinishRequested(false), mbFinished(true), mpAtlas(pAtlas),
    mpKeyFrameDB(pDB), mpORBVocabulary(pVoc), mpMatchedKF(NULL), mLastLoopKFid(0), mbRunningGBA(false), mbFinishedGBA(true),
    mbStopGBA(false), mpThreadGBA(NULL), mbFixScale(bFixScale), mnFullBAIdx(0), mnLoopNumCoincidences(0), mnMergeNumCoincidences(0),
    mbLoopDetected(false), mbMergeDetected(false), mnLoopNumNotFound(0), mnMergeNumNotFound(0), mbActiveLC(bActiveLC)
{
    mnCovisibilityConsistencyTh = 3;
    mpLastCurrentKF = static_cast<KeyFrame*>(NULL);

#ifdef REGISTER_TIMES

    vdDataQuery_ms.clear();
    vdEstSim3_ms.clear();
    vdPRTotal_ms.clear();

    vdMergeMaps_ms.clear();
    vdWeldingBA_ms.clear();
    vdMergeOptEss_ms.clear();
    vdMergeTotal_ms.clear();
    vnMergeKFs.clear();
    vnMergeMPs.clear();
    nMerges = 0;

    vdLoopFusion_ms.clear();
    vdLoopOptEss_ms.clear();
    vdLoopTotal_ms.clear();
    vnLoopKFs.clear();
    nLoop = 0;

    vdGBA_ms.clear();
    vdUpdateMap_ms.clear();
    vdFGBATotal_ms.clear();
    vnGBAKFs.clear();
    vnGBAMPs.clear();
    nFGBA_exec = 0;
    nFGBA_abort = 0;

#endif

    mstrFolderSubTraj = "SubTrajectories/";
    mnNumCorrection = 0;
    mnCorrectionGBA = 0;
}

void LoopClosing::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

void LoopClosing::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

std::vector<LoopClosing::CorrectionRejection> LoopClosing::GetCorrectionRejections()
{
    unique_lock<mutex> lock(mMutexCorrectionRejections);
    return mvCorrectionRejections;
}

unsigned long LoopClosing::RegisterGBAWorkerLocked()
{
    const unsigned long generation=++mnFullBAIdx;
    mGBAGenerations.insert(generation);
    mbRunningGBA=true;
    mbFinishedGBA=false;
    mbStopGBA=false;
    return generation;
}

void LoopClosing::FinishGBAWorker(unsigned long generation)
{
    unique_lock<mutex> lock(mMutexGBA);
    mGBAGenerations.erase(generation);
    mbRunningGBA=!mGBAGenerations.empty();
    mbFinishedGBA=!mbRunningGBA;
}

bool LoopClosing::IsGBACommitCurrent(Map* map, bool initialImu, unsigned long generation)
{
    // Called in the commit-lock scope, including the early-return regression.
    return generation==mnFullBAIdx && (initialImu || !map->isImuInitialized());
}

void LoopClosing::CancelGlobalBundleAdjustment()
{
    unique_lock<mutex> lock(mMutexGBA);
    mbStopGBA=true;
    ++mnFullBAIdx;
    if(mpThreadGBA) {
        if(mpThreadGBA->joinable()) mpThreadGBA->detach();
        delete mpThreadGBA;
        mpThreadGBA=nullptr;
    }
    // Invalidate its result now; only the worker's return unregisters it.
}

void LoopClosing::LaunchGlobalBundleAdjustment(Map* map, KeyFrame* keyframe)
{
    unique_lock<mutex> lock(mMutexGBA);
    if(mpThreadGBA) {
        if(mpThreadGBA->joinable()) mpThreadGBA->detach();
        delete mpThreadGBA;
        mpThreadGBA=nullptr;
    }
    const unsigned long generation=RegisterGBAWorkerLocked();
    mnCorrectionGBA=mnNumCorrection;
    try {
        // Registration precedes dispatch while holding the same mutex used
        // by completion; even an immediately returning worker is accounted.
        mpThreadGBA=new thread(&LoopClosing::RunGlobalBundleAdjustment,this,map,
                              keyframe->mnId,keyframe->mTimeStamp,generation);
    } catch(...) {
        mGBAGenerations.erase(generation);
        mbRunningGBA=!mGBAGenerations.empty();
        mbFinishedGBA=!mbRunningGBA;
        throw;
    }
}

bool LoopClosing::RejectUnsupportedCorrection(const std::string &kind, Map* current, Map* other,
                                              unsigned long keyframeId, double timestamp)
{
    // The caller holds the Atlas correction gate, so a first marker cannot be
    // inserted between this check and the map update. Do not rescale a physical
    // marker or silently throw away its gauge with a background-only Sim3.
    const bool missing=!current || !other || current->IsBad() || other->IsBad();
    const bool metricGeometry=!missing &&
        (current->HasMetricTagGeometry() || other->HasMetricTagGeometry());
    if(!missing && !metricGeometry)
        return false;
    // A visually verified monocular merge may safely metricize the active
    // arbitrary-scale map when the matched (destination) map already owns the
    // metric marker gauge. MergeLocal applies the recovered Sim(3) only to the
    // arbitrary map's cameras/points and keeps the destination world frame.
    // The arbitrary map cannot contribute active metric marker factors here;
    // those require the joint marker-graph path below.
    if(!missing && kind=="merge" && other->mbMetric && std::isfinite(mSold_new.scale())) {
        const bool arbitraryIntoMetric=!current->mbMetric && !current->HasMetricTagGeometry() &&
            mSold_new.scale()>0.05 && mSold_new.scale()<20.0;
        const double rotationDeg=Eigen::AngleAxisd(mSold_new.rotation()).angle()*57.29577951308232;
        const bool sameMarkerWorld=current->mbMetric && ShareConsistentMetricMarker(current,other) &&
            std::abs(std::log(mSold_new.scale()))<=0.03 &&
            mSold_new.translation().norm()<=0.10 && rotationDeg<=10.0;
        if(current->mbMetric && ShareConsistentMetricMarker(current,other))
            cout << "METRIC_MARKER_MERGE_CANDIDATE scale=" << mSold_new.scale()
                 << " translation_m=" << mSold_new.translation().norm()
                 << " rotation_deg=" << rotationDeg
                 << " accepted=" << (sameMarkerWorld?1:0) << endl;
        if(arbitraryIntoMetric || sameMarkerWorld) return false;
    }
    // A same-map visual loop is safe in a metric map: the essential graph has
    // one fixed gauge keyframe and original marker-corner factors on every
    // eligible marker keyframe, including the current one.
    const bool sameMetricLoop=!missing && kind=="loop" && current==other && current->mbMetric;
    // CorrectLoop mutates poses AND fuses/deletes point associations. A later
    // rejected BA cannot undo it. Until a complete staged transaction exists,
    // free-scale metric candidates must never enter this in-place path, even
    // when the offline experiment is enabled. Reanchor uses its separate
    // validated proposal/commit path; pure visual arbitrary maps are unchanged.
    if(sameMetricLoop && std::isfinite(mg2oLoopScw.scale()) &&
       std::abs(mg2oLoopScw.scale()-1.0)<=1e-6)
        return false;
    CorrectionRejection rejection;
    rejection.timestamp=timestamp;
    rejection.keyframeId=keyframeId;
    rejection.mapId=current?current->GetId():0;
    rejection.otherMapId=other?other->GetId():0;
    rejection.kind=kind;
    rejection.reason=missing?"map_unavailable":sameMetricLoop?
        "metric_sim3_requires_staged_validation":
        "fixed_marker_geometry_requires_joint_optimization";
    {
        unique_lock<mutex> lock(mMutexCorrectionRejections);
        rejection.sequence=mvCorrectionRejections.size()+1;
        mvCorrectionRejections.push_back(rejection);
    }
    cout << "POSE_GRAPH_REJECTED " << kind << " map=" << rejection.mapId
         << " other_map=" << rejection.otherMapId << " reason=" << rejection.reason << endl;
    return true;
}


void LoopClosing::Run()
{
    mbFinished =false;

    struct ProcessingGuard
    {
        explicit ProcessingGuard(std::atomic<bool> &flag) : mFlag(flag)
        {
            mFlag.store(true, std::memory_order_release);
        }
        ~ProcessingGuard()
        {
            mFlag.store(false, std::memory_order_release);
        }
        std::atomic<bool> &mFlag;
    };

    while(1)
    {

        //NEW LOOP AND MERGE DETECTION ALGORITHM
        //----------------------------


        if(CheckNewKeyFrames())
        {
            // This remains true after the queue pop and across every early
            // continue, so offline callers cannot mistake an in-flight loop
            // or merge correction for an idle backend.
            ProcessingGuard processing(mbProcessingKeyFrame);
            if(mpLastCurrentKF)
            {
                mpLastCurrentKF->mvpLoopCandKFs.clear();
                mpLastCurrentKF->mvpMergeCandKFs.clear();
            }
#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_StartPR = std::chrono::steady_clock::now();
#endif

            bool bFindedRegion = NewDetectCommonRegions();

#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_EndPR = std::chrono::steady_clock::now();

            double timePRTotal = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndPR - time_StartPR).count();
            vdPRTotal_ms.push_back(timePRTotal);
#endif
            if(bFindedRegion)
            {
                if(mbMergeDetected)
                {
                    bool mergeAccepted=false;
                    if ((mpTracker->mSensor==System::IMU_MONOCULAR || mpTracker->mSensor==System::IMU_STEREO || mpTracker->mSensor==System::IMU_RGBD) &&
                        (!mpCurrentKF->GetMap()->isImuInitialized()))
                    {
                        cout << "IMU is not initilized, merge is aborted" << endl;
                    }
                    else
                    {
                        Sophus::SE3d mTmw = mpMergeMatchedKF->GetPose().cast<double>();
                        g2o::Sim3 gSmw2(mTmw.unit_quaternion(), mTmw.translation(), 1.0);
                        Sophus::SE3d mTcw = mpCurrentKF->GetPose().cast<double>();
                        g2o::Sim3 gScw1(mTcw.unit_quaternion(), mTcw.translation(), 1.0);
                        g2o::Sim3 gSw2c = mg2oMergeSlw.inverse();
                        g2o::Sim3 gSw1m = mg2oMergeSlw;

                        mSold_new = (gSw2c * gScw1);


                        if(mpCurrentKF->GetMap()->IsInertial() && mpMergeMatchedKF->GetMap()->IsInertial())
                        {
                            cout << "Merge check transformation with IMU" << endl;
                            if(mSold_new.scale()<0.90||mSold_new.scale()>1.1){
                                mpMergeLastCurrentKF->SetErase();
                                mpMergeMatchedKF->SetErase();
                                mnMergeNumCoincidences = 0;
                                mvpMergeMatchedMPs.clear();
                                mvpMergeMPs.clear();
                                mnMergeNumNotFound = 0;
                                mbMergeDetected = false;
                                Verbose::PrintMess("scale bad estimated. Abort merging", Verbose::VERBOSITY_NORMAL);
                                continue;
                            }
                            // If inertial, force only yaw
                            if ((mpTracker->mSensor==System::IMU_MONOCULAR || mpTracker->mSensor==System::IMU_STEREO || mpTracker->mSensor==System::IMU_RGBD) &&
                                   mpCurrentKF->GetMap()->GetIniertialBA1())
                            {
                                Eigen::Vector3d phi = LogSO3(mSold_new.rotation().toRotationMatrix());
                                phi(0)=0;
                                phi(1)=0;
                                mSold_new = g2o::Sim3(ExpSO3(phi),mSold_new.translation(),1.0);
                            }
                        }

                        mg2oMergeSmw = gSmw2 * gSw2c * gScw1;

                        mg2oMergeScw = mg2oMergeSlw;

                        //mpTracker->SetStepByStep(true);

                        Verbose::PrintMess("*Merge detected", Verbose::VERBOSITY_QUIET);

#ifdef REGISTER_TIMES
                        std::chrono::steady_clock::time_point time_StartMerge = std::chrono::steady_clock::now();

#endif
                        // TODO UNCOMMENT
                        if (mpTracker->mSensor==System::IMU_MONOCULAR ||mpTracker->mSensor==System::IMU_STEREO || mpTracker->mSensor==System::IMU_RGBD)
                            mergeAccepted=MergeLocal2();
                        else
                            mergeAccepted=MergeLocal();

#ifdef REGISTER_TIMES
                        std::chrono::steady_clock::time_point time_EndMerge = std::chrono::steady_clock::now();

                        double timeMergeTotal = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndMerge - time_StartMerge).count();
                        vdMergeTotal_ms.push_back(timeMergeTotal);
                        if(mergeAccepted) nMerges += 1;
#endif

                        if(mergeAccepted) Verbose::PrintMess("Merge finished!", Verbose::VERBOSITY_QUIET);
                    }

                    if(mergeAccepted) {
                        vdPR_CurrentTime.push_back(mpCurrentKF->mTimeStamp);
                        vdPR_MatchedTime.push_back(mpMergeMatchedKF->mTimeStamp);
                        vnPR_TypeRecogn.push_back(1);
                    }

                    // Reset all variables
                    mpMergeLastCurrentKF->SetErase();
                    mpMergeMatchedKF->SetErase();
                    mnMergeNumCoincidences = 0;
                    mvpMergeMatchedMPs.clear();
                    mvpMergeMPs.clear();
                    mnMergeNumNotFound = 0;
                    mbMergeDetected = false;

                    if(mbLoopDetected)
                    {
                        // Reset Loop variables
                        mpLoopLastCurrentKF->SetErase();
                        mpLoopMatchedKF->SetErase();
                        mnLoopNumCoincidences = 0;
                        mvpLoopMatchedMPs.clear();
                        mvpLoopMPs.clear();
                        mnLoopNumNotFound = 0;
                        mbLoopDetected = false;
                    }

                }

                if(mbLoopDetected)
                {
                    bool bGoodLoop = true;

                    Verbose::PrintMess("*Loop detected", Verbose::VERBOSITY_QUIET);

                    mg2oLoopScw = mg2oLoopSlw; //*mvg2oSim3LoopTcw[nCurrentIndex];
                    if(mpCurrentKF->GetMap()->IsInertial())
                    {
                        Sophus::SE3d Twc = mpCurrentKF->GetPoseInverse().cast<double>();
                        g2o::Sim3 g2oTwc(Twc.unit_quaternion(),Twc.translation(),1.0);
                        g2o::Sim3 g2oSww_new = g2oTwc*mg2oLoopScw;

                        Eigen::Vector3d phi = LogSO3(g2oSww_new.rotation().toRotationMatrix());
                        cout << "phi = " << phi.transpose() << endl; 
                        if (fabs(phi(0))<0.008f && fabs(phi(1))<0.008f && fabs(phi(2))<0.349f)
                        {
                            if(mpCurrentKF->GetMap()->IsInertial())
                            {
                                // If inertial, force only yaw
                                if ((mpTracker->mSensor==System::IMU_MONOCULAR ||mpTracker->mSensor==System::IMU_STEREO || mpTracker->mSensor==System::IMU_RGBD) &&
                                        mpCurrentKF->GetMap()->GetIniertialBA2())
                                {
                                    phi(0)=0;
                                    phi(1)=0;
                                    g2oSww_new = g2o::Sim3(ExpSO3(phi),g2oSww_new.translation(),1.0);
                                    mg2oLoopScw = g2oTwc.inverse()*g2oSww_new;
                                }
                            }

                        }
                        else
                        {
                            cout << "BAD LOOP!!!" << endl;
                            bGoodLoop = false;
                        }

                    }

                    if (bGoodLoop) {

                        mvpLoopMapPoints = mvpLoopMPs;

#ifdef REGISTER_TIMES
                        std::chrono::steady_clock::time_point time_StartLoop = std::chrono::steady_clock::now();

#endif
                        const bool loopAccepted=CorrectLoop();
#ifdef REGISTER_TIMES
                        std::chrono::steady_clock::time_point time_EndLoop = std::chrono::steady_clock::now();

                        double timeLoopTotal = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndLoop - time_StartLoop).count();
                        vdLoopTotal_ms.push_back(timeLoopTotal);
                        if(loopAccepted) nLoop += 1;
#endif

                        if(loopAccepted) {
                            mnNumCorrection += 1;
                            vdPR_CurrentTime.push_back(mpCurrentKF->mTimeStamp);
                            vdPR_MatchedTime.push_back(mpLoopMatchedKF->mTimeStamp);
                            vnPR_TypeRecogn.push_back(0);
                        }
                    }

                    // Reset all variables
                    mpLoopLastCurrentKF->SetErase();
                    mpLoopMatchedKF->SetErase();
                    mnLoopNumCoincidences = 0;
                    mvpLoopMatchedMPs.clear();
                    mvpLoopMPs.clear();
                    mnLoopNumNotFound = 0;
                    mbLoopDetected = false;
                }

            }
            mpLastCurrentKF = mpCurrentKF;
        }

        ResetIfRequested();

        if(CheckFinish()){
            // Offline callers stop Tracking before requesting finish. Drain
            // every keyframe already published by Local Mapping so a return
            // to the start at the tail still receives the normal three-KF
            // place-recognition and loop-correction sequence. Also wait for
            // the correction's asynchronous GBA before publishing the Atlas.
            if(!mbActiveLC) {
                unique_lock<mutex> lock(mMutexLoopQueue);
                mlpLoopKeyFrameQueue.clear();
            }
            if(!CheckNewKeyFrames() && !isRunningGBA())
                break;
        }

        usleep(5000);
    }

    SetFinish();
}

void LoopClosing::InsertKeyFrame(KeyFrame *pKF)
{
    if(!mbActiveLC) return;
    unique_lock<mutex> lock(mMutexLoopQueue);
    // Marker-factor keyframes without a background landmark neighborhood are
    // useful to marker BA but cannot support geometric place recognition.
    if(pKF->mnId!=0 && pKF->TrackedMapPoints(1)>=15)
        mlpLoopKeyFrameQueue.push_back(pKF);
}

bool LoopClosing::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexLoopQueue);
    return(!mlpLoopKeyFrameQueue.empty());
}

void LoopClosing::RunOfflineLoopSearch()
{
    const char* requested=std::getenv("ORB_SLAM3_OFFLINE_LOOP_SEARCH");
    if(!requested || !mbActiveLC) return;
    const std::string mode(requested);
    if(mode!="1" && mode!="audit") return;
    if(!isFinished() || !mpLocalMapper->isFinished() || isRunningGBA()) return;
    const auto start=std::chrono::steady_clock::now();
    auto keyframes=mpAtlas->GetAllKeyFrames();
    std::sort(keyframes.begin(),keyframes.end(),KeyFrame::lId);
    // A fresh past-only index avoids querying a keyframe against itself or
    // inserting duplicate postings into the live database. Reset the shared
    // per-query scratch IDs because these historical IDs have been used before.
    KeyFrameDatabase database(*mpORBVocabulary);
    for(KeyFrame* kf:keyframes) if(kf)
        kf->mnPlaceRecognitionQuery=std::numeric_limits<unsigned long>::max();
    mbOfflineLoopSearch=true;
    unsigned int queries=0, accepted=0, validated=0;
    for(KeyFrame* query:keyframes) {
        if(!query || query->isBad()) continue;
        Map* map=query->GetMap();
        // Historical recall is only for staged same-map metric correction.
        // Nonmetric maps, cross-map merge and inertial correction keep native paths.
        if(IsStagedMetricLoop(map,map) && !map->IsBad() && !map->IsInertial() &&
           query->TrackedMapPoints(1)>=15) {
            mpCurrentKF=query;
            query->SetNotErase();
            std::vector<KeyFrame*> candidates, merges;
            database.DetectNBestCandidates(query,candidates,merges,12);
            const auto existing=query->GetLoopEdges();
            candidates.erase(std::remove_if(candidates.begin(),candidates.end(),
                [&](KeyFrame* kf){return !kf || kf==query || kf->isBad() ||
                    kf->GetMap()!=map || existing.count(kf);}),candidates.end());
            // Offline recall supplement: covisibility aggregation can collapse
            // all BoW candidates into one location. Rank individual past views
            // and retain temporally diverse alternatives; these are candidates
            // only and still face the unchanged complete geometric validation.
            std::vector<std::pair<double,KeyFrame*>> ranked;
            const auto connected=query->GetConnectedKeyFrames();
            for(KeyFrame* past:keyframes) {
                if(!past || past->isBad() || past->GetMap()!=map || past->mnId>=query->mnId ||
                   query->mTimeStamp-past->mTimeStamp<10.0 || connected.count(past) ||
                   existing.count(past)) continue;
                const double score=mpORBVocabulary->score(query->mBowVec,past->mBowVec);
                if(score>0) ranked.emplace_back(score,past);
            }
            std::sort(ranked.begin(),ranked.end(),[](const auto& a,const auto& b) {
                return a.first!=b.first?a.first>b.first:a.second->mnId<b.second->mnId;
            });
            int extra=0;
            for(const auto& candidate:ranked) {
                bool nearby=false;
                for(KeyFrame* kept:candidates)
                    if(std::abs(kept->mTimeStamp-candidate.second->mTimeStamp)<2.0) {nearby=true;break;}
                if(nearby) continue;
                candidates.push_back(candidate.second);
                if(++extra>=32) break;
            }
            ++queries;
            std::cout << "OFFLINE_LOOP_QUERY query=" << query->mnId
                      << " timestamp=" << query->mTimeStamp
                      << " candidates=" << candidates.size() << std::endl;
            KeyFrame* matched=nullptr;
            KeyFrame* last=nullptr;
            int coincidences=0;
            g2o::Sim3 transform;
            std::vector<MapPoint*> points,matches;
            const bool valid=!candidates.empty() && DetectCommonRegionsFromBoW(
                candidates,matched,last,transform,coincidences,points,matches);
            if(valid) {
                ++validated;
                bool committed=false;
                if(mode=="1") {
                    mpLoopMatchedKF=matched;
                    mg2oLoopScw=transform;
                    mvpLoopMapPoints=points;
                    mvpLoopMatchedMPs=matches;
                    committed=CorrectLoop();
                    // Ordinary monocular correction may launch asynchronous BA.
                    while(isRunningGBA()) usleep(1000);
                    if(committed) ++accepted;
                }
                std::cout << "OFFLINE_LOOP_RESULT query=" << query->mnId
                    << " matched=" << matched->mnId << " committed=" << committed
                    << " audit=" << (mode=="audit") << std::endl;
            }
            if(matched) matched->SetErase();
            query->SetErase();
        }
        database.add(query);
    }
    mbOfflineLoopSearch=false;
    const double ms=std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now()-start).count();
    std::cout << "OFFLINE_LOOP_SUMMARY queries=" << queries << " validated=" << validated
              << " committed=" << accepted << " elapsed_ms=" << ms << std::endl;
}

bool LoopClosing::NewDetectCommonRegions()
{
    const char* incrementalMode=std::getenv("ORB_SLAM3_INCREMENTAL_LOOP_SEARCH");
    const bool incremental=incrementalMode && std::string(incrementalMode)=="1";
    // To deactivate placerecognition. No loopclosing nor merging will be performed
    if(!mbActiveLC)
        return false;

    {
        unique_lock<mutex> lock(mMutexLoopQueue);
        mpCurrentKF = mlpLoopKeyFrameQueue.front();
        mlpLoopKeyFrameQueue.pop_front();
        // Avoid that a keyframe can be erased while it is being process by this thread
        mpCurrentKF->SetNotErase();
        mpCurrentKF->mbCurrentPlaceRecognition = true;

        mpLastMap = mpCurrentKF->GetMap();
    }

    if(mpLastMap->IsInertial() && !mpLastMap->GetIniertialBA2())
    {
        mpKeyFrameDB->add(mpCurrentKF);
        mpCurrentKF->SetErase();
        return false;
    }

    if(mpTracker->mSensor == System::STEREO && mpLastMap->GetAllKeyFrames().size() < 5) //12
    {
        // cout << "LoopClousure: Stereo KF inserted without check: " << mpCurrentKF->mnId << endl;
        mpKeyFrameDB->add(mpCurrentKF);
        mpCurrentKF->SetErase();
        return false;
    }

    if(mpLastMap->GetAllKeyFrames().size() < 12)
    {
        // cout << "LoopClousure: Stereo KF inserted without check, map is small: " << mpCurrentKF->mnId << endl;
        mpKeyFrameDB->add(mpCurrentKF);
        mpCurrentKF->SetErase();
        return false;
    }

    //cout << "LoopClousure: Checking KF: " << mpCurrentKF->mnId << endl;

    //Check the last candidates with geometric validation
    // Loop candidates
    bool bLoopDetectedInKF = false;
    bool bCheckSpatial = false;

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartEstSim3_1 = std::chrono::steady_clock::now();
#endif
    if(mnLoopNumCoincidences > 0)
    {
        bCheckSpatial = true;
        // Find from the last KF candidates
        Sophus::SE3d mTcl = (mpCurrentKF->GetPose() * mpLoopLastCurrentKF->GetPoseInverse()).cast<double>();
        g2o::Sim3 gScl(mTcl.unit_quaternion(),mTcl.translation(),1.0);
        g2o::Sim3 gScw = gScl * mg2oLoopSlw;
        int numProjMatches = 0;
        vector<MapPoint*> vpMatchedMPs;
        bool bCommonRegion = DetectAndReffineSim3FromLastKF(mpCurrentKF, mpLoopMatchedKF, gScw, numProjMatches, mvpLoopMPs, vpMatchedMPs);
        if(bCommonRegion)
        {

            bLoopDetectedInKF = true;

            mnLoopNumCoincidences++;
            mpLoopLastCurrentKF->SetErase();
            mpLoopLastCurrentKF = mpCurrentKF;
            mg2oLoopSlw = gScw;
            mvpLoopMatchedMPs = vpMatchedMPs;


            mbLoopDetected = mnLoopNumCoincidences >= 3;
            mnLoopNumNotFound = 0;

            if(!mbLoopDetected)
            {
                cout << "PR: Loop detected with Reffine Sim3" << endl;
            }
        }
        else
        {
            bLoopDetectedInKF = false;

            mnLoopNumNotFound++;
            if(mnLoopNumNotFound >= 2)
            {
                mpLoopLastCurrentKF->SetErase();
                mpLoopMatchedKF->SetErase();
                mnLoopNumCoincidences = 0;
                mvpLoopMatchedMPs.clear();
                mvpLoopMPs.clear();
                mnLoopNumNotFound = 0;
            }

        }
    }

    //Merge candidates
    bool bMergeDetectedInKF = false;
    if(mnMergeNumCoincidences > 0)
    {
        // Find from the last KF candidates
        Sophus::SE3d mTcl = (mpCurrentKF->GetPose() * mpMergeLastCurrentKF->GetPoseInverse()).cast<double>();

        g2o::Sim3 gScl(mTcl.unit_quaternion(), mTcl.translation(), 1.0);
        g2o::Sim3 gScw = gScl * mg2oMergeSlw;
        int numProjMatches = 0;
        vector<MapPoint*> vpMatchedMPs;
        bool bCommonRegion = DetectAndReffineSim3FromLastKF(mpCurrentKF, mpMergeMatchedKF, gScw, numProjMatches, mvpMergeMPs, vpMatchedMPs);
        if(bCommonRegion)
        {
            bMergeDetectedInKF = true;

            mnMergeNumCoincidences++;
            mpMergeLastCurrentKF->SetErase();
            mpMergeLastCurrentKF = mpCurrentKF;
            mg2oMergeSlw = gScw;
            mvpMergeMatchedMPs = vpMatchedMPs;

            mbMergeDetected = mnMergeNumCoincidences >= 3;
        }
        else
        {
            mbMergeDetected = false;
            bMergeDetectedInKF = false;

            mnMergeNumNotFound++;
            if(mnMergeNumNotFound >= 2)
            {
                mpMergeLastCurrentKF->SetErase();
                mpMergeMatchedKF->SetErase();
                mnMergeNumCoincidences = 0;
                mvpMergeMatchedMPs.clear();
                mvpMergeMPs.clear();
                mnMergeNumNotFound = 0;
            }


        }
    }  
#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_EndEstSim3_1 = std::chrono::steady_clock::now();

        double timeEstSim3 = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndEstSim3_1 - time_StartEstSim3_1).count();
#endif

    if(mbMergeDetected || mbLoopDetected)
    {
#ifdef REGISTER_TIMES
        vdEstSim3_ms.push_back(timeEstSim3);
#endif
        mpKeyFrameDB->add(mpCurrentKF);
        return true;
    }

    //TODO: This is only necessary if we use a minimun score for pick the best candidates
    const vector<KeyFrame*> vpConnectedKeyFrames = mpCurrentKF->GetVectorCovisibleKeyFrames();

    // Extract candidates from the bag of words
    vector<KeyFrame*> vpMergeBowCand, vpLoopBowCand;
    if(!bMergeDetectedInKF || !bLoopDetectedInKF)
    {
        // Search in BoW
#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_StartQuery = std::chrono::steady_clock::now();
#endif
        Map* currentMap=mpCurrentKF->GetMap();
        const bool enhancedRecall=incremental && IsStagedMetricLoop(currentMap,currentMap) &&
            !currentMap->IsInertial();
        mpKeyFrameDB->DetectNBestCandidates(mpCurrentKF, vpLoopBowCand, vpMergeBowCand,
                                          enhancedRecall ? 12 : 3);
        // The extra candidates are same-map only; cross-map merge keeps top-3.
        if(vpMergeBowCand.size()>3) vpMergeBowCand.resize(3);
        if(enhancedRecall) {
            // Query only already observed frames. Use the same recall supplement
            // as finalization, but commit through the ordinary loop worker now.
            const auto connected=mpCurrentKF->GetConnectedKeyFrames();
            const auto existing=mpCurrentKF->GetLoopEdges();
            std::vector<std::pair<double,KeyFrame*>> ranked;
            for(KeyFrame* past:mpCurrentKF->GetMap()->GetAllKeyFrames()) {
                if(!past || past->isBad() || past->mnId>=mpCurrentKF->mnId ||
                   mpCurrentKF->mTimeStamp-past->mTimeStamp<10.0 ||
                   connected.count(past) || existing.count(past)) continue;
                const double score=mpORBVocabulary->score(mpCurrentKF->mBowVec,past->mBowVec);
                if(score>0) ranked.emplace_back(score,past);
            }
            std::sort(ranked.begin(),ranked.end(),[](const auto& a,const auto& b) {
                return a.first!=b.first?a.first>b.first:a.second->mnId<b.second->mnId;
            });
            int extra=0;
            for(const auto& candidate:ranked) {
                bool nearby=false;
                for(KeyFrame* kept:vpLoopBowCand)
                    if(std::abs(kept->mTimeStamp-candidate.second->mTimeStamp)<2.0) {nearby=true;break;}
                if(nearby) continue;
                vpLoopBowCand.push_back(candidate.second);
                if(++extra>=32) break;
            }
        }
#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_EndQuery = std::chrono::steady_clock::now();

        double timeDataQuery = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndQuery - time_StartQuery).count();
        vdDataQuery_ms.push_back(timeDataQuery);
#endif
    }

#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_StartEstSim3_2 = std::chrono::steady_clock::now();
#endif
    // Check the BoW candidates if the geometric candidate list is empty
    //Loop candidates
    if(!bLoopDetectedInKF && !vpLoopBowCand.empty())
    {
        mbLoopDetected = DetectCommonRegionsFromBoW(vpLoopBowCand, mpLoopMatchedKF, mpLoopLastCurrentKF, mg2oLoopSlw, mnLoopNumCoincidences, mvpLoopMPs, mvpLoopMatchedMPs);
    }
    // Merge candidates
    if(!bMergeDetectedInKF && !vpMergeBowCand.empty())
    {
        mbMergeDetected = DetectCommonRegionsFromBoW(vpMergeBowCand, mpMergeMatchedKF, mpMergeLastCurrentKF, mg2oMergeSlw, mnMergeNumCoincidences, mvpMergeMPs, mvpMergeMatchedMPs);
    }

#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_EndEstSim3_2 = std::chrono::steady_clock::now();

        timeEstSim3 += std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndEstSim3_2 - time_StartEstSim3_2).count();
        vdEstSim3_ms.push_back(timeEstSim3);
#endif

    mpKeyFrameDB->add(mpCurrentKF);

    if(mbMergeDetected || mbLoopDetected)
    {
        return true;
    }

    mpCurrentKF->SetErase();
    mpCurrentKF->mbCurrentPlaceRecognition = false;

    return false;
}

bool LoopClosing::DetectAndReffineSim3FromLastKF(KeyFrame* pCurrentKF, KeyFrame* pMatchedKF, g2o::Sim3 &gScw, int &nNumProjMatches,
                                                 std::vector<MapPoint*> &vpMPs, std::vector<MapPoint*> &vpMatchedMPs)
{
    set<MapPoint*> spAlreadyMatchedMPs;
    nNumProjMatches = FindMatchesByProjection(pCurrentKF, pMatchedKF, gScw, spAlreadyMatchedMPs, vpMPs, vpMatchedMPs);

    int nProjMatches = 30;
    int nProjOptMatches = 50;
    int nProjMatchesRep = 100;

    if(nNumProjMatches >= nProjMatches)
    {
        //Verbose::PrintMess("Sim3 reffine: There are " + to_string(nNumProjMatches) + " initial matches ", Verbose::VERBOSITY_DEBUG);
        Sophus::SE3d mTwm = pMatchedKF->GetPoseInverse().cast<double>();
        g2o::Sim3 gSwm(mTwm.unit_quaternion(),mTwm.translation(),1.0);
        g2o::Sim3 gScm = gScw * gSwm;
        Eigen::Matrix<double, 7, 7> mHessian7x7;

        // A loop inside an already metric marker map is an SE(3) closure.
        // Fix the scale during hypothesis refinement too, not only later in
        // OptimizeEssentialGraph.  Otherwise the accepted Sim(3) seed can
        // rescale the current covisible window before the fixed-scale graph
        // optimization gets a chance to run.
        const bool metricGauge =
            (pCurrentKF->GetMap()==pMatchedKF->GetMap() &&
             pCurrentKF->GetMap()->HasMetricTagGeometry()) ||
            ShareConsistentMetricMarker(pCurrentKF->GetMap(),pMatchedKF->GetMap());
        bool bFixedScale = mbFixScale || metricGauge;
        if(!mbFixScale && MetricLoopScaleExperiment(mbOfflineLoopSearch,pCurrentKF->GetMap(),pMatchedKF->GetMap()))
            bFixedScale=false;
        if(mpTracker->mSensor==System::IMU_MONOCULAR && !pCurrentKF->GetMap()->GetIniertialBA2())
            bFixedScale=false;
        int numOptMatches = Optimizer::OptimizeSim3(mpCurrentKF, pMatchedKF, vpMatchedMPs, gScm, 10, bFixedScale, mHessian7x7, true);

        //Verbose::PrintMess("Sim3 reffine: There are " + to_string(numOptMatches) + " matches after of the optimization ", Verbose::VERBOSITY_DEBUG);

        if(numOptMatches > nProjOptMatches)
        {
            // OptimizeSim3 returns matched-camera -> current-camera. Compose
            // with world -> matched-camera, retaining the optimized scale.
            // Reusing the old seed (and forcing s=1) discards refinement.
            g2o::Sim3 gScw_estimation = gScm * gSwm.inverse();

            vector<MapPoint*> vpMatchedMP;
            vpMatchedMP.resize(mpCurrentKF->GetMapPointMatches().size(), static_cast<MapPoint*>(NULL));

            nNumProjMatches = FindMatchesByProjection(pCurrentKF, pMatchedKF, gScw_estimation, spAlreadyMatchedMPs, vpMPs, vpMatchedMPs);
            if(nNumProjMatches >= nProjMatchesRep)
            {
                gScw = gScw_estimation;
                return true;
            }
        }
    }
    return false;
}

bool LoopClosing::DetectCommonRegionsFromBoW(std::vector<KeyFrame*> &vpBowCand, KeyFrame* &pMatchedKF2, KeyFrame* &pLastCurrentKF, g2o::Sim3 &g2oScw,
                                             int &nNumCoincidences, std::vector<MapPoint*> &vpMPs, std::vector<MapPoint*> &vpMatchedMPs)
{
    int nBoWMatches = 20;
    int nBoWInliers = 15;
    int nSim3Inliers = 20;
    int nProjMatches = 50;
    int nProjOptMatches = 80;

    set<KeyFrame*> spConnectedKeyFrames = mpCurrentKF->GetConnectedKeyFrames();

    int nNumCovisibles = 10;

    ORBmatcher matcherBoW(0.9, true);
    ORBmatcher matcher(0.75, true);

    // Varibles to select the best numbe
    KeyFrame* pBestMatchedKF;
    int nBestMatchesReproj = 0;
    int nBestNumCoindicendes = 0;
    g2o::Sim3 g2oBestScw;
    std::vector<MapPoint*> vpBestMapPoints;
    std::vector<MapPoint*> vpBestMatchedMapPoints;

    int numCandidates = vpBowCand.size();
    vector<int> vnStage(numCandidates, 0);
    vector<int> vnMatchesStage(numCandidates, 0);

    int index = 0;
    //Verbose::PrintMess("BoW candidates: There are " + to_string(vpBowCand.size()) + " possible candidates ", Verbose::VERBOSITY_DEBUG);
    for(KeyFrame* pKFi : vpBowCand)
    {
        OfflineCandidateDiagnostic diagnostic{mbOfflineLoopSearch,mpCurrentKF->mnId,
            pKFi?pKFi->mnId:0};
        if(!pKFi || pKFi->isBad() || pKFi->TrackedMapPoints(1)<15)
            continue;

        // std::cout << "KF candidate: " << pKFi->mnId << std::endl;
        // Current KF against KF with covisibles version
        std::vector<KeyFrame*> vpCovKFi = pKFi->GetBestCovisibilityKeyFrames(nNumCovisibles);
        if(vpCovKFi.empty())
        {
            std::cout << "Covisible list empty" << std::endl;
            vpCovKFi.push_back(pKFi);
        }
        else
        {
            vpCovKFi.push_back(vpCovKFi[0]);
            vpCovKFi[0] = pKFi;
        }


        bool bAbortByNearKF = false;
        for(int j=0; j<vpCovKFi.size(); ++j)
        {
            if(spConnectedKeyFrames.find(vpCovKFi[j]) != spConnectedKeyFrames.end())
            {
                bAbortByNearKF = true;
                break;
            }
        }
        if(bAbortByNearKF)
        {
            diagnostic.stage="connected_neighborhood";
            //std::cout << "Check BoW aborted because is close to the matched one " << std::endl;
            continue;
        }
        //std::cout << "Check BoW continue because is far to the matched one " << std::endl;


        std::vector<std::vector<MapPoint*> > vvpMatchedMPs;
        vvpMatchedMPs.resize(vpCovKFi.size());
        std::set<MapPoint*> spMatchedMPi;
        int numBoWMatches = 0;

        KeyFrame* pMostBoWMatchesKF = pKFi;
        int nMostBoWNumMatches = 0;

        std::vector<MapPoint*> vpMatchedPoints = std::vector<MapPoint*>(mpCurrentKF->GetMapPointMatches().size(), static_cast<MapPoint*>(NULL));
        std::vector<KeyFrame*> vpKeyFrameMatchedMP = std::vector<KeyFrame*>(mpCurrentKF->GetMapPointMatches().size(), static_cast<KeyFrame*>(NULL));

        int nIndexMostBoWMatchesKF=0;
        for(int j=0; j<vpCovKFi.size(); ++j)
        {
            if(!vpCovKFi[j] || vpCovKFi[j]->isBad())
                continue;

            int num = matcherBoW.SearchByBoW(mpCurrentKF, vpCovKFi[j], vvpMatchedMPs[j]);
            if (num > nMostBoWNumMatches)
            {
                nMostBoWNumMatches = num;
                nIndexMostBoWMatchesKF = j;
            }
        }

        // Several neighbors may match the same current feature. Do not let
        // traversal order overwrite it with a worse match or count it twice.
        std::vector<std::tuple<int,int,int,MapPoint*>> rankedMatches;
        for(int j=0; j<vpCovKFi.size(); ++j)
            for(int k=0; k<vvpMatchedMPs[j].size(); ++k) {
                MapPoint* point=vvpMatchedMPs[j][k];
                if(!point || point->isBad()) continue;
                rankedMatches.emplace_back(ORBmatcher::DescriptorDistance(
                    mpCurrentKF->mDescriptors.row(k),point->GetDescriptor()),k,j,point);
            }
        std::sort(rankedMatches.begin(),rankedMatches.end(),[](const auto& a,const auto& b) {
            if(std::get<0>(a)!=std::get<0>(b)) return std::get<0>(a)<std::get<0>(b);
            if(std::get<1>(a)!=std::get<1>(b)) return std::get<1>(a)<std::get<1>(b);
            return std::get<3>(a)->mnId<std::get<3>(b)->mnId;
        });
        for(const auto& match:rankedMatches) {
            const int k=std::get<1>(match),j=std::get<2>(match);
            MapPoint* point=std::get<3>(match);
            if(vpMatchedPoints[k] || !spMatchedMPi.insert(point).second) continue;
            vpMatchedPoints[k]=point;vpKeyFrameMatchedMP[k]=vpCovKFi[j];++numBoWMatches;
        }
        pMostBoWMatchesKF = vpCovKFi[nIndexMostBoWMatchesKF];

        diagnostic.bow=numBoWMatches;
        diagnostic.stage="bow_matches";
        if(numBoWMatches >= nBoWMatches) // TODO pick a good threshold
        {
            // Geometric validation
            // Preserve scale only for a visual loop within the same metric
            // map.  Cross-map merge candidates must retain Sim(3) freedom so
            // an arbitrary-scale map can still align to a metric one.
            const bool metricGauge =
                (mpCurrentKF->GetMap()==pMostBoWMatchesKF->GetMap() &&
                 mpCurrentKF->GetMap()->HasMetricTagGeometry()) ||
                ShareConsistentMetricMarker(mpCurrentKF->GetMap(),pMostBoWMatchesKF->GetMap());
            bool bFixedScale = mbFixScale || metricGauge;
            if(!mbFixScale && MetricLoopScaleExperiment(mbOfflineLoopSearch,mpCurrentKF->GetMap(),pMostBoWMatchesKF->GetMap()))
                bFixedScale=false;
            if(mpTracker && mpTracker->mSensor==System::IMU_MONOCULAR && !mpCurrentKF->GetMap()->GetIniertialBA2())
                bFixedScale=false;

            Sim3Solver solver = Sim3Solver(mpCurrentKF, pMostBoWMatchesKF, vpMatchedPoints, bFixedScale, vpKeyFrameMatchedMP);
            solver.SetRansacParameters(0.99, nBoWInliers, 300); // at least 15 inliers

            bool bNoMore = false;
            vector<bool> vbInliers;
            int nInliers=0;
            bool bConverge = false;
            Eigen::Matrix4f mTcm;
            while(!bConverge && !bNoMore)
            {
                mTcm = solver.iterate(20,bNoMore, vbInliers, nInliers, bConverge);
                //Verbose::PrintMess("BoW guess: Solver achieve " + to_string(nInliers) + " geometrical inliers among " + to_string(nBoWInliers) + " BoW matches", Verbose::VERBOSITY_DEBUG);
            }

            diagnostic.ransac=nInliers;
            diagnostic.stage="ransac_geometry";
            if(bConverge)
            {
                //std::cout << "Check BoW: SolverSim3 converged" << std::endl;

                //Verbose::PrintMess("BoW guess: Convergende with " + to_string(nInliers) + " geometrical inliers among " + to_string(nBoWInliers) + " BoW matches", Verbose::VERBOSITY_DEBUG);
                // Match by reprojection
                vpCovKFi.clear();
                vpCovKFi = pMostBoWMatchesKF->GetBestCovisibilityKeyFrames(nNumCovisibles);
                vpCovKFi.push_back(pMostBoWMatchesKF);
                set<KeyFrame*> spCheckKFs(vpCovKFi.begin(), vpCovKFi.end());

                //std::cout << "There are " << vpCovKFi.size() <<" near KFs" << std::endl;

                set<MapPoint*> spMapPoints;
                vector<MapPoint*> vpMapPoints;
                vector<KeyFrame*> vpKeyFrames;
                for(KeyFrame* pCovKFi : vpCovKFi)
                {
                    for(MapPoint* pCovMPij : pCovKFi->GetMapPointMatches())
                    {
                        if(!pCovMPij || pCovMPij->isBad())
                            continue;

                        if(spMapPoints.find(pCovMPij) == spMapPoints.end())
                        {
                            spMapPoints.insert(pCovMPij);
                            vpMapPoints.push_back(pCovMPij);
                            vpKeyFrames.push_back(pCovKFi);
                        }
                    }
                }

                //std::cout << "There are " << vpKeyFrames.size() <<" KFs which view all the mappoints" << std::endl;

                g2o::Sim3 gScm(solver.GetEstimatedRotation().cast<double>(),solver.GetEstimatedTranslation().cast<double>(), (double) solver.GetEstimatedScale());
                g2o::Sim3 gSmw(pMostBoWMatchesKF->GetRotation().cast<double>(),pMostBoWMatchesKF->GetTranslation().cast<double>(),1.0);
                g2o::Sim3 gScw = gScm*gSmw; // Similarity matrix of current from the world position
                Sophus::Sim3f mScw = Converter::toSophus(gScw);

                vector<MapPoint*> vpMatchedMP;
                vpMatchedMP.resize(mpCurrentKF->GetMapPointMatches().size(), static_cast<MapPoint*>(NULL));
                vector<KeyFrame*> vpMatchedKF;
                vpMatchedKF.resize(mpCurrentKF->GetMapPointMatches().size(), static_cast<KeyFrame*>(NULL));
                int numProjMatches = matcher.SearchByProjection(mpCurrentKF, mScw, vpMapPoints, vpKeyFrames, vpMatchedMP, vpMatchedKF, 8, 1.5);
                //cout <<"BoW: " << numProjMatches << " matches between " << vpMapPoints.size() << " points with coarse Sim3" << endl;

                diagnostic.projection=numProjMatches;
                diagnostic.stage="projection_matches";
                if(numProjMatches >= nProjMatches)
                {
                    // Optimize Sim3 transformation with every matches
                    Eigen::Matrix<double, 7, 7> mHessian7x7;

                    const bool metricGauge =
                        (mpCurrentKF->GetMap()==pKFi->GetMap() &&
                         mpCurrentKF->GetMap()->HasMetricTagGeometry()) ||
                        ShareConsistentMetricMarker(mpCurrentKF->GetMap(),pKFi->GetMap());
                    bool bFixedScale = mbFixScale || metricGauge;
                    if(!mbFixScale && MetricLoopScaleExperiment(mbOfflineLoopSearch,mpCurrentKF->GetMap(),pMostBoWMatchesKF->GetMap()))
                        bFixedScale=false;
                    if(mpTracker && mpTracker->mSensor==System::IMU_MONOCULAR && !mpCurrentKF->GetMap()->GetIniertialBA2())
                        bFixedScale=false;

                    // gScm is expressed in the strongest covisible keyframe's
                    // camera coordinates, not necessarily the BoW seed's.
                    int numOptMatches = Optimizer::OptimizeSim3(mpCurrentKF, pMostBoWMatchesKF, vpMatchedMP, gScm, 10, bFixedScale, mHessian7x7, true);

                    diagnostic.optimized=numOptMatches;
                    diagnostic.stage="optimized_inliers";
                    if(numOptMatches >= nSim3Inliers)
                    {
                        g2o::Sim3 gSmw(pMostBoWMatchesKF->GetRotation().cast<double>(),pMostBoWMatchesKF->GetTranslation().cast<double>(),1.0);
                        g2o::Sim3 gScw = gScm*gSmw; // Similarity matrix of current from the world position
                        Sophus::Sim3f mScw = Converter::toSophus(gScw);

                        vector<MapPoint*> vpMatchedMP;
                        vpMatchedMP.resize(mpCurrentKF->GetMapPointMatches().size(), static_cast<MapPoint*>(NULL));
                        int numProjOptMatches = matcher.SearchByProjection(mpCurrentKF, mScw, vpMapPoints, vpMatchedMP, 5, 1.0);

                        diagnostic.refined=numProjOptMatches;
                        diagnostic.stage="refined_projection_matches";
                        // Staged metric loops: a moderate-support candidate may
                        // proceed to the unchanged three-neighbor validation.
                        // Spatial coverage prevents one small texture patch
                        // from supplying all the additional evidence.
                        const char* moderateMode=std::getenv("ORB_SLAM3_MODERATE_LOOP_SUPPORT");
                        bool moderateSupport=false;
                        const char* incrementalMode=std::getenv("ORB_SLAM3_INCREMENTAL_LOOP_SEARCH");
                        const bool incremental=incrementalMode && std::string(incrementalMode)=="1";
                        if((mbOfflineLoopSearch || incremental) &&
                           IsStagedMetricLoop(mpCurrentKF->GetMap(),pMostBoWMatchesKF->GetMap()) &&
                           (!moderateMode || std::string(moderateMode)!="0") &&
                           numProjOptMatches>=50 && numProjOptMatches<nProjOptMatches) {
                            std::set<std::pair<int,int>> cells;
                            const float width=mpCurrentKF->mnMaxX-mpCurrentKF->mnMinX;
                            const float height=mpCurrentKF->mnMaxY-mpCurrentKF->mnMinY;
                            if(width>0 && height>0)
                                for(std::size_t k=0;k<vpMatchedMP.size();++k) {
                                    if(!vpMatchedMP[k] || vpMatchedMP[k]->isBad()) continue;
                                    const auto& pt=mpCurrentKF->mvKeysUn[k].pt;
                                    const int x=int(4*(pt.x-mpCurrentKF->mnMinX)/width);
                                    const int y=int(4*(pt.y-mpCurrentKF->mnMinY)/height);
                                    if(x>=0 && x<4 && y>=0 && y<4) cells.emplace(x,y);
                                }
                            const bool markerEvidence=CandidateMarkerEvidence(mpCurrentKF,pMostBoWMatchesKF,gScw);
                            moderateSupport=cells.size()>=4 && markerEvidence;
                            if(mbOfflineLoopSearch || incremental)
                                std::cout << "LOOP_MARKER_EVIDENCE query=" << mpCurrentKF->mnId
                                          << " matched=" << pMostBoWMatchesKF->mnId
                                          << " timestamp=" << mpCurrentKF->mTimeStamp
                                          << " refined=" << numProjOptMatches << " cells=" << cells.size()
                                          << " candidate_marker_support=" << markerEvidence
                                          << " moderate_allowed=" << moderateSupport << std::endl;
                        }
                        if(numProjOptMatches >= nProjOptMatches || moderateSupport)
                        {
                            int max_x = -1, min_x = 1000000;
                            int max_y = -1, min_y = 1000000;
                            for(MapPoint* pMPi : vpMatchedMP)
                            {
                                if(!pMPi || pMPi->isBad())
                                {
                                    continue;
                                }

                                tuple<size_t,size_t> indexes = pMPi->GetIndexInKeyFrame(pKFi);
                                int index = get<0>(indexes);
                                if(index >= 0)
                                {
                                    int coord_x = pKFi->mvKeysUn[index].pt.x;
                                    if(coord_x < min_x)
                                    {
                                        min_x = coord_x;
                                    }
                                    if(coord_x > max_x)
                                    {
                                        max_x = coord_x;
                                    }
                                    int coord_y = pKFi->mvKeysUn[index].pt.y;
                                    if(coord_y < min_y)
                                    {
                                        min_y = coord_y;
                                    }
                                    if(coord_y > max_y)
                                    {
                                        max_y = coord_y;
                                    }
                                }
                            }

                            int nNumKFs = 0;
                            //vpMatchedMPs = vpMatchedMP;
                            //vpMPs = vpMapPoints;
                            // Check the Sim3 transformation with the current KeyFrame covisibles
                            vector<KeyFrame*> vpCurrentCovKFs = mpCurrentKF->GetBestCovisibilityKeyFrames(nNumCovisibles);

                            int j = 0;
                            while(nNumKFs < 3 && j<vpCurrentCovKFs.size())
                            {
                                KeyFrame* pKFj = vpCurrentCovKFs[j];
                                Sophus::SE3d mTjc = (pKFj->GetPose() * mpCurrentKF->GetPoseInverse()).cast<double>();
                                g2o::Sim3 gSjc(mTjc.unit_quaternion(),mTjc.translation(),1.0);
                                g2o::Sim3 gSjw = gSjc * gScw;
                                int numProjMatches_j = 0;
                                vector<MapPoint*> vpMatchedMPs_j;
                                bool bValid = DetectCommonRegionsFromLastKF(pKFj,pMostBoWMatchesKF, gSjw,numProjMatches_j, vpMapPoints, vpMatchedMPs_j);

                                if(bValid)
                                {
                                    Sophus::SE3f Tc_w = mpCurrentKF->GetPose();
                                    Sophus::SE3f Tw_cj = pKFj->GetPoseInverse();
                                    Sophus::SE3f Tc_cj = Tc_w * Tw_cj;
                                    Eigen::Vector3f vector_dist = Tc_cj.translation();
                                    nNumKFs++;
                                }
                                j++;
                            }

                            diagnostic.neighbors=nNumKFs;
                            diagnostic.stage=nNumKFs>=3?"geometry_validated":"neighbor_consistency";
                            if(nNumKFs < 3)
                            {
                                vnStage[index] = 8;
                                vnMatchesStage[index] = nNumKFs;
                            }

                            // An unvalidated high-count candidate must not
                            // displace a candidate supported by three views.
                            if((nNumKFs>=3 && nBestNumCoindicendes<3) ||
                               ((nNumKFs>=3)==(nBestNumCoindicendes>=3) &&
                                nBestMatchesReproj < numProjOptMatches))
                            {
                                nBestMatchesReproj = numProjOptMatches;
                                nBestNumCoindicendes = nNumKFs;
                                pBestMatchedKF = pMostBoWMatchesKF;
                                g2oBestScw = gScw;
                                vpBestMapPoints = vpMapPoints;
                                vpBestMatchedMapPoints = vpMatchedMP;
                            }
                        }
                    }
                }
            }
            /*else
            {
                Verbose::PrintMess("BoW candidate: it don't match with the current one", Verbose::VERBOSITY_DEBUG);
            }*/
        }
        index++;
    }

    if(nBestMatchesReproj > 0)
    {
        pLastCurrentKF = mpCurrentKF;
        nNumCoincidences = nBestNumCoindicendes;
        pMatchedKF2 = pBestMatchedKF;
        pMatchedKF2->SetNotErase();
        g2oScw = g2oBestScw;
        vpMPs = vpBestMapPoints;
        vpMatchedMPs = vpBestMatchedMapPoints;

        return nNumCoincidences >= 3;
    }
    else
    {
        int maxStage = -1;
        int maxMatched;
        for(int i=0; i<vnStage.size(); ++i)
        {
            if(vnStage[i] > maxStage)
            {
                maxStage = vnStage[i];
                maxMatched = vnMatchesStage[i];
            }
        }
    }
    return false;
}

bool LoopClosing::DetectCommonRegionsFromLastKF(KeyFrame* pCurrentKF, KeyFrame* pMatchedKF, g2o::Sim3 &gScw, int &nNumProjMatches,
                                                std::vector<MapPoint*> &vpMPs, std::vector<MapPoint*> &vpMatchedMPs)
{
    set<MapPoint*> spAlreadyMatchedMPs(vpMatchedMPs.begin(), vpMatchedMPs.end());
    nNumProjMatches = FindMatchesByProjection(pCurrentKF, pMatchedKF, gScw, spAlreadyMatchedMPs, vpMPs, vpMatchedMPs);

    int nProjMatches = 30;
    if(nNumProjMatches >= nProjMatches)
    {
        return true;
    }

    return false;
}

int LoopClosing::FindMatchesByProjection(KeyFrame* pCurrentKF, KeyFrame* pMatchedKFw, g2o::Sim3 &g2oScw,
                                         set<MapPoint*> &spMatchedMPinOrigin, vector<MapPoint*> &vpMapPoints,
                                         vector<MapPoint*> &vpMatchedMapPoints)
{
    int nNumCovisibles = 10;
    vector<KeyFrame*> vpCovKFm = pMatchedKFw->GetBestCovisibilityKeyFrames(nNumCovisibles);
    int nInitialCov = vpCovKFm.size();
    vpCovKFm.push_back(pMatchedKFw);
    set<KeyFrame*> spCheckKFs(vpCovKFm.begin(), vpCovKFm.end());
    set<KeyFrame*> spCurrentCovisbles = pCurrentKF->GetConnectedKeyFrames();
    if(nInitialCov < nNumCovisibles)
    {
        for(int i=0; i<nInitialCov; ++i)
        {
            vector<KeyFrame*> vpKFs = vpCovKFm[i]->GetBestCovisibilityKeyFrames(nNumCovisibles);
            int nInserted = 0;
            int j = 0;
            while(j < vpKFs.size() && nInserted < nNumCovisibles)
            {
                if(spCheckKFs.find(vpKFs[j]) == spCheckKFs.end() && spCurrentCovisbles.find(vpKFs[j]) == spCurrentCovisbles.end())
                {
                    spCheckKFs.insert(vpKFs[j]);
                    ++nInserted;
                }
                ++j;
            }
            vpCovKFm.insert(vpCovKFm.end(), vpKFs.begin(), vpKFs.end());
        }
    }
    set<MapPoint*> spMapPoints;
    vpMapPoints.clear();
    vpMatchedMapPoints.clear();
    for(KeyFrame* pKFi : vpCovKFm)
    {
        for(MapPoint* pMPij : pKFi->GetMapPointMatches())
        {
            if(!pMPij || pMPij->isBad())
                continue;

            if(spMapPoints.find(pMPij) == spMapPoints.end())
            {
                spMapPoints.insert(pMPij);
                vpMapPoints.push_back(pMPij);
            }
        }
    }

    Sophus::Sim3f mScw = Converter::toSophus(g2oScw);
    ORBmatcher matcher(0.9, true);

    vpMatchedMapPoints.resize(pCurrentKF->GetMapPointMatches().size(), static_cast<MapPoint*>(NULL));
    int num_matches = matcher.SearchByProjection(pCurrentKF, mScw, vpMapPoints, vpMatchedMapPoints, 3, 1.5);

    return num_matches;
}

bool LoopClosing::CorrectLoop()
{
    unique_lock<mutex> correctionGate(mpAtlas->mMutexPoseGraphCorrection);
    Map* metricMap=mpCurrentKF->GetMap();
    if(metricMap==mpLoopMatchedKF->GetMap() && metricMap->mbMetric) {
        mpLocalMapper->RequestStop();
        if(isRunningGBA())CancelGlobalBundleAdjustment();
        while(!mpLocalMapper->isStopped())usleep(1000);
        bool accepted=false;std::string reason;
        try {
            unique_lock<mutex> lock(metricMap->mMutexMapUpdate);
            MarkerGraphOptimizer::Options loopOptions;
            if(MetricLoopScaleExperiment(mbOfflineLoopSearch,metricMap,metricMap)) {
                // A verified offline loop can move the entire map. The
                // frame/local-BA budget stops before its background settles.
                // Keep one continuous, bounded LM call (with native early
                // termination), not repeated restarts or weaker validation.
                loopOptions.baIterations=60;
                std::cout << "MARKER_LOOP_BA_BUDGET iterations=60 offline_validated=1" << std::endl;
            }
            const auto proposal=MarkerGraphOptimizer::ProposeVisualLoop(metricMap,mpCurrentKF,
                mpLoopMatchedKF,mg2oLoopScw,mvpLoopMatchedMPs,loopOptions);
            reason=proposal.reason;
            if(proposal.accepted) {
                MarkerGraphEvent event;
                event.frameId=event.candidateFrameId=mpCurrentKF->mnFrameId;
                event.timestamp=event.candidateTimestamp=mpCurrentKF->mTimeStamp;
                accepted=MarkerGraphCoordinator::CommitRefine(*mpAtlas,metricMap,proposal,event);
                if(accepted) {
                    // Association mutation happens only after trial fusion was
                    // optimized and validated against pre-loop observations.
                    for(const auto& pair:proposal.pointAliases)pair.first->Replace(pair.second);
                    for(const auto& pose:proposal.keyframePoses)pose.first->UpdateConnections();
                    mpCurrentKF->AddLoopEdge(mpLoopMatchedKF);
                    mpLoopMatchedKF->AddLoopEdge(mpCurrentKF);
                    mpAtlas->mMarkerGraphEvents.back().revision=metricMap->mnRevision;
                    mLastLoopKFid=mpCurrentKF->mnId;
                } else reason=event.reason.empty()?"staged_loop_commit_validation_failed":event.reason;
            }
        } catch(...) {mpLocalMapper->Release();throw;}
        mpLocalMapper->Release();
        if(!accepted) {
            CorrectionRejection rejection;
            rejection.timestamp=mpCurrentKF->mTimeStamp;rejection.keyframeId=mpCurrentKF->mnId;
            rejection.mapId=rejection.otherMapId=metricMap->GetId();rejection.kind="loop";rejection.reason=reason;
            unique_lock<mutex> lock(mMutexCorrectionRejections);
            rejection.sequence=mvCorrectionRejections.size()+1;mvCorrectionRejections.push_back(rejection);
        }
        std::cout << "STAGED_METRIC_LOOP current=" << mpCurrentKF->mnId << " matched=" << mpLoopMatchedKF->mnId
                  << " timestamp=" << mpCurrentKF->mTimeStamp << " accepted=" << accepted
                  << " reason=" << reason << " seed_scale=" << mg2oLoopScw.scale()
                  << " seed_translation_norm=" << mg2oLoopScw.translation().norm() << std::endl;
        return accepted;
    }
    if(RejectUnsupportedCorrection("loop", mpCurrentKF->GetMap(), mpLoopMatchedKF->GetMap(),
                                   mpCurrentKF->mnId, mpCurrentKF->mTimeStamp))
        return false;
    //cout << "Loop detected!" << endl;

    // Send a stop signal to Local Mapping
    // Avoid new keyframes are inserted while correcting the loop
    mpLocalMapper->RequestStop();
    mpLocalMapper->EmptyQueue(); // Proccess keyframes in the queue

    // If a Global Bundle Adjustment is running, abort it
    if(isRunningGBA())
    {
        cout << "Stoping Global Bundle Adjustment...";
        CancelGlobalBundleAdjustment();
        cout << "  Done!!" << endl;
    }

    // Wait until Local Mapping has effectively stopped
    while(!mpLocalMapper->isStopped())
    {
        usleep(1000);
    }

    // Ensure current keyframe is updated
    //cout << "Start updating connections" << endl;
    //assert(mpCurrentKF->GetMap()->CheckEssentialGraph());
    mpCurrentKF->UpdateConnections();
    //assert(mpCurrentKF->GetMap()->CheckEssentialGraph());

    // Retrive keyframes connected to the current keyframe and compute corrected Sim3 pose by propagation
    mvpCurrentConnectedKFs = mpCurrentKF->GetVectorCovisibleKeyFrames();
    mvpCurrentConnectedKFs.push_back(mpCurrentKF);

    //std::cout << "Loop: number of connected KFs -> " + to_string(mvpCurrentConnectedKFs.size()) << std::endl;

    KeyFrameAndPose CorrectedSim3, NonCorrectedSim3;
    const bool markerGauge=mpCurrentKF->GetMap()->HasMetricTagGeometry();
    const bool metricScaleExperiment=!mbFixScale && MetricLoopScaleExperiment(
        mbOfflineLoopSearch,mpCurrentKF->GetMap(),mpLoopMatchedKF->GetMap());
    CorrectedSim3[mpCurrentKF]=mg2oLoopScw;
    Sophus::SE3f Twc = mpCurrentKF->GetPoseInverse();
    Sophus::SE3f Tcw = mpCurrentKF->GetPose();
    g2o::Sim3 g2oScw(Tcw.unit_quaternion().cast<double>(),Tcw.translation().cast<double>(),1.0);
    NonCorrectedSim3[mpCurrentKF]=g2oScw;

    // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
    Sophus::SE3d correctedTcw(mg2oLoopScw.rotation(),mg2oLoopScw.translation() / mg2oLoopScw.scale());
    mpCurrentKF->SetPose(correctedTcw.cast<float>());

    Map* pLoopMap = mpCurrentKF->GetMap();

#ifdef REGISTER_TIMES
    /*KeyFrame* pKF = mpCurrentKF;
    int numKFinLoop = 0;
    while(pKF && pKF->mnId > mpLoopMatchedKF->mnId)
    {
        pKF = pKF->GetParent();
        numKFinLoop += 1;
    }
    vnLoopKFs.push_back(numKFinLoop);*/

    std::chrono::steady_clock::time_point time_StartFusion = std::chrono::steady_clock::now();
#endif

    {
        // Get Map Mutex
        unique_lock<mutex> lock(pLoopMap->mMutexMapUpdate);

        const bool bImuInit = pLoopMap->isImuInitialized();

        for(vector<KeyFrame*>::iterator vit=mvpCurrentConnectedKFs.begin(), vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++)
        {
            KeyFrame* pKFi = *vit;

            if(pKFi!=mpCurrentKF)
            {
                Sophus::SE3f Tiw = pKFi->GetPose();
                if(pKFi==mpCurrentKF->GetMap()->GetOriginKF() ||
                   (markerGauge && !metricScaleExperiment && pKFi->mbHasTagObservation && pKFi->mbTagObservationActive))
                {
                    g2o::Sim3 anchored(Tiw.unit_quaternion().cast<double>(),
                                      Tiw.translation().cast<double>(),1.0);
                    CorrectedSim3[pKFi]=anchored;
                    NonCorrectedSim3[pKFi]=anchored;
                    continue;
                }
                Sophus::SE3d Tic = (Tiw * Twc).cast<double>();
                g2o::Sim3 g2oSic(Tic.unit_quaternion(),Tic.translation(),1.0);
                g2o::Sim3 g2oCorrectedSiw = g2oSic*mg2oLoopScw;
                //Pose corrected with the Sim3 of the loop closure
                CorrectedSim3[pKFi]=g2oCorrectedSiw;

                // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
                Sophus::SE3d correctedTiw(g2oCorrectedSiw.rotation(),g2oCorrectedSiw.translation() / g2oCorrectedSiw.scale());
                pKFi->SetPose(correctedTiw.cast<float>());

                //Pose without correction
                g2o::Sim3 g2oSiw(Tiw.unit_quaternion().cast<double>(),Tiw.translation().cast<double>(),1.0);
                NonCorrectedSim3[pKFi]=g2oSiw;
            }  
        }

        // Correct all MapPoints obsrved by current keyframe and neighbors, so that they align with the other side of the loop
        for(KeyFrameAndPose::iterator mit=CorrectedSim3.begin(), mend=CorrectedSim3.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;
            g2o::Sim3 g2oCorrectedSiw = mit->second;
            g2o::Sim3 g2oCorrectedSwi = g2oCorrectedSiw.inverse();

            g2o::Sim3 g2oSiw =NonCorrectedSim3[pKFi];

            // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
            /*Sophus::SE3d correctedTiw(g2oCorrectedSiw.rotation(),g2oCorrectedSiw.translation() / g2oCorrectedSiw.scale());
            pKFi->SetPose(correctedTiw.cast<float>());*/

            vector<MapPoint*> vpMPsi = pKFi->GetMapPointMatches();
            for(size_t iMP=0, endMPi = vpMPsi.size(); iMP<endMPi; iMP++)
            {
                MapPoint* pMPi = vpMPsi[iMP];
                if(!pMPi)
                    continue;
                if(pMPi->isBad())
                    continue;
                if(pMPi->mnCorrectedByKF==mpCurrentKF->mnId)
                    continue;

                // Project with non-corrected pose and project back with corrected pose
                Eigen::Vector3d P3Dw = pMPi->GetWorldPos().cast<double>();
                Eigen::Vector3d eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oSiw.map(P3Dw));

                pMPi->SetWorldPos(eigCorrectedP3Dw.cast<float>());
                pMPi->mnCorrectedByKF = mpCurrentKF->mnId;
                pMPi->mnCorrectedReference = pKFi->mnId;
                pMPi->UpdateNormalAndDepth();
            }

            // Correct velocity according to orientation correction
            if(bImuInit)
            {
                Eigen::Quaternionf Rcor = (g2oCorrectedSiw.rotation().inverse()*g2oSiw.rotation()).cast<float>();
                pKFi->SetVelocity(Rcor*pKFi->GetVelocity());
            }

            // Make sure connections are updated
            pKFi->UpdateConnections();
        }
        // TODO Check this index increasement
        mpAtlas->GetCurrentMap()->IncreaseChangeIndex();


        // Start Loop Fusion
        // Update matched map points and replace if duplicated
        for(size_t i=0; i<mvpLoopMatchedMPs.size(); i++)
        {
            if(mvpLoopMatchedMPs[i])
            {
                MapPoint* pLoopMP = mvpLoopMatchedMPs[i];
                MapPoint* pCurMP = mpCurrentKF->GetMapPoint(i);
                if(pCurMP)
                    pCurMP->Replace(pLoopMP);
                else
                {
                    mpCurrentKF->AddMapPoint(pLoopMP,i);
                    pLoopMP->AddObservation(mpCurrentKF,i);
                    pLoopMP->ComputeDistinctiveDescriptors();
                }
            }
        }
        //cout << "LC: end replacing duplicated" << endl;
    }

    // Project MapPoints observed in the neighborhood of the loop keyframe
    // into the current keyframe and neighbors using corrected poses.
    // Fuse duplications.
    SearchAndFuse(CorrectedSim3, mvpLoopMapPoints);

    // After the MapPoint fusion, new links in the covisibility graph will appear attaching both sides of the loop
    map<KeyFrame*, set<KeyFrame*> > LoopConnections;

    for(vector<KeyFrame*>::iterator vit=mvpCurrentConnectedKFs.begin(), vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;
        vector<KeyFrame*> vpPreviousNeighbors = pKFi->GetVectorCovisibleKeyFrames();

        // Update connections. Detect new links.
        pKFi->UpdateConnections();
        LoopConnections[pKFi]=pKFi->GetConnectedKeyFrames();
        for(vector<KeyFrame*>::iterator vit_prev=vpPreviousNeighbors.begin(), vend_prev=vpPreviousNeighbors.end(); vit_prev!=vend_prev; vit_prev++)
        {
            LoopConnections[pKFi].erase(*vit_prev);
        }
        for(vector<KeyFrame*>::iterator vit2=mvpCurrentConnectedKFs.begin(), vend2=mvpCurrentConnectedKFs.end(); vit2!=vend2; vit2++)
        {
            LoopConnections[pKFi].erase(*vit2);
        }
    }

    // Optimize graph
    // Once a map has a validated metric marker gauge, a background-only
    // visual loop may correct SE(3) drift but must not assign independent
    // Sim(3) scales along the keyframe graph. Metric scale changes belong to
    // the marker graph's multi-frame re-anchoring path, which validates raw
    // corners, background residuals and depth before an atomic commit.
    // Leaving scale free here can turn an otherwise plausible long visual
    // loop into temporally discontinuous historical poses.
    bool bFixedScale = mbFixScale || markerGauge;
    if(metricScaleExperiment) {
        bFixedScale=false;
        std::cout << "MARKER_SIM3_LOOP seed_scale=" << mg2oLoopScw.scale()
                  << " origin=" << pLoopMap->GetOriginKF()->mnId << std::endl;
    }
    // TODO CHECK; Solo para el monocular inertial
    if(mpTracker->mSensor==System::IMU_MONOCULAR && !mpCurrentKF->GetMap()->GetIniertialBA2())
        bFixedScale=false;

#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_EndFusion = std::chrono::steady_clock::now();

        double timeFusion = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndFusion - time_StartFusion).count();
        vdLoopFusion_ms.push_back(timeFusion);
#endif
    //cout << "Optimize essential graph" << endl;
    if(pLoopMap->IsInertial() && pLoopMap->isImuInitialized())
    {
        Optimizer::OptimizeEssentialGraph4DoF(pLoopMap, mpLoopMatchedKF, mpCurrentKF, NonCorrectedSim3, CorrectedSim3, LoopConnections);
    }
    else
    {
        //cout << "Loop -> Scale correction: " << mg2oLoopScw.scale() << endl;
        Optimizer::OptimizeEssentialGraph(pLoopMap, mpLoopMatchedKF, mpCurrentKF, NonCorrectedSim3, CorrectedSim3, LoopConnections, bFixedScale);
    }
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndOpt = std::chrono::steady_clock::now();

    double timeOptEss = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndOpt - time_EndFusion).count();
    vdLoopOptEss_ms.push_back(timeOptEss);
#endif

    mpAtlas->InformNewBigChange();

    // Add loop edge
    mpLoopMatchedKF->AddLoopEdge(mpCurrentKF);
    mpCurrentKF->AddLoopEdge(mpLoopMatchedKF);

    const char* rematch=std::getenv("ORB_SLAM3_POST_LOOP_REMATCH");
    if(rematch && (std::string(rematch)=="1" || std::string(rematch)=="audit"))
        PostLoopRematch::Run(pLoopMap,mpLoopMatchedKF,mpCurrentKF,std::string(rematch)=="1");

    // Launch a new thread to perform Global Bundle Adjustment (Only if few keyframes, if not it would take too much time)
    if(!pLoopMap->HasMetricTagGeometry() &&
       (!pLoopMap->isImuInitialized() || (pLoopMap->KeyFramesInMap()<200 && mpAtlas->CountMaps()==1)))
    {
        LaunchGlobalBundleAdjustment(pLoopMap,mpCurrentKF);
    }

    // Loop closed. Release Local Mapping.
    mpLocalMapper->Release();    

    mLastLoopKFid = mpCurrentKF->mnId; //TODO old varible, it is not use in the new algorithm
    return true;
}

bool LoopClosing::MergeLocal()
{
    unique_lock<mutex> correctionGate(mpAtlas->mMutexPoseGraphCorrection);
    if(RejectUnsupportedCorrection("merge", mpCurrentKF->GetMap(), mpMergeMatchedKF->GetMap(),
                                   mpCurrentKF->mnId, mpCurrentKF->mTimeStamp))
        return false;
    int numTemporalKFs = 25; //Temporal KFs in the local window if the map is inertial.

    //Relationship to rebuild the essential graph, it is used two times, first in the local window and later in the rest of the map
    KeyFrame* pNewChild;
    KeyFrame* pNewParent;

    vector<KeyFrame*> vpLocalCurrentWindowKFs;
    vector<KeyFrame*> vpMergeConnectedKFs;

    // Flag that is true only when we stopped a running BA, in this case we need relaunch at the end of the merge
    bool bRelaunchBA = false;

    //Verbose::PrintMess("MERGE-VISUAL: Check Full Bundle Adjustment", Verbose::VERBOSITY_DEBUG);
    // If a Global Bundle Adjustment is running, abort it
    if(isRunningGBA())
    {
        CancelGlobalBundleAdjustment();
        bRelaunchBA = true;
    }

    //Verbose::PrintMess("MERGE-VISUAL: Request Stop Local Mapping", Verbose::VERBOSITY_DEBUG);
    //cout << "Request Stop Local Mapping" << endl;
    mpLocalMapper->RequestStop();
    // Wait until Local Mapping has effectively stopped
    while(!mpLocalMapper->isStopped())
    {
        usleep(1000);
    }
    //cout << "Local Map stopped" << endl;

    mpLocalMapper->EmptyQueue();

    // Merge map will become in the new active map with the local window of KFs and MPs from the current map.
    // Later, the elements of the current map will be transform to the new active map reference, in order to keep real time tracking
    Map* pCurrentMap = mpCurrentKF->GetMap();
    Map* pMergeMap = mpMergeMatchedKF->GetMap();
    const bool metricTargetMerge = pMergeMap->mbMetric &&
        ((!pCurrentMap->mbMetric && !pCurrentMap->HasMetricTagGeometry()) ||
         ShareConsistentMetricMarker(pCurrentMap,pMergeMap));
    const unsigned long retiredMapId = pCurrentMap->GetId();
    const unsigned long metricMapId = pMergeMap->GetId();

    //std::cout << "Merge local, Active map: " << pCurrentMap->GetId() << std::endl;
    //std::cout << "Merge local, Non-Active map: " << pMergeMap->GetId() << std::endl;

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartMerge = std::chrono::steady_clock::now();
#endif

    // Ensure current keyframe is updated
    mpCurrentKF->UpdateConnections();

    //Get the current KF and its neighbors(visual->covisibles; inertial->temporal+covisibles)
    set<KeyFrame*> spLocalWindowKFs;
    //Get MPs in the welding area from the current map
    set<MapPoint*> spLocalWindowMPs;
    if(pCurrentMap->IsInertial() && pMergeMap->IsInertial()) //TODO Check the correct initialization
    {
        KeyFrame* pKFi = mpCurrentKF;
        int nInserted = 0;
        while(pKFi && nInserted < numTemporalKFs)
        {
            spLocalWindowKFs.insert(pKFi);
            pKFi = mpCurrentKF->mPrevKF;
            nInserted++;

            set<MapPoint*> spMPi = pKFi->GetMapPoints();
            spLocalWindowMPs.insert(spMPi.begin(), spMPi.end());
        }

        pKFi = mpCurrentKF->mNextKF;
        while(pKFi)
        {
            spLocalWindowKFs.insert(pKFi);

            set<MapPoint*> spMPi = pKFi->GetMapPoints();
            spLocalWindowMPs.insert(spMPi.begin(), spMPi.end());

            pKFi = mpCurrentKF->mNextKF;
        }
    }
    else
    {
        spLocalWindowKFs.insert(mpCurrentKF);
    }

    vector<KeyFrame*> vpCovisibleKFs = mpCurrentKF->GetBestCovisibilityKeyFrames(numTemporalKFs);
    spLocalWindowKFs.insert(vpCovisibleKFs.begin(), vpCovisibleKFs.end());
    spLocalWindowKFs.insert(mpCurrentKF);
    const int nMaxTries = 5;
    int nNumTries = 0;
    while(spLocalWindowKFs.size() < numTemporalKFs && nNumTries < nMaxTries)
    {
        vector<KeyFrame*> vpNewCovKFs;
        vpNewCovKFs.empty();
        for(KeyFrame* pKFi : spLocalWindowKFs)
        {
            vector<KeyFrame*> vpKFiCov = pKFi->GetBestCovisibilityKeyFrames(numTemporalKFs/2);
            for(KeyFrame* pKFcov : vpKFiCov)
            {
                if(pKFcov && !pKFcov->isBad() && spLocalWindowKFs.find(pKFcov) == spLocalWindowKFs.end())
                {
                    vpNewCovKFs.push_back(pKFcov);
                }

            }
        }

        spLocalWindowKFs.insert(vpNewCovKFs.begin(), vpNewCovKFs.end());
        nNumTries++;
    }

    for(KeyFrame* pKFi : spLocalWindowKFs)
    {
        if(!pKFi || pKFi->isBad())
            continue;

        set<MapPoint*> spMPs = pKFi->GetMapPoints();
        spLocalWindowMPs.insert(spMPs.begin(), spMPs.end());
    }

    //std::cout << "[Merge]: Ma = " << to_string(pCurrentMap->GetId()) << "; #KFs = " << to_string(spLocalWindowKFs.size()) << "; #MPs = " << to_string(spLocalWindowMPs.size()) << std::endl;

    set<KeyFrame*> spMergeConnectedKFs;
    if(pCurrentMap->IsInertial() && pMergeMap->IsInertial()) //TODO Check the correct initialization
    {
        KeyFrame* pKFi = mpMergeMatchedKF;
        int nInserted = 0;
        while(pKFi && nInserted < numTemporalKFs/2)
        {
            spMergeConnectedKFs.insert(pKFi);
            pKFi = mpCurrentKF->mPrevKF;
            nInserted++;
        }

        pKFi = mpMergeMatchedKF->mNextKF;
        while(pKFi && nInserted < numTemporalKFs)
        {
            spMergeConnectedKFs.insert(pKFi);
            pKFi = mpCurrentKF->mNextKF;
        }
    }
    else
    {
        spMergeConnectedKFs.insert(mpMergeMatchedKF);
    }
    vpCovisibleKFs = mpMergeMatchedKF->GetBestCovisibilityKeyFrames(numTemporalKFs);
    spMergeConnectedKFs.insert(vpCovisibleKFs.begin(), vpCovisibleKFs.end());
    spMergeConnectedKFs.insert(mpMergeMatchedKF);
    nNumTries = 0;
    while(spMergeConnectedKFs.size() < numTemporalKFs && nNumTries < nMaxTries)
    {
        vector<KeyFrame*> vpNewCovKFs;
        for(KeyFrame* pKFi : spMergeConnectedKFs)
        {
            vector<KeyFrame*> vpKFiCov = pKFi->GetBestCovisibilityKeyFrames(numTemporalKFs/2);
            for(KeyFrame* pKFcov : vpKFiCov)
            {
                if(pKFcov && !pKFcov->isBad() && spMergeConnectedKFs.find(pKFcov) == spMergeConnectedKFs.end())
                {
                    vpNewCovKFs.push_back(pKFcov);
                }

            }
        }

        spMergeConnectedKFs.insert(vpNewCovKFs.begin(), vpNewCovKFs.end());
        nNumTries++;
    }

    set<MapPoint*> spMapPointMerge;
    for(KeyFrame* pKFi : spMergeConnectedKFs)
    {
        set<MapPoint*> vpMPs = pKFi->GetMapPoints();
        spMapPointMerge.insert(vpMPs.begin(),vpMPs.end());
    }

    vector<MapPoint*> vpCheckFuseMapPoint;
    vpCheckFuseMapPoint.reserve(spMapPointMerge.size());
    std::copy(spMapPointMerge.begin(), spMapPointMerge.end(), std::back_inserter(vpCheckFuseMapPoint));

    //std::cout << "[Merge]: Mm = " << to_string(pMergeMap->GetId()) << "; #KFs = " << to_string(spMergeConnectedKFs.size()) << "; #MPs = " << to_string(spMapPointMerge.size()) << std::endl;


    //
    Sophus::SE3d Twc = mpCurrentKF->GetPoseInverse().cast<double>();
    g2o::Sim3 g2oNonCorrectedSwc(Twc.unit_quaternion(),Twc.translation(),1.0);
    g2o::Sim3 g2oNonCorrectedScw = g2oNonCorrectedSwc.inverse();
    g2o::Sim3 g2oCorrectedScw = mg2oMergeScw; //TODO Check the transformation

    KeyFrameAndPose vCorrectedSim3, vNonCorrectedSim3;
    vCorrectedSim3[mpCurrentKF]=g2oCorrectedScw;
    vNonCorrectedSim3[mpCurrentKF]=g2oNonCorrectedScw;


#ifdef REGISTER_TIMES
    vnMergeKFs.push_back(spLocalWindowKFs.size() + spMergeConnectedKFs.size());
    vnMergeMPs.push_back(spLocalWindowMPs.size() + spMapPointMerge.size());
#endif
    for(KeyFrame* pKFi : spLocalWindowKFs)
    {
        if(!pKFi || pKFi->isBad())
        {
            Verbose::PrintMess("Bad KF in correction", Verbose::VERBOSITY_DEBUG);
            continue;
        }

        if(pKFi->GetMap() != pCurrentMap)
            Verbose::PrintMess("Other map KF, this should't happen", Verbose::VERBOSITY_DEBUG);

        g2o::Sim3 g2oCorrectedSiw;

        if(pKFi!=mpCurrentKF)
        {
            Sophus::SE3d Tiw = (pKFi->GetPose()).cast<double>();
            g2o::Sim3 g2oSiw(Tiw.unit_quaternion(),Tiw.translation(),1.0);
            //Pose without correction
            vNonCorrectedSim3[pKFi]=g2oSiw;

            Sophus::SE3d Tic = Tiw*Twc;
            g2o::Sim3 g2oSic(Tic.unit_quaternion(),Tic.translation(),1.0);
            g2oCorrectedSiw = g2oSic*mg2oMergeScw;
            vCorrectedSim3[pKFi]=g2oCorrectedSiw;
        }
        else
        {
            g2oCorrectedSiw = g2oCorrectedScw;
        }
        pKFi->mTcwMerge  = pKFi->GetPose();

        // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
        double s = g2oCorrectedSiw.scale();
        pKFi->mfScale = s;
        Sophus::SE3d correctedTiw(g2oCorrectedSiw.rotation(), g2oCorrectedSiw.translation() / s);

        pKFi->mTcwMerge = correctedTiw.cast<float>();

        if(pCurrentMap->isImuInitialized())
        {
            Eigen::Quaternionf Rcor = (g2oCorrectedSiw.rotation().inverse() * vNonCorrectedSim3[pKFi].rotation()).cast<float>();
            pKFi->mVwbMerge = Rcor * pKFi->GetVelocity();
        }

        //TODO DEBUG to know which are the KFs that had been moved to the other map
    }

    int numPointsWithCorrection = 0;

    //for(MapPoint* pMPi : spLocalWindowMPs)
    set<MapPoint*>::iterator itMP = spLocalWindowMPs.begin();
    while(itMP != spLocalWindowMPs.end())
    {
        MapPoint* pMPi = *itMP;
        if(!pMPi || pMPi->isBad())
        {
            itMP = spLocalWindowMPs.erase(itMP);
            continue;
        }

        KeyFrame* pKFref = pMPi->GetReferenceKeyFrame();
        if(vCorrectedSim3.find(pKFref) == vCorrectedSim3.end())
        {
            itMP = spLocalWindowMPs.erase(itMP);
            numPointsWithCorrection++;
            continue;
        }
        g2o::Sim3 g2oCorrectedSwi = vCorrectedSim3[pKFref].inverse();
        g2o::Sim3 g2oNonCorrectedSiw = vNonCorrectedSim3[pKFref];

        // Project with non-corrected pose and project back with corrected pose
        Eigen::Vector3d P3Dw = pMPi->GetWorldPos().cast<double>();
        Eigen::Vector3d eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oNonCorrectedSiw.map(P3Dw));
        Eigen::Quaterniond Rcor = g2oCorrectedSwi.rotation() * g2oNonCorrectedSiw.rotation();

        pMPi->mPosMerge = eigCorrectedP3Dw.cast<float>();
        pMPi->mNormalVectorMerge = Rcor.cast<float>() * pMPi->GetNormal();

        itMP++;
    }
    /*if(numPointsWithCorrection>0)
    {
        std::cout << "[Merge]: " << std::to_string(numPointsWithCorrection) << " points removed from Ma due to its reference KF is not in welding area" << std::endl;
        std::cout << "[Merge]: Ma has " << std::to_string(spLocalWindowMPs.size()) << " points" << std::endl;
    }*/

    {
        unique_lock<mutex> currentLock(pCurrentMap->mMutexMapUpdate); // We update the current map with the Merge information
        unique_lock<mutex> mergeLock(pMergeMap->mMutexMapUpdate); // We remove the Kfs and MPs in the merged area from the old map

        //std::cout << "Merge local window: " << spLocalWindowKFs.size() << std::endl;
        //std::cout << "[Merge]: init merging maps " << std::endl;
        for(KeyFrame* pKFi : spLocalWindowKFs)
        {
            if(!pKFi || pKFi->isBad())
            {
                //std::cout << "Bad KF in correction" << std::endl;
                continue;
            }

            //std::cout << "KF id: " << pKFi->mnId << std::endl;

            pKFi->mTcwBefMerge = pKFi->GetPose();
            pKFi->mTwcBefMerge = pKFi->GetPoseInverse();
            pKFi->SetPose(pKFi->mTcwMerge);
            // Scw contains s; converting t/s changes map units by 1/s.
            // Apply once when the pose is committed, not while staging it.
            pKFi->mReplayUnitScale /= pKFi->mfScale;

            // Make sure connections are updated
            pKFi->UpdateMap(pMergeMap);
            pKFi->mnMergeCorrectedForKF = mpCurrentKF->mnId;
            pMergeMap->AddKeyFrame(pKFi);
            pCurrentMap->EraseKeyFrame(pKFi);

            if(pCurrentMap->isImuInitialized())
            {
                pKFi->SetVelocity(pKFi->mVwbMerge);
            }
        }

        for(MapPoint* pMPi : spLocalWindowMPs)
        {
            if(!pMPi || pMPi->isBad())
                continue;

            pMPi->SetWorldPos(pMPi->mPosMerge);
            pMPi->SetNormalVector(pMPi->mNormalVectorMerge);
            pMPi->UpdateMap(pMergeMap);
            pMergeMap->AddMapPoint(pMPi);
            pCurrentMap->EraseMapPoint(pMPi);
        }

        mpAtlas->ChangeMap(pMergeMap);
        mpAtlas->SetMapBad(pCurrentMap);
        pMergeMap->IncreaseChangeIndex();
        // The destination marker map defines the published metric world.
        // Upstream's debug ID swap obscures that identity and makes replay
        // appear to jump maps even though the pointer/gauge is unchanged.
        if(metricTargetMerge)
            mpAtlas->mMarkerMapAliases[retiredMapId]=metricMapId;
        else
            pMergeMap->ChangeId(pCurrentMap->GetId());

        //std::cout << "[Merge]: merging maps finished" << std::endl;
    }

    //Rebuild the essential graph in the local window
    pCurrentMap->GetOriginKF()->SetFirstConnection(false);
    pNewChild = mpCurrentKF->GetParent(); // Old parent, it will be the new child of this KF
    pNewParent = mpCurrentKF; // Old child, now it will be the parent of its own parent(we need eliminate this KF from children list in its old parent)
    mpCurrentKF->ChangeParent(mpMergeMatchedKF);
    while(pNewChild)
    {
        pNewChild->EraseChild(pNewParent); // We remove the relation between the old parent and the new for avoid loop
        KeyFrame * pOldParent = pNewChild->GetParent();

        pNewChild->ChangeParent(pNewParent);

        pNewParent = pNewChild;
        pNewChild = pOldParent;

    }

    //Update the connections between the local window
    mpMergeMatchedKF->UpdateConnections();

    vpMergeConnectedKFs = mpMergeMatchedKF->GetVectorCovisibleKeyFrames();
    vpMergeConnectedKFs.push_back(mpMergeMatchedKF);
    //vpCheckFuseMapPoint.reserve(spMapPointMerge.size());
    //std::copy(spMapPointMerge.begin(), spMapPointMerge.end(), std::back_inserter(vpCheckFuseMapPoint));

    // Project MapPoints observed in the neighborhood of the merge keyframe
    // into the current keyframe and neighbors using corrected poses.
    // Fuse duplications.
    //std::cout << "[Merge]: start fuse points" << std::endl;
    SearchAndFuse(vCorrectedSim3, vpCheckFuseMapPoint);
    //std::cout << "[Merge]: fuse points finished" << std::endl;

    // Update connectivity
    for(KeyFrame* pKFi : spLocalWindowKFs)
    {
        if(!pKFi || pKFi->isBad())
            continue;

        pKFi->UpdateConnections();
    }
    for(KeyFrame* pKFi : spMergeConnectedKFs)
    {
        if(!pKFi || pKFi->isBad())
            continue;

        pKFi->UpdateConnections();
    }

    //std::cout << "[Merge]: Start welding bundle adjustment" << std::endl;

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartWeldingBA = std::chrono::steady_clock::now();

    double timeMergeMaps = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_StartWeldingBA - time_StartMerge).count();
    vdMergeMaps_ms.push_back(timeMergeMaps);
#endif

    bool bStop = false;
    vpLocalCurrentWindowKFs.clear();
    vpMergeConnectedKFs.clear();
    std::copy(spLocalWindowKFs.begin(), spLocalWindowKFs.end(), std::back_inserter(vpLocalCurrentWindowKFs));
    std::copy(spMergeConnectedKFs.begin(), spMergeConnectedKFs.end(), std::back_inserter(vpMergeConnectedKFs));
    if (mpTracker->mSensor==System::IMU_MONOCULAR || mpTracker->mSensor==System::IMU_STEREO || mpTracker->mSensor==System::IMU_RGBD)
    {
        Optimizer::MergeInertialBA(mpCurrentKF,mpMergeMatchedKF,&bStop, pCurrentMap,vCorrectedSim3);
    }
    else
    {
        Optimizer::LocalBundleAdjustment(mpCurrentKF, vpLocalCurrentWindowKFs, vpMergeConnectedKFs,&bStop);
    }

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndWeldingBA = std::chrono::steady_clock::now();

    double timeWeldingBA = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndWeldingBA - time_StartWeldingBA).count();
    vdWeldingBA_ms.push_back(timeWeldingBA);
#endif
    //std::cout << "[Merge]: Welding bundle adjustment finished" << std::endl;

    // Loop closed. Release Local Mapping.
    mpLocalMapper->Release();

    //Update the non critical area from the current map to the merged map
    vector<KeyFrame*> vpCurrentMapKFs = pCurrentMap->GetAllKeyFrames();
    vector<MapPoint*> vpCurrentMapMPs = pCurrentMap->GetAllMapPoints();

    if(vpCurrentMapKFs.size() == 0){}
    else {
        if(mpTracker->mSensor == System::MONOCULAR)
        {
            unique_lock<mutex> currentLock(pCurrentMap->mMutexMapUpdate); // We update the current map with the Merge information

            for(KeyFrame* pKFi : vpCurrentMapKFs)
            {
                if(!pKFi || pKFi->isBad() || pKFi->GetMap() != pCurrentMap)
                {
                    continue;
                }

                g2o::Sim3 g2oCorrectedSiw;

                Sophus::SE3d Tiw = (pKFi->GetPose()).cast<double>();
                g2o::Sim3 g2oSiw(Tiw.unit_quaternion(),Tiw.translation(),1.0);
                //Pose without correction
                vNonCorrectedSim3[pKFi]=g2oSiw;

                Sophus::SE3d Tic = Tiw*Twc;
                g2o::Sim3 g2oSim(Tic.unit_quaternion(),Tic.translation(),1.0);
                g2oCorrectedSiw = g2oSim*mg2oMergeScw;
                vCorrectedSim3[pKFi]=g2oCorrectedSiw;

                // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
                double s = g2oCorrectedSiw.scale();

                pKFi->mfScale = s;

                Sophus::SE3d correctedTiw(g2oCorrectedSiw.rotation(),g2oCorrectedSiw.translation() / s);

                pKFi->mTcwBefMerge = pKFi->GetPose();
                pKFi->mTwcBefMerge = pKFi->GetPoseInverse();

                pKFi->SetPose(correctedTiw.cast<float>());
                pKFi->mReplayUnitScale /= static_cast<float>(s);

                if(pCurrentMap->isImuInitialized())
                {
                    Eigen::Quaternionf Rcor = (g2oCorrectedSiw.rotation().inverse() * vNonCorrectedSim3[pKFi].rotation()).cast<float>();
                    pKFi->SetVelocity(Rcor * pKFi->GetVelocity()); // TODO: should add here scale s
                }

            }
            for(MapPoint* pMPi : vpCurrentMapMPs)
            {
                if(!pMPi || pMPi->isBad()|| pMPi->GetMap() != pCurrentMap)
                    continue;

                KeyFrame* pKFref = pMPi->GetReferenceKeyFrame();
                g2o::Sim3 g2oCorrectedSwi = vCorrectedSim3[pKFref].inverse();
                g2o::Sim3 g2oNonCorrectedSiw = vNonCorrectedSim3[pKFref];

                // Project with non-corrected pose and project back with corrected pose
                Eigen::Vector3d P3Dw = pMPi->GetWorldPos().cast<double>();
                Eigen::Vector3d eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oNonCorrectedSiw.map(P3Dw));
                pMPi->SetWorldPos(eigCorrectedP3Dw.cast<float>());

                pMPi->UpdateNormalAndDepth();
            }
        }

        mpLocalMapper->RequestStop();
        // Wait until Local Mapping has effectively stopped
        while(!mpLocalMapper->isStopped())
        {
            usleep(1000);
        }

        // Optimize graph (and update the loop position for each element form the begining to the end)
        if(mpTracker->mSensor != System::MONOCULAR)
        {
            Optimizer::OptimizeEssentialGraph(mpCurrentKF, vpMergeConnectedKFs, vpLocalCurrentWindowKFs, vpCurrentMapKFs, vpCurrentMapMPs);
        }


        {
            // Get Merge Map Mutex
            unique_lock<mutex> currentLock(pCurrentMap->mMutexMapUpdate); // We update the current map with the Merge information
            unique_lock<mutex> mergeLock(pMergeMap->mMutexMapUpdate); // We remove the Kfs and MPs in the merged area from the old map

            //std::cout << "Merge outside KFs: " << vpCurrentMapKFs.size() << std::endl;
            for(KeyFrame* pKFi : vpCurrentMapKFs)
            {
                if(!pKFi || pKFi->isBad() || pKFi->GetMap() != pCurrentMap)
                {
                    continue;
                }
                //std::cout << "KF id: " << pKFi->mnId << std::endl;

                // Make sure connections are updated
                pKFi->UpdateMap(pMergeMap);
                pMergeMap->AddKeyFrame(pKFi);
                pCurrentMap->EraseKeyFrame(pKFi);
            }

            for(MapPoint* pMPi : vpCurrentMapMPs)
            {
                if(!pMPi || pMPi->isBad())
                    continue;

                pMPi->UpdateMap(pMergeMap);
                pMergeMap->AddMapPoint(pMPi);
                pCurrentMap->EraseMapPoint(pMPi);
            }
        }
    }

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndOptEss = std::chrono::steady_clock::now();

    double timeOptEss = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndOptEss - time_EndWeldingBA).count();
    vdMergeOptEss_ms.push_back(timeOptEss);
#endif


    mpLocalMapper->Release();

    if(bRelaunchBA && (!pCurrentMap->isImuInitialized() || (pCurrentMap->KeyFramesInMap()<200 && mpAtlas->CountMaps()==1)))
    {
        // Launch a new thread to perform Global Bundle Adjustment
        LaunchGlobalBundleAdjustment(pMergeMap,mpCurrentKF);
    }

    mpMergeMatchedKF->AddMergeEdge(mpCurrentKF);
    mpCurrentKF->AddMergeEdge(mpMergeMatchedKF);

    pCurrentMap->IncreaseChangeIndex();
    pMergeMap->IncreaseChangeIndex();

    mpAtlas->RemoveBadMaps();

    return true;
}


bool LoopClosing::MergeLocal2()
{
    unique_lock<mutex> correctionGate(mpAtlas->mMutexPoseGraphCorrection);
    if(RejectUnsupportedCorrection("merge", mpCurrentKF->GetMap(), mpMergeMatchedKF->GetMap(),
                                   mpCurrentKF->mnId, mpCurrentKF->mTimeStamp))
        return false;
    //cout << "Merge detected!!!!" << endl;

    int numTemporalKFs = 11; //TODO (set by parameter): Temporal KFs in the local window if the map is inertial.

    //Relationship to rebuild the essential graph, it is used two times, first in the local window and later in the rest of the map
    KeyFrame* pNewChild;
    KeyFrame* pNewParent;

    vector<KeyFrame*> vpLocalCurrentWindowKFs;
    vector<KeyFrame*> vpMergeConnectedKFs;

    KeyFrameAndPose CorrectedSim3, NonCorrectedSim3;
    // NonCorrectedSim3[mpCurrentKF]=mg2oLoopScw;

    // Flag that is true only when we stopped a running BA, in this case we need relaunch at the end of the merge
    bool bRelaunchBA = false;

    //cout << "Check Full Bundle Adjustment" << endl;
    // If a Global Bundle Adjustment is running, abort it
    if(isRunningGBA())
    {
        CancelGlobalBundleAdjustment();
        bRelaunchBA = true;
    }


    //cout << "Request Stop Local Mapping" << endl;
    mpLocalMapper->RequestStop();
    // Wait until Local Mapping has effectively stopped
    while(!mpLocalMapper->isStopped())
    {
        usleep(1000);
    }
    //cout << "Local Map stopped" << endl;

    Map* pCurrentMap = mpCurrentKF->GetMap();
    Map* pMergeMap = mpMergeMatchedKF->GetMap();

    {
        float s_on = mSold_new.scale();
        Sophus::SE3f T_on(mSold_new.rotation().cast<float>(), mSold_new.translation().cast<float>());

        unique_lock<mutex> lock(mpAtlas->GetCurrentMap()->mMutexMapUpdate);

        //cout << "KFs before empty: " << mpAtlas->GetCurrentMap()->KeyFramesInMap() << endl;
        mpLocalMapper->EmptyQueue();
        //cout << "KFs after empty: " << mpAtlas->GetCurrentMap()->KeyFramesInMap() << endl;

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        //cout << "updating active map to merge reference" << endl;
        //cout << "curr merge KF id: " << mpCurrentKF->mnId << endl;
        //cout << "curr tracking KF id: " << mpTracker->GetLastKeyFrame()->mnId << endl;
        bool bScaleVel=false;
        if(s_on!=1)
            bScaleVel=true;
        mpAtlas->GetCurrentMap()->ApplyScaledRotation(T_on,s_on,bScaleVel);
        mpTracker->UpdateFrameIMU(s_on,mpCurrentKF->GetImuBias(),mpTracker->GetLastKeyFrame());

        std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
    }

    const int numKFnew=pCurrentMap->KeyFramesInMap();

    if((mpTracker->mSensor==System::IMU_MONOCULAR || mpTracker->mSensor==System::IMU_STEREO || mpTracker->mSensor==System::IMU_RGBD)
       && !pCurrentMap->GetIniertialBA2()){
        // Map is not completly initialized
        Eigen::Vector3d bg, ba;
        bg << 0., 0., 0.;
        ba << 0., 0., 0.;
        Optimizer::InertialOptimization(pCurrentMap,bg,ba);
        IMU::Bias b (ba[0],ba[1],ba[2],bg[0],bg[1],bg[2]);
        unique_lock<mutex> lock(mpAtlas->GetCurrentMap()->mMutexMapUpdate);
        mpTracker->UpdateFrameIMU(1.0f,b,mpTracker->GetLastKeyFrame());

        // Set map initialized
        pCurrentMap->SetIniertialBA2();
        pCurrentMap->SetIniertialBA1();
        pCurrentMap->SetImuInitialized();

    }


    //cout << "MergeMap init ID: " << pMergeMap->GetInitKFid() << "       CurrMap init ID: " << pCurrentMap->GetInitKFid() << endl;

    // Load KFs and MPs from merge map
    //cout << "updating current map" << endl;
    {
        // Get Merge Map Mutex (This section stops tracking!!)
        unique_lock<mutex> currentLock(pCurrentMap->mMutexMapUpdate); // We update the current map with the Merge information
        unique_lock<mutex> mergeLock(pMergeMap->mMutexMapUpdate); // We remove the Kfs and MPs in the merged area from the old map


        vector<KeyFrame*> vpMergeMapKFs = pMergeMap->GetAllKeyFrames();
        vector<MapPoint*> vpMergeMapMPs = pMergeMap->GetAllMapPoints();


        for(KeyFrame* pKFi : vpMergeMapKFs)
        {
            if(!pKFi || pKFi->isBad() || pKFi->GetMap() != pMergeMap)
            {
                continue;
            }

            // Make sure connections are updated
            pKFi->UpdateMap(pCurrentMap);
            pCurrentMap->AddKeyFrame(pKFi);
            pMergeMap->EraseKeyFrame(pKFi);
        }

        for(MapPoint* pMPi : vpMergeMapMPs)
        {
            if(!pMPi || pMPi->isBad() || pMPi->GetMap() != pMergeMap)
                continue;

            pMPi->UpdateMap(pCurrentMap);
            pCurrentMap->AddMapPoint(pMPi);
            pMergeMap->EraseMapPoint(pMPi);
        }

        // Save non corrected poses (already merged maps)
        vector<KeyFrame*> vpKFs = pCurrentMap->GetAllKeyFrames();
        for(KeyFrame* pKFi : vpKFs)
        {
            Sophus::SE3d Tiw = (pKFi->GetPose()).cast<double>();
            g2o::Sim3 g2oSiw(Tiw.unit_quaternion(),Tiw.translation(),1.0);
            NonCorrectedSim3[pKFi]=g2oSiw;
        }
    }

    //cout << "MergeMap init ID: " << pMergeMap->GetInitKFid() << "       CurrMap init ID: " << pCurrentMap->GetInitKFid() << endl;

    //cout << "end updating current map" << endl;

    // Critical zone
    //bool good = pCurrentMap->CheckEssentialGraph();
    /*if(!good)
        cout << "BAD ESSENTIAL GRAPH!!" << endl;*/

    //cout << "Update essential graph" << endl;
    // mpCurrentKF->UpdateConnections(); // to put at false mbFirstConnection
    pMergeMap->GetOriginKF()->SetFirstConnection(false);
    pNewChild = mpMergeMatchedKF->GetParent(); // Old parent, it will be the new child of this KF
    pNewParent = mpMergeMatchedKF; // Old child, now it will be the parent of its own parent(we need eliminate this KF from children list in its old parent)
    mpMergeMatchedKF->ChangeParent(mpCurrentKF);
    while(pNewChild)
    {
        pNewChild->EraseChild(pNewParent); // We remove the relation between the old parent and the new for avoid loop
        KeyFrame * pOldParent = pNewChild->GetParent();
        pNewChild->ChangeParent(pNewParent);
        pNewParent = pNewChild;
        pNewChild = pOldParent;

    }


    //cout << "MergeMap init ID: " << pMergeMap->GetInitKFid() << "       CurrMap init ID: " << pCurrentMap->GetInitKFid() << endl;

    //cout << "end update essential graph" << endl;

    /*good = pCurrentMap->CheckEssentialGraph();
    if(!good)
        cout << "BAD ESSENTIAL GRAPH 1!!" << endl;*/

    //cout << "Update relationship between KFs" << endl;
    vector<MapPoint*> vpCheckFuseMapPoint; // MapPoint vector from current map to allow to fuse duplicated points with the old map (merge)
    vector<KeyFrame*> vpCurrentConnectedKFs;

    mvpMergeConnectedKFs.push_back(mpMergeMatchedKF);
    vector<KeyFrame*> aux = mpMergeMatchedKF->GetVectorCovisibleKeyFrames();
    mvpMergeConnectedKFs.insert(mvpMergeConnectedKFs.end(), aux.begin(), aux.end());
    if (mvpMergeConnectedKFs.size()>6)
        mvpMergeConnectedKFs.erase(mvpMergeConnectedKFs.begin()+6,mvpMergeConnectedKFs.end());
    /*mvpMergeConnectedKFs = mpMergeMatchedKF->GetVectorCovisibleKeyFrames();
    mvpMergeConnectedKFs.push_back(mpMergeMatchedKF);*/

    mpCurrentKF->UpdateConnections();
    vpCurrentConnectedKFs.push_back(mpCurrentKF);
    /*vpCurrentConnectedKFs = mpCurrentKF->GetVectorCovisibleKeyFrames();
    vpCurrentConnectedKFs.push_back(mpCurrentKF);*/
    aux = mpCurrentKF->GetVectorCovisibleKeyFrames();
    vpCurrentConnectedKFs.insert(vpCurrentConnectedKFs.end(), aux.begin(), aux.end());
    if (vpCurrentConnectedKFs.size()>6)
        vpCurrentConnectedKFs.erase(vpCurrentConnectedKFs.begin()+6,vpCurrentConnectedKFs.end());

    set<MapPoint*> spMapPointMerge;
    for(KeyFrame* pKFi : mvpMergeConnectedKFs)
    {
        set<MapPoint*> vpMPs = pKFi->GetMapPoints();
        spMapPointMerge.insert(vpMPs.begin(),vpMPs.end());
        if(spMapPointMerge.size()>1000)
            break;
    }

    /*cout << "vpCurrentConnectedKFs.size() " << vpCurrentConnectedKFs.size() << endl;
    cout << "mvpMergeConnectedKFs.size() " << mvpMergeConnectedKFs.size() << endl;
    cout << "spMapPointMerge.size() " << spMapPointMerge.size() << endl;*/


    vpCheckFuseMapPoint.reserve(spMapPointMerge.size());
    std::copy(spMapPointMerge.begin(), spMapPointMerge.end(), std::back_inserter(vpCheckFuseMapPoint));
    //cout << "Finished to update relationship between KFs" << endl;

    //cout << "MergeMap init ID: " << pMergeMap->GetInitKFid() << "       CurrMap init ID: " << pCurrentMap->GetInitKFid() << endl;

    /*good = pCurrentMap->CheckEssentialGraph();
    if(!good)
        cout << "BAD ESSENTIAL GRAPH 2!!" << endl;*/

    //cout << "start SearchAndFuse" << endl;
    SearchAndFuse(vpCurrentConnectedKFs, vpCheckFuseMapPoint);
    //cout << "end SearchAndFuse" << endl;

    //cout << "MergeMap init ID: " << pMergeMap->GetInitKFid() << "       CurrMap init ID: " << pCurrentMap->GetInitKFid() << endl;

    /*good = pCurrentMap->CheckEssentialGraph();
    if(!good)
        cout << "BAD ESSENTIAL GRAPH 3!!" << endl;

    cout << "Init to update connections" << endl;*/


    for(KeyFrame* pKFi : vpCurrentConnectedKFs)
    {
        if(!pKFi || pKFi->isBad())
            continue;

        pKFi->UpdateConnections();
    }
    for(KeyFrame* pKFi : mvpMergeConnectedKFs)
    {
        if(!pKFi || pKFi->isBad())
            continue;

        pKFi->UpdateConnections();
    }
    //cout << "end update connections" << endl;

    //cout << "MergeMap init ID: " << pMergeMap->GetInitKFid() << "       CurrMap init ID: " << pCurrentMap->GetInitKFid() << endl;

    /*good = pCurrentMap->CheckEssentialGraph();
    if(!good)
        cout << "BAD ESSENTIAL GRAPH 4!!" << endl;*/

    // TODO Check: If new map is too small, we suppose that not informaiton can be propagated from new to old map
    if (numKFnew<10){
        mpLocalMapper->Release();
        return true;
    }

    /*good = pCurrentMap->CheckEssentialGraph();
    if(!good)
        cout << "BAD ESSENTIAL GRAPH 5!!" << endl;*/

    // Perform BA
    bool bStopFlag=false;
    KeyFrame* pCurrKF = mpTracker->GetLastKeyFrame();
    //cout << "start MergeInertialBA" << endl;
    Optimizer::MergeInertialBA(pCurrKF, mpMergeMatchedKF, &bStopFlag, pCurrentMap,CorrectedSim3);
    //cout << "end MergeInertialBA" << endl;

    /*good = pCurrentMap->CheckEssentialGraph();
    if(!good)
        cout << "BAD ESSENTIAL GRAPH 6!!" << endl;*/

    // Release Local Mapping.
    mpLocalMapper->Release();


    return true;
}

void LoopClosing::CheckObservations(set<KeyFrame*> &spKFsMap1, set<KeyFrame*> &spKFsMap2)
{
    cout << "----------------------" << endl;
    for(KeyFrame* pKFi1 : spKFsMap1)
    {
        map<KeyFrame*, int> mMatchedMP;
        set<MapPoint*> spMPs = pKFi1->GetMapPoints();

        for(MapPoint* pMPij : spMPs)
        {
            if(!pMPij || pMPij->isBad())
            {
                continue;
            }

            map<KeyFrame*, tuple<int,int>> mMPijObs = pMPij->GetObservations();
            for(KeyFrame* pKFi2 : spKFsMap2)
            {
                if(mMPijObs.find(pKFi2) != mMPijObs.end())
                {
                    if(mMatchedMP.find(pKFi2) != mMatchedMP.end())
                    {
                        mMatchedMP[pKFi2] = mMatchedMP[pKFi2] + 1;
                    }
                    else
                    {
                        mMatchedMP[pKFi2] = 1;
                    }
                }
            }

        }

        if(mMatchedMP.size() == 0)
        {
            cout << "CHECK-OBS: KF " << pKFi1->mnId << " has not any matched MP with the other map" << endl;
        }
        else
        {
            cout << "CHECK-OBS: KF " << pKFi1->mnId << " has matched MP with " << mMatchedMP.size() << " KF from the other map" << endl;
            for(pair<KeyFrame*, int> matchedKF : mMatchedMP)
            {
                cout << "   -KF: " << matchedKF.first->mnId << ", Number of matches: " << matchedKF.second << endl;
            }
        }
    }
    cout << "----------------------" << endl;
}


void LoopClosing::SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap, vector<MapPoint*> &vpMapPoints)
{
    ORBmatcher matcher(0.8);

    int total_replaces = 0;

    //cout << "[FUSE]: Initially there are " << vpMapPoints.size() << " MPs" << endl;
    //cout << "FUSE: Intially there are " << CorrectedPosesMap.size() << " KFs" << endl;
    for(KeyFrameAndPose::const_iterator mit=CorrectedPosesMap.begin(), mend=CorrectedPosesMap.end(); mit!=mend;mit++)
    {
        int num_replaces = 0;
        KeyFrame* pKFi = mit->first;
        Map* pMap = pKFi->GetMap();

        g2o::Sim3 g2oScw = mit->second;
        Sophus::Sim3f Scw = Converter::toSophus(g2oScw);

        vector<MapPoint*> vpReplacePoints(vpMapPoints.size(),static_cast<MapPoint*>(NULL));
        int numFused = matcher.Fuse(pKFi,Scw,vpMapPoints,4,vpReplacePoints);

        // Get Map Mutex
        unique_lock<mutex> lock(pMap->mMutexMapUpdate);
        const int nLP = vpMapPoints.size();
        for(int i=0; i<nLP;i++)
        {
            MapPoint* pRep = vpReplacePoints[i];
            if(pRep)
            {


                num_replaces += 1;
                pRep->Replace(vpMapPoints[i]);

            }
        }

        total_replaces += num_replaces;
    }
    //cout << "[FUSE]: " << total_replaces << " MPs had been fused" << endl;
}


void LoopClosing::SearchAndFuse(const vector<KeyFrame*> &vConectedKFs, vector<MapPoint*> &vpMapPoints)
{
    ORBmatcher matcher(0.8);

    int total_replaces = 0;

    //cout << "FUSE-POSE: Initially there are " << vpMapPoints.size() << " MPs" << endl;
    //cout << "FUSE-POSE: Intially there are " << vConectedKFs.size() << " KFs" << endl;
    for(auto mit=vConectedKFs.begin(), mend=vConectedKFs.end(); mit!=mend;mit++)
    {
        int num_replaces = 0;
        KeyFrame* pKF = (*mit);
        Map* pMap = pKF->GetMap();
        Sophus::SE3f Tcw = pKF->GetPose();
        Sophus::Sim3f Scw(Tcw.unit_quaternion(),Tcw.translation());
        Scw.setScale(1.f);
        /*std::cout << "These should be zeros: " <<
            Scw.rotationMatrix() - Tcw.rotationMatrix() << std::endl <<
            Scw.translation() - Tcw.translation() << std::endl <<
            Scw.scale() - 1.f << std::endl;*/
        vector<MapPoint*> vpReplacePoints(vpMapPoints.size(),static_cast<MapPoint*>(NULL));
        matcher.Fuse(pKF,Scw,vpMapPoints,4,vpReplacePoints);

        // Get Map Mutex
        unique_lock<mutex> lock(pMap->mMutexMapUpdate);
        const int nLP = vpMapPoints.size();
        for(int i=0; i<nLP;i++)
        {
            MapPoint* pRep = vpReplacePoints[i];
            if(pRep)
            {
                num_replaces += 1;
                pRep->Replace(vpMapPoints[i]);
            }
        }
        /*cout << "FUSE-POSE: KF " << pKF->mnId << " ->" << num_replaces << " MPs fused" << endl;
        total_replaces += num_replaces;*/
    }
    //cout << "FUSE-POSE: " << total_replaces << " MPs had been fused" << endl;
}



void LoopClosing::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetRequested = true;
    }

    while(1)
    {
        {
        unique_lock<mutex> lock2(mMutexReset);
        if(!mbResetRequested)
            break;
        }
        usleep(5000);
    }
}

void LoopClosing::RequestResetActiveMap(Map *pMap)
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetActiveMapRequested = true;
        mpMapToReset = pMap;
    }

    while(1)
    {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if(!mbResetActiveMapRequested)
                break;
        }
        usleep(3000);
    }
}

void LoopClosing::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    if(mbResetRequested)
    {
        cout << "Loop closer reset requested..." << endl;
        mlpLoopKeyFrameQueue.clear();
        mLastLoopKFid=0;  //TODO old variable, it is not use in the new algorithm
        mbResetRequested=false;
        mbResetActiveMapRequested = false;
    }
    else if(mbResetActiveMapRequested)
    {

        for (list<KeyFrame*>::const_iterator it=mlpLoopKeyFrameQueue.begin(); it != mlpLoopKeyFrameQueue.end();)
        {
            KeyFrame* pKFi = *it;
            if(pKFi->GetMap() == mpMapToReset)
            {
                it = mlpLoopKeyFrameQueue.erase(it);
            }
            else
                ++it;
        }

        mLastLoopKFid=mpAtlas->GetLastInitKFid(); //TODO old variable, it is not use in the new algorithm
        mbResetActiveMapRequested=false;

    }
}

void LoopClosing::RunGlobalBundleAdjustment(Map* pActiveMap, unsigned long nLoopKF,
                                           double loopTimestamp, unsigned long correctionGeneration)
{  
    // Construct outside the commit-lock scope: on every return those locks
    // are released before completion reacquires mMutexGBA to unregister us.
    GBAWorkerCompletion completion{this,correctionGeneration};
    Verbose::PrintMess("Starting Global Bundle Adjustment", Verbose::VERBOSITY_NORMAL);

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartFGBA = std::chrono::steady_clock::now();

    nFGBA_exec += 1;

    vnGBAKFs.push_back(pActiveMap->GetAllKeyFrames().size());
    vnGBAMPs.push_back(pActiveMap->GetAllMapPoints().size());
#endif

    const bool bImuInit = pActiveMap->isImuInitialized();

    if(!bImuInit)
        Optimizer::GlobalBundleAdjustemnt(pActiveMap,10,&mbStopGBA,nLoopKF,false);
    else
        Optimizer::FullInertialBA(pActiveMap,7,false,nLoopKF,&mbStopGBA);

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndGBA = std::chrono::steady_clock::now();

    double timeGBA = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndGBA - time_StartFGBA).count();
    vdGBA_ms.push_back(timeGBA);

    if(mbStopGBA)
    {
        nFGBA_abort += 1;
    }
#endif

    // Optimizer::GlobalBundleAdjustemnt(mpMap,10,&mbStopGBA,nLoopKF,false);

    // Update all MapPoints and KeyFrames
    // Local Mapping was active during BA, that means that there might be new keyframes
    // not included in the Global BA and they are not consistent with the updated map.
    // We need to propagate the correction through the spanning tree
    {
        // A marker may have established metric units while this asynchronous
        // BA was computing. Check again before any stale result is committed.
        // Keep the same lock order as CorrectLoop/MergeLocal, not GBA -> gate.
        unique_lock<mutex> correctionGate(mpAtlas->mMutexPoseGraphCorrection);
        unique_lock<mutex> lock(mMutexGBA);
        if(!IsGBACommitCurrent(pActiveMap,bImuInit,correctionGeneration))
            return;

        if(!mbStopGBA && RejectUnsupportedCorrection("global_ba", pActiveMap, pActiveMap,
                                                     nLoopKF, loopTimestamp))
        {
            return;
        }

        if(!mbStopGBA)
        {
            Verbose::PrintMess("Global Bundle Adjustment finished", Verbose::VERBOSITY_NORMAL);
            Verbose::PrintMess("Updating map ...", Verbose::VERBOSITY_NORMAL);

            mpLocalMapper->RequestStop();
            // Wait until Local Mapping has effectively stopped

            while(!mpLocalMapper->isStopped() && !mpLocalMapper->isFinished())
            {
                usleep(1000);
            }

            // Get Map Mutex
            unique_lock<mutex> lock(pActiveMap->mMutexMapUpdate);
            // cout << "LC: Update Map Mutex adquired" << endl;

            //pActiveMap->PrintEssentialGraph();
            // Correct keyframes starting at map first keyframe
            list<KeyFrame*> lpKFtoCheck(pActiveMap->mvpKeyFrameOrigins.begin(),pActiveMap->mvpKeyFrameOrigins.end());

            while(!lpKFtoCheck.empty())
            {
                KeyFrame* pKF = lpKFtoCheck.front();
                const set<KeyFrame*> sChilds = pKF->GetChilds();
                //cout << "---Updating KF " << pKF->mnId << " with " << sChilds.size() << " childs" << endl;
                //cout << " KF mnBAGlobalForKF: " << pKF->mnBAGlobalForKF << endl;
                Sophus::SE3f Twc = pKF->GetPoseInverse();
                //cout << "Twc: " << Twc << endl;
                //cout << "GBA: Correct KeyFrames" << endl;
                for(set<KeyFrame*>::const_iterator sit=sChilds.begin();sit!=sChilds.end();sit++)
                {
                    KeyFrame* pChild = *sit;
                    if(!pChild || pChild->isBad())
                        continue;

                    if(pChild->mnBAGlobalForKF!=nLoopKF)
                    {
                        //cout << "++++New child with flag " << pChild->mnBAGlobalForKF << "; LoopKF: " << nLoopKF << endl;
                        //cout << " child id: " << pChild->mnId << endl;
                        Sophus::SE3f Tchildc = pChild->GetPose() * Twc;
                        //cout << "Child pose: " << Tchildc << endl;
                        //cout << "pKF->mTcwGBA: " << pKF->mTcwGBA << endl;
                        pChild->mTcwGBA = Tchildc * pKF->mTcwGBA;//*Tcorc*pKF->mTcwGBA;

                        Sophus::SO3f Rcor = pChild->mTcwGBA.so3().inverse() * pChild->GetPose().so3();
                        if(pChild->isVelocitySet()){
                            pChild->mVwbGBA = Rcor * pChild->GetVelocity();
                        }
                        else
                            Verbose::PrintMess("Child velocity empty!! ", Verbose::VERBOSITY_NORMAL);


                        //cout << "Child bias: " << pChild->GetImuBias() << endl;
                        pChild->mBiasGBA = pChild->GetImuBias();


                        pChild->mnBAGlobalForKF = nLoopKF;

                    }
                    lpKFtoCheck.push_back(pChild);
                }

                //cout << "-------Update pose" << endl;
                pKF->mTcwBefGBA = pKF->GetPose();
                //cout << "pKF->mTcwBefGBA: " << pKF->mTcwBefGBA << endl;
                pKF->SetPose(pKF->mTcwGBA);
                /*cv::Mat Tco_cn = pKF->mTcwBefGBA * pKF->mTcwGBA.inv();
                cv::Vec3d trasl = Tco_cn.rowRange(0,3).col(3);
                double dist = cv::norm(trasl);
                cout << "GBA: KF " << pKF->mnId << " had been moved " << dist << " meters" << endl;
                double desvX = 0;
                double desvY = 0;
                double desvZ = 0;
                if(pKF->mbHasHessian)
                {
                    cv::Mat hessianInv = pKF->mHessianPose.inv();

                    double covX = hessianInv.at<double>(3,3);
                    desvX = std::sqrt(covX);
                    double covY = hessianInv.at<double>(4,4);
                    desvY = std::sqrt(covY);
                    double covZ = hessianInv.at<double>(5,5);
                    desvZ = std::sqrt(covZ);
                    pKF->mbHasHessian = false;
                }
                if(dist > 1)
                {
                    cout << "--To much distance correction: It has " << pKF->GetConnectedKeyFrames().size() << " connected KFs" << endl;
                    cout << "--It has " << pKF->GetCovisiblesByWeight(80).size() << " connected KF with 80 common matches or more" << endl;
                    cout << "--It has " << pKF->GetCovisiblesByWeight(50).size() << " connected KF with 50 common matches or more" << endl;
                    cout << "--It has " << pKF->GetCovisiblesByWeight(20).size() << " connected KF with 20 common matches or more" << endl;

                    cout << "--STD in meters(x, y, z): " << desvX << ", " << desvY << ", " << desvZ << endl;


                    string strNameFile = pKF->mNameFile;
                    cv::Mat imLeft = cv::imread(strNameFile, CV_LOAD_IMAGE_UNCHANGED);

                    cv::cvtColor(imLeft, imLeft, CV_GRAY2BGR);

                    vector<MapPoint*> vpMapPointsKF = pKF->GetMapPointMatches();
                    int num_MPs = 0;
                    for(int i=0; i<vpMapPointsKF.size(); ++i)
                    {
                        if(!vpMapPointsKF[i] || vpMapPointsKF[i]->isBad())
                        {
                            continue;
                        }
                        num_MPs += 1;
                        string strNumOBs = to_string(vpMapPointsKF[i]->Observations());
                        cv::circle(imLeft, pKF->mvKeys[i].pt, 2, cv::Scalar(0, 255, 0));
                        cv::putText(imLeft, strNumOBs, pKF->mvKeys[i].pt, CV_FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 0, 0));
                    }
                    cout << "--It has " << num_MPs << " MPs matched in the map" << endl;

                    string namefile = "./test_GBA/GBA_" + to_string(nLoopKF) + "_KF" + to_string(pKF->mnId) +"_D" + to_string(dist) +".png";
                    cv::imwrite(namefile, imLeft);
                }*/


                if(pKF->bImu)
                {
                    //cout << "-------Update inertial values" << endl;
                    pKF->mVwbBefGBA = pKF->GetVelocity();
                    //if (pKF->mVwbGBA.empty())
                    //    Verbose::PrintMess("pKF->mVwbGBA is empty", Verbose::VERBOSITY_NORMAL);

                    //assert(!pKF->mVwbGBA.empty());
                    pKF->SetVelocity(pKF->mVwbGBA);
                    pKF->SetNewBias(pKF->mBiasGBA);                    
                }

                lpKFtoCheck.pop_front();
            }

            //cout << "GBA: Correct MapPoints" << endl;
            // Correct MapPoints
            const vector<MapPoint*> vpMPs = pActiveMap->GetAllMapPoints();

            for(size_t i=0; i<vpMPs.size(); i++)
            {
                MapPoint* pMP = vpMPs[i];

                if(pMP->isBad())
                    continue;

                if(pMP->mnBAGlobalForKF==nLoopKF)
                {
                    // If optimized by Global BA, just update
                    pMP->SetWorldPos(pMP->mPosGBA);
                }
                else
                {
                    // Update according to the correction of its reference keyframe
                    KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();

                    if(pRefKF->mnBAGlobalForKF!=nLoopKF)
                        continue;

                    /*if(pRefKF->mTcwBefGBA.empty())
                        continue;*/

                    // Map to non-corrected camera
                    // cv::Mat Rcw = pRefKF->mTcwBefGBA.rowRange(0,3).colRange(0,3);
                    // cv::Mat tcw = pRefKF->mTcwBefGBA.rowRange(0,3).col(3);
                    Eigen::Vector3f Xc = pRefKF->mTcwBefGBA * pMP->GetWorldPos();

                    // Backproject using corrected camera
                    pMP->SetWorldPos(pRefKF->GetPoseInverse() * Xc);
                }
            }

            pActiveMap->InformNewBigChange();
            pActiveMap->IncreaseChangeIndex();

            // TODO Check this update
            // mpTracker->UpdateFrameIMU(1.0f, mpTracker->GetLastKeyFrame()->GetImuBias(), mpTracker->GetLastKeyFrame());

            mpLocalMapper->Release();

#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_EndUpdateMap = std::chrono::steady_clock::now();

            double timeUpdateMap = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndUpdateMap - time_EndGBA).count();
            vdUpdateMap_ms.push_back(timeUpdateMap);

            double timeFGBA = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndUpdateMap - time_StartFGBA).count();
            vdFGBATotal_ms.push_back(timeFGBA);
#endif
            Verbose::PrintMess("Map updated!", Verbose::VERBOSITY_NORMAL);
        }

    }
}

void LoopClosing::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    // cout << "LC: Finish requested" << endl;
    mbFinishRequested = true;
}

bool LoopClosing::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LoopClosing::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}

bool LoopClosing::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

std::size_t LoopClosing::KeyframesInQueue()
{
    unique_lock<mutex> lock(mMutexLoopQueue);
    return mlpLoopKeyFrameQueue.size();
}

bool LoopClosing::IsIdle()
{
    return KeyframesInQueue()==0 &&
        !mbProcessingKeyFrame.load(std::memory_order_acquire) &&
        !isRunningGBA();
}


} //namespace ORB_SLAM
