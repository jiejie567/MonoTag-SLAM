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


#include "Tracking.h"
#include "MarkerGraphOptimizer.h"
#include "InitialMetricAlignment.h"
#include "MarkerGraphCoordinator.h"
#include "TemporalFlow.h"

#include "ORBmatcher.h"
#include "FrameDrawer.h"
#include "Converter.h"
#include "G2oTypes.h"
#include "Optimizer.h"
#include "GeometricTools.h"
#include "Pinhole.h"
#include "KannalaBrandt8.h"
#include "MLPnPsolver.h"
#include "GeometricTools.h"

#include <iostream>

#include <algorithm>
#include <cmath>
#include <mutex>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <limits>
#include <tuple>


using namespace std;

namespace ORB_SLAM3
{

extern thread_local bool gLegacyMonoInitializationSelection;
namespace {
struct ScopedLegacyInitializationSelection {
    bool previous;
    explicit ScopedLegacyInitializationSelection(bool enabled)
        : previous(gLegacyMonoInitializationSelection) {
        gLegacyMonoInitializationSelection = enabled;
    }
    ~ScopedLegacyInitializationSelection() {
        gLegacyMonoInitializationSelection = previous;
    }
};
}



Tracking::Tracking(System *pSys, ORBVocabulary* pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Atlas *pAtlas, KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor, Settings* settings, const string &_nameSeq):
    mState(NO_IMAGES_YET), mSensor(sensor), mTrackedFr(0), mbStep(false),
    mbOnlyTracking(false), mbMapUpdated(false), mbVO(false), mpORBVocabulary(pVoc), mpKeyFrameDB(pKFDB),
    mbReadyToInitializate(false), mpSystem(pSys), mpViewer(NULL), bStepByStep(false),
    mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpAtlas(pAtlas), mnLastRelocFrameId(0), time_recently_lost(5.0),
    mnInitialFrameId(0), mbCreatedMap(false), mnFirstFrameId(0), mpCamera2(nullptr), mpLastKeyFrame(static_cast<KeyFrame*>(NULL))
{
    const char* temporalFlow=std::getenv("ORB_SLAM3_TEMPORAL_FLOW");
    mbTemporalFlowEnabled=temporalFlow && std::string(temporalFlow)=="1";
    const char* flowRecovery=std::getenv("ORB_SLAM3_FLOW_RECOVERY");
    mbFlowRecoveryEnabled=flowRecovery && std::string(flowRecovery)=="1";
    const char* reliableRecovery=std::getenv("ORB_SLAM3_RELIABLE_FRAME_RECOVERY");
    mbReliableFrameRecoveryEnabled=!reliableRecovery || std::string(reliableRecovery)!="0";
    const char* reliableCache=std::getenv("ORB_SLAM3_RELIABLE_FRAME_CACHE");
    mbReliableFrameCacheEnabled=!reliableCache || std::string(reliableCache)!="0";
    // Load camera parameters from settings file
    if(settings){
        newParameterLoader(settings);
    }
    else{
        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

        bool b_parse_cam = ParseCamParamFile(fSettings);
        if(!b_parse_cam)
        {
            std::cout << "*Error with the camera parameters in the config file*" << std::endl;
        }

        // Load ORB parameters
        bool b_parse_orb = ParseORBParamFile(fSettings);
        if(!b_parse_orb)
        {
            std::cout << "*Error with the ORB parameters in the config file*" << std::endl;
        }

        bool b_parse_imu = true;
        if(sensor==System::IMU_MONOCULAR || sensor==System::IMU_STEREO || sensor==System::IMU_RGBD)
        {
            b_parse_imu = ParseIMUParamFile(fSettings);
            if(!b_parse_imu)
            {
                std::cout << "*Error with the IMU parameters in the config file*" << std::endl;
            }

            mnFramesToResetIMU = mMaxFrames;
        }

        if(!b_parse_cam || !b_parse_orb || !b_parse_imu)
        {
            std::cerr << "**ERROR in the config file, the format is not correct**" << std::endl;
            try
            {
                throw -1;
            }
            catch(exception &e)
            {

            }
        }
    }

    cv::FileStorage tagSettings(strSettingPath, cv::FileStorage::READ);
    cv::FileNode tagEnabled = tagSettings["TagFusion.enabled"];
    if(!tagEnabled.empty())
        mbTagFusionEnabled = static_cast<int>(tagEnabled) != 0;
    const auto readTagFloat = [&tagSettings](const char* key, float fallback) {
        cv::FileNode node = tagSettings[key];
        return node.empty() ? fallback : static_cast<float>(node.real());
    };
    mbMarkerOnlyInitialization = readTagFloat("TagFusion.markerOnlyInitialization", 0.0f)!=0;
    mbRigidMarkerLayout = readTagFloat("TagFusion.rigidMarkerLayout", 0.0f)!=0;
    mTagMinimumScaleBaselineM = readTagFloat(
        "TagFusion.minimumScaleBaselineM", mTagMinimumScaleBaselineM);
    mTagMinimumVisualBaselineDepthRatio = readTagFloat(
        "TagFusion.minimumVisualBaselineDepthRatio",
        mTagMinimumVisualBaselineDepthRatio);
    mTagMinimumTranslationM = readTagFloat(
        "TagFusion.minimumTranslationM", mTagMinimumTranslationM);
    mTagMinimumRotationDeg = readTagFloat(
        "TagFusion.minimumRotationDeg", mTagMinimumRotationDeg);
    mTagMinimumTrackedRatio = readTagFloat(
        "TagFusion.minimumTrackedRatio", mTagMinimumTrackedRatio);
    mTagPoseWeight = readTagFloat("TagFusion.poseWeight", mTagPoseWeight);
    mTagMaxAlignmentPositionResidualM = readTagFloat(
        "TagFusion.maxAlignmentPositionResidualM", mTagMaxAlignmentPositionResidualM);
    mTagMaxAlignmentRotationResidualDeg = readTagFloat(
        "TagFusion.maxAlignmentRotationResidualDeg", mTagMaxAlignmentRotationResidualDeg);
    const float minimumIntervalS = readTagFloat(
        "TagFusion.minimumKeyFrameIntervalS", 0.2f);
    mTagMinimumKeyFrameFrames = std::max(
        1, static_cast<int>(std::lround(minimumIntervalS * mMaxFrames)));
    if(mbTagFusionEnabled)
    {
        cout << "- fixed-tag metric fusion: enabled" << endl;
        cout << "- tag-aware keyframes: " << mTagMinimumTranslationM
             << " m / " << mTagMinimumRotationDeg << " deg / tracked ratio "
             << mTagMinimumTrackedRatio << ", minimum "
             << mTagMinimumKeyFrameFrames << " frames" << endl;
    }

    initID = 0; lastID = 0;
    mbInitWith3KFs = false;
    mnNumDataset = 0;

    vector<GeometricCamera*> vpCams = mpAtlas->GetAllCameras();
    std::cout << "There are " << vpCams.size() << " cameras in the atlas" << std::endl;
    for(GeometricCamera* pCam : vpCams)
    {
        std::cout << "Camera " << pCam->GetId();
        if(pCam->GetType() == GeometricCamera::CAM_PINHOLE)
        {
            std::cout << " is pinhole" << std::endl;
        }
        else if(pCam->GetType() == GeometricCamera::CAM_FISHEYE)
        {
            std::cout << " is fisheye" << std::endl;
        }
        else
        {
            std::cout << " is unknown" << std::endl;
        }
    }

#ifdef REGISTER_TIMES
    vdRectStereo_ms.clear();
    vdResizeImage_ms.clear();
    vdORBExtract_ms.clear();
    vdStereoMatch_ms.clear();
    vdIMUInteg_ms.clear();
    vdPosePred_ms.clear();
    vdLMTrack_ms.clear();
    vdNewKF_ms.clear();
    vdTrackTotal_ms.clear();
#endif
}

#ifdef REGISTER_TIMES
double calcAverage(vector<double> v_times)
{
    double accum = 0;
    for(double value : v_times)
    {
        accum += value;
    }

    return accum / v_times.size();
}

double calcDeviation(vector<double> v_times, double average)
{
    double accum = 0;
    for(double value : v_times)
    {
        accum += pow(value - average, 2);
    }
    return sqrt(accum / v_times.size());
}

double calcAverage(vector<int> v_values)
{
    double accum = 0;
    int total = 0;
    for(double value : v_values)
    {
        if(value == 0)
            continue;
        accum += value;
        total++;
    }

    return accum / total;
}

double calcDeviation(vector<int> v_values, double average)
{
    double accum = 0;
    int total = 0;
    for(double value : v_values)
    {
        if(value == 0)
            continue;
        accum += pow(value - average, 2);
        total++;
    }
    return sqrt(accum / total);
}

void Tracking::LocalMapStats2File()
{
    ofstream f;
    f.open("LocalMapTimeStats.txt");
    f << fixed << setprecision(6);
    f << "#Stereo rect[ms], MP culling[ms], MP creation[ms], LBA[ms], KF culling[ms], Total[ms]" << endl;
    for(int i=0; i<mpLocalMapper->vdLMTotal_ms.size(); ++i)
    {
        f << mpLocalMapper->vdKFInsert_ms[i] << "," << mpLocalMapper->vdMPCulling_ms[i] << ","
          << mpLocalMapper->vdMPCreation_ms[i] << "," << mpLocalMapper->vdLBASync_ms[i] << ","
          << mpLocalMapper->vdKFCullingSync_ms[i] <<  "," << mpLocalMapper->vdLMTotal_ms[i] << endl;
    }

    f.close();

    f.open("LBA_Stats.txt");
    f << fixed << setprecision(6);
    f << "#LBA time[ms], KF opt[#], KF fixed[#], MP[#], Edges[#]" << endl;
    for(int i=0; i<mpLocalMapper->vdLBASync_ms.size(); ++i)
    {
        f << mpLocalMapper->vdLBASync_ms[i] << "," << mpLocalMapper->vnLBA_KFopt[i] << ","
          << mpLocalMapper->vnLBA_KFfixed[i] << "," << mpLocalMapper->vnLBA_MPs[i] << ","
          << mpLocalMapper->vnLBA_edges[i] << endl;
    }


    f.close();
}

void Tracking::TrackStats2File()
{
    ofstream f;
    f.open("SessionInfo.txt");
    f << fixed;
    f << "Number of KFs: " << mpAtlas->GetAllKeyFrames().size() << endl;
    f << "Number of MPs: " << mpAtlas->GetAllMapPoints().size() << endl;

    f << "OpenCV version: " << CV_VERSION << endl;

    f.close();

    f.open("TrackingTimeStats.txt");
    f << fixed << setprecision(6);

    f << "#Image Rect[ms], Image Resize[ms], ORB ext[ms], Stereo match[ms], IMU preint[ms], Pose pred[ms], LM track[ms], KF dec[ms], Total[ms]" << endl;

    for(int i=0; i<vdTrackTotal_ms.size(); ++i)
    {
        double stereo_rect = 0.0;
        if(!vdRectStereo_ms.empty())
        {
            stereo_rect = vdRectStereo_ms[i];
        }

        double resize_image = 0.0;
        if(!vdResizeImage_ms.empty())
        {
            resize_image = vdResizeImage_ms[i];
        }

        double stereo_match = 0.0;
        if(!vdStereoMatch_ms.empty())
        {
            stereo_match = vdStereoMatch_ms[i];
        }

        double imu_preint = 0.0;
        if(!vdIMUInteg_ms.empty())
        {
            imu_preint = vdIMUInteg_ms[i];
        }

        f << stereo_rect << "," << resize_image << "," << vdORBExtract_ms[i] << "," << stereo_match << "," << imu_preint << ","
          << vdPosePred_ms[i] <<  "," << vdLMTrack_ms[i] << "," << vdNewKF_ms[i] << "," << vdTrackTotal_ms[i] << endl;
    }

    f.close();
}

void Tracking::PrintTimeStats()
{
    // Save data in files
    TrackStats2File();
    LocalMapStats2File();


    ofstream f;
    f.open("ExecMean.txt");
    f << fixed;
    //Report the mean and std of each one
    std::cout << std::endl << " TIME STATS in ms (mean$\\pm$std)" << std::endl;
    f << " TIME STATS in ms (mean$\\pm$std)" << std::endl;
    cout << "OpenCV version: " << CV_VERSION << endl;
    f << "OpenCV version: " << CV_VERSION << endl;
    std::cout << "---------------------------" << std::endl;
    std::cout << "Tracking" << std::setprecision(5) << std::endl << std::endl;
    f << "---------------------------" << std::endl;
    f << "Tracking" << std::setprecision(5) << std::endl << std::endl;
    double average, deviation;
    if(!vdRectStereo_ms.empty())
    {
        average = calcAverage(vdRectStereo_ms);
        deviation = calcDeviation(vdRectStereo_ms, average);
        std::cout << "Stereo Rectification: " << average << "$\\pm$" << deviation << std::endl;
        f << "Stereo Rectification: " << average << "$\\pm$" << deviation << std::endl;
    }

    if(!vdResizeImage_ms.empty())
    {
        average = calcAverage(vdResizeImage_ms);
        deviation = calcDeviation(vdResizeImage_ms, average);
        std::cout << "Image Resize: " << average << "$\\pm$" << deviation << std::endl;
        f << "Image Resize: " << average << "$\\pm$" << deviation << std::endl;
    }

    average = calcAverage(vdORBExtract_ms);
    deviation = calcDeviation(vdORBExtract_ms, average);
    std::cout << "ORB Extraction: " << average << "$\\pm$" << deviation << std::endl;
    f << "ORB Extraction: " << average << "$\\pm$" << deviation << std::endl;

    if(!vdStereoMatch_ms.empty())
    {
        average = calcAverage(vdStereoMatch_ms);
        deviation = calcDeviation(vdStereoMatch_ms, average);
        std::cout << "Stereo Matching: " << average << "$\\pm$" << deviation << std::endl;
        f << "Stereo Matching: " << average << "$\\pm$" << deviation << std::endl;
    }

    if(!vdIMUInteg_ms.empty())
    {
        average = calcAverage(vdIMUInteg_ms);
        deviation = calcDeviation(vdIMUInteg_ms, average);
        std::cout << "IMU Preintegration: " << average << "$\\pm$" << deviation << std::endl;
        f << "IMU Preintegration: " << average << "$\\pm$" << deviation << std::endl;
    }

    average = calcAverage(vdPosePred_ms);
    deviation = calcDeviation(vdPosePred_ms, average);
    std::cout << "Pose Prediction: " << average << "$\\pm$" << deviation << std::endl;
    f << "Pose Prediction: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(vdLMTrack_ms);
    deviation = calcDeviation(vdLMTrack_ms, average);
    std::cout << "LM Track: " << average << "$\\pm$" << deviation << std::endl;
    f << "LM Track: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(vdNewKF_ms);
    deviation = calcDeviation(vdNewKF_ms, average);
    std::cout << "New KF decision: " << average << "$\\pm$" << deviation << std::endl;
    f << "New KF decision: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(vdTrackTotal_ms);
    deviation = calcDeviation(vdTrackTotal_ms, average);
    std::cout << "Total Tracking: " << average << "$\\pm$" << deviation << std::endl;
    f << "Total Tracking: " << average << "$\\pm$" << deviation << std::endl;

    // Local Mapping time stats
    std::cout << std::endl << std::endl << std::endl;
    std::cout << "Local Mapping" << std::endl << std::endl;
    f << std::endl << "Local Mapping" << std::endl << std::endl;

    average = calcAverage(mpLocalMapper->vdKFInsert_ms);
    deviation = calcDeviation(mpLocalMapper->vdKFInsert_ms, average);
    std::cout << "KF Insertion: " << average << "$\\pm$" << deviation << std::endl;
    f << "KF Insertion: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdMPCulling_ms);
    deviation = calcDeviation(mpLocalMapper->vdMPCulling_ms, average);
    std::cout << "MP Culling: " << average << "$\\pm$" << deviation << std::endl;
    f << "MP Culling: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdMPCreation_ms);
    deviation = calcDeviation(mpLocalMapper->vdMPCreation_ms, average);
    std::cout << "MP Creation: " << average << "$\\pm$" << deviation << std::endl;
    f << "MP Creation: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdLBA_ms);
    deviation = calcDeviation(mpLocalMapper->vdLBA_ms, average);
    std::cout << "LBA: " << average << "$\\pm$" << deviation << std::endl;
    f << "LBA: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdKFCulling_ms);
    deviation = calcDeviation(mpLocalMapper->vdKFCulling_ms, average);
    std::cout << "KF Culling: " << average << "$\\pm$" << deviation << std::endl;
    f << "KF Culling: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdLMTotal_ms);
    deviation = calcDeviation(mpLocalMapper->vdLMTotal_ms, average);
    std::cout << "Total Local Mapping: " << average << "$\\pm$" << deviation << std::endl;
    f << "Total Local Mapping: " << average << "$\\pm$" << deviation << std::endl;

    // Local Mapping LBA complexity
    std::cout << "---------------------------" << std::endl;
    std::cout << std::endl << "LBA complexity (mean$\\pm$std)" << std::endl;
    f << "---------------------------" << std::endl;
    f << std::endl << "LBA complexity (mean$\\pm$std)" << std::endl;

    average = calcAverage(mpLocalMapper->vnLBA_edges);
    deviation = calcDeviation(mpLocalMapper->vnLBA_edges, average);
    std::cout << "LBA Edges: " << average << "$\\pm$" << deviation << std::endl;
    f << "LBA Edges: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vnLBA_KFopt);
    deviation = calcDeviation(mpLocalMapper->vnLBA_KFopt, average);
    std::cout << "LBA KF optimized: " << average << "$\\pm$" << deviation << std::endl;
    f << "LBA KF optimized: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vnLBA_KFfixed);
    deviation = calcDeviation(mpLocalMapper->vnLBA_KFfixed, average);
    std::cout << "LBA KF fixed: " << average << "$\\pm$" << deviation << std::endl;
    f << "LBA KF fixed: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vnLBA_MPs);
    deviation = calcDeviation(mpLocalMapper->vnLBA_MPs, average);
    std::cout << "LBA MP: " << average << "$\\pm$" << deviation << std::endl << std::endl;
    f << "LBA MP: " << average << "$\\pm$" << deviation << std::endl << std::endl;

    std::cout << "LBA executions: " << mpLocalMapper->nLBA_exec << std::endl;
    std::cout << "LBA aborts: " << mpLocalMapper->nLBA_abort << std::endl;
    f << "LBA executions: " << mpLocalMapper->nLBA_exec << std::endl;
    f << "LBA aborts: " << mpLocalMapper->nLBA_abort << std::endl;

    // Map complexity
    std::cout << "---------------------------" << std::endl;
    std::cout << std::endl << "Map complexity" << std::endl;
    std::cout << "KFs in map: " << mpAtlas->GetAllKeyFrames().size() << std::endl;
    std::cout << "MPs in map: " << mpAtlas->GetAllMapPoints().size() << std::endl;
    f << "---------------------------" << std::endl;
    f << std::endl << "Map complexity" << std::endl;
    vector<Map*> vpMaps = mpAtlas->GetAllMaps();
    Map* pBestMap = vpMaps[0];
    for(int i=1; i<vpMaps.size(); ++i)
    {
        if(pBestMap->GetAllKeyFrames().size() < vpMaps[i]->GetAllKeyFrames().size())
        {
            pBestMap = vpMaps[i];
        }
    }

    f << "KFs in map: " << pBestMap->GetAllKeyFrames().size() << std::endl;
    f << "MPs in map: " << pBestMap->GetAllMapPoints().size() << std::endl;

    f << "---------------------------" << std::endl;
    f << std::endl << "Place Recognition (mean$\\pm$std)" << std::endl;
    std::cout << "---------------------------" << std::endl;
    std::cout << std::endl << "Place Recognition (mean$\\pm$std)" << std::endl;
    average = calcAverage(mpLoopClosing->vdDataQuery_ms);
    deviation = calcDeviation(mpLoopClosing->vdDataQuery_ms, average);
    f << "Database Query: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Database Query: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdEstSim3_ms);
    deviation = calcDeviation(mpLoopClosing->vdEstSim3_ms, average);
    f << "SE3 estimation: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "SE3 estimation: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdPRTotal_ms);
    deviation = calcDeviation(mpLoopClosing->vdPRTotal_ms, average);
    f << "Total Place Recognition: " << average << "$\\pm$" << deviation << std::endl << std::endl;
    std::cout << "Total Place Recognition: " << average << "$\\pm$" << deviation << std::endl << std::endl;

    f << std::endl << "Loop Closing (mean$\\pm$std)" << std::endl;
    std::cout << std::endl << "Loop Closing (mean$\\pm$std)" << std::endl;
    average = calcAverage(mpLoopClosing->vdLoopFusion_ms);
    deviation = calcDeviation(mpLoopClosing->vdLoopFusion_ms, average);
    f << "Loop Fusion: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Loop Fusion: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdLoopOptEss_ms);
    deviation = calcDeviation(mpLoopClosing->vdLoopOptEss_ms, average);
    f << "Essential Graph: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Essential Graph: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdLoopTotal_ms);
    deviation = calcDeviation(mpLoopClosing->vdLoopTotal_ms, average);
    f << "Total Loop Closing: " << average << "$\\pm$" << deviation << std::endl << std::endl;
    std::cout << "Total Loop Closing: " << average << "$\\pm$" << deviation << std::endl << std::endl;

    f << "Numb exec: " << mpLoopClosing->nLoop << std::endl;
    std::cout << "Num exec: " << mpLoopClosing->nLoop << std::endl;
    average = calcAverage(mpLoopClosing->vnLoopKFs);
    deviation = calcDeviation(mpLoopClosing->vnLoopKFs, average);
    f << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;

    f << std::endl << "Map Merging (mean$\\pm$std)" << std::endl;
    std::cout << std::endl << "Map Merging (mean$\\pm$std)" << std::endl;
    average = calcAverage(mpLoopClosing->vdMergeMaps_ms);
    deviation = calcDeviation(mpLoopClosing->vdMergeMaps_ms, average);
    f << "Merge Maps: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Merge Maps: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdWeldingBA_ms);
    deviation = calcDeviation(mpLoopClosing->vdWeldingBA_ms, average);
    f << "Welding BA: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Welding BA: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdMergeOptEss_ms);
    deviation = calcDeviation(mpLoopClosing->vdMergeOptEss_ms, average);
    f << "Optimization Ess.: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Optimization Ess.: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdMergeTotal_ms);
    deviation = calcDeviation(mpLoopClosing->vdMergeTotal_ms, average);
    f << "Total Map Merging: " << average << "$\\pm$" << deviation << std::endl << std::endl;
    std::cout << "Total Map Merging: " << average << "$\\pm$" << deviation << std::endl << std::endl;

    f << "Numb exec: " << mpLoopClosing->nMerges << std::endl;
    std::cout << "Num exec: " << mpLoopClosing->nMerges << std::endl;
    average = calcAverage(mpLoopClosing->vnMergeKFs);
    deviation = calcDeviation(mpLoopClosing->vnMergeKFs, average);
    f << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vnMergeMPs);
    deviation = calcDeviation(mpLoopClosing->vnMergeMPs, average);
    f << "Number of MPs: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Number of MPs: " << average << "$\\pm$" << deviation << std::endl;

    f << std::endl << "Full GBA (mean$\\pm$std)" << std::endl;
    std::cout << std::endl << "Full GBA (mean$\\pm$std)" << std::endl;
    average = calcAverage(mpLoopClosing->vdGBA_ms);
    deviation = calcDeviation(mpLoopClosing->vdGBA_ms, average);
    f << "GBA: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "GBA: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdUpdateMap_ms);
    deviation = calcDeviation(mpLoopClosing->vdUpdateMap_ms, average);
    f << "Map Update: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Map Update: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdFGBATotal_ms);
    deviation = calcDeviation(mpLoopClosing->vdFGBATotal_ms, average);
    f << "Total Full GBA: " << average << "$\\pm$" << deviation << std::endl << std::endl;
    std::cout << "Total Full GBA: " << average << "$\\pm$" << deviation << std::endl << std::endl;

    f << "Numb exec: " << mpLoopClosing->nFGBA_exec << std::endl;
    std::cout << "Num exec: " << mpLoopClosing->nFGBA_exec << std::endl;
    f << "Numb abort: " << mpLoopClosing->nFGBA_abort << std::endl;
    std::cout << "Num abort: " << mpLoopClosing->nFGBA_abort << std::endl;
    average = calcAverage(mpLoopClosing->vnGBAKFs);
    deviation = calcDeviation(mpLoopClosing->vnGBAKFs, average);
    f << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vnGBAMPs);
    deviation = calcDeviation(mpLoopClosing->vnGBAMPs, average);
    f << "Number of MPs: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Number of MPs: " << average << "$\\pm$" << deviation << std::endl;

    f.close();

}

#endif

Tracking::~Tracking()
{
    //f_track_stats.close();

}

void Tracking::newParameterLoader(Settings *settings) {
    mpCamera = settings->camera1();
    mpCamera = mpAtlas->AddCamera(mpCamera);

    if(settings->needToUndistort()){
        mDistCoef = settings->camera1DistortionCoef();
    }
    else{
        mDistCoef = cv::Mat::zeros(4,1,CV_32F);
    }

    //TODO: missing image scaling and rectification
    mImageScale = 1.0f;

    mK = cv::Mat::eye(3,3,CV_32F);
    mK.at<float>(0,0) = mpCamera->getParameter(0);
    mK.at<float>(1,1) = mpCamera->getParameter(1);
    mK.at<float>(0,2) = mpCamera->getParameter(2);
    mK.at<float>(1,2) = mpCamera->getParameter(3);

    mK_.setIdentity();
    mK_(0,0) = mpCamera->getParameter(0);
    mK_(1,1) = mpCamera->getParameter(1);
    mK_(0,2) = mpCamera->getParameter(2);
    mK_(1,2) = mpCamera->getParameter(3);

    if((mSensor==System::STEREO || mSensor==System::IMU_STEREO || mSensor==System::IMU_RGBD) &&
        settings->cameraType() == Settings::KannalaBrandt){
        mpCamera2 = settings->camera2();
        mpCamera2 = mpAtlas->AddCamera(mpCamera2);

        mTlr = settings->Tlr();

        mpFrameDrawer->both = true;
    }

    if(mSensor==System::STEREO || mSensor==System::RGBD || mSensor==System::IMU_STEREO || mSensor==System::IMU_RGBD ){
        mbf = settings->bf();
        mThDepth = settings->b() * settings->thDepth();
    }

    if(mSensor==System::RGBD || mSensor==System::IMU_RGBD){
        mDepthMapFactor = settings->depthMapFactor();
        if(fabs(mDepthMapFactor)<1e-5)
            mDepthMapFactor=1;
        else
            mDepthMapFactor = 1.0f/mDepthMapFactor;
    }

    mMinFrames = 0;
    mMaxFrames = settings->fps();
    mbRGB = settings->rgb();

    //ORB parameters
    int nFeatures = settings->nFeatures();
    int nLevels = settings->nLevels();
    int fIniThFAST = settings->initThFAST();
    int fMinThFAST = settings->minThFAST();
    float fScaleFactor = settings->scaleFactor();

    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(mSensor==System::STEREO || mSensor==System::IMU_STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR)
        mpIniORBextractor = new ORBextractor(5*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    //IMU parameters
    Sophus::SE3f Tbc = settings->Tbc();
    mInsertKFsLost = settings->insertKFsWhenLost();
    mImuFreq = settings->imuFrequency();
    mImuPer = 0.001; //1.0 / (double) mImuFreq;     //TODO: ESTO ESTA BIEN?
    float Ng = settings->noiseGyro();
    float Na = settings->noiseAcc();
    float Ngw = settings->gyroWalk();
    float Naw = settings->accWalk();

    const float sf = sqrt(mImuFreq);
    mpImuCalib = new IMU::Calib(Tbc,Ng*sf,Na*sf,Ngw/sf,Naw/sf);

    mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);
}

bool Tracking::ParseCamParamFile(cv::FileStorage &fSettings)
{
    mDistCoef = cv::Mat::zeros(4,1,CV_32F);
    cout << endl << "Camera Parameters: " << endl;
    bool b_miss_params = false;

    string sCameraName = fSettings["Camera.type"];
    if(sCameraName == "PinHole")
    {
        float fx, fy, cx, cy;
        mImageScale = 1.f;

        // Camera calibration parameters
        cv::FileNode node = fSettings["Camera.fx"];
        if(!node.empty() && node.isReal())
        {
            fx = node.real();
        }
        else
        {
            std::cerr << "*Camera.fx parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.fy"];
        if(!node.empty() && node.isReal())
        {
            fy = node.real();
        }
        else
        {
            std::cerr << "*Camera.fy parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.cx"];
        if(!node.empty() && node.isReal())
        {
            cx = node.real();
        }
        else
        {
            std::cerr << "*Camera.cx parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.cy"];
        if(!node.empty() && node.isReal())
        {
            cy = node.real();
        }
        else
        {
            std::cerr << "*Camera.cy parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        // Distortion parameters
        node = fSettings["Camera.k1"];
        if(!node.empty() && node.isReal())
        {
            mDistCoef.at<float>(0) = node.real();
        }
        else
        {
            std::cerr << "*Camera.k1 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.k2"];
        if(!node.empty() && node.isReal())
        {
            mDistCoef.at<float>(1) = node.real();
        }
        else
        {
            std::cerr << "*Camera.k2 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.p1"];
        if(!node.empty() && node.isReal())
        {
            mDistCoef.at<float>(2) = node.real();
        }
        else
        {
            std::cerr << "*Camera.p1 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.p2"];
        if(!node.empty() && node.isReal())
        {
            mDistCoef.at<float>(3) = node.real();
        }
        else
        {
            std::cerr << "*Camera.p2 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.k3"];
        if(!node.empty() && node.isReal())
        {
            mDistCoef.resize(5);
            mDistCoef.at<float>(4) = node.real();
        }

        node = fSettings["Camera.imageScale"];
        if(!node.empty() && node.isReal())
        {
            mImageScale = node.real();
        }

        if(b_miss_params)
        {
            return false;
        }

        if(mImageScale != 1.f)
        {
            // K matrix parameters must be scaled.
            fx = fx * mImageScale;
            fy = fy * mImageScale;
            cx = cx * mImageScale;
            cy = cy * mImageScale;
        }

        vector<float> vCamCalib{fx,fy,cx,cy};

        mpCamera = new Pinhole(vCamCalib);

        mpCamera = mpAtlas->AddCamera(mpCamera);

        std::cout << "- Camera: Pinhole" << std::endl;
        std::cout << "- Image scale: " << mImageScale << std::endl;
        std::cout << "- fx: " << fx << std::endl;
        std::cout << "- fy: " << fy << std::endl;
        std::cout << "- cx: " << cx << std::endl;
        std::cout << "- cy: " << cy << std::endl;
        std::cout << "- k1: " << mDistCoef.at<float>(0) << std::endl;
        std::cout << "- k2: " << mDistCoef.at<float>(1) << std::endl;


        std::cout << "- p1: " << mDistCoef.at<float>(2) << std::endl;
        std::cout << "- p2: " << mDistCoef.at<float>(3) << std::endl;

        if(mDistCoef.rows==5)
            std::cout << "- k3: " << mDistCoef.at<float>(4) << std::endl;

        mK = cv::Mat::eye(3,3,CV_32F);
        mK.at<float>(0,0) = fx;
        mK.at<float>(1,1) = fy;
        mK.at<float>(0,2) = cx;
        mK.at<float>(1,2) = cy;

        mK_.setIdentity();
        mK_(0,0) = fx;
        mK_(1,1) = fy;
        mK_(0,2) = cx;
        mK_(1,2) = cy;
    }
    else if(sCameraName == "KannalaBrandt8")
    {
        float fx, fy, cx, cy;
        float k1, k2, k3, k4;
        mImageScale = 1.f;

        // Camera calibration parameters
        cv::FileNode node = fSettings["Camera.fx"];
        if(!node.empty() && node.isReal())
        {
            fx = node.real();
        }
        else
        {
            std::cerr << "*Camera.fx parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }
        node = fSettings["Camera.fy"];
        if(!node.empty() && node.isReal())
        {
            fy = node.real();
        }
        else
        {
            std::cerr << "*Camera.fy parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.cx"];
        if(!node.empty() && node.isReal())
        {
            cx = node.real();
        }
        else
        {
            std::cerr << "*Camera.cx parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.cy"];
        if(!node.empty() && node.isReal())
        {
            cy = node.real();
        }
        else
        {
            std::cerr << "*Camera.cy parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        // Distortion parameters
        node = fSettings["Camera.k1"];
        if(!node.empty() && node.isReal())
        {
            k1 = node.real();
        }
        else
        {
            std::cerr << "*Camera.k1 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }
        node = fSettings["Camera.k2"];
        if(!node.empty() && node.isReal())
        {
            k2 = node.real();
        }
        else
        {
            std::cerr << "*Camera.k2 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.k3"];
        if(!node.empty() && node.isReal())
        {
            k3 = node.real();
        }
        else
        {
            std::cerr << "*Camera.k3 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.k4"];
        if(!node.empty() && node.isReal())
        {
            k4 = node.real();
        }
        else
        {
            std::cerr << "*Camera.k4 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.imageScale"];
        if(!node.empty() && node.isReal())
        {
            mImageScale = node.real();
        }

        if(!b_miss_params)
        {
            if(mImageScale != 1.f)
            {
                // K matrix parameters must be scaled.
                fx = fx * mImageScale;
                fy = fy * mImageScale;
                cx = cx * mImageScale;
                cy = cy * mImageScale;
            }

            vector<float> vCamCalib{fx,fy,cx,cy,k1,k2,k3,k4};
            mpCamera = new KannalaBrandt8(vCamCalib);
            mpCamera = mpAtlas->AddCamera(mpCamera);
            std::cout << "- Camera: Fisheye" << std::endl;
            std::cout << "- Image scale: " << mImageScale << std::endl;
            std::cout << "- fx: " << fx << std::endl;
            std::cout << "- fy: " << fy << std::endl;
            std::cout << "- cx: " << cx << std::endl;
            std::cout << "- cy: " << cy << std::endl;
            std::cout << "- k1: " << k1 << std::endl;
            std::cout << "- k2: " << k2 << std::endl;
            std::cout << "- k3: " << k3 << std::endl;
            std::cout << "- k4: " << k4 << std::endl;

            mK = cv::Mat::eye(3,3,CV_32F);
            mK.at<float>(0,0) = fx;
            mK.at<float>(1,1) = fy;
            mK.at<float>(0,2) = cx;
            mK.at<float>(1,2) = cy;

            mK_.setIdentity();
            mK_(0,0) = fx;
            mK_(1,1) = fy;
            mK_(0,2) = cx;
            mK_(1,2) = cy;
        }

        if(mSensor==System::STEREO || mSensor==System::IMU_STEREO || mSensor==System::IMU_RGBD){
            // Right camera
            // Camera calibration parameters
            cv::FileNode node = fSettings["Camera2.fx"];
            if(!node.empty() && node.isReal())
            {
                fx = node.real();
            }
            else
            {
                std::cerr << "*Camera2.fx parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }
            node = fSettings["Camera2.fy"];
            if(!node.empty() && node.isReal())
            {
                fy = node.real();
            }
            else
            {
                std::cerr << "*Camera2.fy parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera2.cx"];
            if(!node.empty() && node.isReal())
            {
                cx = node.real();
            }
            else
            {
                std::cerr << "*Camera2.cx parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera2.cy"];
            if(!node.empty() && node.isReal())
            {
                cy = node.real();
            }
            else
            {
                std::cerr << "*Camera2.cy parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            // Distortion parameters
            node = fSettings["Camera2.k1"];
            if(!node.empty() && node.isReal())
            {
                k1 = node.real();
            }
            else
            {
                std::cerr << "*Camera2.k1 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }
            node = fSettings["Camera2.k2"];
            if(!node.empty() && node.isReal())
            {
                k2 = node.real();
            }
            else
            {
                std::cerr << "*Camera2.k2 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera2.k3"];
            if(!node.empty() && node.isReal())
            {
                k3 = node.real();
            }
            else
            {
                std::cerr << "*Camera2.k3 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera2.k4"];
            if(!node.empty() && node.isReal())
            {
                k4 = node.real();
            }
            else
            {
                std::cerr << "*Camera2.k4 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }


            int leftLappingBegin = -1;
            int leftLappingEnd = -1;

            int rightLappingBegin = -1;
            int rightLappingEnd = -1;

            node = fSettings["Camera.lappingBegin"];
            if(!node.empty() && node.isInt())
            {
                leftLappingBegin = node.operator int();
            }
            else
            {
                std::cout << "WARNING: Camera.lappingBegin not correctly defined" << std::endl;
            }
            node = fSettings["Camera.lappingEnd"];
            if(!node.empty() && node.isInt())
            {
                leftLappingEnd = node.operator int();
            }
            else
            {
                std::cout << "WARNING: Camera.lappingEnd not correctly defined" << std::endl;
            }
            node = fSettings["Camera2.lappingBegin"];
            if(!node.empty() && node.isInt())
            {
                rightLappingBegin = node.operator int();
            }
            else
            {
                std::cout << "WARNING: Camera2.lappingBegin not correctly defined" << std::endl;
            }
            node = fSettings["Camera2.lappingEnd"];
            if(!node.empty() && node.isInt())
            {
                rightLappingEnd = node.operator int();
            }
            else
            {
                std::cout << "WARNING: Camera2.lappingEnd not correctly defined" << std::endl;
            }

            node = fSettings["Tlr"];
            cv::Mat cvTlr;
            if(!node.empty())
            {
                cvTlr = node.mat();
                if(cvTlr.rows != 3 || cvTlr.cols != 4)
                {
                    std::cerr << "*Tlr matrix have to be a 3x4 transformation matrix*" << std::endl;
                    b_miss_params = true;
                }
            }
            else
            {
                std::cerr << "*Tlr matrix doesn't exist*" << std::endl;
                b_miss_params = true;
            }

            if(!b_miss_params)
            {
                if(mImageScale != 1.f)
                {
                    // K matrix parameters must be scaled.
                    fx = fx * mImageScale;
                    fy = fy * mImageScale;
                    cx = cx * mImageScale;
                    cy = cy * mImageScale;

                    leftLappingBegin = leftLappingBegin * mImageScale;
                    leftLappingEnd = leftLappingEnd * mImageScale;
                    rightLappingBegin = rightLappingBegin * mImageScale;
                    rightLappingEnd = rightLappingEnd * mImageScale;
                }

                static_cast<KannalaBrandt8*>(mpCamera)->mvLappingArea[0] = leftLappingBegin;
                static_cast<KannalaBrandt8*>(mpCamera)->mvLappingArea[1] = leftLappingEnd;

                mpFrameDrawer->both = true;

                vector<float> vCamCalib2{fx,fy,cx,cy,k1,k2,k3,k4};
                mpCamera2 = new KannalaBrandt8(vCamCalib2);
                mpCamera2 = mpAtlas->AddCamera(mpCamera2);

                mTlr = Converter::toSophus(cvTlr);

                static_cast<KannalaBrandt8*>(mpCamera2)->mvLappingArea[0] = rightLappingBegin;
                static_cast<KannalaBrandt8*>(mpCamera2)->mvLappingArea[1] = rightLappingEnd;

                std::cout << "- Camera1 Lapping: " << leftLappingBegin << ", " << leftLappingEnd << std::endl;

                std::cout << std::endl << "Camera2 Parameters:" << std::endl;
                std::cout << "- Camera: Fisheye" << std::endl;
                std::cout << "- Image scale: " << mImageScale << std::endl;
                std::cout << "- fx: " << fx << std::endl;
                std::cout << "- fy: " << fy << std::endl;
                std::cout << "- cx: " << cx << std::endl;
                std::cout << "- cy: " << cy << std::endl;
                std::cout << "- k1: " << k1 << std::endl;
                std::cout << "- k2: " << k2 << std::endl;
                std::cout << "- k3: " << k3 << std::endl;
                std::cout << "- k4: " << k4 << std::endl;

                std::cout << "- mTlr: \n" << cvTlr << std::endl;

                std::cout << "- Camera2 Lapping: " << rightLappingBegin << ", " << rightLappingEnd << std::endl;
            }
        }

        if(b_miss_params)
        {
            return false;
        }

    }
    else
    {
        std::cerr << "*Not Supported Camera Sensor*" << std::endl;
        std::cerr << "Check an example configuration file with the desired sensor" << std::endl;
    }

    if(mSensor==System::STEREO || mSensor==System::RGBD || mSensor==System::IMU_STEREO || mSensor==System::IMU_RGBD )
    {
        cv::FileNode node = fSettings["Camera.bf"];
        if(!node.empty() && node.isReal())
        {
            mbf = node.real();
            if(mImageScale != 1.f)
            {
                mbf *= mImageScale;
            }
        }
        else
        {
            std::cerr << "*Camera.bf parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

    }

    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;

    cout << "- fps: " << fps << endl;


    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if(mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    if(mSensor==System::STEREO || mSensor==System::RGBD || mSensor==System::IMU_STEREO || mSensor==System::IMU_RGBD)
    {
        float fx = mpCamera->getParameter(0);
        cv::FileNode node = fSettings["ThDepth"];
        if(!node.empty()  && node.isReal())
        {
            mThDepth = node.real();
            mThDepth = mbf*mThDepth/fx;
            cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
        }
        else
        {
            std::cerr << "*ThDepth parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }


    }

    if(mSensor==System::RGBD || mSensor==System::IMU_RGBD)
    {
        cv::FileNode node = fSettings["DepthMapFactor"];
        if(!node.empty() && node.isReal())
        {
            mDepthMapFactor = node.real();
            if(fabs(mDepthMapFactor)<1e-5)
                mDepthMapFactor=1;
            else
                mDepthMapFactor = 1.0f/mDepthMapFactor;
        }
        else
        {
            std::cerr << "*DepthMapFactor parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

    }

    if(b_miss_params)
    {
        return false;
    }

    return true;
}

bool Tracking::ParseORBParamFile(cv::FileStorage &fSettings)
{
    bool b_miss_params = false;
    int nFeatures, nLevels, fIniThFAST, fMinThFAST;
    float fScaleFactor;

    cv::FileNode node = fSettings["ORBextractor.nFeatures"];
    if(!node.empty() && node.isInt())
    {
        nFeatures = node.operator int();
    }
    else
    {
        std::cerr << "*ORBextractor.nFeatures parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["ORBextractor.scaleFactor"];
    if(!node.empty() && node.isReal())
    {
        fScaleFactor = node.real();
    }
    else
    {
        std::cerr << "*ORBextractor.scaleFactor parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["ORBextractor.nLevels"];
    if(!node.empty() && node.isInt())
    {
        nLevels = node.operator int();
    }
    else
    {
        std::cerr << "*ORBextractor.nLevels parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["ORBextractor.iniThFAST"];
    if(!node.empty() && node.isInt())
    {
        fIniThFAST = node.operator int();
    }
    else
    {
        std::cerr << "*ORBextractor.iniThFAST parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["ORBextractor.minThFAST"];
    if(!node.empty() && node.isInt())
    {
        fMinThFAST = node.operator int();
    }
    else
    {
        std::cerr << "*ORBextractor.minThFAST parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }

    if(b_miss_params)
    {
        return false;
    }

    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(mSensor==System::STEREO || mSensor==System::IMU_STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR)
        mpIniORBextractor = new ORBextractor(5*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    cout << endl << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    return true;
}

bool Tracking::ParseIMUParamFile(cv::FileStorage &fSettings)
{
    bool b_miss_params = false;

    cv::Mat cvTbc;
    cv::FileNode node = fSettings["Tbc"];
    if(!node.empty())
    {
        cvTbc = node.mat();
        if(cvTbc.rows != 4 || cvTbc.cols != 4)
        {
            std::cerr << "*Tbc matrix have to be a 4x4 transformation matrix*" << std::endl;
            b_miss_params = true;
        }
    }
    else
    {
        std::cerr << "*Tbc matrix doesn't exist*" << std::endl;
        b_miss_params = true;
    }
    cout << endl;
    cout << "Left camera to Imu Transform (Tbc): " << endl << cvTbc << endl;
    Eigen::Matrix<float,4,4,Eigen::RowMajor> eigTbc(cvTbc.ptr<float>(0));
    Sophus::SE3f Tbc(eigTbc);

    node = fSettings["InsertKFsWhenLost"];
    mInsertKFsLost = true;
    if(!node.empty() && node.isInt())
    {
        mInsertKFsLost = (bool) node.operator int();
    }

    if(!mInsertKFsLost)
        cout << "Do not insert keyframes when lost visual tracking " << endl;



    float Ng, Na, Ngw, Naw;

    node = fSettings["IMU.Frequency"];
    if(!node.empty() && node.isInt())
    {
        mImuFreq = node.operator int();
        mImuPer = 0.001; //1.0 / (double) mImuFreq;
    }
    else
    {
        std::cerr << "*IMU.Frequency parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.NoiseGyro"];
    if(!node.empty() && node.isReal())
    {
        Ng = node.real();
    }
    else
    {
        std::cerr << "*IMU.NoiseGyro parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.NoiseAcc"];
    if(!node.empty() && node.isReal())
    {
        Na = node.real();
    }
    else
    {
        std::cerr << "*IMU.NoiseAcc parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.GyroWalk"];
    if(!node.empty() && node.isReal())
    {
        Ngw = node.real();
    }
    else
    {
        std::cerr << "*IMU.GyroWalk parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.AccWalk"];
    if(!node.empty() && node.isReal())
    {
        Naw = node.real();
    }
    else
    {
        std::cerr << "*IMU.AccWalk parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.fastInit"];
    mFastInit = false;
    if(!node.empty())
    {
        mFastInit = static_cast<int>(fSettings["IMU.fastInit"]) != 0;
    }

    if(mFastInit)
        cout << "Fast IMU initialization. Acceleration is not checked \n";

    if(b_miss_params)
    {
        return false;
    }

    const float sf = sqrt(mImuFreq);
    cout << endl;
    cout << "IMU frequency: " << mImuFreq << " Hz" << endl;
    cout << "IMU gyro noise: " << Ng << " rad/s/sqrt(Hz)" << endl;
    cout << "IMU gyro walk: " << Ngw << " rad/s^2/sqrt(Hz)" << endl;
    cout << "IMU accelerometer noise: " << Na << " m/s^2/sqrt(Hz)" << endl;
    cout << "IMU accelerometer walk: " << Naw << " m/s^3/sqrt(Hz)" << endl;

    mpImuCalib = new IMU::Calib(Tbc,Ng*sf,Na*sf,Ngw/sf,Naw/sf);

    mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);


    return true;
}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing=pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer=pViewer;
}

void Tracking::SetStepByStep(bool bSet)
{
    bStepByStep = bSet;
}

bool Tracking::GetStepByStep()
{
    return bStepByStep;
}

void Tracking::SetFeatureMask(const cv::Mat &mask)
{
    mpORBextractorLeft->SetAllowedMask(mask);
    mpIniORBextractor->SetAllowedMask(mask);
}

void Tracking::SetExternalTagObservation(
    const Sophus::SE3f &inputTwc,
    const float confidence,
    const std::vector<Eigen::Vector3f> &inputWorldPoints,
    const std::vector<cv::Point2f> &imagePoints,
    const bool valid,
    const std::vector<float> &pointWeights,
    const std::vector<int> &tagIds,
    bool partial, int trackedCorners, float trackAgeS, bool inputInAtlasWorld)
{
    Sophus::SE3f Twc=inputTwc;
    std::vector<Eigen::Vector3f> worldPoints=inputWorldPoints;
    std::string gaugeReason;
    const bool gaugeValid=!valid || MarkerGraphCoordinator::AlignMarkerInput(
        mpAtlas->GetCurrentMap(),Twc,worldPoints,tagIds,pointWeights,partial,gaugeReason,
        mpCamera,imagePoints,inputInAtlasWorld);
    mMarkerTrackingStatus=MarkerTrackingStatus();
    auto& status=mMarkerTrackingStatus;
    status.partial=partial; status.ageS=trackAgeS;
    mbHasTrackedTagObservation=false;
    mbHasExternalTagObservation =
        mbTagFusionEnabled && valid && gaugeValid && confidence >= (partial ? 0.15f : 0.35f) &&
        !worldPoints.empty() && worldPoints.size() == imagePoints.size() &&
        (pointWeights.empty() || pointWeights.size() == worldPoints.size());
    for(const float weight : pointWeights)
        mbHasExternalTagObservation = mbHasExternalTagObservation &&
            std::isfinite(weight) && weight > 0.0f && weight <= 1.0f;
    if(mbHasExternalTagObservation) {
        const Sophus::SE3f Tcw=Twc.inverse();
        double error=0.0;
        int strong=0;
        for(size_t i=0;i<worldPoints.size();++i) {
            const Eigen::Vector3f p=Tcw*worldPoints[i];
            if(!p.allFinite() || p.z()<=0) {mbHasExternalTagObservation=false;break;}
            const Eigen::Vector2f uv=mpCamera->project(p);
            const Eigen::Vector2f observation(imagePoints[i].x,imagePoints[i].y);
            error+=(uv-observation).squaredNorm();
            if(pointWeights.empty() || pointWeights[i]>=0.99f) ++strong;
        }
        status.reprojectionPx=std::sqrt(error/worldPoints.size());
        mbHasExternalTagObservation = mbHasExternalTagObservation &&
            (partial ? worldPoints.size()>=3 && strong==0 : strong>=4) &&
            status.reprojectionPx <= (partial ? 1.5 : 3.0);
    }
    status.reason=!gaugeValid ? gaugeReason : (valid ? "geometry_rejected" : "no_observation");
    if(!partial && mbHasExternalTagObservation && mState==MARKER_TRACKING &&
       mLastFrame.HasPose() && mpAtlas->GetCurrentMap()->mbMetric)
    {
        std::map<int,int> strongCounts;
        for(size_t i=0;i<tagIds.size();++i)
            if(pointWeights.empty() || (i<pointWeights.size() && pointWeights[i]>=0.99f))
                ++strongCounts[tagIds[i]];
        std::set<int> completeStrongMarkerIds;
        for(const auto &item:strongCounts)
            if(item.second>=4) completeStrongMarkerIds.insert(item.first);
        std::map<int,int> previousStrongCounts;
        for(size_t i=0;i<mvExternalTagIds.size();++i)
            if(mvExternalTagPointWeights.empty() ||
               (i<mvExternalTagPointWeights.size() && mvExternalTagPointWeights[i]>=0.99f))
                ++previousStrongCounts[mvExternalTagIds[i]];
        std::set<int> previousStrongMarkerIds;
        for(const auto &item:previousStrongCounts)
            if(item.second>=4) previousStrongMarkerIds.insert(item.first);
        const bool switchesSingleMarker=completeStrongMarkerIds.size()==1 &&
            previousStrongMarkerIds.size()==1 &&
            *completeStrongMarkerIds.begin()!=*previousStrongMarkerIds.begin();
        const auto markerCenter=[](int id,const std::vector<int>& ids,
                                   const std::vector<Eigen::Vector3f>& points,
                                   Eigen::Vector3f& center) {
            center.setZero(); int count=0;
            for(size_t i=0;i<ids.size() && i<points.size();++i)
                if(ids[i]==id) {center+=points[i]; ++count;}
            if(count<4) return false;
            center/=float(count); return center.allFinite();
        };
        Eigen::Vector3f currentMarkerCenter,previousMarkerCenter;
        const bool nearbyMarkerSwitch=switchesSingleMarker &&
            markerCenter(*completeStrongMarkerIds.begin(),tagIds,worldPoints,currentMarkerCenter) &&
            markerCenter(*previousStrongMarkerIds.begin(),mvExternalTagIds,
                         mvExternalTagWorldPoints,previousMarkerCenter) &&
            (currentMarkerCenter-previousMarkerCenter).norm()<=0.30f;
        if(nearbyMarkerSwitch)
        {
            const Sophus::SE3f previousTwc=mLastFrame.GetPose().inverse();
            const float distance=(Twc.translation()-previousTwc.translation()).norm();
            const float angle=(Twc.so3()*previousTwc.so3().inverse()).log().norm();
            // Marker-only localization has no simultaneous visual-map check.
            // When the sole decoded ID changes between nearby markers, reject
            // a physically implausible one-frame pose jump rather than publish
            // stale data or false wrist motion. A changed view of the same tag,
            // or a cut/relocalization between distant tags, is still allowed;
            // thresholds
            // scale with input FPS (1.5 m/s, 360 deg/s) and retain conservative
            // floors for timestamp/fps rounding.
            const float fps=std::max(1,mMaxFrames);
            const float maximumDistance=std::max(0.015f,1.5f/fps);
            const float maximumAngle=std::max(0.0872665f,6.2831853f/fps);
            if(distance>maximumDistance || angle>maximumAngle)
            {
                mbHasExternalTagObservation=false;
                status.reason="single_marker_motion_gate";
            }
        }
    }
    if(partial && mbHasExternalTagObservation) {
        Map* map=mpAtlas->GetCurrentMap();
        bool known=tagIds.size()==worldPoints.size();
        float maxArea=0;
        for(size_t i=1;i<imagePoints.size();++i)
            for(size_t j=i+1;j<imagePoints.size();++j) {
                const auto a=imagePoints[i]-imagePoints[0],b=imagePoints[j]-imagePoints[0];
                maxArea=std::max(maxArea,std::abs(a.x*b.y-a.y*b.x)*0.5f);
            }
        known=known && maxArea>=40.0f;
        for(float weight:pointWeights) known=known && weight<=0.25f;
        for(size_t i=0;known && i<worldPoints.size();++i) {
            auto marker=map->mStaticTags.find(tagIds[i]);
            bool found=false;
            if(marker!=map->mStaticTags.end())
                for(size_t j=0;j+2<marker->second.size();j+=3)
                    if((worldPoints[i]-Eigen::Vector3f(marker->second[j],marker->second[j+1],marker->second[j+2])).norm()<1e-5f)
                        found=true;
            known=known && found;
            for(size_t j=0;j<i;++j)
                if(tagIds[i]==tagIds[j] && (worldPoints[i]-worldPoints[j]).norm()<1e-5f) known=false;
        }
        bool prior=map->mbMetric && mLastFrame.HasPose() &&
            (mState==OK || mState==MARKER_TRACKING) && known &&
            trackAgeS>0 && trackAgeS<=0.30f && trackedCorners==int(worldPoints.size());
        if(prior && worldPoints.size()==3 && trackAgeS>0.10f) {
            prior=false;
            status.reason="three_corner_age_gate";
        }
        if(prior) {
            const Sophus::SE3f previous=mLastFrame.GetPose().inverse();
            const float distance=(Twc.translation()-previous.translation()).norm();
            const float angle=(Twc.so3()*previous.so3().inverse()).log().norm();
            // A three-corner fit has no residual redundancy. Mirror the
            // Python admission gate, including for externally cached hints.
            if(worldPoints.size()==3 && (distance>0.010f || angle>0.0523599f)) {
                prior=false;
                status.reason="three_corner_motion_gate";
            }
            else prior=distance<=0.03f && angle<=0.20944f;
        }
        mbHasExternalTagObservation=prior;
        if(!prior && status.reason!="three_corner_motion_gate" && status.reason!="three_corner_age_gate")
            status.reason="partial_requires_known_metric_map_and_recent_pose";
    }
    const bool accepted=mbHasExternalTagObservation;
    mbHasTrackedTagObservation=accepted && partial;
    mbHasExternalTagObservation=accepted && !partial;
    status.accepted=accepted;
    status.confidence=accepted ? confidence : 0;
    if(accepted) {
        status.reason=partial ? "partial_tracked" : "decoded";
        status.trackedCorners=std::max(0,std::min(trackedCorners,int(imagePoints.size())));
        for(size_t i=imagePoints.size()-status.trackedCorners;i<imagePoints.size();++i) {
            status.pixels.push_back(imagePoints[i]);
            status.ids.push_back(i<tagIds.size()?tagIds[i]:-1);
        }
    }
    mvExternalTagIds = tagIds;
    mExternalTagConfidence = accepted ? confidence : 0.0f;
    mvExternalTagWorldPoints = accepted
        ? worldPoints : std::vector<Eigen::Vector3f>();
    mvExternalTagImagePoints = accepted
        ? imagePoints : std::vector<cv::Point2f>();
    mvExternalTagPointWeights = accepted
        ? pointWeights : std::vector<float>();
    if(accepted)
        mExternalTagTwc = Twc;
}

float Tracking::GetRecoveredTagMetricScale() const
{
    return mRecoveredTagMetricScale;
}

bool Tracking::IsTagMetricAligned() const
{
    return mbTagMetricAligned;
}

unsigned int Tracking::GetTagKeyFramesAccepted() const
{
    return mnTagKeyFramesAccepted;
}

unsigned int Tracking::GetTagKeyFramesRejected() const
{
    return mnTagKeyFramesRejected;
}

unsigned int Tracking::GetTagPoseConstraintsApplied() const
{
    return mnTagPoseConstraintsApplied;
}

void Tracking::AttachCurrentTagObservation(
    KeyFrame* pKF, const bool initialObservation)
{
    const bool valid = initialObservation
        ? mbHasInitialTagObservation : mbHasExternalTagObservation;
    if(!valid)
        return;
    pKF->mvTagIds = initialObservation ? mvInitialTagIds : mvExternalTagIds;
    pKF->mbHasTagObservation = true;
    pKF->mbTagObservationActive = mbTagMetricAligned;
    pKF->mTagObservationConfidence = initialObservation
        ? mInitialTagConfidence : mExternalTagConfidence;
    pKF->mvTagWorldPoints = initialObservation
        ? mvInitialTagWorldPoints : mvExternalTagWorldPoints;
    pKF->mvTagImagePoints = initialObservation
        ? mvInitialTagImagePoints : mvExternalTagImagePoints;
    pKF->mvTagPointWeights = initialObservation
        ? mvInitialTagPointWeights : mvExternalTagPointWeights;
    if(pKF->GetMap()->mbMetric) {
        pKF->GetMap()->mbRigidMarkerLayout =
            pKF->GetMap()->mbRigidMarkerLayout || mbRigidMarkerLayout;
        for(size_t i=0;i+3<pKF->mvTagWorldPoints.size() && i<pKF->mvTagIds.size();i+=4) {
            const int id=pKF->mvTagIds[i];
            if(i+3>=pKF->mvTagIds.size() || pKF->mvTagIds[i+1]!=id ||
               pKF->mvTagIds[i+2]!=id || pKF->mvTagIds[i+3]!=id) continue;
            if(pKF->GetMap()->mStaticTags.count(id)) continue;
            std::vector<float> corners;
            for(size_t j=i;j<i+4;++j)
                for(int axis=0;axis<3;++axis) corners.push_back(pKF->mvTagWorldPoints[j](axis));
            pKF->GetMap()->mStaticTags[id]=corners;
        }
    }
    RecordMarkerKeyFrame(pKF);
}

// Only complete, strong decoded markers produce information-gain events.
// Weak/flow-tracked corners never request a keyframe or a scale anchor.
static std::set<int> StrongMarkerIds(const std::vector<int>& ids,
                                   const std::vector<float>& weights)
{
    std::map<int,int> counts;
    for(size_t i=0;i<ids.size();++i)
        if(weights.empty() || (i<weights.size() && weights[i]>=0.99f)) ++counts[ids[i]];
    std::set<int> result;
    for(const auto& item:counts) if(item.second>=4) result.insert(item.first);
    return result;
}

void Tracking::UpdateMarkerKeyFrameEvents()
{
    mMarkerKeyFrameEvent.clear();
    mnMarkerEventKeyFrameId=-1;
    Map* map=mpAtlas->GetCurrentMap();
    if(mpMarkerEventMap!=map) {
        mpMarkerEventMap=map;
        mLastLocalizedFrameTime=-1.0;
        mPendingMarkerEvents.clear(); mKeyframedMarkerIds.clear();
        mLastDecodedMarkerTime.clear();
        for(KeyFrame* keyframe:map->GetAllKeyFrames()) {
            const auto ids=StrongMarkerIds(keyframe->mvTagIds,keyframe->mvTagPointWeights);
            mKeyframedMarkerIds.insert(ids.begin(),ids.end());
        }
    }
    // Losing one decoded ID is not losing localization. ORB or another tag
    // may have continuously localized the camera during that decoding gap.
    const bool previouslyLocalized=(mState==OK || mState==MARKER_TRACKING) && mLastFrame.HasPose();
    if(previouslyLocalized) mLastLocalizedFrameTime=mLastFrame.mTimeStamp;
    const bool recoveredAfterLoss=!previouslyLocalized && mLastLocalizedFrameTime>=0 &&
        mCurrentFrame.mTimeStamp-mLastLocalizedFrameTime>=0.5;
    if(!mbHasExternalTagObservation) return;
    for(int id:StrongMarkerIds(mvExternalTagIds,mvExternalTagPointWeights)) {
        if(!mKeyframedMarkerIds.count(id)) mPendingMarkerEvents[id]="first_seen";
        else if(recoveredAfterLoss) mPendingMarkerEvents[id]="relocalized";
        else {
            const auto last=mLastDecodedMarkerTime.find(id);
            // A known anchor can return for only a few decoded frames while
            // ORB remains OK. Preserve that new metric evidence in a KF;
            // ordinary geometric spacing can otherwise miss the entire visit.
            // Brief decode flicker and continuous visibility do not qualify.
            if(map->mbMetric && map->mStaticTags.count(id) &&
               last!=mLastDecodedMarkerTime.end() &&
               mCurrentFrame.mTimeStamp-last->second>=0.5 &&
               !mPendingMarkerEvents.count(id))
                mPendingMarkerEvents[id]="anchor_reobserved";
        }
        mLastDecodedMarkerTime[id]=mCurrentFrame.mTimeStamp;
    }
}

bool Tracking::HasMarkerKeyFrameEvent() const
{
    if(!mbHasExternalTagObservation) return false;
    for(int id:StrongMarkerIds(mvExternalTagIds,mvExternalTagPointWeights))
        if(mPendingMarkerEvents.count(id)) return true;
    return false;
}

void Tracking::RecordMarkerKeyFrame(KeyFrame* keyframe)
{
    bool recordedRecovery=false;
    for(int id:StrongMarkerIds(keyframe->mvTagIds,keyframe->mvTagPointWeights)) {
        mKeyframedMarkerIds.insert(id);
        auto pending=mPendingMarkerEvents.find(id);
        if(pending==mPendingMarkerEvents.end()) continue;
        if(keyframe->mnFrameId==mCurrentFrame.mnId) {
            if(!mMarkerKeyFrameEvent.empty()) mMarkerKeyFrameEvent+=",";
            mMarkerKeyFrameEvent+=pending->second+":"+std::to_string(id);
            mnMarkerEventKeyFrameId=keyframe->mnId;
            recordedRecovery=recordedRecovery || pending->second=="relocalized";
        }
        mPendingMarkerEvents.erase(pending);
    }
    // One loss/recovery episode warrants one information keyframe, not a
    // later duplicate for each other ID visible when recovery was requested.
    if(recordedRecovery)
        for(auto it=mPendingMarkerEvents.begin();it!=mPendingMarkerEvents.end();)
            if(it->second=="relocalized") it=mPendingMarkerEvents.erase(it);
            else ++it;
}

void Tracking::InsertMarkerOnlyEventKeyFrame()
{
    if(!HasMarkerKeyFrameEvent() || mpLocalMapper->isStopped() ||
       mpLocalMapper->stopRequested() || !mpLocalMapper->AcceptKeyFrames()) return;
    if(!mpLocalMapper->SetNotStop(true)) return;
    Map* map=mpAtlas->GetCurrentMap();
    Frame measured(mCurrentFrame);
    // A failed ORB attempt may leave candidate matches. Do not turn them
    // into valid map observations merely because marker localization works.
    std::fill(measured.mvpMapPoints.begin(),measured.mvpMapPoints.end(),nullptr);
    auto* keyframe=new KeyFrame(measured,map,mpKeyFrameDB);
    keyframe->ComputeBoW();
    AttachCurrentTagObservation(keyframe);
    mpAtlas->AddKeyFrame(keyframe);
    if(mpReferenceKF) keyframe->ChangeParent(mpReferenceKF);
    // Store tag factors in Atlas (including final BA), but do not run the
    // background mapper/culling or place-recognition database on an image
    // with no measured ORB matches. Common-marker merging has its own
    // validated path and does not need an empty BoW landmark.
    mpLocalMapper->SetNotStop(false);
}

void Tracking::CancelPendingTagAlignment()
{
    mvTagScaleSamples.clear();
    mbHasTagScaleReference = false;
    mTagScaleReferenceTimestamp = 0.0;
    mpTagScaleReferenceMap = nullptr;
    if(!mbTagAlignmentPending)
        return;
    mbTagAlignmentPending = false;
    mpTagAlignmentMap = nullptr;
    mPendingTagMetricScale = 0.0f;
    mpLocalMapper->ReleaseTagAlignmentStop();
}

bool Tracking::TryAlignMapToTagWorld()
{
    if(mbTagAlignmentPending &&
       (mpTagAlignmentMap != mpAtlas->GetCurrentMap() || mbTagMetricAligned))
        CancelPendingTagAlignment();
    if(!mbTagFusionEnabled || mbTagMetricAligned || !mCurrentFrame.HasPose())
        return false;

    if(mbTagAlignmentPending)
    {
        if(!mpLocalMapper->isStoppedForTagAlignment())
            return false;
    }
    else
    {
        if(!mbHasExternalTagObservation)
            return false;

        Map* currentMap = mpAtlas->GetCurrentMap();
        const double currentTimestamp = mCurrentFrame.mTimeStamp;
        if(!std::isfinite(currentTimestamp))
        {
            mvTagScaleSamples.clear();
            mbHasTagScaleReference = false;
            mTagScaleReferenceTimestamp = 0.0;
            mpTagScaleReferenceMap = nullptr;
            return false;
        }
        if(!mvTagScaleSamples.empty())
        {
            const TagScaleSample &last = mvTagScaleSamples.back();
            if(last.map != currentMap || !std::isfinite(last.timestamp) ||
               currentTimestamp < last.timestamp || currentTimestamp - last.timestamp > 0.30)
            {
                mvTagScaleSamples.clear();
                mbHasTagScaleReference = false;
                mTagScaleReferenceTimestamp = 0.0;
                mpTagScaleReferenceMap = nullptr;
            }
        }
        const Sophus::SE3f currentSlamTwc = mCurrentFrame.GetPose().inverse();
        if(mvTagScaleSamples.empty() && mbHasTagScaleReference)
        {
            if(mpTagScaleReferenceMap != currentMap ||
               !std::isfinite(mTagScaleReferenceTimestamp) ||
               currentTimestamp < mTagScaleReferenceTimestamp ||
               currentTimestamp - mTagScaleReferenceTimestamp > 0.30)
            {
                mbHasTagScaleReference = false;
                mTagScaleReferenceTimestamp = 0.0;
                mpTagScaleReferenceMap = nullptr;
            }
            else
            {
                TagScaleSample reference;
                reference.metricTwc = mTagScaleReferenceMetricTwc;
                reference.slamTwc = mTagScaleReferenceSlamTwc;
                reference.timestamp = mTagScaleReferenceTimestamp;
                reference.map = mpTagScaleReferenceMap;
                mvTagScaleSamples.push_back(reference);
            }
        }
        TagScaleSample current;
        current.metricTwc = mExternalTagTwc;
        current.slamTwc = currentSlamTwc;
        current.timestamp = currentTimestamp;
        current.map = currentMap;
        mvTagScaleSamples.push_back(current);
        if(mvTagScaleSamples.size() > 240)
        {
            mvTagScaleSamples.erase(mvTagScaleSamples.begin());
        }
        if(mvTagScaleSamples.size() < 8 || currentMap->KeyFramesInMap()<3)
            return false;

        float metricBaseline = 0.0f;
        float visualBaseline = 0.0f;
        std::vector<float> scaleCandidates;
        for(size_t first = 0; first + 1 < mvTagScaleSamples.size(); ++first)
        {
            for(size_t second = first + 1;
                second < mvTagScaleSamples.size(); ++second)
            {
                const float metricDistance =
                    (mvTagScaleSamples[second].metricTwc.translation() -
                     mvTagScaleSamples[first].metricTwc.translation()).norm();
                metricBaseline = std::max(metricBaseline, metricDistance);
                if(metricDistance < 0.5f * mTagMinimumScaleBaselineM)
                    continue;
                const float slamDistance =
                    (mvTagScaleSamples[second].slamTwc.translation() -
                     mvTagScaleSamples[first].slamTwc.translation()).norm();
                visualBaseline = std::max(visualBaseline, slamDistance);
                if(slamDistance > 1e-5f)
                    scaleCandidates.push_back(metricDistance / slamDistance);
            }
        }
        if(metricBaseline < mTagMinimumScaleBaselineM || scaleCandidates.size() < 6)
            return false;

        // ORB-SLAM normalizes a monocular initialization by scene depth, so
        // visualBaseline / medianDepth is invariant to its arbitrary unit.
        // Without this check, centimetres of marker-PnP jitter divided by a
        // near-zero visual baseline can produce a numerically consistent but
        // physically catastrophic scale (observed as a 351x map expansion).
        KeyFrame* scaleOrigin = currentMap->GetOriginKF();
        const float medianSceneDepth = scaleOrigin && !scaleOrigin->isBad()
            ? scaleOrigin->ComputeSceneMedianDepth(2) : -1.0f;
        const float visualBaselineDepthRatio =
            medianSceneDepth > 1e-6f ? visualBaseline / medianSceneDepth : 0.0f;
        if(!std::isfinite(visualBaselineDepthRatio) ||
           visualBaselineDepthRatio < mTagMinimumVisualBaselineDepthRatio)
        {
            const double evidenceSpan = currentTimestamp -
                mvTagScaleSamples.front().timestamp;
            // At high input rates eight observations may span only a few
            // hundredths of a second.  Marker-PnP jitter can already exceed
            // the metric-baseline threshold then, while the true visual
            // translation needed to observe scale arrives a little later.
            // Keep the bounded 300 ms episode until it has had a fair chance
            // to acquire parallax; only then restart from the newest sample.
            // This preserves the catastrophic-scale guard without making it
            // accidentally frame-rate dependent.
            if(evidenceSpan < 0.25)
                return false;
            cout << "Tag metric alignment waiting for visual parallax: marker baseline "
                 << metricBaseline << " m, visual/depth ratio "
                 << visualBaselineDepthRatio << endl;
            const TagScaleSample newest = mvTagScaleSamples.back();
            mvTagScaleSamples.clear();
            mvTagScaleSamples.push_back(newest);
            mbHasTagScaleReference = false;
            mTagScaleReferenceTimestamp = 0.0;
            mpTagScaleReferenceMap = nullptr;
            return false;
        }

        const auto median = [](std::vector<float> values) {
            const size_t middle = values.size() / 2;
            std::nth_element(values.begin(), values.begin() + middle, values.end());
            float value = values[middle];
            if(values.size() % 2 == 0)
            {
                std::nth_element(
                    values.begin(), values.begin() + middle - 1, values.end());
                value = 0.5f * (value + values[middle - 1]);
            }
            return value;
        };
        const float candidateMedian = median(scaleCandidates);
        std::vector<float> deviations;
        deviations.reserve(scaleCandidates.size());
        for(const float candidate : scaleCandidates)
            deviations.push_back(std::abs(candidate - candidateMedian));
        const float mad = median(deviations);
        std::vector<float> retainedScales;
        retainedScales.reserve(scaleCandidates.size());
        const float tolerance = std::max(1e-4f, 3.5f * 1.4826f * mad);
        for(const float candidate : scaleCandidates)
            if(std::abs(candidate - candidateMedian) <= tolerance)
                retainedScales.push_back(candidate);
        const float scale = median(retainedScales);
        if(!std::isfinite(scale) || scale < 1e-4f || scale > 1000.0f)
            return false;

        Eigen::Quaternionf referenceAlignment(
            mvTagScaleSamples.front().metricTwc.rotationMatrix() *
            mvTagScaleSamples.front().slamTwc.rotationMatrix().transpose());
        Eigen::Vector4f quaternionSum = Eigen::Vector4f::Zero();
        for(size_t index = 0; index < mvTagScaleSamples.size(); ++index)
        {
            Eigen::Quaternionf candidate(
                mvTagScaleSamples[index].metricTwc.rotationMatrix() *
                mvTagScaleSamples[index].slamTwc.rotationMatrix().transpose());
            if(referenceAlignment.dot(candidate) < 0.0f)
                candidate.coeffs() *= -1.0f;
            quaternionSum += candidate.coeffs();
        }
        Eigen::Quaternionf averageAlignment;
        averageAlignment.coeffs() = quaternionSum.normalized();
        const Eigen::Matrix3f alignment =
            averageAlignment.normalized().toRotationMatrix();
        std::vector<float> translations[3];
        for(size_t index = 0; index < mvTagScaleSamples.size(); ++index)
        {
            const Eigen::Vector3f translation =
                mvTagScaleSamples[index].metricTwc.translation() -
                scale * alignment * mvTagScaleSamples[index].slamTwc.translation();
            for(int axis = 0; axis < 3; ++axis)
                translations[axis].push_back(translation(axis));
        }
        const Eigen::Vector3f robustTranslation(
            median(translations[0]), median(translations[1]), median(translations[2]));
        std::vector<float> positionResiduals;
        std::vector<float> rotationResiduals;
        positionResiduals.reserve(mvTagScaleSamples.size());
        rotationResiduals.reserve(mvTagScaleSamples.size());
        for(const TagScaleSample &sample : mvTagScaleSamples)
        {
            const Eigen::Vector3f predictedPosition = robustTranslation +
                scale * alignment * sample.slamTwc.translation();
            positionResiduals.push_back(
                (sample.metricTwc.translation() - predictedPosition).norm());
            const Eigen::Matrix3f rotationDelta = sample.metricTwc.rotationMatrix() *
                (alignment * sample.slamTwc.rotationMatrix()).transpose();
            const float cosine = std::max(-1.0f, std::min(
                1.0f, 0.5f * (rotationDelta.trace() - 1.0f)));
            rotationResiduals.push_back(std::acos(cosine) * 57.2957795f);
        }
        const auto percentile90 = [](std::vector<float> values) {
            std::sort(values.begin(), values.end());
            const size_t index = static_cast<size_t>(
                std::ceil(0.9 * static_cast<double>(values.size() - 1)));
            return values[index];
        };
        const float medianPositionResidual = median(positionResiduals);
        const float medianRotationResidual = median(rotationResiduals);
        const float p90PositionResidual = percentile90(positionResiduals);
        const float p90RotationResidual = percentile90(rotationResiduals);
        if(medianPositionResidual > mTagMaxAlignmentPositionResidualM ||
           p90PositionResidual > 2.0f * mTagMaxAlignmentPositionResidualM ||
           medianRotationResidual > mTagMaxAlignmentRotationResidualDeg ||
           p90RotationResidual > 2.0f * mTagMaxAlignmentRotationResidualDeg)
        {
            cout << "Tag metric alignment rejected: marker/SLAM residual "
                 << medianPositionResidual << " m median, "
                 << p90PositionResidual << " m p90, "
                 << medianRotationResidual << " deg median, "
                 << p90RotationResidual << " deg p90" << endl;
            CancelPendingTagAlignment();
            return false;
        }
        mPendingTagWorldFromSlamWorld =
            Sophus::SE3f(alignment, robustTranslation);
        mPendingTagMetricScale = scale;
        mbTagAlignmentPending = true;
        mpTagAlignmentMap = mpAtlas->GetCurrentMap();
        mpLocalMapper->RequestTagAlignmentStop();
        return false;
    }

    Sophus::SE3f tagWorldFromSlamWorld =
        mPendingTagWorldFromSlamWorld;
    float scale = mPendingTagMetricScale;

    // A short run of per-frame PnP centres is only a proposal, not independent
    // evidence of background scale. Use the existing raw-corner multi-view
    // estimator after LocalMapping has stopped; do not change live geometry.
    auto scaleViews=mpAtlas->GetCurrentMap()->GetAllKeyFrames();
    const auto cornerScale=MarkerGraphOptimizer::EstimateInitialCornerScale(scaleViews);
    if(!cornerScale.valid) {
        cout << "Tag metric alignment waiting for independent corner scale" << endl;
        CancelPendingTagAlignment();
        return false;
    }
    std::vector<std::pair<Eigen::Vector3f,Eigen::Vector3f>> alignmentSamples;
    for(const auto& sample:mvTagScaleSamples)
        alignmentSamples.emplace_back(sample.slamTwc.translation(),sample.metricTwc.translation());
    Eigen::Vector3f refitTranslation;
    if(!RefitInitialMetricTranslation(alignmentSamples,tagWorldFromSlamWorld.rotationMatrix(),
        static_cast<float>(cornerScale.scale),refitTranslation)) {
        CancelPendingTagAlignment();return false;
    }
    tagWorldFromSlamWorld.translation()=refitTranslation;
    cout << "TAG_INITIAL_CORNER_SCALE old=" << scale << " new=" << cornerScale.scale
         << " sigma=" << cornerScale.sigma << " markers=" << cornerScale.markers
         << " rms=" << cornerScale.rms << endl;
    scale=static_cast<float>(cornerScale.scale);

    if(mbHasExternalTagObservation)
    {
        const Sophus::SE3f slamTwc = mCurrentFrame.GetPose().inverse();
        const Eigen::Vector3f predictedPosition = tagWorldFromSlamWorld.translation() +
            scale * tagWorldFromSlamWorld.rotationMatrix() * slamTwc.translation();
        const float positionResidual =
            (mExternalTagTwc.translation() - predictedPosition).norm();
        const Eigen::Matrix3f rotationDelta = mExternalTagTwc.rotationMatrix() *
            (tagWorldFromSlamWorld.rotationMatrix() * slamTwc.rotationMatrix()).transpose();
        const float cosine = std::max(-1.0f, std::min(
            1.0f, 0.5f * (rotationDelta.trace() - 1.0f)));
        const float rotationResidual = std::acos(cosine) * 57.2957795f;
        if(positionResidual > 2.0f * mTagMaxAlignmentPositionResidualM ||
           rotationResidual > 2.0f * mTagMaxAlignmentRotationResidualDeg)
        {
            cout << "Tag metric alignment rejected at commit: current marker/SLAM residual "
                 << positionResidual << " m, " << rotationResidual << " deg" << endl;
            CancelPendingTagAlignment();
            return false;
        }
    }

    Map* pMap = mpAtlas->GetCurrentMap();
    const MarkerGraphOptimizer::Proposal metricProposal =
        MarkerGraphOptimizer::InitializeMetric(pMap, tagWorldFromSlamWorld, scale);
    if(!metricProposal.accepted)
    {
        cout << "Tag metric joint optimization rejected: "
             << metricProposal.reason << ", tag RMS "
             << metricProposal.before.tagRmsPx << " -> "
             << metricProposal.after.tagRmsPx << " px, ORB RMS "
             << metricProposal.before.backgroundRmsPx << " -> "
             << metricProposal.after.backgroundRmsPx << " px, ORB observations "
             << metricProposal.after.backgroundObservations << endl;
        CancelPendingTagAlignment();
        return false;
    }

    std::vector<float> optimizedScales;
    optimizedScales.reserve(metricProposal.replayScaleMultipliers.size());
    for(const auto &item : metricProposal.replayScaleMultipliers)
        if(std::isfinite(item.second) && item.second > 0.0f)
            optimizedScales.push_back(item.second);
    float optimizedScale = scale;
    if(!optimizedScales.empty())
    {
        const size_t middle = optimizedScales.size() / 2;
        std::nth_element(
            optimizedScales.begin(), optimizedScales.begin() + middle,
            optimizedScales.end());
        optimizedScale = optimizedScales[middle];
        if(optimizedScales.size() % 2 == 0)
        {
            const float upper = optimizedScale;
            std::nth_element(
                optimizedScales.begin(), optimizedScales.begin() + middle - 1,
                optimizedScales.end());
            optimizedScale = 0.5f * (upper + optimizedScales[middle - 1]);
        }
    }

    struct StagedFramePose
    {
        Frame* frame = nullptr;
        KeyFrame* reference = nullptr;
        Sophus::SE3f relative;
        bool followsOptimizedReference = false;
    };
    const auto stageFrame = [&metricProposal, pMap](Frame &frame) {
        StagedFramePose staged;
        staged.frame = &frame;
        staged.reference = frame.mpReferenceKF;
        staged.followsOptimizedReference = frame.HasPose() && staged.reference &&
            !staged.reference->isBad() && staged.reference->GetMap() == pMap &&
            metricProposal.keyframePoses.count(staged.reference);
        if(staged.followsOptimizedReference)
            staged.relative = frame.GetPose() * staged.reference->GetPoseInverse();
        return staged;
    };
    const StagedFramePose stagedCurrent = stageFrame(mCurrentFrame);
    const StagedFramePose stagedLast = stageFrame(mLastFrame);

    // The proposal was solved without touching live geometry. LocalMapping is
    // stopped and Track owns the map-update lock, so publish one coherent
    // marker+ORB metric revision only after all validation has passed.
    for(const auto &item : metricProposal.keyframePoses)
    {
        item.first->SetPose(item.second);
        item.first->mReplayUnitScale *=
            metricProposal.replayScaleMultipliers.at(item.first);
    }
    for(const auto &item : metricProposal.tagWorldCorners)
        item.first->mvTagWorldPoints = item.second;
    for(const auto &tag : metricProposal.staticTags)
    {
        auto &values = pMap->mStaticTags[tag.first];
        values.clear();
        values.reserve(12);
        for(const auto &corner : tag.second)
            for(int axis = 0; axis < 3; ++axis)
                values.push_back(corner(axis));
    }
    for(const auto &item : metricProposal.pointPositions)
        item.first->SetWorldPos(item.second);
    for(const auto &item : metricProposal.pointPositions)
        item.first->UpdateNormalAndDepth();
    auto reference = mlpReferences.begin();
    for(Sophus::SE3f &relativePose : mlRelativeFramePoses) {
        if(*reference && (*reference)->GetMap()==pMap) {
            const auto recovered = metricProposal.replayScaleMultipliers.find(*reference);
            relativePose.translation() *= recovered == metricProposal.replayScaleMultipliers.end()
                ? scale : recovered->second;
        }
        ++reference;
    }
    const auto publishFrame = [&metricProposal, &tagWorldFromSlamWorld, scale](
            const StagedFramePose &staged) {
        if(!staged.frame->HasPose())
            return;
        if(staged.followsOptimizedReference)
        {
            Sophus::SE3f relative = staged.relative;
            relative.translation() *=
                metricProposal.replayScaleMultipliers.at(staged.reference);
            staged.frame->SetPose(
                relative * metricProposal.keyframePoses.at(staged.reference));
            return;
        }
        Sophus::SE3f Twc = staged.frame->GetPose().inverse();
        Twc.translation() *= scale;
        staged.frame->SetPose((tagWorldFromSlamWorld * Twc).inverse());
    };
    publishFrame(stagedCurrent);
    publishFrame(stagedLast);
    for(KeyFrame* pKF : pMap->GetAllKeyFrames())
        if(pKF->mbHasTagObservation)
            pKF->mbTagObservationActive = true;

    pMap->mbMetric = true;
    pMap->mMetricScale = optimizedScale;
    pMap->IncreaseChangeIndex();
    mRecoveredTagMetricScale = optimizedScale;
    mbTagMetricAligned = true;
    mbVelocity = false;
    CancelPendingTagAlignment();
    cout << "Tag metric map jointly aligned: initial " << scale
         << ", optimized " << optimizedScale
         << " m/ORB-unit, tag RMS " << metricProposal.before.tagRmsPx
         << " -> " << metricProposal.after.tagRmsPx
         << " px, ORB RMS " << metricProposal.before.backgroundRmsPx
         << " -> " << metricProposal.after.backgroundRmsPx << " px" << endl;
    return true;
}

void Tracking::CaptureMarkerGraphVisualPose()
{
    if(mbHasExternalTagObservation && !mbTagAlignmentPending &&
       mpAtlas->GetCurrentMap()->mbMetric && mpAtlas->GetCurrentMap()->mbBackgroundReady &&
       mCurrentFrame.HasPose()) {
        mMarkerGraphVisualTwc=mCurrentFrame.GetPose().inverse();
        mnMarkerGraphVisualFrameId=long(mCurrentFrame.mnId);
    }
}

void Tracking::ProcessMarkerGraph(bool final)
{
    if(!mbTagFusionEnabled || mSensor!=System::MONOCULAR || (mbOnlyTracking && !final)) return;
    if(!mpMarkerGraphCoordinator)
        mpMarkerGraphCoordinator.reset(new MarkerGraphCoordinator(*this));
    mpMarkerGraphCoordinator->OnFrameEnd(final);
}

void Tracking::CancelMarkerGraph()
{
    if(mpMarkerGraphCoordinator) mpMarkerGraphCoordinator->Cancel();
    mpMarkerGraphCoordinator.reset();
    mnMarkerGraphVisualFrameId=-1;
}

void Tracking::ApplyExternalTagPoseConstraint()
{
    if(!mbTagMetricAligned || (!mbHasExternalTagObservation && !mbHasTrackedTagObservation) ||
       !mCurrentFrame.HasPose())
        return;
    const Sophus::SE3f currentTwc = mCurrentFrame.GetPose().inverse();
    const float positionError =
        (currentTwc.translation() - mExternalTagTwc.translation()).norm();
    const Eigen::Matrix3f rotationError =
        currentTwc.rotationMatrix().transpose() *
        mExternalTagTwc.rotationMatrix();
    const float cosine = std::max(
        -1.0f, std::min(1.0f, 0.5f * (rotationError.trace() - 1.0f)));
    const float angleDeg = std::acos(cosine) * 180.0f / static_cast<float>(M_PI);
    mMarkerTrackingStatus.posePositionResidualM = positionError;
    mMarkerTrackingStatus.poseRotationResidualDeg = angleDeg;

    const std::set<int> strongMarkerIds = StrongMarkerIds(
        mvExternalTagIds, mvExternalTagPointWeights);
    // TrackMarkerSeed starts background recovery from a measured marker pose.
    // Keep one complete, strong tag in that ONE transition solve so the new
    // visual pose cannot introduce a seam between marker_world and Atlas.
    // Ordinary OK tracking still requires two complete tags before an
    // instantaneous correction is possible.
    const bool markerRecoveryAssist =
        mState == MARKER_TRACKING && mbHasExternalTagObservation &&
        !mbHasTrackedTagObservation && strongMarkerIds.size() == 1;
    // Three independently decoded markers contribute at least twelve strong
    // corners. That is enough to recover an isolated visual branch error
    // without granting the same authority to the two-marker subset that
    // caused the previously observed 83 mm jump.
    const bool strongMarkerSupport = strongMarkerIds.size() >= 3;
    // A brief collapse in natural-feature support can put an otherwise
    // localized frame just outside the ordinary 3 cm agreement basin.  When
    // at least three decoded markers independently provide twelve strong,
    // low-residual corners, allow the existing joint pose solve a bounded
    // 5 cm rescue basin.  This is deliberately unavailable to one- or
    // two-marker subsets, so it cannot reintroduce the reproduced 83 mm
    // subset-switch failure.
    const bool strongMarkerVisualRescue = strongMarkerSupport &&
        mnMatchesInliers < 100 && mExternalTagConfidence >= 0.8f &&
        mMarkerTrackingStatus.reprojectionPx <= 1.0f;
    if(strongMarkerIds.size() < 2 && !markerRecoveryAssist)
    {
        mpInstantTagConsistencyMap = nullptr;
        mInstantTagMarkerIds.clear();
        mnInstantTagConsistencyFrameId = -1;
        mnInstantTagConsistencyFrames = 0;
        mMarkerTrackingStatus.poseConstraintReason =
            mbHasTrackedTagObservation ? "partial_marker_observation_only" :
            "single_marker_graph_only";
        return;
    }

    // A decoded marker remains useful evidence for keyframes, interval-scale
    // estimation and joint BA, but one observation must not instantaneously
    // drag an already localized camera away from the committed Atlas pose.
    // Large disagreements are corrected only after multi-view graph checks.
    // Direct marker initialization and marker-only recovery do not pass
    // through this branch, so they remain available when visual tracking is
    // genuinely absent.
    // The very first marker->background joint solve may start a few
    // centimetres away because the visual map has only just recovered. Its
    // same-frame tag has already passed decode, reprojection and marker-only
    // motion gates; allow a bounded 5 cm basin only for that transition.
    // Multiple coplanar markers share calibration and planar-pose failure
    // modes, so marker count alone must never widen this gate.  Larger
    // disagreements remain graph evidence instead of instant pose updates.
    const float maximumPositionResidual =
        (markerRecoveryAssist || strongMarkerVisualRescue)
        ? std::max(mTagMaxAlignmentPositionResidualM, 0.05f)
        : mTagMaxAlignmentPositionResidualM;
    if(positionError > maximumPositionResidual ||
       angleDeg > mTagMaxAlignmentRotationResidualDeg)
    {
        mpInstantTagConsistencyMap = nullptr;
        mInstantTagMarkerIds.clear();
        mnInstantTagConsistencyFrameId = -1;
        mnInstantTagConsistencyFrames = 0;
        mMarkerTrackingStatus.poseConstraintReason = "deferred_to_marker_graph";
        return;
    }

    // A visibility-set change is exactly where independent planar PnP poses
    // tend to shift. Keep the decoded corners for keyframe BA immediately,
    // but require three consecutive, geometrically consistent observations
    // before they may alter an ordinary tracking frame. The exception is a
    // frame with weak visual support: two complete fixed tags are then the
    // stronger measurement and prevent a barely-accepted ORB pose from
    // becoming a one-frame world-coordinate spike.
    Map* currentMap = mpAtlas->GetCurrentMap();
    const Sophus::SE3f correction = mExternalTagTwc * currentTwc.inverse();
    // Three independently decoded board markers provide 12 strong corners
    // and do not need a three-frame delay once they already agree with the
    // committed visual pose inside the normal residual gate.
    const bool immediateTagAssist =
        mnMatchesInliers < 50 || markerRecoveryAssist || strongMarkerSupport;
    if(!immediateTagAssist)
    {
        bool consistent = mpInstantTagConsistencyMap == currentMap &&
            mInstantTagMarkerIds == strongMarkerIds &&
            mnInstantTagConsistencyFrameId + 1 == long(mCurrentFrame.mnId);
        if(consistent)
        {
            const Sophus::SE3f change = correction * mInstantTagCorrection.inverse();
            consistent = change.translation().norm() <= 0.005f &&
                change.so3().log().norm() <= 0.0523599f;
        }
        mnInstantTagConsistencyFrames = consistent
            ? mnInstantTagConsistencyFrames + 1 : 1;
        mpInstantTagConsistencyMap = currentMap;
        mInstantTagMarkerIds = strongMarkerIds;
        mInstantTagCorrection = correction;
        mnInstantTagConsistencyFrameId = long(mCurrentFrame.mnId);
        if(mnInstantTagConsistencyFrames < 3)
        {
            mMarkerTrackingStatus.poseConstraintReason = "marker_set_unconfirmed";
            return;
        }
    }

    // Re-optimize actual ORB + tag pixel observations in one pose solve.
    // Do not average the PnP pose into an already optimized camera estimate.
    std::vector<float> instantWeights = mvExternalTagPointWeights;
    if(instantWeights.empty())
        instantWeights.assign(mvExternalTagWorldPoints.size(), 1.0f);
    const float instantScale = std::max(0.0f, std::min(1.0f, mTagPoseWeight));
    for(float& weight : instantWeights) weight *= instantScale;
    // Start from the committed ORB pose. Marker corners refine it jointly;
    // they do not replace the seed with a distant PnP basin.
    const Sophus::SE3f visualPose=mCurrentFrame.GetPose();
    const auto visualOutliers=mCurrentFrame.mvbOutlier;
    std::vector<std::size_t> visualSupport;
    if(mState==OK && !mCurrentFrame.mpCamera2) {
        for(std::size_t i=0;i<mCurrentFrame.mvpMapPoints.size();++i) {
            MapPoint* point=mCurrentFrame.mvpMapPoints[i];
            if(point && !point->isBad() && point->Observations()>0 &&
               i<visualOutliers.size() && !visualOutliers[i] &&
               i<mCurrentFrame.mvKeysUn.size() && mCurrentFrame.mvuRight[i]<0)
                visualSupport.push_back(i);
        }
    }
    // Compare the SAME established background observations before and after.
    // The joint solver may reclassify them as outliers; evaluating only its
    // surviving inliers would hide a destructive marker/layout correction.
    const auto backgroundRms=[&]() {
        double squared=0;
        for(std::size_t i:visualSupport) {
            const Eigen::Vector3f p=mCurrentFrame.GetPose()*mCurrentFrame.mvpMapPoints[i]->GetWorldPos();
            if(!p.allFinite() || p.z()<=0) return std::numeric_limits<double>::infinity();
            const Eigen::Vector2f uv=mCurrentFrame.mpCamera->project(p);
            const auto& key=mCurrentFrame.mvKeysUn[i];
            const Eigen::Vector2f error=uv-Eigen::Vector2f(key.pt.x,key.pt.y);
            squared+=error.squaredNorm()*mCurrentFrame.mvInvLevelSigma2[key.octave];
        }
        return visualSupport.empty()?0.:std::sqrt(squared/visualSupport.size());
    };
    const double beforeBackground=backgroundRms();
    Optimizer::PoseOptimization(&mCurrentFrame, mvExternalTagWorldPoints,
                                mvExternalTagImagePoints, instantWeights);
    const double afterBackground=backgroundRms();
    if(visualSupport.size()>=15 && beforeBackground<=3.0 &&
       (!std::isfinite(afterBackground) || afterBackground>3.0 ||
        afterBackground>beforeBackground+0.5)) {
        mCurrentFrame.SetPose(visualPose);
        mCurrentFrame.mvbOutlier=visualOutliers;
        mMarkerTrackingStatus.poseConstraintApplied=false;
        mMarkerTrackingStatus.poseConstraintReason="deferred_to_marker_graph_background_conflict";
        std::cout << "TAG_POSE_BACKGROUND_GATE timestamp=" << mCurrentFrame.mTimeStamp
                  << " points=" << visualSupport.size() << " rms_before=" << beforeBackground
                  << " rms_after=" << afterBackground << std::endl;
        return;
    }
    mnTagPoseConstraintsApplied++;
    mnTagPoseConstraintFrameId = long(mCurrentFrame.mnId);
    mMarkerTrackingStatus.poseConstraintApplied = true;
    mMarkerTrackingStatus.poseConstraintReason = markerRecoveryAssist
        ? "fused_marker_recovery" :
        (strongMarkerVisualRescue ? "fused_strong_marker_visual_rescue" :
         (strongMarkerSupport ? "fused_three_marker_support" :
          (mnMatchesInliers < 50 ? "fused_low_visual_support" : "fused")));
}



Sophus::SE3f Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp, string filename)
{
    //cout << "GrabImageStereo" << endl;

    mImGray = imRectLeft;
    cv::Mat imGrayRight = imRectRight;
    mImRight = imRectRight;

    if(mImGray.channels()==3)
    {
        //cout << "Image with 3 channels" << endl;
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,cv::COLOR_RGB2GRAY);
            cvtColor(imGrayRight,imGrayRight,cv::COLOR_RGB2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,cv::COLOR_BGR2GRAY);
            cvtColor(imGrayRight,imGrayRight,cv::COLOR_BGR2GRAY);
        }
    }
    else if(mImGray.channels()==4)
    {
        //cout << "Image with 4 channels" << endl;
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,cv::COLOR_RGBA2GRAY);
            cvtColor(imGrayRight,imGrayRight,cv::COLOR_RGBA2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,cv::COLOR_BGRA2GRAY);
            cvtColor(imGrayRight,imGrayRight,cv::COLOR_BGRA2GRAY);
        }
    }

    //cout << "Incoming frame creation" << endl;

    if (mSensor == System::STEREO && !mpCamera2)
        mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera);
    else if(mSensor == System::STEREO && mpCamera2)
        mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera,mpCamera2,mTlr);
    else if(mSensor == System::IMU_STEREO && !mpCamera2)
        mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera,&mLastFrame,*mpImuCalib);
    else if(mSensor == System::IMU_STEREO && mpCamera2)
        mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera,mpCamera2,mTlr,&mLastFrame,*mpImuCalib);

    //cout << "Incoming frame ended" << endl;

    mCurrentFrame.mNameFile = filename;
    mCurrentFrame.mnDataset = mnNumDataset;

#ifdef REGISTER_TIMES
    vdORBExtract_ms.push_back(mCurrentFrame.mTimeORB_Ext);
    vdStereoMatch_ms.push_back(mCurrentFrame.mTimeStereoMatch);
#endif

    //cout << "Tracking start" << endl;
    Track();
    //cout << "Tracking end" << endl;

    return mCurrentFrame.GetPose();
}


Sophus::SE3f Tracking::GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp, string filename)
{
    mImGray = imRGB;
    cv::Mat imDepth = imD;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,cv::COLOR_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,cv::COLOR_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,cv::COLOR_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,cv::COLOR_BGRA2GRAY);
    }

    if((fabs(mDepthMapFactor-1.0f)>1e-5) || imDepth.type()!=CV_32F)
        imDepth.convertTo(imDepth,CV_32F,mDepthMapFactor);

    if (mSensor == System::RGBD)
        mCurrentFrame = Frame(mImGray,imDepth,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera);
    else if(mSensor == System::IMU_RGBD)
        mCurrentFrame = Frame(mImGray,imDepth,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera,&mLastFrame,*mpImuCalib);






    mCurrentFrame.mNameFile = filename;
    mCurrentFrame.mnDataset = mnNumDataset;

#ifdef REGISTER_TIMES
    vdORBExtract_ms.push_back(mCurrentFrame.mTimeORB_Ext);
#endif

    Track();

    return mCurrentFrame.GetPose();
}


Sophus::SE3f Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp, string filename)
{
    mImGray = im;
    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,cv::COLOR_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,cv::COLOR_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,cv::COLOR_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,cv::COLOR_BGRA2GRAY);
    }

    // Only ordinary monocular initialization uses legacy feature selection.
    // Marker-first initialization and subsequent tracking keep stable selection.
    const char* legacyInit = std::getenv("ORB_SLAM3_LEGACY_MONO_INITIALIZATION");
    ScopedLegacyInitializationSelection selectionScope(
        legacyInit && std::string(legacyInit)=="1" &&
        mSensor==System::MONOCULAR && !mbMarkerOnlyInitialization &&
        !mbHasExternalTagObservation &&
        (mState==NOT_INITIALIZED || mState==NO_IMAGES_YET));

    if (mSensor == System::MONOCULAR)
    {
        if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET ||
           (lastID - initID) < mMaxFrames)
            mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth);
        else
            mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth);
    }
    else if(mSensor == System::IMU_MONOCULAR)
    {
        if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET)
        {
            mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth,&mLastFrame,*mpImuCalib);
        }
        else
            mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth,&mLastFrame,*mpImuCalib);
    }

    if (mState==NO_IMAGES_YET)
        t0=timestamp;

    mCurrentFrame.mNameFile = filename;
    mCurrentFrame.mnDataset = mnNumDataset;

#ifdef REGISTER_TIMES
    vdORBExtract_ms.push_back(mCurrentFrame.mTimeORB_Ext);
#endif

    lastID = mCurrentFrame.mnId;
    Track();

    if(mbTemporalFlowEnabled || mbFlowRecoveryEnabled) {
        mTemporalPreviousImage=mImGray.clone();
        mTemporalPreviousTime=timestamp;
    }

    return mCurrentFrame.GetPose();
}


void Tracking::GrabImuData(const IMU::Point &imuMeasurement)
{
    unique_lock<mutex> lock(mMutexImuQueue);
    mlQueueImuData.push_back(imuMeasurement);
}

void Tracking::PreintegrateIMU()
{

    if(!mCurrentFrame.mpPrevFrame)
    {
        Verbose::PrintMess("non prev frame ", Verbose::VERBOSITY_NORMAL);
        mCurrentFrame.setIntegrated();
        return;
    }

    mvImuFromLastFrame.clear();
    mvImuFromLastFrame.reserve(mlQueueImuData.size());
    if(mlQueueImuData.size() == 0)
    {
        Verbose::PrintMess("Not IMU data in mlQueueImuData!!", Verbose::VERBOSITY_NORMAL);
        mCurrentFrame.setIntegrated();
        return;
    }

    while(true)
    {
        bool bSleep = false;
        {
            unique_lock<mutex> lock(mMutexImuQueue);
            if(!mlQueueImuData.empty())
            {
                IMU::Point* m = &mlQueueImuData.front();
                cout.precision(17);
                if(m->t<mCurrentFrame.mpPrevFrame->mTimeStamp-mImuPer)
                {
                    mlQueueImuData.pop_front();
                }
                else if(m->t<mCurrentFrame.mTimeStamp-mImuPer)
                {
                    mvImuFromLastFrame.push_back(*m);
                    mlQueueImuData.pop_front();
                }
                else
                {
                    mvImuFromLastFrame.push_back(*m);
                    break;
                }
            }
            else
            {
                break;
                bSleep = true;
            }
        }
        if(bSleep)
            usleep(500);
    }

    const int n = mvImuFromLastFrame.size()-1;
    if(n==0){
        cout << "Empty IMU measurements vector!!!\n";
        return;
    }

    IMU::Preintegrated* pImuPreintegratedFromLastFrame = new IMU::Preintegrated(mLastFrame.mImuBias,mCurrentFrame.mImuCalib);

    for(int i=0; i<n; i++)
    {
        float tstep;
        Eigen::Vector3f acc, angVel;
        if((i==0) && (i<(n-1)))
        {
            float tab = mvImuFromLastFrame[i+1].t-mvImuFromLastFrame[i].t;
            float tini = mvImuFromLastFrame[i].t-mCurrentFrame.mpPrevFrame->mTimeStamp;
            acc = (mvImuFromLastFrame[i].a+mvImuFromLastFrame[i+1].a-
                    (mvImuFromLastFrame[i+1].a-mvImuFromLastFrame[i].a)*(tini/tab))*0.5f;
            angVel = (mvImuFromLastFrame[i].w+mvImuFromLastFrame[i+1].w-
                    (mvImuFromLastFrame[i+1].w-mvImuFromLastFrame[i].w)*(tini/tab))*0.5f;
            tstep = mvImuFromLastFrame[i+1].t-mCurrentFrame.mpPrevFrame->mTimeStamp;
        }
        else if(i<(n-1))
        {
            acc = (mvImuFromLastFrame[i].a+mvImuFromLastFrame[i+1].a)*0.5f;
            angVel = (mvImuFromLastFrame[i].w+mvImuFromLastFrame[i+1].w)*0.5f;
            tstep = mvImuFromLastFrame[i+1].t-mvImuFromLastFrame[i].t;
        }
        else if((i>0) && (i==(n-1)))
        {
            float tab = mvImuFromLastFrame[i+1].t-mvImuFromLastFrame[i].t;
            float tend = mvImuFromLastFrame[i+1].t-mCurrentFrame.mTimeStamp;
            acc = (mvImuFromLastFrame[i].a+mvImuFromLastFrame[i+1].a-
                    (mvImuFromLastFrame[i+1].a-mvImuFromLastFrame[i].a)*(tend/tab))*0.5f;
            angVel = (mvImuFromLastFrame[i].w+mvImuFromLastFrame[i+1].w-
                    (mvImuFromLastFrame[i+1].w-mvImuFromLastFrame[i].w)*(tend/tab))*0.5f;
            tstep = mCurrentFrame.mTimeStamp-mvImuFromLastFrame[i].t;
        }
        else if((i==0) && (i==(n-1)))
        {
            acc = mvImuFromLastFrame[i].a;
            angVel = mvImuFromLastFrame[i].w;
            tstep = mCurrentFrame.mTimeStamp-mCurrentFrame.mpPrevFrame->mTimeStamp;
        }

        if (!mpImuPreintegratedFromLastKF)
            cout << "mpImuPreintegratedFromLastKF does not exist" << endl;
        mpImuPreintegratedFromLastKF->IntegrateNewMeasurement(acc,angVel,tstep);
        pImuPreintegratedFromLastFrame->IntegrateNewMeasurement(acc,angVel,tstep);
    }

    mCurrentFrame.mpImuPreintegratedFrame = pImuPreintegratedFromLastFrame;
    mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;
    mCurrentFrame.mpLastKeyFrame = mpLastKeyFrame;

    mCurrentFrame.setIntegrated();

    //Verbose::PrintMess("Preintegration is finished!! ", Verbose::VERBOSITY_DEBUG);
}


bool Tracking::PredictStateIMU()
{
    if(!mCurrentFrame.mpPrevFrame)
    {
        Verbose::PrintMess("No last frame", Verbose::VERBOSITY_NORMAL);
        return false;
    }

    if(mbMapUpdated && mpLastKeyFrame)
    {
        const Eigen::Vector3f twb1 = mpLastKeyFrame->GetImuPosition();
        const Eigen::Matrix3f Rwb1 = mpLastKeyFrame->GetImuRotation();
        const Eigen::Vector3f Vwb1 = mpLastKeyFrame->GetVelocity();

        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);
        const float t12 = mpImuPreintegratedFromLastKF->dT;

        Eigen::Matrix3f Rwb2 = IMU::NormalizeRotation(Rwb1 * mpImuPreintegratedFromLastKF->GetDeltaRotation(mpLastKeyFrame->GetImuBias()));
        Eigen::Vector3f twb2 = twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*mpImuPreintegratedFromLastKF->GetDeltaPosition(mpLastKeyFrame->GetImuBias());
        Eigen::Vector3f Vwb2 = Vwb1 + t12*Gz + Rwb1 * mpImuPreintegratedFromLastKF->GetDeltaVelocity(mpLastKeyFrame->GetImuBias());
        mCurrentFrame.SetImuPoseVelocity(Rwb2,twb2,Vwb2);

        mCurrentFrame.mImuBias = mpLastKeyFrame->GetImuBias();
        mCurrentFrame.mPredBias = mCurrentFrame.mImuBias;
        return true;
    }
    else if(!mbMapUpdated)
    {
        const Eigen::Vector3f twb1 = mLastFrame.GetImuPosition();
        const Eigen::Matrix3f Rwb1 = mLastFrame.GetImuRotation();
        const Eigen::Vector3f Vwb1 = mLastFrame.GetVelocity();
        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);
        const float t12 = mCurrentFrame.mpImuPreintegratedFrame->dT;

        Eigen::Matrix3f Rwb2 = IMU::NormalizeRotation(Rwb1 * mCurrentFrame.mpImuPreintegratedFrame->GetDeltaRotation(mLastFrame.mImuBias));
        Eigen::Vector3f twb2 = twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1 * mCurrentFrame.mpImuPreintegratedFrame->GetDeltaPosition(mLastFrame.mImuBias);
        Eigen::Vector3f Vwb2 = Vwb1 + t12*Gz + Rwb1 * mCurrentFrame.mpImuPreintegratedFrame->GetDeltaVelocity(mLastFrame.mImuBias);

        mCurrentFrame.SetImuPoseVelocity(Rwb2,twb2,Vwb2);

        mCurrentFrame.mImuBias = mLastFrame.mImuBias;
        mCurrentFrame.mPredBias = mCurrentFrame.mImuBias;
        return true;
    }
    else
        cout << "not IMU prediction!!" << endl;

    return false;
}

void Tracking::ResetFrameIMU()
{
    // TODO To implement...
}


void Tracking::Track()
{
    mMarkerBootstrapStatus = MarkerBootstrapStatus();
    mnMatchesInliers = 0;
    // Marker map selection below may change NO_IMAGES_YET to MARKER_TRACKING.
    // Capture whether an actual previous frame exists before that transition.
    const bool hasPreviousFrame = mState!=NO_IMAGES_YET &&
        std::isfinite(mLastFrame.mTimeStamp);

    if (bStepByStep)
    {
        std::cout << "Tracking: Waiting to the next step" << std::endl;
        while(!mbStep && bStepByStep)
            usleep(500);
        mbStep = false;
    }

    if(mpLocalMapper->mbBadImu)
    {
        cout << "TRACK: Reset map because local mapper set the bad imu flag " << endl;
        mpSystem->ResetActiveMap();
        return;
    }

    Map* pCurrentMap = mpAtlas->GetCurrentMap();
    // A known static ID in the same configured gauge can immediately select
    // a loaded map; this is marker relocalization, not a fabricated map merge.
    if(pCurrentMap->KeyFramesInMap()==0 && mbHasExternalTagObservation && !mvExternalTagIds.empty()) {
        for(Map* candidate:mpAtlas->GetAllMaps()) {
            if(candidate==pCurrentMap || !candidate->mbMetric || candidate->IsBad()) continue;
            bool matched=false, consistent=true;
            for(size_t i=0;i+3<mvExternalTagIds.size();i+=4) {
                auto tag=candidate->mStaticTags.find(mvExternalTagIds[i]);
                if(tag==candidate->mStaticTags.end()) continue;
                matched=true;
                for(size_t j=0;j<4;++j)
                    for(size_t axis=0;axis<3;++axis)
                        if(std::abs(tag->second[j*3+axis]-mvExternalTagWorldPoints[i+j](axis))>1e-4f)
                            consistent=false;
            }
            if(!matched || !consistent) continue;
            Map* empty=pCurrentMap;
            mpAtlas->ChangeMap(candidate);
            mpAtlas->SetMapBad(empty);
            pCurrentMap=candidate;
            mpReferenceKF=candidate->GetOriginKF();
            mpLastKeyFrame=mpReferenceKF;
            mvpLocalKeyFrames=candidate->GetAllKeyFrames();
            mvpLocalMapPoints=candidate->GetAllMapPoints();
            mState=MARKER_TRACKING;
            cout << "MARKER_RELOCALIZATION: loaded map " << candidate->GetId() << endl;
            break;
        }
    }
    if(!pCurrentMap)
    {
        cout << "ERROR: There is not an active map in the atlas" << endl;
    }

    if(hasPreviousFrame)
    {
        if(mLastFrame.mTimeStamp>mCurrentFrame.mTimeStamp)
        {
            cerr << "ERROR: Frame with a timestamp older than previous frame detected!" << endl;
            unique_lock<mutex> lock(mMutexImuQueue);
            mlQueueImuData.clear();
            CreateMapInAtlas();
            return;
        }
        else if(mCurrentFrame.mTimeStamp>mLastFrame.mTimeStamp+1.0)
        {
            // cout << mCurrentFrame.mTimeStamp << ", " << mLastFrame.mTimeStamp << endl;
            // cout << "id last: " << mLastFrame.mnId << "    id curr: " << mCurrentFrame.mnId << endl;
            if(mpAtlas->isInertial())
            {

                if(mpAtlas->isImuInitialized())
                {
                    cout << "Timestamp jump detected. State set to LOST. Reseting IMU integration..." << endl;
                    if(!pCurrentMap->GetIniertialBA2())
                    {
                        mpSystem->ResetActiveMap();
                    }
                    else
                    {
                        CreateMapInAtlas();
                    }
                }
                else
                {
                    cout << "Timestamp jump detected, before IMU initialization. Reseting..." << endl;
                    mpSystem->ResetActiveMap();
                }
                return;
            }

        }
    }


    if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && mpLastKeyFrame)
        mCurrentFrame.SetNewBias(mpLastKeyFrame->GetImuBias());

    if(mState==NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED;
    }

    mLastProcessedState=mState;

    if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && !mbCreatedMap)
    {
#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_StartPreIMU = std::chrono::steady_clock::now();
#endif
        PreintegrateIMU();
#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_EndPreIMU = std::chrono::steady_clock::now();

        double timePreImu = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndPreIMU - time_StartPreIMU).count();
        vdIMUInteg_ms.push_back(timePreImu);
#endif

    }
    mbCreatedMap = false;

    // Get Map Mutex -> Map cannot be changed
    unique_lock<mutex> lock(pCurrentMap->mMutexMapUpdate);

    // Normal point/KF creation changes mnRevision too; it is not a gauge
    // change. Only graph commits, scale changes and map identity invalidate
    // this short-lived image cache. Point IDs are resolved against live map
    // membership before use, so culled/replaced points cannot be resurrected.
    if(mpReliableFlowMap &&
       (mpReliableFlowMap!=pCurrentMap || mnReliableFlowMapId!=pCurrentMap->GetId() ||
        mnReliableFlowBigChange!=pCurrentMap->GetLastBigChangeIdx() ||
        mnReliableFlowGraphSequence!=pCurrentMap->mnMarkerGraphSequence ||
        mbReliableFlowMetric!=pCurrentMap->mbMetric ||
        mReliableFlowScale!=pCurrentMap->mMetricScale ||
        !std::isfinite(mCurrentFrame.mTimeStamp) ||
        !std::isfinite(mReliableFlowFrame.mTimeStamp) ||
        mCurrentFrame.mTimeStamp-mReliableFlowFrame.mTimeStamp>.25 ||
        mCurrentFrame.mTimeStamp<=mReliableFlowFrame.mTimeStamp ||
        mSensor!=System::MONOCULAR))
        ClearReliableFlowFrame();

    mbTagMetricAligned = pCurrentMap->mbMetric;
    mRecoveredTagMetricScale = pCurrentMap->mMetricScale;
    if(mbTagAlignmentPending &&
       (mpTagAlignmentMap != pCurrentMap || mbTagMetricAligned))
        CancelPendingTagAlignment();
    if(mbHasExternalTagObservation && mbRigidMarkerLayout)
        pCurrentMap->mbRigidMarkerLayout = true;
    if(mbHasExternalTagObservation && pCurrentMap->mbMetric) {
        for(size_t i=0; i+3<mvExternalTagWorldPoints.size() && i<mvExternalTagIds.size(); i+=4) {
            const int id=mvExternalTagIds[i];
            if(i+3>=mvExternalTagIds.size() || mvExternalTagIds[i+1]!=id ||
               mvExternalTagIds[i+2]!=id || mvExternalTagIds[i+3]!=id) continue;
            if(pCurrentMap->mStaticTags.count(id)) continue;
            std::vector<float> corners;
            for(size_t j=i;j<i+4;++j)
                for(int axis=0;axis<3;++axis) corners.push_back(mvExternalTagWorldPoints[j](axis));
            pCurrentMap->mStaticTags[id]=corners;
        }
    }
    UpdateMarkerKeyFrameEvents();
    if(mSensor == System::MONOCULAR && TrackMarkerSeed())
        return;

    mbMapUpdated = false;

    int nCurMapChangeIndex = pCurrentMap->GetMapChangeIndex();
    int nMapChangeIndex = pCurrentMap->GetLastMapChange();
    if(nCurMapChangeIndex>nMapChangeIndex)
    {
        pCurrentMap->SetLastMapChange(nCurMapChangeIndex);
        mbMapUpdated = true;
    }


    if(mState==NOT_INITIALIZED)
    {
        if(mSensor==System::STEREO || mSensor==System::RGBD || mSensor==System::IMU_STEREO || mSensor==System::IMU_RGBD)
        {
            StereoInitialization();
        }
        else
        {
            MonocularInitialization();
        }

        //mpFrameDrawer->Update(this);

        if(mState!=OK) // If rightly initialized, mState=OK
        {
            mLastFrame = Frame(mCurrentFrame);
            return;
        }

        if(mpAtlas->GetAllMaps().size() == 1)
        {
            mnFirstFrameId = mCurrentFrame.mnId;
        }
    }
    else
    {
        // System is initialized. Track Frame.
        bool bOK;

#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_StartPosePred = std::chrono::steady_clock::now();
#endif

        // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
        if(!mbOnlyTracking)
        {

            // State OK
            // Local Mapping is activated. This is the normal behaviour, unless
            // you explicitly activate the "only tracking" mode.
            if(mState==OK)
            {

                // Local Mapping might have changed some MapPoints tracked in last frame
                CheckReplacedInLastFrame();

                if((!mbVelocity && !pCurrentMap->isImuInitialized()) || mCurrentFrame.mnId<mnLastRelocFrameId+2)
                {
                    Verbose::PrintMess("TRACK: Track with respect to the reference KF ", Verbose::VERBOSITY_DEBUG);
                    bOK = TrackReferenceKeyFrame();
                }
                else
                {
                    Verbose::PrintMess("TRACK: Track with motion model", Verbose::VERBOSITY_DEBUG);
                    bOK = TrackWithMotionModel();
                    if(!bOK)
                        bOK = TrackReferenceKeyFrame();
                }


                if (!bOK)
                {
                    if ( mCurrentFrame.mnId<=(mnLastRelocFrameId+mnFramesToResetIMU) &&
                         (mSensor==System::IMU_MONOCULAR || mSensor==System::IMU_STEREO || mSensor == System::IMU_RGBD))
                    {
                        mState = LOST;
                    }
                    else if(pCurrentMap->KeyFramesInMap()>10 ||
                            (mSensor==System::MONOCULAR && pCurrentMap->KeyFramesInMap()>0))
                    {
                        // cout << "KF in map: " << pCurrentMap->KeyFramesInMap() << endl;
                        mState = RECENTLY_LOST;
                        mTimeStampLost = mCurrentFrame.mTimeStamp;
                    }
                    else
                    {
                        mState = LOST;
                    }
                }
            }
            else
            {

                if (mState == RECENTLY_LOST)
                {
                    Verbose::PrintMess("Lost for a short time", Verbose::VERBOSITY_NORMAL);

                    bOK = true;
                    if((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD))
                    {
                        if(pCurrentMap->isImuInitialized())
                            PredictStateIMU();
                        else
                            bOK = false;

                        if (mCurrentFrame.mTimeStamp-mTimeStampLost>time_recently_lost)
                        {
                            mState = LOST;
                            Verbose::PrintMess("Track Lost...", Verbose::VERBOSITY_NORMAL);
                            bOK=false;
                        }
                    }
                    else
                    {
                        // A one-frame dip below the local-map inlier threshold
                        // still has a useful last pose and frame-to-frame ORB
                        // associations.  Preserve that short-baseline evidence
                        // before falling back to global BoW relocalization.
                        // This is especially important for high-rate head
                        // cameras, where adjacent frames remain highly
                        // matchable while the old local map leaves the FOV.
                        bOK = mbVelocity && TrackWithMotionModel();
                        if(!bOK)
                            bOK = TrackReferenceKeyFrame();
                        if(!bOK)
                            bOK = Relocalization();
                        //std::cout << "mCurrentFrame.mTimeStamp:" << to_string(mCurrentFrame.mTimeStamp) << std::endl;
                        //std::cout << "mTimeStampLost:" << to_string(mTimeStampLost) << std::endl;
                        if(mCurrentFrame.mTimeStamp-mTimeStampLost>3.0f && !bOK)
                        {
                            mState = LOST;
                            Verbose::PrintMess("Track Lost...", Verbose::VERBOSITY_NORMAL);
                            bOK=false;
                        }
                    }
                }
                else if (mState == LOST)
                {

                    Verbose::PrintMess("A new map is started...", Verbose::VERBOSITY_NORMAL);

                    if (pCurrentMap->KeyFramesInMap()==0)
                    {
                        mpSystem->ResetActiveMap();
                        Verbose::PrintMess("Reseting current map...", Verbose::VERBOSITY_NORMAL);
                    }else
                        CreateMapInAtlas();

                    if(mpLastKeyFrame)
                        mpLastKeyFrame = static_cast<KeyFrame*>(NULL);

                    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

                    return;
                }
            }

        }
        else
        {
            // Localization Mode: Local Mapping is deactivated (TODO Not available in inertial mode)
            if(mState==LOST)
            {
                if(mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
                    Verbose::PrintMess("IMU. State LOST", Verbose::VERBOSITY_NORMAL);
                bOK = Relocalization();
            }
            else
            {
                if(!mbVO)
                {
                    // In last frame we tracked enough MapPoints in the map
                    if(mbVelocity)
                    {
                        bOK = TrackWithMotionModel();
                    }
                    else
                    {
                        bOK = TrackReferenceKeyFrame();
                    }
                }
                else
                {
                    // In last frame we tracked mainly "visual odometry" points.

                    // We compute two camera poses, one from motion model and one doing relocalization.
                    // If relocalization is sucessfull we choose that solution, otherwise we retain
                    // the "visual odometry" solution.

                    bool bOKMM = false;
                    bool bOKReloc = false;
                    vector<MapPoint*> vpMPsMM;
                    vector<bool> vbOutMM;
                    Sophus::SE3f TcwMM;
                    if(mbVelocity)
                    {
                        bOKMM = TrackWithMotionModel();
                        vpMPsMM = mCurrentFrame.mvpMapPoints;
                        vbOutMM = mCurrentFrame.mvbOutlier;
                        TcwMM = mCurrentFrame.GetPose();
                    }
                    bOKReloc = Relocalization();

                    if(bOKMM && !bOKReloc)
                    {
                        mCurrentFrame.SetPose(TcwMM);
                        mCurrentFrame.mvpMapPoints = vpMPsMM;
                        mCurrentFrame.mvbOutlier = vbOutMM;

                        if(mbVO)
                        {
                            for(int i =0; i<mCurrentFrame.N; i++)
                            {
                                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                                {
                                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                                }
                            }
                        }
                    }
                    else if(bOKReloc)
                    {
                        mbVO = false;
                    }

                    bOK = bOKReloc || bOKMM;
                }
            }
        }

#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_EndPosePred = std::chrono::steady_clock::now();

        double timePosePred = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndPosePred - time_StartPosePred).count();
        vdPosePred_ms.push_back(timePosePred);
#endif


#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_StartLMTrack = std::chrono::steady_clock::now();
#endif
        // If we have an initial estimation of the camera pose and matching. Track the local map.
        if(!mbOnlyTracking)
        {
            if(bOK)
            {
                bOK = TrackLocalMap();

            }
            if(mbFlowRecoveryEnabled &&
               (mCurrentFrame.isSet() ||
                (!bOK && mLastFrame.HasPose() && !mlRelativeFramePoses.empty())) &&
               (!bOK || mnMatchesInliers<40)) {
                if(TryTemporalFlowRecovery(mnMatchesInliers)) bOK=true;
            }
            if(!bOK)
                cout << "Fail to track local map!" << endl;
        }
        else
        {
            // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
            // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
            // the camera we will use the local map again.
            if(bOK && !mbVO)
                bOK = TrackLocalMap();
        }

        if(!bOK)
            CancelPendingTagAlignment();
        if(!bOK && (mbHasExternalTagObservation || mbHasTrackedTagObservation) && pCurrentMap->mbMetric) {
            mCurrentFrame.SetPose(mExternalTagTwc.inverse());
            mState = MARKER_TRACKING;
            InsertMarkerOnlyEventKeyFrame();
            StoreMarkerFrame();
            return;
        }
        if(bOK)
            mState = OK;
        else if (mState == OK)
        {
            if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
            {
                Verbose::PrintMess("Track lost for less than one second...", Verbose::VERBOSITY_NORMAL);
                if(!pCurrentMap->isImuInitialized() || !pCurrentMap->GetIniertialBA2())
                {
                    cout << "IMU is not or recently initialized. Reseting active map..." << endl;
                    mpSystem->ResetActiveMap();
                }

                mState=RECENTLY_LOST;
            }
            else
                mState=RECENTLY_LOST; // visual to lost

            /*if(mCurrentFrame.mnId>mnLastRelocFrameId+mMaxFrames)
            {*/
                mTimeStampLost = mCurrentFrame.mTimeStamp;
            //}
        }
        // Save frame if recent relocalization, since they are used for IMU reset (as we are making copy, it shluld be once mCurrFrame is completely modified)
        if((mCurrentFrame.mnId<(mnLastRelocFrameId+mnFramesToResetIMU)) && (mCurrentFrame.mnId > mnFramesToResetIMU) &&
           (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && pCurrentMap->isImuInitialized())
        {
            // TODO check this situation
            Verbose::PrintMess("Saving pointer to frame. imu needs reset...", Verbose::VERBOSITY_NORMAL);
            Frame* pF = new Frame(mCurrentFrame);
            pF->mpPrevFrame = new Frame(mLastFrame);

            // Load preintegration
            pF->mpImuPreintegratedFrame = new IMU::Preintegrated(mCurrentFrame.mpImuPreintegratedFrame);
        }

        if(pCurrentMap->isImuInitialized())
        {
            if(bOK)
            {
                if(mCurrentFrame.mnId==(mnLastRelocFrameId+mnFramesToResetIMU))
                {
                    cout << "RESETING FRAME!!!" << endl;
                    ResetFrameIMU();
                }
                else if(mCurrentFrame.mnId>(mnLastRelocFrameId+30))
                    mLastBias = mCurrentFrame.mImuBias;
            }
        }

#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_EndLMTrack = std::chrono::steady_clock::now();

        double timeLMTrack = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndLMTrack - time_StartLMTrack).count();
        vdLMTrack_ms.push_back(timeLMTrack);
#endif

        if(bOK && mbTagFusionEnabled)
        {
            CaptureMarkerGraphVisualPose();
            TryAlignMapToTagWorld();
            ApplyExternalTagPoseConstraint();
        }

        // Update drawer
        mpFrameDrawer->Update(this);
        if(mCurrentFrame.isSet())
            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.GetPose());

        if(bOK || mState==RECENTLY_LOST)
        {
            // Update motion model
            if(mLastFrame.isSet() && mCurrentFrame.isSet())
            {
                Sophus::SE3f LastTwc = mLastFrame.GetPose().inverse();
                mVelocity = mCurrentFrame.GetPose() * LastTwc;
                mbVelocity = true;
            }
            else {
                mbVelocity = false;
            }

            if(mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
                mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.GetPose());

            // Clean VO matches
            for(int i=0; i<mCurrentFrame.N; i++)
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(pMP)
                    if(pMP->Observations()<1)
                    {
                        mCurrentFrame.mvbOutlier[i] = false;
                        mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                    }
            }

            // Delete temporal MapPoints
            for(list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend =  mlpTemporalPoints.end(); lit!=lend; lit++)
            {
                MapPoint* pMP = *lit;
                delete pMP;
            }
            mlpTemporalPoints.clear();

#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_StartNewKF = std::chrono::steady_clock::now();
#endif
            bool bNeedKF = NeedNewKeyFrame();

            // Check if we need to insert a new keyframe
            // if(bNeedKF && bOK)
            if(bNeedKF && (bOK || (mInsertKFsLost && mState==RECENTLY_LOST &&
                                   (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD))))
                CreateNewKeyFrame();

#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_EndNewKF = std::chrono::steady_clock::now();

            double timeNewKF = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndNewKF - time_StartNewKF).count();
            vdNewKF_ms.push_back(timeNewKF);
#endif

            // We allow points with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame. Only has effect if lastframe is tracked
            for(int i=0; i<mCurrentFrame.N;i++)
            {
                if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
            }
        }

        // Reset if the camera get lost soon after initialization
        if(mState==LOST)
        {
            if(pCurrentMap->KeyFramesInMap()==0)
            {
                mpSystem->ResetActiveMap();
                return;
            }
            if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
                if (!pCurrentMap->isImuInitialized())
                {
                    Verbose::PrintMess("Track lost before IMU initialisation, reseting...", Verbose::VERBOSITY_QUIET);
                    mpSystem->ResetActiveMap();
                    return;
                }

            CreateMapInAtlas();

            return;
        }

        if(!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

        UpdateReliableFlowFrame();
        mLastFrame = Frame(mCurrentFrame);
    }




    if(mState==OK || mState==RECENTLY_LOST)
    {
        // Store frame pose information to retrieve the complete camera trajectory afterwards.
        if(mCurrentFrame.isSet())
        {
            Sophus::SE3f Tcr_ = mCurrentFrame.GetPose() * mCurrentFrame.mpReferenceKF->GetPoseInverse();
            mlRelativeFramePoses.push_back(Tcr_);
            mlpReferences.push_back(mCurrentFrame.mpReferenceKF);
            Sophus::SE3f referencePose;
            float referenceScale=1.f;
            Map* referenceMap=nullptr;
            if(!mCurrentFrame.mpReferenceKF->GetReplayReference(
                   referencePose,referenceScale,referenceMap))
                referenceScale=1.f;
            mlReferenceUnitScales.push_back(referenceScale);
            mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
            mlbLost.push_back(mState==RECENTLY_LOST);
            if(mState==OK && CurrentPoseHasTagConstraint() && mpAtlas->GetCurrentMap()->mbMetric) {
                MarkerMetricFrame anchored;
                anchored.historyIndex=mlRelativeFramePoses.size()-1;
                anchored.worldFromCamera=mCurrentFrame.GetPose().inverse();
                if(mnMarkerGraphVisualFrameId==long(mCurrentFrame.mnId) &&
                   mCurrentFrame.mpReferenceKF) {
                    anchored.visualCameraFromReference=
                        mMarkerGraphVisualTwc.inverse()*
                        mCurrentFrame.mpReferenceKF->GetPoseInverse();
                    anchored.hasVisualRelative=
                        anchored.visualCameraFromReference.matrix().allFinite();
                }
                mMarkerMetricFrames[mCurrentFrame.mnId]=anchored;
            }
        }
        else
        {
            // This can happen if tracking is lost
            mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
            mlpReferences.push_back(mlpReferences.back());
            mlReferenceUnitScales.push_back(mlReferenceUnitScales.back());
            mlFrameTimes.push_back(mlFrameTimes.back());
            mlbLost.push_back(true);
        }

    }

#ifdef REGISTER_LOOP
    if (Stop()) {

        // Safe area to stop
        while(isStopped())
        {
            usleep(3000);
        }
    }
#endif
}


void Tracking::StereoInitialization()
{
    if(mCurrentFrame.N>500)
    {
        if (mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
        {
            if (!mCurrentFrame.mpImuPreintegrated || !mLastFrame.mpImuPreintegrated)
            {
                cout << "not IMU meas" << endl;
                return;
            }

            if (!mFastInit && (mCurrentFrame.mpImuPreintegratedFrame->avgA-mLastFrame.mpImuPreintegratedFrame->avgA).norm()<0.5)
            {
                cout << "not enough acceleration" << endl;
                return;
            }

            if(mpImuPreintegratedFromLastKF)
                delete mpImuPreintegratedFromLastKF;

            mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);
            mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;
        }

        // Set Frame pose to the origin (In case of inertial SLAM to imu)
        if (mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
        {
            Eigen::Matrix3f Rwb0 = mCurrentFrame.mImuCalib.mTcb.rotationMatrix();
            Eigen::Vector3f twb0 = mCurrentFrame.mImuCalib.mTcb.translation();
            Eigen::Vector3f Vwb0;
            Vwb0.setZero();
            mCurrentFrame.SetImuPoseVelocity(Rwb0, twb0, Vwb0);
        }
        else
            mCurrentFrame.SetPose(Sophus::SE3f());

        // Create KeyFrame
        KeyFrame* pKFini = new KeyFrame(mCurrentFrame,mpAtlas->GetCurrentMap(),mpKeyFrameDB);

        // Insert KeyFrame in the map
        mpAtlas->AddKeyFrame(pKFini);

        // Create MapPoints and asscoiate to KeyFrame
        if(!mpCamera2){
            for(int i=0; i<mCurrentFrame.N;i++)
            {
                float z = mCurrentFrame.mvDepth[i];
                if(z>0)
                {
                    Eigen::Vector3f x3D;
                    mCurrentFrame.UnprojectStereo(i, x3D);
                    MapPoint* pNewMP = new MapPoint(x3D, pKFini, mpAtlas->GetCurrentMap());
                    pNewMP->AddObservation(pKFini,i);
                    pKFini->AddMapPoint(pNewMP,i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpAtlas->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                }
            }
        } else{
            for(int i = 0; i < mCurrentFrame.Nleft; i++){
                int rightIndex = mCurrentFrame.mvLeftToRightMatch[i];
                if(rightIndex != -1){
                    Eigen::Vector3f x3D = mCurrentFrame.mvStereo3Dpoints[i];

                    MapPoint* pNewMP = new MapPoint(x3D, pKFini, mpAtlas->GetCurrentMap());

                    pNewMP->AddObservation(pKFini,i);
                    pNewMP->AddObservation(pKFini,rightIndex + mCurrentFrame.Nleft);

                    pKFini->AddMapPoint(pNewMP,i);
                    pKFini->AddMapPoint(pNewMP,rightIndex + mCurrentFrame.Nleft);

                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpAtlas->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                    mCurrentFrame.mvpMapPoints[rightIndex + mCurrentFrame.Nleft]=pNewMP;
                }
            }
        }

        Verbose::PrintMess("New Map created with " + to_string(mpAtlas->MapPointsInMap()) + " points", Verbose::VERBOSITY_QUIET);

        //cout << "Active map: " << mpAtlas->GetCurrentMap()->GetId() << endl;

        mpLocalMapper->InsertKeyFrame(pKFini);

        mLastFrame = Frame(mCurrentFrame);
        mnLastKeyFrameId = mCurrentFrame.mnId;
        mpLastKeyFrame = pKFini;
        //mnLastRelocFrameId = mCurrentFrame.mnId;

        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints=mpAtlas->GetAllMapPoints();
        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;

        mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

        mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.push_back(pKFini);

        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.GetPose());

        mState=OK;
    }
}



void Tracking::StoreMarkerFrame()
{
    mCurrentFrame.mpReferenceKF = mpReferenceKF;
    mlRelativeFramePoses.push_back(mCurrentFrame.GetPose() * mpReferenceKF->GetPoseInverse());
    mlpReferences.push_back(mpReferenceKF);
    Sophus::SE3f referencePose;
    float referenceScale=1.f;
    Map* referenceMap=nullptr;
    if(!mpReferenceKF->GetReplayReference(referencePose,referenceScale,referenceMap))
        referenceScale=1.f;
    mlReferenceUnitScales.push_back(referenceScale);
    mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
    mlbLost.push_back(false);
    if(CurrentPoseHasTagConstraint() && mpAtlas->GetCurrentMap()->mbMetric) {
        MarkerMetricFrame anchored;
        anchored.historyIndex=mlRelativeFramePoses.size()-1;
        anchored.worldFromCamera=mCurrentFrame.GetPose().inverse();
        if(mnMarkerGraphVisualFrameId==long(mCurrentFrame.mnId) && mpReferenceKF) {
            anchored.visualCameraFromReference=
                mMarkerGraphVisualTwc.inverse()*mpReferenceKF->GetPoseInverse();
            anchored.hasVisualRelative=
                anchored.visualCameraFromReference.matrix().allFinite();
        }
        mMarkerMetricFrames[mCurrentFrame.mnId]=anchored;
    }
    mLastFrame = Frame(mCurrentFrame);
    mbVelocity = false;
}

bool Tracking::MarkerRecoveryPoseConsistent()
{
    const Sophus::SE3f Twc=mCurrentFrame.GetPose().inverse();
    if(!Twc.matrix().allFinite() ||
       (Twc.translation()-mExternalTagTwc.translation()).norm()>
           mTagMaxAlignmentPositionResidualM) return false;
    const float cosine=std::max(-1.f,std::min(1.f,0.5f*
        ((Twc.rotationMatrix().transpose()*mExternalTagTwc.rotationMatrix()).trace()-1.f)));
    if(std::acos(cosine)*180.f/float(M_PI)>mTagMaxAlignmentRotationResidualDeg)
        return false;
    double error=0.; int count=0;
    for(size_t i=0;i<mvExternalTagWorldPoints.size();++i) {
        if(!mvExternalTagPointWeights.empty() && mvExternalTagPointWeights[i]<0.99f)
            continue;
        const Eigen::Vector3f p=mCurrentFrame.GetPose()*mvExternalTagWorldPoints[i];
        if(!p.allFinite() || p.z()<=0.f) return false;
        const Eigen::Vector2f uv=mpCamera->project(p);
        error+=(uv-Eigen::Vector2f(mvExternalTagImagePoints[i].x,
                                   mvExternalTagImagePoints[i].y)).squaredNorm();
        ++count;
    }
    return count>=4 && std::isfinite(error) && error/count<=9.;
}

// This path publishes measured tag poses, not synthetic background depths.
bool Tracking::TrackMarkerSeed()
{
    Map* map = mpAtlas->GetCurrentMap();
    if(map->mbBackgroundReady && mState==MARKER_TRACKING) {
        if(mbHasExternalTagObservation || mbHasTrackedTagObservation) {
            mCurrentFrame.SetPose(mExternalTagTwc.inverse());
            // A new Frame has no ORB associations. Calling TrackLocalMap
            // directly clears the local map (zero keyframe votes), so marker
            // tracking would permanently suppress background recovery.
            // Seed actual descriptor matches at the measured marker pose.
            ORBmatcher matcher(0.8,true);
            std::set<MapPoint*> found;
            std::vector<KeyFrame*> references;
            if(mpReferenceKF && !mpReferenceKF->isBad()) {
                references=mpReferenceKF->GetBestCovisibilityKeyFrames(10);
                references.push_back(mpReferenceKF);
            }
            int matches=0;
            for(KeyFrame* reference:references) {
                if(reference->isBad() || reference->GetMap()!=map) continue;
                matches+=matcher.SearchByProjection(mCurrentFrame,reference,found,10,100);
                for(MapPoint* point:mCurrentFrame.mvpMapPoints) if(point) found.insert(point);
            }
            // Failure-only rescue at a measured, decoded marker pose. Keep the
            // ordinary search unchanged, and never let weak/partial tags widen it.
            static const bool boundedSearch = !std::getenv("HT_MARKER_RECOVERY_SEARCH") ||
                std::string(std::getenv("HT_MARKER_RECOVERY_SEARCH")) != "0";
            bool expandedSeed=false;
            std::unique_ptr<Frame> narrowFrame;
            if(boundedSearch && matches<15 && mbHasExternalTagObservation &&
               !mbHasTrackedTagObservation && mExternalTagConfidence>=0.35f) {
                const auto start=std::chrono::steady_clock::now();
                narrowFrame.reset(new Frame(mCurrentFrame));
                const int originalMatches=matches;
                for(KeyFrame* reference:references) {
                    if(reference->isBad() || reference->GetMap()!=map) continue;
                    matches+=matcher.SearchByProjection(mCurrentFrame,reference,found,20,64);
                    for(MapPoint* point:mCurrentFrame.mvpMapPoints) if(point) found.insert(point);
                }
                const int inliers=matches>=30 ? Optimizer::PoseOptimization(&mCurrentFrame) : 0;
                expandedSeed=inliers>=30 && MarkerRecoveryPoseConsistent();
                if(expandedSeed) {
                    for(int i=0;i<mCurrentFrame.N;++i)
                        if(mCurrentFrame.mvbOutlier[i]) mCurrentFrame.mvpMapPoints[i]=nullptr;
                    matches=inliers;
                } else {
                    mCurrentFrame=*narrowFrame;
                    matches=originalMatches;
                }
                cout << "MARKER_RECOVERY_SEARCH frame=" << mCurrentFrame.mnId
                     << " first=" << originalMatches << " inliers=" << inliers
                     << " seed_accepted=" << expandedSeed << " ms="
                     << std::chrono::duration<double,std::milli>(
                            std::chrono::steady_clock::now()-start).count() << endl;
            }
            const bool visualSeed=matches>=15 || Relocalization();
            const bool localRecovered=visualSeed && TrackLocalMap();
            bool recovered=localRecovered && (!expandedSeed || MarkerRecoveryPoseConsistent());
            if(expandedSeed)
                cout << "MARKER_RECOVERY_SEARCH_RESULT frame=" << mCurrentFrame.mnId
                     << " recovered=" << recovered << endl;
            if(expandedSeed && !recovered) {
                // A failed rescue must not suppress the original BoW
                // relocalization path that the narrow-search failure would use.
                mCurrentFrame=*narrowFrame;
                expandedSeed=false;
                recovered=Relocalization() && TrackLocalMap();
            }
            if(recovered) {
                // Recovery must use the same tag+ORB solve as normal tracking.
                // A marker pose is an initialization, not the final constraint.
                if(mbTagFusionEnabled) {
                    CaptureMarkerGraphVisualPose();
                    ApplyExternalTagPoseConstraint();
                }
                mState=OK;
            }
            else mCurrentFrame.SetPose(mExternalTagTwc.inverse());
            if(mState==MARKER_TRACKING) InsertMarkerOnlyEventKeyFrame();
            else if(NeedNewKeyFrame()) CreateNewKeyFrame();
            StoreMarkerFrame();
            return true;
        }
        mState=RECENTLY_LOST;
        mTimeStampLost=mCurrentFrame.mTimeStamp;
    }
    const bool seed = !map->mbBackgroundReady &&
        (map->mbMarkerSeed || (mState==NOT_INITIALIZED && mbHasExternalTagObservation));
    if(!seed)
        return false;
    mMarkerBootstrapStatus.active = true;
    mMarkerBootstrapStatus.referenceFrame = mpMarkerSeedKF ? long(mInitialFrame.mnId) : -1;
    if(!mbHasExternalTagObservation && !mbHasTrackedTagObservation) {
        mMarkerBootstrapStatus.reason = "marker_missing";
        mState = LOST;
        mnMatchesInliers = 0;
        if(mLastReliableMarkerTime >= 0 &&
           mCurrentFrame.mTimeStamp - mLastReliableMarkerTime > 3.0)
            CreateMapInAtlas();
        return true;
    }
    if(mbHasExternalTagObservation) mLastReliableMarkerTime = mCurrentFrame.mTimeStamp;
    mCurrentFrame.SetPose(mExternalTagTwc.inverse());
    map->mbMetric = true;
    map->mbMarkerSeed = true;
    map->mbRigidMarkerLayout = map->mbRigidMarkerLayout || mbRigidMarkerLayout;
    map->mMetricScale = 1.0f;
    mbTagMetricAligned = true;
    mRecoveredTagMetricScale = 1.0f;
    // A restored seed retains its gauge. A new visual reference may be needed
    // because transient Frame descriptors are not serialized in Atlas.
    if(!mpMarkerSeedKF || mpMarkerSeedKF->GetMap()!=map) {
        SetMarkerBootstrapReference();
        mpMarkerSeedKF = new KeyFrame(mCurrentFrame,map,mpKeyFrameDB);
        mpMarkerSeedKF->ComputeBoW();
        AttachCurrentTagObservation(mpMarkerSeedKF);
        mpAtlas->AddKeyFrame(mpMarkerSeedKF);
        if(map->mvpKeyFrameOrigins.empty())
            map->mvpKeyFrameOrigins.push_back(mpMarkerSeedKF);
        mpReferenceKF = mpMarkerSeedKF;
        mpLastKeyFrame = mpMarkerSeedKF;
        mnLastKeyFrameId = mCurrentFrame.mnId;
        mvpLocalKeyFrames = {mpMarkerSeedKF};
        ++map->mnRevision;
        cout << "MARKER_TRACKING: metric seed; background points=0" << endl;
    }
    mState = MARKER_TRACKING;
    if(mbHasExternalTagObservation) BootstrapMarkerBackground();
    else mMarkerBootstrapStatus.reason="partial_marker_tracking";
    if(mState==MARKER_TRACKING) InsertMarkerOnlyEventKeyFrame();
    StoreMarkerFrame();
    return true;
}

void Tracking::SetMarkerBootstrapReference()
{
    // The world gauge and historical keyframes stay unchanged. This is only
    // a candidate image; it becomes a keyframe if triangulation succeeds.
    mInitialFrame = Frame(mCurrentFrame);
    mbHasInitialTagObservation = mbHasExternalTagObservation;
    mInitialTagTwc = mExternalTagTwc;
    mInitialTagConfidence = mExternalTagConfidence;
    mvInitialTagWorldPoints = mvExternalTagWorldPoints;
    mvInitialTagImagePoints = mvExternalTagImagePoints;
    mvInitialTagPointWeights = mvExternalTagPointWeights;
    mvInitialTagIds = mvExternalTagIds;
    mvbPrevMatched.clear();
    for(const auto &key : mInitialFrame.mvKeysUn)
        mvbPrevMatched.push_back(key.pt);
    if(mMarkerBootstrapStatus.referenceFrame < 0)
        mMarkerBootstrapStatus.referenceFrame = long(mInitialFrame.mnId);
    mMarkerBootstrapStatus.referenceChanged = true;
}

void Tracking::BootstrapMarkerBackground()
{
    auto &status = mMarkerBootstrapStatus;
    status.reason = "waiting_baseline";
    // SetExternalTagObservation already checks strong corners, confidence,
    // positive depth and reprojection. Keep single-tag views eligible; the
    // two-view geometry below rejects inconsistent depth hypotheses.
    if(mCurrentFrame.N < 100) {
        status.reason = "insufficient_features";
        return;
    }
    if(!mbHasInitialTagObservation || mInitialFrame.N < 100) {
        SetMarkerBootstrapReference();
        status.reason = "stronger_reference";
        return;
    }
    status.baselineM = (mCurrentFrame.GetCameraCenter()-mInitialFrame.GetCameraCenter()).norm();
    const double age = mCurrentFrame.mTimeStamp-mInitialFrame.mTimeStamp;
    const float angle = (mCurrentFrame.GetPose().so3()*mInitialFrame.GetPose().so3().inverse()).log().norm();
    // Marker localization is already metric and remains available while
    // stationary. Only triangulation of the background needs this baseline;
    // a shallow seed otherwise conflicts with later marker-constrained BA.
    if(mCurrentFrame.mnId == mInitialFrame.mnId ||
       status.baselineM < std::max(0.01f,mTagMinimumScaleBaselineM)) {
        if(age > 0.25 && angle > 0.35f) {
            SetMarkerBootstrapReference();
            status.reason = "reference_view_changed";
        }
        return;
    }
    ORBmatcher matcher(0.8,true);
    std::vector<int> matches;
    status.matches = matcher.SearchForInitialization(mInitialFrame,mCurrentFrame,mvbPrevMatched,matches,100,true);
    if(status.matches < 50) {
        status.reason = "insufficient_matches";
        if(age > 0.25) SetMarkerBootstrapReference();
        return;
    }
    Eigen::Matrix<float,3,4> firstProjection = mInitialFrame.GetPose().matrix3x4();
    Eigen::Matrix<float,3,4> secondProjection = mCurrentFrame.GetPose().matrix3x4();
    struct Candidate { size_t first, second; Eigen::Vector3f point; };
    std::vector<Candidate> candidates;
    for(size_t i=0;i<matches.size();++i) {
        if(matches[i]<0) continue;
        const size_t j=matches[i];
        Eigen::Vector3f ray1=mpCamera->unprojectEig(mInitialFrame.mvKeysUn[i].pt);
        Eigen::Vector3f ray2=mpCamera->unprojectEig(mCurrentFrame.mvKeysUn[j].pt);
        Eigen::Vector3f worldRay1=mInitialFrame.GetPose().rotationMatrix().transpose()*ray1;
        Eigen::Vector3f worldRay2=mCurrentFrame.GetPose().rotationMatrix().transpose()*ray2;
        const float cosine=worldRay1.normalized().dot(worldRay2.normalized());
        if(cosine<=0 || cosine>0.9998f) continue;
        Eigen::Vector3f point;
        if(!GeometricTools::Triangulate(ray1,ray2,firstProjection,secondProjection,point)) continue;
        Eigen::Vector3f p1=mInitialFrame.GetPose()*point;
        Eigen::Vector3f p2=mCurrentFrame.GetPose()*point;
        if(!point.allFinite() || p1.z()<=0 || p2.z()<=0) continue;
        Eigen::Vector2f e1=mpCamera->project(p1)-
            Eigen::Vector2f(mInitialFrame.mvKeysUn[i].pt.x,mInitialFrame.mvKeysUn[i].pt.y);
        Eigen::Vector2f e2=mpCamera->project(p2)-
            Eigen::Vector2f(mCurrentFrame.mvKeysUn[j].pt.x,mCurrentFrame.mvKeysUn[j].pt.y);
        if(e1.squaredNorm()>5.991f*mInitialFrame.mvLevelSigma2[mInitialFrame.mvKeysUn[i].octave] ||
           e2.squaredNorm()>5.991f*mCurrentFrame.mvLevelSigma2[mCurrentFrame.mvKeysUn[j].octave])
            continue;
        const float ratio=(point-mInitialFrame.GetCameraCenter()).norm()/
                          (point-mCurrentFrame.GetCameraCenter()).norm();
        const float octaveRatio=mInitialFrame.mvScaleFactors[mInitialFrame.mvKeysUn[i].octave]/
                                mCurrentFrame.mvScaleFactors[mCurrentFrame.mvKeysUn[j].octave];
        const float tolerance=1.5f*mCurrentFrame.mfScaleFactor;
        if(ratio* tolerance<octaveRatio || ratio>octaveRatio*tolerance) continue;
        candidates.push_back({i,j,point});
    }
    status.triangulated = candidates.size();
    if(candidates.size()<50) {
        status.reason = "insufficient_geometry";
        // Keep a small-baseline pair so slow translation can accumulate
        // parallax. Replace a stale failed pair only after meaningful motion.
        if(age > 1.0 && status.baselineM > 0.03f)
            SetMarkerBootstrapReference();
        return;
    }
    Map* map=mpAtlas->GetCurrentMap();
    KeyFrame* first = mpMarkerSeedKF;
    if(mInitialFrame.mnId != mpMarkerSeedKF->mnFrameId) {
        first = new KeyFrame(mInitialFrame,map,mpKeyFrameDB);
        first->ComputeBoW();
        AttachCurrentTagObservation(first, true);
        mpAtlas->AddKeyFrame(first);
    }
    KeyFrame* current=new KeyFrame(mCurrentFrame,map,mpKeyFrameDB);
    current->ComputeBoW();
    AttachCurrentTagObservation(current);
    mpAtlas->AddKeyFrame(current);
    for(const Candidate &candidate:candidates) {
        MapPoint* point=new MapPoint(candidate.point,current,map);
        point->AddObservation(first,candidate.first);
        point->AddObservation(current,candidate.second);
        first->AddMapPoint(point,candidate.first);
        current->AddMapPoint(point,candidate.second);
        point->ComputeDistinctiveDescriptors();
        point->UpdateNormalAndDepth();
        mpAtlas->AddMapPoint(point);
        mCurrentFrame.mvpMapPoints[candidate.second]=point;
        mCurrentFrame.mvbOutlier[candidate.second]=false;
    }
    first->UpdateConnections();
    current->UpdateConnections();
    if(first != mpMarkerSeedKF) {
        // Preserve an acyclic spanning tree rooted at the original marker
        // keyframe. Do not invent a visual covisibility edge to that image.
        if(first->GetParent()) first->GetParent()->EraseChild(first);
        first->ChangeParent(mpMarkerSeedKF);
    }
    mpReferenceKF=current;
    mpLastKeyFrame=current;
    mnLastKeyFrameId=mCurrentFrame.mnId;
    mvpLocalKeyFrames={first,current};
    mvpLocalMapPoints=map->GetAllMapPoints();
    mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);
    map->mbBackgroundReady=true;
    ++map->mnRevision;
    mnMatchesInliers=candidates.size();
    mState=OK;
    status.reason = "background_ready";
    mpLocalMapper->InsertKeyFrame(first);
    mpLocalMapper->InsertKeyFrame(current);
    mpLocalMapper->mFirstTs=current->mTimeStamp;
    cout << "MARKER_BACKGROUND_READY: " << candidates.size() << " metric points" << endl;
}

void Tracking::MonocularInitialization()
{
    if(mbMarkerOnlyInitialization)
        return;

    if(!mbReadyToInitializate)
    {
        // Set Reference Frame
        if(mCurrentFrame.mvKeys.size()>100)
        {

            mInitialFrame = Frame(mCurrentFrame);
            mLastFrame = Frame(mCurrentFrame);
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            for(size_t i=0; i<mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i]=mCurrentFrame.mvKeysUn[i].pt;

            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);

            if (mSensor == System::IMU_MONOCULAR)
            {
                if(mpImuPreintegratedFromLastKF)
                {
                    delete mpImuPreintegratedFromLastKF;
                }
                mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);
                mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;

            }

            mbHasInitialTagObservation = mbHasExternalTagObservation;
            if(mbHasInitialTagObservation)
            {
                mInitialTagTwc = mExternalTagTwc;
                mInitialTagConfidence = mExternalTagConfidence;
                mvInitialTagWorldPoints = mvExternalTagWorldPoints;
                mvInitialTagImagePoints = mvExternalTagImagePoints;
                mvInitialTagPointWeights = mvExternalTagPointWeights;
                mvInitialTagIds = mvExternalTagIds;
            }

            mbReadyToInitializate = true;

            return;
        }
    }
    else
    {
        if (((int)mCurrentFrame.mvKeys.size()<=100)||((mSensor == System::IMU_MONOCULAR)&&(mLastFrame.mTimeStamp-mInitialFrame.mTimeStamp>1.0)))
        {
            mbReadyToInitializate = false;
            mbHasInitialTagObservation = false;

            return;
        }

        // Find correspondences
        ORBmatcher matcher(0.9,true);
        int nmatches = matcher.SearchForInitialization(mInitialFrame,mCurrentFrame,mvbPrevMatched,mvIniMatches,100);

        // Check if there are enough correspondences
        if(nmatches<100)
        {
            mbReadyToInitializate = false;
            mbHasInitialTagObservation = false;
            return;
        }

        Sophus::SE3f Tcw;
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

        if(mpCamera->ReconstructWithTwoViews(mInitialFrame.mvKeysUn,mCurrentFrame.mvKeysUn,mvIniMatches,Tcw,mvIniP3D,vbTriangulated))
        {
            for(size_t i=0, iend=mvIniMatches.size(); i<iend;i++)
            {
                if(mvIniMatches[i]>=0 && !vbTriangulated[i])
                {
                    mvIniMatches[i]=-1;
                    nmatches--;
                }
            }

            // Set Frame Poses
            mInitialFrame.SetPose(Sophus::SE3f());
            mCurrentFrame.SetPose(Tcw);

            CreateInitialMapMonocular();
        }
    }
}



void Tracking::CreateInitialMapMonocular()
{
    // Create KeyFrames
    KeyFrame* pKFini = new KeyFrame(mInitialFrame,mpAtlas->GetCurrentMap(),mpKeyFrameDB);
    KeyFrame* pKFcur = new KeyFrame(mCurrentFrame,mpAtlas->GetCurrentMap(),mpKeyFrameDB);

    if(mSensor == System::IMU_MONOCULAR)
        pKFini->mpImuPreintegrated = (IMU::Preintegrated*)(NULL);


    pKFini->ComputeBoW();
    pKFcur->ComputeBoW();

    // Insert KFs in the map
    mpAtlas->AddKeyFrame(pKFini);
    mpAtlas->AddKeyFrame(pKFcur);

    for(size_t i=0; i<mvIniMatches.size();i++)
    {
        if(mvIniMatches[i]<0)
            continue;

        //Create MapPoint.
        Eigen::Vector3f worldPos;
        worldPos << mvIniP3D[i].x, mvIniP3D[i].y, mvIniP3D[i].z;
        MapPoint* pMP = new MapPoint(worldPos,pKFcur,mpAtlas->GetCurrentMap());

        pKFini->AddMapPoint(pMP,i);
        pKFcur->AddMapPoint(pMP,mvIniMatches[i]);

        pMP->AddObservation(pKFini,i);
        pMP->AddObservation(pKFcur,mvIniMatches[i]);

        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateNormalAndDepth();

        //Fill Current Frame structure
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

        //Add to Map
        mpAtlas->AddMapPoint(pMP);
    }


    // Update Connections
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    std::set<MapPoint*> sMPs;
    sMPs = pKFini->GetMapPoints();

    // Bundle Adjustment
    Verbose::PrintMess("New Map created with " + to_string(mpAtlas->MapPointsInMap()) + " points", Verbose::VERBOSITY_QUIET);
    Optimizer::GlobalBundleAdjustemnt(mpAtlas->GetCurrentMap(),20);

    float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    float invMedianDepth;
    if(mSensor == System::IMU_MONOCULAR)
        invMedianDepth = 4.0f/medianDepth; // 4.0f
    else
        invMedianDepth = 1.0f/medianDepth;

    if(medianDepth<0 || pKFcur->TrackedMapPoints(1)<50) // TODO Check, originally 100 tracks
    {
        Verbose::PrintMess("Wrong initialization, reseting...", Verbose::VERBOSITY_QUIET);
        mpSystem->ResetActiveMap();
        return;
    }

    // Scale initial baseline
    Sophus::SE3f Tc2w = pKFcur->GetPose();
    Tc2w.translation() *= invMedianDepth;
    pKFcur->SetPose(Tc2w);

    // Scale points
    vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
    for(size_t iMP=0; iMP<vpAllMapPoints.size(); iMP++)
    {
        if(vpAllMapPoints[iMP])
        {
            MapPoint* pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
            pMP->UpdateNormalAndDepth();
        }
    }

    AttachCurrentTagObservation(pKFini, true);
    AttachCurrentTagObservation(pKFcur, false);
    mCurrentFrame.SetPose(pKFcur->GetPose());
    if(mbHasInitialTagObservation)
    {
        mTagScaleReferenceMetricTwc = mInitialTagTwc;
        mTagScaleReferenceSlamTwc = pKFini->GetPoseInverse();
        mTagScaleReferenceTimestamp = pKFini->mTimeStamp;
        mpTagScaleReferenceMap = pKFini->GetMap();
        mbHasTagScaleReference = true;
        TryAlignMapToTagWorld();
    }

    if (mSensor == System::IMU_MONOCULAR)
    {
        pKFcur->mPrevKF = pKFini;
        pKFini->mNextKF = pKFcur;
        pKFcur->mpImuPreintegrated = mpImuPreintegratedFromLastKF;

        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(pKFcur->mpImuPreintegrated->GetUpdatedBias(),pKFcur->mImuCalib);
    }


    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);
    mpLocalMapper->mFirstTs=pKFcur->mTimeStamp;

    mCurrentFrame.SetPose(pKFcur->GetPose());
    mnLastKeyFrameId=mCurrentFrame.mnId;
    mpLastKeyFrame = pKFcur;
    //mnLastRelocFrameId = mInitialFrame.mnId;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints=mpAtlas->GetAllMapPoints();
    mpReferenceKF = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFcur;

    // Compute here initial velocity
    vector<KeyFrame*> vKFs = mpAtlas->GetAllKeyFrames();

    Sophus::SE3f deltaT = vKFs.back()->GetPose() * vKFs.front()->GetPoseInverse();
    mbVelocity = false;
    Eigen::Vector3f phi = deltaT.so3().log();

    double aux = (mCurrentFrame.mTimeStamp-mLastFrame.mTimeStamp)/(mCurrentFrame.mTimeStamp-mInitialFrame.mTimeStamp);
    phi *= aux;

    mLastFrame = Frame(mCurrentFrame);

    mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

    mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.push_back(pKFini);

    mState=OK;
    mpAtlas->GetCurrentMap()->mbBackgroundReady = true;

    initID = pKFcur->mnId;
}


void Tracking::CreateMapInAtlas()
{
    ClearReliableFlowFrame();
    CancelMarkerGraph();
    CancelPendingTagAlignment();
    mnLastInitFrameId = mCurrentFrame.mnId;
    mpAtlas->CreateNewMap();
    mbTagMetricAligned = false;
    mbTagAlignmentPending = false;
    mbHasTagScaleReference = false;
    mbHasInitialTagObservation = false;
    mRecoveredTagMetricScale = 0.0f;
    mvTagScaleSamples.clear();
    mpMarkerSeedKF = nullptr;
    mLastReliableMarkerTime = -1.0;
    if (mSensor==System::IMU_STEREO || mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_RGBD)
        mpAtlas->SetInertialSensor();
    mbSetInit=false;

    mnInitialFrameId = mCurrentFrame.mnId+1;
    mState = NO_IMAGES_YET;

    // Restart the variable with information about the last KF
    mbVelocity = false;
    //mnLastRelocFrameId = mnLastInitFrameId; // The last relocation KF_id is the current id, because it is the new starting point for new map
    Verbose::PrintMess("First frame id in map: " + to_string(mnLastInitFrameId+1), Verbose::VERBOSITY_NORMAL);
    mbVO = false; // Init value for know if there are enough MapPoints in the last KF
    if(mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR)
    {
        mbReadyToInitializate = false;
    }

    if((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && mpImuPreintegratedFromLastKF)
    {
        delete mpImuPreintegratedFromLastKF;
        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);
    }

    if(mpLastKeyFrame)
        mpLastKeyFrame = static_cast<KeyFrame*>(NULL);

    if(mpReferenceKF)
        mpReferenceKF = static_cast<KeyFrame*>(NULL);

    mLastFrame = Frame();
    mCurrentFrame = Frame();
    mvIniMatches.clear();

    mbCreatedMap = true;
}

void Tracking::CheckReplacedInLastFrame()
{
    for(int i =0; i<mLastFrame.N; i++)
    {
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];

        if(pMP)
        {
            MapPoint* pRep = pMP->GetReplaced();
            if(pRep)
            {
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }
}


bool Tracking::TrackReferenceKeyFrame()
{
    // Compute Bag of Words vector
    mCurrentFrame.ComputeBoW();

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.7,true);
    vector<MapPoint*> vpMapPointMatches;

    int nmatches = matcher.SearchByBoW(mpReferenceKF,mCurrentFrame,vpMapPointMatches);

    if(nmatches<15)
    {
        cout << "TRACK_REF_KF: Less than 15 matches!!\n";
        return false;
    }

    mCurrentFrame.mvpMapPoints = vpMapPointMatches;
    mCurrentFrame.SetPose(mLastFrame.GetPose());

    //mCurrentFrame.PrintPointDistribution();


    // cout << " TrackReferenceKeyFrame mLastFrame.mTcw:  " << mLastFrame.mTcw << endl;
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        //if(i >= mCurrentFrame.Nleft) break;
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                if(i < mCurrentFrame.Nleft){
                    pMP->mbTrackInView = false;
                }
                else{
                    pMP->mbTrackInViewR = false;
                }
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }

    if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
        return true;
    else
        return nmatchesMap>=10;
}

void Tracking::UpdateLastFrame()
{
    // Update pose according to reference keyframe
    KeyFrame* pRef = mLastFrame.mpReferenceKF;
    Sophus::SE3f Tlr = mlRelativeFramePoses.back();
    Sophus::SE3f worldFromReference;
    float currentReferenceScale=1.f;
    Map* referenceMap=nullptr;
    if(pRef && !mlReferenceUnitScales.empty() &&
       pRef->GetReplayReference(worldFromReference,currentReferenceScale,referenceMap) &&
       std::isfinite(mlReferenceUnitScales.back()) && mlReferenceUnitScales.back()>0.f) {
        // Resolve a culled reference through its surviving spanning-tree
        // parent. Using pRef->GetPose() directly can mix a pre-correction
        // culled keyframe with a post-correction map for exactly one frame.
        Tlr.translation() *= currentReferenceScale/mlReferenceUnitScales.back();
        mLastFrame.SetPose(Tlr * worldFromReference.inverse());
    } else
        mLastFrame.SetPose(Tlr * pRef->GetPose());

    if(mnLastKeyFrameId==mLastFrame.mnId || mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR || !mbOnlyTracking)
        return;

    // Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo/RGB-D sensor
    vector<pair<float,int> > vDepthIdx;
    const int Nfeat = mLastFrame.Nleft == -1? mLastFrame.N : mLastFrame.Nleft;
    vDepthIdx.reserve(Nfeat);
    for(int i=0; i<Nfeat;i++)
    {
        float z = mLastFrame.mvDepth[i];
        if(z>0)
        {
            vDepthIdx.push_back(make_pair(z,i));
        }
    }

    if(vDepthIdx.empty())
        return;

    sort(vDepthIdx.begin(),vDepthIdx.end());

    // We insert all close points (depth<mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    int nPoints = 0;
    for(size_t j=0; j<vDepthIdx.size();j++)
    {
        int i = vDepthIdx[j].second;

        bool bCreateNew = false;

        MapPoint* pMP = mLastFrame.mvpMapPoints[i];

        if(!pMP)
            bCreateNew = true;
        else if(pMP->Observations()<1)
            bCreateNew = true;

        if(bCreateNew)
        {
            Eigen::Vector3f x3D;

            if(mLastFrame.Nleft == -1){
                mLastFrame.UnprojectStereo(i, x3D);
            }
            else{
                x3D = mLastFrame.UnprojectStereoFishEye(i);
            }

            MapPoint* pNewMP = new MapPoint(x3D,mpAtlas->GetCurrentMap(),&mLastFrame,i);
            mLastFrame.mvpMapPoints[i]=pNewMP;

            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
        }
        else
        {
            nPoints++;
        }

        if(vDepthIdx[j].first>mThDepth && nPoints>100)
            break;

    }
}

namespace {
// Matching uses scratch fields on live points. Save each scalar's object
// representation separately: unused projections may not be initialized yet,
// and no assumption about padding or neighboring class members is required.
class TemporalRecoveryPointScratch {
    MapPoint* point;
    std::vector<unsigned char> bytes;
    template<class Visitor> void fields(Visitor visitor) {
        visitor(point->mTrackProjX); visitor(point->mTrackProjY);
        visitor(point->mTrackDepth); visitor(point->mTrackDepthR);
        visitor(point->mTrackProjXR); visitor(point->mTrackProjYR);
        visitor(point->mbTrackInView); visitor(point->mbTrackInViewR);
        visitor(point->mnTrackScaleLevel); visitor(point->mnTrackScaleLevelR);
        visitor(point->mTrackViewCos); visitor(point->mTrackViewCosR);
        visitor(point->mnTrackReferenceForFrame); visitor(point->mnLastFrameSeen);
    }
public:
    explicit TemporalRecoveryPointScratch(MapPoint* p):point(p) {
        bytes.reserve(128);
        fields([&](const auto& field) {
            const size_t offset=bytes.size(); bytes.resize(offset+sizeof(field));
            std::memcpy(bytes.data()+offset,&field,sizeof(field));
        });
    }
    ~TemporalRecoveryPointScratch() noexcept {
        size_t offset=0;
        fields([&](auto& field) {
            std::memcpy(&field,bytes.data()+offset,sizeof(field)); offset+=sizeof(field);
        });
    }
    TemporalRecoveryPointScratch(const TemporalRecoveryPointScratch&)=delete;
    TemporalRecoveryPointScratch& operator=(const TemporalRecoveryPointScratch&)=delete;
};

bool TemporalRecoveryCorrectionBound(const Frame& frame,const Sophus::SE3f& predicted)
{
    // A short rescue must not jump by a large fraction of scene depth or
    // rotate 45 degrees. This bound does not assume that map units are metres.
    std::vector<float> depths;
    for(int i=0;i<frame.N;++i) if(frame.mvpMapPoints[i] && !frame.mvbOutlier[i]) {
        const auto point=predicted*frame.mvpMapPoints[i]->GetWorldPos();
        if(point.allFinite() && point.z()>0) depths.push_back(point.z());
    }
    if(depths.empty()) return false;
    std::nth_element(depths.begin(),depths.begin()+depths.size()/2,depths.end());
    const auto correction=frame.GetPose()*predicted.inverse();
    return correction.matrix().allFinite() && correction.so3().log().norm()<.7853982f &&
        correction.translation().norm()<.25f*depths[depths.size()/2];
}
}

void Tracking::ClearReliableFlowFrame()
{
    mpReliableFlowMap=nullptr;
    mReliableFlowImage.release();
    mvReliableFlowPointIds.clear();
    mReliableFlowFrame=Frame();
}

void Tracking::UpdateReliableFlowFrame()
{
    if(!mbFlowRecoveryEnabled || !mbReliableFrameRecoveryEnabled || !mbReliableFrameCacheEnabled ||
       mSensor!=System::MONOCULAR || mState!=OK ||
       !mCurrentFrame.HasPose() || !std::isfinite(mCurrentFrame.mTimeStamp) ||
       mnMatchesInliers<60 || mImGray.empty()) return;
    Map* map=mpAtlas->GetCurrentMap();
    if(!map || map->IsBad()) { ClearReliableFlowFrame(); return; }
    std::vector<unsigned long> ids(mCurrentFrame.N,static_cast<unsigned long>(-1));
    std::set<MapPoint*> seen;
    std::set<int> cells;
    for(int i=0;i<mCurrentFrame.N;++i) {
        MapPoint* point=mCurrentFrame.mvpMapPoints[i];
        if(!point || mCurrentFrame.mvbOutlier[i] || point->isBad() ||
           point->GetMap()!=map || point->Observations()<2 || !seen.insert(point).second) continue;
        const auto xy=mCurrentFrame.mvKeys[i].pt;
        if(xy.x<0 || xy.y<0 || xy.x>=mImGray.cols || xy.y>=mImGray.rows) continue;
        ids[i]=point->mnId;
        cells.insert(int(xy.y*6/mImGray.rows)*8+int(xy.x*8/mImGray.cols));
    }
    const int eligible=std::count_if(ids.begin(),ids.end(),[](unsigned long id) {
        return id!=static_cast<unsigned long>(-1);
    });
    if(eligible<60 || cells.size()<8) return;
    mReliableFlowFrame=Frame(mCurrentFrame);
    // IDs, not cached raw pointers, own correspondence identity between calls.
    std::fill(mReliableFlowFrame.mvpMapPoints.begin(),mReliableFlowFrame.mvpMapPoints.end(),nullptr);
    mReliableFlowFrame.mpReferenceKF=nullptr;
    mReliableFlowImage=mImGray.clone();
    mvReliableFlowPointIds=std::move(ids);
    mpReliableFlowMap=map;
    mnReliableFlowMapId=map->GetId();
    mnReliableFlowBigChange=map->GetLastBigChangeIdx();
    mnReliableFlowGraphSequence=map->mnMarkerGraphSequence;
    mbReliableFlowMetric=map->mbMetric;
    mReliableFlowScale=map->mMetricScale;
}

bool Tracking::TryTemporalFlowRecovery(int minimumImprovement)
{
    const Frame original=mCurrentFrame;
    const int originalInliers=mnMatchesInliers;
    auto restore=[&]() { mCurrentFrame=original; mnMatchesInliers=originalInliers; };
    try {
    if(!mCurrentFrame.HasPose()) {
        const Frame originalLast=mLastFrame;
        UpdateLastFrame();
        mCurrentFrame.SetPose(mLastFrame.GetPose());
        if(mbReliableFrameRecoveryEnabled) mLastFrame=originalLast;
    }
    const Frame predicted=mCurrentFrame;
    auto validate=[&](const Frame* source,const cv::Mat* image) {
        mCurrentFrame=predicted;
        if(!TrackWithTemporalFlow(true,source,image)) return false;
        // The explicit off switch retains the pre-change immediate recovery
        // path, so recorded A/B runs isolate this bounded recovery change.
        if(!mbReliableFrameRecoveryEnabled) {
            const bool recovered=TrackLocalMap();
            return recovered && mnMatchesInliers>=30 && mnMatchesInliers>minimumImprovement;
        }
        // Reuse the already selected local map. Running UpdateLocalMap twice
        // for the same frame can discard points stamped by the first trial.
        // This guard restores local-point scratch on success, failure and
        // OpenCV exceptions; the trial never writes KF/reference statistics.
        std::set<MapPoint*> touched(mvpLocalMapPoints.begin(),mvpLocalMapPoints.end());
        touched.insert(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end());
        std::vector<std::unique_ptr<TemporalRecoveryPointScratch>> scratch;
        for(MapPoint* point:touched) if(point)
            scratch.emplace_back(new TemporalRecoveryPointScratch(point));
        const bool recovered=TrackLocalMap(true);
        const bool correctionOK=!source || TemporalRecoveryCorrectionBound(mCurrentFrame,predicted.GetPose());
        const bool accepted=recovered && mnMatchesInliers>=30 &&
                            mnMatchesInliers>minimumImprovement && correctionOK;
        std::cout<<"FLOW_RECOVERY_LOCAL frame="<<mCurrentFrame.mnId
                 <<" source_frame="<<(source?source->mnId:mLastFrame.mnId)
                 <<" inliers="<<mnMatchesInliers<<" accepted="<<accepted
                 <<" reason="<<(!correctionOK?"relative_correction":(accepted?"accepted":"local_inliers"))<<std::endl;
        if(accepted) {
            std::set<MapPoint*> counted;
            // Only a completed local-map attempt counted these points. A
            // failed initial motion/BoW match can leave provisional matches
            // while originalInliers is still zero.
            if(originalInliers>0)
                for(int i=0;i<original.N;++i)
                    if(original.mvpMapPoints[i] && !original.mvbOutlier[i]) counted.insert(original.mvpMapPoints[i]);
            for(int i=0;i<mCurrentFrame.N;++i)
                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i] &&
                   counted.insert(mCurrentFrame.mvpMapPoints[i]).second) {
                    mCurrentFrame.mvpMapPoints[i]->IncreaseVisible();
                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                }
            mpLocalMapper->mnMatchesInliers=mnMatchesInliers;
        }
        return accepted;
    };
    if(validate(nullptr,nullptr)) return true;
    restore();
    Map* map=mpAtlas->GetCurrentMap();
    if(!mbReliableFrameRecoveryEnabled || !mbReliableFrameCacheEnabled || !map || map->IsBad() ||
       mpReliableFlowMap!=map || mnReliableFlowMapId!=map->GetId() ||
       mnReliableFlowBigChange!=map->GetLastBigChangeIdx() ||
       mnReliableFlowGraphSequence!=map->mnMarkerGraphSequence ||
       mbReliableFlowMetric!=map->mbMetric || mReliableFlowScale!=map->mMetricScale ||
       mReliableFlowImage.empty() ||
       mReliableFlowFrame.mnId==mLastFrame.mnId) return false;
    const double age=mCurrentFrame.mTimeStamp-mReliableFlowFrame.mTimeStamp;
    if(!std::isfinite(age) || age<=0 || age>.25) return false;
    // Resolve only current map members, discarding culled/replaced IDs rather
    // than dereferencing old points or giving replacements old descriptors.
    std::map<unsigned long,MapPoint*> live;
    for(MapPoint* point:map->GetAllMapPoints())
        if(point && !point->isBad()) live.emplace(point->mnId,point);
    Frame source=mReliableFlowFrame;
    for(int i=0;i<source.N;++i) {
        const auto found=live.find(mvReliableFlowPointIds[i]);
        if(found!=live.end()) source.mvpMapPoints[i]=found->second;
    }
    if(validate(&source,&mReliableFlowImage)) return true;
    restore();
    return false;
    } catch(const cv::Exception& error) {
        restore();
        std::cout<<"FLOW_RECOVERY_LOCAL frame="<<mCurrentFrame.mnId
                 <<" accepted=0 reason=opencv_exception code="<<error.code<<std::endl;
        return false;
    }
}

bool Tracking::TrackWithTemporalFlow(bool recovery,const Frame* sourceFrame,const cv::Mat* sourceImage)
{
    const Frame& previous=sourceFrame?*sourceFrame:mLastFrame;
    const cv::Mat& previousImage=sourceImage?*sourceImage:mTemporalPreviousImage;
    const double age=mCurrentFrame.mTimeStamp-previous.mTimeStamp;
    const bool reliable=sourceFrame!=nullptr;
    if((!recovery && !mbTemporalFlowEnabled) || mSensor!=System::MONOCULAR ||
       (mState!=OK && !(recovery && mState==RECENTLY_LOST)) ||
       previousImage.empty() || previousImage.size()!=mImGray.size() ||
       (!reliable && mTemporalPreviousTime!=previous.mTimeStamp) ||
       !std::isfinite(age) || age<=0 || age>(reliable?.25:.075) ||
       mCurrentFrame.mnId<=mnLastRelocFrameId+2) return false;
    const auto started=std::chrono::steady_clock::now();
    const Frame original=mCurrentFrame;
    const auto predicted=mCurrentFrame.GetPose();
    std::vector<cv::Point2f> source;
    std::vector<cv::Point3f> cameraPoints;
    std::vector<int> indices;
    std::set<MapPoint*> seen;
    for(int i=0;i<previous.N;++i) {
        MapPoint* p=previous.mvpMapPoints[i];
        if(!p || previous.mvbOutlier[i] || p->isBad() || p->Observations()<2 ||
           p->GetMap()!=mpAtlas->GetCurrentMap() || !seen.insert(p).second) continue;
        const Eigen::Vector3f xyz=predicted*p->GetWorldPos();
        if(!xyz.allFinite() || xyz.z()<=0) continue;
        source.push_back(previous.mvKeys[i].pt);
        cameraPoints.emplace_back(xyz.x(),xyz.y(),xyz.z());indices.push_back(i);
    }
    if(source.size()<(recovery ? 30 : 100)) {
        if(recovery) std::cout<<"TEMPORAL_FLOW frame="<<mCurrentFrame.mnId
            <<" source_frame="<<previous.mnId<<" age_s="<<age<<" reliable="<<reliable
            <<" seeds="<<source.size()<<" accepted=0 reason=insufficient_seeds"<<std::endl;
        return false;
    }
    std::vector<cv::Point2f> target;
    cv::projectPoints(cameraPoints,cv::Vec3d(0,0,0),cv::Vec3d(0,0,0),
                      mCurrentFrame.mK,mCurrentFrame.mDistCoef,target);
    // Keep a wildly inaccurate motion prior away from LK's image accesses.
    for(size_t i=0;i<target.size();++i)
        if(!std::isfinite(target[i].x) || !std::isfinite(target[i].y) ||
           target[i].x<0 || target[i].x>=mImGray.cols ||
           target[i].y<0 || target[i].y>=mImGray.rows) target[i]=source[i];
    const auto valid=TemporalFlow(previousImage,mImGray,source,target,
                                  mpORBextractorLeft->mAllowedMask);
    std::vector<cv::Point2f> undistorted;
    cv::undistortPoints(target,undistorted,mCurrentFrame.mK,mCurrentFrame.mDistCoef,
                        cv::noArray(),mCurrentFrame.mK);
    struct Pair {int distance;int current;int previous;};
    std::vector<Pair> pairs;
    for(size_t i=0;i<source.size();++i) {
        if(!valid[i]) continue;
        const int old=indices[i],level=previous.mvKeys[old].octave;
        const auto candidates=mCurrentFrame.GetFeaturesInArea(undistorted[i].x,
            undistorted[i].y,std::min(6.f,3.f*previous.mvScaleFactors[level]),
            std::max(0,level-1),std::min(mCurrentFrame.mnScaleLevels-1,level+1));
        int best=-1,distance=65,second=257;
        for(size_t j:candidates) {
            const int d=ORBmatcher::DescriptorDistance(previous.mDescriptors.row(old),
                                                        mCurrentFrame.mDescriptors.row(j));
            if(d<distance) {second=distance;distance=d;best=j;}
            else if(d<second) second=d;
        }
        if(best>=0 && distance<.8*second) pairs.push_back({distance,best,old});
    }
    std::sort(pairs.begin(),pairs.end(),[](const Pair& a,const Pair& b) {
        return std::tie(a.distance,a.current,a.previous)<std::tie(b.distance,b.current,b.previous);
    });
    std::fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),nullptr);
    std::fill(mCurrentFrame.mvbOutlier.begin(),mCurrentFrame.mvbOutlier.end(),false);
    int matches=0;std::set<int> cells;
    for(const auto& p:pairs) if(!mCurrentFrame.mvpMapPoints[p.current]) {
        mCurrentFrame.mvpMapPoints[p.current]=previous.mvpMapPoints[p.previous];
        ++matches;
        const auto xy=mCurrentFrame.mvKeys[p.current].pt;
        cells.insert(std::min(5,int(xy.y*6/mImGray.rows))*8+std::min(7,int(xy.x*8/mImGray.cols)));
    }
    bool accepted=matches>=std::max(recovery ? 30 : 80,int(source.size()*(recovery ? .3 : .6))) && cells.size()>=8;
    // A 20--29 point hypothesis is only a seed for the unchanged >=30-inlier
    // local-map validation in TryTemporalFlowRecovery. Never publish it here.
    const char* sparseOption=std::getenv("ORB_SLAM3_SPARSE_RECOVERY_SEED");
    const bool sparseSeed=recovery && mbReliableFrameRecoveryEnabled && sparseOption &&
        std::string(sparseOption)=="1" && !accepted && matches>=20 && matches<30 && cells.size()>=8;
    if(sparseSeed) accepted=true;
    // A provisional pose only: TryTemporalFlowRecovery still requires at
    // least 30 native local-map inliers before committing the recovered frame.
    const int minimumSeedInliers=sparseSeed?20:30;
    int inliers=0,pnpInliers=0;
    std::string reason=accepted?"pose_optimization":"matches_or_spatial_coverage";
    if(accepted && recovery) {
        std::vector<cv::Point3f> world;
        std::vector<cv::Point2f> pixels;
        for(int i=0;i<mCurrentFrame.N;++i) if(mCurrentFrame.mvpMapPoints[i]) {
            const auto p=mCurrentFrame.mvpMapPoints[i]->GetWorldPos();
            world.emplace_back(p.x(),p.y(),p.z());
            pixels.push_back(mCurrentFrame.mvKeysUn[i].pt);
        }
        cv::Mat r,t,ids;
        accepted=cv::solvePnPRansac(world,pixels,mCurrentFrame.mK,cv::Mat(),r,t,false,
                                   200,3.,.999,ids,cv::SOLVEPNP_EPNP) && ids.total()>=std::size_t(minimumSeedInliers);
        pnpInliers=int(ids.total());
        reason=accepted?"pose_optimization":"pnp_ransac";
        if(accepted) {
            cv::Mat R,T=cv::Mat::eye(4,4,CV_32F);
            cv::Rodrigues(r,R); R.convertTo(T(cv::Rect(0,0,3,3)),CV_32F);
            t.convertTo(T(cv::Rect(3,0,1,3)),CV_32F);
            mCurrentFrame.SetPose(Converter::toSophus(T));
        }
    }
    if(accepted) {
        Optimizer::PoseOptimization(&mCurrentFrame);
        for(int i=0;i<mCurrentFrame.N;++i) {
            if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i]) ++inliers;
            else mCurrentFrame.mvpMapPoints[i]=nullptr;
        }
        accepted=inliers>=std::max(recovery ? minimumSeedInliers : 60,int(matches*(recovery ? .6 : .8)));
        if(accepted && (reliable || sparseSeed)) {
            accepted=TemporalRecoveryCorrectionBound(mCurrentFrame,predicted);
            if(!accepted) reason="relative_correction";
        }
    }
    if(!accepted) {
        mCurrentFrame=original;
    }
    const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
    std::cout<<"TEMPORAL_FLOW frame="<<mCurrentFrame.mnId<<" source_frame="<<previous.mnId
             <<" age_s="<<age<<" reliable="<<reliable<<" seeds="<<source.size()
             <<" lk_valid="<<std::count(valid.begin(),valid.end(),1)<<" cells="<<cells.size()
             <<" matches="<<matches<<" sparse_seed="<<sparseSeed<<" pnp_inliers="<<pnpInliers<<" inliers="<<inliers
             <<" accepted="<<accepted<<" reason="<<(accepted?"accepted":reason)<<" ms="<<ms<<std::endl;
    return accepted;
}

bool Tracking::TrackWithMotionModel()
{
    ORBmatcher matcher(0.9,true);

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points if in Localization Mode
    UpdateLastFrame();

    if (mpAtlas->isImuInitialized() && (mCurrentFrame.mnId>mnLastRelocFrameId+mnFramesToResetIMU))
    {
        // Predict state with IMU if it is initialized and it doesnt need reset
        PredictStateIMU();
        return true;
    }
    else
    {
        mCurrentFrame.SetPose(mVelocity * mLastFrame.GetPose());
    }

    // Experimental matching accelerator only. Full ORB features, local-map
    // validation, keyframe creation and all marker/BA/loop policies remain.
    if(TrackWithTemporalFlow()) return true;

    fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));

    // Project points seen in previous frame
    int th;

    if(mSensor==System::STEREO)
        th=7;
    else
        th=15;

    int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR);

    // If few matches, uses a wider window search
    if(nmatches<20)
    {
        Verbose::PrintMess("Not enough matches, wider window search!!", Verbose::VERBOSITY_NORMAL);
        fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));

        nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR);
        Verbose::PrintMess("Matches with wider search: " + to_string(nmatches), Verbose::VERBOSITY_NORMAL);

    }

    if(nmatches<20)
    {
        Verbose::PrintMess("Not enough matches!!", Verbose::VERBOSITY_NORMAL);
        if(mSensor==System::MONOCULAR && mpAtlas->GetCurrentMap()->mbMetric)
        {
            // A low-frame-rate head turn can move a correctly predicted map
            // point beyond the ordinary 30 px window.  One final bounded
            // 60 px search still requires real map-point associations and
            // downstream robust pose optimization; it is not 2-D VO.
            fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),
                 static_cast<MapPoint*>(NULL));
            nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,4*th,true);
        }
        const int minimumMotionMatches =
            (mSensor==System::MONOCULAR && mpAtlas->GetCurrentMap()->mbMetric)
            ? 15 : 20;
        if(nmatches<minimumMotionMatches)
        {
            if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
                return true;
            else
                return false;
        }
    }

    // Optimize frame pose with all matches
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                if(i < mCurrentFrame.Nleft){
                    pMP->mbTrackInView = false;
                }
                else{
                    pMP->mbTrackInViewR = false;
                }
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }

    if(mbOnlyTracking)
    {
        mbVO = nmatchesMap<10;
        return nmatches>20;
    }

    if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
        return true;
    else
        return nmatchesMap>=10;
}

bool Tracking::TrackLocalMap(bool recoveryTrial)
{

    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.
    if(!recoveryTrial) mTrackedFr++;

    if(!recoveryTrial) UpdateLocalMap();
    SearchLocalPoints(recoveryTrial);

    // TOO check outliers before PO
    int aux1 = 0, aux2=0;
    for(int i=0; i<mCurrentFrame.N; i++)
        if( mCurrentFrame.mvpMapPoints[i])
        {
            aux1++;
            if(mCurrentFrame.mvbOutlier[i])
                aux2++;
        }

    int inliers;
    if (!mpAtlas->isImuInitialized())
        Optimizer::PoseOptimization(&mCurrentFrame);
    else
    {
        if(mCurrentFrame.mnId<=mnLastRelocFrameId+mnFramesToResetIMU)
        {
            Verbose::PrintMess("TLM: PoseOptimization ", Verbose::VERBOSITY_DEBUG);
            Optimizer::PoseOptimization(&mCurrentFrame);
        }
        else
        {
            // if(!mbMapUpdated && mState == OK) //  && (mnMatchesInliers>30))
            if(!mbMapUpdated) //  && (mnMatchesInliers>30))
            {
                Verbose::PrintMess("TLM: PoseInertialOptimizationLastFrame ", Verbose::VERBOSITY_DEBUG);
                inliers = Optimizer::PoseInertialOptimizationLastFrame(&mCurrentFrame); // , !mpLastKeyFrame->GetMap()->GetIniertialBA1());
            }
            else
            {
                Verbose::PrintMess("TLM: PoseInertialOptimizationLastKeyFrame ", Verbose::VERBOSITY_DEBUG);
                inliers = Optimizer::PoseInertialOptimizationLastKeyFrame(&mCurrentFrame); // , !mpLastKeyFrame->GetMap()->GetIniertialBA1());
            }
        }
    }

    aux1 = 0, aux2 = 0;
    for(int i=0; i<mCurrentFrame.N; i++)
        if( mCurrentFrame.mvpMapPoints[i])
        {
            aux1++;
            if(mCurrentFrame.mvbOutlier[i])
                aux2++;
        }

    mnMatchesInliers = 0;

    // Update MapPoints Statistics
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(!mCurrentFrame.mvbOutlier[i])
            {
                if(!recoveryTrial) mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                if(!mbOnlyTracking)
                {
                    if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                        mnMatchesInliers++;
                }
                else
                    mnMatchesInliers++;
            }
            else if(mSensor==System::STEREO)
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
        }
    }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    if(!recoveryTrial) mpLocalMapper->mnMatchesInliers=mnMatchesInliers;
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers<50)
        return false;

    if((mnMatchesInliers>10)&&(mState==RECENTLY_LOST))
        return true;


    if (mSensor == System::IMU_MONOCULAR)
    {
        if((mnMatchesInliers<15 && mpAtlas->isImuInitialized())||(mnMatchesInliers<50 && !mpAtlas->isImuInitialized()))
        {
            return false;
        }
        else
            return true;
    }
    else if (mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
    {
        if(mnMatchesInliers<15)
        {
            return false;
        }
        else
            return true;
    }
    else
    {
        // Metric scale supplies no additional current-frame visual evidence.
        // Keep the monocular visual floor even in a metric map; a failed
        // visual solve can still use the separately validated marker path.
        const int minimumInliers = 30;
        if(mnMatchesInliers<minimumInliers)
            return false;
        else
            return true;
    }
}

bool Tracking::NeedNewKeyFrame()
{
    if((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && !mpAtlas->GetCurrentMap()->isImuInitialized())
    {
        if (mSensor == System::IMU_MONOCULAR && (mCurrentFrame.mTimeStamp-mpLastKeyFrame->mTimeStamp)>=0.25)
            return true;
        else if ((mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && (mCurrentFrame.mTimeStamp-mpLastKeyFrame->mTimeStamp)>=0.25)
            return true;
        else
            return false;
    }

    if(mbOnlyTracking)
        return false;

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested()) {
        /*if(mSensor == System::MONOCULAR)
        {
            std::cout << "NeedNewKeyFrame: localmap stopped" << std::endl;
        }*/
        return false;
    }

    const int nKFs = mpAtlas->KeyFramesInMap();

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    const bool markerEvent=HasMarkerKeyFrameEvent();
    if(!markerEvent && mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && nKFs>mMaxFrames)
    {
        return false;
    }

    // Tracked MapPoints in the reference keyframe
    int nMinObs = 3;
    if(nKFs<=2)
        nMinObs=2;
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);
    // Several marker-only information frames can coexist with just TWO
    // background keyframes. Their freshly triangulated points are not yet
    // observed three times; do not misread this as a fully tracked reference.
    if(nRefMatches==0 && mbTagFusionEnabled && mbTagMetricAligned)
        nRefMatches=mpReferenceKF->TrackedMapPoints(2);

    // Local Mapping accept keyframes?
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    // First observations, known anchors returning after a sustained decode
    // gap, and localization recovery preserve metric evidence. Brief decode
    // flicker is not such an event.
    if(markerEvent) {
        if(bLocalMappingIdle || mpLocalMapper->IsInitializing()) return true;
        mpLocalMapper->InterruptBA();
        return false; // The pending event is consumed only on actual insertion.
    }

    // ORB-SLAM3's monocular default permits c1b as soon as LocalMapping is
    // idle (mMinFrames is normally zero).  On high-rate offline video, an
    // arbitrary-scale weak-tracking segment can therefore insert dozens of
    // nearly identical keyframes per second and starve the mapper.  Keep the
    // ordinary geometric/coverage test below, but cap redundant proposals at
    // 10 Hz once initialization has its first three keyframes.  Marker events
    // and relocalization are handled above and are never delayed by this cap.
    if(mSensor==System::MONOCULAR && !mbTagMetricAligned && nKFs>=3 && mpLastKeyFrame) {
        const unsigned int arbitraryScaleMinimumGap =
            static_cast<unsigned int>(std::max(
                1, static_cast<int>(std::lround(0.10f * mMaxFrames))));
        if(mCurrentFrame.mnId - mnLastKeyFrameId < arbitraryScaleMinimumGap)
            return false;
    }

    if(mbTagFusionEnabled && mbTagMetricAligned &&
       mSensor == System::MONOCULAR && mpLastKeyFrame)
    {
        const unsigned int frameGap = mCurrentFrame.mnId - mnLastKeyFrameId;
        const float translationM =
            (mCurrentFrame.GetCameraCenter() -
             mpLastKeyFrame->GetCameraCenter()).norm();
        const Eigen::Matrix3f relativeRotation =
            mCurrentFrame.GetRotationInverse().transpose() *
            mpLastKeyFrame->GetPoseInverse().rotationMatrix();
        const float cosine = std::max(
            -1.0f, std::min(1.0f,
                0.5f * (relativeRotation.trace() - 1.0f)));
        const float rotationDeg = std::acos(cosine) *
            57.29577951308232f;
        int unmatchedFeatures = 0;
        int measuredInliers = 0;
        for(int index = 0; index < mCurrentFrame.N; ++index) {
            if(mCurrentFrame.mvpMapPoints[index] && !mCurrentFrame.mvbOutlier[index])
                ++measuredInliers;
            if(!mCurrentFrame.mvpMapPoints[index] &&
               !mCurrentFrame.mvbOutlier[index])
                unmatchedFeatures++;
        }
        // The tag+ORB solve may reject matches from the earlier ORB-only
        // pose. Use surviving matches, not the stale pre-fusion inlier count.
        const float trackedRatio = nRefMatches > 0
            ? static_cast<float>(measuredInliers)/nRefMatches : 1.0f;

        // Do not let the normal anti-spam interval suppress the last useful
        // bridge into a new view.  At high frame rates there may still be
        // enough geometrically verified points for a keyframe even though
        // the old reference is about to leave the image.  A shorter 100 ms
        // interval is allowed only under that explicit coverage loss.
        const bool coverageAtRisk = measuredInliers >= 15 &&
            (measuredInliers < 80 || trackedRatio < 0.35f) &&
            unmatchedFeatures > std::max(80, mCurrentFrame.N / 4);
        const unsigned int minimumFrameGap = coverageAtRisk
            ? static_cast<unsigned int>(std::max(
                1, static_cast<int>(std::lround(0.10f * mMaxFrames))))
            : static_cast<unsigned int>(mTagMinimumKeyFrameFrames);
        if(frameGap < minimumFrameGap)
        {
            mnTagKeyFramesRejected++;
            return false;
        }

        const bool metricMotion =
            translationM >= mTagMinimumTranslationM ||
            rotationDeg >= mTagMinimumRotationDeg;
        const bool featureNovelty =
            (measuredInliers > 30 || (mbHasExternalTagObservation &&
             mnTagPoseConstraintFrameId==long(mCurrentFrame.mnId) && measuredInliers >= 15)) &&
            trackedRatio < mTagMinimumTrackedRatio &&
            unmatchedFeatures > std::max(80, mCurrentFrame.N / 5);
        if(!metricMotion && !featureNovelty && !coverageAtRisk)
        {
            mnTagKeyFramesRejected++;
            return false;
        }

        if(bLocalMappingIdle || mpLocalMapper->IsInitializing())
        {
            mnTagKeyFramesAccepted++;
            return true;
        }
        mpLocalMapper->InterruptBA();
        mnTagKeyFramesRejected++;
        return false;
    }

    // Check how many "close" points are being tracked and how many could be potentially created.
    int nNonTrackedClose = 0;
    int nTrackedClose= 0;

    if(mSensor!=System::MONOCULAR && mSensor!=System::IMU_MONOCULAR)
    {
        int N = (mCurrentFrame.Nleft == -1) ? mCurrentFrame.N : mCurrentFrame.Nleft;
        for(int i =0; i<N; i++)
        {
            if(mCurrentFrame.mvDepth[i]>0 && mCurrentFrame.mvDepth[i]<mThDepth)
            {
                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                    nTrackedClose++;
                else
                    nNonTrackedClose++;

            }
        }
        //Verbose::PrintMess("[NEEDNEWKF]-> closed points: " + to_string(nTrackedClose) + "; non tracked closed points: " + to_string(nNonTrackedClose), Verbose::VERBOSITY_NORMAL);// Verbose::VERBOSITY_DEBUG);
    }

    bool bNeedToInsertClose;
    bNeedToInsertClose = (nTrackedClose<100) && (nNonTrackedClose>70);

    // Thresholds
    float thRefRatio = 0.75f;
    if(nKFs<2)
        thRefRatio = 0.4f;

    /*int nClosedPoints = nTrackedClose + nNonTrackedClose;
    const int thStereoClosedPoints = 15;
    if(nClosedPoints < thStereoClosedPoints && (mSensor==System::STEREO || mSensor==System::IMU_STEREO))
    {
        //Pseudo-monocular, there are not enough close points to be confident about the stereo observations.
        thRefRatio = 0.9f;
    }*/

    if(mSensor==System::MONOCULAR)
        thRefRatio = 0.9f;

    if(mpCamera2) thRefRatio = 0.75f;

    if(mSensor==System::IMU_MONOCULAR)
    {
        if(mnMatchesInliers>350) // Points tracked from the local map
            thRefRatio = 0.75f;
        else
            thRefRatio = 0.90f;
    }

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    const bool c1b = ((mCurrentFrame.mnId>=mnLastKeyFrameId+mMinFrames) && bLocalMappingIdle); //mpLocalMapper->KeyframesInQueue() < 2);
    //Condition 1c: tracking is weak
    const bool c1c = mSensor!=System::MONOCULAR && mSensor!=System::IMU_MONOCULAR && mSensor!=System::IMU_STEREO && mSensor!=System::IMU_RGBD && (mnMatchesInliers<nRefMatches*0.25 || bNeedToInsertClose) ;
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    const bool c2 = (((mnMatchesInliers<nRefMatches*thRefRatio || bNeedToInsertClose)) && mnMatchesInliers>15);

    //std::cout << "NeedNewKF: c1a=" << c1a << "; c1b=" << c1b << "; c1c=" << c1c << "; c2=" << c2 << std::endl;
    // Temporal condition for Inertial cases
    bool c3 = false;
    if(mpLastKeyFrame)
    {
        if (mSensor==System::IMU_MONOCULAR)
        {
            if ((mCurrentFrame.mTimeStamp-mpLastKeyFrame->mTimeStamp)>=0.5)
                c3 = true;
        }
        else if (mSensor==System::IMU_STEREO || mSensor == System::IMU_RGBD)
        {
            if ((mCurrentFrame.mTimeStamp-mpLastKeyFrame->mTimeStamp)>=0.5)
                c3 = true;
        }
    }

    bool c4 = false;
    if ((((mnMatchesInliers<75) && (mnMatchesInliers>15)) || mState==RECENTLY_LOST) && (mSensor == System::IMU_MONOCULAR)) // MODIFICATION_2, originally ((((mnMatchesInliers<75) && (mnMatchesInliers>15)) || mState==RECENTLY_LOST) && ((mSensor == System::IMU_MONOCULAR)))
        c4=true;
    else
        c4=false;

    if(((c1a||c1b||c1c) && c2)||c3 ||c4)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        if(bLocalMappingIdle || mpLocalMapper->IsInitializing())
        {
            return true;
        }
        else
        {
            mpLocalMapper->InterruptBA();
            if(mSensor!=System::MONOCULAR  && mSensor!=System::IMU_MONOCULAR)
            {
                if(mpLocalMapper->KeyframesInQueue()<3)
                    return true;
                else
                    return false;
            }
            else
            {
                //std::cout << "NeedNewKeyFrame: localmap is busy" << std::endl;
                return false;
            }
        }
    }
    else
        return false;
}

void Tracking::CreateNewKeyFrame()
{
    if(mpLocalMapper->IsInitializing() && !mpAtlas->isImuInitialized())
        return;

    if(!mpLocalMapper->SetNotStop(true))
        return;

    KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpAtlas->GetCurrentMap(),mpKeyFrameDB);
    AttachCurrentTagObservation(pKF);

    if(mpAtlas->isImuInitialized()) //  || mpLocalMapper->IsInitializing())
        pKF->bImu = true;

    pKF->SetNewBias(mCurrentFrame.mImuBias);
    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

    if(mpLastKeyFrame)
    {
        pKF->mPrevKF = mpLastKeyFrame;
        mpLastKeyFrame->mNextKF = pKF;
    }
    else
        Verbose::PrintMess("No last KF in KF creation!!", Verbose::VERBOSITY_NORMAL);

    // Reset preintegration from last KF (Create new object)
    if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
    {
        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(pKF->GetImuBias(),pKF->mImuCalib);
    }

    if(mSensor!=System::MONOCULAR && mSensor != System::IMU_MONOCULAR) // TODO check if incluide imu_stereo
    {
        mCurrentFrame.UpdatePoseMatrices();
        // cout << "create new MPs" << endl;
        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
        int maxPoint = 100;
        if(mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
            maxPoint = 100;

        vector<pair<float,int> > vDepthIdx;
        int N = (mCurrentFrame.Nleft != -1) ? mCurrentFrame.Nleft : mCurrentFrame.N;
        vDepthIdx.reserve(mCurrentFrame.N);
        for(int i=0; i<N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                vDepthIdx.push_back(make_pair(z,i));
            }
        }

        if(!vDepthIdx.empty())
        {
            sort(vDepthIdx.begin(),vDepthIdx.end());

            int nPoints = 0;
            for(size_t j=0; j<vDepthIdx.size();j++)
            {
                int i = vDepthIdx[j].second;

                bool bCreateNew = false;

                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(!pMP)
                    bCreateNew = true;
                else if(pMP->Observations()<1)
                {
                    bCreateNew = true;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                }

                if(bCreateNew)
                {
                    Eigen::Vector3f x3D;

                    if(mCurrentFrame.Nleft == -1){
                        mCurrentFrame.UnprojectStereo(i, x3D);
                    }
                    else{
                        x3D = mCurrentFrame.UnprojectStereoFishEye(i);
                    }

                    MapPoint* pNewMP = new MapPoint(x3D,pKF,mpAtlas->GetCurrentMap());
                    pNewMP->AddObservation(pKF,i);

                    //Check if it is a stereo observation in order to not
                    //duplicate mappoints
                    if(mCurrentFrame.Nleft != -1 && mCurrentFrame.mvLeftToRightMatch[i] >= 0){
                        mCurrentFrame.mvpMapPoints[mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]]=pNewMP;
                        pNewMP->AddObservation(pKF,mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]);
                        pKF->AddMapPoint(pNewMP,mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]);
                    }

                    pKF->AddMapPoint(pNewMP,i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpAtlas->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                    nPoints++;
                }
                else
                {
                    nPoints++;
                }

                if(vDepthIdx[j].first>mThDepth && nPoints>maxPoint)
                {
                    break;
                }
            }
            //Verbose::PrintMess("new mps for stereo KF: " + to_string(nPoints), Verbose::VERBOSITY_NORMAL);
        }
    }


    mpLocalMapper->InsertKeyFrame(pKF);

    mpLocalMapper->SetNotStop(false);

    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;
}

void Tracking::SearchLocalPoints(bool recoveryTrial)
{
    std::set<MapPoint*> trialAssociated;
    if(recoveryTrial)
        trialAssociated.insert(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end());
    // Do not search map points already matched
    for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(), vend=mCurrentFrame.mvpMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP)
        {
            if(pMP->isBad())
            {
                *vit = static_cast<MapPoint*>(NULL);
            }
            else
            {
                if(!recoveryTrial) pMP->IncreaseVisible();
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                pMP->mbTrackInView = false;
                pMP->mbTrackInViewR = false;
            }
        }
    }

    int nToMatch=0;

    // Project points in frame and check its visibility
    for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;

        if(recoveryTrial ? trialAssociated.count(pMP)>0 : pMP->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        if(pMP->isBad())
            continue;
        // Project (this fills MapPoint variables for matching)
        if(mCurrentFrame.isInFrustum(pMP,0.5))
        {
            if(!recoveryTrial) pMP->IncreaseVisible();
            nToMatch++;
        }
        if(pMP->mbTrackInView)
        {
            mCurrentFrame.mmProjectPoints[pMP->mnId] = cv::Point2f(pMP->mTrackProjX, pMP->mTrackProjY);
        }
    }

    if(nToMatch>0)
    {
        ORBmatcher matcher(0.8);
        int th = 1;
        if(mSensor==System::RGBD || mSensor==System::IMU_RGBD)
            th=3;
        if(mpAtlas->isImuInitialized())
        {
            if(mpAtlas->GetCurrentMap()->GetIniertialBA2())
                th=2;
            else
                th=6;
        }
        else if(!mpAtlas->isImuInitialized() && (mSensor==System::IMU_MONOCULAR || mSensor==System::IMU_STEREO || mSensor == System::IMU_RGBD))
        {
            th=10;
        }

        // If the camera has been relocalised recently, perform a coarser search
        if(mCurrentFrame.mnId<mnLastRelocFrameId+2)
            th=5;

        if(mState==LOST || mState==RECENTLY_LOST) // Lost for less than 1 second
            th=15; // 15

        int matches = matcher.SearchByProjection(mCurrentFrame, mvpLocalMapPoints, th, mpLocalMapper->mbFarPoints, mpLocalMapper->mThFarPoints);
    }
}

void Tracking::UpdateLocalMap()
{
    // This is for visualization
    mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

    // Update
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
}

void Tracking::UpdateLocalPoints()
{
    mvpLocalMapPoints.clear();

    int count_pts = 0;

    for(vector<KeyFrame*>::const_reverse_iterator itKF=mvpLocalKeyFrames.rbegin(), itEndKF=mvpLocalKeyFrames.rend(); itKF!=itEndKF; ++itKF)
    {
        KeyFrame* pKF = *itKF;
        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

        for(vector<MapPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
        {

            MapPoint* pMP = *itMP;
            if(!pMP)
                continue;
            if(pMP->mnTrackReferenceForFrame==mCurrentFrame.mnId)
                continue;
            if(!pMP->isBad())
            {
                count_pts++;
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame=mCurrentFrame.mnId;
            }
        }
    }
}


void Tracking::UpdateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    map<KeyFrame*,int> keyframeCounter;
    if(!mpAtlas->isImuInitialized() || (mCurrentFrame.mnId<mnLastRelocFrameId+2))
    {
        for(int i=0; i<mCurrentFrame.N; i++)
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    const map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();
                    for(map<KeyFrame*,tuple<int,int>>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                        keyframeCounter[it->first]++;
                }
                else
                {
                    mCurrentFrame.mvpMapPoints[i]=NULL;
                }
            }
        }
    }
    else
    {
        for(int i=0; i<mLastFrame.N; i++)
        {
            // Using lastframe since current frame has not matches yet
            if(mLastFrame.mvpMapPoints[i])
            {
                MapPoint* pMP = mLastFrame.mvpMapPoints[i];
                if(!pMP)
                    continue;
                if(!pMP->isBad())
                {
                    const map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();
                    for(map<KeyFrame*,tuple<int,int>>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                        keyframeCounter[it->first]++;
                }
                else
                {
                    // MODIFICATION
                    mLastFrame.mvpMapPoints[i]=NULL;
                }
            }
        }
    }


    int max=0;
    KeyFrame* pKFmax= static_cast<KeyFrame*>(NULL);

    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3*keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    vector<pair<KeyFrame*,int>> orderedKeyframeVotes(
        keyframeCounter.begin(),keyframeCounter.end());
    sort(orderedKeyframeVotes.begin(),orderedKeyframeVotes.end(),
         [](const pair<KeyFrame*,int>& first,const pair<KeyFrame*,int>& second) {
             return first.first->mnId<second.first->mnId;
         });
    for(const auto& vote:orderedKeyframeVotes)
    {
        KeyFrame* pKF = vote.first;

        if(pKF->isBad())
            continue;

        if(vote.second>max ||
           (vote.second==max && (!pKFmax || pKF->mnId<pKFmax->mnId)))
        {
            max=vote.second;
            pKFmax=pKF;
        }

        mvpLocalKeyFrames.push_back(pKF);
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }

    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if(mvpLocalKeyFrames.size()>80) // 80
            break;

        KeyFrame* pKF = *itKF;

        const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);


        for(vector<KeyFrame*>::const_iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++)
        {
            KeyFrame* pNeighKF = *itNeighKF;
            if(!pNeighKF->isBad())
            {
                if(pNeighKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        const set<KeyFrame*> spChilds = pKF->GetChilds();
        vector<KeyFrame*> orderedChildren(spChilds.begin(),spChilds.end());
        sort(orderedChildren.begin(),orderedChildren.end(),KeyFrame::lId);
        for(KeyFrame* pChildKF:orderedChildren)
        {
            if(!pChildKF->isBad())
            {
                if(pChildKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        KeyFrame* pParent = pKF->GetParent();
        if(pParent)
        {
            if(pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                break;
            }
        }
    }

    // Add 10 last temporal KFs (mainly for IMU)
    if((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) &&mvpLocalKeyFrames.size()<80)
    {
        KeyFrame* tempKeyFrame = mCurrentFrame.mpLastKeyFrame;

        const int Nd = 20;
        for(int i=0; i<Nd; i++){
            if (!tempKeyFrame)
                break;
            if(tempKeyFrame->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(tempKeyFrame);
                tempKeyFrame->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                tempKeyFrame=tempKeyFrame->mPrevKF;
            }
        }
    }

    if(pKFmax)
    {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}

bool Tracking::Relocalization()
{
    Verbose::PrintMess("Starting relocalization", Verbose::VERBOSITY_NORMAL);
    // Compute Bag of Words Vector
    mCurrentFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(
        &mCurrentFrame, mpAtlas->GetCurrentMap());

    if(vpCandidateKFs.empty()) {
        Verbose::PrintMess("There are not candidates", Verbose::VERBOSITY_NORMAL);
        return false;
    }

    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.75,true);

    vector<MLPnPsolver*> vpMLPnPsolvers;
    vpMLPnPsolvers.resize(nKFs);

    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates=0;

    for(int i=0; i<nKFs; i++)
    {
        KeyFrame* pKF = vpCandidateKFs[i];
        if(pKF->isBad())
            vbDiscarded[i] = true;
        else
        {
            int nmatches = matcher.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]);
            if(nmatches<15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                MLPnPsolver* pSolver = new MLPnPsolver(mCurrentFrame,vvpMapPointMatches[i]);
                pSolver->SetRansacParameters(0.99,10,300,6,0.5,5.991);  //This solver needs at least 6 points
                vpMLPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;
    ORBmatcher matcher2(0.9,true);

    while(nCandidates>0 && !bMatch)
    {
        for(int i=0; i<nKFs; i++)
        {
            if(vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            MLPnPsolver* pSolver = vpMLPnPsolvers[i];
            Eigen::Matrix4f eigTcw;
            bool bTcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers, eigTcw);

            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if(bTcw)
            {
                Sophus::SE3f Tcw(eigTcw);
                mCurrentFrame.SetPose(Tcw);
                // Tcw.copyTo(mCurrentFrame.mTcw);

                set<MapPoint*> sFound;

                const int np = vbInliers.size();

                for(int j=0; j<np; j++)
                {
                    if(vbInliers[j])
                    {
                        mCurrentFrame.mvpMapPoints[j]=vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j]=NULL;
                }

                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                if(nGood<10)
                    continue;

                for(int io =0; io<mCurrentFrame.N; io++)
                    if(mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL);

                // If few inliers, search by projection in a coarse window and optimize again
                if(nGood<50)
                {
                    int nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,10,100);

                    if(nadditional+nGood>=50)
                    {
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if(nGood>30 && nGood<50)
                        {
                            sFound.clear();
                            for(int ip =0; ip<mCurrentFrame.N; ip++)
                                if(mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,3,64);

                            // Final optimization
                            if(nGood+nadditional>=50)
                            {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                for(int io =0; io<mCurrentFrame.N; io++)
                                    if(mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io]=NULL;
                            }
                        }
                    }
                }


                // If the pose is supported by enough inliers stop ransacs and continue
                if(nGood>=50)
                {
                    bMatch = true;
                    break;
                }
            }
        }
    }

    if(!bMatch)
    {
        return false;
    }
    else
    {
        mnLastRelocFrameId = mCurrentFrame.mnId;
        cout << "Relocalized!!" << endl;
        return true;
    }

}

void Tracking::Reset(bool bLocMap)
{
    ClearReliableFlowFrame();
    CancelMarkerGraph();
    Verbose::PrintMess("System Reseting", Verbose::VERBOSITY_NORMAL);
    CancelPendingTagAlignment();

    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            usleep(3000);
    }

    // Reset Local Mapping
    if (!bLocMap)
    {
        Verbose::PrintMess("Reseting Local Mapper...", Verbose::VERBOSITY_NORMAL);
        mpLocalMapper->RequestReset();
        Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);
    }


    // Reset Loop Closing
    Verbose::PrintMess("Reseting Loop Closing...", Verbose::VERBOSITY_NORMAL);
    mpLoopClosing->RequestReset();
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear BoW Database
    Verbose::PrintMess("Reseting Database...", Verbose::VERBOSITY_NORMAL);
    mpKeyFrameDB->clear();
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear Map (this erase MapPoints and KeyFrames)
    mpAtlas->clearAtlas();
    mpAtlas->CreateNewMap();
    if (mSensor==System::IMU_STEREO || mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_RGBD)
        mpAtlas->SetInertialSensor();
    mnInitialFrameId = 0;

    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    mbReadyToInitializate = false;
    mbSetInit=false;
    mbHasInitialTagObservation = false;
    mbHasTagScaleReference = false;
    mbTagMetricAligned = false;
    mbTagAlignmentPending = false;
    mRecoveredTagMetricScale = 0.0f;
    mPendingTagMetricScale = 0.0f;
    mvTagScaleSamples.clear();

    mlRelativeFramePoses.clear();
    mMarkerMetricFrames.clear();
    mpMarkerSeedKF = nullptr;
    mLastReliableMarkerTime = -1.0;
    mlpReferences.clear();
    mlReferenceUnitScales.clear();
    mlFrameTimes.clear();
    mlbLost.clear();
    mCurrentFrame = Frame();
    mnLastRelocFrameId = 0;
    mLastFrame = Frame();
    mpReferenceKF = static_cast<KeyFrame*>(NULL);
    mpLastKeyFrame = static_cast<KeyFrame*>(NULL);
    mvIniMatches.clear();

    if(mpViewer)
        mpViewer->Release();

    Verbose::PrintMess("   End reseting! ", Verbose::VERBOSITY_NORMAL);
}

void Tracking::ResetActiveMap(bool bLocMap)
{
    ClearReliableFlowFrame();
    CancelMarkerGraph();
    Verbose::PrintMess("Active map Reseting", Verbose::VERBOSITY_NORMAL);
    CancelPendingTagAlignment();
    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            usleep(3000);
    }

    Map* pMap = mpAtlas->GetCurrentMap();

    if (!bLocMap)
    {
        Verbose::PrintMess("Reseting Local Mapper...", Verbose::VERBOSITY_VERY_VERBOSE);
        mpLocalMapper->RequestResetActiveMap(pMap);
        Verbose::PrintMess("done", Verbose::VERBOSITY_VERY_VERBOSE);
    }

    // Reset Loop Closing
    Verbose::PrintMess("Reseting Loop Closing...", Verbose::VERBOSITY_NORMAL);
    mpLoopClosing->RequestResetActiveMap(pMap);
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear BoW Database
    Verbose::PrintMess("Reseting Database", Verbose::VERBOSITY_NORMAL);
    mpKeyFrameDB->clearMap(pMap); // Only clear the active map references
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear Map (this erase MapPoints and KeyFrames)
    mpAtlas->clearMap();


    //KeyFrame::nNextId = mpAtlas->GetLastInitKFid();
    //Frame::nNextId = mnLastInitFrameId;
    mnLastInitFrameId = Frame::nNextId;
    //mnLastRelocFrameId = mnLastInitFrameId;
    mState = NO_IMAGES_YET; //NOT_INITIALIZED;

    mbReadyToInitializate = false;
    mbHasInitialTagObservation = false;
    mbHasTagScaleReference = false;
    mbTagMetricAligned = false;
    mbTagAlignmentPending = false;
    mRecoveredTagMetricScale = 0.0f;
    mPendingTagMetricScale = 0.0f;
    mvTagScaleSamples.clear();

    list<bool> lbLost;
    mpMarkerSeedKF = nullptr;
    mLastReliableMarkerTime = -1.0;
    // lbLost.reserve(mlbLost.size());
    unsigned int index = mnFirstFrameId;
    cout << "mnFirstFrameId = " << mnFirstFrameId << endl;
    for(Map* pMap : mpAtlas->GetAllMaps())
    {
        if(pMap->GetAllKeyFrames().size() > 0)
        {
            if(index > pMap->GetLowerKFID())
                index = pMap->GetLowerKFID();
        }
    }

    //cout << "First Frame id: " << index << endl;
    int num_lost = 0;
    cout << "mnInitialFrameId = " << mnInitialFrameId << endl;

    for(list<bool>::iterator ilbL = mlbLost.begin(); ilbL != mlbLost.end(); ilbL++)
    {
        if(index < mnInitialFrameId)
            lbLost.push_back(*ilbL);
        else
        {
            lbLost.push_back(true);
            num_lost += 1;
        }

        index++;
    }
    cout << num_lost << " Frames set to lost" << endl;

    mlbLost = lbLost;

    mnInitialFrameId = mCurrentFrame.mnId;
    mnLastRelocFrameId = mCurrentFrame.mnId;

    mCurrentFrame = Frame();
    mLastFrame = Frame();
    mpReferenceKF = static_cast<KeyFrame*>(NULL);
    mpLastKeyFrame = static_cast<KeyFrame*>(NULL);
    mvIniMatches.clear();

    mbVelocity = false;

    if(mpViewer)
        mpViewer->Release();

    Verbose::PrintMess("   End reseting! ", Verbose::VERBOSITY_NORMAL);
}

vector<MapPoint*> Tracking::GetLocalMapMPS()
{
    return mvpLocalMapPoints;
}

void Tracking::ChangeCalibration(const string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    mK_.setIdentity();
    mK_(0,0) = fx;
    mK_(1,1) = fy;
    mK_(0,2) = cx;
    mK_(1,2) = cy;

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}

void Tracking::UpdateFrameIMU(const float s, const IMU::Bias &b, KeyFrame* pCurrentKeyFrame)
{
    Map * pMap = pCurrentKeyFrame->GetMap();
    unsigned int index = mnFirstFrameId;
    list<ORB_SLAM3::KeyFrame*>::iterator lRit = mlpReferences.begin();
    list<bool>::iterator lbL = mlbLost.begin();
    for(auto lit=mlRelativeFramePoses.begin(),lend=mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lbL++)
    {
        if(*lbL)
            continue;

        KeyFrame* pKF = *lRit;

        while(pKF->isBad())
        {
            pKF = pKF->GetParent();
        }

        if(pKF->GetMap() == pMap)
        {
            (*lit).translation() *= s;
        }
    }

    mLastBias = b;

    mpLastKeyFrame = pCurrentKeyFrame;

    mLastFrame.SetNewBias(mLastBias);
    mCurrentFrame.SetNewBias(mLastBias);

    while(!mCurrentFrame.imuIsPreintegrated())
    {
        usleep(500);
    }


    if(mLastFrame.mnId == mLastFrame.mpLastKeyFrame->mnFrameId)
    {
        mLastFrame.SetImuPoseVelocity(mLastFrame.mpLastKeyFrame->GetImuRotation(),
                                      mLastFrame.mpLastKeyFrame->GetImuPosition(),
                                      mLastFrame.mpLastKeyFrame->GetVelocity());
    }
    else
    {
        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);
        const Eigen::Vector3f twb1 = mLastFrame.mpLastKeyFrame->GetImuPosition();
        const Eigen::Matrix3f Rwb1 = mLastFrame.mpLastKeyFrame->GetImuRotation();
        const Eigen::Vector3f Vwb1 = mLastFrame.mpLastKeyFrame->GetVelocity();
        float t12 = mLastFrame.mpImuPreintegrated->dT;

        mLastFrame.SetImuPoseVelocity(IMU::NormalizeRotation(Rwb1*mLastFrame.mpImuPreintegrated->GetUpdatedDeltaRotation()),
                                      twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*mLastFrame.mpImuPreintegrated->GetUpdatedDeltaPosition(),
                                      Vwb1 + Gz*t12 + Rwb1*mLastFrame.mpImuPreintegrated->GetUpdatedDeltaVelocity());
    }

    if (mCurrentFrame.mpImuPreintegrated)
    {
        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);

        const Eigen::Vector3f twb1 = mCurrentFrame.mpLastKeyFrame->GetImuPosition();
        const Eigen::Matrix3f Rwb1 = mCurrentFrame.mpLastKeyFrame->GetImuRotation();
        const Eigen::Vector3f Vwb1 = mCurrentFrame.mpLastKeyFrame->GetVelocity();
        float t12 = mCurrentFrame.mpImuPreintegrated->dT;

        mCurrentFrame.SetImuPoseVelocity(IMU::NormalizeRotation(Rwb1*mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaRotation()),
                                      twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaPosition(),
                                      Vwb1 + Gz*t12 + Rwb1*mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaVelocity());
    }

    mnFirstImuFrameId = mCurrentFrame.mnId;
}

void Tracking::NewDataset()
{
    mnNumDataset++;
}

int Tracking::GetNumberDataset()
{
    return mnNumDataset;
}

int Tracking::GetMatchesInliers()
{
    return mnMatchesInliers;
}

void Tracking::SaveSubTrajectory(string strNameFile_frames, string strNameFile_kf, string strFolder)
{
    mpSystem->SaveTrajectoryEuRoC(strFolder + strNameFile_frames);
    //mpSystem->SaveKeyFrameTrajectoryEuRoC(strFolder + strNameFile_kf);
}

void Tracking::SaveSubTrajectory(string strNameFile_frames, string strNameFile_kf, Map* pMap)
{
    mpSystem->SaveTrajectoryEuRoC(strNameFile_frames, pMap);
    if(!strNameFile_kf.empty())
        mpSystem->SaveKeyFrameTrajectoryEuRoC(strNameFile_kf, pMap);
}

float Tracking::GetImageScale()
{
    return mImageScale;
}

#ifdef REGISTER_LOOP
void Tracking::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = true;
}

bool Tracking::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(mbStopRequested && !mbNotStop)
    {
        mbStopped = true;
        cout << "Tracking STOP" << endl;
        return true;
    }

    return false;
}

bool Tracking::stopRequested()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}

bool Tracking::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

void Tracking::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopped = false;
    mbStopRequested = false;
}
#endif

} //namespace ORB_SLAM
