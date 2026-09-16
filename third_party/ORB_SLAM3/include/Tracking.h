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


#ifndef TRACKING_H
#define TRACKING_H

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "Viewer.h"
#include "FrameDrawer.h"
#include "Atlas.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "Frame.h"
#include "ORBVocabulary.h"
#include "KeyFrameDatabase.h"
#include "ORBextractor.h"
#include "MapDrawer.h"
#include "System.h"
#include "ImuTypes.h"
#include "Settings.h"

#include "GeometricCamera.h"

#include <mutex>
#include <memory>
#include <unordered_set>
#include <Eigen/StdVector>

namespace ORB_SLAM3
{

class Viewer;
class FrameDrawer;
class Atlas;
class LocalMapping;
class LoopClosing;
class System;
class Settings;
class MarkerGraphCoordinator;

class Tracking
{  
    friend class MarkerGraphCoordinator;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Tracking(System* pSys, ORBVocabulary* pVoc, FrameDrawer* pFrameDrawer, MapDrawer* pMapDrawer, Atlas* pAtlas,
             KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor, Settings* settings, const string &_nameSeq=std::string());

    ~Tracking();

    // Parse the config file
    bool ParseCamParamFile(cv::FileStorage &fSettings);
    bool ParseORBParamFile(cv::FileStorage &fSettings);
    bool ParseIMUParamFile(cv::FileStorage &fSettings);

    // Preprocess the input and call Track(). Extract features and performs stereo matching.
    Sophus::SE3f GrabImageStereo(const cv::Mat &imRectLeft,const cv::Mat &imRectRight, const double &timestamp, string filename);
    Sophus::SE3f GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp, string filename);
    Sophus::SE3f GrabImageMonocular(const cv::Mat &im, const double &timestamp, string filename);

    void GrabImuData(const IMU::Point &imuMeasurement);

    void SetLocalMapper(LocalMapping* pLocalMapper);
    void SetLoopClosing(LoopClosing* pLoopClosing);
    void SetViewer(Viewer* pViewer);
    void SetStepByStep(bool bSet);
    bool GetStepByStep();

    void SetFeatureMask(const cv::Mat &mask);
    void SetExternalTagObservation(
        const Sophus::SE3f &Twc,
        const float confidence,
        const std::vector<Eigen::Vector3f> &worldPoints,
        const std::vector<cv::Point2f> &imagePoints,
        const bool valid,
        const std::vector<float> &pointWeights = {},
        const std::vector<int> &tagIds = {},
        bool partial = false, int trackedCorners = 0, float trackAgeS = 0.0f,
        bool inputInAtlasWorld = false);
    float GetRecoveredTagMetricScale() const;
    bool IsTagMetricAligned() const;
    unsigned int GetTagKeyFramesAccepted() const;
    unsigned int GetTagKeyFramesRejected() const;
    unsigned int GetTagPoseConstraintsApplied() const;
    bool CurrentPoseHasTagConstraint() const {
        return mnTagPoseConstraintFrameId == long(mCurrentFrame.mnId) ||
            (mState == MARKER_TRACKING && (mbHasExternalTagObservation || mbHasTrackedTagObservation));
    }

    struct MarkerBootstrapStatus {
        bool active = false;
        std::string reason = "inactive";
        long referenceFrame = -1;
        bool referenceChanged = false;
        int matches = 0;
        int triangulated = 0;
        float baselineM = 0.0f;
    };
    const MarkerBootstrapStatus &GetMarkerBootstrapStatus() const { return mMarkerBootstrapStatus; }
    const std::string &GetMarkerKeyFrameEvent() const { return mMarkerKeyFrameEvent; }
    long GetMarkerEventKeyFrameId() const { return mnMarkerEventKeyFrameId; }
    struct MarkerTrackingStatus {
        bool partial=false, accepted=false;
        bool poseConstraintApplied=false;
        int trackedCorners=0;
        float confidence=0, ageS=0, reprojectionPx=0;
        float posePositionResidualM=-1, poseRotationResidualDeg=-1;
        std::string reason="no_observation";
        std::string poseConstraintReason="not_evaluated";
        std::vector<cv::Point2f> pixels;
        std::vector<int> ids;
    };
    const MarkerTrackingStatus &GetMarkerTrackingStatus() const { return mMarkerTrackingStatus; }
    void ProcessMarkerGraph(bool final = false);
    void CancelMarkerGraph();

    // Load new settings
    // The focal lenght should be similar or scale prediction will fail when projecting points
    void ChangeCalibration(const string &strSettingPath);

    // Use this function if you have deactivated local mapping and you only want to localize the camera.
    void InformOnlyTracking(const bool &flag);

    void UpdateFrameIMU(const float s, const IMU::Bias &b, KeyFrame* pCurrentKeyFrame);
    KeyFrame* GetLastKeyFrame()
    {
        return mpLastKeyFrame;
    }

    void CreateMapInAtlas();
    //std::mutex mMutexTracks;

    //--
    void NewDataset();
    int GetNumberDataset();
    int GetMatchesInliers();

    //DEBUG
    void SaveSubTrajectory(string strNameFile_frames, string strNameFile_kf, string strFolder="");
    void SaveSubTrajectory(string strNameFile_frames, string strNameFile_kf, Map* pMap);

    float GetImageScale();

#ifdef REGISTER_LOOP
    void RequestStop();
    bool isStopped();
    void Release();
    bool stopRequested();
#endif

public:

    // Tracking states
    enum eTrackingState{
        SYSTEM_NOT_READY=-1,
        NO_IMAGES_YET=0,
        NOT_INITIALIZED=1,
        OK=2,
        RECENTLY_LOST=3,
        LOST=4,
        OK_KLT=5,
        MARKER_TRACKING=6
    };

    eTrackingState mState;
    eTrackingState mLastProcessedState;

    // Input sensor
    int mSensor;

    // Current Frame
    Frame mCurrentFrame;
    Frame mLastFrame;

    cv::Mat mImGray;

    // Initialization Variables (Monocular)
    std::vector<int> mvIniLastMatches;
    std::vector<int> mvIniMatches;
    std::vector<cv::Point2f> mvbPrevMatched;
    std::vector<cv::Point3f> mvIniP3D;
    Frame mInitialFrame;

    // Lists used to recover the full camera trajectory at the end of the execution.
    // Basically we store the reference keyframe for each frame and its relative transformation
    list<Sophus::SE3f> mlRelativeFramePoses;
    list<KeyFrame*> mlpReferences;
    // Reference-map unit scale when each relative frame pose was stored.
    // Later metricization or Sim(3) corrections may change the surviving
    // keyframe scale, so trajectory recovery needs the original value.
    list<float> mlReferenceUnitScales;
    list<double> mlFrameTimes;
    list<bool> mlbLost;
    struct MarkerMetricFrame {
        std::size_t historyIndex = 0;
        Sophus::SE3f worldFromCamera;
        Sophus::SE3f visualCameraFromReference;
        bool hasVisualRelative = false;
    };
    // Sparse records of actual tag-constrained metre poses, not visual
    // relative translations. Preserve these through scale BA; update their
    // separate rigid correction chain after merge or joint marker-pose BA.
    // The mapped value contains Sophus::SE3f.  Keep map nodes aligned across
    // libstdc++/libc++ so long offline histories do not depend on allocator
    // luck when poses are revisited during retrospective recovery.
    using MarkerMetricFrameEntry =
        std::pair<const unsigned long, MarkerMetricFrame>;
    using MarkerMetricFrameMap = std::map<unsigned long, MarkerMetricFrame,
                                          std::less<unsigned long>,
                                          Eigen::aligned_allocator<MarkerMetricFrameEntry>>;
    MarkerMetricFrameMap mMarkerMetricFrames;

    // frames with estimated pose
    int mTrackedFr;
    bool mbStep;

    // True if local mapping is deactivated and we are performing only localization
    bool mbOnlyTracking;

    void Reset(bool bLocMap = false);
    void ResetActiveMap(bool bLocMap = false);

    float mMeanTrack;
    bool mbInitWith3KFs;
    double t0; // time-stamp of first read frame
    double t0vis; // time-stamp of first inserted keyframe
    double t0IMU; // time-stamp of IMU initialization
    bool mFastInit = false;


    vector<MapPoint*> GetLocalMapMPS();

    bool mbWriteStats;

#ifdef REGISTER_TIMES
    void LocalMapStats2File();
    void TrackStats2File();
    void PrintTimeStats();

    vector<double> vdRectStereo_ms;
    vector<double> vdResizeImage_ms;
    vector<double> vdORBExtract_ms;
    vector<double> vdStereoMatch_ms;
    vector<double> vdIMUInteg_ms;
    vector<double> vdPosePred_ms;
    vector<double> vdLMTrack_ms;
    vector<double> vdNewKF_ms;
    vector<double> vdTrackTotal_ms;
#endif

protected:

    // Main tracking function. It is independent of the input sensor.
    void Track();

    // Map initialization for stereo and RGB-D
    void StereoInitialization();

    // Map initialization for monocular
    void MonocularInitialization();
    //void CreateNewMapPoints();
    void CreateInitialMapMonocular();

    void CheckReplacedInLastFrame();
    bool TrackReferenceKeyFrame();
    void UpdateLastFrame();
    bool TrackWithMotionModel();
    bool TrackWithTemporalFlow(bool recovery=false, const Frame* sourceFrame=nullptr,
                               const cv::Mat* sourceImage=nullptr);
    bool TryTemporalFlowRecovery(int minimumImprovement);
    void UpdateReliableFlowFrame();
    void ClearReliableFlowFrame();
    Frame mReliableFlowFrame;
    cv::Mat mReliableFlowImage;
    Map* mpReliableFlowMap = nullptr;
    unsigned long mnReliableFlowMapId = 0, mnReliableFlowGraphSequence = 0;
    int mnReliableFlowBigChange = -1;
    bool mbReliableFlowMetric = false;
    float mReliableFlowScale = 0.f;
    std::vector<unsigned long> mvReliableFlowPointIds;
    bool mbFlowRecoveryEnabled=false;
    bool mbReliableFrameRecoveryEnabled=true;
    bool mbReliableFrameCacheEnabled=true;
    bool mbTemporalFlowEnabled = false;
    cv::Mat mTemporalPreviousImage;
    double mTemporalPreviousTime = -1.;
    bool PredictStateIMU();

    bool Relocalization();

    void UpdateLocalMap();
    void UpdateLocalPoints();
    void UpdateLocalKeyFrames();

    bool TrackLocalMap(bool recoveryTrial=false);
    void SearchLocalPoints(bool recoveryTrial=false);

    bool NeedNewKeyFrame();
    void CreateNewKeyFrame();

    // Perform preintegration from last frame
    void PreintegrateIMU();

    // Reset IMU biases and compute frame velocity
    void ResetFrameIMU();

    bool mbMapUpdated;

    // Imu preintegration from last frame
    IMU::Preintegrated *mpImuPreintegratedFromLastKF;

    // Queue of IMU measurements between frames
    std::list<IMU::Point> mlQueueImuData;

    // Vector of IMU measurements from previous to current frame (to be filled by PreintegrateIMU)
    std::vector<IMU::Point> mvImuFromLastFrame;
    std::mutex mMutexImuQueue;

    // Imu calibration parameters
    IMU::Calib *mpImuCalib;

    // Last Bias Estimation (at keyframe creation)
    IMU::Bias mLastBias;

    // In case of performing only localization, this flag is true when there are no matches to
    // points in the map. Still tracking will continue if there are enough matches with temporal points.
    // In that case we are doing visual odometry. The system will try to do relocalization to recover
    // "zero-drift" localization to the map.
    bool mbVO;

    //Other Thread Pointers
    LocalMapping* mpLocalMapper;
    LoopClosing* mpLoopClosing;

    //ORB
    ORBextractor* mpORBextractorLeft, *mpORBextractorRight;
    ORBextractor* mpIniORBextractor;

    //BoW
    ORBVocabulary* mpORBVocabulary;
    KeyFrameDatabase* mpKeyFrameDB;

    // Initalization (only for monocular)
    bool mbReadyToInitializate;
    bool mbSetInit;

    //Local Map
    KeyFrame* mpReferenceKF;
    std::vector<KeyFrame*> mvpLocalKeyFrames;
    std::vector<MapPoint*> mvpLocalMapPoints;
    
    // System
    System* mpSystem;
    
    //Drawers
    Viewer* mpViewer;
    FrameDrawer* mpFrameDrawer;
    MapDrawer* mpMapDrawer;
    bool bStepByStep;

    //Atlas
    Atlas* mpAtlas;

    //Calibration matrix
    cv::Mat mK;
    Eigen::Matrix3f mK_;
    cv::Mat mDistCoef;
    float mbf;
    float mImageScale;

    float mImuFreq;
    double mImuPer;
    bool mInsertKFsLost;

    //New KeyFrame rules (according to fps)
    int mMinFrames;
    int mMaxFrames;

    // Optional fixed-tag metric fusion for the local offline monocular runner.
    bool mbTagFusionEnabled = false;
    bool mbRigidMarkerLayout = false;
    std::unique_ptr<MarkerGraphCoordinator> mpMarkerGraphCoordinator;
    Sophus::SE3f mMarkerGraphVisualTwc;
    long mnMarkerGraphVisualFrameId = -1;
    void CaptureMarkerGraphVisualPose();
    bool mbMarkerOnlyInitialization = false;
    KeyFrame* mpMarkerSeedKF = nullptr;
    MarkerBootstrapStatus mMarkerBootstrapStatus;
    Map* mpMarkerEventMap = nullptr;
    double mLastLocalizedFrameTime = -1.0;
    std::set<int> mKeyframedMarkerIds;
    std::map<int,double> mLastDecodedMarkerTime;
    std::map<int,std::string> mPendingMarkerEvents;
    std::string mMarkerKeyFrameEvent;
    long mnMarkerEventKeyFrameId = -1;
    void UpdateMarkerKeyFrameEvents();
    bool HasMarkerKeyFrameEvent() const;
    void RecordMarkerKeyFrame(KeyFrame* keyframe);
    void InsertMarkerOnlyEventKeyFrame();
    double mLastReliableMarkerTime = -1.0;
    std::vector<int> mvExternalTagIds;
    std::vector<int> mvInitialTagIds;
    bool mbHasExternalTagObservation = false;
    bool mbHasTrackedTagObservation = false;
    MarkerTrackingStatus mMarkerTrackingStatus;
    bool mbHasInitialTagObservation = false;
    bool mbHasTagScaleReference = false;
    bool mbTagMetricAligned = false;
    bool mbTagAlignmentPending = false;
    Map* mpTagAlignmentMap = nullptr;
    Sophus::SE3f mExternalTagTwc;
    Sophus::SE3f mInitialTagTwc;
    Sophus::SE3f mTagScaleReferenceMetricTwc;
    Sophus::SE3f mTagScaleReferenceSlamTwc;
    double mTagScaleReferenceTimestamp = 0.0;
    Map* mpTagScaleReferenceMap = nullptr;
    Sophus::SE3f mPendingTagWorldFromSlamWorld;
    float mExternalTagConfidence = 0.0f;
    float mInitialTagConfidence = 0.0f;
    float mRecoveredTagMetricScale = 0.0f;
    float mPendingTagMetricScale = 0.0f;
    float mTagMinimumScaleBaselineM = 0.04f;
    // Metric scale is a ratio of marker motion to visual-map motion.  A
    // physical marker baseline alone is insufficient: pose noise can appear
    // to move by centimetres while a freshly initialized monocular map has
    // essentially zero translational parallax.  Keep this gate scale-free by
    // normalizing the visual baseline by the map's median scene depth.
    float mTagMinimumVisualBaselineDepthRatio = 0.01f;
    float mTagMinimumTranslationM = 0.02f;
    float mTagMinimumRotationDeg = 5.0f;
    float mTagMinimumTrackedRatio = 0.60f;
    float mTagPoseWeight = 0.85f;
    float mTagMaxAlignmentPositionResidualM = 0.03f;
    float mTagMaxAlignmentRotationResidualDeg = 10.0f;
    Map* mpInstantTagConsistencyMap = nullptr;
    std::set<int> mInstantTagMarkerIds;
    Sophus::SE3f mInstantTagCorrection;
    long mnInstantTagConsistencyFrameId = -1;
    int mnInstantTagConsistencyFrames = 0;
    int mTagMinimumKeyFrameFrames = 1;
    unsigned int mnTagKeyFramesAccepted = 0;
    unsigned int mnTagKeyFramesRejected = 0;
    unsigned int mnTagPoseConstraintsApplied = 0;
    long mnTagPoseConstraintFrameId = -1;
    std::vector<Eigen::Vector3f> mvExternalTagWorldPoints;
    std::vector<cv::Point2f> mvExternalTagImagePoints;
    std::vector<float> mvExternalTagPointWeights;
    std::vector<Eigen::Vector3f> mvInitialTagWorldPoints;
    std::vector<cv::Point2f> mvInitialTagImagePoints;
    std::vector<float> mvInitialTagPointWeights;
    struct TagScaleSample {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Sophus::SE3f metricTwc;
        Sophus::SE3f slamTwc;
        double timestamp = 0.0;
        Map* map = nullptr;
    };
    using TagScaleSampleVector = std::vector<TagScaleSample,
                                              Eigen::aligned_allocator<TagScaleSample>>;
    TagScaleSampleVector mvTagScaleSamples;

    int mnFirstImuFrameId;
    int mnFramesToResetIMU;

    // Threshold close/far points
    // Points seen as close by the stereo/RGBD sensor are considered reliable
    // and inserted from just one frame. Far points requiere a match in two keyframes.
    float mThDepth;

    // For RGB-D inputs only. For some datasets (e.g. TUM) the depthmap values are scaled.
    float mDepthMapFactor;

    //Current matches in frame
    int mnMatchesInliers;

    //Last Frame, KeyFrame and Relocalisation Info
    KeyFrame* mpLastKeyFrame;
    unsigned int mnLastKeyFrameId;
    unsigned int mnLastRelocFrameId;
    double mTimeStampLost;
    double time_recently_lost;

    unsigned int mnFirstFrameId;
    unsigned int mnInitialFrameId;
    unsigned int mnLastInitFrameId;

    bool mbCreatedMap;

    //Motion Model
    bool mbVelocity{false};
    Sophus::SE3f mVelocity;

    //Color order (true RGB, false BGR, ignored if grayscale)
    bool mbRGB;

    list<MapPoint*> mlpTemporalPoints;

    //int nMapChangeIndex;

    int mnNumDataset;

    ofstream f_track_stats;

    ofstream f_track_times;
    double mTime_PreIntIMU;
    double mTime_PosePred;
    double mTime_LocalMapTrack;
    double mTime_NewKF_Dec;

    GeometricCamera* mpCamera, *mpCamera2;

    int initID, lastID;

    Sophus::SE3f mTlr;

    void newParameterLoader(Settings* settings);
    bool TrackMarkerSeed();
    bool MarkerRecoveryPoseConsistent();
    void SetMarkerBootstrapReference();
    void BootstrapMarkerBackground();
    void StoreMarkerFrame();
    void AttachCurrentTagObservation(KeyFrame* pKF, const bool initialObservation = false);
    bool TryAlignMapToTagWorld();
    void CancelPendingTagAlignment();
    void ApplyExternalTagPoseConstraint();

#ifdef REGISTER_LOOP
    bool Stop();

    bool mbStopped;
    bool mbStopRequested;
    bool mbNotStop;
    std::mutex mMutexStop;
#endif

public:
    cv::Mat mImRight;
};

} //namespace ORB_SLAM

#endif // TRACKING_H
