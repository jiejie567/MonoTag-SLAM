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



#include "System.h"
#include "Converter.h"
#include "PostLoopRematch.h"
#include <array>
#include <limits>
#include <thread>
#include <pangolin/pangolin.h>
#include <iomanip>
#include <openssl/md5.h>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/string.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>

namespace ORB_SLAM3
{

namespace
{
using ReplayPoint = std::array<float,3>;
using ReplayMapPoints = std::map<unsigned long,ReplayPoint>;
using ReplayAtlasPoints = std::map<unsigned long,ReplayMapPoints>;

struct ReplayPublicationState
{
    bool initialized=false;
    double lastTimestamp=0.0;
    double lastCheckpointTimestamp=0.0;
    ReplayAtlasPoints points;
};

std::mutex gReplayPublicationMutex;
std::map<const System*,ReplayPublicationState> gReplayPublicationStates;
}

Verbose::eLevel Verbose::th = Verbose::VERBOSITY_NORMAL;

System::System(const string &strVocFile, const string &strSettingsFile, const eSensor sensor,
               const bool bUseViewer, const int initFr, const string &strSequence, const bool bViewerMainThread):
    mSensor(sensor), mpViewer(static_cast<Viewer*>(NULL)), mbReset(false), mbResetActiveMap(false),
    mbActivateLocalizationMode(false), mbDeactivateLocalizationMode(false), mbShutDown(false)
{
    // Output welcome message
    cout << endl <<
    "ORB-SLAM3 Copyright (C) 2017-2020 Carlos Campos, Richard Elvira, Juan J. Gómez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza." << endl <<
    "ORB-SLAM2 Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza." << endl <<
    "This program comes with ABSOLUTELY NO WARRANTY;" << endl  <<
    "This is free software, and you are welcome to redistribute it" << endl <<
    "under certain conditions. See LICENSE.txt." << endl << endl;

    cout << "Input sensor was set to: ";

    if(mSensor==MONOCULAR)
        cout << "Monocular" << endl;
    else if(mSensor==STEREO)
        cout << "Stereo" << endl;
    else if(mSensor==RGBD)
        cout << "RGB-D" << endl;
    else if(mSensor==IMU_MONOCULAR)
        cout << "Monocular-Inertial" << endl;
    else if(mSensor==IMU_STEREO)
        cout << "Stereo-Inertial" << endl;
    else if(mSensor==IMU_RGBD)
        cout << "RGB-D-Inertial" << endl;

    //Check settings file
    cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
       cerr << "Failed to open settings file at: " << strSettingsFile << endl;
       exit(-1);
    }

    cv::FileNode node = fsSettings["File.version"];
    if(!node.empty() && node.isString() && node.string() == "1.0"){
        settings_ = new Settings(strSettingsFile,mSensor);

        mStrLoadAtlasFromFile = settings_->atlasLoadFile();
        mStrSaveAtlasToFile = settings_->atlasSaveFile();

        cout << (*settings_) << endl;
    }
    else{
        settings_ = nullptr;
        cv::FileNode node = fsSettings["System.LoadAtlasFromFile"];
        if(!node.empty() && node.isString())
        {
            mStrLoadAtlasFromFile = (string)node;
        }

        node = fsSettings["System.SaveAtlasToFile"];
        if(!node.empty() && node.isString())
        {
            mStrSaveAtlasToFile = (string)node;
        }
    }

    node = fsSettings["loopClosing"];
    bool activeLC = true;
    if(!node.empty())
    {
        activeLC = static_cast<int>(fsSettings["loopClosing"]) != 0;
    }

    mStrVocabularyFilePath = strVocFile;

    bool loadedAtlas = false;

    if(mStrLoadAtlasFromFile.empty())
    {
        //Load ORB Vocabulary
        cout << endl << "Loading ORB Vocabulary. This could take a while..." << endl;

        mpVocabulary = new ORBVocabulary();
        bool bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);
        if(!bVocLoad)
        {
            cerr << "Wrong path to vocabulary. " << endl;
            cerr << "Falied to open at: " << strVocFile << endl;
            exit(-1);
        }
        cout << "Vocabulary loaded!" << endl << endl;

        //Create KeyFrame Database
        mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);

        //Create the Atlas
        cout << "Initialization of Atlas from scratch " << endl;
        mpAtlas = new Atlas(0);
    }
    else
    {
        //Load ORB Vocabulary
        cout << endl << "Loading ORB Vocabulary. This could take a while..." << endl;

        mpVocabulary = new ORBVocabulary();
        bool bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);
        if(!bVocLoad)
        {
            cerr << "Wrong path to vocabulary. " << endl;
            cerr << "Falied to open at: " << strVocFile << endl;
            exit(-1);
        }
        cout << "Vocabulary loaded!" << endl << endl;

        //Create KeyFrame Database
        mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);

        cout << "Load File" << endl;

        // Load the file with an earlier session
        //clock_t start = clock();
        cout << "Initialization of Atlas from file: " << mStrLoadAtlasFromFile << endl;
        bool isRead = LoadAtlas(FileType::BINARY_FILE);

        if(!isRead)
        {
            cout << "Error to load the file, please try with other session file or vocabulary file" << endl;
            exit(-1);
        }
        //mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);


        //cout << "KF in DB: " << mpKeyFrameDatabase->mnNumKFs << "; words: " << mpKeyFrameDatabase->mnNumWords << endl;

        loadedAtlas = true;

        mpAtlas->CreateNewMap();

        //clock_t timeElapsed = clock() - start;
        //unsigned msElapsed = timeElapsed / (CLOCKS_PER_SEC / 1000);
        //cout << "Binary file read in " << msElapsed << " ms" << endl;

        //usleep(10*1000*1000);
    }


    if (mSensor==IMU_STEREO || mSensor==IMU_MONOCULAR || mSensor==IMU_RGBD)
        mpAtlas->SetInertialSensor();

    //Create Drawers. These are used by the Viewer
    mpFrameDrawer = new FrameDrawer(mpAtlas);
    mpMapDrawer = new MapDrawer(mpAtlas, strSettingsFile, settings_);

    //Initialize the Tracking thread
    //(it will live in the main thread of execution, the one that called this constructor)
    cout << "Seq. Name: " << strSequence << endl;
    mpTracker = new Tracking(this, mpVocabulary, mpFrameDrawer, mpMapDrawer,
                             mpAtlas, mpKeyFrameDatabase, strSettingsFile, mSensor, settings_, strSequence);

    //Initialize the Local Mapping thread and launch
    mpLocalMapper = new LocalMapping(this, mpAtlas, mSensor==MONOCULAR || mSensor==IMU_MONOCULAR,
                                     mSensor==IMU_MONOCULAR || mSensor==IMU_STEREO || mSensor==IMU_RGBD, strSequence);
    mptLocalMapping = new thread(&ORB_SLAM3::LocalMapping::Run,mpLocalMapper);
    mpLocalMapper->mInitFr = initFr;
    if(settings_)
        mpLocalMapper->mThFarPoints = settings_->thFarPoints();
    else
        mpLocalMapper->mThFarPoints = fsSettings["thFarPoints"];
    if(mpLocalMapper->mThFarPoints!=0)
    {
        cout << "Discard points further than " << mpLocalMapper->mThFarPoints << " m from current camera" << endl;
        mpLocalMapper->mbFarPoints = true;
    }
    else
        mpLocalMapper->mbFarPoints = false;

    //Initialize the Loop Closing thread and launch
    // mSensor!=MONOCULAR && mSensor!=IMU_MONOCULAR
    mpLoopCloser = new LoopClosing(mpAtlas, mpKeyFrameDatabase, mpVocabulary, mSensor!=MONOCULAR, activeLC); // mSensor!=MONOCULAR);
    mptLoopClosing = new thread(&ORB_SLAM3::LoopClosing::Run, mpLoopCloser);

    //Set pointers between threads
    mpTracker->SetLocalMapper(mpLocalMapper);
    mpTracker->SetLoopClosing(mpLoopCloser);

    mpLocalMapper->SetTracker(mpTracker);
    mpLocalMapper->SetLoopCloser(mpLoopCloser);

    mpLoopCloser->SetTracker(mpTracker);
    mpLoopCloser->SetLocalMapper(mpLocalMapper);

    //usleep(10*1000*1000);

    //Initialize the Viewer thread and launch
    if(bUseViewer)
    //if(false) // TODO
    {
        mpViewer = new Viewer(this, mpFrameDrawer,mpMapDrawer,mpTracker,strSettingsFile,settings_);
        if(bViewerMainThread)
            mptViewer = static_cast<thread*>(NULL);
        else
            mptViewer = new thread(&Viewer::Run, mpViewer);
        mpTracker->SetViewer(mpViewer);
        mpLoopCloser->mpViewer = mpViewer;
        mpViewer->both = mpFrameDrawer->both;
    }

    // Fix verbosity
    Verbose::SetTh(Verbose::VERBOSITY_QUIET);

}

Sophus::SE3f System::TrackStereo(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp, const vector<IMU::Point>& vImuMeas, string filename)
{
    if(mSensor!=STEREO && mSensor!=IMU_STEREO)
    {
        cerr << "ERROR: you called TrackStereo but input sensor was not set to Stereo nor Stereo-Inertial." << endl;
        exit(-1);
    }

    cv::Mat imLeftToFeed, imRightToFeed;
    if(settings_ && settings_->needToRectify()){
        cv::Mat M1l = settings_->M1l();
        cv::Mat M2l = settings_->M2l();
        cv::Mat M1r = settings_->M1r();
        cv::Mat M2r = settings_->M2r();

        cv::remap(imLeft, imLeftToFeed, M1l, M2l, cv::INTER_LINEAR);
        cv::remap(imRight, imRightToFeed, M1r, M2r, cv::INTER_LINEAR);
    }
    else if(settings_ && settings_->needToResize()){
        cv::resize(imLeft,imLeftToFeed,settings_->newImSize());
        cv::resize(imRight,imRightToFeed,settings_->newImSize());
    }
    else{
        imLeftToFeed = imLeft.clone();
        imRightToFeed = imRight.clone();
    }

    // Check mode change
    {
        unique_lock<mutex> lock(mMutexMode);
        if(mbActivateLocalizationMode)
        {
            mpLocalMapper->RequestStop();

            // Wait until Local Mapping has effectively stopped
            while(!mpLocalMapper->isStopped())
            {
                usleep(1000);
            }

            mpTracker->InformOnlyTracking(true);
            mbActivateLocalizationMode = false;
        }
        if(mbDeactivateLocalizationMode)
        {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }

    // Check reset
    {
        unique_lock<mutex> lock(mMutexReset);
        if(mbReset)
        {
            mpTracker->Reset();
            mbReset = false;
            mbResetActiveMap = false;
        }
        else if(mbResetActiveMap)
        {
            mpTracker->ResetActiveMap();
            mbResetActiveMap = false;
        }
    }

    // Reset waits for LoopClosing and must remain outside this gate.
    unique_lock<mutex> correctionGate(mpAtlas->mMutexPoseGraphCorrection);
    if (mSensor == System::IMU_STEREO)
        for(size_t i_imu = 0; i_imu < vImuMeas.size(); i_imu++)
            mpTracker->GrabImuData(vImuMeas[i_imu]);

    // std::cout << "start GrabImageStereo" << std::endl;
    Sophus::SE3f Tcw = mpTracker->GrabImageStereo(imLeftToFeed,imRightToFeed,timestamp,filename);

    // std::cout << "out grabber" << std::endl;

    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;

    return Tcw;
}

Sophus::SE3f System::TrackRGBD(const cv::Mat &im, const cv::Mat &depthmap, const double &timestamp, const vector<IMU::Point>& vImuMeas, string filename)
{
    if(mSensor!=RGBD  && mSensor!=IMU_RGBD)
    {
        cerr << "ERROR: you called TrackRGBD but input sensor was not set to RGBD." << endl;
        exit(-1);
    }

    cv::Mat imToFeed = im.clone();
    cv::Mat imDepthToFeed = depthmap.clone();
    if(settings_ && settings_->needToResize()){
        cv::Mat resizedIm;
        cv::resize(im,resizedIm,settings_->newImSize());
        imToFeed = resizedIm;

        cv::resize(depthmap,imDepthToFeed,settings_->newImSize());
    }

    // Check mode change
    {
        unique_lock<mutex> lock(mMutexMode);
        if(mbActivateLocalizationMode)
        {
            mpLocalMapper->RequestStop();

            // Wait until Local Mapping has effectively stopped
            while(!mpLocalMapper->isStopped())
            {
                usleep(1000);
            }

            mpTracker->InformOnlyTracking(true);
            mbActivateLocalizationMode = false;
        }
        if(mbDeactivateLocalizationMode)
        {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }

    // Check reset
    {
        unique_lock<mutex> lock(mMutexReset);
        if(mbReset)
        {
            mpTracker->Reset();
            mbReset = false;
            mbResetActiveMap = false;
        }
        else if(mbResetActiveMap)
        {
            mpTracker->ResetActiveMap();
            mbResetActiveMap = false;
        }
    }

    unique_lock<mutex> correctionGate(mpAtlas->mMutexPoseGraphCorrection);
    if (mSensor == System::IMU_RGBD)
        for(size_t i_imu = 0; i_imu < vImuMeas.size(); i_imu++)
            mpTracker->GrabImuData(vImuMeas[i_imu]);

    Sophus::SE3f Tcw = mpTracker->GrabImageRGBD(imToFeed,imDepthToFeed,timestamp,filename);

    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
    return Tcw;
}

Sophus::SE3f System::TrackMonocular(const cv::Mat &im, const double &timestamp, const vector<IMU::Point>& vImuMeas, string filename)
{

    {
        unique_lock<mutex> lock(mMutexReset);
        if(mbShutDown)
            return Sophus::SE3f();
    }

    if(mSensor!=MONOCULAR && mSensor!=IMU_MONOCULAR)
    {
        cerr << "ERROR: you called TrackMonocular but input sensor was not set to Monocular nor Monocular-Inertial." << endl;
        exit(-1);
    }

    cv::Mat imToFeed = im.clone();
    if(settings_ && settings_->needToResize()){
        cv::Mat resizedIm;
        cv::resize(im,resizedIm,settings_->newImSize());
        imToFeed = resizedIm;
    }

    // Check mode change
    {
        unique_lock<mutex> lock(mMutexMode);
        if(mbActivateLocalizationMode)
        {
            mpLocalMapper->RequestStop();

            // Wait until Local Mapping has effectively stopped
            while(!mpLocalMapper->isStopped())
            {
                usleep(1000);
            }

            mpTracker->InformOnlyTracking(true);
            mbActivateLocalizationMode = false;
        }
        if(mbDeactivateLocalizationMode)
        {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }

    // Check reset
    {
        unique_lock<mutex> lock(mMutexReset);
        if(mbReset)
        {
            mpTracker->Reset();
            mbReset = false;
            mbResetActiveMap = false;
        }
        else if(mbResetActiveMap)
        {
            cout << "SYSTEM-> Reseting active map in monocular case" << endl;
            mpTracker->ResetActiveMap();
            mbResetActiveMap = false;
        }
    }

    unique_lock<mutex> correctionGate(mpAtlas->mMutexPoseGraphCorrection);
    if (mSensor == System::IMU_MONOCULAR)
        for(size_t i_imu = 0; i_imu < vImuMeas.size(); i_imu++)
            mpTracker->GrabImuData(vImuMeas[i_imu]);

    Sophus::SE3f Tcw = mpTracker->GrabImageMonocular(imToFeed,timestamp,filename);
    mpTracker->ProcessMarkerGraph();
    if(mpTracker->mCurrentFrame.HasPose()) Tcw=mpTracker->mCurrentFrame.GetPose();

    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;

    return Tcw;
}



void System::ActivateLocalizationMode()
{
    unique_lock<mutex> lock(mMutexMode);
    mbActivateLocalizationMode = true;
}

void System::DeactivateLocalizationMode()
{
    unique_lock<mutex> lock(mMutexMode);
    mbDeactivateLocalizationMode = true;
}

bool System::MapChanged()
{
    static int n=0;
    int curn = mpAtlas->GetLastBigChangeIdx();
    if(n<curn)
    {
        n=curn;
        return true;
    }
    else
        return false;
}

void System::Reset()
{
    unique_lock<mutex> lock(mMutexReset);
    mbReset = true;
}

void System::ResetActiveMap()
{
    unique_lock<mutex> lock(mMutexReset);
    mbResetActiveMap = true;
}

void System::Shutdown()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbShutDown = true;
    }

    cout << "Shutdown" << endl;

    // Local Mapping is the producer of the loop-closing queue. Finish and
    // join that producer first, then let LoopClosing consume the complete
    // finite queue. Requesting both finishes together used to let the loop
    // thread exit after one item, dropping tail loop candidates.
    mpLocalMapper->RequestFinish();
    while(!mpLocalMapper->isFinished())
        usleep(1000);
    cout << "OFFLINE_FINALIZE loop_queue=" << mpLoopCloser->KeyframesInQueue() << endl;
    mpLoopCloser->RequestFinish();
    /*if(mpViewer)
    {
        mpViewer->RequestFinish();
        while(!mpViewer->isFinished())
            usleep(5000);
    }*/

    // Wait until all thread have effectively stopped
    /*while(!mpLocalMapper->isFinished() || !mpLoopCloser->isFinished() || mpLoopCloser->isRunningGBA())
    {
        if(!mpLocalMapper->isFinished())
            cout << "mpLocalMapper is not finished" << endl;*/
        /*if(!mpLoopCloser->isFinished())
            cout << "mpLoopCloser is not finished" << endl;
        if(mpLoopCloser->isRunningGBA()){
            cout << "mpLoopCloser is running GBA" << endl;
            cout << "break anyway..." << endl;
            break;
        }*/
        /*usleep(5000);
    }*/

    // Serialization and final export must not race local BA / loop closure.
    while(!mpLocalMapper->isFinished() || !mpLoopCloser->isFinished() ||
          mpLoopCloser->isRunningGBA())
        usleep(1000);

    // Explicit diagnostic prefixes preserve the pre-finalization graph for
    // repeatable solver experiments. Workers are already stopped above;
    // normal Atlas saving below still performs its documented cleanup.
    const char* diagnosticNoFinal=std::getenv("ORB_SLAM3_DIAGNOSTIC_NO_FINAL_OPTIMIZATION");
    const bool skipFinalOptimization=diagnosticNoFinal && std::string(diagnosticNoFinal)=="1";
    if(skipFinalOptimization)
        std::cout << "DIAGNOSTIC_SHUTDOWN_OPTIMIZATION_SKIPPED pre_finalization_snapshot=1" << std::endl;
    else mpLoopCloser->RunOfflineLoopSearch();

    // Explicit saved-Atlas experiment: only already committed loop edges.
    const char* savedRematch=std::getenv("ORB_SLAM3_POST_LOOP_REMATCH_SAVED");
    if(!skipFinalOptimization && savedRematch && !mStrLoadAtlasFromFile.empty() &&
       (std::string(savedRematch)=="1" || std::string(savedRematch)=="audit")) {
        unique_lock<mutex> correctionGate(mpAtlas->mMutexPoseGraphCorrection);
        for(KeyFrame* kf:mpAtlas->GetAllKeyFrames()) if(kf && !kf->isBad())
            for(KeyFrame* other:kf->GetLoopEdges())
                if(other && !other->isBad() && other->GetMap()==kf->GetMap() && other->mnId<kf->mnId)
                    PostLoopRematch::Run(kf->GetMap(),other,kf,std::string(savedRematch)=="1");
    }

    if(!skipFinalOptimization) {
        unique_lock<mutex> correctionGate(mpAtlas->mMutexPoseGraphCorrection);
        mpTracker->ProcessMarkerGraph(true);
    }

    if(!mStrSaveAtlasToFile.empty())
    {
        Verbose::PrintMess("Atlas saving to file " + mStrSaveAtlasToFile, Verbose::VERBOSITY_NORMAL);
        SaveAtlas(FileType::BINARY_FILE);
    }

    /*if(mpViewer)
        pangolin::BindToContext("ORB-SLAM2: Map Viewer");*/

#ifdef REGISTER_TIMES
    mpTracker->PrintTimeStats();
#endif


}

bool System::isShutDown() {
    unique_lock<mutex> lock(mMutexReset);
    return mbShutDown;
}

void System::RunViewer()
{
    if(mpViewer)
        mpViewer->Run();
}

void System::RequestViewerFinish()
{
    if(mpViewer)
        mpViewer->RequestFinish();
}

void System::SaveTrajectoryTUM(const string &filename)
{
    cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
    if(mSensor==MONOCULAR)
    {
        cerr << "ERROR: SaveTrajectoryTUM cannot be used for monocular." << endl;
        return;
    }

    vector<KeyFrame*> vpKFs = mpAtlas->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    Sophus::SE3f Two = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    list<ORB_SLAM3::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    list<bool>::iterator lbL = mpTracker->mlbLost.begin();
    for(list<Sophus::SE3f>::iterator lit=mpTracker->mlRelativeFramePoses.begin(),
        lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++, lbL++)
    {
        if(*lbL)
            continue;

        KeyFrame* pKF = *lRit;

        Sophus::SE3f Trw;

        // If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
        while(pKF->isBad())
        {
            Trw = Trw * pKF->mTcp;
            pKF = pKF->GetParent();
        }

        Trw = Trw * pKF->GetPose() * Two;

        Sophus::SE3f Tcw = (*lit) * Trw;
        Sophus::SE3f Twc = Tcw.inverse();

        Eigen::Vector3f twc = Twc.translation();
        Eigen::Quaternionf q = Twc.unit_quaternion();

        f << setprecision(6) << *lT << " " <<  setprecision(9) << twc(0) << " " << twc(1) << " " << twc(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
    }
    f.close();
    // cout << endl << "trajectory saved!" << endl;
}

void System::SaveFrameTrajectoryTUM(const string &filename)
{
    cout << endl << "Saving frame trajectory to " << filename << " ..." << endl;

    vector<KeyFrame*> vpKFs = mpAtlas->GetAllKeyFrames();
    if(vpKFs.empty())
    {
        ofstream empty(filename.c_str());
        return;
    }

    ofstream f(filename.c_str());
    f << fixed;

    list<ORB_SLAM3::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    list<bool>::iterator lbL = mpTracker->mlbLost.begin();
    for(list<Sophus::SE3f>::iterator lit=mpTracker->mlRelativeFramePoses.begin(),
        lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++, lbL++)
    {
        if(*lbL)
            continue;

        KeyFrame* pKF = *lRit;
        Sophus::SE3f Trw;
        while(pKF->isBad())
        {
            Trw = Trw * pKF->mTcp;
            pKF = pKF->GetParent();
        }

        Trw = Trw * pKF->GetPose();
        const Sophus::SE3f Twc = ((*lit) * Trw).inverse();
        const Eigen::Vector3f twc = Twc.translation();
        const Eigen::Quaternionf q = Twc.unit_quaternion();
        f << setprecision(9) << *lT << " " << twc(0) << " " << twc(1) << " "
          << twc(2) << " " << q.x() << " " << q.y() << " " << q.z() << " "
          << q.w() << endl;
    }
}


void System::SaveReplaySnapshot(std::ostream &out, double timestamp, bool final, bool compact)
{
    // A merge commits in multiple map-locked stages; do not capture between
    // those stages, or enumerate maps while their ownership is changing.
    unique_lock<mutex> correctionGate(mpAtlas->mMutexPoseGraphCorrection);
    std::vector<Map*> maps=mpAtlas->GetAllMaps();
    std::sort(maps.begin(),maps.end(),[](Map* a,Map* b){return a->GetId()<b->GetId();});
    // Lock a complete published map version, never an intermediate BA iteration.
    std::vector<std::unique_lock<std::mutex>> locks;
    for(Map* map:maps) locks.emplace_back(map->mMutexMapUpdate,std::defer_lock);
    for(;;) {
        bool acquired=true;
        for(auto &lock:locks) if(!lock.try_lock()) {acquired=false;break;}
        if(acquired) break;
        for(auto &lock:locks) if(lock.owns_lock()) lock.unlock();
        usleep(1000);
    }
    // Copy one coherent point publication before writing any feature
    // association. A matched feature is emitted only if its point is present
    // in this exact publication, so compact logs cannot contain torn IDs.
    ReplayAtlasPoints currentPoints;
    std::set<unsigned long> publishedPointIds;
    for(Map* map:maps) {
        if(map->IsBad()) continue;
        ReplayMapPoints &points=currentPoints[map->GetId()];
        for(MapPoint* point:map->GetAllMapPoints()) {
            if(!point || point->isBad() || point->GetMap()!=map) continue;
            const Eigen::Vector3f position=point->GetWorldPos();
            if(!position.allFinite()) continue;
            points[point->mnId]={position.x(),position.y(),position.z()};
            publishedPointIds.insert(point->mnId);
        }
    }
    ReplayAtlasPoints previousPoints;
    bool pointCheckpoint=false;
    {
        unique_lock<mutex> stateLock(gReplayPublicationMutex);
        ReplayPublicationState &state=gReplayPublicationStates[this];
        if(state.initialized && timestamp<state.lastTimestamp)
            state=ReplayPublicationState();
        pointCheckpoint=!state.initialized || final || timestamp-state.lastCheckpointTimestamp>=5.0;
        previousPoints=state.points;
        state.initialized=true;
        state.lastTimestamp=timestamp;
        if(pointCheckpoint) state.lastCheckpointTimestamp=timestamp;
        state.points=currentPoints;
        if(final) gReplayPublicationStates.erase(this);
    }
    const auto pose=[&out](const Sophus::SE3f &Twc) {
        const auto t=Twc.translation(); const auto q=Twc.unit_quaternion();
        out << "[" << t.x() << "," << t.y() << "," << t.z() << ","
            << q.x() << "," << q.y() << "," << q.z() << "," << q.w() << "]";
    };
    const auto markerGraph=[&out](const MarkerGraphTransform& g) {
        out << "[" << g.sequence << "," << g.scale << ","
            << g.translation.x() << "," << g.translation.y() << "," << g.translation.z() << ","
            << g.rotation.x() << "," << g.rotation.y() << "," << g.rotation.z() << "," << g.rotation.w() << "]";
    };
    const auto residual=[&out](double value) {
        if(std::isfinite(value)) out << value; else out << "null";
    };
    Map* active=mpAtlas->GetCurrentMap();
    const bool valid=mpTracker->mCurrentFrame.HasPose() &&
        (mpTracker->mState==Tracking::OK || mpTracker->mState==Tracking::MARKER_TRACKING);
    const auto& tagStatus=mpTracker->GetMarkerTrackingStatus();
    const bool markerPoseUsed=valid && mpTracker->CurrentPoseHasTagConstraint();

    // Tracking archives Tcr and its reference KF atomically before Track()
    // returns.  A local/loop BA can update (or cull) that KF before this
    // snapshot is written, while mCurrentFrame still contains the pose from
    // the preceding map revision.  Recomputing Tcr from that mixed pair
    // creates a one-frame, otherwise well-tracked trajectory spike.  Prefer
    // the archived pair for ordinary visual frames; direct marker poses keep
    // their independent rigid-gauge publication below.
    KeyFrame* ref=mpTracker->mCurrentFrame.mpReferenceKF;
    Sophus::SE3f archivedRelative;
    float archivedReferenceScale=1.f;
    bool hasArchivedRelative=false;
    if(valid && !markerPoseUsed && !mpTracker->mlRelativeFramePoses.empty() &&
       !mpTracker->mlpReferences.empty() && !mpTracker->mlFrameTimes.empty() &&
       !mpTracker->mlReferenceUnitScales.empty() &&
       !mpTracker->mlbLost.empty() && !mpTracker->mlbLost.back() &&
       std::abs(mpTracker->mlFrameTimes.back()-timestamp)<1e-4) {
        ref=mpTracker->mlpReferences.back();
        archivedRelative=mpTracker->mlRelativeFramePoses.back();
        archivedReferenceScale=mpTracker->mlReferenceUnitScales.back();
        hasArchivedRelative=ref!=nullptr && archivedRelative.matrix().allFinite() &&
            std::isfinite(archivedReferenceScale) && archivedReferenceScale>0.f;
    }
    Sophus::SE3f effectiveReference;
    float effectiveReferenceScale=1.f;
    Map* effectiveReferenceMap=nullptr;
    const bool referenceValid=ref && ref->GetReplayReference(
        effectiveReference,effectiveReferenceScale,effectiveReferenceMap);
    if(!referenceValid) effectiveReferenceScale=1.f;
    Sophus::SE3f publishedWorldFromCamera;
    const bool coherentArchivedPose=valid && hasArchivedRelative && referenceValid;
    if(coherentArchivedPose) {
        Sophus::SE3f correctedRelative=archivedRelative;
        correctedRelative.translation()*=effectiveReferenceScale/archivedReferenceScale;
        publishedWorldFromCamera=effectiveReference*correctedRelative.inverse();
    }
    else if(valid)
        publishedWorldFromCamera=mpTracker->mCurrentFrame.GetPose().inverse();
    out << std::setprecision(9) << "{\"timestamp\":" << timestamp
        << ",\"final\":" << (final?"true":"false")
        << ",\"point_protocol\":\"map-delta-v1\""
        << ",\"state\":" << mpTracker->mState
        << ",\"map_tracking_inliers\":" << mpTracker->GetMatchesInliers()
        << ",\"active_map\":" << active->GetId() << ",\"pose\":";
    if(valid) pose(publishedWorldFromCamera); else out << "null";
    out << ",\"tag_anchored\":" << (markerPoseUsed ? "true" : "false")
        << ",\"marker_observation_valid\":" << (tagStatus.accepted ? "true" : "false")
        << ",\"marker_factor_eligible\":"
        << (tagStatus.accepted && !tagStatus.partial ? "true" : "false")
        << ",\"marker_pose_used\":" << (markerPoseUsed ? "true" : "false")
        << ",\"marker_pose_constraint_applied\":"
        << (tagStatus.poseConstraintApplied ? "true" : "false");
    out << ",\"marker_keyframe_event\":\"" << mpTracker->GetMarkerKeyFrameEvent()
        << "\",\"marker_event_keyframe_id\":" << mpTracker->GetMarkerEventKeyFrameId();
    out << ",\"marker_tracking\":{\"partial\":" << (tagStatus.partial?"true":"false")
        << ",\"accepted\":" << (tagStatus.accepted?"true":"false")
        << ",\"corners\":" << tagStatus.trackedCorners << ",\"confidence\":" << tagStatus.confidence
        << ",\"age_s\":" << tagStatus.ageS << ",\"reprojection_px\":" << tagStatus.reprojectionPx
        << ",\"reason\":\"" << tagStatus.reason << "\""
        << ",\"pose_constraint_applied\":" << (tagStatus.poseConstraintApplied?"true":"false")
        << ",\"pose_constraint_reason\":\"" << tagStatus.poseConstraintReason << "\""
        << ",\"pose_position_residual_m\":" << tagStatus.posePositionResidualM
        << ",\"pose_rotation_residual_deg\":" << tagStatus.poseRotationResidualDeg
        << ",\"points_undistorted\":[";
    for(size_t i=0;i<tagStatus.pixels.size();++i) {
        if(i) out << ",";
        out << "[" << tagStatus.pixels[i].x << "," << tagStatus.pixels[i].y << "," << tagStatus.ids[i] << "]";
    }
    out << "]}";
    out << ",\"reference\":"; if(valid && ref) out << ref->mnId; else out << "null";
    out << ",\"reference_scale\":"
        << (hasArchivedRelative ? archivedReferenceScale : effectiveReferenceScale);
    out << ",\"reference_marker_graph\":";
    MarkerGraphTransform currentGraph;
    if(ref && ref->GetReplayMarkerGraph(currentGraph)) markerGraph(currentGraph);
    else out << "null";
    out << ",\"reference_marker_gauge\":";
    MarkerGraphTransform currentGauge;
    if(ref && ref->GetReplayMarkerGauge(currentGauge)) markerGraph(currentGauge);
    else out << "null";
    out << ",\"relative\":";
    if(valid && referenceValid)
        pose(hasArchivedRelative ? archivedRelative
                                 : mpTracker->mCurrentFrame.GetPose()*effectiveReference);
    else out << "null";
    out << ",\"visual_relative\":";
    const auto visualFrame=mpTracker->mMarkerMetricFrames.find(
        mpTracker->mCurrentFrame.mnId);
    if(valid && referenceValid && visualFrame!=mpTracker->mMarkerMetricFrames.end() &&
       visualFrame->second.hasVisualRelative)
        pose(visualFrame->second.visualCameraFromReference);
    else if(valid && referenceValid && !markerPoseUsed)
        pose(hasArchivedRelative ? archivedRelative
                                 : mpTracker->mCurrentFrame.GetPose()*effectiveReference);
    else out << "null";
    const auto &bootstrap=mpTracker->GetMarkerBootstrapStatus();
    if(bootstrap.active)
        out << ",\"marker_bootstrap\":{\"reason\":\"" << bootstrap.reason << "\""
            << ",\"reference_frame\":" << bootstrap.referenceFrame
            << ",\"next_reference_frame\":" << (bootstrap.referenceChanged?long(mpTracker->mCurrentFrame.mnId):bootstrap.referenceFrame)
            << ",\"reference_changed\":" << (bootstrap.referenceChanged?"true":"false")
            << ",\"matches\":" << bootstrap.matches
            << ",\"triangulated\":" << bootstrap.triangulated
            << ",\"baseline_m\":" << bootstrap.baselineM << "}";
    const auto &frame=mpTracker->mCurrentFrame;
    out << ",\"feature_count\":" << frame.mvKeys.size()
        << ",\"matched_features\":[";
    bool firstFeature=true;
    for(size_t i=0;i<frame.mvKeys.size();++i) {
        MapPoint* p=i<frame.mvpMapPoints.size()?frame.mvpMapPoints[i]:nullptr;
        if(!p || p->isBad() || !valid || i>=frame.mvbOutlier.size() || frame.mvbOutlier[i]
           || !publishedPointIds.count(p->mnId))
            continue;
        if(!firstFeature) out << ","; firstFeature=false;
        out << "[" << frame.mvKeys[i].pt.x << "," << frame.mvKeys[i].pt.y
            << "," << p->mnId << "]";
    }
    out << "],\"maps\":[";
    bool firstMap=true;
    for(Map* map:maps) {
        if(map->IsBad()) continue;
        if(!firstMap) out << ","; firstMap=false;
        out << "{\"id\":" << map->GetId()
            << ",\"metric\":" << (map->mbMetric?"true":"false")
            << ",\"seed\":" << (map->mbMarkerSeed?"true":"false")
            << ",\"background\":" << (map->mbBackgroundReady?"true":"false")
            << ",\"rigid_marker_layout\":" << (map->mbRigidMarkerLayout?"true":"false")
            << ",\"scale\":" << map->mMetricScale
            << ",\"marker_graph_sequence\":" << map->mnMarkerGraphSequence
            << ",\"scale_anchor_keyframe\":" << map->mnMarkerScaleAnchorKFId
            << ",\"revision\":" << map->mnRevision
            << ",\"point_count\":" << currentPoints.at(map->GetId()).size()
            << ",\"keyframe_count\":" << map->KeyFramesInMap();
        if(compact && !final) {
            out << ",\"points_mode\":\"omitted\",\"points\":[],\"deleted_points\":[],"
                   "\"keyframes\":[],\"markers\":{},\"loops\":[],\"merges\":[]}";
            continue;
        }
        const unsigned long mapId=map->GetId();
        const ReplayMapPoints &points=currentPoints.at(mapId);
        const auto previousMap=previousPoints.find(mapId);
        const bool fullPoints=pointCheckpoint || previousMap==previousPoints.end();
        out << ",\"points_mode\":\"" << (fullPoints?"full":"delta") << "\""
            << ",\"points\":[";
        bool first=true;
        for(const auto &entry:points) {
            if(!fullPoints) {
                const auto old=previousMap->second.find(entry.first);
                if(old!=previousMap->second.end() && old->second==entry.second) continue;
            }
            if(!first) out << ","; first=false;
            out << "[" << entry.first << "," << entry.second[0] << ","
                << entry.second[1] << "," << entry.second[2] << "]";
        }
        out << "],\"deleted_points\":[";
        first=true;
        if(!fullPoints)
            for(const auto &entry:previousMap->second)
                if(!points.count(entry.first)) {
                    if(!first) out << ","; first=false;
                    out << entry.first;
                }
        out << "],\"keyframes\":[";
        first=true;
        for(KeyFrame* kf:map->GetAllKeyFrames()) {
            if(kf->isBad() || kf->GetMap()!=map) continue;
            if(!first) out << ","; first=false;
            out << "[" << kf->mnId << "," << kf->mTimeStamp << ",";
            pose(kf->GetPoseInverse());
            out << "]";
        }
        out << "],\"markers\":{";
        first=true;
        for(const auto &tag:map->mStaticTags) {
            if(!first) out << ","; first=false;
            out << "\"" << tag.first << "\":[";
            for(size_t i=0;i<tag.second.size();++i) {
                if(i) out << ","; out << tag.second[i];
            }
            out << "]";
        }
        out << "},\"loops\":[";
        first=true;
        for(KeyFrame* kf:map->GetAllKeyFrames())
            for(KeyFrame* other:kf->GetLoopEdges())
                if(kf->mnId<other->mnId) {
                    if(!first) out << ","; first=false;
                    out << "[" << kf->mnId << "," << other->mnId << "]";
                }
        out << "],\"merges\":[";
        first=true;
        for(KeyFrame* kf:map->GetAllKeyFrames())
            for(KeyFrame* other:kf->GetMergeEdges())
                if(kf->mnId<other->mnId) {
                    if(!first) out << ","; first=false;
                    out << "[" << kf->mnId << "," << other->mnId << "]";
                }
        out << "]}";
    }
    // Retain transforms of culled reference KFs too. Their spanning-tree
    // parent correction is needed to reproject past wrists after BA/loop.
    out << "],\"references\":[";
    std::set<KeyFrame*> references(mpTracker->mlpReferences.begin(),mpTracker->mlpReferences.end());
    if(ref) references.insert(ref);
    bool first=true;
    for(KeyFrame* original:references) {
        if(compact && !final) break;
        if(!original) continue;
        Sophus::SE3f worldFromReference;
        float referenceScale;
        Map* referenceMap=nullptr;
        if(!original->GetReplayReference(worldFromReference,referenceScale,referenceMap)) continue;
        if(!first) out << ","; first=false;
        out << "[" << original->mnId << "," << referenceMap->GetId() << ",";
        pose(worldFromReference);
        out << "," << referenceScale << ",";
        MarkerGraphTransform graph;
        if(original->GetReplayMarkerGraph(graph)) markerGraph(graph); else out << "null";
        out << ",";
        MarkerGraphTransform gauge;
        if(original->GetReplayMarkerGauge(gauge)) markerGraph(gauge); else out << "null";
        out << "]";
    }
    out << "],\"correction_rejections\":[";
    first=true;
    for(const auto &event:mpLoopCloser->GetCorrectionRejections()) {
        if(compact && !final) break;
        if(!first) out << ","; first=false;
        out << "{\"sequence\":" << event.sequence << ",\"timestamp\":" << event.timestamp
            << ",\"keyframe_id\":" << event.keyframeId << ",\"map_id\":" << event.mapId
            << ",\"other_map_id\":" << event.otherMapId << ",\"kind\":\"" << event.kind
            << "\",\"reason\":\"" << event.reason << "\"}";
    }
    out << "],\"marker_graph_capabilities\":{\"interval_scale_reanchor\":true,\"marker_map_merge\":true,"
           "\"rigid_marker_pose_optimization\":true,\"calibrated_rigid_board_optimization\":true,"
           "\"metric_tag_global_ba\":true,\"metric_tag_visual_loop\":true,"
           "\"metric_tag_pose_graph_factors\":true,"
           "\"metric_tag_visual_merge\":true}"
        << ",\"marker_map_aliases\":{";
    first=true;
    for(const auto& alias:mpAtlas->mMarkerMapAliases) {
        if(!first) out << ","; first=false;
        out << "\"" << alias.first << "\":" << alias.second;
    }
    out << "},\"marker_graph_events\":[";
    first=true;
    for(const auto& event:mpAtlas->mMarkerGraphEvents) {
        if(compact && !final) break;
        if(!first) out << ","; first=false;
        out << "{\"sequence\":" << event.sequence << ",\"type\":\"" << event.type
            << "\",\"status\":\"" << event.status << "\",\"reason\":\"" << event.reason
            << "\",\"frame\":" << event.frameId << ",\"timestamp\":" << event.timestamp
            << ",\"candidate_frame\":" << event.candidateFrameId << ",\"candidate_timestamp\":" << event.candidateTimestamp
            << ",\"map_id\":" << event.mapId << ",\"source_map_id\":" << event.sourceMapId
            << ",\"target_map_id\":" << event.targetMapId << ",\"revision\":" << event.revision
            << ",\"scale\":" << event.scale << ",\"sigma\":" << event.sigma
            << ",\"before_reprojection_error_px\":";
        residual(event.beforeTagRms);
        out << ",\"after_reprojection_error_px\":"; residual(event.afterTagRms);
        out << ",\"before_background_error_px\":"; residual(event.beforeBackgroundRms);
        out << ",\"after_background_error_px\":"; residual(event.afterBackgroundRms);
        out << ",\"marker_ids\":[";
        for(std::size_t i=0;i<event.markerIds.size();++i) { if(i) out << ","; out << event.markerIds[i]; }
        out << "],\"affected_keyframes\":[";
        for(std::size_t i=0;i<event.affectedKeyframes.size();++i) { if(i) out << ","; out << event.affectedKeyframes[i]; }
        out << "],\"excluded_tag_keyframes\":[";
        for(std::size_t i=0;i<event.excludedTagKeyframes.size();++i) { if(i) out << ","; out << event.excludedTagKeyframes[i]; }
        out << "],\"excluded_tag_groups\":[";
        for(std::size_t i=0;i<event.excludedTagGroups.size();++i) {
            if(i) out << ",";
            out << "{\"keyframe\":" << event.excludedTagGroups[i].first
                << ",\"marker_id\":" << event.excludedTagGroups[i].second << "}";
        }
        out << "],\"marker_residuals_px\":{";
        for(std::size_t i=0;i<event.diagnosticMarkerIds.size();++i) {
            if(i) out << ",";
            out << "\"" << event.diagnosticMarkerIds[i] << "\":{\"before\":";
            residual(i<event.beforeMarkerRms.size()?event.beforeMarkerRms[i]:0.);
            out << ",\"after\":";
            residual(i<event.afterMarkerRms.size()?event.afterMarkerRms[i]:0.);
            out << "}";
        }
        out << "},\"worst_tag_keyframe_id\":" << event.worstTagKeyframeId
            << ",\"worst_tag_marker_id\":" << event.worstTagMarkerId
            << ",\"worst_tag_rms_px\":";
        residual(event.worstTagRms);
        out << "}";
    }
    out << "]}\n";
}

void System::SaveMapPointsXYZ(const string &filename)
{
    cout << endl << "Saving map points to " << filename << " ..." << endl;
    ofstream f(filename.c_str());
    f << fixed;
    const vector<MapPoint*> points = mpAtlas->GetAllMapPoints();
    for(MapPoint* point : points)
    {
        if(point == nullptr || point->isBad())
            continue;
        const Eigen::Vector3f p = point->GetWorldPos();
        f << setprecision(9) << p(0) << " " << p(1) << " " << p(2) << endl;
    }
}

void System::SaveKeyFrameTrajectoryTUM(const string &filename)
{
    cout << endl << "Saving keyframe trajectory to " << filename << " ..." << endl;

    vector<KeyFrame*> vpKFs = mpAtlas->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];

       // pKF->SetPose(pKF->GetPose()*Two);

        if(pKF->isBad())
            continue;

        Sophus::SE3f Twc = pKF->GetPoseInverse();
        Eigen::Quaternionf q = Twc.unit_quaternion();
        Eigen::Vector3f t = Twc.translation();
        f << setprecision(6) << pKF->mTimeStamp << setprecision(7) << " " << t(0) << " " << t(1) << " " << t(2)
          << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;

    }

    f.close();
}

void System::SaveTrajectoryEuRoC(const string &filename)
{

    cout << endl << "Saving trajectory to " << filename << " ..." << endl;
    /*if(mSensor==MONOCULAR)
    {
        cerr << "ERROR: SaveTrajectoryEuRoC cannot be used for monocular." << endl;
        return;
    }*/

    vector<Map*> vpMaps = mpAtlas->GetAllMaps();
    int numMaxKFs = 0;
    Map* pBiggerMap;
    std::cout << "There are " << std::to_string(vpMaps.size()) << " maps in the atlas" << std::endl;
    for(Map* pMap :vpMaps)
    {
        std::cout << "  Map " << std::to_string(pMap->GetId()) << " has " << std::to_string(pMap->GetAllKeyFrames().size()) << " KFs" << std::endl;
        if(pMap->GetAllKeyFrames().size() > numMaxKFs)
        {
            numMaxKFs = pMap->GetAllKeyFrames().size();
            pBiggerMap = pMap;
        }
    }

    vector<KeyFrame*> vpKFs = pBiggerMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    Sophus::SE3f Twb; // Can be word to cam0 or world to b depending on IMU or not.
    if (mSensor==IMU_MONOCULAR || mSensor==IMU_STEREO || mSensor==IMU_RGBD)
        Twb = vpKFs[0]->GetImuPose();
    else
        Twb = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    // cout << "file open" << endl;
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    list<ORB_SLAM3::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    list<bool>::iterator lbL = mpTracker->mlbLost.begin();

    //cout << "size mlpReferences: " << mpTracker->mlpReferences.size() << endl;
    //cout << "size mlRelativeFramePoses: " << mpTracker->mlRelativeFramePoses.size() << endl;
    //cout << "size mpTracker->mlFrameTimes: " << mpTracker->mlFrameTimes.size() << endl;
    //cout << "size mpTracker->mlbLost: " << mpTracker->mlbLost.size() << endl;


    for(auto lit=mpTracker->mlRelativeFramePoses.begin(),
        lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++, lbL++)
    {
        //cout << "1" << endl;
        if(*lbL)
            continue;


        KeyFrame* pKF = *lRit;
        //cout << "KF: " << pKF->mnId << endl;

        Sophus::SE3f Trw;

        // If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
        if (!pKF)
            continue;

        //cout << "2.5" << endl;

        while(pKF->isBad())
        {
            //cout << " 2.bad" << endl;
            Trw = Trw * pKF->mTcp;
            pKF = pKF->GetParent();
            //cout << "--Parent KF: " << pKF->mnId << endl;
        }

        if(!pKF || pKF->GetMap() != pBiggerMap)
        {
            //cout << "--Parent KF is from another map" << endl;
            continue;
        }

        //cout << "3" << endl;

        Trw = Trw * pKF->GetPose()*Twb; // Tcp*Tpw*Twb0=Tcb0 where b0 is the new world reference

        // cout << "4" << endl;

        if (mSensor == IMU_MONOCULAR || mSensor == IMU_STEREO || mSensor==IMU_RGBD)
        {
            Sophus::SE3f Twb = (pKF->mImuCalib.mTbc * (*lit) * Trw).inverse();
            Eigen::Quaternionf q = Twb.unit_quaternion();
            Eigen::Vector3f twb = Twb.translation();
            f << setprecision(6) << 1e9*(*lT) << " " <<  setprecision(9) << twb(0) << " " << twb(1) << " " << twb(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
        }
        else
        {
            Sophus::SE3f Twc = ((*lit)*Trw).inverse();
            Eigen::Quaternionf q = Twc.unit_quaternion();
            Eigen::Vector3f twc = Twc.translation();
            f << setprecision(6) << 1e9*(*lT) << " " <<  setprecision(9) << twc(0) << " " << twc(1) << " " << twc(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
        }

        // cout << "5" << endl;
    }
    //cout << "end saving trajectory" << endl;
    f.close();
    cout << endl << "End of saving trajectory to " << filename << " ..." << endl;
}

void System::SaveTrajectoryEuRoC(const string &filename, Map* pMap)
{

    cout << endl << "Saving trajectory of map " << pMap->GetId() << " to " << filename << " ..." << endl;
    /*if(mSensor==MONOCULAR)
    {
        cerr << "ERROR: SaveTrajectoryEuRoC cannot be used for monocular." << endl;
        return;
    }*/

    int numMaxKFs = 0;

    vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    Sophus::SE3f Twb; // Can be word to cam0 or world to b dependingo on IMU or not.
    if (mSensor==IMU_MONOCULAR || mSensor==IMU_STEREO || mSensor==IMU_RGBD)
        Twb = vpKFs[0]->GetImuPose();
    else
        Twb = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    // cout << "file open" << endl;
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    list<ORB_SLAM3::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    list<bool>::iterator lbL = mpTracker->mlbLost.begin();

    //cout << "size mlpReferences: " << mpTracker->mlpReferences.size() << endl;
    //cout << "size mlRelativeFramePoses: " << mpTracker->mlRelativeFramePoses.size() << endl;
    //cout << "size mpTracker->mlFrameTimes: " << mpTracker->mlFrameTimes.size() << endl;
    //cout << "size mpTracker->mlbLost: " << mpTracker->mlbLost.size() << endl;


    for(auto lit=mpTracker->mlRelativeFramePoses.begin(),
        lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++, lbL++)
    {
        //cout << "1" << endl;
        if(*lbL)
            continue;


        KeyFrame* pKF = *lRit;
        //cout << "KF: " << pKF->mnId << endl;

        Sophus::SE3f Trw;

        // If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
        if (!pKF)
            continue;

        //cout << "2.5" << endl;

        while(pKF->isBad())
        {
            //cout << " 2.bad" << endl;
            Trw = Trw * pKF->mTcp;
            pKF = pKF->GetParent();
            //cout << "--Parent KF: " << pKF->mnId << endl;
        }

        if(!pKF || pKF->GetMap() != pMap)
        {
            //cout << "--Parent KF is from another map" << endl;
            continue;
        }

        //cout << "3" << endl;

        Trw = Trw * pKF->GetPose()*Twb; // Tcp*Tpw*Twb0=Tcb0 where b0 is the new world reference

        // cout << "4" << endl;

        if (mSensor == IMU_MONOCULAR || mSensor == IMU_STEREO || mSensor==IMU_RGBD)
        {
            Sophus::SE3f Twb = (pKF->mImuCalib.mTbc * (*lit) * Trw).inverse();
            Eigen::Quaternionf q = Twb.unit_quaternion();
            Eigen::Vector3f twb = Twb.translation();
            f << setprecision(6) << 1e9*(*lT) << " " <<  setprecision(9) << twb(0) << " " << twb(1) << " " << twb(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
        }
        else
        {
            Sophus::SE3f Twc = ((*lit)*Trw).inverse();
            Eigen::Quaternionf q = Twc.unit_quaternion();
            Eigen::Vector3f twc = Twc.translation();
            f << setprecision(6) << 1e9*(*lT) << " " <<  setprecision(9) << twc(0) << " " << twc(1) << " " << twc(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
        }

        // cout << "5" << endl;
    }
    //cout << "end saving trajectory" << endl;
    f.close();
    cout << endl << "End of saving trajectory to " << filename << " ..." << endl;
}

/*void System::SaveTrajectoryEuRoC(const string &filename)
{

    cout << endl << "Saving trajectory to " << filename << " ..." << endl;
    if(mSensor==MONOCULAR)
    {
        cerr << "ERROR: SaveTrajectoryEuRoC cannot be used for monocular." << endl;
        return;
    }

    vector<Map*> vpMaps = mpAtlas->GetAllMaps();
    Map* pBiggerMap;
    int numMaxKFs = 0;
    for(Map* pMap :vpMaps)
    {
        if(pMap->GetAllKeyFrames().size() > numMaxKFs)
        {
            numMaxKFs = pMap->GetAllKeyFrames().size();
            pBiggerMap = pMap;
        }
    }

    vector<KeyFrame*> vpKFs = pBiggerMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    Sophus::SE3f Twb; // Can be word to cam0 or world to b dependingo on IMU or not.
    if (mSensor==IMU_MONOCULAR || mSensor==IMU_STEREO || mSensor==IMU_RGBD)
        Twb = vpKFs[0]->GetImuPose_();
    else
        Twb = vpKFs[0]->GetPoseInverse_();

    ofstream f;
    f.open(filename.c_str());
    // cout << "file open" << endl;
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    list<ORB_SLAM3::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    list<bool>::iterator lbL = mpTracker->mlbLost.begin();

    //cout << "size mlpReferences: " << mpTracker->mlpReferences.size() << endl;
    //cout << "size mlRelativeFramePoses: " << mpTracker->mlRelativeFramePoses.size() << endl;
    //cout << "size mpTracker->mlFrameTimes: " << mpTracker->mlFrameTimes.size() << endl;
    //cout << "size mpTracker->mlbLost: " << mpTracker->mlbLost.size() << endl;


    for(list<Sophus::SE3f>::iterator lit=mpTracker->mlRelativeFramePoses.begin(),
        lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++, lbL++)
    {
        //cout << "1" << endl;
        if(*lbL)
            continue;


        KeyFrame* pKF = *lRit;
        //cout << "KF: " << pKF->mnId << endl;

        Sophus::SE3f Trw;

        // If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
        if (!pKF)
            continue;

        //cout << "2.5" << endl;

        while(pKF->isBad())
        {
            //cout << " 2.bad" << endl;
            Trw = Trw * pKF->mTcp;
            pKF = pKF->GetParent();
            //cout << "--Parent KF: " << pKF->mnId << endl;
        }

        if(!pKF || pKF->GetMap() != pBiggerMap)
        {
            //cout << "--Parent KF is from another map" << endl;
            continue;
        }

        //cout << "3" << endl;

        Trw = Trw * pKF->GetPose()*Twb; // Tcp*Tpw*Twb0=Tcb0 where b0 is the new world reference

        // cout << "4" << endl;


        if (mSensor == IMU_MONOCULAR || mSensor == IMU_STEREO || mSensor==IMU_RGBD)
        {
            Sophus::SE3f Tbw = pKF->mImuCalib.Tbc_ * (*lit) * Trw;
            Sophus::SE3f Twb = Tbw.inverse();

            Eigen::Vector3f twb = Twb.translation();
            Eigen::Quaternionf q = Twb.unit_quaternion();
            f << setprecision(6) << 1e9*(*lT) << " " <<  setprecision(9) << twb(0) << " " << twb(1) << " " << twb(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
        }
        else
        {
            Sophus::SE3f Tcw = (*lit) * Trw;
            Sophus::SE3f Twc = Tcw.inverse();

            Eigen::Vector3f twc = Twc.translation();
            Eigen::Quaternionf q = Twc.unit_quaternion();
            f << setprecision(6) << 1e9*(*lT) << " " <<  setprecision(9) << twc(0) << " " << twc(1) << " " << twc(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
        }

        // cout << "5" << endl;
    }
    //cout << "end saving trajectory" << endl;
    f.close();
    cout << endl << "End of saving trajectory to " << filename << " ..." << endl;
}*/


/*void System::SaveKeyFrameTrajectoryEuRoC_old(const string &filename)
{
    cout << endl << "Saving keyframe trajectory to " << filename << " ..." << endl;

    vector<Map*> vpMaps = mpAtlas->GetAllMaps();
    Map* pBiggerMap;
    int numMaxKFs = 0;
    for(Map* pMap :vpMaps)
    {
        if(pMap->GetAllKeyFrames().size() > numMaxKFs)
        {
            numMaxKFs = pMap->GetAllKeyFrames().size();
            pBiggerMap = pMap;
        }
    }

    vector<KeyFrame*> vpKFs = pBiggerMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];

       // pKF->SetPose(pKF->GetPose()*Two);

        if(pKF->isBad())
            continue;
        if (mSensor == IMU_MONOCULAR || mSensor == IMU_STEREO || mSensor==IMU_RGBD)
        {
            cv::Mat R = pKF->GetImuRotation().t();
            vector<float> q = Converter::toQuaternion(R);
            cv::Mat twb = pKF->GetImuPosition();
            f << setprecision(6) << 1e9*pKF->mTimeStamp  << " " <<  setprecision(9) << twb.at<float>(0) << " " << twb.at<float>(1) << " " << twb.at<float>(2) << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;

        }
        else
        {
            cv::Mat R = pKF->GetRotation();
            vector<float> q = Converter::toQuaternion(R);
            cv::Mat t = pKF->GetCameraCenter();
            f << setprecision(6) << 1e9*pKF->mTimeStamp << " " <<  setprecision(9) << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2) << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;
        }
    }
    f.close();
}*/

void System::SaveKeyFrameTrajectoryEuRoC(const string &filename)
{
    cout << endl << "Saving keyframe trajectory to " << filename << " ..." << endl;

    vector<Map*> vpMaps = mpAtlas->GetAllMaps();
    Map* pBiggerMap;
    int numMaxKFs = 0;
    for(Map* pMap :vpMaps)
    {
        if(pMap && pMap->GetAllKeyFrames().size() > numMaxKFs)
        {
            numMaxKFs = pMap->GetAllKeyFrames().size();
            pBiggerMap = pMap;
        }
    }

    if(!pBiggerMap)
    {
        std::cout << "There is not a map!!" << std::endl;
        return;
    }

    vector<KeyFrame*> vpKFs = pBiggerMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];

       // pKF->SetPose(pKF->GetPose()*Two);

        if(!pKF || pKF->isBad())
            continue;
        if (mSensor == IMU_MONOCULAR || mSensor == IMU_STEREO || mSensor==IMU_RGBD)
        {
            Sophus::SE3f Twb = pKF->GetImuPose();
            Eigen::Quaternionf q = Twb.unit_quaternion();
            Eigen::Vector3f twb = Twb.translation();
            f << setprecision(6) << 1e9*pKF->mTimeStamp  << " " <<  setprecision(9) << twb(0) << " " << twb(1) << " " << twb(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;

        }
        else
        {
            Sophus::SE3f Twc = pKF->GetPoseInverse();
            Eigen::Quaternionf q = Twc.unit_quaternion();
            Eigen::Vector3f t = Twc.translation();
            f << setprecision(6) << 1e9*pKF->mTimeStamp << " " <<  setprecision(9) << t(0) << " " << t(1) << " " << t(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
        }
    }
    f.close();
}

void System::SaveKeyFrameTrajectoryEuRoC(const string &filename, Map* pMap)
{
    cout << endl << "Saving keyframe trajectory of map " << pMap->GetId() << " to " << filename << " ..." << endl;

    vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];

        if(!pKF || pKF->isBad())
            continue;
        if (mSensor == IMU_MONOCULAR || mSensor == IMU_STEREO || mSensor==IMU_RGBD)
        {
            Sophus::SE3f Twb = pKF->GetImuPose();
            Eigen::Quaternionf q = Twb.unit_quaternion();
            Eigen::Vector3f twb = Twb.translation();
            f << setprecision(6) << 1e9*pKF->mTimeStamp  << " " <<  setprecision(9) << twb(0) << " " << twb(1) << " " << twb(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;

        }
        else
        {
            Sophus::SE3f Twc = pKF->GetPoseInverse();
            Eigen::Quaternionf q = Twc.unit_quaternion();
            Eigen::Vector3f t = Twc.translation();
            f << setprecision(6) << 1e9*pKF->mTimeStamp << " " <<  setprecision(9) << t(0) << " " << t(1) << " " << t(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
        }
    }
    f.close();
}

/*void System::SaveTrajectoryKITTI(const string &filename)
{
    cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
    if(mSensor==MONOCULAR)
    {
        cerr << "ERROR: SaveTrajectoryKITTI cannot be used for monocular." << endl;
        return;
    }

    vector<KeyFrame*> vpKFs = mpAtlas->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    cv::Mat Two = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    list<ORB_SLAM3::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    for(list<cv::Mat>::iterator lit=mpTracker->mlRelativeFramePoses.begin(), lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++)
    {
        ORB_SLAM3::KeyFrame* pKF = *lRit;

        cv::Mat Trw = cv::Mat::eye(4,4,CV_32F);

        while(pKF->isBad())
        {
            Trw = Trw * Converter::toCvMat(pKF->mTcp.matrix());
            pKF = pKF->GetParent();
        }

        Trw = Trw * pKF->GetPoseCv() * Two;

        cv::Mat Tcw = (*lit)*Trw;
        cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t();
        cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3);

        f << setprecision(9) << Rwc.at<float>(0,0) << " " << Rwc.at<float>(0,1)  << " " << Rwc.at<float>(0,2) << " "  << twc.at<float>(0) << " " <<
             Rwc.at<float>(1,0) << " " << Rwc.at<float>(1,1)  << " " << Rwc.at<float>(1,2) << " "  << twc.at<float>(1) << " " <<
             Rwc.at<float>(2,0) << " " << Rwc.at<float>(2,1)  << " " << Rwc.at<float>(2,2) << " "  << twc.at<float>(2) << endl;
    }
    f.close();
}*/

void System::SaveTrajectoryKITTI(const string &filename)
{
    cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
    if(mSensor==MONOCULAR)
    {
        cerr << "ERROR: SaveTrajectoryKITTI cannot be used for monocular." << endl;
        return;
    }

    vector<KeyFrame*> vpKFs = mpAtlas->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    Sophus::SE3f Tow = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    list<ORB_SLAM3::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    for(list<Sophus::SE3f>::iterator lit=mpTracker->mlRelativeFramePoses.begin(),
        lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++)
    {
        ORB_SLAM3::KeyFrame* pKF = *lRit;

        Sophus::SE3f Trw;

        if(!pKF)
            continue;

        while(pKF->isBad())
        {
            Trw = Trw * pKF->mTcp;
            pKF = pKF->GetParent();
        }

        Trw = Trw * pKF->GetPose() * Tow;

        Sophus::SE3f Tcw = (*lit) * Trw;
        Sophus::SE3f Twc = Tcw.inverse();
        Eigen::Matrix3f Rwc = Twc.rotationMatrix();
        Eigen::Vector3f twc = Twc.translation();

        f << setprecision(9) << Rwc(0,0) << " " << Rwc(0,1)  << " " << Rwc(0,2) << " "  << twc(0) << " " <<
             Rwc(1,0) << " " << Rwc(1,1)  << " " << Rwc(1,2) << " "  << twc(1) << " " <<
             Rwc(2,0) << " " << Rwc(2,1)  << " " << Rwc(2,2) << " "  << twc(2) << endl;
    }
    f.close();
}


void System::SaveDebugData(const int &initIdx)
{
    // 0. Save initialization trajectory
    SaveTrajectoryEuRoC("init_FrameTrajectoy_" +to_string(mpLocalMapper->mInitSect)+ "_" + to_string(initIdx)+".txt");

    // 1. Save scale
    ofstream f;
    f.open("init_Scale_" + to_string(mpLocalMapper->mInitSect) + ".txt", ios_base::app);
    f << fixed;
    f << mpLocalMapper->mScale << endl;
    f.close();

    // 2. Save gravity direction
    f.open("init_GDir_" +to_string(mpLocalMapper->mInitSect)+ ".txt", ios_base::app);
    f << fixed;
    f << mpLocalMapper->mRwg(0,0) << "," << mpLocalMapper->mRwg(0,1) << "," << mpLocalMapper->mRwg(0,2) << endl;
    f << mpLocalMapper->mRwg(1,0) << "," << mpLocalMapper->mRwg(1,1) << "," << mpLocalMapper->mRwg(1,2) << endl;
    f << mpLocalMapper->mRwg(2,0) << "," << mpLocalMapper->mRwg(2,1) << "," << mpLocalMapper->mRwg(2,2) << endl;
    f.close();

    // 3. Save computational cost
    f.open("init_CompCost_" +to_string(mpLocalMapper->mInitSect)+ ".txt", ios_base::app);
    f << fixed;
    f << mpLocalMapper->mCostTime << endl;
    f.close();

    // 4. Save biases
    f.open("init_Biases_" +to_string(mpLocalMapper->mInitSect)+ ".txt", ios_base::app);
    f << fixed;
    f << mpLocalMapper->mbg(0) << "," << mpLocalMapper->mbg(1) << "," << mpLocalMapper->mbg(2) << endl;
    f << mpLocalMapper->mba(0) << "," << mpLocalMapper->mba(1) << "," << mpLocalMapper->mba(2) << endl;
    f.close();

    // 5. Save covariance matrix
    f.open("init_CovMatrix_" +to_string(mpLocalMapper->mInitSect)+ "_" +to_string(initIdx)+".txt", ios_base::app);
    f << fixed;
    for(int i=0; i<mpLocalMapper->mcovInertial.rows(); i++)
    {
        for(int j=0; j<mpLocalMapper->mcovInertial.cols(); j++)
        {
            if(j!=0)
                f << ",";
            f << setprecision(15) << mpLocalMapper->mcovInertial(i,j);
        }
        f << endl;
    }
    f.close();

    // 6. Save initialization time
    f.open("init_Time_" +to_string(mpLocalMapper->mInitSect)+ ".txt", ios_base::app);
    f << fixed;
    f << mpLocalMapper->mInitTime << endl;
    f.close();
}


int System::GetTrackingState()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackingState;
}

bool System::WaitForLocalMappingIdle(int maxMilliseconds)
{
    if(!mpLocalMapper || maxMilliseconds <= 0)
        return true;
    // A marker-graph stop is itself a mapper safe point. Otherwise wait for
    // every submitted keyframe to finish culling, triangulation, local BA and
    // publication to LoopClosing. Queue-empty alone is insufficient because
    // ProcessNewKeyFrame pops before those mutations begin.
    if(mpLocalMapper->isStoppedForTagAlignment() || mpLocalMapper->isFinished())
        return true;
    return mpLocalMapper->WaitUntilKeyFramesProcessed(maxMilliseconds);
}

bool System::WaitForLoopClosingIdle(int maxMilliseconds)
{
    if(!mpLoopCloser || maxMilliseconds <= 0)
        return true;
    const auto deadline = std::chrono::steady_clock::now() +
        std::chrono::milliseconds(maxMilliseconds);
    while(std::chrono::steady_clock::now() < deadline)
    {
        if(mpLoopCloser->IsIdle()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return mpLoopCloser->IsIdle();
}

std::size_t System::GetLoopClosingQueueSize()
{
    return mpLoopCloser ? mpLoopCloser->KeyframesInQueue() : 0;
}

bool System::OfflineFinalizationConverged()
{
    return mpLocalMapper && mpLoopCloser && mpLocalMapper->isFinished() &&
        mpLoopCloser->isFinished() && !mpLoopCloser->isRunningGBA() &&
        mpLoopCloser->KeyframesInQueue()==0;
}

void System::SelectLargestMapForOfflineFinalization()
{
    Map* selected=nullptr;
    for(Map* map:mpAtlas->GetAllMaps())
        if(map && !map->IsBad() && (!selected || map->KeyFramesInMap()>selected->KeyFramesInMap()))
            selected=map;
    if(selected) mpAtlas->ChangeMap(selected);
}

vector<MapPoint*> System::GetTrackedMapPoints()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackedMapPoints;
}

vector<cv::KeyPoint> System::GetTrackedKeyPointsUn()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackedKeyPointsUn;
}

double System::GetTimeFromIMUInit()
{
    double aux = mpLocalMapper->GetCurrKFTime()-mpLocalMapper->mFirstTs;
    if ((aux>0.) && mpAtlas->isImuInitialized())
        return mpLocalMapper->GetCurrKFTime()-mpLocalMapper->mFirstTs;
    else
        return 0.f;
}

bool System::isLost()
{
    if (!mpAtlas->isImuInitialized())
        return false;
    else
    {
        if ((mpTracker->mState==Tracking::LOST)) //||(mpTracker->mState==Tracking::RECENTLY_LOST))
            return true;
        else
            return false;
    }
}


bool System::isFinished()
{
    return (GetTimeFromIMUInit()>0.1);
}

void System::ChangeDataset()
{
    if(mpAtlas->GetCurrentMap()->KeyFramesInMap() < 12)
    {
        mpTracker->ResetActiveMap();
    }
    else
    {
        mpTracker->CreateMapInAtlas();
    }

    mpTracker->NewDataset();
}

void System::StartNewMapForMarkerComponent()
{
    mpTracker->CreateMapInAtlas();
    mpTracker->NewDataset();
}

unsigned long System::GetCurrentMapId()
{
    Map* map = mpAtlas->GetCurrentMap();
    return map ? map->GetId() : std::numeric_limits<unsigned long>::max();
}

float System::GetImageScale()
{
    return mpTracker->GetImageScale();
}

void System::SetFeatureMask(const cv::Mat &mask)
{
    mpTracker->SetFeatureMask(mask);
}

void System::SetExternalTagObservation(
    const Sophus::SE3f &Twc,
    const float confidence,
    const std::vector<Eigen::Vector3f> &worldPoints,
    const std::vector<cv::Point2f> &imagePoints,
    const bool valid,
    const std::vector<float> &pointWeights,
    const std::vector<int> &tagIds,
    bool partial, int trackedCorners, float trackAgeS, bool inputInAtlasWorld)
{
    unique_lock<mutex> correctionGate(mpAtlas->mMutexPoseGraphCorrection);
    Map* map=mpAtlas->GetCurrentMap();
    unique_lock<mutex> mapLock(map->mMutexMapUpdate);
    mpTracker->SetExternalTagObservation(
        Twc, confidence, worldPoints, imagePoints, valid, pointWeights, tagIds, partial, trackedCorners, trackAgeS,
        inputInAtlasWorld);
}

float System::GetRecoveredTagMetricScale()
{
    return mpTracker->GetRecoveredTagMetricScale();
}

bool System::IsTagMetricAligned()
{
    return mpTracker->IsTagMetricAligned();
}

unsigned int System::GetTagKeyFramesAccepted()
{
    return mpTracker->GetTagKeyFramesAccepted();
}

unsigned int System::GetTagKeyFramesRejected()
{
    return mpTracker->GetTagKeyFramesRejected();
}

unsigned int System::GetTagPoseConstraintsApplied()
{
    return mpTracker->GetTagPoseConstraintsApplied();
}

#ifdef REGISTER_TIMES
void System::InsertRectTime(double& time)
{
    mpTracker->vdRectStereo_ms.push_back(time);
}

void System::InsertResizeTime(double& time)
{
    mpTracker->vdResizeImage_ms.push_back(time);
}

void System::InsertTrackTime(double& time)
{
    mpTracker->vdTrackTotal_ms.push_back(time);
}
#endif

void System::SaveAtlas(int type){
    if(!mStrSaveAtlasToFile.empty())
    {
        //clock_t start = clock();

        // Save the current session
        mpAtlas->PreSave();

        string pathSaveFileName = mStrSaveAtlasToFile;
        if(pathSaveFileName.size()<4 || pathSaveFileName.substr(pathSaveFileName.size()-4)!=".osa")
            pathSaveFileName += ".osa";

        string strVocabularyChecksum = CalculateCheckSum(mStrVocabularyFilePath,TEXT_FILE);
        std::size_t found = mStrVocabularyFilePath.find_last_of("/\\");
        string strVocabularyName = mStrVocabularyFilePath.substr(found+1);

        if(type == TEXT_FILE) // File text
        {
            cout << "Starting to write the save text file " << endl;
            std::remove(pathSaveFileName.c_str());
            std::ofstream ofs(pathSaveFileName, std::ios::binary);
            boost::archive::text_oarchive oa(ofs);

            oa << strVocabularyName;
            oa << strVocabularyChecksum;
            oa << std::string("marker-orb-atlas/v2");
            oa << mpAtlas;
            cout << "End to write the save text file" << endl;
        }
        else if(type == BINARY_FILE) // File binary
        {
            cout << "Starting to write the save binary file" << endl;
            std::remove(pathSaveFileName.c_str());
            std::ofstream ofs(pathSaveFileName, std::ios::binary);
            boost::archive::binary_oarchive oa(ofs);
            oa << strVocabularyName;
            oa << strVocabularyChecksum;
            oa << std::string("marker-orb-atlas/v2");
            oa << mpAtlas;
            cout << "End to write save binary file" << endl;
        }
    }
}

bool System::LoadAtlas(int type)
{
    string strFileVoc, strVocChecksum;
    bool isRead = false;

    string pathLoadFileName = mStrLoadAtlasFromFile;
    if(pathLoadFileName.size()<4 || pathLoadFileName.substr(pathLoadFileName.size()-4)!=".osa")
        pathLoadFileName += ".osa";

    if(type == TEXT_FILE) // File text
    {
        cout << "Starting to read the save text file " << endl;
        std::ifstream ifs(pathLoadFileName, std::ios::binary);
        if(!ifs.good())
        {
            cout << "Load file not found" << endl;
            return false;
        }
        boost::archive::text_iarchive ia(ifs);
        ia >> strFileVoc;
        ia >> strVocChecksum;
        std::string format; ia >> format;
        if(format!="marker-orb-atlas/v1" && format!="marker-orb-atlas/v2") return false;
        ia >> mpAtlas;
        cout << "End to load the save text file " << endl;
        isRead = true;
    }
    else if(type == BINARY_FILE) // File binary
    {
        cout << "Starting to read the save binary file"  << endl;
        std::ifstream ifs(pathLoadFileName, std::ios::binary);
        if(!ifs.good())
        {
            cout << "Load file not found" << endl;
            return false;
        }
        boost::archive::binary_iarchive ia(ifs);
        ia >> strFileVoc;
        ia >> strVocChecksum;
        std::string format; ia >> format;
        if(format!="marker-orb-atlas/v1" && format!="marker-orb-atlas/v2") return false;
        ia >> mpAtlas;
        cout << "End to load the save binary file" << endl;
        isRead = true;
    }

    if(isRead)
    {
        //Check if the vocabulary is the same
        string strInputVocabularyChecksum = CalculateCheckSum(mStrVocabularyFilePath,TEXT_FILE);

        if(strInputVocabularyChecksum.compare(strVocChecksum) != 0)
        {
            cout << "The vocabulary load isn't the same which the load session was created " << endl;
            cout << "-Vocabulary name: " << strFileVoc << endl;
            return false; // Both are differents
        }

        mpAtlas->SetKeyFrameDababase(mpKeyFrameDatabase);
        mpAtlas->SetORBVocabulary(mpVocabulary);
        mpAtlas->PostLoad();

        return true;
    }
    return false;
}

string System::CalculateCheckSum(string filename, int type)
{
    string checksum = "";

    unsigned char c[MD5_DIGEST_LENGTH];

    std::ios_base::openmode flags = std::ios::in;
    if(type == BINARY_FILE) // Binary file
        flags = std::ios::in | std::ios::binary;

    ifstream f(filename.c_str(), flags);
    if ( !f.is_open() )
    {
        cout << "[E] Unable to open the in file " << filename << " for Md5 hash." << endl;
        return checksum;
    }

    MD5_CTX md5Context;
    char buffer[1024];

    MD5_Init (&md5Context);
    while ( int count = f.readsome(buffer, sizeof(buffer)))
    {
        MD5_Update(&md5Context, buffer, count);
    }

    f.close();

    MD5_Final(c, &md5Context );

    for(int i = 0; i < MD5_DIGEST_LENGTH; i++)
    {
        char aux[10];
        sprintf(aux,"%02x", c[i]);
        checksum = checksum + aux;
    }

    return checksum;
}

} //namespace ORB_SLAM
