#ifndef ORB_SLAM3_MARKER_GRAPH_COORDINATOR_H
#define ORB_SLAM3_MARKER_GRAPH_COORDINATOR_H

#include "MarkerGraphState.h"
#include "MarkerGraphOptimizer.h"
#include "MarkerMapMerge.h"
#include <map>
#include <set>
#include <vector>
#include <Eigen/StdVector>
#include <opencv2/core/types.hpp>

namespace ORB_SLAM3 {
class Atlas;
class Tracking;
class Map;
class KeyFrame;
class GeometricCamera;

// Called after Track(), under the Atlas correction gate, without Track's map
// lock. Requests commit only after LocalMapping acknowledges their owned stop.
class MarkerGraphCoordinator {
public:
    struct ScaleSample {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        unsigned long frameId = 0;
        double timestamp = 0;
        int correctionEpoch = 0;
        Sophus::SE3f visualTwc, markerTwc;
    };
    using ScaleSampleVector = std::vector<ScaleSample,
                                          Eigen::aligned_allocator<ScaleSample>>;
    struct ScaleEvidence {
        bool reliable = false;
        // A geometrically consistent multi-view metric observation remains
        // useful as a new interval anchor even when its scale is already
        // statistically indistinguishable from one.
        bool geometricallyValid = false;
        std::string reason;
        double metricPerVisual = 1.0, sigma = 1.0, baselineM = 0;
        std::size_t observations = 0;
    };
    // Only raw-KF selection metadata survives short decoding gaps. No copied
    // camera pose or inter-segment motion ratio is retained by this window.
    struct CornerRevisitWindow {
        Map* map = nullptr;
        int correctionEpoch = -1;
        unsigned long firstFrame = 0, lastFrame = 0;
        double firstTime = 0, lastTime = 0;
        bool fragmented = false;
        std::set<int> markerIds;
        void Observe(Map* currentMap, unsigned long frame, double timestamp,
                     const std::set<int>& strongIds);
    };
    explicit MarkerGraphCoordinator(Tracking& tracker) : tracker_(tracker) {}
    void OnFrameEnd(bool final = false);
    void Cancel();
    static ScaleEvidence EstimateScale(const ScaleSampleVector& samples);
    // Compatibility overload: copy ordinary storage into aligned Eigen storage.
    static ScaleEvidence EstimateScale(const std::vector<ScaleSample>& samples);
    static bool ShouldScheduleScale(const ScaleEvidence& evidence, bool intervalClosure);
    static bool CanTryCornerInterval(const ScaleEvidence& evidence, bool intervalClosure,
                                     std::size_t keyframes);
    static bool CanRetryScale(std::size_t previousViews, std::size_t currentViews,
                             unsigned attempts, double elapsedSeconds);
    static std::vector<unsigned long> SelectCornerRevisit(
        const ScaleEvidence& evidence, const CornerRevisitWindow& window,
        KeyFrame* anchor, const std::vector<KeyFrame*>& strongKeyframes);
    // Caller holds correction gate, map-update locks, and stops local mapping.
    static bool CommitScale(Atlas& atlas, Map* map,
        const MarkerGraphOptimizer::Proposal& proposal, KeyFrame* nextAnchor,
        MarkerGraphEvent& event);
    static bool CommitMerge(Atlas& atlas, Map* target, Map* source,
        const MarkerMapMerge::Proposal& proposal, MarkerGraphEvent& event);
    static bool CommitRefine(Atlas& atlas, Map* map,
        const MarkerGraphOptimizer::Proposal& proposal, MarkerGraphEvent& event);
    // Incoming known-layout hints may use a different rigid gauge from Atlas.
    // Registered strong IDs determine SE3 only; dimensions cannot be changed.
    // With image evidence, independently refined Atlas corners supersede the
    // cached layout and the pose hint is refitted in the same committed world.
    static bool AlignMarkerInput(Map* map, Sophus::SE3f& Twc,
        std::vector<Eigen::Vector3f>& corners, const std::vector<int>& ids,
        const std::vector<float>& weights, bool partial, std::string& reason,
        GeometricCamera* camera=nullptr, const std::vector<cv::Point2f>& pixels={},
        bool inputInAtlasWorld=false);
private:
    enum class Kind { None, Scale, Merge, Refine };
    Tracking& tracker_;
    Map* observedMap_ = nullptr;
    int observedCorrectionEpoch_ = -1;
    int pendingCorrectionEpoch_ = -1;
    ScaleSampleVector samples_;
    CornerRevisitWindow cornerRevisit_;
    Kind pendingKind_ = Kind::None;
    Map* pendingSource_ = nullptr;
    Map* pendingTarget_ = nullptr;
    Map* pendingActive_ = nullptr;
    std::vector<unsigned long> pendingB_;
    ScaleEvidence pendingScale_;
    long candidateFrame_ = -1;
    double candidateTime_ = 0;
    bool ownsStop_ = false;
    std::map<std::pair<unsigned long,unsigned long>,std::pair<unsigned long,unsigned long>> mergeAttempts_;
    std::map<unsigned long,unsigned long> scaleAttempts_;
    // Retry geometry-limited intervals once after a successful final BA.
    std::map<unsigned long,std::vector<unsigned long>> deferredScaleWindows_;
    // A marker episode starts after the fixed-marker stream has been absent
    // for long enough to represent a genuine revisit.  This lets a repeated
    // registered anchor close a near-unit metric interval once, without
    // re-running graph optimization throughout one continuous observation.
    long lastMarkerObservationFrame_ = -1;
    double lastMarkerObservationTime_ = -1.0;
    unsigned long markerEpisode_ = 0;
    std::map<unsigned long,unsigned long> scaleAnchorEpisodes_;
    std::map<unsigned long,unsigned long> scaleAttemptEpisodes_;
    struct ScaleRetry {
        unsigned long episode=0;
        unsigned attempts=0;
        std::set<unsigned long> frames;
        double timestamp=0;
    };
    std::map<unsigned long,ScaleRetry> scaleRetries_;
    std::map<unsigned long,std::pair<unsigned long,unsigned long>> refineAttempts_;
    void Schedule(Kind kind, Map* source, Map* target);
    void ProcessPending(bool final);
};
} // namespace ORB_SLAM3
#endif
