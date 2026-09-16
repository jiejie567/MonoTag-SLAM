#ifndef ORB_SLAM3_MARKER_GRAPH_OPTIMIZER_H
#define ORB_SLAM3_MARKER_GRAPH_OPTIMIZER_H

#include <cstddef>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <sophus/se3.hpp>
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

namespace ORB_SLAM3 {

class KeyFrame;
class Map;
class MapPoint;

// All methods are proposals only. The caller must keep the referenced maps,
// cameras, raw observations and KF/MP lifetimes stable until computation ends.
class MarkerGraphOptimizer {
public:
    // Sophus::SE3f and Eigen::Vector3f are fixed-size/aligned objects.  The
    // default node allocator used by libstdc++ is not required to preserve
    // their alignment in C++14 builds, which can turn an otherwise valid BA
    // proposal into a platform-dependent gauge drift or crash.  Keep every
    // public map carrying these values explicitly aligned.
    using PoseEntry = std::pair<KeyFrame* const, Sophus::SE3f>;
    using PointEntry = std::pair<MapPoint* const, Eigen::Vector3f>;
    using TagCornerEntry = std::pair<KeyFrame* const, std::vector<Eigen::Vector3f>>;
    using StaticTagEntry = std::pair<const int, std::vector<Eigen::Vector3f>>;
    using PoseMap = std::map<KeyFrame*, Sophus::SE3f, std::less<KeyFrame*>,
                             Eigen::aligned_allocator<PoseEntry>>;
    using PointMap = std::map<MapPoint*, Eigen::Vector3f, std::less<MapPoint*>,
                              Eigen::aligned_allocator<PointEntry>>;
    using TagCornerMap = std::map<KeyFrame*, std::vector<Eigen::Vector3f>, std::less<KeyFrame*>,
                                  Eigen::aligned_allocator<TagCornerEntry>>;
    using StaticTagMap = std::map<int, std::vector<Eigen::Vector3f>, std::less<int>,
                                  Eigen::aligned_allocator<StaticTagEntry>>;
    using ScaleMap = std::map<KeyFrame*, float>;

    struct Options {
        // Regression A/B only; production resolves stale partial geometry.
        bool canonicalizeWeakMarkerCorners = true;
        bool retryIsolatedMarkerGroups = true;
        // Regression A/B only: independent markers must not all become a
        // surveyed rigid layout just because their observing KF is fixed.
        bool fixAllObservedGaugeMarkers = false;
        // Regression A/B only. Independent markers are unsurveyed rigid
        // landmarks, not world-pose priors. The fixed camera defines gauge;
        // known marker dimensions and shared observations determine scale.
        bool legacyIndependentMarkerWorldPrior = false;
        // Only independently validated multi-marker corner geometry may
        // authorize an interval repair beyond the ordinary scale envelope.
        // The fixed gauge and all post-BA validation still apply.
        bool allowLargeCornerScaleRepair = true;
        // A known world marker may independently validate a natural-feature
        // Sim3 using a second, translated strong view. Admission is not commit.
        bool allowLargeKnownMarkerLoopRepair = true;
        // One bounded post-BA rescue of background points that crossed behind
        // a camera, followed by the same joint BA and unchanged validation.
        // False retains the original solver path for regression A/B.
        bool repairBackgroundCheirality = true;
        // Match ordinary ORB EssentialGraph sparsification. Keep
        // every parent/loop edge, and only covisibility links of weight >=100.
        // This selects initialization priors, never raw BA observations;
        // false retains the legacy graph for explicit regression A/B.
        bool useEssentialGraphCovisibility = true;
        // An earlier tag rejection must not mask independently
        // failed background validation of NEW, uncommitted loop aliases.
        // False retains the original retry trigger for regression A/B.
        bool retryLoopAliasesAfterTagFailure = true;
        int graphIterations = 30;
        int baIterations = 15;
        int covisibleNeighbors = 10;
        double maximumTagRmsPx = 2.5;
        double maximumBackgroundRmsPx = 3.0;
        double maximumBackgroundRmsIncreasePx = 0.5;
        double minimumPositiveDepthFraction = 0.98;
        double maximumAnchorTranslationM = 0.002;
        double maximumAnchorRotationRad = 0.005;
        double maximumScaleAnchorRelativeError = 0.03;
        // Legacy/rigid-board policy only; not a prior on independent markers.
        double maximumMarkerCornerDisplacementM = 0.10;
        double minimumScale = 0.5;
        double maximumScale = 2.0;
    };

    struct ResidualSummary {
        std::size_t tagCorners = 0;
        std::size_t backgroundObservations = 0;
        std::size_t positiveDepth = 0;
        double tagRmsPx = 0.0;
        double backgroundRmsPx = 0.0;
        double positiveDepthFraction = 0.0;
        std::map<KeyFrame*, double> tagRmsByKeyframe;
        std::map<int, double> tagRmsByMarker;
        std::map<std::pair<KeyFrame*, int>, double> tagRmsByKeyframeMarker;
        std::map<KeyFrame*, double> backgroundRmsByKeyframe;
        std::map<KeyFrame*, double> backgroundNormalizedRmsByKeyframe;
    };

    struct StagedBAInput {
        // Offline full-map refinement may continue while the robust objective
        // is still decreasing. Online/window solves retain their fixed budget.
        bool convergeOffline = false;
        std::vector<KeyFrame*> keyframes;
        std::set<KeyFrame*> fixedKeyframes;
        // A calibrated board has one rigid pose variable; online-discovered
        // marker maps retain one variable per marker.
        bool rigidMarkerLayout = false;
        // Independent station markers first seen after the last accepted
        // scale anchor have unsurveyed world placement until interval BA
        // validates them. Registration alone does not establish that pose.
        std::set<int> provisionalMarkerIds;
        // Initial metricization must validate stored tag factors before the
        // live map marks them active. Ordinary BA leaves this false.
        bool includeInactiveTagObservations = false;
        // Initial metricization may retry after rejecting a whole damaged tag
        // keyframe. Ordinary reanchor/merge inputs leave this empty.
        std::set<KeyFrame*> excludedTagKeyframes;
        // Retry-only exclusions; do not erase the live keyframe observation.
        std::set<std::pair<KeyFrame*, int>> excludedTagGroups;
        // A final full-map solve may expose a small number of stale ORB
        // association groups. Retry without that keyframe's background
        // pixels, while retaining its camera, marker and graph constraints.
        std::set<KeyFrame*> excludedBackgroundKeyframes;
        // Interval retry: discard only independently contradicted pixels,
        // not their keyframe, marker observations, or the whole map point.
        std::set<std::pair<KeyFrame*, MapPoint*>> excludedBackgroundObservations;
        // A native ORB map can retain stale/high-octave observations until its
        // next culling pass. Initial metricization uses only observations that
        // are already chi-square inliers under the unchanged visual geometry.
        bool filterInitialBackgroundOutliers = false;
        bool useCommittedAdmission = false;
        // Trial point fusion: source observations constrain the destination
        // vertex without changing any live associations before acceptance.
        std::map<MapPoint*,MapPoint*> pointAliases;
        // Boundary keyframes are deliberately held fixed during an interval
        // correction. Their pre-existing stale observations must neither
        // steer the changed interval nor veto an otherwise improving result.
        bool filterFixedBackgroundOutliers = false;
        // Missing KF poses use their current Tcw. Only explicitly supplied
        // points are optimized; raw observation indices/pixels stay unchanged.
        PoseMap keyframePoses;
        PointMap pointPositions;
        // Optional same-order replacements for mvTagWorldPoints. These stay
        // as the initial geometry (e.g. a source map in a target gauge).
        TagCornerMap tagWorldCorners;
        ScaleMap replayScaleMultipliers;
    };

    struct Proposal {
        std::map<MapPoint*,MapPoint*> pointAliases;
        bool accepted = false;
        std::string reason;
        PoseMap keyframePoses;
        PointMap pointPositions;
        TagCornerMap tagWorldCorners;
        StaticTagMap staticTags;
        ScaleMap replayScaleMultipliers;
        std::vector<int> optimizedMarkerIds;
        std::vector<unsigned long> excludedTagKeyFrameIds;
        std::vector<std::pair<unsigned long,int>> excludedTagGroupIds;
        std::vector<unsigned long> excludedBackgroundKeyFrameIds;
        ResidualSummary before;
        ResidualSummary after;
        std::vector<unsigned long> affectedKeyFrameIds;
        double cornerScale = 0, cornerScaleSigma = 0;
    };

    struct CornerScaleEvidence {
        bool valid=false;
        double scale=1, sigma=1, rms=0;
        std::size_t markers=0;
    };
    // Triangulate decoded rigid corners in the current visual map, then
    // compare their dimensions with the physical marker. Leave-one-view-out
    // reprojection validates the estimate; no live geometry is modified.
    static CornerScaleEvidence EstimateCornerScale(const std::vector<KeyFrame*>& views);
    // Arbitrary-unit initialization: physical parallax/cross-validation apply,
    // but the unit magnitude is not an extreme correction to a metric map.
    static CornerScaleEvidence EstimateInitialCornerScale(const std::vector<KeyFrame*>& views);

    struct KnownMarkerLoopEvidence {
        bool valid=false;
        int markerId=-1;
        unsigned long holdoutKeyframeId=0;
        double selfRms=0, holdoutRms=0, baselineM=0, logScaleTolerance=0;
    };
    // Does not fit corners or a PnP pose. Checks a frozen, naturally estimated
    // seed against canonical corners observed at the fixed origin and in two
    // local strong views. The caller still supplies native-validated matches.
    static KnownMarkerLoopEvidence ValidateKnownMarkerLoopScale(Map* map,
        KeyFrame* current, const g2o::Sim3& seed,
        const std::vector<MapPoint*>& matches, const Options& options);

    // metricPerVisual is a local correction (e.g. .9 shrinks visual lengths
    // by 10% at B); sigma is its positive relative/log-scale uncertainty.
    // A is fixed. A->B spanning-tree paths and their covisible neighborhood
    // receive a spatial Sim3 correction, never a time-interpolated rescale.
    // Returned replay multipliers follow observable post-BA local depth
    // changes; B must agree with the measured scale. Fixed gauges stay at 1.
    static Proposal Reanchor(Map* map, KeyFrame* anchorA,
                             const std::vector<KeyFrame*>& anchorB,
                             double metricPerVisual, double sigma,
                             const Options& options);
    static Proposal Reanchor(Map* map, KeyFrame* anchorA,
                             const std::vector<KeyFrame*>& anchorB,
                             double metricPerVisual, double sigma);

    // Shared staged SE3-camera/XYZ-point BA using raw ORB and fixed-tag pixel
    // factors. No real pose, point, observation, flag or scale is written.
    static Proposal RefineAndValidate(const StagedBAInput& input,
                                      const Options& options,
                                      const ResidualSummary* committedBaseline = nullptr);
    static Proposal RefineAndValidate(const StagedBAInput& input);
    // Final-map-only policy; ordinary initialization/reanchor/loop admission
    // retain the world-displacement guard. Also exposes frozen-graph A/B.
    static Proposal RefineAndValidate(const StagedBAInput& input,
                                      const Options& options,
                                      const ResidualSummary* committedBaseline,
                                      bool observerRelativeMarkerGate);
    static Proposal RefineAndValidate(const StagedBAInput& input,
                                      const Options& options,
                                      const ResidualSummary* committedBaseline,
                                      bool observerRelativeMarkerGate,
                                      bool finalBackgroundPolicy);

    // Use a robust closed-form visual->metric Sim(3) only as the initial
    // estimate, then jointly refine all live ORB points/keyframes against the
    // stored raw fixed-tag corners. This method never mutates the map.
    static Proposal InitializeMetric(Map* map,
                                     const Sophus::SE3f& metricWorldFromVisualWorld,
                                     double metricPerVisual,
                                     const Options& options);
    static Proposal InitializeMetric(Map* map,
                                     const Sophus::SE3f& metricWorldFromVisualWorld,
                                     double metricPerVisual);

    // Full-map metric BA used by the offline finalization path. Static marker
    // dimensions stay fixed while multi-view marker SE3 poses, camera poses
    // and ORB points are optimized in one gauge-fixed problem.
    static Proposal RefineMetricMap(Map* map, const Options& options);
    static Proposal RefineMetricMap(Map* map, const Options& options,
                                   bool observerRelativeMarkerGate);
    static Proposal RefineMetricMap(Map* map, const Options& options,
                                   bool observerRelativeMarkerGate, bool finalBackgroundPolicy);
    static Proposal RefineMetricMap(Map* map);
    static Proposal ProposeVisualLoop(Map* map, KeyFrame* current, KeyFrame* matched,
        const g2o::Sim3& currentFromWorld, const std::vector<MapPoint*>& matches);
    static Proposal ProposeVisualLoop(Map* map, KeyFrame* current, KeyFrame* matched,
        const g2o::Sim3& currentFromWorld, const std::vector<MapPoint*>& matches,
        const Options& options);
};

} // namespace ORB_SLAM3

#endif
