#ifndef ORB_SLAM3_MARKER_MAP_MERGE_H
#define ORB_SLAM3_MARKER_MAP_MERGE_H

#include "MarkerGraphOptimizer.h"

namespace ORB_SLAM3 {

class Map;

// Computes a proposal only. The caller stops local mapping and holds the Atlas
// correction gate plus both maps' update locks until validation/commit ends.
// Marker IDs must name the same fixed physical marker throughout the Atlas.
class MarkerMapMerge {
public:
    using StaticTags = std::map<int, std::vector<float>>;

    struct Options {
        std::size_t minimumIndependentFrames = 3;
        double minimumTimeSpanS = 0.1;
        double minimumConfidence = 0.35;
        double maximumMarkerRmsPx = 3.0;
        // Sizes are known, not estimated from images. Allow only numerical
        // error from storing the same rigid square in another float gauge.
        double maximumSizeRelativeError = 1e-4;
        double maximumLayoutErrorM = 0.005;
        MarkerGraphOptimizer::Options optimizer;
    };

    struct MarkerEvidence {
        std::size_t targetFrames = 0;
        std::size_t sourceFrames = 0;
        double targetTimeSpanS = 0.0;
        double sourceTimeSpanS = 0.0;
        double targetRmsPx = 0.0;
        double sourceRmsPx = 0.0;
        double targetSideM = 0.0;
        double sourceSideM = 0.0;
        double alignmentRmsM = 0.0;
        double alignmentMaximumM = 0.0;
    };

    struct Proposal {
        bool accepted = false;
        std::string reason;
        unsigned long targetMapId = 0;
        unsigned long sourceMapId = 0;
        unsigned long targetRevision = 0;
        unsigned long sourceRevision = 0;
        // X_target = sourceToTarget * X_source. Always SE3, never a scale.
        Sophus::SE3f sourceToTarget;
        std::vector<int> commonMarkerIds;
        std::vector<int> verifiedMarkerIds;
        std::map<int, MarkerEvidence> evidence;
        // Target registry unchanged; source-only markers transformed rigidly.
        StaticTags staticTags;
        MarkerGraphOptimizer::Proposal graph;
    };

    static Proposal Propose(Map* target, Map* source, const Options& options);
    static Proposal Propose(Map* target, Map* source);
};

} // namespace ORB_SLAM3

#endif
