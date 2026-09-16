#ifndef ORB_SLAM3_MARKER_GRAPH_STATE_H
#define ORB_SLAM3_MARKER_GRAPH_STATE_H

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>
#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>

namespace ORB_SLAM3 {

// Only committed marker-graph corrections enter this accumulator. Ordinary
// visual BA must not change it: reliable absolute marker measurements in the
// replay are deliberately protected from unrelated reference-KF motion.
struct MarkerGraphTransform {
    unsigned long sequence = 0;
    double scale = 1.0;
    Eigen::Quaterniond rotation = Eigen::Quaterniond::Identity();
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();

    template<class Archive> void serialize(Archive& ar, const unsigned int) {
        ar & sequence & scale;
        ar & rotation.x() & rotation.y() & rotation.z() & rotation.w();
        ar & translation.x() & translation.y() & translation.z();
    }
    bool finite() const {
        return std::isfinite(scale) && scale > 0 && translation.allFinite() &&
            rotation.coeffs().allFinite() && std::abs(rotation.norm()-1.0)<1e-4;
    }
    MarkerGraphTransform inverse() const {
        MarkerGraphTransform r;
        r.sequence=sequence; r.scale=1.0/scale; r.rotation=rotation.conjugate();
        r.translation=-(r.rotation*translation)/scale;
        return r;
    }
    MarkerGraphTransform operator*(const MarkerGraphTransform& other) const {
        MarkerGraphTransform r;
        r.sequence=std::max(sequence,other.sequence);
        r.scale=scale*other.scale;
        r.rotation=(rotation*other.rotation).normalized();
        r.translation=scale*(rotation*other.translation)+translation;
        return r;
    }
    Sophus::SE3f apply(const Sophus::SE3f& worldFromCamera) const {
        return Sophus::SE3f((rotation*worldFromCamera.unit_quaternion().cast<double>()).cast<float>(),
            (scale*(rotation*worldFromCamera.translation().cast<double>())+translation).cast<float>());
    }
    static MarkerGraphTransform between(const Sophus::SE3f& beforeTwc,
            const Sophus::SE3f& afterTwc, double localScale, unsigned long event) {
        MarkerGraphTransform r;
        r.sequence=event; r.scale=localScale;
        r.rotation=(afterTwc.unit_quaternion().cast<double>()*
                    beforeTwc.unit_quaternion().cast<double>().conjugate()).normalized();
        r.translation=afterTwc.translation().cast<double>()-
            localScale*(r.rotation*beforeTwc.translation().cast<double>());
        return r;
    }
};

struct MarkerGraphEvent {
    unsigned long sequence = 0;
    std::string type, status, reason;
    long frameId = -1, candidateFrameId = -1;
    double timestamp = 0.0, candidateTimestamp = 0.0;
    long mapId = -1, sourceMapId = -1, targetMapId = -1;
    unsigned long revision = 0;
    double scale = 1.0, sigma = 0.0;
    double beforeTagRms = 0.0, afterTagRms = 0.0;
    double beforeBackgroundRms = 0.0, afterBackgroundRms = 0.0;
    std::vector<int> markerIds;
    std::vector<unsigned long> affectedKeyframes;
    // Run-local diagnostics. They are deliberately not serialized so older
    // Atlas archives remain readable; a loaded archive simply lacks the
    // original run's detailed rejection breakdown.
    long worstTagKeyframeId = -1;
    int worstTagMarkerId = -1;
    double worstTagRms = 0.0;
    std::vector<int> diagnosticMarkerIds;
    std::vector<double> beforeMarkerRms, afterMarkerRms;
    std::vector<unsigned long> excludedTagKeyframes;
    std::vector<std::pair<unsigned long,int>> excludedTagGroups;
    template<class Archive> void serialize(Archive& ar, const unsigned int) {
        ar & sequence & type & status & reason & frameId & candidateFrameId;
        ar & timestamp & candidateTimestamp & mapId & sourceMapId & targetMapId & revision;
        ar & scale & sigma & beforeTagRms & afterTagRms & beforeBackgroundRms & afterBackgroundRms;
        ar & markerIds & affectedKeyframes;
    }
};

} // namespace ORB_SLAM3
#endif
