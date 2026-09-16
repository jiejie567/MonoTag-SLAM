#pragma once
#include <cmath>
#include <cstddef>

namespace ORB_SLAM3 {
// Candidate scheduling only. No interpolation, no odometry, no acceptance relaxation.
inline bool RetainSparseMarkerWindow(double gap, double span, bool commonStrongId) {
    return commonStrongId && std::isfinite(gap) && std::isfinite(span) &&
        gap >= 0.0 && gap <= 1.0 && span >= 0.0 && span <= 3.0;
}
inline bool TrySparseMarkerCorners(bool fragmented, std::size_t observations,
                                   std::size_t keyframes, double baseline) {
    return fragmented && observations >= 8 && keyframes >= 2 &&
        std::isfinite(baseline) && baseline >= .04;
}
}
