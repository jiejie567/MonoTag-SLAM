#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

namespace ORB_SLAM3 {
// A trust-region check on a SAME-FRAME local BA proposal, not a constraint on
// camera motion between frames. Explicit interval/loop Sim3 optimization is
// separate and must remain capable of correcting metric scale.
inline bool LocalMetricDepthChangeAccepted(std::vector<double> ratios,
                                          double& median)
{
    median = 1.0;
    if(ratios.size() < 30) return true; // insufficient evidence for this check
    for(double ratio : ratios)
        if(!std::isfinite(ratio)) return false;
    auto middle = ratios.begin() + ratios.size()/2;
    std::nth_element(ratios.begin(), middle, ratios.end());
    median = *middle;
    return median >= 0.8 && median <= 1.25;
}
}
