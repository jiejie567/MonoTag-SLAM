#pragma once
#include <cmath>
namespace ORB_SLAM3 {
inline bool RetainComponentMeasurements(double now,double first,double last,
                                        bool healthy,bool sameMap,bool unchangedGauge) {
    return healthy && sameMap && unchangedGauge && std::isfinite(now) &&
        std::isfinite(first) && std::isfinite(last) && first<=last &&
        now>=last && now-last<=.5 && now-first<=1.;
}
}
