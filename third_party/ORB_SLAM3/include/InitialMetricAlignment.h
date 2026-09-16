#pragma once
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace ORB_SLAM3 {
// At a known scale and rotation, fit the translation to the ORIGINAL paired
// observations. An alignment fitted at another scale is not an anchor.
inline bool RefitInitialMetricTranslation(
    const std::vector<std::pair<Eigen::Vector3f,Eigen::Vector3f>>& visualMetric,
    const Eigen::Matrix3f& rotation, float scale, Eigen::Vector3f& translation)
{
    if(visualMetric.size()<3 || !rotation.allFinite() ||
       !std::isfinite(scale) || scale<=0) return false;
    std::vector<float> coordinates[3];
    for(const auto& sample:visualMetric) {
        if(!sample.first.allFinite() || !sample.second.allFinite()) return false;
        const Eigen::Vector3f offset=sample.second-scale*rotation*sample.first;
        for(int axis=0;axis<3;++axis) coordinates[axis].push_back(offset[axis]);
    }
    Eigen::Vector3f result;
    for(int axis=0;axis<3;++axis) {
        auto& values=coordinates[axis];
        std::sort(values.begin(),values.end());
        const size_t mid=values.size()/2;
        result[axis]=values.size()%2 ? values[mid] : .5f*(values[mid-1]+values[mid]);
    }
    if(!result.allFinite()) return false;
    translation=result;
    return true;
}
}
