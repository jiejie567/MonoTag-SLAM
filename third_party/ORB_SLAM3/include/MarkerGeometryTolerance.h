#pragma once
#include <Eigen/Core>
#include <limits>

namespace ORB_SLAM3 {
// Corners have already been stored as float world coordinates. Subtract in
// double, and budget the float rounding of BOTH endpoints. This is numerical
// precision, not permission to change a physical marker's size.
inline double MarkerDistanceRoundoff(const Eigen::Vector3d& a,
                                    const Eigen::Vector3d& b)
{
    return std::numeric_limits<float>::epsilon()*(a.norm()+b.norm());
}
}
