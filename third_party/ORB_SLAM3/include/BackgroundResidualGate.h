#ifndef ORB_SLAM3_BACKGROUND_RESIDUAL_GATE_H
#define ORB_SLAM3_BACKGROUND_RESIDUAL_GATE_H
#include <cmath>
namespace ORB_SLAM3 {
// RMS of whitened 2-D residuals: sqrt(mean(||pixel error||^2 / octave sigma^2)).
// Do not compare mixed pyramid octaves to a level-zero pixel threshold.
inline bool BackgroundResidualFrameConsistent(double before, double after)
{
    return std::isfinite(before) && std::isfinite(after) && before>=0 && after>=0 &&
        !(before<=3.0 && after>3.0) && after<=before+.5;
}
// Final joint BA redistributes error between views. A healthy frame need not
// improve individually; keep the same absolute noise envelope. Pre-existing
// out-of-envelope damage still must not materially worsen. Other BA paths
// keep BackgroundResidualFrameConsistent unchanged.
inline bool FinalBackgroundResidualFrameConsistent(double before, double after)
{
    return std::isfinite(before) && std::isfinite(after) && before>=0 && after>=0 &&
        (after<=3.0 || BackgroundResidualFrameConsistent(before,after));
}
}
#endif
