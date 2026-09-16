#ifndef ORB_SLAM3_MARKER_COMPONENT_REGISTRATION_H
#define ORB_SLAM3_MARKER_COMPONENT_REGISTRATION_H

#include <sophus/se3.hpp>
#include <Eigen/SVD>
#include <vector>

namespace ORB_SLAM3 {

// Refresh a cached component gauge from ONE already committed marker. Other
// independent markers remain BA variables; this is only a new-ID pose seed.
inline bool RefreshMarkerComponentTransform(const std::vector<Eigen::Vector3f>& local,
    const std::vector<float>& world, Sophus::SE3f& transform)
{
    if(local.size()!=4 || world.size()!=12)return false;
    Eigen::Matrix<double,3,4> a,b;
    for(int j=0;j<4;++j) {
        a.col(j)=local[j].cast<double>();
        b.col(j)<<world[j*3],world[j*3+1],world[j*3+2];
    }
    if(!a.allFinite() || !b.allFinite())return false;
    const double side=(a.col(1)-a.col(0)).norm();
    if(side<1e-5)return false;
    for(int j=0;j<4;++j)for(int k=j+1;k<4;++k) {
        const double expected=side*((k-j==2)?std::sqrt(2.):1.);
        if(std::abs((a.col(k)-a.col(j)).norm()-expected)>1e-4 ||
           std::abs((b.col(k)-b.col(j)).norm()-expected)>1e-4)return false;
    }
    const Eigen::Vector3d ca=a.rowwise().mean(),cb=b.rowwise().mean();
    const Eigen::Matrix3d covariance=(a.colwise()-ca)*(b.colwise()-cb).transpose();
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(covariance,Eigen::ComputeFullU|Eigen::ComputeFullV);
    if(svd.singularValues()[1]<1e-10)return false;
    Eigen::Matrix3d sign=Eigen::Matrix3d::Identity();
    sign(2,2)=(svd.matrixV()*svd.matrixU().transpose()).determinant()<0?-1:1;
    const Eigen::Matrix3d R=svd.matrixV()*sign*svd.matrixU().transpose();
    const Eigen::Vector3d t=cb-R*ca;
    if(((R*a).colwise()+t-b).norm()/2>1e-4)return false;
    const Sophus::SE3f result(Eigen::Quaternionf(R.cast<float>()).normalized(),t.cast<float>());
    if(!result.matrix().allFinite())return false;
    transform=result;return true;
}

// Both transforms map the SAME marker component into the SAME Atlas map.
// Compare that component's origin positions, not the translation of the
// world-space correction candidate*reference.inverse(). The latter includes
// (I-R)*world_origin and spuriously rejects tiny rotations far from world zero.
inline bool MarkerComponentTransformsConsistent(const Sophus::SE3f& reference,
                                                const Sophus::SE3f& candidate)
{
    if(!reference.matrix().allFinite() || !candidate.matrix().allFinite())
        return false;
    const float translation=(candidate.translation()-reference.translation()).norm();
    const float rotation=(candidate.so3()*reference.so3().inverse()).log().norm();
    return translation<=0.03f && rotation<=0.174533f;
}

} // namespace ORB_SLAM3
#endif
