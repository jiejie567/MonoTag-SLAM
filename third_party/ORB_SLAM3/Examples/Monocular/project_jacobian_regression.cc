// Exercise the production BA edge against central finite differences.
// No images, vocabulary, Atlas, mapper, or tracking worker is needed.
#include "OptimizableTypes.h"
#include "CameraModels/Pinhole.h"
#include "Thirdparty/g2o/g2o/core/jacobian_workspace.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

int main()
{
    using Edge = ORB_SLAM3::EdgeSE3ProjectXYZ;
    using Base = g2o::BaseBinaryEdge<2,Eigen::Vector2d,
        g2o::VertexSBAPointXYZ,g2o::VertexSE3Expmap>;
    ORB_SLAM3::Pinhole camera(std::vector<float>{1089.5f,1088.3f,970.7f,554.3f});
    double maximumPointError=0.,maximumPoseError=0.;
    bool finite=true;
    for(int test=0;test<40;++test) {
        g2o::VertexSBAPointXYZ point;
        g2o::VertexSE3Expmap pose;
        const Eigen::Vector3d position(.13*(test%5)-.2,.09*(test%7)-.3,
                                      .4+.17*test);
        const g2o::SE3Quat transform(
            Eigen::Quaterniond(Eigen::AngleAxisd(.013*test,
                Eigen::Vector3d(.3,.2,.1).normalized())),
            Eigen::Vector3d(.02*(test%3),-.03*(test%4),.07));
        point.setEstimate(position);
        pose.setEstimate(transform);
        Edge edge;
        edge.pCamera=&camera;
        edge.setVertex(0,&point);
        edge.setVertex(1,&pose);
        edge.setMeasurement(camera.project(transform.map(position))+
                            Eigen::Vector2d(.2,-.1));
        edge.setInformation(Eigen::Matrix2d::Identity());
        g2o::JacobianWorkspace workspace;
        workspace.updateSize(&edge);
        if(!workspace.allocate()) throw std::runtime_error("Jacobian workspace allocation failed");
        // Base overload binds storage and dispatches the real virtual method
        // defined in src/OptimizableTypes.cpp, not an independent formula.
        static_cast<Base&>(edge).linearizeOplus(workspace);
        const Eigen::Matrix<double,2,3> analyticPoint=edge.jacobianOplusXi();
        const Eigen::Matrix<double,2,6> analyticPose=edge.jacobianOplusXj();
        Eigen::Matrix<double,2,3> numericPoint;
        Eigen::Matrix<double,2,6> numericPose;
        const double h=1e-6;
        for(int axis=0;axis<3;++axis) {
            Eigen::Vector3d offset=Eigen::Vector3d::Zero();
            offset[axis]=h;
            point.setEstimate(position+offset); edge.computeError();
            const Eigen::Vector2d plus=edge.error();
            point.setEstimate(position-offset); edge.computeError();
            numericPoint.col(axis)=(plus-edge.error())/(2*h);
        }
        point.setEstimate(position);
        for(int axis=0;axis<6;++axis) {
            Eigen::Matrix<double,6,1> offset=Eigen::Matrix<double,6,1>::Zero();
            offset[axis]=h;
            pose.setEstimate(g2o::SE3Quat::exp(offset)*transform); edge.computeError();
            const Eigen::Vector2d plus=edge.error();
            pose.setEstimate(g2o::SE3Quat::exp(-offset)*transform); edge.computeError();
            numericPose.col(axis)=(plus-edge.error())/(2*h);
        }
        pose.setEstimate(transform);
        finite=finite && analyticPoint.allFinite() && analyticPose.allFinite();
        const double pointError=(analyticPoint-numericPoint).cwiseAbs().maxCoeff();
        const double poseError=(analyticPose-numericPose).cwiseAbs().maxCoeff();
        maximumPointError=std::max(maximumPointError,pointError);
        maximumPoseError=std::max(maximumPoseError,poseError);
        if(test==0 || pointError>1e-3 || poseError>1e-3)
            std::cout << "case=" << test << " point_max_abs=" << pointError
                      << " pose_max_abs=" << poseError << '\n';
    }
    std::cout << std::setprecision(12)
              << "{\"cases\":40,\"point_max_abs\":" << maximumPointError
              << ",\"pose_max_abs\":" << maximumPoseError
              << ",\"finite\":" << (finite?"true":"false") << "}" << std::endl;
    return finite && maximumPointError<1e-3 && maximumPoseError<1e-3 ? 0 : 1;
}
