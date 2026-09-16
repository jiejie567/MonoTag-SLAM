#include "MarkerComponentRegistration.h"
#include <iostream>
#include <stdexcept>
using namespace ORB_SLAM3;
static void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
int main() {
    const std::vector<Eigen::Vector3f> local{{-.024f,.024f,0},{.024f,.024f,0},{.024f,-.024f,0},{-.024f,-.024f,0}};
    const Sophus::SE3f old(Eigen::Quaternionf::Identity(),Eigen::Vector3f(12,0,-5));
    const Sophus::SE3f committed(Eigen::Quaternionf(Eigen::AngleAxisf(1.2f,Eigen::Vector3f::UnitY())),Eigen::Vector3f(12.5f,-.1f,-5.1f));
    std::vector<float> corners;for(auto p:local){auto q=committed*p;for(int i=0;i<3;++i)corners.push_back(q[i]);}
    auto recovered=old;require(RefreshMarkerComponentTransform(local,corners,recovered),"valid known marker rejected");
    require((recovered.matrix()-committed.matrix()).norm()<1e-4,"BA component gauge not recovered");
    const Eigen::Vector3f unseen(-.63f,.018f,0);
    require((old*unseen-committed*unseen).norm()>.5,"fixture did not expose stale registration");
    require((recovered*unseen-committed*unseen).norm()<1e-4,"new marker retained stale world position");
    const Sophus::SE3f correction=recovered*old.inverse();
    // A mixed batch can include a new complete marker and incomplete corners
    // of its known neighbour. Both went through the same stale component map.
    for(int j:{0,2,3})require((correction*(old*local[j])-committed*local[j]).norm()<1e-4,
        "partial known-marker corners retained stale component coordinates");
    const Sophus::SE3f camera(Eigen::Quaternionf::Identity(),Eigen::Vector3f(0,0,-1));
    require(((recovered*camera).inverse()*(recovered*unseen)-camera.inverse()*unseen).norm()<1e-5,"camera-relative measurement changed");
    auto bad=corners;bad[0]+=.01f;auto unchanged=recovered;
    require(!RefreshMarkerComponentTransform(local,bad,recovered),"damaged square accepted");
    require((unchanged.matrix()-recovered.matrix()).norm()==0,"rejection mutated gauge");
    require(!RefreshMarkerComponentTransform({},corners,recovered),"unknown reference invented");
    std::cout<<"COMPONENT_REFRESH_OK: new ID / BA gauge / camera-relative invariant / invalid unchanged\n";
}
