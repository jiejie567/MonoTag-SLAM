// Independent layout admission must not depend on having emitted a loop event.
#include "MarkerGraphCoordinator.h"
#include "Map.h"
#include "CameraModels/Pinhole.h"
#include <iostream>
#include <stdexcept>

static void require(bool ok, const char* message) {
    if(!ok) throw std::runtime_error(message);
}
int main() {
    using namespace ORB_SLAM3;
    Pinhole camera({700,700,320,240});
    for(int sequence:{0,1,9}) {
        Map map(0); map.mbMetric=true; map.mbRigidMarkerLayout=false;
        map.mnMarkerGraphSequence=sequence;
        std::vector<Eigen::Vector3f> cached, canonical;
        std::vector<int> ids; std::vector<cv::Point2f> pixels;
        for(int tag=0;tag<2;++tag) for(const auto& offset:
                std::vector<Eigen::Vector2f>{{-1,-1},{1,-1},{1,1},{-1,1}}) {
            Eigen::Vector3f p(.2f*tag+offset.x()*.024f,offset.y()*.024f,1.f);
            cached.push_back(p); p.z()+=tag*.035f; canonical.push_back(p);
            ids.push_back(20+tag);
            for(int axis=0;axis<3;++axis) map.mStaticTags[20+tag].push_back(p[axis]);
            Eigen::Vector2f uv=camera.project(p);pixels.emplace_back(uv.x(),uv.y());
        }
        auto run=[&](std::vector<Eigen::Vector3f> corners,std::vector<cv::Point2f> image,
                     bool expected) {
            Sophus::SE3f pose;std::string reason;auto original=corners;
            const auto gauge=map.mMarkerInputToWorld;
            bool ok=MarkerGraphCoordinator::AlignMarkerInput(&map,pose,corners,ids,
                {},false,reason,&camera,image);
            require(ok==expected,"unexpected marker layout admission");
            require(map.mnMarkerGraphSequence==sequence,"admission fabricated graph event");
            if(ok) {
                require(corners==canonical,"factor retained stale world corners");
                require(pose.translation().norm()<1e-4,"wrong measured pose");
            } else {
                require(corners==original,"rejection mutated corners");
                require((map.mMarkerInputToWorld.matrix()-gauge.matrix()).norm()<1e-6,
                        "rejection mutated gauge");
            }
        };
        run(cached,pixels,true);
        map.mbRigidMarkerLayout=true;run(cached,pixels,false);
        map.mbRigidMarkerLayout=false;
        auto bad=cached;bad[1].x()+=.01f;run(bad,pixels,false);
        auto bad_pixels=pixels;bad_pixels[0].x+=80;run(cached,bad_pixels,false);
    }
    std::cout<<"PASS marker layout initialization / independent / rigid / size / pixels / no fake event\n";
}
