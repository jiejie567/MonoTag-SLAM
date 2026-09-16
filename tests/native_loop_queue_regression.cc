// Exercise the production LoopClosing queue, without starting SLAM or workers.
#include "KeyFrame.h"
#include "LoopClosing.h"
#include "Map.h"
#include "MapPoint.h"
#include "ORBextractor.h"
#include "CameraModels/Pinhole.h"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace ORB_SLAM3;

namespace {
void require(bool condition, const char* message) {
    if(!condition) throw std::runtime_error(message);
}
}

int main() {
    try {
        std::vector<std::unique_ptr<MapPoint>> owned;
        std::vector<MapPoint*> points;
        for(int index=0; index<15; ++index) {
            owned.emplace_back(new MapPoint());
            owned.back()->nObs=1;
            points.push_back(owned.back().get());
        }
        ORBVocabulary vocabulary;
        ORBextractor extractor(200, 1.2f, 8, 20, 7);
        Pinhole camera(std::vector<float>{500.f, 500.f, 320.f, 240.f});
        cv::Mat distortion=cv::Mat::zeros(4, 1, CV_32F);
        cv::Mat image(480, 640, CV_8UC1);
        cv::RNG random(271828);
        random.fill(image, cv::RNG::UNIFORM, 0, 256);
        Frame frame(image, 0., &extractor, &vocabulary, &camera, distortion, 0.f, 40.f);
        frame.SetPose(Sophus::SE3f());
        require(frame.N>=15, "fixture must extract at least 15 features");
        for(std::size_t index=0; index<points.size(); ++index)
            frame.mvpMapPoints[index]=points[index];
        Map map;
        KeyFrame origin(frame, &map, nullptr), supported(frame, &map, nullptr);
        KeyFrame weak(frame, &map, nullptr);
        origin.mnId=0;
        supported.mnId=1;
        weak.mnId=2;
        weak.EraseMapPointMatch(14);
        require(supported.TrackedMapPoints(1)==15, "fixture must have 15 observed points");
        require(weak.TrackedMapPoints(1)==14, "fixture must have 14 observed points");

        LoopClosing disabled(nullptr, nullptr, nullptr, false, false);
        require(disabled.IsIdle(), "disabled loop closer must start idle");
        for(int frame=0; frame<382; ++frame) {
            disabled.InsertKeyFrame(&supported);
            require(disabled.KeyframesInQueue()==0, "disabled loop closer enqueued a keyframe");
            require(disabled.IsIdle(), "disabled loop closer stopped being idle");
        }

        LoopClosing enabled(nullptr, nullptr, nullptr, false, true);
        enabled.InsertKeyFrame(&origin);
        enabled.InsertKeyFrame(&weak);
        require(enabled.KeyframesInQueue()==0, "enabled origin/support filters changed");
        enabled.InsertKeyFrame(&supported);
        require(enabled.KeyframesInQueue()==1, "enabled supported keyframe must enqueue");
        require(!enabled.IsIdle(), "enabled pending queue must not be idle");
        enabled.InsertKeyFrame(&weak);
        require(enabled.KeyframesInQueue()==1, "weak keyframe unexpectedly enqueued");
        enabled.InsertKeyFrame(&supported);
        require(enabled.KeyframesInQueue()==2, "enabled queue semantics changed");
        std::cout << "LOOP_QUEUE_SAFETY_OK: disabled 382 inserts stay idle; "
                     "enabled origin/14/15-point filters and queue semantics retained\n";
        return 0;
    } catch(const std::exception& error) {
        std::cerr << "LOOP_QUEUE_SAFETY_FAIL: " << error.what() << '\n';
        return 1;
    }
}
