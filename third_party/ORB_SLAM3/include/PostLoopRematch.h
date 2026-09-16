#ifndef ORB_SLAM3_POST_LOOP_REMATCH_H
#define ORB_SLAM3_POST_LOOP_REMATCH_H
namespace ORB_SLAM3 {
class Map;
class MapPoint;
class KeyFrame;
namespace PostLoopRematch {
// Read-only, conservative check against all stored monocular observations.
bool ConsistentDuplicate(MapPoint* first, MapPoint* second);
// Called only after a validated visual loop and its essential-graph correction.
// apply=false produces diagnostics without changing observations or points.
void Run(Map* map, KeyFrame* first, KeyFrame* last, bool apply);
}
}
#endif
