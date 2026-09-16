#!/usr/bin/env bash
set -euo pipefail

PROJECT=$(cd "$(dirname "$0")/.." && pwd)
DEPENDENCIES=${1:?Usage: run_native_smoke.sh DEPENDENCY_INSTALL_PREFIX}
NATIVE="$PROJECT/third_party/ORB_SLAM3"

export LD_LIBRARY_PATH="$DEPENDENCIES/pangolin/lib:$DEPENDENCIES/opencv-4.10/lib:$NATIVE/lib:$NATIVE/Thirdparty/DBoW2/lib:$NATIVE/Thirdparty/g2o/lib:${LD_LIBRARY_PATH:-}"

for test_name in \
  marker_graph_regression \
  marker_map_merge_regression \
  marker_graph_coordinator_regression \
  atlas_scale_regression \
  portable_frontend_regression \
  portable_random_regression
do
  echo "[native-smoke] $test_name"
  "$NATIVE/Examples/Monocular/$test_name"
done
