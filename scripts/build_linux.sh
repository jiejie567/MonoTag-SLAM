#!/usr/bin/env bash
# Dependencies must already be installed; never patches algorithm source.
set -euo pipefail
PROJECT=$(cd "$(dirname "$0")/.." && pwd)
DEPENDENCIES=${1:?Usage: build_linux.sh DEPENDENCY_INSTALL_PREFIX [jobs]}
BUILD_JOBS=${2:-4}
NATIVE="$PROJECT/third_party/ORB_SLAM3"
COMMON=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17
  '-DCMAKE_CXX_FLAGS=-DEIGEN_MAX_ALIGN_BYTES=16 -DEIGEN_MAX_STATIC_ALIGN_BYTES=16'
  "-DOpenCV_DIR=$DEPENDENCIES/opencv-4.10/lib/cmake/opencv4"
  "-DPangolin_DIR=$DEPENDENCIES/pangolin/lib/cmake/Pangolin")
cmake -S "$NATIVE/Thirdparty/DBoW2" -B "$NATIVE/build-dbow-linux" "${COMMON[@]}"
cmake --build "$NATIVE/build-dbow-linux" -j "$BUILD_JOBS"
cmake -S "$NATIVE" -B "$NATIVE/build" "${COMMON[@]}"
cmake --build "$NATIVE/build" -j "$BUILD_JOBS" --target mono_tum_headless relocalize_prefix_readonly relocalize_gap \
  marker_graph_regression marker_map_merge_regression marker_graph_coordinator_regression atlas_scale_regression \
  portable_frontend_regression portable_random_regression
