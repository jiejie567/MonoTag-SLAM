# Ubuntu installation

Use Ubuntu 24.04 and Python 3.10 or newer. macOS capture/viewing utilities are
not the validated reconstruction runtime. GPU hand inference is optional;
ordinary MonoTag reconstruction does not require a GPU or learned weights.

## Python and system dependencies

```bash
sudo apt-get install build-essential cmake git python3-venv python3-dev \
  libeigen3-dev libboost-serialization-dev libssl-dev ffmpeg
make setup
```

Build OpenCV 4.10 (including video, calib3d and features2d) and Pangolin in a
dependency prefix. Follow their upstream build instructions; this release does
not install system packages or download third-party dependencies automatically.
The build helper expects:

```text
DEPS_PREFIX/
  opencv-4.10/lib/cmake/opencv4/OpenCVConfig.cmake
  pangolin/lib/cmake/Pangolin/PangolinConfig.cmake
```

Native OpenCV and the pinned Python opencv-contrib package serve different
parts of the pipeline. Retain both versions for the reference configuration.

## Native build and runtime

```bash
make native DEPS_PREFIX=/path/to/dependencies JOBS=4
make native-smoke DEPS_PREFIX=/path/to/dependencies
make smoke
```

Obtain ORBvoc.txt from the official ORB-SLAM3 distribution under its applicable
terms. Then bind this checkout to its own newly built binaries:

```bash
.venv/bin/python scripts/init_runtime.py \
  --deps-prefix /path/to/dependencies --vocabulary /path/to/ORBvoc.txt
```

This creates a local, ignored `.monotag/runtime.json` and a vocabulary symlink.
It checks required files and records binary hashes. It does not download hand
models, change algorithm settings, or replace an existing runtime. A rebuild
requires explicit rebinding; do not reuse another machine's runtime JSON.

## Calibration and processing

Use your measured marker black-square size and camera intrinsics. The hardware
repository provides printable targets, not universal camera calibration.
Choose a fresh output directory for each run:

```bash
.venv/bin/python process_monotag.py input.mp4 \
  --calib camera.json --head-slam --slam-init auto --auto-marker-map \
  --static-marker-ids 20-49 --static-marker-size-mm 48 \
  --no-hand-joints --slam-replay --save-atlas runs/demo/atlas.osa \
  --output runs/demo/actions.jsonl
```

The 48 mm size and ID range are examples, not defaults suitable for every
recording. For full exporter options use `tools/export_action_labels.py --help`.
`--no-hand-joints` skips learned hand inference, not geometric wrist-marker
observations. Hand inference requires separately obtained licensed models;
MANO assets and model weights are not distributed here.

```bash
.venv/bin/python tools/render_slam_replay.py runs/demo/actions.jsonl --hybrid
```

Use the local replay server rather than opening the HTML via `file://`.
See `tools/replay_orb_slam.py --help` for the serving interface. The renderer needs
the original video, calibration and cached native replay referenced by the
actions metadata; an actions JSONL alone is not a self-contained replay package.
