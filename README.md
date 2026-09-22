<div align="center">

# MonoTag SLAM
**Marker-aided monocular metric reconstruction**

<a href="https://anyverse.com/"><img src="docs/images/anyverse-dynamics-logo.png" width="280" alt="Anyverse Dynamics"></a>

[Capture system & hardware](https://github.com/jiejie567/MonoEgo) · [Installation](docs/INSTALL.md) · [Reproducibility](REPRODUCIBILITY.md)

</div>

English | [中文](README_CN.md)

MonoTag combines natural image features with known-size square markers to reconstruct metric camera motion from one RGB camera. Built on ORB-SLAM3, it uses marker corners in geometric optimization, revisits earlier frames offline, and maintains an Atlas across multiple maps.

**MonoTag-SLAM** hosts the publicly available backend; **MonoEgo** is the complete capture system.

## Video demonstrations

Click a preview to open its MP4. These are excerpts of existing offline results, not real-time performance demonstrations. The left Atlas panel shows the final map; the right panel shows recorded construction and committed updates. Privacy masking is retained.

### Retrospective map matches (12 s)

[![Offline retrospective recovery](docs/videos/retrospective.gif)](docs/videos/retrospective.mp4)

Later map evidence recovers earlier image-to-map matches. This example is not a guarantee of frame-zero recovery, nor a vanilla ORB-SLAM3 benchmark.

### Metric re-anchoring (8 s)

[![Metric re-anchoring](docs/videos/reanchoring.gif)](docs/videos/reanchoring.mp4)

Accepted update at source time 121.73 s: scale ×1.0295, with highlighted geometry.

### Common-anchor merge and visual loop closure (18 s)

[![Map merge and loop update](docs/videos/merge-loop.gif)](docs/videos/merge-loop.mp4)

Maps merge at source time 187.73 s; a visual-loop update is published at 188.40 s. Playback is slowed to 0.5×. [Clip provenance](docs/videos/README.md).

## Method at a glance

- **Marker-first initialization:** reliable fixed markers seed metric camera poses before background features have sufficient triangulation parallax.
- **Metric re-anchoring:** new marker observations constrain scale along the connected visual trajectory, with geometric checks before committing updates.
- **Map recovery and merging:** visual and common-anchor evidence propose loop closures or map merges. A matching ID alone is not unconditional permission to merge.
- **Offline retrospective recovery:** verified image-to-map correspondences can recover earlier and short-gap poses. This is not interpolation or a guarantee of frame-zero recovery.
- **Separate marker roles:** workstation markers constrain the static map; calibrated wrist markers belong to moving fixtures.

<p align="center"><img src="docs/images/retrospective-recovery.png" width="760" alt="Forward processing and offline retrospective recovery"></p>

## What's included

| Path | Contents |
|---|---|
| `third_party/ORB_SLAM3/` | Modified C++ backend and native regressions |
| `aruco_track/` | Python observation, wrist geometry, replay and export |
| `process_monotag.py` | Hash-bound Ubuntu processing entry |
| `config/monotag_ubuntu_profile.json` | Validated feature profile |
| `scripts/` | Build, runtime setup, evaluation and diagnostics |
| `tools/` | Capture, calibration, marker generation and standalone utilities ([commands](tools/README.md)) |
| `tests/` | Release tests and historical experimental tests |
| `export_lerobot_dataset.py` | Optional validity-aware training export |

Recordings, binaries, model weights, vocabulary, device-specific camera calibration and generated caches are excluded. Diagnostic scripts do not define alternative production defaults.

## Installation

Ubuntu 24.04 is the validated reconstruction platform. Follow the [installation guide](docs/INSTALL.md) to provision OpenCV/Pangolin and vocabulary before building:

```bash
git clone https://github.com/jiejie567/MonoTag-SLAM.git
cd MonoTag-SLAM
make setup
make native DEPS_PREFIX=/path/to/dependencies
.venv/bin/python scripts/init_runtime.py \
  --deps-prefix /path/to/dependencies --vocabulary /path/to/ORBvoc.txt
```

Hand networks are optional and separately licensed. SLAM-only processing needs neither a GPU nor hand-model weights.

## Quick start

```bash
.venv/bin/python process_monotag.py input.mp4 \
  --calib camera.json --head-slam --slam-init auto --auto-marker-map \
  --static-marker-ids 20-49 --static-marker-size-mm 48 \
  --no-hand-joints --slam-replay --save-atlas runs/demo/atlas.osa \
  --output runs/demo/actions.jsonl

.venv/bin/python render_slam_replay.py runs/demo/actions.jsonl --hybrid
```

Use your actual marker IDs, measured black-square dimensions and calibrated intrinsics. These values are examples. Each run requires a fresh output location.

## Outputs and validity

Outputs distinguish localization, metric scale, map identity and revision. World-frame wrist output requires a supported camera pose, a metric frame and an accepted fixture observation. Unsupported intervals remain invalid; display interpolation is not a measurement.

Replay separates final offline reconstruction from the recorded mapping process. Hardware, example videos and compact result tables are in [MonoEgo](https://github.com/jiejie567/MonoEgo).

## Tests and reproducibility

```bash
make smoke
make native-smoke DEPS_PREFIX=/path/to/dependencies
```

[Reproducibility notes](REPRODUCIBILITY.md) identify the frozen runtime and release test boundary. Rebuilds do not imply bitwise-identical trajectories. Odin odometry is an external camera reference, not estimator input or independently certified ground truth. Static wrist scatter measures precision, not dynamic anatomical accuracy.

## Acknowledgements and license

MonoTag extends [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3), with OpenCV, Eigen, Pangolin and its included third-party libraries.

Code uses [GPL-3.0](LICENSE); third-party notices remain intact. Original project documentation is additionally licensed under [CC BY 4.0](docs/ASSET_LICENSE.md). The company logo is excluded; no trademark rights are granted. Learned models and datasets retain their own terms.
