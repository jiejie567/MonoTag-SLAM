# MonoTag SLAM

Anonymous review code for **MonoTag SLAM**, the marker-aided monocular backend used by the MonoEgo capture system.

MonoTag combines ORB-SLAM3 geometry with known-size square-marker observations. It supports marker-first metric initialization, monocular-map metricization, marker-assisted loop and map-merge proposals, interval metric re-anchoring, multimap Atlas persistence, and offline recovery of earlier or short-gap frames when the final map provides independent image-to-map support. Dynamic wrist markers are processed separately and never become static camera anchors.

## Repository layout

- `third_party/ORB_SLAM3/`: modified native ORB-SLAM3 source.
- `aruco_track/`: marker admission, wrist constellations, replay, and export modules.
- `process_monotag.py`: validated offline entry point.
- `config/monotag_ubuntu_profile.json`: paper configuration and feature switches.
- `scripts/`: build, validation, replay, and packaging utilities.
- `tests/`: Python and native regression tests.
- `export_lerobot_dataset.py`: optional validity-aware training export.

Raw recordings, learned hand-model weights, vocabularies, calibration files tied to a particular device, and generated outputs are intentionally excluded.

## Platform

The paper experiments use Ubuntu 24.04. The Python environment can be created with:

```bash
bash setup.sh
```

The native backend additionally requires the normal ORB-SLAM3 dependencies and a vocabulary file. Build the modified native source before running the complete pipeline. The portable source snapshot is included here; machine-specific binaries and absolute runtime paths are not.

## Offline processing

```bash
./.venv/bin/python process_monotag.py VIDEO \
  --calib CAMERA.json \
  --head-slam --slam-init auto \
  --auto-marker-map \
  --static-marker-ids 20-49 \
  --static-marker-size-mm 48 \
  --no-hand-joints \
  --save-atlas RUN/atlas.osa \
  --output RUN/actions.jsonl
```

Marker size, ID ranges, and intrinsics must match the recording. The 48 mm setting is an example from the camera-reference experiment, not a universal default.

To render the synchronized video and interactive Atlas after processing:

```bash
./.venv/bin/python render_slam_replay.py \
  --actions RUN/actions.jsonl \
  --output RUN/actions_replay
```

Use each command's `--help` for the complete interface.

## Output contract

Every exported frame distinguishes tracking validity from metric scale. World-frame wrist output requires a valid camera pose, a metric map, and an accepted wrist-constellation observation. Unsupported motion remains missing; the pipeline does not freeze a previous pose or convert interpolation into a measurement. Outputs include map identity, map revision, localization source, metric state, and quality fields.

## Tests

```bash
./.venv/bin/python scripts/run_release_smoke.py
```

This is the portable release gate. Native regression binaries require a completed ORB-SLAM3 build. The broader `tests/` tree intentionally retains historical and experimental source-contract tests; it is not the release gate. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the exact review snapshot and test-suite boundary.

After `scripts/build_linux.sh`, run the six native release regressions with:

```bash
bash scripts/run_native_smoke.sh DEPENDENCY_INSTALL_PREFIX
```

## License

This repository contains a modified ORB-SLAM3 codebase and is released under the GNU General Public License v3.0. Third-party components retain their original notices.
