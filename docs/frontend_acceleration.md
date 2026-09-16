# Offline frontend acceleration experiments (2026-09-08)

These are incremental adaptations, **not a FastORB-SLAM or OV²SLAM port**.
The established marker/ORB initialization, keyframe policy, metric graph, BA,
loop admission and finalization remain in place. The offline headless entry
defaults to descriptor parallelism and image prefetch, as requested after testing.
LK remains opt-in. Other library users retain their previous defaults.

## Switches

| Environment variable | `1` enables | Default |
|---|---|---|
| `ORB_SLAM3_TEMPORAL_FLOW` | Prediction-guided half-resolution forward/backward LK matching proposal; current ORB descriptors and pose optimization still validate matches | off |
| `ORB_SLAM3_PARALLEL_DESCRIPTORS` | One additional worker computes level-zero ORB descriptors while the extraction thread computes other pyramid levels | on (offline) |
| `ORB_SLAM3_PREFETCH_IMAGES` | Decode one future image/mask pair concurrently with current-frame SLAM | on (offline) |

The default offline combination is `PARALLEL_DESCRIPTORS=1`,
`PREFETCH_IMAGES=1`, `TEMPORAL_FLOW=0`, prefixed with `ORB_SLAM3_` as above.
Set all three to `0` to use the baseline implementation.
Defaults apply only when the environment variable is absent; an explicit `0`
is never overwritten. The benchmark's `--modes default off` exercises this contract.

No source FPS, pixels, timestamps, feature budget, pyramid levels, BA iterations
or acceptance thresholds are reduced by the conservative combination. The
prefetch worker never accesses the Atlas, drops frames or decodes past the
selected input range. It holds at most one future decoded image/mask pair.
Native timing output records selected switches and input wait time.

## Why the LK proposal is not enabled by default

The current ORB implementation requires fresh descriptors for local-map
matching, relocalization and keyframe insertion. This first-stage LK experiment
retains full extraction and only accelerates motion-model matching. It does
**not** implement FastORB's ordinary-frame descriptor-free path.

Only healthy, consecutive monocular frames with sufficient observed map points
are eligible. Forward/backward, photometric, foreground-mask, descriptor,
spatial-coverage and robust-pose checks gate proposals. Rejected proposals return
to the existing projection matcher. Local-map verification still runs.

In the initial 2700-frame A/B, LK added 9.26 s of proposal work. Tracking mean
increased from 26.15 to 26.94 ms even though total time changed from 118.22 to
114.17 s. That is not evidence of an effective frontend speedup; map and machine
timing also varied. A descriptor-free frontend needs a separate design for
keyframe promotion, feature-index consistency and loss recovery, not stale
descriptors copied into new frames.

## Verification and reproducibility

- `temporal_flow_regression`: known image translation, excluded mask,
  textureless images and empty input.
- `descriptor_parallel_regression IMAGE ...`: alternates serial/parallel order,
  checks keypoint fields, output ordering and descriptor bytes exactly; tests
  2000/10000-feature extraction with and without an exclusion mask.
- `scripts/benchmark_temporal_flow.py`: sequential native A/B on the same cache,
  calibration and tag observations. `--accelerator flow|descriptors|prefetch|safe`
  selects a candidate (`safe` means the conservative combination, not proven
  trajectory accuracy). `--modes on off` reverses run order.
- The benchmark defaults to **2700 frames**, not the full sequence. Pass
  `--frames 13252` for this full UVC90 recording.

Benchmarks exclude RGB/marker preprocessing and final MP4/HTML generation;
their speedup must not be presented as full dataset-export speedup. Compact
native history is identical between paired runs. Camera trajectory differences
are repeatability measures, **not ground-truth errors**. Even byte-identical
feature extraction can change native worker scheduling and subsequent map
optimization slightly. The 13252-frame full-sequence test retained three
accepted loops; external trajectory accuracy equivalence is still not proven.

## Primary references

- [OV²SLAM official implementation](https://github.com/ov2slam/ov2slam):
  prediction-guided LK tracking and separation of tracking, mapping and optimization.
- [FastORB-SLAM paper](https://arxiv.org/abs/2008.09870):
  descriptor-independent adjacent-frame matching using sparse optical flow.

The descriptor-level parallelism and bounded image prefetch above are local
engineering changes; they are not claimed as algorithms reproduced from either paper.
