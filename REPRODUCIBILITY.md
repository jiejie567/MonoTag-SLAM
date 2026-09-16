# Reproducibility notes

## Review snapshot

This repository was packaged from the Ubuntu 24.04 processing tree used for the final submission audit on 2026-09-16. The accepted paper profile is stored in `config/monotag_ubuntu_profile.json`.

The frozen native library used for the reported reference run had SHA-256:

```text
a2309ac4767729da32ceb0fcd90a9af374668adda054bce6c8be7943a2785ab5
```

Machine-specific binaries are not committed. The hash is included to identify the exact experimental runtime and to prevent an unverified rebuild from being presented as the paper binary.

## Validation status

The portable release gate is:

```bash
./.venv/bin/python scripts/run_release_smoke.py
```

It currently passes 140 tests on a clean Ubuntu 24.04 source checkout. It
covers marker admission and temporal replay, wrist precision/rejection,
geometry quality, metric re-anchoring, native-source recovery and loop queues,
final-map and hybrid replay, the local range-capable replay server, and build
and processing-host contracts. All six native release regressions built and
passed on Ubuntu 24.04; these include 14 coordinator cases and 10 map-merge
cases. The broader historical Python suite retains stale fallback and proposal
source-contract tests rather than hiding them; it is not the release gate and
does not authorize silently enabling rejected experiments.

A clean-source Ubuntu smoke rebuild should be interpreted separately from the
frozen paper binary: rebuilding validates portability, whereas the SHA-256
above identifies the exact native library used for the reported measurements.

The source release excludes raw or identifiable recordings, generated maps and caches, device-specific calibration, learned model weights, private host details, native binaries, and vocabulary data.

## Evaluation scope

The camera-reference experiments use odometry from a lidar--visual system as a reference trajectory, not as an input to MonoTag and not as independently verified absolute ground truth. SE(3) and Sim(3) results must be reported separately. Static wrist-constellation sequences evaluate precision, not absolute anatomical wrist accuracy.

Small source tables and representative videos are provided in the companion anonymous project repository. Raw recordings are withheld during review because they may contain identifiable indoor imagery.
