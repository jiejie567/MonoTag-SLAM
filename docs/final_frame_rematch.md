# Final-frame background re-matching

The ordinary exporter enables this pass after final-Atlas frame-pose refinement.
It repairs measured, metric camera poses whose final fixed-marker projection
residual exceeds 8 px, or which rely on a single strong fixed marker without
background tracking, plus a 0.5-second neighborhood in the same map/revision.
Low single-marker reprojection error alone cannot rule out planar ambiguity.
It does not modify native tracking history, the Atlas, marker geometry, or wrist
calibration. Missing camera/wrist observations remain missing.

The read-only native adapter extracts fresh masked ORB features and matches
them to final map-point descriptors. Its `final-frame-pose-evidence` mode emits
support-only evidence, **not accepted replacement poses**. Python verifies map
identity, actual point IDs, unique pixels, image coverage and reprojection.
At least 40 background matches are required. A pose-only refinement then uses
these pixels and admitted marker corners with the existing 40 mm / 3-degree
envelope; corrections over 20 mm still require disjoint background cross-fits.
Re-matched background RMS and marker-corner residuals must remain within 3 px.

No wrist position, stationary reference, zero-velocity constraint, or gap
interpolation is used to correct the camera. Failed candidates leave the prior
pose and its quality warning intact. Refined final labels are marked with
`camera_localization_recovery.method = final-map-rematched-pose-refinement`
and `original_tracking_valid = true`; this is distinct from prefix/gap recovery.

Build the updated read-only adapter with:

```sh
python scripts/build_prefix_localizer.py
```

For an explicit baseline ablation, set `ORB_SLAM3_FINAL_FRAME_REMATCH=0`.
The normal default is enabled. Per-window diagnostics are saved under the
replay directory's `final_frame_rematch/`. Unsupported/older adapters fail
closed with a diagnostic report; they never silently replace poses.

Validation: the September 14 REV5 stationary-wrist batch is retained under
`output/samsung_static4_rev5_20260914/`. The A/B experiment holds native SLAM,
the Atlas, camera intrinsics and wrist observations fixed, isolating final-pose
reconstruction from native SLAM randomness. Static wrist scatter measures
precision in these clips, not externally referenced absolute accuracy.
