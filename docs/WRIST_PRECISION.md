# Wrist precision screening

The Ubuntu exporter now adds `hands.<band>.wrist_precision_qualified` by default
to graph-based wrist labels. This is **estimated local precision**, not a
guarantee about absolute world-pose error or hand-joint accuracy.

- `true`: local maximum-direction translation standard deviation is within
  the configured budget.
- `false`: the estimate exceeds that budget. The original measurement remains
  valid if it was valid before; nothing is removed or interpolated.
- `null`: no valid metric world wrist or no usable geometry estimate.

`wrist_precision` supplies the estimate in mm, reason, budget, assumed pixel
noise, and model limitations. `actions.meta.json.wrist_precision_policy` records
the policy. Historical outputs without the field mean **unknown**, not `true`.
Non-graph export leaves the flag unknown and marks the policy disabled.

Defaults: `--wrist-precision-budget-mm 10 --wrist-corner-sigma-px 0.5`.
The criterion is maximum-direction **1-sigma**, not a 95% interval or a bound
on 3D Euclidean error. Both defaults are provisional application/model choices,
not parameters fitted to the 39 mm example. Independent calibration and held-out
testing are required before using this flag as a hard training-quality filter.

Only accepted wrist marker corners participate. The diagnostic uses a local
unregularized PnP fit; it does not change the pose, accepted IDs, confidence,
existing validity mask, trajectory, or SLAM optimizer. It does not assume the
wrist is stationary. Camera pose, intrinsics, layout, branch ambiguity and
systematic corner errors are excluded, so a `true` flag does not certify that
the world error is below the budget. Do not use it to certify finger joints.

No automatic changes to replay rendering or existing dataset files are made.
New exports include the flag; prior files must be recomputed/annotated explicitly.
