# Release check — 2026-09-21

Checked in an isolated Ubuntu 24.04 source directory, without replacing the
production runtime.

| Check | Result |
|---|---|
| Existing Python release gate | 140 tests run, 2 conditionally skipped; no failures |
| New runtime-setup tests | 2 passed |
| Clean native build | Passed, including short-gap and prefix adapters |
| Native release suite | All 6 regression programs passed |
| Fresh runtime creation | Passed, using locally rebuilt binaries |
| Runtime hash and executable binding | Passed through production validator |
| Current source/private-host path scan | No known private-host identifiers found |
| Git history credential-pattern scan | No matches in the scanned patterns |
| Whitespace/diff validation | Passed |

The native suite covers marker graph optimization, map merge, coordinator,
Atlas scale, portable frontend and portable random behavior. The original
140-test gate still has two conditional skips; the count is not represented as
140 executed passes.

The build reused provisioned OpenCV 4.10 and Pangolin dependencies. It validates
the source build, not the dependency-install procedure on an empty OS image.
No long-sequence precision benchmark was rerun for this documentation and
packaging change. No algorithm thresholds, defaults or paper results changed.

Automated scans are limited checks, not a guarantee that every form of sensitive
information can be detected. Raw recordings, model weights, compiled artifacts,
private runtime JSON and the unblurred publicity review are not in this release.

Both GitHub repositories remain private at the author's request.
