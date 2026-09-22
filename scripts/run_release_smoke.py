#!/usr/bin/env python3
"""Run the portable release-contract smoke suite.

This deliberately names the submission-facing contracts instead of treating
the larger historical test archive as a release gate.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))


MODULES = (
    "test_auto_process",
    "test_marker_quality",
    "test_marker_temporal_admission",
    "test_static_marker_admission_integration",
    "test_marker_temporal_replay",
    "test_wrist_precision",
    "test_wrist_rejection_labels",
    "test_wrist_geometry_quality",
    "test_reanchor_metric_audit",
    "test_reanchor_observation_contract",
    "test_native_reliable_recovery",
    "test_reliable_recovery_contract",
    "test_native_loop_queue",
    "test_offline_loop_budget_contract",
    "test_final_map_replay",
    "test_hybrid_replay_frontend",
    "test_replay_server",
    "test_slam_build_contract",
    "test_processing_host",
)


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromNames(MODULES)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
