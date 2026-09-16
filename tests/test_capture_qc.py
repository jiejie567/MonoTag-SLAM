#!/usr/bin/env python3
import unittest

import cv2
import numpy as np

from aruco_track.capture_qc import (
    CaptureQualitySummary,
    evaluate_capture_quality,
)


class CaptureQualityTests(unittest.TestCase):
    def test_sharp_well_exposed_large_marker_is_ready(self):
        grid = (np.indices((480, 640)).sum(axis=0) % 2 * 150 + 50).astype(np.uint8)
        corners = np.array([[100, 100], [180, 100], [180, 180], [100, 180]])

        quality = evaluate_capture_quality(grid, {20: corners})

        self.assertTrue(quality.ready)
        self.assertEqual(quality.marker_count, 1)
        self.assertAlmostEqual(quality.median_marker_side_px, 80.0)

    def test_blur_exposure_and_small_marker_are_reported(self):
        dark = np.zeros((480, 640), dtype=np.uint8)
        corners = np.array([[10, 10], [30, 10], [30, 30], [10, 30]])

        quality = evaluate_capture_quality(dark, {20: corners})

        self.assertIn("BLUR", quality.warnings)
        self.assertIn("DARK", quality.warnings)
        self.assertIn("TAG_SMALL", quality.warnings)
        self.assertFalse(quality.ready)

    def test_missing_marker_and_summary_are_explicit(self):
        image = np.full((480, 640), 120, dtype=np.uint8)
        quality = evaluate_capture_quality(image, {})
        summary = CaptureQualitySummary()
        summary.add(quality)
        payload = summary.to_dict()

        self.assertIn("NO_TAG", quality.warnings)
        self.assertEqual(payload["samples"], 1)
        self.assertEqual(payload["ready_fraction"], 0.0)
        self.assertEqual(payload["warning_counts"]["NO_TAG"], 1)


if __name__ == "__main__":
    unittest.main()
