#!/usr/bin/env python3
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from aruco_track.models import BandLayout, Calibration
from tools.make_band import band_layout


class LayoutTests(unittest.TestCase):
    def test_hands_have_disjoint_ids(self):
        left = band_layout("left", 0, 69, 55, 56)
        right = band_layout("right", 6, 69, 55, 56)
        self.assertFalse(set(left.markers) & set(right.markers))
        self.assertEqual(set(left.markers), set(range(6)))
        self.assertEqual(set(right.markers), set(range(6, 12)))

    def test_layout_json_round_trip(self):
        original = band_layout("left", 0, 69, 55, 56)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "layout.json"
            original.save(path, reconstructed=True)
            restored = BandLayout.load(path)
        self.assertEqual(restored.name, original.name)
        for marker_id in original.markers:
            np.testing.assert_allclose(restored.markers[marker_id], original.markers[marker_id])

    def test_layout_without_name_uses_filename(self):
        original = band_layout("left", 0, 69, 55, 56)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "strap_band_L.json"
            original.save(path)
            payload = json.loads(path.read_text())
            del payload["name"]
            path.write_text(json.dumps(payload))
            restored = BandLayout.load(path)
        self.assertEqual(restored.name, "strap_band_L")

    def test_calibration_round_trip(self):
        original = Calibration(np.eye(3), np.zeros(5), (1920, 1080))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "camera.json"
            original.save(path, rms_reprojection_error=0.5)
            restored = Calibration.load(path)
            payload = json.loads(path.read_text())
        np.testing.assert_array_equal(restored.camera_matrix, original.camera_matrix)
        self.assertEqual(restored.image_size, (1920, 1080))
        self.assertEqual(payload["rms_reprojection_error"], 0.5)

    def test_calibration_scales_for_resized_video(self):
        original = Calibration(
            np.array([[1000.0, 0.0, 960.0], [0.0, 900.0, 540.0], [0.0, 0.0, 1.0]]),
            np.arange(5, dtype=np.float64),
            (1920, 1080),
        )

        scaled = original.scaled_to((1280, 720))

        np.testing.assert_allclose(
            scaled.camera_matrix,
            np.array(
                [[666.6666667, 0.0, 640.0], [0.0, 600.0, 360.0], [0.0, 0.0, 1.0]]
            ),
        )
        np.testing.assert_array_equal(scaled.dist_coeffs, original.dist_coeffs)

    def test_calibration_rejects_aspect_ratio_change(self):
        calibration = Calibration(np.eye(3), np.zeros(5), (1920, 1080))

        with self.assertRaises(ValueError):
            calibration.scaled_to((1280, 800))


if __name__ == "__main__":
    unittest.main()
