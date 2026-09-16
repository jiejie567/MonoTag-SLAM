import unittest
from unittest.mock import patch

import cv2
import numpy as np

from aruco_track.marker_cover_tracking import CoverTracker, validate_marker_patch


class MarkerCoverTrackingTests(unittest.TestCase):
    def setUp(self):
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.image = np.full((180, 180), 255, np.uint8)
        self.image[30:150, 30:150] = cv2.aruco.generateImageMarker(self.dictionary, 8, 120)
        self.quad = np.array([[29.5, 29.5], [149.5, 29.5],
                              [149.5, 149.5], [29.5, 149.5]], np.float32)
        self.tracker = CoverTracker(self.dictionary)

    def quality(self, image=None, quad=None, marker_id=8):
        return validate_marker_patch(self.image if image is None else image,
                                     self.quad if quad is None else quad,
                                     marker_id, self.dictionary)

    def test_fresh_only_detections_always_reset_age(self):
        for i in range(30):
            quads, info = self.tracker.update(self.image, {8: self.quad}, i / 30)
            self.assertIn(8, quads)
            self.assertEqual(info[8]["source"], "observed")
            self.assertEqual(info[8]["age_s"], 0)
        quads, info = self.tracker.update(self.image, {}, 1.0)
        self.assertIn(8, quads)
        self.assertEqual(info[8]["source"], "flow")
        self.assertAlmostEqual(info[8]["age_s"], 1 / 30)

    def test_stationary_real_marker_continues_after_six_frames(self):
        self.tracker.update(self.image, {8: self.quad}, 0.)
        for i in range(1, 24):
            quads, info = self.tracker.update(self.image, {}, i / 30)
            self.assertIn(8, quads, info)
            self.assertEqual(info[8]["source"], "flow")
            self.assertEqual(info[8]["reason"], "verified_flow")
            self.assertAlmostEqual(info[8]["age_s"], i / 30)

    def test_gap_uses_seconds_and_expiry_remains_visible(self):
        self.tracker.update(self.image, {8: self.quad}, 2.)
        for stamp in (2.81, 2.9, 3.):
            quads, info = self.tracker.update(self.image, {}, stamp)
            self.assertNotIn(8, quads)
            self.assertFalse(info[8]["valid"])
            self.assertEqual(info[8]["reason"], "expired")
            self.assertAlmostEqual(info[8]["age_s"], stamp - 2.)
        quads, info = self.tracker.update(self.image, {8: self.quad}, 3.1)
        self.assertIn(8, quads)
        self.assertEqual(info[8]["age_s"], 0.)

    def test_exact_gap_boundary_tolerates_timestamp_roundoff(self):
        self.tracker.update(self.image, {8: self.quad}, 32.)
        quads, info = self.tracker.update(self.image, {}, 32.8)
        self.assertIn(8, quads)
        self.assertEqual(info[8]["source"], "flow")

    def test_correct_and_moderately_blurred_patches_pass(self):
        self.assertTrue(self.quality()["valid"])
        blurred = cv2.GaussianBlur(self.image, (5, 5), 1.0)
        self.assertTrue(self.quality(blurred)["valid"])

    def test_perspective_and_blur_preserve_payload(self):
        dst = np.array([[30, 35], [145, 20], [130, 140], [45, 120]], np.float32)
        transform = cv2.getPerspectiveTransform(self.quad, dst)
        image = cv2.warpPerspective(self.image, transform, (180, 180), borderValue=255)
        image = cv2.GaussianBlur(image, (3, 3), .6)
        self.assertTrue(self.quality(image, dst)["valid"])

    def test_wrong_id_is_rejected(self):
        quality = self.quality(marker_id=9)
        self.assertFalse(quality["valid"])
        self.assertGreater(quality["bit_errors"], 1)

    def test_black_rectangle_with_white_bar_is_not_a_marker(self):
        image = np.zeros((100, 100), np.uint8)
        image[37:63, 37:77] = 255
        quad = np.array([[10, 10], [89, 10], [89, 89], [10, 89]], np.float32)
        for marker_id in range(12):
            quality = self.quality(image, quad, marker_id)
            self.assertFalse(quality["valid"], (marker_id, quality))

    def test_border_and_payload_are_checked_separately(self):
        image = self.image.copy()
        image[30:50, 30:150] = 255
        quality = self.quality(image)
        self.assertFalse(quality["valid"])
        self.assertEqual(quality["reason"], "border_mismatch")
        self.assertEqual(quality["bit_errors"], 0)

    def test_blank_and_low_contrast_are_rejected(self):
        for image in (np.full_like(self.image, 80), (self.image / 20 + 100).astype(np.uint8)):
            self.assertFalse(self.quality(image)["valid"])
            self.assertEqual(self.quality(image)["reason"], "low_contrast")

    def test_invalid_images_quads_and_ids_are_rejected(self):
        for image in (np.zeros((0, 0), np.uint8), np.zeros((180, 180, 3), np.uint8),
                      self.image.astype(float)):
            self.assertEqual(self.quality(image)["reason"], "invalid_image")
        for quad in (np.zeros((3, 2)), np.full((4, 2), np.nan),
                     self.quad[[0, 2, 1, 3]], self.quad + 180):
            self.assertFalse(self.quality(quad=quad)["valid"])
        self.assertEqual(self.quality(marker_id=100)["reason"], "invalid_marker_id")

    def test_invalid_observation_is_not_returned_as_current_polygon(self):
        self.tracker.update(self.image, {8: self.quad}, 0.)
        quads, info = self.tracker.update(self.image, {8: np.full((4, 2), np.nan)}, .1)
        self.assertNotIn(8, quads)
        self.assertEqual(info[8]["source"], "invalid")
        self.assertEqual(info[8]["reason"], "invalid_quad")

    def test_non_monotonic_time_and_shape_changes_reset_tracks(self):
        for new_image, timestamp, reason in ((self.image, 0., "non_monotonic_timestamp"),
                                             (self.image[:170], .1, "image_shape_changed")):
            tracker = CoverTracker(self.dictionary)
            tracker.update(self.image, {8: self.quad}, 0.)
            quads, info = tracker.update(new_image, {}, timestamp)
            self.assertNotIn(8, quads)
            self.assertEqual(info[8]["reason"], reason)
            self.assertEqual(info[8]["source"], "invalid")

    def test_reset_can_accept_a_new_real_observation(self):
        self.tracker.update(self.image, {8: self.quad}, 10.)
        quads, info = self.tracker.update(self.image, {8: self.quad}, 0.)
        self.assertIn(8, quads)
        self.assertEqual(info[8]["source"], "observed")
        self.assertEqual(info[8]["reset_reason"], "non_monotonic_timestamp")

    def test_invalid_image_and_timestamp_reset_without_stale_quads(self):
        for image, timestamp, reason in ((None, .1, "invalid_image"),
                                        (self.image, float("nan"), "invalid_timestamp")):
            tracker = CoverTracker(self.dictionary)
            tracker.update(self.image, {8: self.quad}, 0.)
            quads, info = tracker.update(image, {}, timestamp)
            self.assertEqual(quads, {})
            self.assertEqual(info[8]["reason"], reason)
            self.assertFalse(info[8]["valid"])

    def test_optical_flow_failure_is_reported_and_requires_redetection(self):
        self.tracker.update(self.image, {8: self.quad}, 0.)
        with patch("aruco_track.marker_cover_tracking.cv2.calcOpticalFlowPyrLK",
                   return_value=(None, None, None)):
            quads, info = self.tracker.update(self.image, {}, .1)
        self.assertNotIn(8, quads)
        self.assertEqual(info[8]["reason"], "flow_failed")
        quads, info = self.tracker.update(self.image, {}, .2)
        self.assertNotIn(8, quads)
        self.assertEqual(info[8]["reason"], "flow_failed")

    def test_successful_flow_cannot_carry_an_occluded_marker(self):
        self.tracker.update(self.image, {8: self.quad}, 0.)
        good = np.ones((4, 1), np.uint8)
        with patch("aruco_track.marker_cover_tracking.cv2.calcOpticalFlowPyrLK",
                   return_value=(self.quad[:, None], good, None)):
            quads, info = self.tracker.update(np.full_like(self.image, 100), {}, .1)
        self.assertNotIn(8, quads)
        self.assertEqual(info[8]["reason"], "flow_low_contrast")

    def test_inconsistent_forward_backward_flow_is_rejected(self):
        self.tracker.update(self.image, {8: self.quad}, 0.)
        good = np.ones((4, 1), np.uint8)
        with patch("aruco_track.marker_cover_tracking.cv2.calcOpticalFlowPyrLK", side_effect=[
            (self.quad[:, None], good, None), ((self.quad + 3)[:, None], good, None)
        ]):
            quads, info = self.tracker.update(self.image, {}, .1)
        self.assertNotIn(8, quads)
        self.assertEqual(info[8]["reason"], "flow_forward_backward")

    def test_flow_shape_change_is_rejected_even_with_perfect_fb(self):
        self.tracker.update(self.image, {8: self.quad}, 0.)
        good = np.ones((4, 1), np.uint8)
        distorted = (self.quad - 90) * np.array([.3, 1.]) + 90
        with patch("aruco_track.marker_cover_tracking.cv2.calcOpticalFlowPyrLK", side_effect=[
            (distorted.astype(np.float32)[:, None], good, None),
            (self.quad[:, None], good, None)
        ]):
            quads, info = self.tracker.update(self.image, {}, .1)
        self.assertNotIn(8, quads)
        self.assertEqual(info[8]["reason"], "flow_shape_change")

    def test_implausible_displacement_is_rejected(self):
        quad = self.quad / 6
        self.tracker.update(self.image, {8: quad}, 0.)
        good = np.ones((4, 1), np.uint8)
        with patch("aruco_track.marker_cover_tracking.cv2.calcOpticalFlowPyrLK", side_effect=[
            ((quad + 80)[:, None], good, None), (quad[:, None], good, None)
        ]):
            quads, info = self.tracker.update(self.image, {}, .1)
        self.assertNotIn(8, quads)
        self.assertEqual(info[8]["reason"], "flow_displacement")

    def test_configuration_rejects_invalid_gap(self):
        for gap in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                CoverTracker(self.dictionary, gap)


if __name__ == "__main__":
    unittest.main()
