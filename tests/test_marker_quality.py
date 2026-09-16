import unittest

import cv2
import numpy as np

from aruco_track.detector import ArucoDetector
from aruco_track.marker_quality import (
    MarkerBoundaryQuality,
    evaluate_marker_boundary,
    is_assist_only_marker,
    weak_corner_information_weights,
)


class MarkerBoundaryTests(unittest.TestCase):
    def setUp(self):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.image = np.full((160, 160), 255, np.uint8)
        self.image[20:140, 20:140] = cv2.aruco.generateImageMarker(dictionary, 20, 120)
        self.template = cv2.aruco.generateImageMarker(dictionary, 20, 120)
        self.corners = np.array(
            [[19.5, 19.5], [139.5, 19.5], [139.5, 139.5], [19.5, 139.5]], np.float32
        )

    def test_complete_single_marker_is_retained(self):
        self.assertTrue(evaluate_marker_boundary(self.image, self.corners, self.template).accepted)
        self.assertIn(20, ArucoDetector(validate_corners=True).detect(self.image))

    def test_intact_rejected_grid_can_vote_but_missing_border_cannot(self):
        weak = MarkerBoundaryQuality(
            False, (1.0,) * 4, (0.875,) * 4, 80.0,
            template_error_fraction=0.09, information_weight=0.0,
            reason="wrist_grid_mismatch", template_interior_error_fraction=0.01,
        )
        self.assertTrue(is_assist_only_marker(weak))
        self.assertFalse(is_assist_only_marker(MarkerBoundaryQuality(
            False, (1.0, 1.0, 0.4, 1.0), (1.0,) * 4, 80.0,
            template_error_fraction=0.09, information_weight=0.0,
            reason="wrist_boundary", template_interior_error_fraction=0.01,
        )))

    def test_rejected_marker_exposes_only_supported_weak_corners(self):
        quality = MarkerBoundaryQuality(
            False, (1.0, 1.0, 0.4, 1.0), (1.0, 1.0, 0.5, 1.0), 80.0,
            template_error_fraction=0.08, information_weight=0.0,
            reason="boundary", template_interior_error_fraction=0.01,
        )
        self.assertEqual(
            weak_corner_information_weights(quality),
            (0.05, 0.05, 0.0, 0.0),
        )

    def test_weak_corners_require_reliable_payload_interior(self):
        quality = MarkerBoundaryQuality(
            False, (1.0,) * 4, (1.0,) * 4, 80.0,
            template_error_fraction=0.08, information_weight=0.0,
            reason="grid_mismatch", template_interior_error_fraction=0.03,
        )
        self.assertEqual(weak_corner_information_weights(quality), (0.0,) * 4)

    def test_one_occluded_corner_is_not_a_strong_pose_observation(self):
        occluded = self.image.copy()
        occluded[118:149, 118:149] = 170
        quality = evaluate_marker_boundary(occluded, self.corners)
        self.assertFalse(quality.accepted)
        self.assertLess(quality.corner_support[2], 0.5)

    def test_displaced_corner_is_rejected_even_if_payload_id_is_known(self):
        displaced = self.corners.copy()
        displaced[2] -= 12.0
        self.assertFalse(evaluate_marker_boundary(self.image, displaced).accepted)

    def test_oblique_blurred_complete_marker_is_retained(self):
        destination = np.array([[25, 35], [150, 15], [120, 130], [40, 115]], np.float32)
        warped = cv2.warpPerspective(
            self.image, cv2.getPerspectiveTransform(self.corners, destination),
            (180, 160), borderValue=255,
        )
        warped = cv2.GaussianBlur(warped, (3, 3), 0.6)
        self.assertTrue(evaluate_marker_boundary(warped, destination).accepted)

    def test_missing_contrast_and_clipped_border_are_rejected(self):
        self.assertFalse(evaluate_marker_boundary(np.full_like(self.image, 100), self.corners).accepted)
        clipped = self.image[20:, 20:]
        self.assertFalse(evaluate_marker_boundary(clipped, self.corners - 20).accepted)

    def test_rejected_observation_is_not_kept_as_an_optical_flow_seed(self):
        class KnownIdDetector:
            def detectMarkers(inner_self, gray):
                return [self.corners.reshape(1, 4, 2)], np.array([[20]]), []

            def refineDetectedMarkers(inner_self, gray, board, corners, ids, rejected, *args):
                return corners, ids, rejected, None

        detector = ArucoDetector(
            board_markers=[{20: np.zeros((4, 3), np.float32)}], track_marker_gaps=2,
            validate_corners=True,
        )
        detector._detector = KnownIdDetector()
        self.assertIn(20, detector.detect(self.image))
        occluded = self.image.copy()
        occluded[118:149, 118:149] = 170
        self.assertNotIn(20, detector.detect(occluded))
        self.assertIn(20, detector.last_rejected_detections)
        self.assertNotIn(20, detector._tracks)

    def test_guard_does_not_filter_unconfigured_wrist_markers(self):
        detector = ArucoDetector(validate_corners=True, boundary_marker_ids={26})
        detector.detect(self.image)
        self.assertEqual(detector.last_boundary_quality, {})
        wrist_only = ArucoDetector(wrist_marker_ids={0})
        self.assertIn(20, wrist_only.detect(self.image))
        self.assertEqual(wrist_only.last_boundary_quality, {})

    def test_wrist_guard_rejects_decodable_occlusion_and_clears_flow_seed(self):
        detector = ArucoDetector(
            validate_corners=True, boundary_marker_ids={26}, wrist_marker_ids={20},
            board_markers=[{20: np.zeros((4, 3), np.float32)}], track_marker_gaps=2,
        )
        image = np.full((180, 180), 255, np.uint8)
        image[20:140, 20:140] = self.template
        self.assertIn(20, detector.detect(image))
        cv2.ellipse(image, (142, 140), (14, 12), 0, 0, 360, 170, -1)
        self.assertIn(20, ArucoDetector().detect(image))
        self.assertNotIn(20, detector.detect(image))
        self.assertTrue(detector.last_boundary_quality[20].reason.startswith('wrist_'))
        self.assertNotIn(20, detector._tracks)

    def test_wrist_profile_preserves_small_oblique_complete_marker(self):
        destination = np.array([[25, 45], [115, 25], [132, 54], [42, 74]], np.float32)
        image = cv2.warpPerspective(
            self.image, cv2.getPerspectiveTransform(self.corners, destination),
            (160, 100), borderValue=255,
        )
        image = cv2.GaussianBlur(image, (3, 3), 0.6)
        result = evaluate_marker_boundary(image, destination, self.template, wrist_marker=True)
        self.assertTrue(result.accepted)

    def test_small_oblique_wrist_with_displaced_corner_is_rejected(self):
        destination = np.array([[25, 45], [115, 25], [132, 54], [42, 74]], np.float32)
        image = cv2.warpPerspective(
            self.image, cv2.getPerspectiveTransform(self.corners, destination),
            (160, 100), borderValue=255,
        )
        image = cv2.GaussianBlur(image, (3, 3), 0.6)
        destination[2] -= [4, 4]
        result = evaluate_marker_boundary(image, destination, self.template, wrist_marker=True)
        self.assertFalse(result.accepted)

    def test_wrist_profile_does_not_accept_wrong_payload_or_blank_image(self):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        wrong = cv2.aruco.generateImageMarker(dictionary, 21, 120)
        self.assertFalse(evaluate_marker_boundary(self.image, self.corners, wrong,
                                                   wrist_marker=True).accepted)
        self.assertFalse(evaluate_marker_boundary(np.full_like(self.image, 100), self.corners,
                                                   self.template, wrist_marker=True).accepted)

    def test_grid_mismatch_is_rejected_even_with_complete_outer_border(self):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        wrong_template = cv2.aruco.generateImageMarker(dictionary, 21, 120)
        quality = evaluate_marker_boundary(self.image, self.corners, wrong_template)
        self.assertGreater(min(quality.corner_support), 0.9)
        self.assertGreater(quality.template_error_fraction, 0.04)
        self.assertFalse(quality.accepted)

    def test_small_cell_edge_difference_is_soft_not_bad_border(self):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        template = cv2.aruco.generateImageMarker(dictionary, 21, 120)
        image = self.image.copy()
        image[20:140, 20:140] = cv2.dilate(template, np.ones((6, 6), np.uint8))
        quality = evaluate_marker_boundary(image, self.corners, template)
        self.assertTrue(quality.accepted)
        self.assertEqual(quality.reason, "soft_grid")
        self.assertEqual(quality.information_weight, 0.25)
        self.assertGreater(quality.template_error_fraction, 0.04)
        self.assertEqual(quality.template_interior_error_fraction, 0.0)
        strict = evaluate_marker_boundary(image, self.corners, template, allow_soft_grid=False)
        self.assertFalse(strict.accepted)
        self.assertEqual(strict.reason, "grid_mismatch")

    def test_payload_interior_corruption_is_not_softened(self):
        # Keep the outer corners intact, but corrupt ~5% of the compared
        # pixels inside white cells, not along their edges.
        image = self.image.copy()
        core = (np.arange(120) % 20 >= 4) & (np.arange(120) % 20 < 16)
        ys, xs = np.where((self.template > 127) & core[:, None] & core[None, :])
        image[20 + ys[:600], 20 + xs[:600]] = 0
        quality = evaluate_marker_boundary(image, self.corners, self.template)
        self.assertFalse(quality.accepted)
        self.assertEqual(quality.information_weight, 0.0)
        self.assertGreater(quality.template_interior_error_fraction, 0.005)

    def test_reliable_corners_keep_full_weight(self):
        quality = evaluate_marker_boundary(self.image, self.corners, self.template)
        self.assertTrue(quality.accepted)
        self.assertEqual(quality.information_weight, 1.0)
        self.assertEqual(quality.reason, "ok")

    def test_decodable_marker_with_rounded_corner_occlusion_is_rejected(self):
        image = np.full((180, 180), 255, np.uint8)
        image[20:140, 20:140] = self.template
        cv2.ellipse(image, (142, 140), (14, 12), 0, 0, 360, 170, -1)
        raw = ArucoDetector(validate_corners=False).detect(image)
        self.assertIn(20, raw)
        error = np.linalg.norm(raw[20] - self.corners, axis=1).max()
        self.assertGreater(error, 3.0)
        checked = ArucoDetector(validate_corners=True)
        self.assertNotIn(20, checked.detect(image))
        self.assertGreater(checked.last_boundary_quality[20].template_error_fraction, 0.04)

    def test_calibrated_lens_distortion_is_not_treated_as_bad_geometry(self):
        camera = np.array([[220.0, 0.0, 150.0], [0.0, 220.0, 150.0], [0.0, 0.0, 1.0]])
        distortion = np.array([-0.35, 0.08, 0.0, 0.0, 0.0])
        undistorted_image = np.full((300, 300), 255, np.uint8)
        undistorted_image[60:180, 150:270] = self.template
        corners = self.corners + np.array([130.0, 40.0])
        rays = np.column_stack(((corners - [150, 150]) / 220.0, np.ones(4)))
        distorted_corners = cv2.projectPoints(
            rays, np.zeros(3), np.zeros(3), camera, distortion
        )[0].reshape(4, 2)
        y, x = np.indices((300, 300), dtype=np.float32)
        grid = np.stack((x, y), axis=-1).reshape(-1, 1, 2)
        undistorted_grid = cv2.undistortPoints(grid, camera, distortion, P=camera).reshape(300, 300, 2)
        distorted_image = cv2.remap(
            undistorted_image, undistorted_grid[:, :, 0], undistorted_grid[:, :, 1],
            cv2.INTER_LINEAR, borderValue=255,
        )

        class KnownIdDetector:
            def detectMarkers(inner_self, gray):
                return [distorted_corners.reshape(1, 4, 2)], np.array([[20]]), []

        detector = ArucoDetector(
            validate_corners=True, camera_matrix=camera, dist_coeffs=distortion
        )
        detector._detector = KnownIdDetector()
        self.assertIn(20, detector.detect(distorted_image))



if __name__ == "__main__":
    unittest.main()
