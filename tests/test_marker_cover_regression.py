"""Marker-cover safety regressions; no camera, SLAM, or learned model needed.

The optional image checks use local, unchanged input frames. Their small manual
regions are pixel-level examples, not a claim of complete foreground ground truth.
"""
import json
from pathlib import Path
import unittest

import cv2
import numpy as np

from aruco_track.marker_cover import build_mask, foreground_protection, project_paper_quad


ROOT = Path(__file__).resolve().parents[1]
PREPARED = ROOT / 'output/inpainting_preview_Ao5kwb/prepared_small'
RECORDS = ROOT / 'recordings/raw_20260831_111003_047657_actions.jsonl'
BOARD = ROOT / 'output/pdf/world_reference_board_A4_aruco_20_27.json'

# Coordinates are in the original preview's 960 x 540 image, not millimetres.
CABLE_CENTERLINE_FRAME_0 = np.array(
    [[640, 143], [680, 130], [720, 117], [760, 101]], np.int32)
PAPER_FRAME_89 = np.array(
    [[468, 157], [610, 122], [629, 209], [485, 247]], np.float32)
WHITE_STRIPS_FRAME_89 = (
    np.array([[612, 167], [617, 166], [625, 204], [619, 206]], np.float32),
    np.array([[475, 175], [480, 174], [490, 234], [484, 236]], np.float32),
)


def polygon(shape, corners):
    mask = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(mask, np.rint(corners).astype(np.int32), 255)
    return mask


def synthetic_scene(marker_colour=(0, 0, 0)):
    """A paper sticker with a real ArUco bitmap isolated from its outer border."""
    frame = np.full((220, 320, 3), 185, np.uint8)
    paper = polygon(frame.shape[:2], [[45, 35], [275, 35], [275, 185], [45, 185]])
    frame[paper > 0] = 240
    quad = np.array([[110, 70], [201, 70], [201, 161], [110, 161]], np.float32)
    payload = polygon(frame.shape[:2], quad)
    bitmap = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 11, 92)
    patch = frame[70:162, 110:202]
    patch[bitmap == 0] = marker_colour
    return frame, paper, payload


class MarkerCoverGeometryTests(unittest.TestCase):
    def test_paper_uv_supports_unequal_horizontal_and_vertical_margins(self):
        quad = np.array([[100, 100], [200, 100], [200, 180], [100, 180]], np.float32)
        paper_uv = np.array([[-.25, -.16], [1.6, -.16],
                             [1.6, 1.25], [-.25, 1.25]], np.float32)
        expected = np.array([[75, 87.2], [260, 87.2],
                             [260, 200], [75, 200]], np.float32)
        np.testing.assert_allclose(project_paper_quad(quad, paper_uv), expected, atol=1e-4)

    def test_nonrectangular_paper_outline_follows_marker_perspective(self):
        quad = np.array([[130, 90], [220, 105], [190, 187], [95, 160]], np.float32)
        paper_uv = np.array([[-.2, -.1], [1.4, -.2],
                             [1.25, 1.3], [-.3, 1.15]], np.float32)
        canonical = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
        projected = project_paper_quad(quad, paper_uv)
        back = cv2.perspectiveTransform(
            np.asarray(projected, np.float32)[None],
            cv2.getPerspectiveTransform(quad, canonical))[0]
        np.testing.assert_allclose(back, paper_uv, atol=1e-4)

    def test_unsafe_paper_projection_does_not_abort_or_fill_unknown_area(self):
        # The detected tag is convex and in frame, but extrapolating its paper
        # perimeter crosses, or approaches, the projective horizon. The latter
        # is finite yet expands to coordinates around +/-13,000 in a 220px frame.
        frame = np.full((220, 220, 3), 240, np.uint8)
        quad = np.array([[70, 60], [110, 60], [170, 160], [10, 160]], np.float32)
        for pad in (.375, .33):
            with self.subTest(padding=pad):
                paper_uv = np.array([[-pad, -pad], [1+pad, -pad],
                                     [1+pad, 1+pad], [-pad, 1+pad]], np.float32)
                result = build_mask(frame, {11: quad}, {}, {11: paper_uv})
                region = result.diagnostics['regions']['11']
                self.assertIn(region['status'], ('partial', 'invalid'))
                self.assertTrue(region['reason'])
                self.assertFalse(np.any(result.mask[polygon(frame.shape[:2], quad) == 0]))


class MarkerCoverForegroundTests(unittest.TestCase):
    def test_isolated_reddish_marker_ink_is_not_treated_as_a_hand(self):
        # This colour passes the old global HSV "coloured foreground" gate.
        frame, paper, payload = synthetic_scene((35, 45, 100))
        frame_before, paper_before, payload_before = frame.copy(), paper.copy(), payload.copy()
        protected, diagnostics = foreground_protection(frame, paper, payload)
        self.assertEqual(np.count_nonzero(protected[payload > 0]), 0)
        self.assertEqual(diagnostics['protected_payload_pixels'], 0)
        self.assertFalse(diagnostics['partial'])
        np.testing.assert_array_equal(frame, frame_before)
        np.testing.assert_array_equal(paper, paper_before)
        np.testing.assert_array_equal(payload, payload_before)

    def test_curved_cable_crossing_paper_preserves_its_inside_pixels(self):
        frame, paper, payload = synthetic_scene()
        points = np.array([[12, 130], [46, 151], [82, 173], [117, 178],
                           [147, 172], [176, 178], [213, 175], [252, 151],
                           [300, 138]], np.int32)
        cv2.polylines(frame, [points], False, (20, 20, 20), 5)
        center = np.zeros(paper.shape, np.uint8)
        cv2.polylines(center, [points], False, 255, 1)
        protected, diagnostics = foreground_protection(frame, paper, payload)
        targets = (center > 0) & (paper > 0)
        self.assertGreater(np.count_nonzero(targets), 150)
        self.assertTrue(np.all(protected[targets] > 0))
        self.assertGreaterEqual(diagnostics['crossing_components'], 1)
        # The curve does not touch the tag: do not preserve all black ink.
        self.assertEqual(np.count_nonzero(protected[payload > 0]), 0)

    def test_cable_connected_to_tag_ink_is_explicitly_partial(self):
        frame, paper, payload = synthetic_scene()
        points = np.array([[18, 85], [65, 85], [105, 99], [137, 119],
                           [179, 125], [233, 139], [301, 139]], np.int32)
        cv2.polylines(frame, [points], False, (0, 0, 0), 7)
        center = np.zeros(paper.shape, np.uint8)
        cv2.polylines(center, [points], False, 255, 1)
        protected, diagnostics = foreground_protection(frame, paper, payload)
        self.assertTrue(np.all(protected[(center > 0) & (paper > 0)] > 0))
        self.assertTrue(diagnostics['partial'])
        self.assertGreater(diagnostics['protected_payload_pixels'], 0)
        # Ambiguity must preserve pixels, not silently erase the occluder.
        self.assertEqual(protected[99, 105], 255)

    def test_explicit_hand_mask_wins_even_over_marker_payload(self):
        frame, paper, payload = synthetic_scene((35, 45, 100))
        hand = np.zeros(paper.shape, np.uint8)
        cv2.rectangle(hand, (143, 55), (165, 131), 255, -1)
        frame[hand > 0] = (100, 145, 190)
        hand_before = hand.copy()
        protected, diagnostics = foreground_protection(frame, paper, payload, hand_mask=hand)
        self.assertTrue(np.all(protected[(hand > 0) & (paper > 0)] > 0))
        self.assertTrue(diagnostics['partial'])
        self.assertGreater(diagnostics['protected_payload_pixels'], 0)
        np.testing.assert_array_equal(hand, hand_before)

    def test_nonpaper_object_outside_payload_is_not_painted_over(self):
        frame, paper, payload = synthetic_scene()
        object_mask = polygon(paper.shape, [[60, 55], [91, 55], [91, 115], [60, 115]])
        frame[object_mask > 0] = (65, 110, 165)
        protected, _ = foreground_protection(frame, paper, payload)
        self.assertTrue(np.all(protected[object_mask > 0] > 0))


@unittest.skipUnless(PREPARED.is_dir() and RECORDS.is_file() and BOARD.is_file(),
                     'optional local source-video regression frames are unavailable')
class MarkerCoverLocalVideoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = {}
        with RECORDS.open() as stream:
            for index, line in enumerate(stream):
                if index in (360, 480, 538):
                    cls.records[index] = json.loads(line)
                if index >= 538:
                    break

    def test_known_bent_cable_pixels_in_source_frame_360_are_preserved(self):
        frame = cv2.imread(str(PREPARED / 'input_frames/00000.png'))
        self.assertIsNotNone(frame)
        layout = {str(item['id']): np.asarray(item['object_points_m'], np.float32)[:, :2]
                  for item in json.loads(BOARD.read_text())['markers']}
        observed = self.records[360]['detected_marker_corners']
        ids = [mid for mid in observed if mid in layout]
        homography, _ = cv2.findHomography(
            np.concatenate([layout[mid] for mid in ids]),
            np.concatenate([np.asarray(observed[mid], np.float32) * .5 for mid in ids]),
            cv2.RANSAC, 2.5)
        page_xy = np.array([[[-.105, .1485], [.105, .1485],
                             [.105, -.1485], [-.105, -.1485]]], np.float32)
        paper = polygon(frame.shape[:2], cv2.perspectiveTransform(page_xy, homography)[0])
        payload = np.zeros(paper.shape, np.uint8)
        for points in layout.values():
            projected = cv2.perspectiveTransform(points[None], homography)[0]
            uv = np.array([[-.05, -.05], [1.05, -.05],
                           [1.05, 1.05], [-.05, 1.05]], np.float32)
            payload |= polygon(paper.shape, project_paper_quad(projected, uv))
        center = np.zeros(paper.shape, np.uint8)
        cv2.polylines(center, [CABLE_CENTERLINE_FRAME_0], False, 255, 3)
        targets = (center > 0) & (paper > 0)
        self.assertGreater(np.count_nonzero(targets), 500)
        self.assertTrue(np.all(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[targets] < 65))
        protected, diagnostics = foreground_protection(frame, paper, payload)
        self.assertTrue(np.all(protected[targets] > 0))
        self.assertGreaterEqual(diagnostics['crossing_components'], 1)

    def test_annotated_paper_outline_includes_both_missed_white_strips(self):
        frame = cv2.imread(str(PREPARED / 'input_frames/00089.png'))
        self.assertIsNotNone(frame)
        marker = np.asarray(self.records[538]['detected_marker_corners']['11'], np.float32) * .5
        canonical = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
        paper_uv = cv2.perspectiveTransform(
            PAPER_FRAME_89[None], cv2.getPerspectiveTransform(marker, canonical))[0]
        projected = project_paper_quad(marker, paper_uv)
        paper = polygon(frame.shape[:2], projected)
        payload = polygon(frame.shape[:2], marker)
        protected, _ = foreground_protection(frame, paper, payload)
        for strip in WHITE_STRIPS_FRAME_89:
            with self.subTest(strip=strip.tolist()):
                strip_mask = polygon(paper.shape, strip)
                targets = strip_mask > 0
                self.assertGreater(np.count_nonzero(targets), 200)
                self.assertTrue(np.all(paper[targets] > 0))
                # Permit a one-pixel conservative safety margin at the band edge,
                # but not the previous complete loss of these white strips.
                interior = cv2.erode(strip_mask, np.ones((3, 3), np.uint8)) > 0
                self.assertTrue(np.all(protected[interior] == 0))
                self.assertGreater(np.mean(protected[targets] == 0), .97)


if __name__ == '__main__':
    unittest.main()
