import unittest
from unittest.mock import patch

import cv2
import numpy as np

from aruco_track.camera_state import prepare_slam_frame


def legacy_frame(frame, allowed_mask):
    softened = cv2.GaussianBlur(frame, (31, 31), 0)
    return np.where(allowed_mask[:, :, None] > 0, frame, softened)


class SlamPreprocessingTests(unittest.TestCase):
    def test_identical_pixels_and_jpeg_without_modifying_inputs(self):
        rng = np.random.default_rng(42)
        frame = rng.integers(0, 256, (97, 131, 3), dtype=np.uint8)
        partial = np.full(frame.shape[:2], 255, np.uint8)
        partial[0:40, 0:37] = 0
        partial[60:, 90:] = 0
        masks = [partial, np.zeros(frame.shape[:2], np.uint8),
                 np.full(frame.shape[:2], 255, np.uint8),
                 rng.choice(np.array([0, 1, 128, 255], np.uint8), frame.shape[:2])]
        for mask in masks:
            with self.subTest(excluded=int(np.count_nonzero(mask == 0))):
                frame_before, mask_before = frame.copy(), mask.copy()
                expected, actual = legacy_frame(frame, mask), prepare_slam_frame(frame, mask)
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(frame, frame_before)
                np.testing.assert_array_equal(mask, mask_before)
                self.assertFalse(np.shares_memory(frame, actual))
                options = [cv2.IMWRITE_JPEG_QUALITY, 92]
                np.testing.assert_array_equal(cv2.imencode('.jpg', actual, options)[1],
                                              cv2.imencode('.jpg', expected, options)[1])

    def test_unmasked_frame_does_not_need_blur(self):
        frame = np.full((40, 60, 3), 23, np.uint8)
        mask = np.full(frame.shape[:2], 255, np.uint8)
        with patch('aruco_track.camera_state.cv2.GaussianBlur') as blur:
            actual = prepare_slam_frame(frame, mask)
        blur.assert_not_called()
        np.testing.assert_array_equal(actual, frame)
        self.assertFalse(np.shares_memory(frame, actual))

    def test_noncontiguous_inputs_keep_the_same_border_pixels(self):
        rng = np.random.default_rng(7)
        frame = rng.integers(0, 256, (80, 120, 3), dtype=np.uint8)[::2, ::2]
        mask = rng.choice(np.array([0, 255], np.uint8), (80, 120))[::2, ::2]
        self.assertFalse(frame.flags.c_contiguous)
        np.testing.assert_array_equal(prepare_slam_frame(frame, mask), legacy_frame(frame, mask))


if __name__ == '__main__':
    unittest.main()
