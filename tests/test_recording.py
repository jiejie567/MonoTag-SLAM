#!/usr/bin/env python3
import tempfile
import time
import unittest
import json
from pathlib import Path

import cv2
import numpy as np

from aruco_track.recording import LatestFrameCapture, RawVideoRecorder


class _FakeCapture:
    def __init__(self, frame_count: int):
        self.frame_count = frame_count
        self.index = 0
        self.released = False

    def read(self):
        if self.index >= self.frame_count:
            return False, None
        frame = np.full((8, 8, 3), self.index, dtype=np.uint8)
        self.index += 1
        time.sleep(0.001)
        return True, frame

    def release(self):
        self.released = True


class _CountingRecorder:
    def __init__(self):
        self.values = []

    def write(self, frame, timestamp_s):
        self.values.append(int(frame[0, 0, 0]))


class RawVideoRecorderTests(unittest.TestCase):
    def test_writes_readable_video(self):
        with tempfile.TemporaryDirectory() as temp:
            recorder = RawVideoRecorder(
                (64, 48),
                fps=10.0,
                output_dir=Path(temp),
                metadata={"camera_controls": {"mode": "locked"}},
            )
            path = recorder.start()
            recorder.write(np.zeros((48, 64, 3), dtype=np.uint8))
            recorder.write(np.full((48, 64, 3), 255, dtype=np.uint8))
            saved = recorder.stop()
            self.assertIsNotNone(saved)
            assert saved is not None
            self.assertEqual(saved.path, path)
            self.assertEqual(saved.frames, 2)
            self.assertEqual(saved.dropped_frames, 0)
            self.assertEqual(path.suffix, ".avi")
            self.assertTrue(saved.metadata_path.exists())
            metadata = json.loads(saved.metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["camera_controls"]["mode"], "locked")
            capture = cv2.VideoCapture(str(path))
            self.assertTrue(capture.isOpened())
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 2)
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 64)
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), 48)
            capture.release()

    def test_capture_records_every_frame_while_preview_skips_stale_frames(self):
        capture = _FakeCapture(20)
        recorder = _CountingRecorder()
        stream = LatestFrameCapture(capture, recorder)
        stream.start()
        sequence = -1
        preview_values = []
        while True:
            captured = stream.read_latest(sequence, timeout_s=0.1)
            if captured is None:
                break
            sequence, _, frame = captured
            preview_values.append(int(frame[0, 0, 0]))
            time.sleep(0.005)
        stream.stop()

        self.assertEqual(recorder.values, list(range(20)))
        self.assertLess(len(preview_values), 20)
        self.assertEqual(preview_values[-1], 19)
        self.assertTrue(capture.released)


if __name__ == "__main__":
    unittest.main()
