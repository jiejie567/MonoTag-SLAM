"""Backend protocol checks that do not require Torch or private model assets."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_hawor_hands.py"
_SPEC = importlib.util.spec_from_file_location("hawor_standalone_runner", _SCRIPT)
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)


def box(side, score, x=0):
    return {"handedness_class_id": side, "detection_score": score,
            "bbox_xyxy": [x, 1, x+10, 11]}


class HaWoRRunnerTests(unittest.TestCase):
    def test_highest_candidate_per_side_without_input_mutation(self):
        rows = [{"frame": 7, "boxes": [box(0, .6), box(1, .8), box(0, .9, 20)]},
                {"frame": 8, "boxes": []}]
        before = copy.deepcopy(rows)
        selected = runner.select_detections(rows, 7, 9)
        self.assertEqual(selected["Left"][7]["candidate_index"], 2)
        self.assertEqual(selected["Left"][7]["same_side_candidate_count"], 2)
        self.assertEqual(selected["Right"][7]["detection_score"], .8)
        self.assertNotIn(8, selected["Left"])
        self.assertEqual(rows, before)

    def test_missing_detection_frame_is_not_silently_skipped(self):
        with self.assertRaisesRegex(ValueError, "Missing detection frame 8"):
            runner.select_detections([{"frame": 7, "boxes": []}], 7, 9)

    def test_duplicate_detection_frame_fails(self):
        with self.assertRaisesRegex(ValueError, "Duplicate detection frame 7"):
            runner.select_detections([{"frame": 7, "boxes": []}]*2, 7, 8)

    def test_explicit_device_does_not_silently_fallback(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False),
                                backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)))
        self.assertEqual(runner.choose_device(torch, "auto"), "cpu")
        with self.assertRaisesRegex(RuntimeError, "no silent CPU fallback"):
            runner.choose_device(torch, "mps")
        torch.backends.mps.is_available = lambda: True
        self.assertEqual(runner.choose_device(torch, "auto"), "mps")

    def test_end_frame_is_exclusive_and_model_config_is_explicit(self):
        args = runner.parse_args(["--video", "v.mp4", "--output", "p.jsonl", "--repo", "repo",
                                  "--checkpoint", "w.ckpt", "--mano-dir", "mano", "--detector", "d.pt",
                                  "--start-frame", "3000", "--end-frame", "3016",
                                  "--model-config", "config.yaml"])
        self.assertEqual(args.end_frame-args.start_frame, 16)
        self.assertEqual(args.model_config, Path("config.yaml"))
        self.assertEqual(len(runner.LANDMARK_NAMES), 21)
        self.assertEqual(runner.LANDMARK_NAMES[0], "wrist")
        self.assertEqual(runner.LANDMARK_NAMES[20], "pinky_tip")


if __name__ == "__main__":
    unittest.main()
