#!/usr/bin/env python3
import argparse
import unittest
from pathlib import Path
import io
import tempfile
from unittest.mock import patch

from tools.track import offline_processing_command


class OfflineProcessingCommandTests(unittest.TestCase):
    def test_dynamic_filter_defaults_off_and_can_be_enabled(self):
        from tools.export_action_labels import main

        class ArgumentsCaptured(Exception):
            pass

        parse_args = argparse.ArgumentParser.parse_args
        for flags, expected in (([], False), (["--slam-dynamic-filter"], True),
                                (["--no-slam-dynamic-filter"], False)):
            with self.subTest(flags=flags):
                captured = []

                def capture(parser, *args, **kwargs):
                    captured.append(parse_args(parser, *args, **kwargs))
                    raise ArgumentsCaptured

                arguments = ["tools/export_action_labels.py", "raw.avi", "--head-slam", *flags]
                with patch("sys.argv", arguments), patch.object(argparse.ArgumentParser, "parse_args", capture):
                    with self.assertRaises(ArgumentsCaptured):
                        main()
                self.assertEqual(captured[0].slam_dynamic_filter, expected)

    def test_existing_explicit_debug_video_is_rejected_before_analysis(self):
        from tools.export_action_labels import main
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / 'existing.mp4'
            video.write_bytes(b'keep existing video')
            arguments = ['tools/export_action_labels.py', 'raw.avi', '--band', 'band.json',
                         '--world-board', 'board.json', '--head-slam',
                         '--output', str(root/'new'/'actions.jsonl'),
                         '--slam-debug-video', str(video)]
            with patch('sys.argv', arguments), patch('sys.stderr', new_callable=io.StringIO) as error:
                with self.assertRaises(SystemExit) as raised:
                    main()
            self.assertEqual(raised.exception.code, 2)
            self.assertIn('debug video output exists', error.getvalue())
            self.assertEqual(video.read_bytes(), b'keep existing video')

    def test_world_board_enables_head_slam_and_diagnostics(self):
        command = offline_processing_command(
            Path("recordings/raw.avi"),
            "calib/camera.json",
            ["left.json", "right.json"],
            "world.json",
            "models/hand.task",
        )
        self.assertIn("--head-slam", command)
        self.assertIn("--graph-diagnostics", command)
        self.assertIn("--slam-debug-video", command)
        self.assertEqual(command.count("--band"), 2)
        self.assertEqual(command[command.index('--hand-backend') + 1], 'hawor')
        self.assertEqual(command[command.index('--hawor-device') + 1], 'auto')

    def test_offline_mediapipe_is_explicit_and_hawor_options_do_not_leak(self):
        command = offline_processing_command(
            Path('recordings/raw.avi'), 'calib/camera.json', ['left.json'], None,
            'models/hand.task', hand_backend='mediapipe')
        self.assertEqual(command[command.index('--hand-backend') + 1], 'mediapipe')
        self.assertNotIn('--hawor-config', command)
        self.assertNotIn('--hawor-device', command)

    def test_offline_hawor_device_and_config_are_forwarded(self):
        command = offline_processing_command(
            Path('recordings/raw.avi'), 'calib/camera.json', ['left.json'], None,
            'models/hand.task', hawor_config=Path('models/local_hawor.json'), hawor_device='mps')
        self.assertEqual(command[command.index('--hawor-config') + 1], 'models/local_hawor.json')
        self.assertEqual(command[command.index('--hawor-device') + 1], 'mps')

    def test_without_world_board_defaults_to_independent_marker_map(self):
        command = offline_processing_command(
            Path("recordings/raw.avi"),
            "calib/camera.json",
            ["left.json"],
            None,
            "models/hand.task",
        )
        self.assertIn("--auto-marker-map", command)
        self.assertIn("--graph-diagnostics", command)
        self.assertIn("--hand-model", command)

    def test_independent_marker_map_can_be_disabled(self):
        command = offline_processing_command(
            Path("recordings/raw.avi"),
            "calib/camera.json",
            ["left.json"],
            None,
            "models/hand.task",
            auto_marker_map=False,
        )
        self.assertNotIn("--auto-marker-map", command)
        self.assertNotIn("--head-slam", command)
        self.assertIn("--hand-model", command)

    def test_disabled_hand_joints_are_forwarded_to_offline_export(self):
        command = offline_processing_command(
            Path("recordings/raw.avi"),
            "calib/camera.json",
            ["left.json"],
            None,
            "models/hand.task",
            hand_joints=False,
        )
        self.assertIn("--no-hand-joints", command)

    def test_auto_marker_map_enables_offline_mapping_without_known_layout(self):
        command = offline_processing_command(
            Path("recordings/raw.avi"),
            "calib/camera.json",
            ["left.json"],
            None,
            "models/hand.task",
            auto_marker_map=True,
            static_marker_ids="20-35",
            static_marker_size_mm=48.0,
        )
        self.assertIn("--auto-marker-map", command)
        self.assertIn("--static-marker-ids", command)
        self.assertIn("20-35", command)
        self.assertIn("--graph-diagnostics", command)
        self.assertIn("--slam-debug-video", command)


if __name__ == "__main__":
    unittest.main()
