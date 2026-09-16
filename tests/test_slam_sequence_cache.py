import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

import cv2
import numpy as np
import zstandard as zstd

from aruco_track.models import BandLayout, Calibration, Pose
from aruco_track.slam_sequence_cache import SlamSequenceCache, sequence_cache_key, sequence_cache_root, snapshot_sequence
from export_action_labels import _marker_layouts_by_component, _run_deferred_head_slam


class SlamSequenceCacheTests(unittest.TestCase):
    def test_local_cache_override(self):
        with patch.dict(os.environ, {"SLAM_SEQUENCE_CACHE_DIR": "/tmp/slam-cache-test"}):
            self.assertEqual(sequence_cache_root(Path("/project")), Path("/tmp/slam-cache-test").resolve())
        with patch.dict(os.environ, {"SLAM_SEQUENCE_CACHE_DIR": ""}):
            self.assertEqual(sequence_cache_root(Path("/project")), Path("/project/output/.slam_sequence_cache"))

    def test_cross_device_snapshot_does_not_try_links(self):
        source, destination = Mock(), Mock()
        source.stat.return_value.st_dev = 1
        destination.parent.stat.return_value.st_dev = 2
        with patch("aruco_track.slam_sequence_cache.shutil.copytree") as copy:
            snapshot_sequence(source, destination)
        copy.assert_called_once_with(source, destination)

    def test_same_device_snapshot_preserves_bytes_and_pins_files(self):
        source = self.root / "source"
        source.mkdir()
        (source / "image").write_bytes(b"exact image bytes")
        destination = self.root / "snapshot"
        snapshot_sequence(source, destination)
        self.assertEqual((source / "image").stat().st_ino, (destination / "image").stat().st_ino)
        (source / "image").unlink()
        self.assertEqual((destination / "image").read_bytes(), b"exact image bytes")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.video = self.root / "raw.avi"
        self.video.write_bytes(b"video")
        self.observations = self.root / "observations.jsonl"
        self.observations.write_text("{}\n")
        self.calibration = Calibration(np.eye(3), np.zeros(5), (16, 12))
        self.layout = BandLayout("board", "DICT_4X4_50", {20: np.zeros((4, 3))})

    def tearDown(self):
        self.temporary.cleanup()

    def key(self):
        return sequence_cache_key(
            self.video, self.observations, self.calibration, self.layout, 60.0, 2
        )

    def test_rigid_board_layout_is_keyed_by_native_component_id(self):
        resolved = _marker_layouts_by_component(
            self.layout, None, ["world_board", "world_board"]
        )
        self.assertEqual(set(resolved), {"world_board"})
        self.assertIs(resolved["world_board"], self.layout)

    def test_key_changes_when_observations_change(self):
        first = self.key()
        self.observations.write_text('{"frame":0}\n')
        self.assertNotEqual(first, self.key())

    def test_key_includes_current_marker_pose_admission_and_weight(self):
        pose = Pose(np.zeros((3, 1)), np.array([[0.0], [0.0], [1.0]]), 0.2)

        def key_for(value, accepted=(20,), weight=1.0):
            return sequence_cache_key(
                self.video,
                self.observations,
                self.calibration,
                self.layout,
                60.0,
                1,
                [value],
                [0.9],
                [accepted],
                [{20: weight}],
            )

        baseline = key_for(pose)
        moved = Pose(pose.rvec.copy(), pose.tvec + np.array([[0.01], [0.0], [0.0]]), 0.2)
        self.assertNotEqual(baseline, key_for(moved))
        self.assertNotEqual(baseline, key_for(pose, accepted=()))
        self.assertNotEqual(baseline, key_for(pose, weight=0.25))

    def test_key_includes_runtime_detections(self):
        first = sequence_cache_key(
            self.video, self.observations, self.calibration, self.layout,
            60.0, 1, detections=[{20: np.zeros((4, 2))}],
        )
        second = sequence_cache_key(
            self.video, self.observations, self.calibration, self.layout,
            60.0, 1, detections=[{20: np.ones((4, 2))}],
        )
        self.assertNotEqual(first, second)

    def test_key_includes_disconnected_marker_component_assignment(self):
        layouts = {
            "room_a": self.layout,
            "room_b": BandLayout("room_b", "DICT_4X4_50", {21: np.zeros((4, 3))}),
        }
        first = sequence_cache_key(
            self.video, self.observations, self.calibration, None,
            60.0, 1, marker_layouts=layouts, active_submap_ids=["room_a"],
        )
        second = sequence_cache_key(
            self.video, self.observations, self.calibration, None,
            60.0, 1, marker_layouts=layouts, active_submap_ids=["room_b"],
        )
        self.assertNotEqual(first, second)

    def test_fresh_and_reused_derived_fields_have_same_prepared_key(self):
        raw_landmarks = np.linspace(0.1, 0.9, 42).reshape(21, 2).tolist()
        raw = {
            "frame": 0,
            "detected_marker_corners": {"20": [[1, 1], [2, 1], [2, 2], [1, 2]]},
            "boundary_rejected_marker_corners": {},
        }
        fresh = {
            **raw,
            "hands": {"left": {"joints": {
                "valid": True,
                "image_landmarks_normalized": raw_landmarks,
                "world_landmarks_m": [[99, 99, 99]],
            }}},
            "unassigned_hands": [],
            "camera_world_pose_fused": {"translation_m": [1, 2, 3]},
        }
        reused = {
            **raw,
            "hands": {"right": {"joints": {
                "valid": True,
                "image_landmarks_normalized": raw_landmarks,
                "world_landmarks_m": [[0, 0, 0]],
            }}},
            "unassigned_hands": [],
            "camera_world_pose_fused": None,
        }

        keys = []
        for name, record in (("fresh", fresh), ("reused", reused)):
            path = self.root / f"{name}.jsonl"
            path.write_text(json.dumps(record) + "\n")
            keys.append(sequence_cache_key(
                self.video, path, self.calibration, self.layout, 60.0, 1,
                detections=[{20: np.asarray(raw["detected_marker_corners"]["20"])}],
            ))
        self.assertEqual(keys[0], keys[1])

    def test_publish_lookup_and_corruption_rejection(self):
        cache = SlamSequenceCache(self.root / "cache")
        key = self.key()
        staging = cache.staging_directory(key)
        (staging / "rgb").mkdir()
        (staging / "rgb/000000.jpg").write_bytes(b"jpeg")
        (staging / "rgb.txt").write_text("0 rgb/000000.jpg\n")
        (staging / "tag_observations.txt").write_text("0 invalid\n")

        published = cache.publish(key, staging)
        self.assertFalse(published.hit)
        self.assertEqual(cache.lookup(key).path, published.path)

        (published.path / "rgb/000000.jpg").write_bytes(b"corrupt")
        self.assertIsNone(cache.lookup(key))

        replacement = cache.staging_directory(key)
        self.assertFalse(published.path.exists())
        (replacement / "rgb").mkdir()
        (replacement / "rgb/000000.jpg").write_bytes(b"jpeg-2")
        (replacement / "rgb.txt").write_text("0 rgb/000000.jpg\n")
        (replacement / "tag_observations.txt").write_text("0 invalid\n")
        self.assertEqual(cache.publish(key, replacement).path, published.path)

    def test_prune_never_removes_current_publication(self):
        cache = SlamSequenceCache(self.root / "cache", max_bytes=1, max_entries=1)
        key = self.key()
        staging = cache.staging_directory(key)
        (staging / "rgb").mkdir()
        (staging / "rgb/000000.jpg").write_bytes(b"jpeg")
        (staging / "rgb.txt").write_text("0 rgb/000000.jpg\n")
        (staging / "tag_observations.txt").write_text("0 invalid\n")
        entry = cache.publish(key, staging)
        self.assertTrue(entry.path.is_dir())
        self.assertEqual(json.loads((entry.path / "manifest.json").read_text())["key"], key)

    def test_prune_keeps_recently_used_entry_during_parallel_native_run(self):
        cache = SlamSequenceCache(
            self.root / "cache", max_bytes=1, max_entries=1, active_grace_s=600
        )
        key = self.key()
        staging = cache.staging_directory(key)
        (staging / "rgb").mkdir()
        (staging / "rgb/000000.jpg").write_bytes(b"jpeg")
        (staging / "rgb.txt").write_text("0 rgb/000000.jpg\n")
        (staging / "tag_observations.txt").write_text("0 invalid\n")
        entry = cache.publish(key, staging)
        cache.prune()
        self.assertTrue(entry.path.is_dir())

        old_ns = time.time_ns() - int(601 * 1e9)
        os.utime(entry.path, ns=(old_ns, old_ns))
        cache.prune()
        self.assertFalse(entry.path.exists())

    def test_prune_removes_only_abandoned_staging_directories(self):
        cache = SlamSequenceCache(self.root / "cache")
        old = cache.root / ".old-staging"
        recent = cache.root / ".recent-staging"
        old.mkdir(parents=True)
        recent.mkdir()
        os.utime(old, (1, 1))

        cache.prune()

        self.assertFalse(old.exists())
        self.assertTrue(recent.exists())

    def test_deferred_slam_reuses_prepared_sequence_without_changing_analysis(self):
        video = self.root / "two_frames.avi"
        writer = cv2.VideoWriter(
            str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (16, 12)
        )
        self.assertTrue(writer.isOpened())
        writer.write(np.zeros((12, 16, 3), dtype=np.uint8))
        writer.write(np.full((12, 16, 3), 20, dtype=np.uint8))
        writer.release()
        observations = self.root / "raw_observations.jsonl"
        observations.write_text(
            "\n".join(
                json.dumps({"hands": {}, "unassigned_hands": []})
                for _ in range(2)
            )
            + "\n"
        )
        calibration = Calibration(
            np.array([[20.0, 0.0, 8.0], [0.0, 20.0, 6.0], [0.0, 0.0, 1.0]]),
            np.zeros(5),
            (16, 12),
        )
        cache = SlamSequenceCache(self.root / "prepared")
        timings = []
        private_bootstrap_paths = []
        probe_history_bytes = b'{"timestamp":0.0,"final":true}\n'
        probe_hint_bytes = b'# original bootstrap hints\n0.000000000 0\n'

        def fake_bootstrap(_project, _sequence, work, *_args, **_kwargs):
            probe = work / 'marker_bootstrap/native'
            probe.mkdir(parents=True)
            history, hints = probe / 'frames.txt.history.jsonl', work / 'bootstrap_hints.txt'
            history.write_bytes(probe_history_bytes)
            hints.write_bytes(probe_hint_bytes)
            private_bootstrap_paths.append((history, hints))
            return SimpleNamespace(hints_path=hints, diagnostics=dict(
                accepted=True, reason='test_initialization_hint', published_observations=[],
                probe_history_path=str(history), hints_path=str(hints)))

        def fake_native(_project, sequence, _settings, output, tag_hints, **_kwargs):
            # Archival copies must not redirect the formal native input.
            self.assertEqual(tag_hints, sequence.parent / 'bootstrap_hints.txt')
            self.assertEqual(tag_hints.read_bytes(), probe_hint_bytes)
            self.assertTrue((sequence / "rgb/000001.jpg").is_file())
            (output / "native.log").write_text("ok\n")
            (output / "frames.txt.history.jsonl").write_text("{}\n")
            return {}, {}, np.empty((0, 3)), {}, {"Track": 0.01}

        def fake_result(_history, _poses, _confidences, _observations, _fps, timing, **_kwargs):
            from aruco_track.orbslam3_backend import MetricOrbSlamResult
            timings.append(dict(timing))
            return MetricOrbSlamResult([], np.empty((0, 3)), (), [], None, None,
                                       0, None, None, dict(timing))

        with (
            patch.dict(os.environ, {'ORB_SLAM3_OFFLINE_MARKER_BOOTSTRAP': '1'}),
            patch('aruco_track.marker_bootstrap.bootstrap_initial_marker_observations', side_effect=fake_bootstrap),
            patch("export_action_labels.SlamSequenceCache", return_value=cache),
            patch("export_action_labels.run_orbslam3_sequence", side_effect=fake_native),
            patch("export_action_labels.read_native_result", side_effect=fake_result),
            patch("aruco_track.orbslam3_backend.refine_final_frame_poses", side_effect=lambda result, *_args: result),
            patch("export_action_labels.prepare_slam_frame", wraps=lambda frame, _mask: frame.copy()) as prepare,
        ):
            for run in range(2):
                result = _run_deferred_head_slam(
                    video,
                    observations,
                    calibration,
                    [{}, {}],
                    [None, None],
                    [0.0, 0.0],
                    [(), ()],
                    None,
                    [None, None],
                    self.root / f"replay_{run}",
                    "auto",
                    None,
                    self.root / f"atlas_{run}.osa",
                    marker_weights=[{}, {}],
                )
                replay = self.root / f'replay_{run}'
                diagnostic = json.loads((replay / 'marker_bootstrap.json').read_text())
                self.assertEqual(result.offline_marker_bootstrap, diagnostic)
                self.assertEqual(Path(diagnostic['probe_history_path']), (replay / 'marker_bootstrap_history.jsonl.zst').resolve())
                self.assertEqual(Path(diagnostic['hints_path']), (replay / 'marker_bootstrap_hints.txt').resolve())
                self.assertEqual(Path(diagnostic['hints_path']).read_bytes(), probe_hint_bytes)
                with Path(diagnostic['probe_history_path']).open('rb') as source:
                    with zstd.ZstdDecompressor().stream_reader(source) as reader:
                        self.assertEqual(reader.read(), probe_history_bytes)

        self.assertTrue(all(not path.exists() for pair in private_bootstrap_paths for path in pair))
        self.assertEqual(prepare.call_count, 2)
        self.assertEqual(timings[0]["sequence_cache_hit"], 0.0)
        self.assertGreaterEqual(timings[0]["sequence_preparation_s"], 0.0)
        self.assertEqual(timings[1]["sequence_cache_hit"], 1.0)
        self.assertEqual(timings[1]["sequence_preparation_s"], 0.0)


if __name__ == "__main__":
    unittest.main()
