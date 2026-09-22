import copy
from pathlib import Path
import tempfile
import unittest

import zstandard as zstd

from tools.export_action_labels import _persist_marker_bootstrap_artifacts


class MarkerBootstrapArtifactTests(unittest.TestCase):
    def test_copies_survive_private_directory_cleanup_without_changing_evidence(self):
        history_bytes = b'{"timestamp":0.0,"pose":null}\n{"timestamp":0.1,"final":true}\n'
        hint_bytes = b'# original raw-corner hints\n0.000000000 0\n'
        with tempfile.TemporaryDirectory() as output:
            replay = Path(output) / 'replay'
            with tempfile.TemporaryDirectory() as native:
                history = Path(native) / 'frames.txt.history.jsonl'
                hints = Path(native) / 'hints.txt'
                history.write_bytes(history_bytes)
                hints.write_bytes(hint_bytes)
                diagnostics = dict(accepted=True, available_after_s=.1,
                    probe_history_path=str(history), hints_path=str(hints),
                    published_observations=[{'frame_id': 0, 'available_after_s': .1}])
                original = copy.deepcopy(diagnostics)
                persisted = _persist_marker_bootstrap_artifacts(diagnostics, replay)
                self.assertEqual(diagnostics, original)
                self.assertEqual(history.read_bytes(), history_bytes)
                self.assertEqual(hints.read_bytes(), hint_bytes)
            self.assertFalse(history.exists())
            self.assertFalse(hints.exists())
            self.assertEqual(Path(persisted['probe_history_path']), (replay / 'marker_bootstrap_history.jsonl.zst').resolve())
            self.assertEqual(Path(persisted['hints_path']), (replay / 'marker_bootstrap_hints.txt').resolve())
            with Path(persisted['probe_history_path']).open('rb') as source:
                with zstd.ZstdDecompressor().stream_reader(source) as reader:
                    self.assertEqual(reader.read(), history_bytes)
            self.assertEqual(Path(persisted['hints_path']).read_bytes(), hint_bytes)
            for field in ('accepted', 'available_after_s', 'published_observations'):
                self.assertEqual(persisted[field], original[field])

    def test_missing_old_evidence_is_reported_not_reconstructed(self):
        with tempfile.TemporaryDirectory() as output:
            replay = Path(output) / 'replay'
            diagnostics = dict(accepted=True, reason='existing_result',
                               probe_history_path=str(Path(output) / 'removed/history.jsonl'),
                               hints_path=str(Path(output) / 'removed/hints.txt'))
            original = copy.deepcopy(diagnostics)
            persisted = _persist_marker_bootstrap_artifacts(diagnostics, replay)
            self.assertEqual(diagnostics, original)
            self.assertTrue(persisted['accepted'])
            self.assertEqual(persisted['reason'], 'existing_result')
            self.assertEqual(persisted['missing_artifacts'],
                             {field: original[field] for field in ('probe_history_path', 'hints_path')})
            self.assertIsNone(persisted['probe_history_path'])
            self.assertIsNone(persisted['hints_path'])
            self.assertFalse((replay / 'marker_bootstrap_history.jsonl.zst').exists())
            self.assertFalse((replay / 'marker_bootstrap_hints.txt').exists())

    def test_skipped_probe_does_not_create_fake_artifacts(self):
        with tempfile.TemporaryDirectory() as output:
            replay = Path(output) / 'replay'
            diagnostics = dict(accepted=False, reason='initial_reliable_marker_already_available')
            self.assertEqual(_persist_marker_bootstrap_artifacts(diagnostics, replay), diagnostics)
            self.assertFalse(replay.exists())


if __name__ == '__main__':
    unittest.main()
