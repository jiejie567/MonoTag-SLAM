import json
import tempfile
import unittest
from pathlib import Path
from aruco_track.processing_host import require_processing_host


class ProcessingHostTests(unittest.TestCase):
    def test_mac_refuses_and_linux_accepts(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'policy.json'
            path.write_text(json.dumps(dict(allow_macos_processing=False,
                ssh_host='ubuntu',project='/project',entrypoint='process_monotag.py')))
            with self.assertRaisesRegex(RuntimeError, 'no local fallback'):
                require_processing_host(path, 'darwin')
            require_processing_host(path, 'linux')

    def test_no_policy_preserves_other_installations(self):
        with tempfile.TemporaryDirectory() as directory:
            require_processing_host(Path(directory)/'missing.json', 'darwin')
