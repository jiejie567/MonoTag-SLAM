import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('init_runtime', Path(__file__).resolve().parents[1] / 'scripts/init_runtime.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


class RuntimeSetupTest(unittest.TestCase):
    def test_binds_source_build_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            native = root / 'third_party/ORB_SLAM3'
            for rel in ('lib/libORB_SLAM3.so', 'Examples/Monocular/mono_tum_headless',
                        'Examples/Monocular/relocalize_gap', 'Examples/Monocular/relocalize_prefix_readonly'):
                f = native / rel
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_bytes(b'test binary')
            for d in (root / 'deps/opencv-4.10/lib', root / 'deps/pangolin/lib',
                      native / 'Thirdparty/DBoW2/lib', native / 'Thirdparty/g2o/lib'):
                d.mkdir(parents=True, exist_ok=True)
            vocab = root / 'words.txt'
            vocab.write_text('test vocabulary')
            argv = ['init_runtime', '--deps-prefix', str(root / 'deps'), '--vocabulary', str(vocab)]
            with patch.object(runtime, 'ROOT', root), patch('sys.platform', 'linux'), patch('sys.argv', argv):
                runtime.main()
                cfg = json.loads((root / '.monotag/runtime.json').read_text())
                self.assertEqual(cfg['native_project'], str(root))
                self.assertEqual(cfg['python_paths'], [])
                self.assertEqual(cfg['static_corner_refinement'], 'apriltag')
                self.assertEqual((native / 'Vocabulary/ORBvoc.txt').resolve(), vocab)
                self.assertEqual(len(cfg['gap_adapter_sha256']), 64)
                with self.assertRaises(SystemExit):
                    runtime.main()

    def test_missing_build_writes_nothing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            argv = ['init_runtime', '--deps-prefix', str(root), '--vocabulary', str(root / 'absent')]
            with patch.object(runtime, 'ROOT', root), patch('sys.platform', 'linux'), patch('sys.argv', argv):
                with self.assertRaises(SystemExit):
                    runtime.main()
            self.assertFalse((root / '.monotag').exists())


if __name__ == '__main__':
    unittest.main()
