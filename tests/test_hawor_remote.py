"""Mocked SSH contract tests: identity checks, quoting, cache safety, no fallback."""
import json
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from aruco_track import hawor_remote as remote


class RemoteHaWoRTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='hawor remote ')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        for name, content in {
            'scripts/run_hawor_hands.py': 'print("fixture only")\n',
            'repo/lib/models/hawor.py': '# official engine fixture\n',
            'repo/_DATA/data/mano_mean_params.npz': 'mean',
            'repo/.git/ignored.py': '# ignored',
            'repo/__pycache__/ignored.py': '# ignored',
            'model config.yaml': 'config', 'mano/MANO_LEFT.pkl': 'left',
            'mano/MANO_RIGHT.pkl': 'right', 'clip ; literal.mp4': 'video bytes',
            'calibration.json': '{}',
        }.items():
            path = self.root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        self.video = self.root/'clip ; literal.mp4'
        self.calibration = self.root/'calibration.json'
        self.output = self.root/'result.jsonl'
        self.config_path = self.root/'runtime.json'
        self.cfg = dict(execution='ssh', repo='repo', model_config='model config.yaml', mano_dir='mano',
                        remote=dict(host='root@example.test', port=1022, root='/service ; literal',
                                    python='/env path/python', repo='/model repo', checkpoint='/model checkpoint',
                                    model_config='/model config', mano_dir='/private mano', detector='/detector',
                                    cuda_device='2', checkpoint_sha256='a'*64, detector_sha256='b'*64,
                                    seed_videos={remote._digest(self.video): '/existing video/clip.mp4'}))
        self.config_path.write_text(json.dumps(self.cfg))
        self.calls, self.uploads = [], []
        self.bad_resource = None
        self.bad_engine = False
        self.result_device = 'cuda'
        self.remote_video_missing = False
        self.inference_returncode = 0
        self.inference = None
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.side_effect = lambda key: 4 if key == remote.cv2.CAP_PROP_FRAME_COUNT else 90.
        self.patches = [patch.object(remote, 'ROOT', self.root),
                        patch.object(remote.cv2, 'VideoCapture', return_value=cap),
                        patch.object(remote.subprocess, 'run', side_effect=self.fake_run)]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def prepare(self, device='auto'):
        return remote.prepare_remote_hawor(self.video, self.calibration, self.output, 2,
                                          self.cfg, self.config_path, device)

    def fake_run(self, argv, **kwargs):
        self.calls.append(argv)
        self.assertFalse(kwargs['check'])
        self.assertTrue(kwargs['capture_output'])
        if argv[0] == 'ssh':
            command = shlex.split(argv[-1])
            if remote._AUDIT in command:
                request = json.loads(kwargs['input'])
                fixtures = {'model_config': self.root/'model config.yaml',
                            'mean_params': self.root/'repo/_DATA/data/mano_mean_params.npz',
                            'mano_left': self.root/'mano/MANO_LEFT.pkl',
                            'mano_right': self.root/'mano/MANO_RIGHT.pkl', 'video': self.video}
                files = {key: (remote._digest(fixtures[key]) if key in fixtures else
                               'a'*64 if key == 'checkpoint' else 'b'*64 if key == 'detector' else None)
                         for key in request['paths']}
                if self.bad_resource in files:
                    files[self.bad_resource] = '0'*64
                if self.remote_video_missing and 'video' in files:
                    files['video'] = None
                out = dict(files=files, python='/env path/python', python_version='test')
                if 'repo' in request:
                    out['engine'] = remote._engine_hashes(self.root/'repo')
                    if self.bad_engine:
                        out['engine']['lib/models/hawor.py'] = '0'*64
                return subprocess.CompletedProcess(argv, 0, json.dumps(out), '')
            if command[0] == 'env':
                self.inference = command
                return subprocess.CompletedProcess(argv, self.inference_returncode, '',
                                                   'fixture CUDA error' if self.inference_returncode else '')
            return subprocess.CompletedProcess(argv, 0, '', '')
        self.assertEqual(argv[0], 'scp')
        self.assertIn('-s', argv)  # SFTP, no remote shell expansion of filenames.
        source, target = argv[-2:]
        if not source.startswith('root@example.test:'):
            self.uploads.append((source, target))
        elif source.endswith('predictions.metrics.json'):
            command = self.inference
            option = lambda name: command[command.index(name)+1]
            metrics = dict(status='complete', device=self.result_device,
                           runner_sha256=remote._digest(self.root/'scripts/run_hawor_hands.py'),
                           start_frame=0, end_frame_exclusive=2, frames=2, fps=90.,
                           repo=option('--repo'), checkpoint=option('--checkpoint'), video=option('--video'),
                           marker_camera_wrist_modified=False, slam_used=False, infiller_used=False)
            Path(target).write_text(json.dumps(metrics))
        else:
            Path(target).write_text(''.join(json.dumps(dict(frame=f, timestamp_s=f/90., hands=[]))+'\n' for f in range(2)))
        return subprocess.CompletedProcess(argv, 0, '', '')

    def test_verified_seed_cuda_execution_and_safe_argument_quoting(self):
        path, provenance = self.prepare()
        self.assertEqual(path, self.output)
        self.assertTrue(provenance['reused_video'])
        self.assertEqual(provenance['signature']['frames'], 2)
        self.assertEqual(provenance['signature']['engine_file_count'], 1)
        self.assertIn('CUDA_VISIBLE_DEVICES=2', self.inference)
        self.assertEqual(self.inference[self.inference.index('--device')+1], 'cuda')
        self.assertEqual(self.inference[self.inference.index('--start-frame')+1], '0')
        self.assertEqual(self.inference[self.inference.index('--end-frame')+1], '2')
        self.assertEqual(self.inference[self.inference.index('--repo')+1], '/model repo')
        self.assertTrue(self.inference[4].startswith('/service ; literal/runners/'))
        self.assertNotIn(str(self.video), [source for source, _ in self.uploads])
        self.assertEqual(len(self.uploads), 2)  # Script and per-job calibration only.
        self.assertTrue(all(argv[0] in ('ssh', 'scp') for argv in self.calls))

    def test_resource_hash_mismatch_stops_without_model_execution_or_fallback(self):
        self.bad_resource = 'checkpoint'
        with self.assertRaisesRegex(ValueError, 'checkpoint SHA256 mismatch'):
            self.prepare()
        self.assertIsNone(self.inference)
        self.assertFalse(self.output.exists())
        self.assertTrue(list(self.root.glob('.result.ssh-*/error.json')))
        self.assertTrue(all(argv[0] == 'ssh' for argv in self.calls))

    def test_missing_video_uploaded_to_verified_content_address(self):
        self.remote_video_missing = True
        _, provenance = self.prepare()
        expected = '/service ; literal/input/'+remote._digest(self.video)+'.mp4'
        self.assertEqual(provenance['remote_video'], expected)
        self.assertFalse(provenance['reused_video'])
        self.assertEqual(len([source for source, _ in self.uploads if source == str(self.video)]), 1)
        publications = [shlex.split(argv[-1]) for argv in self.calls
                        if argv[0] == 'ssh' and remote._PUBLISH in shlex.split(argv[-1])]
        self.assertTrue(any(command[-2:] == [expected, remote._digest(self.video)]
                            for command in publications))

    def test_ssh_failure_preserves_diagnostics_without_fallback(self):
        self.inference_returncode = 17
        with self.assertRaisesRegex(RuntimeError, 'command failed \\(17\\)'):
            self.prepare()
        self.assertFalse(self.output.exists())
        logs = list(self.root.glob('.result.ssh-*/diagnostics.log'))
        self.assertEqual(len(logs), 1)
        self.assertIn('fixture CUDA error', logs[0].read_text())
        self.assertTrue(list(self.root.glob('.result.ssh-*/error.json')))
        self.assertTrue(all(argv[0] in ('ssh', 'scp') for argv in self.calls))

    def test_engine_hash_mismatch_stops(self):
        self.bad_engine = True
        with self.assertRaisesRegex(ValueError, 'engine SHA256 lists differ'):
            self.prepare()
        self.assertIsNone(self.inference)
        self.assertFalse(self.uploads)

    def test_cpu_result_not_published(self):
        self.result_device = 'cpu'
        with self.assertRaisesRegex(ValueError, 'provenance/device/frame mismatch'):
            self.prepare()
        self.assertFalse(self.output.exists())
        self.assertTrue(list(self.root.glob('.result.ssh-*/predictions.jsonl')))
        self.assertTrue(list(self.root.glob('.result.ssh-*/diagnostics.log')))

    def test_matching_cache_reused_without_network(self):
        _, first = self.prepare()
        self.calls.clear()
        _, second = self.prepare()
        self.assertEqual(first, second)
        self.assertFalse(self.calls)

    def test_local_engine_change_refuses_to_overwrite_cache(self):
        self.prepare()
        original = self.output.read_bytes()
        (self.root/'repo/lib/models/hawor.py').write_text('# changed engine\n')
        self.calls.clear()
        with self.assertRaisesRegex(ValueError, 'refusing to overwrite'):
            self.prepare()
        self.assertEqual(self.output.read_bytes(), original)
        self.assertFalse(self.calls)

    def test_explicit_mps_and_shell_host_fail_closed(self):
        with self.assertRaisesRegex(ValueError, 'no local fallback'):
            self.prepare(device='mps')
        self.cfg['remote']['host'] = 'host; touch /tmp/not-allowed'
        with self.assertRaisesRegex(ValueError, 'Invalid SSH host'):
            self.prepare()
        self.cfg['remote']['host'] = '-oProxyCommand=command@example.test'
        with self.assertRaisesRegex(ValueError, 'Invalid SSH host'):
            self.prepare()
        self.assertFalse(self.calls)


if __name__ == '__main__':
    unittest.main()
