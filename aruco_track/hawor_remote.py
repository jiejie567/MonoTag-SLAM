"""Verified SSH execution of the exact local HaWoR hand-only runner."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import time
import uuid

import cv2


ROOT = Path(__file__).resolve().parents[1]
_EXCLUDED = {'.git', '__pycache__'}
_AUDIT = r'''
import hashlib, json, os, sys
def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
request = json.load(sys.stdin)
out = {'files': {}}
for name, path in request.get('paths', {}).items():
    out['files'][name] = digest(path) if os.path.isfile(path) else None
if request.get('repo'):
    root = os.path.realpath(request['repo'])
    engine = {}
    for directory, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in ('.git', '__pycache__'))
        for name in sorted(files):
            if name.endswith('.py'):
                path = os.path.join(directory, name)
                if os.path.commonpath([root, os.path.realpath(path)]) != root:
                    raise ValueError('Engine symlink leaves its repository')
                engine[os.path.relpath(path, root).replace(os.sep, '/')] = digest(path)
    out['engine'] = engine
out['python'] = sys.executable
out['python_version'] = sys.version
print(json.dumps(out, sort_keys=True))
'''
_PUBLISH = r'''
import hashlib, os, sys
pending, target, expected = sys.argv[1:]
def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
if digest(pending) != expected:
    raise ValueError('Uploaded content SHA256 mismatch')
try:
    os.link(pending, target)
except FileExistsError:
    if digest(target) != expected:
        raise ValueError('Refusing to overwrite mismatched remote content')
os.unlink(pending)
'''


def _digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def _engine_hashes(repo):
    repo = Path(repo).resolve()
    result = {}
    for path in sorted(repo.rglob('*.py')):
        relative = path.relative_to(repo)
        if _EXCLUDED.intersection(relative.parts):
            continue
        if not path.resolve().is_relative_to(repo):
            raise ValueError('Engine symlink leaves its repository')
        result[relative.as_posix()] = _digest(path)
    if 'lib/models/hawor.py' not in result:
        raise ValueError(f'Official HaWoR engine not found at {repo}')
    return result


def _remote_path(value):
    path = PurePosixPath(str(value))
    if not path.is_absolute() or '..' in path.parts or str(path) == '/':
        raise ValueError(f'Remote path must be explicit and absolute: {value}')
    return str(path)


def prepare_remote_hawor(video, calibration_path, output_path, max_frames,
                         cfg, config_path, device):
    """Return verified raw observations, never silently fall back or relabel gaps."""
    from .hawor_backend import _read_predictions, _stamp

    if cfg.get('execution') != 'ssh' or device not in ('auto', 'cuda'):
        raise ValueError('SSH HaWoR requires execution=ssh and device=auto or cuda; no local fallback')
    video, calibration_path = Path(video).resolve(), Path(calibration_path).resolve()
    config_path, output_path = Path(config_path).resolve(), Path(output_path).resolve()
    remote = cfg['remote']
    host = str(remote['host'])
    if not re.fullmatch(r'(?:[A-Za-z0-9_][A-Za-z0-9_.-]*@)?[A-Za-z0-9_][A-Za-z0-9_.-]*', host):
        raise ValueError('Invalid SSH host; supply a host name, not shell arguments')
    port = int(remote['port'])
    if not 1 <= port <= 65535 or not re.fullmatch(r'\d+', str(remote['cuda_device'])):
        raise ValueError('Invalid SSH port or isolated CUDA device')
    remote_paths = {key: _remote_path(remote[key]) for key in
                    ('root', 'python', 'repo', 'checkpoint', 'model_config', 'mano_dir', 'detector')}
    for key in ('checkpoint_sha256', 'detector_sha256'):
        if not re.fullmatch(r'[0-9a-f]{64}', str(remote[key])):
            raise ValueError(f'Missing verified {key}')

    def local_path(key):
        path = Path(cfg[key]).expanduser()
        return path.resolve() if path.is_absolute() else (config_path.parent/path).resolve()

    def stamped(path):
        return dict(_stamp(path), sha256=_digest(path))

    runner = ROOT/'scripts/run_hawor_hands.py'
    engine = _engine_hashes(local_path('repo'))
    engine_sha = hashlib.sha256(json.dumps(engine, sort_keys=True).encode()).hexdigest()
    local_assets = {'model_config': local_path('model_config'),
                    'mean_params': local_path('repo')/'_DATA/data/mano_mean_params.npz',
                    'mano_left': local_path('mano_dir')/'MANO_LEFT.pkl',
                    'mano_right': local_path('mano_dir')/'MANO_RIGHT.pkl'}
    local_stamps = {key: stamped(path) for key, path in local_assets.items()}
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f'Cannot open video: {video}')
    count, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if max_frames is not None:
        count = min(count, int(max_frames))
    if count <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError('HaWoR requires a positive source frame count and FPS')
    signature = dict(schema='hawor-ssh-observations/v1', video=stamped(video),
                     calibration=stamped(calibration_path), config=stamped(config_path),
                     runner=stamped(runner), engine_sha256=engine_sha, engine_file_count=len(engine),
                     local_assets=local_stamps, frames=count, fps=fps, device='cuda',
                     remote={**remote_paths, 'host': host, 'port': port,
                             'cuda_device': str(remote['cuda_device']),
                             'checkpoint_sha256': remote['checkpoint_sha256'],
                             'detector_sha256': remote['detector_sha256']})
    meta_path, metrics_path = output_path.with_suffix('.meta.json'), output_path.with_suffix('.metrics.json')
    if any(path.exists() for path in (output_path, meta_path, metrics_path)):
        if not all(path.is_file() for path in (output_path, meta_path, metrics_path)):
            raise ValueError('Incomplete HaWoR cache; choose a new output path')
        meta = json.loads(meta_path.read_text())
        if meta.get('signature') != signature or meta.get('prediction_file') != stamped(output_path) or meta.get('metrics_file') != stamped(metrics_path):
            raise ValueError('HaWoR cache differs or is unverified; refusing to overwrite it')
        _read_predictions(output_path, fps, expected_count=count, expected_start=0)
        return output_path, meta

    output_path.parent.mkdir(parents=True, exist_ok=True)
    job_id = uuid.uuid4().hex
    local_job = output_path.parent/f'.{output_path.stem}.ssh-{job_id}'
    local_job.mkdir()
    log_path = local_job/'diagnostics.log'
    remote_job = remote_paths['root']+'/jobs/'+job_id
    runner_sha = signature['runner']['sha256']
    remote_runner = remote_paths['root']+'/runners/'+runner_sha+'.py'
    shared_video = remote_paths['root']+'/input/'+signature['video']['sha256']+video.suffix
    seeded_video = remote.get('seed_videos', {}).get(signature['video']['sha256'])
    remote_video = _remote_path(seeded_video) if seeded_video else shared_video
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=30',
           '-o', 'ServerAliveCountMax=3', '-p', str(port), host]
    # Explicit SFTP avoids a remote shell interpreting scp file names.
    scp = ['scp', '-s', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-P', str(port)]

    def run(argv, payload=None):
        result = subprocess.run(argv, input=None if payload is None else json.dumps(payload),
                                text=True, capture_output=True, check=False)
        with log_path.open('a') as log:
            log.write(shlex.join(argv)+'\n'+result.stdout+'\n'+result.stderr+'\n')
        if result.returncode:
            raise RuntimeError(f'Server HaWoR command failed ({result.returncode}); diagnostics: {log_path}')
        return result.stdout

    def ssh_run(arguments, payload=None):
        return run(ssh+[shlex.join([str(x) for x in arguments])], payload)

    def audit(paths, include_engine=False):
        payload = {'paths': paths}
        if include_engine:
            payload['repo'] = remote_paths['repo']
        return json.loads(ssh_run([remote_paths['python'], '-c', _AUDIT], payload))

    def upload(source, target, expected):
        pending = remote_job+'/'+uuid.uuid4().hex+'.upload'
        run(scp+[str(source), host+':'+pending])
        ssh_run([remote_paths['python'], '-c', _PUBLISH, pending, target, expected])

    started = time.perf_counter()
    try:
        assets = {'checkpoint': remote_paths['checkpoint'], 'detector': remote_paths['detector'],
                  'model_config': remote_paths['model_config'],
                  'mean_params': remote_paths['repo']+'/_DATA/data/mano_mean_params.npz',
                  'mano_left': remote_paths['mano_dir']+'/MANO_LEFT.pkl',
                  'mano_right': remote_paths['mano_dir']+'/MANO_RIGHT.pkl',
                  'runner': remote_runner, 'video': remote_video}
        checked = audit(assets, include_engine=True)
        if checked.get('engine') != engine:
            raise ValueError('Local/server HaWoR Python engine SHA256 lists differ; no execution')
        expected = {key: stamp['sha256'] for key, stamp in local_stamps.items()}
        expected.update(checkpoint=remote['checkpoint_sha256'], detector=remote['detector_sha256'])
        for key, digest in expected.items():
            if checked['files'].get(key) != digest:
                raise ValueError(f'Server {key} SHA256 mismatch; no execution')
        for key, digest in (('runner', runner_sha), ('video', signature['video']['sha256'])):
            if checked['files'].get(key) not in (None, digest):
                raise ValueError(f'Server {key} SHA256 mismatch; refusing overwrite')
        ssh_run(['mkdir', '-p', remote_job, remote_paths['root']+'/runners', remote_paths['root']+'/input'])
        if checked['files'].get('runner') is None:
            upload(runner, remote_runner, runner_sha)
        reused_video = checked['files'].get('video') is not None
        if not reused_video:
            remote_video = shared_video
            existing = audit({'video': remote_video})['files']['video']
            if existing not in (None, signature['video']['sha256']):
                raise ValueError('Content-addressed video SHA256 mismatch; refusing overwrite')
            if existing is None:
                upload(video, remote_video, signature['video']['sha256'])
            else:
                reused_video = True
        remote_calibration = remote_job+'/calibration.json'
        upload(calibration_path, remote_calibration, signature['calibration']['sha256'])
        remote_output = remote_job+'/predictions.jsonl'
        command = ['env', 'CUDA_VISIBLE_DEVICES='+str(remote['cuda_device']), 'PYTHONNOUSERSITE=1',
                   remote_paths['python'], remote_runner, '--video', remote_video, '--output', remote_output,
                   '--start-frame', '0', '--end-frame', str(count), '--device', 'cuda',
                   '--calibration', remote_calibration]
        for key in ('repo', 'checkpoint', 'model_config', 'mano_dir', 'detector'):
            command.extend(['--'+key.replace('_', '-'), remote_paths[key]])
        ssh_run(command)
        pending, pending_metrics = local_job/'predictions.jsonl', local_job/'predictions.metrics.json'
        run(scp+[host+':'+remote_output, str(pending)])
        run(scp+[host+':'+remote_job+'/predictions.metrics.json', str(pending_metrics)])
        _read_predictions(pending, fps, expected_count=count, expected_start=0)
        metrics = json.loads(pending_metrics.read_text())
        required = dict(status='complete', device='cuda', runner_sha256=runner_sha,
                        start_frame=0, end_frame_exclusive=count, frames=count,
                        repo=remote_paths['repo'], checkpoint=remote_paths['checkpoint'], video=remote_video,
                        marker_camera_wrist_modified=False, slam_used=False, infiller_used=False)
        if any(metrics.get(key) != value for key, value in required.items()) or not math.isclose(float(metrics.get('fps', 0)), fps, abs_tol=1e-6):
            raise ValueError('Server result provenance/device/frame mismatch; result not published')
        if any(path.exists() for path in (output_path, meta_path, metrics_path)):
            raise ValueError('Output appeared while job was running; refusing overwrite')
        pending.replace(output_path)
        pending_metrics.replace(metrics_path)
        provenance = dict(backend='hawor', execution='ssh', signature=signature,
                          prediction_file=stamped(output_path), metrics_file=stamped(metrics_path),
                          metrics=str(metrics_path), engine_files=engine, remote_audit=checked,
                          remote_job=remote_job, remote_runner=remote_runner, remote_video=remote_video,
                          reused_video=reused_video, elapsed_s=time.perf_counter()-started,
                          diagnostics=str(log_path), policy=dict(detector_supported_only=True,
                          interpolated_training_labels=False, temporal_context_frames=16,
                          execution='ssh', device='cuda'),
                          frame_semantics='original source frames; future context allowed; unsupported predictions excluded')
        pending_meta = local_job/'provenance.json'
        pending_meta.write_text(json.dumps(provenance, indent=2)+'\n')
        pending_meta.replace(meta_path)
        return output_path, provenance
    except Exception as error:
        (local_job/'error.json').write_text(json.dumps(dict(error=str(error), remote_job=remote_job,
                                                       signature=signature), indent=2)+'\n')
        raise
