"""Offline HaWoR observations, with no unsupported inferred training labels."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import uuid

import cv2
import numpy as np

from .hands import (
    HandJointPose, RawHandJoints, TemporalHandAssignmentGate, _hand_band_geometry,
    assign_hands_to_bands, bend_angles, bind_landmarks_to_wrist,
)
from .models import Calibration

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HAWOR_CONFIG = ROOT / 'models/hawor/runtime.json'


def add_hand_backend_arguments(parser):
    parser.add_argument('--hand-backend', choices=('hawor', 'mediapipe'), default='hawor',
                        help='offline hand backend (default: HaWoR; live recording is unchanged)')
    parser.add_argument('--hawor-config', type=Path, default=DEFAULT_HAWOR_CONFIG,
                        help='HaWoR runtime/model configuration (project default: SSH hand inference)')
    parser.add_argument('--hawor-device', choices=('auto', 'mps', 'cuda', 'cpu'), default='auto')


def hawor_policy(enabled=True):
    return dict(version='hawor-observed-temporal-v1', enabled=enabled,
                temporal_context_frames=16, min_consecutive_frames=2,
                detector_supported_only=True, interpolated_training_labels=False,
                wrist_geometry='same distance and palm-axis gates as MediaPipe',
                execution='configured_hand_inference_only', slam_modified=False)


def _stamp(path, digest=False):
    path = Path(path).resolve()
    s = path.stat()
    out = dict(path=str(path), size=s.st_size, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)
    if digest:
        out['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def _effective_calibration(video, calibration_path, output_path):
    """Use the exporter's same-aspect resize rule without changing source intrinsics."""
    calibration_path = Path(calibration_path).resolve()
    original = Calibration.load(calibration_path)
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError(f'Cannot open video: {video}')
        size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        capture.release()
    if min(size + original.image_size) <= 0:
        raise ValueError('Video and calibration require positive image dimensions')
    if size == original.image_size:
        return calibration_path
    scaled = original.scaled_to(size)  # Reject crop/aspect changes, as the local exporter does.
    content = json.dumps(dict(image_size=list(scaled.image_size),
                             camera_matrix=scaled.camera_matrix.tolist(),
                             dist_coeffs=scaled.dist_coeffs.reshape(-1).tolist(),
                             source_calibration=_stamp(calibration_path, True),
                             transform='same-aspect intrinsic resize'), sort_keys=True, indent=2)+'\n'
    digest = hashlib.sha256(content.encode()).hexdigest()[:16]
    path = Path(output_path).resolve().parent/f'hawor_calibration_{digest}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != content:
            raise ValueError('Derived HaWoR calibration changed; refusing overwrite')
    else:
        with path.open('x') as handle:
            handle.write(content)
    return path


def prepare_hawor_predictions(video, calibration_path, output_path, max_frames=None,
                              config_path=DEFAULT_HAWOR_CONFIG, device='auto'):
    """Run the configured hand-only backend once, then cache raw observations."""
    config_path, output_path = Path(config_path).resolve(), Path(output_path).resolve()
    if not config_path.is_file():
        raise RuntimeError(f'HaWoR runtime missing: {config_path}; configure its resources '
                           'or explicitly select --hand-backend mediapipe')
    cfg = json.loads(config_path.read_text())
    calibration_path = _effective_calibration(video, calibration_path, output_path)
    if cfg.get('execution') == 'ssh':
        from .hawor_remote import prepare_remote_hawor
        return prepare_remote_hawor(video, calibration_path, output_path, max_frames,
                                    cfg, config_path, device)
    paths = {}
    for name in ('python', 'repo', 'checkpoint', 'model_config', 'mano_dir', 'detector'):
        value = Path(cfg[name]).expanduser()
        absolute = value if value.is_absolute() else config_path.parent/value
        # Resolving venv/bin/python's symlink selects the system interpreter and
        # silently loses the isolated ML dependencies. Keep the invocation path.
        paths[name] = absolute.absolute() if name == 'python' else absolute.resolve()
        if not paths[name].exists():
            raise RuntimeError(f'HaWoR {name} missing: {paths[name]}; no automatic cloud upload or fallback')
    runner = ROOT/'scripts/run_hawor_hands.py'
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f'cannot open video: {video}')
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if max_frames is not None:
        count = min(count, int(max_frames))
    if count <= 0 or fps <= 0:
        raise ValueError('HaWoR requires a video with valid frame count and FPS')
    from .hawor_remote import _engine_hashes
    signature = dict(schema='hawor-source-observations/v1', video=_stamp(video),
                     calibration=_stamp(calibration_path, True), config=_stamp(config_path, True),
                     runner=_stamp(runner, True), frames=count, fps=fps, device=device,
                     resources={key: _stamp(paths[key]) for key in ('checkpoint', 'model_config', 'detector')},
                     engine_sha256=hashlib.sha256(json.dumps(_engine_hashes(paths['repo']), sort_keys=True).encode()).hexdigest(),
                     mean_params=_stamp(paths['repo']/'_DATA/data/mano_mean_params.npz', True),
                     mano={side: _stamp(paths['mano_dir']/f'MANO_{side}.pkl') for side in ('LEFT', 'RIGHT')})
    meta_path = output_path.with_suffix('.meta.json')
    if output_path.exists():
        if not meta_path.is_file() or json.loads(meta_path.read_text()).get('signature') != signature:
            raise ValueError(f'HaWoR observation cache is unverified or different: {output_path}; choose a new output path')
        meta = json.loads(meta_path.read_text())
        if meta.get('prediction_file') != _stamp(output_path):
            raise ValueError('HaWoR prediction cache content changed; choose a new output path')
        return output_path, meta
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pending = output_path.with_name(f'.{output_path.stem}.{uuid.uuid4().hex}.jsonl')
    command = [str(paths['python']), str(runner), '--video', str(Path(video).resolve()),
               '--output', str(pending), '--start-frame', '0', '--end-frame', str(count),
               '--device', device, '--calibration', str(Path(calibration_path).resolve())]
    for name in ('repo', 'checkpoint', 'model_config', 'mano_dir', 'detector'):
        command.extend(['--'+name.replace('_', '-'), str(paths[name])])
    print(f'HaWoR: local {device} analysis of {count} source frames; no cloud upload', flush=True)
    subprocess.run(command, check=True, cwd=ROOT)
    _read_predictions(pending, fps, expected_count=count, expected_start=0)
    pending.replace(output_path)
    pending_metrics = pending.with_suffix('.metrics.json')
    metrics_path = output_path.with_suffix('.metrics.json')
    if pending_metrics.exists():
        pending_metrics.replace(metrics_path)
    provenance = dict(backend='hawor', signature=signature, prediction_file=_stamp(output_path),
                      policy=hawor_policy(), metrics=str(metrics_path),
                      frame_semantics='original frames; future context allowed; unsupported predictions excluded from labels')
    meta_path.write_text(json.dumps(provenance, indent=2)+'\n')
    return output_path, provenance


def _read_predictions(path, fps, expected_count=None, expected_start=None):
    with Path(path).open() as handle:
        rows = [json.loads(line) for line in handle]
    if not rows or (expected_count is not None and len(rows) != expected_count):
        raise ValueError('incomplete HaWoR prediction cache')
    start = rows[0]['frame'] if expected_start is None else expected_start
    for index, row in enumerate(rows, start):
        if row['frame'] != index or not np.isclose(row['timestamp_s'], index/fps, atol=1e-6):
            raise ValueError(f'HaWoR source frame/time mismatch at {index}')
        if not isinstance(row.get('hands'), list):
            raise ValueError(f'missing HaWoR hand candidates at {index}')
    return {row['frame']: row for row in rows}


class HaworHandTracker:
    """Bind actual supported HaWoR estimates to measured wrists, never fill gaps."""
    backend = 'hawor'

    def __init__(self, predictions_path, calibration, band_names, fps):
        self.rows = _read_predictions(predictions_path, fps)
        self.calibration, self.band_names, self.fps = calibration, list(band_names), float(fps)
        self.gate = TemporalHandAssignmentGate(self.band_names)
        self.last_frame = None
        self.skipped_predictions = 0

    def process(self, frame, timestamp_ms, band_poses, world_reference=None):
        index = int(round(float(timestamp_ms)*self.fps/1000.))
        if index not in self.rows:
            raise ValueError(f'no HaWoR source observation for frame {index}')
        if self.last_frame is not None and index != self.last_frame + 1:
            self.gate = TemporalHandAssignmentGate(self.band_names)
        self.last_frame = index
        raw, sources = [], {}
        for item in self.rows[index]['hands']:
            if not item.get('detector_supported', False) or item.get('image_supported') is False:
                self.skipped_predictions += 1
                continue
            uv = np.asarray(item['image_landmarks_normalized'], dtype=float)
            xyz = np.asarray(item['model_landmarks_m'], dtype=float)
            if uv.shape == (21, 2):
                uv = np.c_[uv, np.zeros(21)]
            if uv.shape != (21, 3) or xyz.shape != (21, 3) or not np.isfinite(uv).all() or not np.isfinite(xyz).all():
                raise ValueError(f'invalid HaWoR hand geometry at frame {index}')
            hand = RawHandJoints(item['handedness'], None, uv, xyz-xyz[0], 'hawor_temporal')
            raw.append(hand)
            sources[id(hand)] = item
        geometric = assign_hands_to_bands(raw, self.band_names, band_poses, self.calibration)
        matched = {id(hand): name for name, hand in geometric.items() if name in self.band_names}
        assigned = self.gate.update(geometric)
        out = {}
        for name, hand in assigned.items():
            item = sources[id(hand)]
            matched_band = matched.get(id(hand))
            pose = band_poses.get(name) if name in self.band_names else None
            error, angle, ratio = None, None, None
            if matched_band is not None:
                error, angle, length = _hand_band_geometry(hand, band_poses[matched_band], self.calibration)
                ratio = error/length if length > 1e-6 else None
            camera, band, world = bind_landmarks_to_wrist(hand.model_landmarks_m, pose, world_reference)
            out[name] = HandJointPose(
                name, hand.handedness, None, hand.image_landmarks_normalized, hand.model_landmarks_m,
                bend_angles(hand.model_landmarks_m), camera, band, world,
                wrist_anchor_error_px=error, wrist_anchor_axis_ratio=ratio, wrist_axis_error_deg=angle,
                wrist_association_confidence=None,
                association_status=('confirmed_hawor' if pose is not None else
                                    'temporal_unconfirmed' if matched_band else 'geometry_rejected'),
                temporal_confirmation_frames=self.gate.streak(matched_band) if matched_band else 0,
                detection_source='hawor_temporal', prediction_backend='hawor',
                detector_confidence=item.get('detection_score', item.get('detector_score')),
                temporal_future_frames=int(item.get('future_frames_used', 0)),
                keypoint_2d_source='projected_reconstruction', image_supported=True)
        return out

    def process_observations(self, frame, timestamp_ms, raw_hands, band_poses,
                             world_reference=None, protected_assignments=None):
        # Never protect or mix a different network's previous estimates.
        return self.process(frame, timestamp_ms, band_poses, world_reference)

    @property
    def recovery_diagnostics(self):
        return dict(policy=hawor_policy(), unsupported_predictions_excluded=self.skipped_predictions)

    def close(self):
        pass
