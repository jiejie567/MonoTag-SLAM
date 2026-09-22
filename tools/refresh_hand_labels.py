#!/usr/bin/env python3
"""Upgrade hand measurements from source RGB, with the finished SLAM/wrists frozen."""
from __future__ import annotations

# Support direct execution from a source checkout.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import time

import cv2

from aruco_track.hands import (
    HandJointTracker, joint_pose_to_dict, raw_hand_from_dict,
)
from aruco_track.hand_recovery import hand_recovery_policy
from aruco_track.hawor_backend import (
    HaworHandTracker, add_hand_backend_arguments, hawor_policy, prepare_hawor_predictions,
)
from aruco_track.models import BandLayout
from aruco_track.slam_replay import _pose_from_action, _replay_wrist_camera
from tools.export_action_labels import (
    _rebind_cached_joints, _world_landmarks, _observation_input_fingerprints,
)
from tools.render_slam_replay import load_replay_calibration


def frozen_fields(record):
    """All non-hand-label fields, including raw observations and every wrist pose."""
    value = {key: item for key, item in record.items()
             if key not in ('hands', 'unassigned_hands')}
    value['hands'] = {name: {key: item for key, item in hand.items() if key != 'joints'}
                      for name, hand in record.get('hands', {}).items()}
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def refresh_record(record, frame, tracker, calibration, *, redetect=False):
    """Supplement current-image candidates; never create a missing wrist/camera."""
    result = copy.deepcopy(record)
    measurements, protected = [], {}
    reuse_hands = not redetect and getattr(tracker, 'backend', 'mediapipe') == 'mediapipe'
    for name, hand in record['hands'].items() if reuse_hands else ():
        joints = hand.get('joints', {})
        if joints.get('valid'):
            raw = raw_hand_from_dict(joints)
            measurements.append(raw)
            protected[name] = raw
    if reuse_hands:
        measurements.extend(raw_hand_from_dict(joints)
                            for joints in record.get('unassigned_hands', []) if joints.get('valid'))
    poses = {}
    for name, hand in record['hands'].items():
        pose = (_replay_wrist_camera(record, hand, final_labels=True)
                or _pose_from_action(hand.get('wrist_camera_graph')))
        if pose is not None:
            poses[name] = pose
    timestamp_ms = round(float(record['timestamp_s']) * 1000)
    if redetect:
        detected = tracker.process(frame, timestamp_ms, poses)
    else:
        detected = tracker.process_observations(
            frame, timestamp_ms, measurements, poses, protected_assignments=protected)
    joints_by_name = {name: joint_pose_to_dict(joint) for name, joint in detected.items()}
    camera = (_pose_from_action(record.get('camera_world_pose_fused'))
              if record.get('camera_world_source') in ('marker', 'marker+slam', 'head-slam')
              and record.get('scale_status') == 'metric' else None)
    for name, hand in result['hands'].items():
        joints = joints_by_name.get(name, {'valid': False, 'wrist_anchor_valid': False})
        # A finalized wrist veto must not erase the immutable image observation;
        # conversely a formerly bound hand must not retain stale world joints.
        if name in protected and not joints.get('valid'):
            joints = copy.deepcopy(record['hands'][name]['joints'])
        _rebind_cached_joints(joints, poses.get(name), None, calibration)
        same_world = (camera is not None and hand.get('world_submap_id') is not None
                      and hand.get('world_submap_id') == record.get('camera_submap_id'))
        for joint_key, wrist_key in (
            ('world_landmarks_graph_m', 'wrist_world_graph'),
            ('world_landmarks_tracking_m', 'wrist_world_tracking'),
            ('world_landmarks_anchor_verified_m', 'wrist_world_anchor_verified'),
        ):
            wrist = _pose_from_action(hand.get(wrist_key)) if same_world else None
            joints[joint_key] = (_world_landmarks(joints.get('camera_landmarks_m'), camera, wrist)
                                 if wrist is not None and poses.get(name) is not None else None)
        hand['joints'] = joints
    # Some trackers correctly leave protected measurements unassigned if the
    # wrist is absent. They were retained above; do not cache them a second time.
    protected_images = {json.dumps(record['hands'][name]['joints']['image_landmarks_normalized'])
                        for name in protected}
    result['unassigned_hands'] = []
    for name, joints in joints_by_name.items():
        if name in result['hands']:
            continue
        if json.dumps(joints.get('image_landmarks_normalized')) in protected_images:
            continue
        _rebind_cached_joints(joints, None, None, calibration)
        for key in ('world_landmarks_graph_m', 'world_landmarks_tracking_m',
                    'world_landmarks_anchor_verified_m'):
            joints[key] = None
        result['unassigned_hands'].append(joints)
    if frozen_fields(result) != frozen_fields(record):
        raise AssertionError('hand-only refresh changed frozen SLAM/wrist fields')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('actions', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    add_hand_backend_arguments(parser)
    parser.add_argument('--hand-model', type=Path,
                        help='MediaPipe model; default: source MP model or models/hand_landmarker.task')
    args = parser.parse_args()
    if (args.output.resolve() == args.actions.resolve() or args.output.exists()
            or args.output.with_suffix('.meta.json').exists()):
        parser.error('use a new output path; input actions must remain immutable')
    meta_path = args.actions.with_suffix('.meta.json')
    metadata = json.loads(meta_path.read_text())
    # Source pixels must still be the pixels from which these labels were made.
    inputs = metadata.get('observation_cache_contract', {}).get('input_fingerprints', {})
    current_inputs = _observation_input_fingerprints(
        metadata['video'], metadata['calibration'], metadata['bands'],
        metadata.get('world_board'), metadata['hand_model'],
        hand_joints=metadata.get('hand_joints_enabled', True))
    if inputs != current_inputs:
        parser.error('source video/calibration/layout/model changed; refusing cached hand reuse')
    calibration = load_replay_calibration(metadata)
    names = [BandLayout.load(path).name for path in metadata['bands']]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    source_backend = metadata.get('hand_backend', 'mediapipe')
    backend_provenance = None
    redetect = False
    if args.hand_backend == 'hawor':
        prediction_path = args.output.with_name(args.output.stem + '.hawor_observations.jsonl')
        predictions, backend_provenance = prepare_hawor_predictions(
            metadata['video'], metadata['calibration'], prediction_path,
            max_frames=metadata['frames'], config_path=args.hawor_config, device=args.hawor_device)
        tracker = HaworHandTracker(predictions, calibration, names, metadata['fps'])
        hand_model = Path(args.hawor_config).resolve()
        policy = hawor_policy()
        capture = None  # HaWoR has already decoded the source during its offline pass.
    else:
        hand_model = (args.hand_model or Path(metadata['hand_model'] if source_backend == 'mediapipe'
                                            else 'models/hand_landmarker.task')).resolve()
        redetect = (source_backend != 'mediapipe' or not metadata.get('hand_joints_enabled', True)
                    or hand_model != Path(metadata['hand_model']).resolve())
        tracker = HandJointTracker(hand_model, calibration, names,
                                   metadata.get('min_hand_confidence', .4),
                                   initialize_full_frame_detector=redetect)
        policy = hand_recovery_policy(min_confidence=metadata.get('min_hand_confidence', .4))
        capture = cv2.VideoCapture(metadata['video'])
    old_counts, new_counts, world_counts, anchored_counts, source_counts = [Counter() for _ in range(5)]
    unchanged = hashlib.sha256()
    source_digest = hashlib.sha256()
    count, dual, binding_started = 0, 0, time.perf_counter()
    try:
        with args.actions.open('rb') as source, args.output.open('x') as destination:
            for line in source:
                source_digest.update(line)
                row = json.loads(line)
                if row['frame'] != count:
                    raise ValueError('actions must cover every source frame, in order')
                frame = None
                if capture is not None:
                    ok, frame = capture.read()
                    if not ok:
                        raise RuntimeError(f'source video ended at frame {count}')
                updated = refresh_record(row, frame, tracker, calibration, redetect=redetect)
                unchanged.update(frozen_fields(updated))
                for name in names:
                    old_counts[name] += bool(row['hands'][name]['joints'].get('valid'))
                    joints = updated['hands'][name]['joints']
                    new_counts[name] += bool(joints.get('valid'))
                    anchored_counts[name] += joints.get('camera_landmarks_m') is not None
                    world_counts[name] += joints.get('world_landmarks_graph_m') is not None
                    if joints.get('valid'):
                        source_counts[joints.get('association_status', 'unknown')] += 1
                dual += all(updated['hands'][name]['joints'].get('world_landmarks_graph_m') is not None
                            for name in names)
                destination.write(json.dumps(updated, separators=(',', ':'), allow_nan=False) + '\n')
                count += 1
                if count % 450 == 0:
                    print(f'hand refresh {count}/{metadata["frames"]}: {dict(new_counts)}, '
                          f'{time.perf_counter()-started:.1f}s', flush=True)
    finally:
        if capture is not None:
            capture.release()
        tracker.close()
    if count != metadata['frames']:
        raise ValueError('actions frame count differs from metadata')
    provenance = dict(source_actions=str(args.actions.resolve()),
                      source_sha256=source_digest.hexdigest(),
                      frozen_non_hand_fields_sha256=unchanged.hexdigest(),
                      mode='hand-only postprocessing; native Atlas and original SLAM masks unchanged',
                      source_hand_backend=source_backend, hand_backend=args.hand_backend,
                      interpolation=False, frames=count, elapsed_s=time.perf_counter()-started,
                      model_preparation_s=binding_started-started,
                      label_binding_s=time.perf_counter()-binding_started,
                      before=dict(old_counts), after=dict(new_counts),
                      world_joint_frames=dict(world_counts), association_counts=dict(source_counts))
    metadata.update(hand_model=str(hand_model), hand_backend=args.hand_backend,
                    hand_joints_enabled=True, hand_recovery_policy=policy, hand_label_refresh=provenance,
                    hand_backend_provenance=backend_provenance,
                    joint_detection_frames=dict(new_counts),
                    wrist_anchored_joint_frames=dict(anchored_counts),
                    world_joint_frames={name: 0 for name in names},
                    graph_world_joint_frames=dict(world_counts))
    metadata['observation_cache_contract']['input_fingerprints'] = _observation_input_fingerprints(
        metadata['video'], metadata['calibration'], metadata['bands'], metadata.get('world_board'), hand_model)
    if args.hand_backend == 'hawor':
        metadata['hawor_config'] = str(hand_model)
    else:
        metadata.pop('hawor_config', None)
    if isinstance(metadata.get('joint_policy'), dict):
        metadata['joint_policy']['landmarks'] = '21 hand landmarks in wrist/thumb/index/middle/ring/pinky order'
        metadata['joint_policy']['backend'] = args.hand_backend
    # Do not point a new actions file at an HTML page claiming these new labels.
    # native_replay_source retains the immutable history for a new derivative.
    metadata['native_replay_source'] = metadata.get('native_replay_source', metadata.get('replay'))
    metadata['replay'] = None
    for value in metadata.values():
        if isinstance(value, dict):
            for key in ('debug_video', 'orb_map_video', 'orb_map_viewer'):
                if value.get(key):
                    value['original_native_' + key] = value[key]
                    value[key] = None
    if isinstance(metadata.get('quality_control'), dict):
        metadata['quality_control']['hand_refresh_requires_reassessment'] = True
    if isinstance(metadata.get('valid_metric_label_yield'), dict):
        metadata['valid_metric_label_yield'].update(
            conditional_hand_frames=dict(world_counts),
            conditional_hand_fraction={name: world_counts[name] / count for name in names},
            dual_conditional_hand_frames=dual, dual_conditional_hand_fraction=dual / count)
    provenance['roi_inference_calls'] = tracker.recovery_diagnostics.get('roi_inference_calls', 0)
    args.output.with_suffix('.meta.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(provenance, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
