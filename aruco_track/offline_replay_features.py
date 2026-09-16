"""Read-only, late 2D/3D correspondences for the final-map replay prefix.

These are image measurements, not projected point decorations or new poses.
Native tracking records and final action labels are never overwritten.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np

from .camera_state import observed_hands_for_mask
from .slam_replay import final_label_camera_frame


def _excluded_polygons(record, calibration):
    corners = dict(record.get('boundary_rejected_marker_corners', {}))
    corners.update(record.get('detected_marker_corners', {}))
    polygons = list(corners.values())
    for hand in observed_hands_for_mask(record).values():
        xy = np.asarray(hand.get('image_landmarks_normalized', []), float)
        if xy.ndim == 2 and xy.shape[1] >= 2 and len(xy) >= 3:
            polygons.append((xy[:, :2] * calibration.image_size).tolist())
    return polygons


def replay_feature_request(history, actions, calibration, fps, video_path):
    if not history or not history[-1].get('final') or fps <= 0:
        return None
    states = {round(h['timestamp'] * fps): h for h in history if not h.get('final')}
    boundary = next((i for i in sorted(states) if states[i].get('state') == 2
                     and states[i].get('pose') is not None
                     and any(m.get('id') == states[i].get('active_map') and m.get('background')
                             for m in states[i].get('maps', []))), None)
    if boundary is None or boundary <= 0 or boundary >= len(actions):
        return None
    final = history[-1]
    anchor = final_label_camera_frame(actions[boundary], final)
    if anchor.pose is None or not anchor.background_ready:
        return None
    map_id = int(anchor.map_id.removeprefix('atlas_'))
    queries = []
    sampled = sorted({min(len(actions)-1, int(i * fps / 30.))
                      for i in range(int(np.ceil(boundary * 30. / fps)))})
    for i in sampled:
        state = states.get(i, {})
        # Historical LOST states can occur between marker seeding and the first
        # background map. A validated final pose, not that historical state,
        # determines eligibility for read-only late feature matching.
        if (i >= boundary or boundary-i > 5. * fps
                or state.get('active_map') != states[boundary].get('active_map')):
            continue
        selected = final_label_camera_frame(actions[i], final)
        if selected.pose is None or selected.map_id != anchor.map_id:
            continue
        transform = np.eye(4)
        transform[:3, :3] = selected.pose.rotation_matrix
        transform[:3, 3] = selected.pose.tvec.reshape(3)
        queries.append({'frame': i, 'timestamp_s': i/fps,
                        'T_world_camera': transform.tolist(),
                        'excluded_polygons': _excluded_polygons(actions[i], calibration)})
    if not queries:
        return None
    return {'purpose': 'replay-feature-correspondence', 'video': str(Path(video_path).resolve()),
            'map_id': map_id, 'map_revision': anchor.revision,
            'boundary_frame': boundary, 'boundary_time_s': boundary/fps,
            'validation_effective_time_s': len(actions)/fps,
            'camera_matrix': calibration.camera_matrix.tolist(),
            'dist_coeffs': calibration.dist_coeffs.reshape(-1).tolist(),
            'image_width': calibration.image_size[0], 'image_height': calibration.image_size[1],
            'queries': queries,
            'mask_frames': [{'frame': i, 'excluded_polygons': _excluded_polygons(actions[i], calibration)}
                            for i in range(boundary, min(len(actions), boundary + int(2*fps)+1))]}


def validate_replay_feature_rows(rows, request):
    queries = {q['frame']: q for q in request['queries']}
    frames = [r.get('frame') for r in rows if isinstance(r, dict) and r.get('type') == 'frame'
              and type(r.get('frame')) is int]
    duplicates = {f for f, count in Counter(frames).items() if count > 1}
    accepted = {}
    for row in rows:
        if not isinstance(row, dict) or row.get('type') != 'frame':
            continue
        index = row.get('frame')
        if (type(index) is not int or index not in queries or index in duplicates
                or row.get('accepted') is not True
                or row.get('source') != 'offline-final-map-correspondence'
                or row.get('map_id') != request['map_id']
                or row.get('map_revision') != request['map_revision']):
            continue
        try:
            timestamp, effective = float(row['timestamp_s']), float(row['validation_effective_time_s'])
            if not np.isfinite(timestamp) or not np.isfinite(effective):
                continue
            raw_features = row.get('matched_features', [])
            if any(not isinstance(f, list) or len(f) != 4 or type(f[2]) is not int for f in raw_features):
                continue
            features = np.asarray(raw_features, float)
            valid = (features.ndim == 2 and features.shape[1] == 4 and len(features) >= 30
                     and np.all(np.isfinite(features)))
            if not valid:
                continue
            u, v, ids, errors = features.T
            cells = set(zip((u * 4 / request['image_width']).astype(int),
                            (v * 3 / request['image_height']).astype(int)))
            hull = cv2.contourArea(cv2.convexHull(features[:, :2].astype(np.float32))) / (
                request['image_width'] * request['image_height'])
            if (len(set(ids)) != len(ids) or len(set(zip(u, v))) != len(u)
                    or np.any(ids < 0) or np.any(ids != np.floor(ids))
                    or np.any(u < 0) or np.any(u >= request['image_width'])
                    or np.any(v < 0) or np.any(v >= request['image_height'])
                    or np.any(errors < 0) or np.any(errors > 3.)
                    or len(cells) < 5 or hull < .06
                    or row.get('inliers') != len(features)
                    or not 0 <= float(row['rms_px']) <= 3.
                    or not 5 <= int(row['occupied_cells']) <= 12
                    or not .06 <= float(row['hull_fraction']) <= 1.
                    or abs(timestamp - queries[index]['timestamp_s']) > 1e-6
                    or effective < request['validation_effective_time_s']-1e-6):
                continue
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        accepted[index] = row
    return accepted


def prefix_feature_request(history, actions, request, missing):
    """Only remeasure a prefix whose final labels already record late recovery."""
    fps = len(actions) / request['validation_effective_time_s']
    states = {round(h['timestamp'] * fps): h for h in history if not h.get('final')}
    boundary = next((i for i in sorted(states) if states[i].get('pose') is not None
                     and states[i].get('state') in (2, 6)), None)
    if boundary is None or not 0 < boundary < len(actions):
        return None
    queries = []
    for i in range(max(0, boundary - int(5 * fps)), boundary):
        evidence = actions[i].get('camera_localization_recovery') or {}
        state = states.get(i, {})
        if (state.get('state') not in (0, 1) or state.get('pose') is not None
                or state.get('active_map') != states[boundary].get('active_map')
                or evidence.get('method') != 'native-orb-final-atlas-prefix-pnp'
                or evidence.get('accepted') is not True
                or evidence.get('original_tracking_valid') is not False
                or evidence.get('map_id') != request['map_id']
                or evidence.get('map_revision') != request['map_revision']):
            continue
        queries.append({'frame': i, 'timestamp_s': i / fps, 'original_pose_valid': False,
                        'excluded_polygons': _excluded_polygons(actions[i], _request_calibration(request))})
    if len(queries) < 2 or not missing.intersection(q['frame'] for q in queries):
        return None
    masks, support = [], []
    stride = max(1, int(np.ceil(fps / 30.)))
    for i in range(boundary, min(len(actions), boundary + int(2 * fps) + 1)):
        if states.get(i, {}).get('active_map') != states[boundary].get('active_map'):
            break
        query = {'frame': i, 'timestamp_s': i / fps,
                 'excluded_polygons': _excluded_polygons(actions[i], _request_calibration(request))}
        masks.append(query)
        if (i - boundary) % stride == 0:
            support.append(query)
    return {**request, 'purpose': 'prefix-localization', 'queries': queries,
            'support_queries': support, 'mask_frames': masks,
            'boundary_frame': boundary, 'boundary_time_s': boundary / fps}


def _request_calibration(request):
    from .models import Calibration
    return Calibration(np.asarray(request['camera_matrix']), np.asarray(request['dist_coeffs']),
                       (request['image_width'], request['image_height']))


def validate_prefix_feature_rows(rows, prefix_request, display_request, final):
    """Reuse certified PnP measurements, then test them in the frozen display pose."""
    metadata = [r for r in rows if isinstance(r, dict) and r.get('type') == 'metadata']
    if (len(metadata) != 1 or metadata[0].get('schema') != 'readonly-prefix-localization/v1'
            or metadata[0].get('anchor_valid') is not True
            or metadata[0].get('atlas_modified') is not False
            or metadata[0].get('map_id') != display_request['map_id']
            or metadata[0].get('map_revision') != display_request['map_revision']):
        return {}
    try:
        if (not 0 <= float(metadata[0]['anchor_translation_difference_m']) < .05
                or not 0 <= float(metadata[0]['anchor_rotation_difference_deg']) < 5.):
            return {}
    except (KeyError, TypeError, ValueError):
        return {}
    mapping = next((m for m in final.get('maps', []) if m.get('id') == display_request['map_id']
                    and m.get('revision') == display_request['map_revision']), {})
    if not mapping.get('metric') or mapping.get('points_mode') == 'delta':
        return {}
    point_lookup = {p[0]: p[1:4] for p in mapping.get('points', [])}
    queries = {q['frame']: q for q in display_request['queries']}
    eligible = {q['frame'] for q in prefix_request['queries']}
    counts = Counter(r.get('frame') for r in rows if isinstance(r, dict) and type(r.get('frame')) is int)
    if sum(r.get('type') == 'frame' and type(r.get('frame')) is int and r.get('frame') in eligible
           and r.get('accepted') is True and r.get('connected') is True
           for r in rows if isinstance(r, dict)) < 2:
        return {}
    calibration = _request_calibration(display_request)
    converted = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        index = row.get('frame')
        if (type(index) is not int or index not in queries or index not in eligible or counts[index] != 1
                or row.get('type') != 'frame' or row.get('support_only') is not False
                or row.get('source') != 'offline-prefix-relocalization'
                or any(row.get(key) is not True for key in ('accepted', 'candidate', 'connected'))
                or row.get('map_id') != display_request['map_id']
                or row.get('map_revision') != display_request['map_revision']):
            continue
        try:
            raw = row['matched_features']
            if (row['inliers'] < 30 or row['matches'] < 30 or row['inliers'] / row['matches'] < .45
                    or not .45 <= row['inlier_fraction'] <= 1. or row['occupied_cells'] < 5
                    or row['hull_fraction'] < .06 or not 0 <= row['rms_px'] <= 3.
                    or row.get('matched_feature_count') != len(raw) or len(raw) < 30
                    or any(len(p) != 4 or type(p[2]) is not int or p[2] not in point_lookup for p in raw)):
                continue
            features = np.asarray(raw, float)
            if not np.all(np.isfinite(features)) or len(set(features[:, 2])) != len(features):
                continue
            if len(set(map(tuple, features[:, :2]))) != len(features):
                continue
            xyz = np.asarray([point_lookup[p[2]] for p in raw], float)
            transform = np.asarray(queries[index]['T_world_camera'], float)
            rotation = transform[:3, :3].T
            translation = -rotation @ transform[:3, 3]
            projected = cv2.projectPoints(xyz, cv2.Rodrigues(rotation)[0], translation,
                                         calibration.camera_matrix, calibration.dist_coeffs)[0].reshape(-1, 2)
            errors = np.linalg.norm(projected - features[:, :2], axis=1)
            depth = (xyz @ rotation.T + translation)[:, 2]
            mask = np.full((display_request['image_height'], display_request['image_width']), 255, np.uint8)
            polygons = [cv2.convexHull(np.rint(p).astype(np.int32))
                        for p in queries[index]['excluded_polygons'] if len(p) >= 3]
            if polygons:
                cv2.fillPoly(mask, polygons, 0)
            mask = cv2.erode(mask, np.ones((17, 17), np.uint8))
            kept = []
            for feature, error, z in zip(features, errors, depth):
                u, v = np.rint(feature[:2]).astype(int)
                if (np.isfinite(error) and error <= 3. and z > 0 and 0 <= u < mask.shape[1]
                        and 0 <= v < mask.shape[0] and mask[v, u]):
                    kept.append([float(feature[0]), float(feature[1]), int(feature[2]), float(error)])
            if len(kept) < 30:
                continue
            values = np.asarray(kept)
            cells = set(zip((values[:, 0] * 4 / mask.shape[1]).astype(int),
                            (values[:, 1] * 3 / mask.shape[0]).astype(int)))
            hull = cv2.contourArea(cv2.convexHull(values[:, :2].astype(np.float32))) / mask.size
            converted.append({**row, 'source': 'offline-final-map-correspondence',
                'correspondence_origin': 'validated-prefix-pnp', 'prefix_pose_inliers': row['inliers'],
                'prefix_inlier_fraction': row['inlier_fraction'], 'matched_features': kept,
                'matched_feature_count': len(kept), 'inliers': len(kept),
                'rms_px': float(np.sqrt(np.mean(values[:, 3] ** 2))),
                'occupied_cells': len(cells), 'hull_fraction': hull})
        except (KeyError, TypeError, ValueError, IndexError, OverflowError, cv2.error):
            continue
    return validate_replay_feature_rows(converted, display_request)


def _run_prefix_features(project_dir, directory, binary, atlas, inputs, request,
                         history, actions, accepted):
    missing = {q['frame'] for q in request['queries']} - set(accepted)
    prefix = prefix_feature_request(history, actions, request, missing)
    if prefix is None:
        return accepted
    binding = {**inputs, 'prefix_request': prefix, 'policy': 'validated-prefix-correspondence/v1'}
    report_path, candidates = directory / 'prefix_feature_matches.meta.json', directory / 'prefix_feature_candidates.jsonl'
    cached = json.loads(report_path.read_text()) if report_path.is_file() else {}
    if (inputs.get('runtime_identity') is None or cached.get('inputs') != binding
            or cached.get('returncode') != 0 or not candidates.is_file()):
        # The native read-only adapter is launched with ``project_dir`` as its
        # working directory.  Keep every cross-process path absolute so a
        # caller using a relative output directory cannot make a valid request
        # look unreadable (or write candidates into an unexpected directory).
        manifest = (directory / 'prefix_feature_request.json').resolve()
        candidates = candidates.resolve()
        manifest.write_text(json.dumps(prefix, separators=(',', ':'), allow_nan=False))
        started = time.perf_counter()
        run = subprocess.run([str(binary.resolve()), str((project_dir / 'third_party/ORB_SLAM3/Vocabulary/ORBvoc.txt').resolve()),
                              str(Path(atlas).resolve()), str(manifest), str(candidates)], cwd=project_dir,
                             capture_output=True, text=True, timeout=180, check=False)
        (directory / 'prefix_feature_matches.log').write_text(run.stdout + run.stderr)
        cached = {'inputs': binding, 'returncode': run.returncode, 'seconds': time.perf_counter() - started}
        report_path.write_text(json.dumps(cached, indent=2, allow_nan=False))
    rows = [json.loads(line) for line in candidates.read_text().splitlines()] if cached.get('returncode') == 0 else []
    additional = validate_prefix_feature_rows(rows, prefix, request, history[-1])
    combined = {**additional, **accepted}  # Never replace successful double-ratio matches.
    print(f'Validated prefix feature reuse: {len(combined) - len(accepted)} extra display frames', flush=True)
    return combined


def _prepare_prefix_features(project_dir, directory, binary, atlas, inputs, request,
                             history, actions, accepted):
    try:
        return _run_prefix_features(project_dir, directory, binary, atlas, inputs, request,
                                    history, actions, accepted)
    except (OSError, ValueError, TypeError, KeyError, subprocess.TimeoutExpired) as error:
        try:
            (directory / 'prefix_feature_failure.json').write_text(json.dumps(
                {'status': 'optional_prefix_features_failed', 'reason': str(error)}))
        except OSError:
            pass
        print(f'Optional prefix feature reuse unavailable: {error}', flush=True)
        return accepted


def prepare_final_replay_features(project_dir, video_path, source_directory, directory,
                                  history, actions, calibration, fps, *, atlas_path=None):
    accepted = _prepare_frozen_pose_features(project_dir, video_path, source_directory,
        directory, history, actions, calibration, fps, atlas_path=atlas_path)
    from .offline_prefix_display import supplement_prefix_display
    accepted = supplement_prefix_display(project_dir, video_path, source_directory,
        directory, history, actions, calibration, fps, accepted, atlas_path=atlas_path)
    from .offline_gap_display import supplement_gap_display
    return supplement_gap_display(history, actions, calibration, accepted)


def _prepare_frozen_pose_features(project_dir, video_path, source_directory, directory,
                                  history, actions, calibration, fps, *, atlas_path=None):
    request = replay_feature_request(history, actions, calibration, fps, video_path)
    if request is None:
        return {}
    binary = project_dir / 'third_party/ORB_SLAM3/Examples/Monocular/relocalize_prefix_readonly'
    # --save-atlas can place the actual final Atlas outside the replay package.
    # An explicit path is authoritative; never silently match a different map.
    atlas = Path(atlas_path) if atlas_path is not None else source_directory / 'atlas.osa'
    if not binary.is_file() or not atlas.is_file():
        missing = [str(path) for path in (binary, atlas) if not path.is_file()]
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'offline_feature_failure.json').write_text(json.dumps(
            {'status': 'missing_readonly_adapter_or_atlas', 'missing_paths': missing,
             'policy': 'no synthetic correspondences; native history and labels unchanged'}, indent=2))
        print('Offline feature recovery unavailable: missing ' + ', '.join(missing), flush=True)
        return {}
    # Bind cache to actual final Atlas, frozen query poses/masks, video and adapter.
    from .native_cache_identity import adapter_identity, content_identity, video_identity
    vocabulary = (project_dir / 'third_party/ORB_SLAM3/Vocabulary/ORBvoc.txt').resolve()
    atlas = atlas.resolve()
    inputs = {'request': request, 'atlas': content_identity(atlas),
              'video': video_identity(video_path), 'validator': content_identity(__file__),
              'runtime_identity': adapter_identity(binary, vocabulary)}
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / 'offline_feature_matches.meta.json'
    candidates = directory / 'offline_feature_matches.jsonl'
    if report_path.is_file() and candidates.is_file():
        try:
            previous = json.loads(report_path.read_text())
        except (OSError, ValueError):
            previous = {}
        if (inputs['runtime_identity'] is not None and previous.get('inputs') == inputs
                and previous.get('status') == 'completed'):
            accepted = validate_replay_feature_rows([json.loads(line) for line in candidates.read_text().splitlines()], request)
            return _prepare_prefix_features(project_dir, directory, binary, atlas, inputs, request,
                                            history, actions, accepted)
    request_path = (directory / 'offline_feature_request.json').resolve()
    candidates = candidates.resolve()
    request_path.write_text(json.dumps(request, separators=(',', ':'), allow_nan=False))
    started = time.perf_counter()
    result = subprocess.run([str(binary.resolve()), str(vocabulary),
                             str(atlas), str(request_path), str(candidates)],
                            cwd=project_dir, capture_output=True, text=True, timeout=180, check=False)
    (directory / 'offline_feature_matches.log').write_text(result.stdout + result.stderr)
    rows = ([json.loads(line) for line in candidates.read_text().splitlines()]
            if result.returncode == 0 else [])
    accepted = validate_replay_feature_rows(rows, request)
    report = {'inputs': inputs, 'status': 'completed' if result.returncode == 0 else 'adapter_failed',
              'returncode': result.returncode, 'requested_frames': len(request['queries']),
              'accepted_frames': len(accepted), 'seconds': time.perf_counter()-started,
              'policy': 'measured descriptor matches validated in frozen final gauge; no labels or native history changed'}
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(f"Offline feature recovery: {report['status']}, {len(accepted)}/{len(request['queries'])} frames, "
          f"{report['seconds']:.2f}s", flush=True)
    return _prepare_prefix_features(project_dir, directory, binary, atlas, inputs, request,
                                    history, actions, accepted)
