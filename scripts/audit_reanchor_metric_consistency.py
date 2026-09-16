#!/usr/bin/env python3
"""Audit saved reanchor revisions against measured marker pixels and known size.

Read-only with respect to the run and Atlas. This is an internal geometric
consistency check, not external ground truth or a replacement for native BA.
Uses the same retained keyframes and raw corner observations before and after
each event; never measures the already size-constrained Atlas marker vertices.
"""
from __future__ import annotations

import argparse
import bisect
import io
import json
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import zstandard


def square(size_m):
    h = size_m / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])


def fit_square(views, size_m, focal):
    """Triangulate pixels first, then fit a similarity to a physical square."""
    if len(views) < 2:
        raise ValueError('at least two views required')
    # Centre the linear solve to avoid far-from-origin conditioning artifacts.
    origin = np.asarray(views[0]['center'])
    xyz = []
    for corner in range(4):
        rows = []
        for view in views:
            p = np.array(view['projection'], copy=True)
            p[:, 3] += p[:, :3] @ origin
            x, y = view['pixels'][corner]
            rows.extend((x * p[2] - p[0], y * p[2] - p[1]))
        _, singular, vt = np.linalg.svd(rows, full_matrices=False)
        h = vt[-1]
        if singular[-2] < 1e-10 or abs(h[3]) < 1e-10:
            raise ValueError('degenerate triangulation')
        xyz.append(h[:3] / h[3] + origin)
    xyz = np.asarray(xyz)
    template = square(size_m)
    centered = xyz - xyz.mean(axis=0)
    u, values, vt = np.linalg.svd(centered.T @ template / 4)
    sign = np.diag([1., 1., np.linalg.det(u @ vt)])
    fitted_scale = np.sum(values * np.diag(sign)) / np.mean(np.sum(template**2, axis=1))
    if not np.isfinite(fitted_scale) or fitted_scale <= 0:
        raise ValueError('invalid fitted scale')
    rigid = fitted_scale * template @ (u @ sign @ vt).T + xyz.mean(axis=0)
    errors = []
    for view in views:
        p = view['projection']
        camera = rigid @ p[:, :3].T + p[:, 3]
        if np.any(camera[:, 2] <= 1e-6):
            raise ValueError('nonpositive depth')
        errors.extend((camera[:, :2] / camera[:, 2:] - view['pixels']) * focal)
    correction = 1. / fitted_scale
    return {
        'metric_per_reconstructed_unit': float(correction),
        'remaining_correction_pct': float(100 * (correction - 1)),
        'reconstructed_edge_mm': float(1000 * size_m * fitted_scale),
        'size_error_pct': float(100 * (fitted_scale - 1)),
        'square_shape_error_fraction': float(np.sqrt(np.mean(np.sum((xyz-rigid)**2, axis=1))) / (size_m*fitted_scale)),
        'rigid_square_reprojection_rms_px': float(np.sqrt(np.mean(np.sum(np.asarray(errors)**2, axis=1)))),
    }


def evaluate(views, size_m, focal, tolerance_pct=3.):
    result = {'views': len(views), 'keyframe_ids': [v['id'] for v in views]}
    if len(views) < 3:
        return dict(result, status='unverified', reason='fewer_than_three_views')
    if len(set(result['keyframe_ids'])) != len(views):
        return dict(result, status='unverified', reason='duplicate_keyframe_observations')
    try:
        result.update(fit_square(views, size_m, focal))
        # Translation alone does not ensure depth observability for a far
        # marker. Undo camera rotation before checking measured ray parallax.
        for corner in range(4):
            rays = []
            for view in views:
                ray = view['projection'][:, :3].T @ np.r_[view['pixels'][corner], 1.]
                rays.append(ray / np.linalg.norm(ray))
            if not any(0 < np.dot(a, b) < .9998 for i, a in enumerate(rays) for b in rays[i+1:]):
                return dict(result, status='unverified', reason='insufficient_corner_parallax')
        scale = result['metric_per_reconstructed_unit']
        baseline = max(np.linalg.norm(a['center']-b['center']) for a in views for b in views) * scale
        leaveout = [fit_square(subset, size_m, focal)['metric_per_reconstructed_unit']
                    for subset in (views[1:], views[:-1])]
        stability = max(abs(np.log(s / scale)) for s in leaveout)
        result.update(metric_baseline_m=float(baseline), leave_endpoint_out_log_variation=float(stability))
        if baseline < .04 or result['square_shape_error_fraction'] > .05 or result['rigid_square_reprojection_rms_px'] > 2.5 or stability > .10:
            return dict(result, status='unverified', reason='geometry_or_stability_gate')
        # A large uncertainty does NOT widen the physical-size success band.
        # Leave-out stability is a heuristic, not a calibrated confidence bound.
        if stability > np.log1p(tolerance_pct / 100):
            return dict(result, status='unverified', reason='insufficient_size_precision')
        return dict(result, status='consistent' if abs(result['size_error_pct']) <= tolerance_pct else 'residual_scale_error')
    except (ValueError, np.linalg.LinAlgError) as error:
        return dict(result, status='unverified', reason=str(error))


def classify_event(markers, expected_ids):
    """Label evidence, never equate a native commit with metric recovery."""
    expected = {str(mid) for mid in expected_ids}
    if not expected or expected != set(markers):
        return 'scale_unverified'
    after = [markers[mid]['after']['status'] for mid in expected]
    if 'residual_scale_error' in after:
        return 'residual_scale_error'
    if any(status != 'consistent' for status in after):
        return 'scale_unverified'
    before = [markers[mid]['before']['status'] for mid in expected]
    if any(status not in ('consistent', 'residual_scale_error') for status in before):
        return 'metric_consistency_verified'
    if 'residual_scale_error' in before:
        return 'scale_restored'
    return 'metric_consistency_confirmed'


def snapshot_pairs(history, events):
    targets = {round(e['timestamp'], 6) for e in events}
    pairs, previous = {}, None
    with history.open('rb') as source, zstandard.ZstdDecompressor().stream_reader(source) as stream:
        for line in io.TextIOWrapper(stream):
            # Native format starts with timestamp; parse full large snapshots
            # only for requested publications, not all 14,000 point updates.
            timestamp = round(float(line.split(':', 1)[1].split(',', 1)[0]), 6)
            if timestamp in targets and timestamp not in pairs:
                if previous is None:
                    raise ValueError('no before-publication for event')
                pairs[timestamp] = (json.loads(previous), json.loads(line))
            previous = line
    if set(pairs) != targets:
        raise ValueError('requested event missing from native history')
    return pairs


def audit(actions, history, calib, start_s, end_s, size_m, tolerance_pct):
    meta = json.loads(actions.with_suffix('.meta.json').read_text())
    events = [e for e in meta['head_slam']['marker_graph_events']
              if e['type'] == 'scale_reanchor' and e['status'] == 'accepted'
              and start_s <= e['timestamp'] <= end_s]
    if not events:
        raise ValueError('no accepted reanchor in requested window')
    pairs = snapshot_pairs(history, events)
    rows = []
    with actions.open() as source:
        for line in source:
            row = json.loads(line)
            if start_s <= row['timestamp_s'] <= end_s:
                rows.append(row)
    times = [row['timestamp_s'] for row in rows]
    camera = json.loads(calib.read_text())
    k = np.asarray(camera['camera_matrix'])
    distortion = np.asarray(camera['dist_coeffs'])
    results = []
    for event in events:
        before, after = pairs[round(event['timestamp'], 6)]
        maps = [next(m for m in s['maps'] if m['id'] == event['map_id']) for s in (before, after)]
        keyframes = [{v[0]: v for v in m['keyframes']} for m in maps]
        groups = {}
        for kid in sorted(set(keyframes[0]) & set(keyframes[1])):
            _, timestamp, _ = keyframes[0][kid]
            if not start_s <= timestamp <= event['candidate_timestamp']:
                continue
            idx = bisect.bisect_left(times, timestamp)
            candidates = [i for i in (idx-1, idx) if 0 <= i < len(rows)]
            if not candidates:
                continue
            idx = min(candidates, key=lambda i: abs(times[i]-timestamp))
            if abs(times[idx]-timestamp) > 1e-5:
                continue
            row = rows[idx]
            for mid, corners in row['detected_marker_corners'].items():
                if int(mid) not in event['marker_ids']:
                    continue
                if row['marker_boundary_quality'].get(mid, {}).get('information_weight', 0) < .99:
                    continue
                pixels = cv2.undistortPoints(np.asarray(corners, float).reshape(-1, 1, 2), k, distortion).reshape(4, 2)
                group = groups.setdefault(mid, [[], []])
                for side in range(2):
                    pose = keyframes[side][kid][2]
                    rotation = Rotation.from_quat(pose[3:]).as_matrix().T
                    center = np.asarray(pose[:3])
                    group[side].append({'id': kid, 'center': center, 'pixels': pixels,
                                        'projection': np.column_stack((rotation, -rotation @ center))})
        validation_start = time.perf_counter()
        marker_results = {mid: {name: evaluate(views, size_m, np.array([k[0, 0], k[1, 1]]), tolerance_pct)
                                for name, views in zip(('before', 'after'), group)}
                          for mid, group in sorted(groups.items())}
        results.append({
            'timestamp_s': event['timestamp'], 'sequence': event['sequence'],
            'logged_candidate': event['scale'], 'logged_sigma': event['sigma'],
            'optimizer_commit_status': event['status'],
            'metric_verification_outcome': classify_event(marker_results, event['marker_ids']),
            'geometry_validation_ms': 1000 * (time.perf_counter() - validation_start),
            'markers': marker_results,
        })
    return {'actions': str(actions.resolve()), 'history': str(history.resolve()),
            'calibration': str(calib.resolve()), 'window_s': [start_s, end_s],
            'physical_marker_edge_mm': size_m*1000, 'size_tolerance_pct': tolerance_pct,
            'scope': 'Internal local-scale consistency, not external accuracy or proof of the whole corridor scale; raw strong decoded corners at retained KFs, not necessarily identical to native BA factor admission.',
            'events': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--actions', type=Path, required=True)
    parser.add_argument('--history', type=Path, required=True)
    parser.add_argument('--calib', type=Path, required=True)
    parser.add_argument('--window-start-s', type=float, required=True)
    parser.add_argument('--window-end-s', type=float, required=True)
    parser.add_argument('--marker-size-mm', type=float, default=48.)
    parser.add_argument('--size-tolerance-pct', type=float, default=3.)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('refusing to overwrite output')
    if args.marker_size_mm <= 0 or args.size_tolerance_pct <= 0 or args.window_end_s <= args.window_start_s:
        parser.error('invalid size, tolerance or window')
    result = audit(args.actions, args.history, args.calib, args.window_start_s,
                   args.window_end_s, args.marker_size_mm / 1000, args.size_tolerance_pct)
    with args.output.open('x') as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write('\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
