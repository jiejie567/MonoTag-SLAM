#!/usr/bin/env python3
"""Bounded synthetic + saved-real-state validation; does not run/modify SLAM."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_reanchor_metric_consistency import square, evaluate, audit


def views(scale, rng, noise_px):
    physical = square(.048) + [0, 0, 1]
    result = []
    for i, x in enumerate([-.2, -.1, 0, .1, .2]):
        camera = np.array([x, .01*i, 0])
        ray = physical-camera
        pixels = ray[:, :2]/ray[:, 2:] + rng.normal(0, noise_px/1000, (4, 2))
        center = scale*camera
        result.append({'id': i, 'center': center, 'pixels': pixels,
                       'projection': np.column_stack((np.eye(3), -center))})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run', type=Path, default=ROOT/'output/uvc90_reanchor_verified_full_20260909')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('refusing to overwrite output')
    rng = np.random.default_rng(903)
    matrix = []
    for scale in (1., .8, 1.08, 1.2, 8.):
        for noise in (0., .2, .5, 1.):
            counts = Counter()
            durations = []
            for _ in range(50):
                sample = views(scale, rng, noise)
                start = time.perf_counter()
                result = evaluate(sample, .048, np.array([1000., 1000.]))
                durations.append(1000*(time.perf_counter()-start))
                counts[result['status']] += 1
            matrix.append({'map_scale': scale, 'corner_noise_std_px': noise,
                           'trials': 50, 'outcomes': dict(counts),
                           'geometry_ms_median': float(np.median(durations)),
                           'geometry_ms_p95': float(np.percentile(durations, 95))})
    start = time.perf_counter()
    real = audit(args.run/'actions.jsonl', args.run/'actions_replay/native_history.jsonl.zst',
                 ROOT/'calib/camera_usb_1bcf_28c4_1920x1080_v2.json', 101., 108., .048, 3.)
    read_and_audit_s = time.perf_counter()-start
    bad_accepts = sum(row['outcomes'].get('consistent', 0) for row in matrix if row['map_scale'] != 1.)
    result = {'scope': 'Validation of audit/classification only. No native optimizer, production gate, or 8x repair is enabled/tested by this script.',
              'seed': 903, 'synthetic': matrix, 'synthetic_trials': 1000,
              'synthetic_wrong_scale_consistent_count': bad_accepts,
              'real_read_and_audit_seconds': read_and_audit_s, 'real': real}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('synthetic wrong-scale accepted:', bad_accepts)
    for row in matrix:
        print('synthetic', row['map_scale'], row['corner_noise_std_px'], row['outcomes'],
              'median ms', round(row['geometry_ms_median'], 3))
    print('real read + audit s', round(read_and_audit_s, 3))
    for event in real['events']:
        print('real event', event['sequence'], event['metric_verification_outcome'],
              'geometry ms', round(event['geometry_validation_ms'], 3))


if __name__ == '__main__':
    main()
