#!/usr/bin/env python3
"""Sequential native A/B runs on identical cached images and marker observations.

No video decoding, detection, changed calibration, or renderer in the comparison.
Trajectory differences measure repeatability, NOT external accuracy.
"""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aruco_track.models import Calibration
from aruco_track.orbslam3_backend import run_orbslam3_sequence, write_orbslam3_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--calib', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=2700)
    parser.add_argument('--fps', type=float, default=90.)
    parser.add_argument('--modes', nargs='+', choices=['off', 'on', 'default'], default=['off', 'on'])
    parser.add_argument('--accelerator', choices=['flow', 'descriptors', 'prefetch', 'safe'], default='flow')
    args = parser.parse_args()
    cache = args.cache.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    sequence = args.output / 'sequence'
    sequence.mkdir()
    (sequence / 'rgb').symlink_to(cache / 'rgb', target_is_directory=True)
    rows = [r for r in (cache / 'rgb.txt').read_text().splitlines()
            if r.strip() and not r.startswith('#')][:args.frames]
    if not rows:
        raise ValueError('empty frame selection')
    (sequence / 'rgb.txt').write_text('\n'.join(rows)+'\n')
    # Matching timestamps determine the prefix; later tag rows are not used.
    (sequence / 'tag_observations.txt').symlink_to(cache / 'tag_observations.txt')
    calibration = Calibration.load(args.calib)
    results = []
    for index, mode in enumerate(args.modes):
        destination = args.output / f'{index}_{mode}'
        destination.mkdir()
        settings = destination / 'settings.yaml'
        write_orbslam3_settings(settings, calibration, args.fps)
        os.environ['ORB_SLAM3_TEMPORAL_FLOW'] = str(int(mode == 'on' and args.accelerator == 'flow'))
        os.environ['ORB_SLAM3_PARALLEL_DESCRIPTORS'] = str(int(mode == 'on' and args.accelerator in ['descriptors', 'safe']))
        os.environ['ORB_SLAM3_PREFETCH_IMAGES'] = str(int(mode == 'on' and args.accelerator in ['prefetch', 'safe']))
        if mode == 'default':
            for flag in ['TEMPORAL_FLOW', 'PARALLEL_DESCRIPTORS', 'PREFETCH_IMAGES']:
                os.environ.pop('ORB_SLAM3_'+flag, None)
        for name in ['OFFLINE_LOOP_SEARCH', 'INCREMENTAL_LOOP_SEARCH', 'MARKER_SIM3_LOOP']:
            os.environ['ORB_SLAM3_'+name] = '1'
        print(f'Start {index}: {mode}, {len(rows)} frames', flush=True)
        started = time.monotonic()
        _, _, points, observations, timing = run_orbslam3_sequence(
            ROOT, sequence.resolve(), settings.resolve(), destination.resolve(),
            (sequence / 'tag_observations.txt').resolve(), compact_history=True)
        log = (destination / 'native.log').read_text()
        attempts = re.findall(r'TEMPORAL_FLOW .*?accepted=(\d+) ms=([\d.e+-]+)', log)
        result = {'mode': mode, 'accelerator': args.accelerator, 'wall_s': time.monotonic()-started,
                  'timing': timing, 'points': len(points),
                  'flow_attempts': len(attempts),
                  'flow_accepted': sum(int(a[0]) for a in attempts),
                  'flow_ms': sum(float(a[1]) for a in attempts),
                  'tracking_state_counts': {str(s): sum(o.state == s for o in observations.values())
                                            for s in sorted({o.state for o in observations.values()})}}
        results.append(result)
        (args.output / 'summary.json').write_text(json.dumps(results, indent=2))
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
