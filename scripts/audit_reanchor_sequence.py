#!/usr/bin/env python3
"""Replay unchanged native inputs through a bounded reanchor test prefix.

No detection, label regeneration, parameter loosening, or video rendering.
The saved input cache is never modified. Output must be a new directory.
"""
import argparse
import json
import os
from pathlib import Path
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
    parser.add_argument('--fps', type=float, required=True)
    parser.add_argument('--through-s', type=float, default=112.)
    parser.add_argument('--save-atlas', action='store_true',
                        help='save a separate frozen output Atlas for same-map solver comparisons')
    args = parser.parse_args()
    cache = args.cache.resolve()
    rows = [line for line in (cache / 'rgb.txt').read_text().splitlines()
            if line.strip() and not line.startswith('#')
            and float(line.split()[0]) <= args.through_s]
    if not rows:
        parser.error('no frames in prefix')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sequence = output / 'sequence'
    sequence.mkdir()
    (sequence / 'rgb').symlink_to(cache / 'rgb', target_is_directory=True)
    (sequence / 'tag_observations.txt').symlink_to(cache / 'tag_observations.txt')
    (sequence / 'rgb.txt').write_text('\n'.join(rows) + '\n')
    settings = output / 'settings.yaml'
    atlas_path = output / 'atlas.osa' if args.save_atlas else None
    write_orbslam3_settings(settings, Calibration.load(args.calib), args.fps, save_atlas=atlas_path)
    for flag in ('OFFLINE_LOOP_SEARCH', 'INCREMENTAL_LOOP_SEARCH', 'MARKER_SIM3_LOOP',
                 'PARALLEL_DESCRIPTORS', 'PREFETCH_IMAGES'):
        os.environ['ORB_SLAM3_' + flag] = '1'
    for flag in ('TEMPORAL_FLOW', 'COMPACT_HISTORY', 'DYNAMIC_GEOMETRY'):
        os.environ['ORB_SLAM3_' + flag] = '0'
    started = time.monotonic()
    print(f'Start unchanged-cache prefix: {len(rows)} frames through {rows[-1].split()[0]} s', flush=True)
    _, _, points, observations, timing = run_orbslam3_sequence(
        ROOT, sequence, settings, output, sequence / 'tag_observations.txt', compact_history=False)
    summary = {'input_cache': str(cache), 'calibration': str(args.calib.resolve()),
               'saved_atlas': str(atlas_path) if atlas_path else None,
               'frames': len(rows), 'last_timestamp_s': float(rows[-1].split()[0]),
               'wall_s': time.monotonic() - started, 'timing': timing,
               'points': len(points),
               'state_counts': {str(state): sum(o.state == state for o in observations.values())
                                for state in sorted({o.state for o in observations.values()})},
               'scope': 'prefix through reanchor episode; finalization is at prefix end, not full-video end'}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
