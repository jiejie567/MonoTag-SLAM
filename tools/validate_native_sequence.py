#!/usr/bin/env python3
"""Replay cached detections through native ORB, optionally hide tag measurements.

No ArUco or MediaPipe re-analysis. A diagnostic experiment, not another backend.
"""
from __future__ import annotations

# Support direct execution from a source checkout.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np

from aruco_track.models import BandLayout, Calibration
from tools.export_action_labels import _pose_from_output_dict, _run_deferred_head_slam, pose_to_dict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('actions', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hide-start-s', type=float)
    parser.add_argument('--hide-length-s', type=float, default=2.5)
    args = parser.parse_args()
    metadata = json.loads(args.actions.with_suffix('.meta.json').read_text())
    records = [json.loads(line) for line in args.actions.read_text().splitlines()]
    if not all('detected_marker_corners' in r and 'marker_camera_pose_observed' in r for r in records):
        parser.error('requires v3 cached marker observation fields')
    if args.output.exists():
        parser.error('use a new validation output directory')
    args.output.mkdir(parents=True)
    fps = metadata['fps']
    poses = [_pose_from_output_dict(r['marker_camera_pose_observed']) for r in records]
    confidences = [r['marker_camera_confidence'] for r in records]
    accepted = [tuple(r['accepted_marker_ids']) for r in records]
    hidden = []
    if args.hide_start_s is not None:
        for i in range(len(records)):
            if args.hide_start_s <= i/fps < args.hide_start_s+args.hide_length_s:
                poses[i], confidences[i], accepted[i] = None, 0., ()
                hidden.append(i)
    detections = [{int(mid): np.asarray(c, float) for mid, c in r['detected_marker_corners'].items()} for r in records]
    weights = [{int(mid): q['information_weight'] for mid, q in r['marker_boundary_quality'].items()} for r in records]
    layout = BandLayout.load(metadata['world_board']) if metadata['world_board'] else None
    started = time.perf_counter()
    result = _run_deferred_head_slam(Path(metadata['video']), args.actions,
                Calibration.load(metadata['calibration']), detections, poses, confidences,
                accepted, layout, ['world_board']*len(records), args.output, 'auto', None,
                args.output/'atlas.osa', weights)
    comparison = []
    for index in hidden:
        baseline = _pose_from_output_dict(records[index]['camera_world_pose_fused'])
        current = result.frames[index].pose
        if baseline is not None and current is not None:
            comparison.append(float(np.linalg.norm(baseline.tvec-current.tvec)))
    report = {'frames': len(records), 'hidden_frames': len(hidden),
              'hidden_valid_frames': sum(result.frames[i].pose is not None for i in hidden),
              'hidden_sources': {s: sum(result.frames[i].source == s for i in hidden)
                                  for s in sorted({result.frames[i].source for i in hidden})},
              'paired_position_error_median_m': float(np.median(comparison)) if comparison else None,
              'paired_position_error_p95_m': float(np.percentile(comparison, 95)) if comparison else None,
              'elapsed_s': time.perf_counter()-started,
              'peak_python_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'native_timing': result.timing,
              'scope': 'artificial tag observation dropout; unmasked same-run method is reference, not ground truth'}
    (args.output/'report.json').write_text(json.dumps(report, indent=2))
    (args.output/'camera.jsonl').write_text('\n'.join(json.dumps({'frame': i, 'source': f.source,
        'pose': pose_to_dict(f.pose), 'map': f.map_id}) for i, f in enumerate(result.frames))+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
