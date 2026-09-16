#!/usr/bin/env python3
"""Summarize native prefix geometry, without inferring external accuracy."""
import io
import json
import sys

import numpy as np
import zstandard


def summarize(path):
    points = {}
    samples = []
    next_second = 0
    path_length = 0.
    previous = None
    with open(path, 'rb') as source:
        stream = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(source))
        for line in stream:
            row = json.loads(line)
            if row['timestamp'] >= 30 or row.get('final'):
                break
            for mapping in row['maps']:
                mid = mapping['id']
                if mapping.get('points_mode', 'full') == 'full':
                    points[mid] = {}
                target = points.setdefault(mid, {})
                target.update({int(p[0]): p[1:] for p in mapping['points']})
                for pid in mapping.get('deleted_points', []):
                    target.pop(pid, None)
            if row.get('pose') is None:
                previous = None
                continue
            center = np.asarray(row['pose'][:3])
            if previous is not None:
                path_length += float(np.linalg.norm(center-previous))
            previous = center
            if row['timestamp'] + 1.e-5 < next_second:
                continue
            next_second += 1
            active = points.get(row['active_map'], {})
            distances = [float(np.linalg.norm(np.asarray(active[f[2]])-center))
                         for f in row.get('matched_features', []) if f[2] in active]
            samples.append({'time_s': row['timestamp'], 'camera_m': center.tolist(),
                'tracked_points': len(distances),
                'tracked_distance_median_m': float(np.median(distances)) if distances else None,
                'tracked_below_1cm': sum(d < .01 for d in distances)})
    return {'native_history': path, 'path_length_m': path_length,
            'scope': 'first 30 s, publication geometry; not ground-truth accuracy',
            'samples': samples}


if __name__ == '__main__':
    print(json.dumps([summarize(path) for path in sys.argv[1:]], indent=2))
