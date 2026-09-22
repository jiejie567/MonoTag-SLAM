#!/usr/bin/env python3
"""Verify a saved replay against every native frame publication (no SLAM rerun)."""
from __future__ import annotations

# Support direct execution from a source checkout.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import gzip
import json
from pathlib import Path
import struct
import time

import numpy as np

from aruco_track.orbslam3_backend import read_native_history, resolve_native_history_path


def _apply_native_points(points, known_maps, snapshot):
    map_ids = {mapping['id'] for mapping in snapshot['maps']}
    for key in [key for key in points if key[0] not in map_ids]:
        del points[key]
    known_maps.intersection_update(map_ids)
    if any('points_mode' in mapping for mapping in snapshot['maps']):
        for mapping in snapshot['maps']:
            map_id = mapping['id']
            if mapping['points_mode'] == 'full':
                for key in [key for key in points if key[0] == map_id]:
                    del points[key]
                known_maps.add(map_id)
            else:
                assert mapping['points_mode'] == 'delta' and map_id in known_maps
            for point in mapping['points']:
                points[(map_id, point[0])] = tuple(np.asarray(point[1:], dtype=np.float32))
            for point_id in mapping.get('deleted_points', []):
                points.pop((map_id, point_id), None)
            assert sum(key[0] == map_id for key in points) == mapping['point_count']
    else:
        points.clear()
        points.update({(mapping['id'], point[0]): tuple(np.asarray(point[1:], dtype=np.float32))
                       for mapping in snapshot['maps'] for point in mapping['points']})
        known_maps.clear()
        known_maps.update(map_ids)


def verify(directory: Path) -> dict:
    manifest = json.loads((directory/'manifest.json').read_text())
    history_path = resolve_native_history_path(directory)
    history = read_native_history(history_path)
    with gzip.open(directory/'timeline.json.gz', 'rt') as stream:
        timeline = json.load(stream)
    with gzip.open(directory/'video_frames.json.gz', 'rt') as stream:
        frames = json.load(stream)
    binary = gzip.decompress((directory/'points.bin.gz').read_bytes())
    assert len(history) == len(timeline), 'missing native publications'
    assert len(frames) == manifest['frames'], 'video frame count mismatch'
    points, native_points, native_maps = {}, {}, set()
    seek_targets = [len(timeline)-1, 0, len(timeline)//2, 1, len(timeline)-2]
    expected_states = {}
    created, updated, deleted, replay_ms = 0, 0, 0, []
    for sequence, (row, native) in enumerate(zip(timeline, history)):
        before = time.perf_counter()
        assert row['sequence'] == sequence
        if row['checkpoint']:
            points.clear()
        for i in range(row['count']):
            mid, pid, x, y, z = struct.unpack_from('<QQfff', binary, row['offset'] + i*28)
            if (mid, pid) in points:
                updated += 1
            else:
                created += 1
            points[(mid, pid)] = (x, y, z)
        for mid, pid in row['deleted']:
            del points[(mid, pid)]
            deleted += 1
        replay_ms.append((time.perf_counter()-before)*1000)
        _apply_native_points(native_points, native_maps, native)
        assert points == native_points, f'point state differs at publication {sequence}'
        assert [(m['id'], m['point_count']) for m in row['maps']] == [
            (m['id'], m.get('point_count', len(m['points']))) for m in native['maps']]
        if sequence in seek_targets:
            expected_states[sequence] = native_points.copy()
    source_fps = manifest['source_fps']
    for index, frame in enumerate(frames):
        row = timeline[frame['sequence']]
        if manifest.get('replay_mode') in ('final-map', 'hybrid'):
            observation = timeline[frame['observation_sequence']]
            hybrid = manifest['replay_mode'] == 'hybrid'
            offline_sequence = frame['global_sequence'] if hybrid else frame['sequence']
            offline_row = timeline[offline_sequence]
            assert offline_sequence == len(timeline)-1 and offline_row['final']
            assert not observation.get('final')
            assert frame['source_frame'] == round(observation['timestamp']*source_fps)
            if not frame['tail']:
                assert frame['source_frame'] == int(index*source_fps/manifest['fps'])
            if hybrid:
                assert frame['sequence'] == (offline_sequence if frame['tail']
                                              else frame['observation_sequence'])
                assert bool(row.get('final')) == frame['tail']
                assert 'process_hands' not in frame, 'final hands must not leak into process coordinates'
                if frame['process_camera'] is not None:
                    process_mapping = next(m for m in row['maps']
                                           if f"atlas_{m['id']}" == frame['process_map_id'])
                    assert frame['process_map_revision'] == process_mapping['revision']
                    if frame['process_metric']:
                        assert process_mapping['metric']
            else:
                assert not frame['tail']
            if frame['camera'] is not None:
                mapping = next(m for m in offline_row['maps'] if f"atlas_{m['id']}" == frame['map_id'])
                assert frame['metric'] and mapping['metric']
                assert frame['map_revision'] == mapping['revision']
            else:
                assert not frame['hands'] and not any(frame['trails'].values())
            continue
        assert frame['source_frame'] == round(row['timestamp']*source_fps)
        if not frame['tail']:
            assert frame['source_frame'] == int(index*source_fps/manifest['fps'])
            assert not row['final'], 'future final optimization leaked into video'
        else:
            assert row['final'], 'unlabelled final optimization'
    # Random-access reconstruction must start from a checkpoint and reach the
    # same state as sequential playback, including backward seeks.
    seek_ms = []
    for target in seek_targets:
        before = time.perf_counter()
        start = target
        while start and not timeline[start]['checkpoint']:
            start -= 1
        current = {}
        for row in timeline[start:target+1]:
            if row['checkpoint']:
                current.clear()
            for i in range(row['count']):
                mid, pid, *point = struct.unpack_from('<QQfff', binary, row['offset']+i*28)
                current[(mid, pid)] = tuple(point)
            for mid, pid in row['deleted']:
                del current[(mid, pid)]
        seek_ms.append((time.perf_counter()-before)*1000)
        assert current == expected_states[target], f'random seek mismatch at {target}'
    seconds = manifest['analysis_frames']/source_fps
    payloads = ['points.bin.gz', 'timeline.json.gz', 'video_frames.json.gz', 'events.jsonl']
    payload_bytes = sum((directory/name).stat().st_size for name in payloads)
    history_bytes = history_path.stat().st_size
    return {'verified_publications': len(history), 'verified_video_frames': len(frames),
            'source_frames': manifest['analysis_frames'], 'source_seconds': seconds,
            'sequential_apply_ms_p95': float(np.percentile(replay_ms, 95)),
            'python_random_seek_ms_max': max(seek_ms),
            'compressed_replay_bytes': payload_bytes,
            'compressed_replay_MB_per_minute': payload_bytes/1e6*60/seconds,
            'native_history_bytes': history_bytes,
            'native_history_MB_per_minute': history_bytes/1e6*60/seconds,
            'final_map_counts': {m['id']: {'points': m.get('point_count', len(m['points'])),
                                         'keyframes': len(m['keyframes']),
                                         'markers': sorted(m['markers'])} for m in history[-1]['maps']},
            'verified_scope': ('unaltered native journal; final global/video/hands with separate historical local references'
                               if manifest.get('replay_mode') == 'hybrid' else
                               'unaltered native journal plus explicitly final-map video frame references'
                               if manifest.get('replay_mode') == 'final-map' else
                               'all saved frame publications, not intra-frame mutation lifecycles'),
            'result': 'passed'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('replay', type=Path)
    args = parser.parse_args()
    report = verify(args.replay)
    (args.replay/'verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
