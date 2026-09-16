#!/usr/bin/env python3
"""Render a native SLAM replay from cached observations, without re-analysis."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from aruco_track.models import BandLayout, Calibration
from aruco_track.orbslam3_backend import read_native_history, resolve_native_history_path
from aruco_track.slam_replay import write_slam_replay
from aruco_track.offline_replay_features import prepare_final_replay_features


def load_replay_calibration(metadata):
    """Match cached pixel observations to the actual source video, not the MP4 output."""
    capture = cv2.VideoCapture(str(metadata['video']))
    try:
        if not capture.isOpened():
            raise ValueError(f"cannot read source video: {metadata['video']}")
        size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        capture.release()
    if min(size) <= 0:
        raise ValueError(f'invalid source video size: {size}')
    cached_size = metadata.get('image_size')
    if cached_size is not None and tuple(cached_size) != size:
        raise ValueError(f'cached observation size {tuple(cached_size)} differs from source video size {size}')
    return Calibration.load(metadata['calibration']).scaled_to(size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('actions', type=Path)
    parser.add_argument('--replace-render', action='store_true',
                        help='replace generated MP4/HTML, retaining observations and Atlas')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--final-map', action='store_true',
                        help='render a separate offline final-map package, with final labels from frame zero')
    modes.add_argument('--hybrid', action='store_true',
                       help='default: final global Atlas/video/hands plus historical local mapping view')
    modes.add_argument('--process', action='store_true',
                       help='legacy all-process view, without displaying future map states')
    args = parser.parse_args()
    hybrid = not args.final_map and not args.process
    metadata = json.loads(args.actions.with_suffix('.meta.json').read_text())
    replay_source = metadata.get('native_replay_source') or metadata.get('replay')
    if not replay_source:
        parser.error('actions have no native replay package')
    source_directory = Path(replay_source).parent
    target_base = (args.actions.with_name(args.actions.stem + '_replay')
                   if metadata.get('native_replay_source') else source_directory)
    suffix = '_final' if args.final_map else '_hybrid' if hybrid else ''
    directory = target_base.with_name(target_base.name + suffix)
    if (directory / 'process.mp4').exists() and not args.replace_render:
        parser.error('render already exists; use --replace-render to regenerate it')
    try:
        calibration = load_replay_calibration(metadata)
    except ValueError as error:
        parser.error(str(error))
    with args.actions.open() as stream:
        actions = [json.loads(line) for line in stream if line.strip()]
    if not all('detected_marker_corners' in row for row in actions):
        parser.error('requires cached v3 marker observations')
    history_path = resolve_native_history_path(source_directory)
    history = read_native_history(history_path)
    if args.final_map or hybrid or directory != source_directory:
        directory.mkdir(parents=True, exist_ok=True)
        # Keep the input process package immutable and make the derivative
        # independently serveable/verifiable. No detection or SLAM is rerun.
        shutil.copy2(history_path, directory / history_path.name)
        if (source_directory / 'atlas.osa').is_file():
            shutil.copy2(source_directory / 'atlas.osa', directory / 'atlas.osa')
        # Reuse validated prefix feature measurements from an existing final
        # package; prepare_final_replay_features checks their full input binding.
        if hybrid:
            cached_final = source_directory.with_name(source_directory.name + '_final')
            for name in ('offline_feature_matches.meta.json', 'offline_feature_matches.jsonl'):
                if not (directory / name).exists() and (cached_final / name).is_file():
                    shutil.copy2(cached_final / name, directory / name)
    offline_features = (prepare_final_replay_features(
        Path(__file__).resolve().parent, Path(metadata['video']), source_directory, directory,
        history, actions, calibration, metadata['fps'],
        atlas_path=metadata.get('atlas')) if args.final_map or hybrid else {})
    detections = [{int(mid): np.asarray(corners, float)
                   for mid, corners in row['detected_marker_corners'].items()}
                  for row in actions]
    result = write_slam_replay(Path(metadata['video']), args.actions, directory, history,
                              calibration, detections,
                              [tuple(row['accepted_marker_ids']) for row in actions], metadata['fps'],
                              marker_layout=BandLayout.load(metadata['world_board']) if metadata.get('world_board') else None,
                              actions=actions, final_map=args.final_map, offline_features=offline_features,
                              hybrid=hybrid)
    if metadata.get('native_replay_source'):
        metadata['replay'] = str((directory / 'index.html').resolve())
        if isinstance(metadata.get('slam'), dict):
            metadata['slam'].update(debug_video=str((directory / 'process.mp4').resolve()),
                                    orb_map_viewer=metadata['replay'])
        args.actions.with_suffix('.meta.json').write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    print('\n'.join(str(path) for path in result))


if __name__ == '__main__':
    main()
