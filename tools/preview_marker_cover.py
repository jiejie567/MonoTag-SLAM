#!/usr/bin/env python3
"""Render an isolated low-cost cover preview from original RGB and cached labels."""
from __future__ import annotations

# Support direct execution from a source checkout.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

import cv2
import numpy as np

from aruco_track.marker_cover import build_mask, solid_cover
from aruco_track.marker_cover_tracking import CoverTracker, validate_marker_patch


def encoder(path, width, height, fps):
    executable = shutil.which('ffmpeg')
    if executable is None:
        raise RuntimeError('ffmpeg is required for the preview MP4')
    return subprocess.Popen([
        executable, '-hide_banner', '-loglevel', 'error', '-n',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{width}x{height}',
        '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'fast',
        '-crf', '17', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path),
    ], stdin=subprocess.PIPE)


def load_records(path):
    records = {}
    with path.open() as stream:
        for line in stream:
            record = json.loads(line)
            index = int(record['frame'])
            if index in records:
                raise ValueError(f'Duplicate observation frame {index}')
            records[index] = record
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('--observations', type=Path, required=True)
    parser.add_argument('--paper-config', type=Path, required=True)
    parser.add_argument('--board', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start', type=float, default=6.)
    parser.add_argument('--duration', type=float, default=3.)
    parser.add_argument('--fps', type=float, default=30.)
    parser.add_argument('--width', type=int, default=960)
    args = parser.parse_args()
    if args.start < 0 or args.duration <= 0 or args.fps <= 0 or args.width < 64 or args.width % 2:
        parser.error('Invalid clip timing or output width')
    if args.output.exists():
        parser.error('Choose a new output directory; previous results are preserved')
    config = json.loads(args.paper_config.read_text())
    board = json.loads(args.board.read_text())
    papers = {int(mid): np.asarray(item['paper_uv'], np.float32)
              for mid, item in config['markers'].items()}
    layout = {int(item['id']): np.asarray(item['object_points_m'], np.float32)
              for item in board['markers']}
    page_size = np.asarray(board['page_size_mm'], float) / 1000
    records = load_records(args.observations)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open {args.video}')
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    source_width, source_height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if source_fps <= 0 or args.fps > source_fps:
        raise ValueError('This preview does not synthesize or duplicate source frames')
    width = args.width
    height = int(round(width * source_height / source_width / 2) * 2)
    xy_scale = np.array([width/source_width, height/source_height], np.float32)
    count = int(round(args.duration * args.fps))
    output_indices = [int(round((args.start + index/args.fps) * source_fps)) for index in range(count)]
    if len(set(output_indices)) != count:
        raise ValueError('Output sampling would duplicate source frames')
    warm_start = max(0., args.start - 2.)
    warm_indices = {int(round((warm_start + index/args.fps) * source_fps))
                    for index in range(int(round((args.start-warm_start)*args.fps)))}
    selected = {source_index: index for index, source_index in enumerate(output_indices)}
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    tracker = CoverTracker(dictionary, max_gap_s=.8)
    detector = cv2.aruco.ArucoDetector(dictionary)
    args.output.mkdir(parents=True)
    for folder in ('masks', 'protected', 'frames'):
        (args.output / folder).mkdir()
    pure = encoder(args.output / 'covered.mp4', width, height, args.fps)
    comparison = encoder(args.output / 'comparison.mp4', 1280, 448, args.fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(warm_indices | set(output_indices)))
    rows, samples, mask_ms, fill_ms = [], [], [], []
    started = time.perf_counter()
    try:
        for source_index in range(min(warm_indices | set(output_indices)), output_indices[-1]+1):
            ok, raw = cap.read()
            if not ok:
                raise RuntimeError(f'Video ended before requested frame {source_index}')
            if source_index not in warm_indices and source_index not in selected:
                continue
            record = records.get(source_index)
            if record is None:
                raise ValueError(f'Missing original observations for frame {source_index}')
            timestamp = float(record['timestamp_s'])
            if abs(timestamp - source_index/source_fps) > 1/source_fps:
                raise ValueError('Cached observations do not match source-video timing')
            frame = cv2.resize(raw, (width, height), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flowed_ids = {int(mid) for mid in record.get('optical_flow_marker_ids', [])}
            observed = {int(mid): np.asarray(quad, np.float32) * xy_scale
                        for mid, quad in record.get('detected_marker_corners', {}).items()
                        if int(mid) not in flowed_ids and (0 <= int(mid) <= 11 or 20 <= int(mid) <= 27)}
            tick = time.perf_counter()
            # The reduced *original* image can decode small tags that the cached
            # full-resolution detector missed. Never detect on edited RGB to
            # create a new observation or a pose label.
            fresh_corners, fresh_ids, _ = detector.detectMarkers(gray)
            fresh_added, fresh_rejected = [], {}
            for corner, mid in zip(fresh_corners, [] if fresh_ids is None else fresh_ids.flatten()):
                mid = int(mid)
                if mid in observed or not (0 <= mid <= 11 or 20 <= mid <= 27):
                    continue
                quad = corner.reshape(4, 2).astype(np.float32)
                quality = validate_marker_patch(gray, quad, mid, dictionary)
                if quality['valid']:
                    observed[mid] = quad
                    fresh_added.append(mid)
                else:
                    fresh_rejected[str(mid)] = quality
            quads, tracking = tracker.update(gray, observed, timestamp)
            tracking_elapsed = time.perf_counter() - tick
            if source_index not in selected:
                continue
            index = selected[source_index]
            tick = time.perf_counter()
            result = build_mask(frame, quads, record, papers, layout, tuple(page_size))
            mask_ms.append((tracking_elapsed + time.perf_counter() - tick) * 1000)
            tick = time.perf_counter()
            covered = solid_cover(frame, result.mask, result.protected)
            fill_ms.append((time.perf_counter() - tick) * 1000)
            assert np.array_equal(covered[result.mask == 0], frame[result.mask == 0])
            assert np.array_equal(covered[result.protected != 0], frame[result.protected != 0])
            _, remaining, _ = detector.detectMarkers(cv2.cvtColor(covered, cv2.COLOR_BGR2GRAY))
            remaining_ids = [] if remaining is None else sorted(int(mid) for mid in remaining.flatten())
            failures = {str(mid): item for mid, item in tracking.items() if not item['valid']}
            rejected = [int(mid) for mid in record.get('boundary_rejected_marker_corners', {})]
            partial = result.diagnostics['partial'] or bool(failures) or bool(remaining_ids) or bool(rejected)
            status = 'partial' if partial else result.diagnostics['coverage_status']
            row = dict(
                output_frame=index, source_frame=source_index, source_time_s=timestamp,
                output_time_s=index/args.fps, source_sampling_error_s=timestamp-(args.start+index/args.fps),
                coverage_status=status, validated_for_training=False,
                residual_detected_ids=remaining_ids, boundary_rejected_ids=rejected,
                supplemental_original_detections=fresh_added,
                supplemental_rejected_detections=fresh_rejected,
                tracking={str(mid): item for mid, item in tracking.items()},
                mask=result.diagnostics, mask_pixels=int(np.count_nonzero(result.mask)),
                unmasked_changed_pixels_before_encoding=0, protected_changed_pixels_before_encoding=0,
            )
            rows.append(row)
            name = f'{index:05d}.png'
            cv2.imwrite(str(args.output / 'masks' / name), result.mask)
            cv2.imwrite(str(args.output / 'protected' / name), result.protected)
            cv2.imwrite(str(args.output / 'frames' / name), covered)
            pure.stdin.write(covered.tobytes())
            canvas = np.full((448, 1280, 3), (28, 24, 22), np.uint8)
            canvas[54:414, :640] = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
            canvas[54:414, 640:] = cv2.resize(covered, (640, 360), interpolation=cv2.INTER_AREA)
            cv2.putText(canvas, 'ORIGINAL', (20, 36), cv2.FONT_HERSHEY_SIMPLEX, .8, (245, 245, 245), 2, cv2.LINE_AA)
            cv2.putText(canvas, 'FIXED COVER / FOREGROUND GUARD', (660, 36), cv2.FONT_HERSHEY_SIMPLEX, .65, (180, 235, 190), 2, cv2.LINE_AA)
            cv2.putText(canvas, f'Source {timestamp:.3f}s | {args.fps:g} fps | {status.upper()} | uncertainties recorded; source & labels unchanged',
                        (20, 437), cv2.FONT_HERSHEY_SIMPLEX, .48, (180, 205, 245) if partial else (220, 220, 220), 1, cv2.LINE_AA)
            comparison.stdin.write(canvas.tobytes())
            if index in {0, count//3, 2*count//3, count-1}:
                samples.append(cv2.resize(canvas, (960, 336)))
    finally:
        cap.release()
        pure.stdin.close()
        comparison.stdin.close()
        exits = [pure.wait(), comparison.wait()]
    if any(code != 0 for code in exits) or len(rows) != count:
        raise RuntimeError('Incomplete preview; do not use its partial video output')
    elapsed = time.perf_counter() - started
    cv2.imwrite(str(args.output / 'comparison_contact.jpg'), np.vstack(samples))
    report = dict(
        source=str(args.video.resolve()), observations=str(args.observations.resolve()),
        paper_configuration=str(args.paper_config.resolve()),
        paper_geometry_source='manual image outlines for this clip, not millimetre ground truth',
        method='constant cover with explicit paper polygons and conservative foreground guard',
        source_fps=source_fps, fps=args.fps, frames=count, duration_s=count/args.fps,
        size=[width, height], source_interval_s=[args.start, args.start+args.duration],
        neural_inference=False, labels_modified=False, validated_for_training=False,
        partial_frames=sum(row['coverage_status']=='partial' for row in rows),
        frames_with_residual_detections=sum(bool(row['residual_detected_ids']) for row in rows),
        frames_with_supplemental_detections=sum(bool(row['supplemental_original_detections']) for row in rows),
        mask_tracking_ms_median=float(np.median(mask_ms)), mask_tracking_ms_p95=float(np.percentile(mask_ms,95)),
        fill_ms_median=float(np.median(fill_ms)), render_wall_s=elapsed,
        timing_scope='Mask/flow includes supplemental ArUco decoding on reduced original RGB, using cached original observations too. Wall time includes warmup, video and PNG I/O, two encodes, diagnostics. Original hand/SLAM/full-resolution analysis is excluded.',
        unmasked_changed_pixels_before_encoding=0, protected_changed_pixels_before_encoding=0,
        warning='Partial frames retain ambiguous foreground/undetected markers. Residual decoding is only a lower bound on remaining markers. H.264 is lossy.',
    )
    (args.output / 'report.json').write_text(json.dumps(report, indent=2))
    (args.output / 'diagnostics.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
