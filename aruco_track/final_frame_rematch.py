"""Revisit pixel-inconsistent final poses using new, read-only ORB matches.

No new camera poses or wrist observations are created. Native process history,
the final Atlas and static/unknown wrist motion are never optimization targets.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np

from .camera_state import observed_hands_for_mask
from .orbslam3_backend import (OrbSlamObservation, annotate_anchor_consistency,
                              pose_from_native, refine_final_frame_poses)


def suspect_windows(result, calibration, detections, accepted_ids, weights, fps):
    """Re-match a short neighborhood, not only one side of a source transition."""
    if fps <= 0 or not result.history or not result.history[-1].get('final'):
        return []
    suspects = []
    for i, frame in enumerate(result.frames):
        audit = annotate_anchor_consistency(frame, result.maps.get(frame.map_id, {}),
                 calibration, detections[i], accepted_ids[i], weights[i]).anchor_consistency
        strong = [mid for mid in accepted_ids[i]
                  if .99 <= weights[i].get(mid, 1.) <= 1.
                  and str(mid) in result.maps.get(frame.map_id, {}).get('markers', {})]
        single_marker_only = frame.source == 'marker' and len(strong) == 1
        residual_conflict = audit.get('max_rms_px') is not None and audit['max_rms_px'] > 8.
        if (frame.pose is not None and frame.metric and frame.background_ready
                and (residual_conflict or single_marker_only)):
            suspects.append(i)
    selected = set()
    for i in suspects:
        frame = result.frames[i]
        for j in range(max(0, i-round(.5*fps)), min(len(result.frames), i+round(.5*fps)+1)):
            other = result.frames[j]
            if (other.pose is not None and other.metric and other.map_id == frame.map_id
                    and other.revision == frame.revision and not other.localization_recovery):
                selected.add(j)
    windows = []
    for i in sorted(selected):
        if (not windows or i-windows[-1][-1] > .5*fps or i-windows[-1][0] > 3*fps
                or result.frames[i].map_id != result.frames[windows[-1][0]].map_id):
            windows.append([])
        windows[-1].append(i)
    return windows


def validated_observation(row, mapping, calibration, timestamp):
    """Recompute evidence geometry; adapter candidate != permission to publish."""
    try:
        if (row.get('source') != 'offline-final-frame-pose-evidence'
                or row.get('accepted') is not False or row.get('support_only') is not True
                or row.get('candidate') is not True or row.get('connected') is not True
                or row.get('map_id') != mapping['id'] or row.get('map_revision') != mapping['revision']
                or row.get('gauge') != 'final_metric_atlas' or row.get('validated_after_final_atlas') is not True
                or abs(row['timestamp_s']-timestamp) > 1e-5):
            return None
        native = np.asarray(row['evidence_pose'], float)
        if native.shape != (7,) or not np.isfinite(native).all() or abs(np.linalg.norm(native[3:])-1.) > 1e-4:
            return None
        pose = pose_from_native(native)
        features = np.asarray(row['matched_features'], float)
        n = len(features)
        if (pose is None or features.shape != (n, 4) or n < 40 or not np.isfinite(features).all()
                or n != row['matched_feature_count'] or n != row['inliers']
                or np.any(features[:, 2] != np.floor(features[:, 2]))
                or len(set(features[:, 2])) != n or len(set(map(tuple, features[:, :2]))) != n
                or not .45 <= n / row['matches'] <= 1.):
            return None
        points = {p[0]: p[1:] for p in mapping['points']}
        world = np.asarray([points[mid] for mid in features[:, 2]], float)
        r = pose.rotation_matrix.T
        t = -r @ pose.tvec
        if not np.isfinite(world).all() or np.any((r @ world.T+t)[2] <= 0):
            return None
        pixels = cv2.projectPoints(world, cv2.Rodrigues(r)[0], t,
                    calibration.camera_matrix, calibration.dist_coeffs)[0].reshape(-1, 2)
        errors = np.linalg.norm(pixels-features[:, :2], axis=1)
        width, height = calibration.image_size
        uv = features[:, :2]
        if np.any(uv < 0) or np.any(uv >= [width, height]):
            return None
        cells = len(set(map(tuple, (uv * [4/width, 3/height]).astype(int))))
        hull = cv2.contourArea(cv2.convexHull(uv.astype(np.float32))) / (width*height)
        if (cells < 5 or hull < .06 or not np.isfinite(errors).all() or np.max(errors) > 3.001
                or not np.allclose(errors, features[:, 3], atol=1e-3, rtol=0)
                or abs(np.sqrt(np.mean(errors**2))-row['rms_px']) > 1e-3):
            return None
        undistorted = cv2.undistortPoints(uv.reshape(-1, 1, 2), calibration.camera_matrix,
                      calibration.dist_coeffs, P=calibration.camera_matrix).reshape(-1, 2)
        return OrbSlamObservation(n, undistorted, np.empty((0, 2)), 2, features[:, 2].astype(np.int64))
    except (ValueError, TypeError, KeyError, ZeroDivisionError, IndexError):
        return None


def refine_inconsistent_frames(result, project_dir, video_path, atlas_path, records_path,
                               calibration, detections, accepted_ids, weights, fps, diagnostics_dir,
                               *, adapter_path=None):
    windows = suspect_windows(result, calibration, detections, accepted_ids, weights, fps)
    if not windows:
        return result
    binary = (Path(adapter_path) if adapter_path is not None else
              Path(project_dir)/'third_party/ORB_SLAM3/Examples/Monocular/relocalize_prefix_readonly').resolve()
    atlas_path = Path(atlas_path).resolve()
    if not binary.is_file() or not atlas_path.is_file():
        return result
    start = time.monotonic()
    directory = Path(diagnostics_dir)/'final_frame_rematch'
    directory.mkdir(parents=True, exist_ok=True)
    masks = {}
    with Path(records_path).open() as stream:
        for line in stream:
            record = json.loads(line)
            corners = dict(record.get('boundary_rejected_marker_corners', {}))
            corners.update(record.get('detected_marker_corners', {}))
            corners.update(record.get('marker_mask_corners', {}))
            polygons = list(corners.values())
            for hand in observed_hands_for_mask(record).values():
                p = np.asarray(hand.get('image_landmarks_normalized', []), float)
                if p.ndim == 2 and len(p) >= 3:
                    polygons.append((p[:, :2]*calibration.image_size).tolist())
            masks[record['frame']] = polygons
    history = {round(h['timestamp']*fps): h for h in result.history if not h.get('final')}
    frames = list(result.frames)
    # Native final publication can use the last frame's timestamp, whereas
    # the video duration includes that frame's exposure interval.
    effective_time = max(float(result.history[-1]['timestamp']), len(result.frames)/fps)
    report = []
    for window in windows:
        mapping = result.maps[frames[window[0]].map_id]
        if any(i not in masks or i not in history for i in window):
            continue
        centre = window[len(window)//2]
        request = dict(purpose='final-frame-pose-evidence', video=str(Path(video_path).resolve()),
            camera_matrix=calibration.camera_matrix.tolist(), dist_coeffs=calibration.dist_coeffs.ravel().tolist(),
            image_width=calibration.image_size[0], image_height=calibration.image_size[1],
            map_id=mapping['id'], map_revision=mapping['revision'], boundary_frame=centre, boundary_time_s=centre/fps,
            validation_effective_time_s=effective_time,
            queries=[dict(frame=i, timestamp_s=i/fps, original_pose_valid=1, excluded_polygons=masks[i]) for i in window],
            mask_frames=[dict(frame=i, excluded_polygons=masks[i]) for i in masks if abs(i-centre) <= 2.1*fps])
        stem = directory/str(window[0])
        manifest, output = stem.with_suffix('.request.json').resolve(), stem.with_suffix('.evidence.jsonl').resolve()
        manifest.write_text(json.dumps(request, allow_nan=False))
        detail = dict(first_frame=window[0], last_frame=window[-1], evidence_frames=0, refined_frames=0)
        try:
            with stem.with_suffix('.log').open('w') as log:
                run = subprocess.run([str(binary), str((Path(project_dir)/'third_party/ORB_SLAM3/Vocabulary/ORBvoc.txt').resolve()),
                    str(atlas_path), str(manifest), str(output)], stdout=log, stderr=subprocess.STDOUT, timeout=120)
            if run.returncode:
                detail['status'] = 'adapter_failed'
                report.append(detail)
                continue
            rows = [json.loads(s) for s in output.read_text().splitlines()]
            meta = [r for r in rows if r.get('type') == 'metadata']
            if (len(meta) != 1 or meta[0].get('schema') != 'readonly-final-frame-pose-evidence/v1'
                    or meta[0].get('atlas_modified') is not False or meta[0].get('anchor_valid') is not True
                    or meta[0].get('map_id') != mapping['id'] or meta[0].get('map_revision') != mapping['revision']):
                detail['status'] = 'anchor_rejected'
                report.append(detail)
                continue
            counts = Counter(r.get('frame') for r in rows)
            for row in rows:
                i = row.get('frame')
                if type(i) is not int or i not in window or counts[i] != 1:
                    continue
                if abs(row.get('validation_effective_time_s', -1)-effective_time) > 1e-5:
                    continue
                observation = validated_observation(row, mapping, calibration, i/fps)
                if observation is None:
                    continue
                detail['evidence_frames'] += 1
                one = replace(result, frames=[frames[i]], observations=[observation], history=[history[i], result.history[-1]])
                fit = refine_final_frame_poses(one, calibration, [detections[i]], [accepted_ids[i]], [weights[i]], rematched=True)
                if fit.timing['dense_pose_accepted']:
                    detail['refined_frames'] += 1
                    frames[i] = replace(fit.frames[0], localization_recovery=dict(
                        method='final-map-rematched-pose-refinement', accepted=True, original_tracking_valid=True,
                        available_after_timestamp_s=effective_time, inliers=observation.inliers,
                        map_id=mapping['id'], map_revision=mapping['revision']))
            detail['status'] = 'completed'
        except (OSError, ValueError, subprocess.TimeoutExpired) as error:
            detail.update(status='failed', reason=str(error))
        report.append(detail)
    elapsed = time.monotonic()-start
    (directory/'report.json').write_text(json.dumps(dict(windows=report, seconds=elapsed,
        history_modified=False, atlas_modified=False, missing_frames_filled=0), indent=2))
    count = sum(r['refined_frames'] for r in report)
    print(f'Final-map rematch: {count} existing poses refined, {elapsed:.2f}s', flush=True)
    return replace(result, frames=frames, timing=dict(result.timing, final_frame_rematch_seconds=elapsed,
                   final_frame_rematch_refined=count))
