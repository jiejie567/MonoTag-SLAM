"""Late prefix correspondences for display only; labels/history stay immutable."""
import json
from pathlib import Path
import subprocess

import numpy as np


def supplement_prefix_display(project_dir, video, source, directory, history,
                              actions, calibration, fps, accepted, *, atlas_path=None):
    from .offline_replay_features import _excluded_polygons, validate_prefix_feature_rows
    from .slam_replay import final_label_camera_frame
    if not history or not history[-1].get('final') or fps <= 0:
        return accepted
    states = {round(h['timestamp'] * fps): h for h in history if not h.get('final')}
    boundary = next((i for i in sorted(states) if states[i].get('state') == 2
        and states[i].get('pose') is not None and any(
            m.get('id') == states[i].get('active_map') and m.get('background')
            for m in states[i].get('maps', []))), None)
    if boundary is None or not 0 < boundary < len(actions):
        return accepted
    anchor = final_label_camera_frame(actions[boundary], history[-1])
    if anchor.pose is None or not anchor.metric or not anchor.background_ready:
        return accepted
    map_id = int(anchor.map_id.removeprefix('atlas_'))
    indices = [i for i in range(max(0, boundary-int(5*fps)), boundary)
               if states.get(i, {}).get('active_map') == states[boundary].get('active_map')]
    displayed = {int(i*fps/30.) for i in range(int(np.ceil(boundary*30./fps)))}
    if len(indices) < 2 or all(i in accepted for i in set(indices).intersection(displayed)):
        return accepted
    def query(i):
        return dict(frame=i, timestamp_s=i/fps, original_pose_valid=False,
                    excluded_polygons=_excluded_polygons(actions[i], calibration))
    support = [i for i in range(boundary, min(len(actions), boundary+int(2*fps)+1))
               if states.get(i, {}).get('active_map') == states[boundary].get('active_map')]
    request = dict(purpose='prefix-localization', guided_prefix_matching=1, video=str(Path(video).resolve()),
        map_id=map_id, map_revision=anchor.revision, boundary_frame=boundary,
        boundary_time_s=boundary/fps, validation_effective_time_s=len(actions)/fps,
        camera_matrix=calibration.camera_matrix.tolist(),
        dist_coeffs=calibration.dist_coeffs.reshape(-1).tolist(),
        image_width=calibration.image_size[0], image_height=calibration.image_size[1],
        queries=[query(i) for i in indices], support_queries=[query(i) for i in support],
        mask_frames=[query(i) for i in support])
    binary = (Path(project_dir)/'third_party/ORB_SLAM3/Examples/Monocular/relocalize_prefix_readonly').resolve()
    atlas = (Path(atlas_path) if atlas_path is not None else Path(source)/'atlas.osa').resolve()
    if not binary.is_file() or not atlas.is_file():
        return accepted
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    from .native_cache_identity import adapter_identity, content_identity, video_identity
    binding = dict(policy='prefix-display-only/v2', request=request,
        atlas=content_identity(atlas), video=video_identity(video),
        validators=[content_identity(__file__),content_identity(Path(__file__).with_name('offline_replay_features.py'))],
        runtime_identity=adapter_identity(binary, (Path(project_dir)/'third_party/ORB_SLAM3/Vocabulary/ORBvoc.txt').resolve()))
    report_path = directory/'prefix_display.meta.json'
    candidates = directory/'prefix_display.jsonl'
    try:
        cached = json.loads(report_path.read_text()) if report_path.exists() else {}
        if (binding['runtime_identity'] is None or cached.get('binding') != binding
                or cached.get('returncode') != 0 or not candidates.exists()):
            manifest = (directory/'prefix_display_request.json').resolve()
            candidates = (directory/'prefix_display.jsonl').resolve()
            manifest.write_text(json.dumps(request, allow_nan=False))
            run = subprocess.run([str(binary), str((Path(project_dir)/'third_party/ORB_SLAM3/Vocabulary/ORBvoc.txt').resolve()),
                str(atlas), str(manifest), str(candidates)], cwd=Path(project_dir),
                capture_output=True, text=True, timeout=180)
            (directory/'prefix_display.log').write_text(run.stdout+run.stderr)
            cached = dict(binding=binding, returncode=run.returncode)
            report_path.write_text(json.dumps(cached, allow_nan=False))
        if cached['returncode'] != 0:
            return accepted
        rows = [json.loads(l) for l in candidates.read_text().splitlines() if l.strip()]
        additional = validate_display_rows(rows, request, history[-1])
        print(f'Late prefix display: {len(set(additional)-set(accepted))} extra verified frames', flush=True)
        return {**additional, **accepted}
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
        (directory/'prefix_display_failure.json').write_text(json.dumps(dict(reason=str(error))))
        return accepted


def validate_display_rows(rows, request, final):
    """Recheck matches in the certified late pose, without adopting that pose."""
    from .offline_replay_features import validate_prefix_feature_rows
    queries = {q['frame']: q for q in request['queries']}
    by_frame = {row.get('frame'): row for row in rows if row.get('type') in ('frame', 'support')}
    display_queries = []
    for row in rows:
        if row.get('type') != 'frame' or row.get('frame') not in queries:
            continue
        try:
            if row.get('guided_matching') is True:
                before, after = by_frame.get(row.get('guide_before'), {}), by_frame.get(row.get('guide_after'), {})
                if (not before.get('candidate') or not before.get('connected')
                    or not after.get('candidate') or not after.get('connected')
                    or before.get('guided_matching') or after.get('guided_matching')
                    or not 0 < row['timestamp_s']-before['timestamp_s'] <= .25
                    or not 0 < after['timestamp_s']-row['timestamp_s'] <= .25
                    or before.get('map_id') != row.get('map_id')
                    or after.get('map_id') != row.get('map_id')
                    or before.get('map_revision') != row.get('map_revision')
                    or after.get('map_revision') != row.get('map_revision')):
                    continue
            transform = np.asarray(row['T_world_camera'], float)
            if (transform.shape != (4, 4) or not np.isfinite(transform).all()
                or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-6)
                or not np.allclose(transform[:3,:3].T@transform[:3,:3], np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(transform[:3,:3]), 1., atol=1e-5)):
                continue
            display_queries.append({**queries[row['frame']], 'T_world_camera': transform.tolist()})
        except (ValueError, TypeError, KeyError):
            continue
    display_request = {**request, 'queries': display_queries}
    validated = validate_prefix_feature_rows(rows, request, display_request, final)
    return {i: {**row, 'source': 'offline-prefix-display-correspondence',
                'display_only': True, 'labels_modified': False} for i,row in validated.items()}
