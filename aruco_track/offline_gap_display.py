"""Use verified late gap measurements for display without changing history."""
import cv2
import numpy as np


def supplement_gap_display(history, actions, calibration, accepted):
    from .slam_replay import final_label_camera_frame
    from .offline_replay_features import validate_replay_feature_rows
    if not history or not history[-1].get('final'):
        return accepted
    final = history[-1]
    points = {m['id']: {int(p[0]): p[1:] for p in m.get('points', [])} for m in final['maps']}
    output = dict(accepted)
    for action in actions:
        evidence = action.get('camera_localization_recovery') or {}
        if evidence.get('method') != 'native-orb-final-atlas-short-gap-pnp' or evidence.get('accepted') is not True:
            continue
        try:
            index = action['frame']
            if type(index) is not int or index in output or evidence['frame'] != index or evidence['original_tracking_valid'] is not False:
                continue
            camera = final_label_camera_frame(action, final)
            map_id = evidence['map_id']
            if camera.pose is None or camera.map_id != f'atlas_{map_id}' or camera.revision != evidence['map_revision']:
                continue
            effective = evidence['available_after_timestamp_s']
            if not np.isfinite(effective) or abs(effective-final['timestamp']) > 1e-4:
                continue
            row = dict(type='frame', frame=index, timestamp_s=evidence['timestamp_s'], accepted=True,
                       map_id=map_id, map_revision=camera.revision, source='offline-final-map-correspondence',
                       validation_effective_time_s=effective, matched_features=evidence['matched_features'],
                       inliers=evidence['inliers'], rms_px=evidence['rms_px'],
                       occupied_cells=evidence['occupied_cells'], hull_fraction=evidence['hull_fraction'],
                       correspondence_origin='validated-short-gap-pnp')
            request = dict(map_id=map_id, map_revision=camera.revision, validation_effective_time_s=effective,
                           image_width=calibration.image_size[0], image_height=calibration.image_size[1],
                           queries=[dict(frame=index, timestamp_s=action['timestamp_s'])])
            if index not in validate_replay_feature_rows([row], request):
                continue
            features = np.asarray(row['matched_features'], float)
            world = np.asarray([points[map_id][f[2]] for f in row['matched_features']], float)
            R = camera.pose.rotation_matrix.T
            t = -R @ camera.pose.tvec
            if not np.isfinite(world).all() or np.any((R @ world.T + t)[2] <= 0):
                continue
            projected = cv2.projectPoints(world, cv2.Rodrigues(R)[0], t,
                                          calibration.camera_matrix, calibration.dist_coeffs)[0].reshape(-1, 2)
            if np.max(np.abs(np.linalg.norm(projected-features[:, :2], axis=1)-features[:, 3])) > .01:
                continue
            output[index] = row
        except (KeyError, TypeError, ValueError, OverflowError, cv2.error):
            continue
    return output
