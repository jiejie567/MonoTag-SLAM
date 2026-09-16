"""Read-only local geometric sensitivity of wrist observations.

No temporal prior, gate, or pose mutation. This conditional, local linear
covariance excludes PnP branch ambiguity, camera and layout uncertainty.
"""
from __future__ import annotations
import itertools
import cv2
import numpy as np


def _fit(points, corners, calibration, seed, corner_sigma_px):
    # OpenCV can mutate rvec/tvec passed as guesses: never pass seed views.
    ok, rvec, tvec = cv2.solvePnP(
        np.asarray(points, np.float64), np.asarray(corners, np.float64),
        calibration.camera_matrix, calibration.dist_coeffs,
        seed.rvec.copy(), seed.tvec.copy(), True, cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok or not np.all(np.isfinite(np.r_[rvec.ravel(), tvec.ravel()])):
        return {'status': 'fit_failed'}
    rotation = cv2.Rodrigues(rvec)[0]
    if np.any((np.asarray(points) @ rotation.T + tvec.ravel())[:, 2] <= 0):
        return {'status': 'nonpositive_depth'}
    image, jacobian = cv2.projectPoints(
        points, rvec, tvec, calibration.camera_matrix, calibration.dist_coeffs)
    J = jacobian[:, :6]
    _, singular, vt = np.linalg.svd(J, full_matrices=False)
    if len(singular) < 6 or singular[-1] <= singular[0] * 1e-10:
        return {'status': 'rank_deficient'}
    covariance = (vt.T * (corner_sigma_px / singular)**2) @ vt
    translation_covariance = covariance[3:, 3:]
    eigenvalues = np.linalg.eigvalsh(translation_covariance)
    residual = image.reshape(-1, 2) - np.asarray(corners).reshape(-1, 2)
    return {
        'status': 'ok',
        'translation_m': tvec.ravel().tolist(),
        'translation_covariance_m2': translation_covariance.tolist(),
        'translation_sigma_max_m': float(np.sqrt(max(0., eigenvalues[-1]))),
        'translation_sigma_rms_m': float(np.sqrt(max(0., eigenvalues.sum()))),
        'reprojection_rms_px': float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))),
    }


def wrist_geometry_quality(detections, accepted_ids, layout, calibration,
                           seed, corner_sigma_px=0.5):
    """Diagnose retained strong corners only; do not admit rejected markers."""
    if not np.isfinite(corner_sigma_px) or corner_sigma_px <= 0:
        raise ValueError('corner_sigma_px must be positive and finite')
    ids = sorted(set(accepted_ids).intersection(detections).intersection(layout.markers))
    result = {
        'schema': 'wrist-local-geometry/v1',
        'corner_sigma_px_assumed': float(corner_sigma_px),
        'scope': 'local PnP branch; fixed camera intrinsics/layout; no temporal prior',
        'status': 'unavailable', 'faces': {}, 'pairs': [],
    }
    if seed is None or not ids:
        return result
    points = np.concatenate([layout.markers[i] for i in ids]).astype(np.float64)
    corners = np.concatenate([np.asarray(detections[i]).reshape(4, 2) for i in ids])
    try:
        result['joint'] = _fit(points, corners, calibration, seed, corner_sigma_px)
        for marker_id in ids:
            uv = np.asarray(detections[marker_id], np.float64).reshape(4, 2)
            face = _fit(layout.markers[marker_id].astype(np.float64), uv,
                        calibration, seed, corner_sigma_px)
            edges = np.linalg.norm(np.roll(uv, -1, axis=0) - uv, axis=1)
            area = abs(cv2.contourArea(uv.astype(np.float32)))
            face.update(area_px2=float(area),
                        altitude_px=float(area / max(float(edges.max()), 1e-12)))
            result['faces'][str(marker_id)] = face
        for a, b in itertools.combinations(ids, 2):
            first, second = (result['faces'][str(i)] for i in (a, b))
            if first['status'] != 'ok' or second['status'] != 'ok':
                continue
            delta = np.asarray(first['translation_m']) - second['translation_m']
            cov = (np.asarray(first['translation_covariance_m2'])
                   + np.asarray(second['translation_covariance_m2']))
            result['pairs'].append({
                'ids': [a, b],
                'translation_disagreement_m': float(np.linalg.norm(delta)),
                # Diagnostic score only: fitted faces may share systematic
                # pixel/layout errors. It is not a calibrated rejection test.
                'normalized_squared_disagreement': float(delta @ np.linalg.solve(cov, delta)),
            })
        result['status'] = result['joint']['status']
    except (cv2.error, np.linalg.LinAlgError):
        result['status'] = 'geometry_failed'
    return result
