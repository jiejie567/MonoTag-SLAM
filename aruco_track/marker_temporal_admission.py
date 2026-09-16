"""Offline admission of isolated, geometrically inconsistent static-tag quads.

Raw detections are never repaired, averaged or overwritten.  Each frame gets
its own free camera-to-square fit (both planar solutions), so real camera
motion is not tested against a zero-motion/pixel-displacement assumption.
Nearby independently decoded frames only validate a suspicious measurement;
they never supply replacement pixels or a pose for the rejected frame.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import cv2
import numpy as np

from .models import Calibration


POLICY_VERSION = "static-quad-temporal-v1"
_SQUARE = np.array([[-.5, .5, 0], [.5, .5, 0], [.5, -.5, 0], [-.5, -.5, 0]])


@dataclass
class MarkerAdmissionResult:
    detections: list[dict[int, np.ndarray]]
    diagnostics: list[dict[int, dict]]
    excluded_ids: list[set[int]]
    summary: dict


def _square_rms(corners: np.ndarray, calibration: Calibration) -> float:
    """Best positive-depth square fit, in source-image pixels, not world BA."""
    pixels = np.asarray(corners, dtype=np.float64)
    if pixels.shape != (4, 2) or not np.isfinite(pixels).all():
        return float("inf")
    if not cv2.isContourConvex(pixels.astype(np.float32)):
        return float("inf")
    try:
        result = cv2.solvePnPGeneric(
            _SQUARE, pixels, calibration.camera_matrix, calibration.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        candidates = []
        for rotation, translation in zip(result[1], result[2]):
            if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
                continue
            projected = cv2.projectPoints(
                _SQUARE, rotation, translation,
                calibration.camera_matrix, calibration.dist_coeffs,
            )[0].reshape(4, 2)
            error = float(np.sqrt(np.mean(np.sum((projected - pixels)**2, axis=1))))
            # Most clean quads require no iterative solve. Refine a potentially
            # suspicious IPPE fit before blaming its pixels, using either branch.
            if error > .5:
                rotation, translation = cv2.solvePnPRefineLM(
                    _SQUARE, pixels, calibration.camera_matrix,
                    calibration.dist_coeffs, rotation.copy(), translation.copy(),
                )
                projected = cv2.projectPoints(
                    _SQUARE, rotation, translation,
                    calibration.camera_matrix, calibration.dist_coeffs,
                )[0].reshape(4, 2)
                error = float(np.sqrt(np.mean(np.sum((projected - pixels)**2, axis=1))))
            depth = (_SQUARE @ cv2.Rodrigues(rotation)[0].T + translation.reshape(3))[:, 2]
            if np.isfinite(error) and np.all(depth > 1e-8):
                candidates.append(error)
        return min(candidates, default=float("inf"))
    except cv2.error:
        return float("inf")


def review_static_marker_sequence(
    detections: list[dict[int, np.ndarray]],
    timestamps_s: list[float],
    calibration: Calibration,
    static_marker_ids: set[int],
    marker_weights: list[dict[int, float]] | None = None,
    nondecoded_ids: list[set[int]] | None = None,
) -> MarkerAdmissionResult:
    """Review five source frames around a suspicious strong static marker.

    A normal first/isolated frame is immediately usable: this is NOT a blanket
    N-frame initialization delay. An outlier needs two clean neighboring
    measurements of the same ID; otherwise only a suspicious new/reappearing
    observation is deferred. Persistent mature-stream disagreement is retained
    as inconclusive, not discarded by a majority vote. Original SLAM validation
    remains responsible for such systematic problems.
    """
    started = time.perf_counter()
    n = len(detections)
    times = np.asarray(timestamps_s, dtype=float)
    if times.shape != (n,) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("marker admission requires matching, strictly increasing timestamps")
    if marker_weights is not None and len(marker_weights) != n:
        raise ValueError("marker weights must match the frame count")
    if nondecoded_ids is not None and len(nondecoded_ids) != n:
        raise ValueError("nondecoded marker IDs must match the frame count")
    weights = marker_weights if marker_weights is not None else [{} for _ in range(n)]
    nondecoded = nondecoded_ids if nondecoded_ids is not None else [set() for _ in range(n)]
    filtered = [{mid: np.array(corners, copy=True) for mid, corners in frame.items()}
                for frame in detections]
    diagnostics: list[dict[int, dict]] = [{} for _ in range(n)]
    excluded: list[set[int]] = [set() for _ in range(n)]
    errors: list[dict[int, float]] = [{} for _ in range(n)]
    last_decoded: dict[int, float] = {}
    confirmed_episode: dict[int, bool] = {}
    recent: list[set[int]] = [set() for _ in range(n)]
    tested = 0
    for index, frame in enumerate(detections):
        for marker_id in sorted(static_marker_ids.intersection(frame)):
            weight = float(weights[index].get(marker_id, 1.0))
            if marker_id in nondecoded[index] or not .99 <= weight <= 1.0:
                # Weak/flow/recovered measurements are not independent votes.
                continue
            last = last_decoded.get(marker_id)
            if last is None or times[index] - last >= .5:
                confirmed_episode[marker_id] = False
            last_decoded[marker_id] = float(times[index])
            rms = _square_rms(frame[marker_id], calibration)
            if rms <= .75:
                confirmed_episode[marker_id] = True
            if not confirmed_episode[marker_id]:
                # Repeatedly decoding the same bad boundary is not evidence.
                # A new episode stays unconfirmed until a clean real quad,
                # regardless of how many suspicious frames have accumulated.
                recent[index].add(marker_id)
            errors[index][marker_id] = rms
            diagnostics[index][marker_id] = dict(
                state="accepted", reason="square_geometry_consistent",
                square_rms_px=float(rms) if np.isfinite(rms) else None,
                evidence_frame_indices=[], evidence_timestamps_s=[], uses_future_evidence=False,
            )
            tested += 1
    for index, frame_errors in enumerate(errors):
        for marker_id, rms in frame_errors.items():
            if rms <= .75:
                continue
            diagnostic = diagnostics[index][marker_id]
            references = [j for j in range(max(0, index - 2), min(n, index + 3))
                          if j != index and abs(times[j] - times[index]) <= .15
                          and errors[j].get(marker_id, float("inf")) <= .5]
            diagnostic.update(
                evidence_frame_indices=references,
                evidence_timestamps_s=[float(times[j]) for j in references],
                uses_future_evidence=any(j > index for j in references),
            )
            baseline = (float(np.median([errors[j][marker_id] for j in references]))
                        if references else None)
            diagnostic["neighbor_median_rms_px"] = baseline
            if not np.isfinite(rms):
                diagnostic.update(state="rejected", reason="invalid_square_geometry")
            elif len(references) >= 2 and rms > 4 * max(.05, baseline):
                diagnostic.update(state="rejected", reason="temporal_geometry_outlier")
            elif marker_id in recent[index]:
                diagnostic.update(state="pending", reason="pending_geometry_confirmation")
            else:
                diagnostic.update(state="inconclusive", reason="insufficient_independent_evidence")
            if diagnostic["state"] in {"rejected", "pending"}:
                excluded[index].add(marker_id)
                filtered[index].pop(marker_id, None)
    # A raw optical-flow/board-recovery cache can descend from a quad rejected
    # above. Do not let that precomputed successor re-enter as a strong factor,
    # or restart a weak-corner streak, before an independent clean decode.
    blocked: set[int] = set()
    for index in range(n):
        for marker_id, rms in errors[index].items():
            if rms <= .75 and marker_id not in excluded[index]:
                blocked.discard(marker_id)
        blocked.update(excluded[index])
        for marker_id in blocked:
            excluded[index].add(marker_id)
            filtered[index].pop(marker_id, None)
            if marker_id in detections[index] and marker_id not in diagnostics[index]:
                diagnostics[index][marker_id] = dict(
                    state="pending", reason="unverified_recovery_from_rejected_quad",
                    square_rms_px=None, evidence_frame_indices=[],
                    evidence_timestamps_s=[], uses_future_evidence=False,
                )
            elif marker_id in diagnostics[index] and diagnostics[index][marker_id]["state"] == "inconclusive":
                diagnostics[index][marker_id].update(
                    state="pending", reason="awaiting_clean_decode_after_rejection")
    summary = dict(
        policy=POLICY_VERSION, window_source_frames=5, maximum_neighbor_dt_s=.15,
        suspicious_rms_px=.75, clean_reference_rms_px=.5, minimum_references=2,
        residual_ratio=4.0, reappearance_gap_s=.5, first_clean_frame_delayed=False,
        tested_strong_observations=tested,
        excluded_observations=sum(len(ids.intersection(frame)) for ids, frame in zip(excluded, detections)),
        excluded_frames=sum(bool(ids.intersection(frame)) for ids, frame in zip(excluded, detections)),
        blocked_tracker_frames=sum(bool(ids) for ids in excluded),
        pending_observations=sum(d["state"] == "pending" for row in diagnostics for d in row.values()),
        raw_corners_modified=False, seconds=time.perf_counter() - started,
    )
    return MarkerAdmissionResult(filtered, diagnostics, excluded, summary)
