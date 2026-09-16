"""Conservative image-space tracking for appearance covers, never pose labels.

``observed`` must contain actual decoded detections, not projected marker faces.
Invalid tracks retain diagnostics but never return a last-known cover polygon.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


def _valid_image(gray):
    return (isinstance(gray, np.ndarray) and gray.ndim == 2
            and gray.dtype == np.uint8 and min(gray.shape, default=0) >= 8)


def _quad_error(quad, shape):
    try:
        quad = np.asarray(quad, np.float32)
    except (TypeError, ValueError, OverflowError):
        return "invalid_quad", None
    if quad.shape != (4, 2) or not np.isfinite(quad).all():
        return "invalid_quad", None
    if not cv2.isContourConvex(quad) or cv2.contourArea(quad, oriented=True) < 25:
        return "invalid_quad", None
    edges = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
    if np.min(edges) < 3:
        return "quad_too_small", None
    height, width = shape
    if (quad[:, 0].min() < 0 or quad[:, 0].max() > width - 1
            or quad[:, 1].min() < 0 or quad[:, 1].max() > height - 1):
        return "quad_out_of_bounds", None
    return None, quad.copy()


def validate_marker_patch(
    gray: np.ndarray, quad: np.ndarray, marker_id: int, dictionary,
) -> dict:
    """Check black border and decoded payload separately, allowing one bit error.

    Cell centres, rather than cell edges, are sampled to tolerate moderate blur.
    This verifies appearance only; it does not infer foreground occlusion/depth.
    """
    result = dict(valid=False, reason="invalid_image", contrast=0.0,
                  bit_errors=None, border_score=0.0)
    if not _valid_image(gray):
        return result
    reason, quad = _quad_error(quad, gray.shape)
    if reason:
        result["reason"] = reason
        return result
    if not isinstance(marker_id, (int, np.integer)) or not 0 <= marker_id < len(dictionary.bytesList):
        result["reason"] = "invalid_marker_id"
        return result
    grid = int(dictionary.markerSize) + 2
    cell = 8
    side = grid * cell
    target = np.array([[-.5, -.5], [side - .5, -.5],
                       [side - .5, side - .5], [-.5, side - .5]], np.float32)
    patch = cv2.warpPerspective(gray, cv2.getPerspectiveTransform(quad, target), (side, side))
    cores = patch.reshape(grid, cell, grid, cell).transpose(0, 2, 1, 3)[:, :, 2:6, 2:6]
    values = cores.mean(axis=(2, 3))
    dark, light = np.percentile(values, [10, 90])
    contrast = float(light - dark)
    threshold = (dark + light) * .5
    border = np.ones((grid, grid), bool)
    border[1:-1, 1:-1] = False
    expected = cv2.aruco.generateImageMarker(dictionary, int(marker_id), grid) > 127
    bits = values[1:-1, 1:-1] > threshold
    bit_errors = int(np.count_nonzero(bits != expected[1:-1, 1:-1]))
    border_score = float(np.mean(cores[border] < threshold))
    uncertain = int(np.count_nonzero(np.abs(values[1:-1, 1:-1] - threshold) < .12 * contrast))
    result.update(contrast=contrast, bit_errors=bit_errors, border_score=border_score,
                  uncertain_bits=uncertain)
    if contrast < 35:
        result["reason"] = "low_contrast"
    elif border_score < .8:
        result["reason"] = "border_mismatch"
    elif bit_errors > min(1, int(dictionary.maxCorrectionBits)):
        result["reason"] = "payload_mismatch"
    elif uncertain > 2:
        result["reason"] = "ambiguous_payload"
    else:
        result.update(valid=True, reason="ok")
    return result


@dataclass
class _Track:
    quad: np.ndarray | None
    observed_at: float
    reason: str = "ok"


class CoverTracker:
    """Short bounded LK continuation with explicit failures and elapsed-time age."""

    def __init__(self, dictionary, max_gap_s: float = 0.8):
        if not math.isfinite(max_gap_s) or max_gap_s < 0:
            raise ValueError("max_gap_s must be finite and nonnegative")
        self.dictionary = dictionary
        self.max_gap_s = float(max_gap_s)
        self._tracks = {}
        self._previous_gray = None
        self._previous_time = None

    def _flow(self, gray, quad, marker_id):
        options = dict(winSize=(25, 25), maxLevel=3)
        try:
            forward, good, _ = cv2.calcOpticalFlowPyrLK(
                self._previous_gray, gray, quad[:, None], None, **options)
            if forward is None or good is None or not good.all() or not np.isfinite(forward).all():
                return None, dict(reason="flow_failed")
            backward, back_good, _ = cv2.calcOpticalFlowPyrLK(
                gray, self._previous_gray, forward, None, **options)
            if backward is None or back_good is None or not back_good.all() or not np.isfinite(backward).all():
                return None, dict(reason="flow_failed")
        except cv2.error:
            return None, dict(reason="flow_failed")
        fb = float(np.max(np.linalg.norm(backward[:, 0] - quad, axis=1)))
        extra = dict(reason="flow_forward_backward", fb_error_px=fb)
        if fb > 1.5:
            return None, extra
        reason, flowed = _quad_error(forward[:, 0], gray.shape)
        if reason:
            extra["reason"] = reason
            return None, extra
        area = cv2.contourArea(quad)
        area_ratio = cv2.contourArea(flowed) / area
        edges = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
        new_edges = np.linalg.norm(np.roll(flowed, -1, axis=0) - flowed, axis=1)
        displacement = float(np.max(np.linalg.norm(flowed - quad, axis=1)))
        extra.update(area_ratio=float(area_ratio), displacement_px=displacement)
        if not .5 <= area_ratio <= 2 or np.min(new_edges / edges) < .5 or np.max(new_edges / edges) > 2:
            extra["reason"] = "flow_shape_change"
            return None, extra
        if displacement > max(24., 2 * math.sqrt(area)):
            extra["reason"] = "flow_displacement"
            return None, extra
        quality = validate_marker_patch(gray, flowed, marker_id, self.dictionary)
        extra.update(quality)
        if not quality["valid"]:
            extra["reason"] = "flow_" + quality["reason"]
            return None, extra
        extra["reason"] = "verified_flow"
        return flowed, extra

    def update(
        self, gray: np.ndarray, observed: dict[int, np.ndarray], timestamp_s: float,
    ) -> tuple[dict[int, np.ndarray], dict[int, dict]]:
        """Return current valid quads and per-ID status, including failed IDs.

        Real detections always reset age, regardless of whether they came from a
        cache or a fresh detector. A reset accepts new detections but never reuses
        old image coordinates. Rejected tracks require a real detection to rearm.
        """
        timestamp_s = float(timestamp_s)
        reset = None
        if not math.isfinite(timestamp_s):
            reset = "invalid_timestamp"
        elif not _valid_image(gray):
            reset = "invalid_image"
        elif self._previous_gray is not None and gray.shape != self._previous_gray.shape:
            reset = "image_shape_changed"
        elif self._previous_time is not None and timestamp_s <= self._previous_time:
            reset = "non_monotonic_timestamp"
        now = timestamp_s if math.isfinite(timestamp_s) else 0.0
        if reset:
            for track in self._tracks.values():
                track.quad, track.reason, track.observed_at = None, reset, now
            self._previous_gray = None
            self._previous_time = None
        if reset in ("invalid_image", "invalid_timestamp"):
            for marker_id in observed:
                self._tracks[marker_id] = _Track(None, now, reset)
            return {}, {mid: dict(source="invalid", valid=False, reason=reset, age_s=0.0)
                        for mid in self._tracks}

        quads, diagnostics = {}, {}
        for marker_id in set(self._tracks) | set(observed):
            track = self._tracks.get(marker_id)
            extra = {}
            if marker_id in observed:
                reason, quad = _quad_error(observed[marker_id], gray.shape)
                if not isinstance(marker_id, (int, np.integer)) or not 0 <= marker_id < len(self.dictionary.bytesList):
                    reason, quad = "invalid_marker_id", None
                track = _Track(quad, now, reason or "observed")
                self._tracks[marker_id] = track
                source = "observed" if quad is not None else "invalid"
            elif track.quad is None:
                source = "invalid"
            elif now - track.observed_at - self.max_gap_s > 1e-9:
                track.quad, track.reason = None, "expired"
                source = "invalid"
            elif self._previous_gray is None:
                track.quad, track.reason = None, "no_previous_image"
                source = "invalid"
            else:
                quad, extra = self._flow(gray, track.quad, marker_id)
                track.quad, track.reason = quad, extra["reason"]
                source = "flow" if quad is not None else "invalid"
            if track.quad is not None:
                quads[marker_id] = track.quad.copy()
            diagnostics[marker_id] = dict(extra, source=source, valid=track.quad is not None,
                                          reason=track.reason, age_s=max(0., now - track.observed_at))
            if reset:
                diagnostics[marker_id]["reset_reason"] = reset
        self._previous_gray = gray.copy()
        self._previous_time = now
        return quads, diagnostics
