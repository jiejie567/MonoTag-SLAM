"""Conservative image-only marker masking; never changes SLAM/action labels.

Paper extents are explicit polygons in marker coordinates, not a universal
scale factor. Ambiguous foreground is retained and reported as partial coverage.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


UNIT_QUAD = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)


def polygon_mask(shape, points):
    points = np.asarray(points, np.float32)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError('A finite four-corner polygon is required')
    if not cv2.isContourConvex(points) or cv2.contourArea(points) <= 0:
        raise ValueError('A nondegenerate convex polygon is required')
    mask = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(mask, np.rint(points).astype(np.int32), 255)
    return mask


def project_paper_quad(marker_quad, paper_uv):
    marker_quad = np.asarray(marker_quad, np.float32)
    paper_uv = np.asarray(paper_uv, np.float32)
    for points in (marker_quad, paper_uv):
        if (points.shape != (4, 2) or not np.isfinite(points).all()
                or not cv2.isContourConvex(points) or cv2.contourArea(points) < 1e-8):
            raise ValueError('Invalid marker or paper quadrilateral')
    transform = cv2.getPerspectiveTransform(UNIT_QUAD, marker_quad)
    homogeneous = np.column_stack((paper_uv, np.ones(4))) @ transform.T
    if (np.any(np.abs(homogeneous[:, 2]) < 1e-6)
            or np.any(np.sign(homogeneous[:, 2]) != np.sign(homogeneous[0, 2]))):
        raise ValueError('Paper projection crosses the homography horizon')
    projected = (homogeneous[:, :2] / homogeneous[:, 2:]).astype(np.float32)
    if not np.isfinite(projected).all() or not cv2.isContourConvex(projected):
        raise ValueError('Invalid projected paper polygon')
    return projected


def hand_protection(frame, record):
    """Reuse measured 2D joints; skin expansion is local, never frame-wide."""
    height, width = frame.shape[:2]
    protected = np.zeros((height, width), np.uint8)
    hands = [hand.get('joints', {}) for hand in record.get('hands', {}).values()]
    hands.extend(record.get('unassigned_hands', []))
    for hand in hands:
        points = hand.get('image_landmarks_normalized')
        if points is None:
            continue
        points = np.asarray(points, float)
        if points.ndim != 2 or points.shape[0] != 21 or points.shape[1] < 2 or not np.isfinite(points).all():
            continue
        points = points[:, :2] * [width, height]
        radius = int(np.clip(np.linalg.norm(points[5] - points[17]) * .09, 4, 15))
        palm = cv2.convexHull(points[[0, 1, 5, 9, 13, 17]].astype(np.float32))
        cv2.fillConvexPoly(protected, np.rint(palm).astype(np.int32), 255)
        for finger in [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12],
                       [13, 14, 15, 16], [17, 18, 19, 20]]:
            for a, b in zip(finger, finger[1:]):
                cv2.line(protected, tuple(np.rint(points[a]).astype(int)),
                         tuple(np.rint(points[b]).astype(int)), 255, radius * 2)
            cv2.circle(protected, tuple(np.rint(points[finger[-1]]).astype(int)), radius, 255, -1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    vicinity = cv2.dilate(protected, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    skin = cv2.inRange(hsv, (0, 45, 80), (28, 210, 255)) & vicinity
    return cv2.dilate(protected | skin, np.ones((3, 3), np.uint8))


def foreground_protection(frame, paper_mask, payload_mask, hand_mask=None):
    """Protect dark/coloured objects entering paper without assuming straightness.

    A dark component needs evidence *outside* the target paper to be treated as
    a crossing occluder. An isolated tinted black marker is not such evidence.
    If a cable connects to a printed black area, keep the ambiguous connected
    region and flag partial coverage instead of inventing an occlusion boundary.
    """
    shape = frame.shape[:2]
    if paper_mask.shape != shape or payload_mask.shape != shape:
        raise ValueError('Mask dimensions do not match image')
    paper = paper_mask != 0
    payload = payload_mask != 0
    if hand_mask is None:
        hand_mask = np.zeros(shape, np.uint8)
    if hand_mask.shape != shape:
        raise ValueError('Hand mask dimensions do not match image')
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    paper_appearance = (hsv[:, :, 1] <= 85) & (hsv[:, :, 2] >= 90)
    # Colours/shadows outside known payloads cannot be assumed to be white paper.
    nonpaper = paper & ~payload & ~paper_appearance
    neighborhood = cv2.dilate(paper_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))) != 0
    # Use strong dark evidence: dim white paper (often 110--140 here) must not
    # connect an otherwise isolated printed border to the black plastic band.
    dark = ((gray < 100) & neighborhood).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    outside_counts = np.bincount(labels[neighborhood & ~paper], minlength=count)
    inside_counts = np.bincount(labels[paper], minlength=count)
    crossing_ids = [i for i in range(1, count)
                    if outside_counts[i] >= 6 and inside_counts[i] >= 3 and stats[i, cv2.CC_STAT_AREA] >= 15]
    crossing = np.isin(labels, crossing_ids).astype(np.uint8) * 255
    crossing = cv2.dilate(crossing, np.ones((3, 3), np.uint8))
    protected = hand_mask.copy()
    protected[nonpaper] = 255
    protected |= crossing & paper_mask
    protected_payload = int(np.count_nonzero((protected != 0) & payload & paper))
    return protected, {
        'partial': bool(protected_payload),
        'protected_payload_pixels': protected_payload,
        'crossing_components': len(crossing_ids),
        'reason': 'ambiguous_foreground_retained' if protected_payload else 'no_payload_occlusion_found',
    }


def _board_geometry(board_layout, quads, page_size_m):
    ids = [mid for mid in quads if mid in board_layout]
    if len(ids) < 2:
        return None, {}, 'need_multiple_tags_for_page_extent'
    obj = np.concatenate([np.asarray(board_layout[mid], np.float32)[:, :2] for mid in ids])
    img = np.concatenate([quads[mid] for mid in ids]).astype(np.float32)
    transform, inliers = cv2.findHomography(obj, img, cv2.RANSAC, 2.5)
    if transform is None or inliers is None:
        return None, {}, 'page_homography_failed'
    inlier_tags = sum(np.count_nonzero(inliers.reshape(-1)[4*i:4*i+4]) >= 3 for i in range(len(ids)))
    projected = cv2.perspectiveTransform(obj[None], transform)[0]
    error = np.linalg.norm(projected - img, axis=1)
    if inlier_tags < 2 or np.median(error[inliers.reshape(-1) != 0]) > 2:
        return None, {}, 'page_geometry_inconsistent'
    width, height = page_size_m
    page = np.array([[-width/2, height/2], [width/2, height/2],
                     [width/2, -height/2], [-width/2, -height/2]], np.float32)
    corners = cv2.perspectiveTransform(page[None], transform)[0]
    if not np.isfinite(corners).all() or not cv2.isContourConvex(corners):
        return None, {}, 'invalid_page_projection'
    payloads = {mid: cv2.perspectiveTransform(np.asarray(points, np.float32)[None, :, :2], transform)[0]
                for mid, points in board_layout.items()}
    return corners, payloads, 'multi_tag_page_geometry'


@dataclass
class CoverMaskResult:
    mask: np.ndarray
    protected: np.ndarray
    payload_mask: np.ndarray
    diagnostics: dict


def build_mask(frame, quads, record, paper_polygons, board_layout=None,
               page_size_m=(.210, .297)):
    """Build only image appearance masks, with explicit per-region limitations."""
    shape = frame.shape[:2]
    board_layout = board_layout or {}
    paper_mask = np.zeros(shape, np.uint8)
    payload_mask = np.zeros(shape, np.uint8)
    payload_regions, regions = {}, {}
    for marker_id, quad in quads.items():
        try:
            payload = polygon_mask(shape, quad)
        except ValueError:
            regions[str(marker_id)] = {'status': 'invalid', 'reason': 'invalid_marker_quad'}
            continue
        payload_regions[marker_id] = payload
        # Detector corners are subpixel estimates; a one-pixel guard covers the
        # antialiased printed border without treating it as a foreground object.
        payload_mask |= cv2.dilate(payload, np.ones((3, 3), np.uint8))
        if marker_id in paper_polygons:
            try:
                paper = project_paper_quad(quad, paper_polygons[marker_id])
                paper_region = polygon_mask(shape, paper)
                if (np.any(np.abs(paper) > max(shape) * 2)
                        or np.count_nonzero(paper_region) > np.prod(shape) * .25
                        or cv2.contourArea(paper) > cv2.contourArea(np.asarray(quad, np.float32)) * 12):
                    raise ValueError('Unreliable paper extrapolation')
            except ValueError:
                paper_mask |= payload
                regions[str(marker_id)] = {'status': 'partial', 'reason': 'paper_projection_invalid'}
            else:
                paper_mask |= paper_region
                regions[str(marker_id)] = {'status': 'candidate', 'paper_source': 'explicit_paper_polygon'}
        else:
            # With unknown cut-paper geometry, cover only what was located, and
            # explicitly report that surrounding paper has not been removed.
            paper_mask |= payload
            regions[str(marker_id)] = {'status': 'partial', 'reason': 'paper_extent_unknown'}
    page, board_payloads, page_reason = _board_geometry(board_layout, quads, page_size_m)
    if page is not None:
        try:
            page_mask = polygon_mask(shape, page)
        except ValueError:
            page_mask = None
        if page_mask is not None and 100 < np.count_nonzero(page_mask) < np.prod(shape) * .4:
            paper_mask |= page_mask
            for marker_id, quad in board_payloads.items():
                try:
                    payload = polygon_mask(shape, quad)
                except ValueError:
                    continue
                payload_mask |= cv2.dilate(payload, np.ones((3, 3), np.uint8))
                payload_regions[marker_id] = payload
                if np.any(payload):
                    regions[str(marker_id)] = {'status': 'candidate', 'paper_source': page_reason}
        else:
            page_reason = 'page_projection_out_of_bounds'
    hand_mask = hand_protection(frame, record)
    protected, foreground = foreground_protection(frame, paper_mask, payload_mask, hand_mask)
    mask = paper_mask.copy()
    mask[protected != 0] = 0
    for mid, payload in payload_regions.items():
        region = regions.get(str(mid))
        if region is None:
            continue
        total = int(np.count_nonzero(payload))
        covered = int(np.count_nonzero((payload != 0) & (mask != 0)))
        region['payload_pixels'] = total
        region['covered_payload_pixels'] = covered
        region['protected_payload_pixels'] = int(np.count_nonzero((payload != 0) & (protected != 0)))
        if total and covered < total:
            region.update(status='partial', reason='foreground_or_uncertain_boundary')
        elif region['status'] == 'candidate':
            region['status'] = 'located_region_covered'
    partial = any(row['status'] != 'located_region_covered' for row in regions.values())
    diagnostics = {
        'partial': partial, 'regions': regions, 'foreground': foreground,
        'page_reason': page_reason,
        'coverage_status': ('no_reliable_regions' if not np.any(mask) else
                            'partial' if partial else 'located_regions_covered'),
        'scope': 'Located regions only; absence of detections does not prove absence of markers.',
    }
    return CoverMaskResult(mask, protected, payload_mask, diagnostics)


def solid_cover(frame, mask, protected, color=(128, 128, 128)):
    if frame.shape[:2] != mask.shape or mask.shape != protected.shape:
        raise ValueError('Frame/mask shape mismatch')
    if np.any((mask != 0) & (protected != 0)):
        raise ValueError('Cover mask must not overlap protected foreground')
    result = frame.copy()
    result[mask != 0] = color
    return result
