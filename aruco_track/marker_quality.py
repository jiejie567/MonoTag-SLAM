"""Image evidence for a decoded marker's boundary, independent of PnP."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class MarkerBoundaryQuality:
    accepted: bool
    edge_support: tuple[float, ...]
    corner_support: tuple[float, ...]
    contrast: float
    template_error_fraction: float | None = None
    information_weight: float = 1.0
    reason: str = "ok"
    template_interior_error_fraction: float | None = None


def is_assist_only_marker(quality: MarkerBoundaryQuality) -> bool:
    """Retain identity/branch evidence without admitting a corner factor."""
    return bool(
        not quality.accepted
        and quality.reason in {"grid_mismatch", "wrist_grid_mismatch"}
        and quality.contrast >= 35.0
        and min(quality.edge_support) >= 0.9
        and min(quality.corner_support) >= 0.75
        and quality.template_interior_error_fraction is not None
        and quality.template_interior_error_fraction <= 0.02
    )


def weak_corner_information_weights(
    quality: MarkerBoundaryQuality, information_weight: float = 0.05,
) -> tuple[float, float, float, float]:
    """Return conservative per-corner weights for a decoded rejected tag.

    These weights are only candidates for an already localized metric map.
    Pose consistency and a multi-frame streak are checked by the corner
    tracker before any factor is emitted.
    """
    if (
        quality.accepted
        or quality.reason not in {"boundary", "grid_mismatch"}
        or quality.contrast < 35.0
        or quality.template_interior_error_fraction is None
        or quality.template_interior_error_fraction > 0.02
    ):
        return (0.0,) * 4
    weights = []
    for index in range(4):
        adjacent_edges = min(
            quality.edge_support[index], quality.edge_support[(index - 1) % 4]
        )
        usable = quality.corner_support[index] >= 0.75 and adjacent_edges >= 0.5
        weights.append(float(information_weight) if usable else 0.0)
    return tuple(weights)


def evaluate_marker_boundary(
    gray: np.ndarray, corners: np.ndarray, marker_template: np.ndarray | None = None,
    allow_soft_grid: bool = True,
    wrist_marker: bool = False,
) -> MarkerBoundaryQuality:
    corners = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    invalid = MarkerBoundaryQuality(
        False, (0.0,) * 4, (0.0,) * 4, 0.0,
        information_weight=0.0, reason="invalid_quad",
    )
    if not np.all(np.isfinite(corners)) or not cv2.isContourConvex(corners):
        return invalid
    minimum_edge_pixels = float(np.min(np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)))
    if minimum_edge_pixels < 10:
        return invalid
    target = np.array([[16, 16], [80, 16], [80, 80], [16, 80]], np.float32)
    patch = cv2.warpPerspective(
        gray, cv2.getPerspectiveTransform(corners, target), (96, 96),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    ).astype(np.float32)
    dark, light = np.percentile(patch[16:80, 16:80], [10, 90])
    contrast = float(light - dark)
    threshold = max(20.0, 0.15 * contrast)
    support = []
    for index in range(4):
        edge = np.rot90(patch, index)
        # Sample both sides of the outer black border. A decoded payload alone
        # does not prove that a finger/occluder left the four corners intact.
        outside = edge[12:15, 19:78].mean(axis=0)
        inside = edge[18:21, 19:78].mean(axis=0)
        support.append(outside - inside > threshold)
    edge_support = tuple(float(np.mean(edge)) for edge in support)
    corner_support = tuple(
        min(float(np.mean(support[index][:8])),
            float(np.mean(support[(index - 1) % 4][-8:])))
        for index in range(4)
    )
    accepted = (
        contrast >= 35.0
        and min(edge_support) >= 0.15
        and min(corner_support) >= 0.25
    )
    template_error = None
    interior_error = None
    reason = "ok" if accepted else "low_contrast" if contrast < 35.0 else "boundary"
    weight = 1.0 if accepted else 0.0
    if marker_template is not None:
        side = marker_template.shape[0]
        target = np.array(
            [[-0.5, -0.5], [side - 0.5, -0.5],
             [side - 0.5, side - 0.5], [-0.5, side - 0.5]], np.float32,
        )
        rectified = cv2.warpPerspective(
            gray, cv2.getPerspectiveTransform(corners, target), (side, side)
        )
        _, binary = cv2.threshold(
            rectified, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        # Templates use 20 pixels/module. Ignore one pixel on either side of
        # every cell boundary, allowing blur/subpixel noise, but not a warped
        # grid whose ID is still decodable by majority voting in each cell.
        core = (np.arange(side) % 20 >= 1) & (np.arange(side) % 20 < 19)
        mismatch = (binary > 127) != (marker_template > 127)
        template_error = float(np.mean(mismatch[np.ix_(core, core)]))
        interior = (np.arange(side) % 20 >= 4) & (np.arange(side) % 20 < 16)
        interior_error = float(np.mean(mismatch[np.ix_(interior, interior)]))
        if accepted and template_error > 0.04:
            # A small difference confined to cell edges is not evidence of a
            # missing outer corner. Keep it as a weak observation, never a new
            # map anchor. Missing borders and payload-interior errors stay hard
            # failures; this is not a blanket relaxation of the template test.
            accepted = bool(
                allow_soft_grid and template_error <= 0.06
                and interior_error <= 0.005
                and min(edge_support) >= 0.9 and min(corner_support) >= 0.9
            )
            weight = 0.25 if accepted else 0.0
            reason = "soft_grid" if accepted else "grid_mismatch"
    if wrist_marker and marker_template is not None:
        # Small, oblique band tags have only a few source pixels per module:
        # rectifying them magnifies blur and printing errors. Do not apply the
        # fixed-board subpixel grid thresholds to them. A substantially missing
        # outer edge or a corrupted payload must still invalidate the whole tag,
        # including tags supplied by short-gap optical flow.
        module_pixels = minimum_edge_pixels / (marker_template.shape[0] / 20.0)
        grid_limit = float(np.clip(0.02 + 0.8 / module_pixels, 0.07, 0.20))
        interior_limit = float(np.clip(0.005 + 0.55 / module_pixels, 0.025, 0.15))
        boundary_ok = (contrast >= 35.0
                       and min(edge_support) >= (0.5 if module_pixels >= 8.0 else 0.35)
                       and (module_pixels < 8.0 or min(corner_support) >= 0.25))
        grid_ok = template_error <= grid_limit and interior_error <= interior_limit
        accepted = bool(boundary_ok and grid_ok)
        weight = 1.0 if accepted else 0.0
        reason = "ok" if accepted else "wrist_boundary" if not boundary_ok else "wrist_grid_mismatch"
    return MarkerBoundaryQuality(
        accepted, edge_support, corner_support, contrast, template_error,
        weight, reason, interior_error,
    )
