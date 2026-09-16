"""Additive wrist precision screening; never changes a pose or validity mask.

This is a conditional image-geometry estimate, not calibrated world accuracy.
"""
from __future__ import annotations

import argparse
import math

from aruco_track.wrist_geometry_quality import wrist_geometry_quality


def _positive_finite(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError('must be positive and finite')
    return value


def add_wrist_precision_arguments(parser):
    parser.add_argument('--wrist-precision-budget-mm', type=_positive_finite,
                        default=10.0, help='estimated local 1-sigma position budget; '
                        'diagnostic only, not a rejection gate (default: 10 mm)')
    parser.add_argument('--wrist-corner-sigma-px', type=_positive_finite,
                        default=0.5, help='assumed corner noise for wrist precision '
                        'screening; not empirically calibrated (default: 0.5 px)')


def wrist_precision_policy(budget_mm=10.0, corner_sigma_px=0.5):
    _positive_finite(budget_mm)
    _positive_finite(corner_sigma_px)
    return {
        'schema': 'wrist-precision-screen/v1',
        'budget_mm': float(budget_mm),
        'corner_sigma_px_assumed': float(corner_sigma_px),
        'criterion': 'local_translation_sigma_max_mm <= budget_mm',
        'calibration_status': 'assumed_noise_not_empirically_calibrated',
        'scope': 'retained wrist corners, local PnP branch, fixed intrinsics/layout; '
                 'not final world-pose accuracy or hand-joint accuracy',
        'excludes': ['camera_pose_error', 'intrinsics_error', 'layout_error',
                     'PnP_branch_ambiguity', 'systematic_corner_error'],
        'changes_pose_or_validity': False,
    }


def precision_fields_from_quality(quality, *, world_valid, budget_mm=10.0,
                                  corner_sigma_px=0.5):
    detail = wrist_precision_policy(budget_mm, corner_sigma_px)
    sigma = quality.get('joint', {}).get('translation_sigma_max_m')
    available = (quality.get('status') == 'ok' and sigma is not None
                 and math.isfinite(sigma) and sigma >= 0)
    detail['local_translation_sigma_max_mm'] = float(sigma * 1000) if available else None
    # Unknown is distinct from failing a budget. Never certify an invalid world
    # label merely because its camera-relative marker geometry looks good.
    if not world_valid:
        qualified, reason = None, 'no_valid_metric_world_wrist'
    elif not available:
        qualified, reason = None, 'geometry_unavailable'
    else:
        qualified = bool(sigma * 1000 <= budget_mm)
        reason = 'estimated_within_budget' if qualified else 'estimated_over_budget'
    detail['reason'] = reason
    return {'wrist_precision_qualified': qualified, 'wrist_precision': detail}


def wrist_precision_fields(detections, accepted_ids, layout, calibration, seed,
                           *, world_valid, budget_mm=10.0, corner_sigma_px=0.5):
    # Validate even on frames without a pose. The geometry function copies all
    # PnP guesses and never admits rejected/assist-only markers.
    wrist_precision_policy(budget_mm, corner_sigma_px)
    quality = (wrist_geometry_quality(detections, accepted_ids, layout,
                                     calibration, seed, corner_sigma_px)
               if world_valid and seed is not None else {'status': 'unavailable'})
    return precision_fields_from_quality(quality, world_valid=world_valid,
                                         budget_mm=budget_mm,
                                         corner_sigma_px=corner_sigma_px)
