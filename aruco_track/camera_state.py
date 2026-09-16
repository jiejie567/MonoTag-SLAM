from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np

from .models import Pose


@dataclass(frozen=True)
class FusedCameraFrame:
    pose: Pose | None
    source: str
    confidence: float
    slam_inliers: int
    graph_reprojection_error_px: float | None
    map_id: str | None = None
    revision: int = 0
    metric: bool = False
    initialization_source: str | None = None
    background_ready: bool = False
    metric_recovered_later: bool = False
    anchor_consistency: dict | None = None
    localization_recovery: dict | None = None



def observed_hands_for_mask(record: dict) -> dict:
    """Mask detected hands even when no left/right action identity was assigned."""
    hands = {name: hand["joints"] for name, hand in record["hands"].items()
             if hand["joints"].get("valid")}
    hands.update({f"unassigned_{i}": hand
                  for i, hand in enumerate(record.get("unassigned_hands", []))
                  if hand.get("valid")})
    return hands


def prepare_slam_frame(frame: np.ndarray, allowed_mask: np.ndarray) -> np.ndarray:
    """Same masked Gaussian pixels as np.where, without its RGB broadcast copy."""
    processed = frame.copy()
    excluded = cv2.compare(allowed_mask, 0, cv2.CMP_EQ)
    if cv2.countNonZero(excluded):
        softened = cv2.GaussianBlur(frame, (31, 31), 0)
        cv2.copyTo(softened, excluded, processed)
    return processed


def make_exclusion_mask(
    image_shape: tuple[int, ...],
    detections: dict[int, np.ndarray],
    hands: dict[str, object] | None = None,
) -> np.ndarray:
    height, width = image_shape[:2]
    excluded = np.zeros((height, width), dtype=np.uint8)

    def draw_padded_hull(points: np.ndarray, padding: int) -> None:
        hull = cv2.convexHull(
            np.rint(points).astype(np.int32).reshape(-1, 1, 2)
        )
        if len(hull) == 1:
            cv2.circle(excluded, tuple(hull[0, 0]), padding, 255, -1)
        elif len(hull) == 2:
            cv2.line(
                excluded,
                tuple(hull[0, 0]),
                tuple(hull[1, 0]),
                255,
                2 * padding + 1,
            )
        else:
            cv2.fillConvexPoly(excluded, hull, 255)
            cv2.polylines(
                excluded,
                [hull],
                True,
                255,
                2 * padding + 1,
                lineType=cv2.LINE_8,
            )

    marker_padding = max(8, int(round(0.012 * np.hypot(width, height))))
    for corners in detections.values():
        draw_padded_hull(np.asarray(corners), marker_padding)

    hand_padding = max(12, int(round(0.018 * np.hypot(width, height))))
    for hand in (hands or {}).values():
        points = getattr(hand, "image_landmarks_normalized", None)
        if points is None and isinstance(hand, dict):
            points = hand.get("image_landmarks_normalized")
        if points is None:
            continue
        points = np.asarray(points, dtype=np.float64)
        if points.shape[0] < 3:
            continue
        pixels = points[:, :2] * np.array([width, height], dtype=np.float64)
        draw_padded_hull(pixels, hand_padding)
    return cv2.bitwise_not(excluded)
