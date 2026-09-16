from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CaptureQuality:
    blur_score: float
    mean_luma: float
    dark_fraction: float
    bright_fraction: float
    marker_count: int
    median_marker_side_px: float | None
    smallest_marker_side_px: float | None
    warnings: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return not self.warnings

    def hud(self) -> str:
        marker = (
            f"TAGS {self.marker_count} MED {self.median_marker_side_px:.0f}px"
            if self.median_marker_side_px is not None
            else "NO TAG"
        )
        status = "QC OK" if self.ready else "QC " + "/".join(self.warnings)
        return f"{status}   SHARP {self.blur_score:.0f}   {marker}"

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["ready"] = self.ready
        return payload


def evaluate_capture_quality(
    frame: np.ndarray,
    detections: dict[int, np.ndarray] | None = None,
    *,
    min_marker_side_px: float = 40.0,
) -> CaptureQuality:
    """Compute inexpensive, advisory capture checks on one preview frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    scale = min(1.0, 640.0 / max(gray.shape))
    preview = (
        cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else gray
    )
    blur_score = float(cv2.Laplacian(preview, cv2.CV_64F).var())
    mean_luma = float(np.mean(gray))
    dark_fraction = float(np.mean(gray <= 5))
    bright_fraction = float(np.mean(gray >= 250))

    marker_sides: list[float] = []
    for corners in (detections or {}).values():
        points = np.asarray(corners, dtype=np.float64).reshape(4, 2)
        marker_sides.append(
            float(np.median(np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)))
        )
    warnings: list[str] = []
    if blur_score < 80.0:
        warnings.append("BLUR")
    if mean_luma < 45.0 or dark_fraction > 0.35:
        warnings.append("DARK")
    elif mean_luma > 220.0 or bright_fraction > 0.35:
        warnings.append("BRIGHT")
    if not marker_sides:
        warnings.append("NO_TAG")
    elif float(np.median(marker_sides)) < min_marker_side_px:
        warnings.append("TAG_SMALL")
    return CaptureQuality(
        blur_score=blur_score,
        mean_luma=mean_luma,
        dark_fraction=dark_fraction,
        bright_fraction=bright_fraction,
        marker_count=len(marker_sides),
        median_marker_side_px=(float(np.median(marker_sides)) if marker_sides else None),
        smallest_marker_side_px=(min(marker_sides) if marker_sides else None),
        warnings=tuple(warnings),
    )


class CaptureQualitySummary:
    def __init__(self) -> None:
        self._samples: list[CaptureQuality] = []

    def add(self, quality: CaptureQuality) -> None:
        self._samples.append(quality)

    def clear(self) -> None:
        self._samples.clear()

    def to_dict(self) -> dict[str, object]:
        if not self._samples:
            return {"samples": 0}
        blur = np.asarray([sample.blur_score for sample in self._samples])
        marker_count = np.asarray([sample.marker_count for sample in self._samples])
        warning_counts = {
            warning: sum(warning in sample.warnings for sample in self._samples)
            for warning in ("BLUR", "DARK", "BRIGHT", "NO_TAG", "TAG_SMALL")
        }
        return {
            "samples": len(self._samples),
            "ready_fraction": float(np.mean([sample.ready for sample in self._samples])),
            "blur_score_median": float(np.median(blur)),
            "blur_score_p05": float(np.percentile(blur, 5)),
            "marker_count_median": float(np.median(marker_count)),
            "warning_counts": warning_counts,
            "thresholds": {
                "blur_score_min": 80.0,
                "mean_luma_range": [45.0, 220.0],
                "clipped_fraction_max": 0.35,
            },
        }
