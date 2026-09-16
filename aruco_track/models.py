from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Calibration:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_size: tuple[int, int]

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        data = json.loads(Path(path).read_text())
        return cls(
            camera_matrix=np.asarray(data["camera_matrix"], dtype=np.float64),
            dist_coeffs=np.asarray(data["dist_coeffs"], dtype=np.float64),
            image_size=tuple(data["image_size"]),
        )

    def save(self, path: str | Path, **metadata: object) -> None:
        data = {
            "image_size": list(self.image_size),
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.reshape(-1).tolist(),
            **metadata,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data, indent=2) + "\n")

    def scaled_to(self, image_size: tuple[int, int]) -> "Calibration":
        if image_size == self.image_size:
            return self
        source_aspect = self.image_size[0] / self.image_size[1]
        target_aspect = image_size[0] / image_size[1]
        if not np.isclose(source_aspect, target_aspect, rtol=1e-6):
            raise ValueError(
                f"cannot scale calibration from {self.image_size} to {image_size}: aspect ratio changed"
            )
        scale_x = image_size[0] / self.image_size[0]
        scale_y = image_size[1] / self.image_size[1]
        camera_matrix = self.camera_matrix.copy()
        camera_matrix[0, :] *= scale_x
        camera_matrix[1, :] *= scale_y
        return Calibration(camera_matrix, self.dist_coeffs.copy(), image_size)


@dataclass(frozen=True)
class BandLayout:
    name: str
    dictionary: str
    markers: dict[int, np.ndarray]

    @classmethod
    def load(cls, path: str | Path) -> "BandLayout":
        path = Path(path)
        data = json.loads(path.read_text())
        markers = {
            int(marker["id"]): np.asarray(marker["object_points_m"], dtype=np.float64)
            for marker in data["markers"]
        }
        if any(points.shape != (4, 3) for points in markers.values()):
            raise ValueError("each marker must contain four 3D object points")
        return cls(data.get("name", path.stem), data.get("dictionary", "DICT_4X4_50"), markers)

    def save(self, path: str | Path, **metadata: object) -> None:
        data = {
            "name": self.name,
            "dictionary": self.dictionary,
            "coordinate_system": {
                "origin": "band axis midpoint",
                "+Y": "toward elbow",
                "+Z": "ridge between F5 and F0",
                "unit": "metre",
            },
            "markers": [
                {"id": marker_id, "object_points_m": points.tolist()}
                for marker_id, points in sorted(self.markers.items())
            ],
            **metadata,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data, indent=2) + "\n")


@dataclass
class Pose:
    rvec: np.ndarray
    tvec: np.ndarray
    reprojection_error_px: float
    marker_ids: tuple[int, ...] = ()
    inlier_count: int = 0
    ambiguous: bool = False

    @property
    def rotation_matrix(self) -> np.ndarray:
        import cv2

        return cv2.Rodrigues(self.rvec)[0]
