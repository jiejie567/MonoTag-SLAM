#!/usr/bin/env python3
"""Create a non-destructive +6 mm inner-ellipse preview from the existing STL.

The source band has a nominal 63 x 49 mm inner ellipse.  Scaling the two
cross-section axes to 69 x 55 mm leaves the 56 mm axial width unchanged.  For
the combined print STL, each separated half is scaled about its own seam.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import struct

import numpy as np


FACET_DTYPE = np.dtype(
    [("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")]
)
SCALE_X = 55.0 / 49.0  # minor inner diameter: 49 -> 55 mm
SCALE_Y = 69.0 / 63.0  # major inner diameter: 63 -> 69 mm


def read_binary_stl(path: Path) -> tuple[bytes, np.ndarray]:
    payload = path.read_bytes()
    if len(payload) < 84:
        raise ValueError(f"not a binary STL: {path}")
    triangle_count = struct.unpack_from("<I", payload, 80)[0]
    if len(payload) != 84 + 50 * triangle_count:
        raise ValueError(f"unexpected STL size: {path}")
    facets = np.frombuffer(
        payload, dtype=FACET_DTYPE, count=triangle_count, offset=84
    ).copy()
    return payload[:80], facets


def component_masks(vertices: np.ndarray) -> list[np.ndarray]:
    centroids = vertices[:, :, 0].mean(axis=1)
    ordered = np.sort(centroids)
    gaps = np.diff(ordered)
    if len(gaps) and float(gaps.max()) > 10.0:
        split = 0.5 * (ordered[int(np.argmax(gaps))] + ordered[int(np.argmax(gaps)) + 1])
        return [centroids < split, centroids >= split]
    return [np.ones(len(vertices), dtype=bool)]


def resize_facets(facets: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]]]:
    vertices = facets["vertices"].astype(np.float64)
    masks = component_masks(vertices)
    components: list[dict[str, object]] = []
    for component_index, mask in enumerate(masks):
        part = vertices[mask]
        minimum = part.min(axis=(0, 1))
        maximum = part.max(axis=(0, 1))
        # Tenons cross the split plane, so geometric min/max is not the seam.
        # The true split plane appears hundreds of times in the triangulation.
        rounded_x = np.round(part[:, :, 0].reshape(-1), 4)
        values, counts = np.unique(rounded_x, return_counts=True)
        seam_candidates = values[counts >= max(100, int(0.2 * counts.max()))]
        if len(masks) == 1:
            seam_x = float(seam_candidates[np.argmin(np.abs(seam_candidates))])
        elif component_index == 0:
            seam_x = float(seam_candidates.max())
        else:
            seam_x = float(seam_candidates.min())
        part[:, :, 0] = seam_x + (part[:, :, 0] - seam_x) * SCALE_X
        part[:, :, 1] *= SCALE_Y
        vertices[mask] = part
        resized_minimum = part.min(axis=(0, 1))
        resized_maximum = part.max(axis=(0, 1))
        components.append(
            {
                "triangles": int(mask.sum()),
                "seam_x_mm": float(seam_x),
                "source_extent_mm": (maximum - minimum).tolist(),
                "output_extent_mm": (resized_maximum - resized_minimum).tolist(),
            }
        )
    edge_a = vertices[:, 1] - vertices[:, 0]
    edge_b = vertices[:, 2] - vertices[:, 0]
    normals = np.cross(edge_a, edge_b)
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths <= 1e-10):
        raise ValueError("source STL contains a degenerate triangle")
    facets["vertices"] = vertices.astype(np.float32)
    facets["normal"] = (normals / lengths[:, None]).astype(np.float32)
    return facets, components


def write_binary_stl(path: Path, facets: np.ndarray) -> None:
    label = b"hand-tracking strap inner ellipse +6 mm preview"
    output_header = (label + b" " * 80)[:80]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(output_header)
        stream.write(struct.pack("<I", len(facets)))
        stream.write(facets.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    _, facets = read_binary_stl(args.input)
    resized, components = resize_facets(facets)
    write_binary_stl(args.output, resized)
    print(f"wrote {args.output}")
    print(f"inner ellipse: 63 x 49 mm -> 69 x 55 mm; axial width unchanged")
    for index, component in enumerate(components):
        print(f"component {index}: {component}")


if __name__ == "__main__":
    main()
