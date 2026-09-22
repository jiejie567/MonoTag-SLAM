#!/usr/bin/env python3
"""Create software-layout JSON and printable marker sheets for a six-face band.

This reconstructs the coordinate contract from PROJECT_HANDOFF.md. It does not
recreate the missing mechanical STL and must not be paired with an unknown STL.
"""
from __future__ import annotations

# Support direct execution from a source checkout.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from aruco_track.models import BandLayout


DPI = 300
MM_TO_PX = DPI / 25.4
A4 = (round(210 * MM_TO_PX), round(297 * MM_TO_PX))


def band_layout(name: str, first_id: int, axis_x_mm: float, axis_z_mm: float, width_mm: float) -> BandLayout:
    vertices = []
    for index in range(6):
        angle = math.radians(index * 60.0)
        vertices.append(np.array([axis_x_mm / 2 * math.sin(angle), 0.0, axis_z_mm / 2 * math.cos(angle)]))
    markers: dict[int, np.ndarray] = {}
    for face in range(6):
        a = vertices[face]
        b = vertices[(face + 1) % 6]
        center = (a + b) / 2.0
        right = (a - b) / np.linalg.norm(a - b)
        marker_mm = 30.75 if np.linalg.norm(a - b) > 30 else 21.5
        half = marker_mm / 2.0
        up = np.array([0.0, 1.0, 0.0])
        corners = np.array([
            center - right * half + up * half,
            center + right * half + up * half,
            center + right * half - up * half,
            center - right * half - up * half,
        ]) / 1000.0
        markers[first_id + face] = corners
    return BandLayout(name, "DICT_4X4_50", markers)


def marker_image(marker_id: int, size_px: int) -> Image.Image:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    pixels = cv2.aruco.generateImageMarker(dictionary, marker_id, size_px)
    return Image.fromarray(pixels)


def marker_size_mm(layout: BandLayout, marker_id: int) -> float:
    points = layout.markers[marker_id]
    return float(np.linalg.norm(points[1] - points[0]) * 1000.0)


def save_marker_sheet(layout: BandLayout, marker_ids: list[int], output: Path) -> None:
    page = Image.new("RGB", A4, "white")
    draw = ImageDraw.Draw(page)
    y = round(24 * MM_TO_PX)
    draw.text((round(20 * MM_TO_PX), y), f"{layout.name}: IDs {marker_ids[0]}-{marker_ids[-1]} | print at 100%", fill="black")
    y += round(18 * MM_TO_PX)
    for marker_id in marker_ids:
        size_mm = marker_size_mm(layout, marker_id)
        size_px = round(size_mm * MM_TO_PX)
        marker = marker_image(marker_id, size_px).convert("RGB")
        x = round((210 * MM_TO_PX - size_px) / 2)
        page.paste(marker, (x, y))
        draw.text((x + size_px + 30, y + size_px // 2), f"F{marker_id - min(layout.markers)} / ID {marker_id} / {size_mm:.2f} mm", fill="black")
        y += size_px + round(15 * MM_TO_PX)
    output.parent.mkdir(parents=True, exist_ok=True)
    page.save(output, "PDF", resolution=DPI)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reconstructed six-face band JSON and marker PDFs (no STL)")
    parser.add_argument("--output", default="print")
    parser.add_argument("--axis-x-mm", type=float, default=69.0)
    parser.add_argument("--axis-z-mm", type=float, default=55.0)
    parser.add_argument("--width-mm", type=float, default=56.0)
    args = parser.parse_args()
    output = Path(args.output)
    for hand, first_id in (("L", 0), ("R", 6)):
        layout = band_layout(f"strap_band_{hand}", first_id, args.axis_x_mm, args.axis_z_mm, args.width_mm)
        layout.save(
            output / f"strap_band_{hand}.json",
            reconstructed=True,
            warning="Not the original mechanical layout. Regenerate after measuring the physical band.",
            nominal_outer_axes_mm=[args.axis_x_mm, args.axis_z_mm],
            nominal_arm_width_mm=args.width_mm,
        )
        save_marker_sheet(layout, list(range(first_id, first_id + 3)), output / f"strap_band_{hand}_half0.pdf")
        save_marker_sheet(layout, list(range(first_id + 3, first_id + 6)), output / f"strap_band_{hand}_half1.pdf")
    print(f"wrote reconstructed JSON/PDF files to {output}")
    print("WARNING: no STL was reconstructed; do not use these JSON files with an unmeasured band")


if __name__ == "__main__":
    main()
