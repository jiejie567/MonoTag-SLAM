#!/usr/bin/env python3
from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path

import cv2
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


MM = 72.0 / 25.4
MARKER_SIZE_MM = 48.0
HORIZONTAL_GAP_MM = 15.0
VERTICAL_GAP_MM = 10.0
MARKER_IDS = tuple(range(20, 28))


def marker_centers_mm() -> list[tuple[float, float]]:
    x = (MARKER_SIZE_MM + HORIZONTAL_GAP_MM) / 2.0
    pitch_y = MARKER_SIZE_MM + VERTICAL_GAP_MM
    return [
        (column_x, (1.5 - row) * pitch_y)
        for row in range(4)
        for column_x in (-x, x)
    ]


def object_points_m(center_x_mm: float, center_y_mm: float) -> list[list[float]]:
    half = MARKER_SIZE_MM / 2.0
    return [
        [(center_x_mm - half) / 1000.0, (center_y_mm + half) / 1000.0, 0.0],
        [(center_x_mm + half) / 1000.0, (center_y_mm + half) / 1000.0, 0.0],
        [(center_x_mm + half) / 1000.0, (center_y_mm - half) / 1000.0, 0.0],
        [(center_x_mm - half) / 1000.0, (center_y_mm - half) / 1000.0, 0.0],
    ]


def marker_image(marker_id: int) -> ImageReader:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    image = cv2.aruco.generateImageMarker(dictionary, marker_id, 1200, borderBits=1)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"failed to encode marker {marker_id}")
    return ImageReader(BytesIO(encoded.tobytes()))


def create_pdf(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    page_width, page_height = A4
    pdf = canvas.Canvas(str(path), pagesize=A4, pageCompression=1)
    pdf.setTitle("A4 World Reference Board - ArUco IDs 20-27")
    pdf.setAuthor("hand-tracking")
    pdf.setCreator("make_world_board.py")
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawCentredString(
        page_width / 2.0,
        page_height - 11.0 * MM,
        "WORLD REFERENCE BOARD - DICT_4X4_50 - IDs 20-27",
    )
    pdf.setFont("Helvetica", 7.5)
    pdf.setFillGray(0.35)
    pdf.drawCentredString(
        page_width / 2.0,
        page_height - 16.0 * MM,
        "A4 portrait - print at 100% / Actual Size - disable Fit to Page",
    )

    for marker_id, (center_x_mm, center_y_mm) in zip(MARKER_IDS, marker_centers_mm()):
        x = page_width / 2.0 + (center_x_mm - MARKER_SIZE_MM / 2.0) * MM
        y = page_height / 2.0 + (center_y_mm - MARKER_SIZE_MM / 2.0) * MM
        pdf.drawImage(
            marker_image(marker_id),
            x,
            y,
            MARKER_SIZE_MM * MM,
            MARKER_SIZE_MM * MM,
            preserveAspectRatio=True,
        )
        pdf.setFont("Helvetica-Bold", 7)
        pdf.setFillGray(0.35)
        pdf.drawCentredString(
            page_width / 2.0 + center_x_mm * MM,
            page_height / 2.0 + (center_y_mm - MARKER_SIZE_MM / 2.0 - 3.0) * MM,
            f"ID {marker_id}",
        )

    pdf.setStrokeGray(0.55)
    pdf.setLineWidth(0.25)
    center_x = page_width / 2.0
    center_y = page_height / 2.0
    pdf.line(center_x - 3.0 * MM, center_y, center_x + 3.0 * MM, center_y)
    pdf.line(center_x, center_y - 3.0 * MM, center_x, center_y + 3.0 * MM)

    ruler_left = center_x - 50.0 * MM
    ruler_y = 23.0 * MM
    pdf.setStrokeGray(0.0)
    pdf.setLineWidth(0.35)
    pdf.line(ruler_left, ruler_y, ruler_left + 100.0 * MM, ruler_y)
    for tick_mm in range(0, 101, 10):
        height_mm = 3.0 if tick_mm in (0, 50, 100) else 2.0
        x = ruler_left + tick_mm * MM
        pdf.line(x, ruler_y - height_mm / 2.0 * MM, x, ruler_y + height_mm / 2.0 * MM)
    pdf.setFont("Helvetica", 7)
    pdf.drawCentredString(center_x, 17.5 * MM, "100 mm verification ruler")
    pdf.setFillGray(0.35)
    pdf.drawCentredString(
        center_x,
        11.5 * MM,
        "Origin: page center   +X: right   +Y: up   +Z: out of printed face",
    )
    pdf.save()


def create_layout(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    markers = [
        {"id": marker_id, "object_points_m": object_points_m(center_x, center_y)}
        for marker_id, (center_x, center_y) in zip(MARKER_IDS, marker_centers_mm())
    ]
    payload = {
        "name": "world_board_A4",
        "dictionary": "DICT_4X4_50",
        "page_size_mm": [210.0, 297.0],
        "marker_size_mm": MARKER_SIZE_MM,
        "print_scale": "100% / Actual Size",
        "coordinate_system": {
            "origin": "physical center of A4 page",
            "+X": "page right",
            "+Y": "page up",
            "+Z": "out of printed face",
            "unit": "metre",
        },
        "markers": markers,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    output_dir = Path("output/pdf")
    create_pdf(output_dir / "world_reference_board_A4_aruco_20_27.pdf")
    create_layout(output_dir / "world_reference_board_A4_aruco_20_27.json")


if __name__ == "__main__":
    main()
