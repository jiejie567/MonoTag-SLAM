#!/usr/bin/env python3
"""Generate independently placeable fixed ArUco markers for workstations."""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path

import cv2
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


MM = 72.0 / 25.4
DICTIONARY_NAME = "DICT_4X4_50"
MARKER_SIZE_MM = 48.0
MARKER_IDS = tuple(range(28, 50))
PAGE_CENTERS_MM = ((58.0, 197.0), (152.0, 197.0), (58.0, 99.0), (152.0, 99.0))


def marker_pixels(marker_id: int, size_px: int = 1200):
    dictionary = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, DICTIONARY_NAME)
    )
    return cv2.aruco.generateImageMarker(
        dictionary, marker_id, size_px, borderBits=1
    )


def marker_image(marker_id: int) -> ImageReader:
    ok, encoded = cv2.imencode(".png", marker_pixels(marker_id))
    if not ok:
        raise RuntimeError(f"failed to encode marker {marker_id}")
    return ImageReader(BytesIO(encoded.tobytes()))


def draw_cut_guides(pdf: canvas.Canvas, center_x_mm: float, center_y_mm: float) -> None:
    half = 31.0
    tick = 4.0
    left = (center_x_mm - half) * MM
    right = (center_x_mm + half) * MM
    bottom = (center_y_mm - half) * MM
    top = (center_y_mm + half) * MM
    pdf.setStrokeGray(0.72)
    pdf.setLineWidth(0.25)
    for x, direction in ((left, 1), (right, -1)):
        pdf.line(x, bottom, x + direction * tick * MM, bottom)
        pdf.line(x, top, x + direction * tick * MM, top)
    for y, direction in ((bottom, 1), (top, -1)):
        pdf.line(left, y, left, y + direction * tick * MM)
        pdf.line(right, y, right, y + direction * tick * MM)


def create_pdf(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    page_width, page_height = A4
    pdf = canvas.Canvas(str(path), pagesize=A4, pageCompression=1)
    pdf.setTitle("Independent fixed ArUco markers - IDs 28-49")
    pdf.setAuthor("hand-tracking")
    pdf.setCreator("make_fixed_marker_pack.py")

    for page_start in range(0, len(MARKER_IDS), len(PAGE_CENTERS_MM)):
        page_ids = MARKER_IDS[page_start : page_start + len(PAGE_CENTERS_MM)]
        pdf.setFillGray(0.0)
        pdf.setFont("Helvetica-Bold", 11)
        pdf.drawCentredString(
            page_width / 2.0,
            page_height - 13.0 * MM,
            f"FIXED WORKSTATION MARKERS - {DICTIONARY_NAME} - IDs "
            f"{page_ids[0]}-{page_ids[-1]}",
        )
        pdf.setFillGray(0.35)
        pdf.setFont("Helvetica", 7.5)
        pdf.drawCentredString(
            page_width / 2.0,
            page_height - 18.0 * MM,
            "Print at 100% / Actual Size - disable Fit to Page",
        )

        for marker_id, (center_x_mm, center_y_mm) in zip(page_ids, PAGE_CENTERS_MM):
            x = (center_x_mm - MARKER_SIZE_MM / 2.0) * MM
            y = (center_y_mm - MARKER_SIZE_MM / 2.0) * MM
            pdf.drawImage(
                marker_image(marker_id),
                x,
                y,
                MARKER_SIZE_MM * MM,
                MARKER_SIZE_MM * MM,
                preserveAspectRatio=True,
                mask=None,
            )
            draw_cut_guides(pdf, center_x_mm, center_y_mm)
            pdf.setFillGray(0.25)
            pdf.setFont("Helvetica-Bold", 8)
            pdf.drawCentredString(
                center_x_mm * MM,
                (center_y_mm - MARKER_SIZE_MM / 2.0 - 6.0) * MM,
                f"ID {marker_id} - 48 mm",
            )

        ruler_left = page_width / 2.0 - 25.0 * MM
        ruler_y = 20.0 * MM
        pdf.setStrokeGray(0.0)
        pdf.setLineWidth(0.35)
        pdf.line(ruler_left, ruler_y, ruler_left + 50.0 * MM, ruler_y)
        for tick_mm in range(0, 51, 10):
            height_mm = 3.0 if tick_mm in (0, 50) else 2.0
            x = ruler_left + tick_mm * MM
            pdf.line(
                x,
                ruler_y - height_mm / 2.0 * MM,
                x,
                ruler_y + height_mm / 2.0 * MM,
            )
        pdf.setFillGray(0.35)
        pdf.setFont("Helvetica", 7)
        pdf.drawCentredString(page_width / 2.0, 14.5 * MM, "50 mm verification ruler")
        pdf.drawCentredString(
            page_width / 2.0,
            9.5 * MM,
            "Cut and place independently; relative poses are learned from video, not this sheet.",
        )
        pdf.showPage()
    pdf.save()


def create_assets(output_dir: Path) -> None:
    png_dir = output_dir / "fixed_markers_aruco_28_49_png"
    png_dir.mkdir(parents=True, exist_ok=True)
    for marker_id in MARKER_IDS:
        if not cv2.imwrite(str(png_dir / f"aruco_{marker_id:02d}.png"), marker_pixels(marker_id)):
            raise RuntimeError(f"failed to write marker {marker_id}")

    manifest = {
        "name": "fixed_workstation_markers_28_49",
        "dictionary": DICTIONARY_NAME,
        "marker_ids": list(MARKER_IDS),
        "marker_size_mm": MARKER_SIZE_MM,
        "print_scale": "100% / Actual Size",
        "placement": "independent static anchors; do not move after mapping",
        "mapping": {
            "mode": "auto-marker-map",
            "static_marker_ids": "20-49",
            "relative_pose_policy": (
                "learn from reliable video observations; the packing layout on the PDF "
                "does not define a rigid multi-marker board"
            ),
        },
    }
    (output_dir / "fixed_markers_A4_aruco_28_49.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


def main() -> None:
    output_dir = Path("output/pdf")
    create_pdf(output_dir / "fixed_markers_A4_aruco_28_49.pdf")
    create_assets(output_dir)


if __name__ == "__main__":
    main()
