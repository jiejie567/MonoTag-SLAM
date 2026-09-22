#!/usr/bin/env python3
from __future__ import annotations

# Support direct execution from a source checkout.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import cv2
from PIL import Image


DPI = 300
MM_TO_PX = DPI / 25.4


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an A4 PDF or board-only PNG ChArUco target")
    parser.add_argument("--output", default="print/charuco_A4.pdf")
    parser.add_argument("--squares-x", type=int, default=7)
    parser.add_argument("--squares-y", type=int, default=10)
    parser.add_argument("--square-mm", type=float, default=24.9)
    parser.add_argument("--marker-mm", type=float, default=19.0)
    args = parser.parse_args()
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard(
        (args.squares_x, args.squares_y), args.square_mm / 1000.0, args.marker_mm / 1000.0, dictionary
    )
    board_w = round(args.squares_x * args.square_mm * MM_TO_PX)
    board_h = round(args.squares_y * args.square_mm * MM_TO_PX)
    if board_w > round(190 * MM_TO_PX) or board_h > round(277 * MM_TO_PX):
        raise SystemExit("board does not fit inside A4 with 10 mm margins")
    pixels = board.generateImage((board_w, board_h), marginSize=0, borderBits=1)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".png":
        Image.fromarray(pixels).save(output)
        print(f"wrote board-only image {output}; square size must be displayed at {args.square_mm} mm")
    else:
        page = Image.new("L", (round(210 * MM_TO_PX), round(297 * MM_TO_PX)), 255)
        page.paste(Image.fromarray(pixels), ((page.width - board_w) // 2, (page.height - board_h) // 2))
        page.save(output, "PDF", resolution=DPI)
        print(f"wrote {output}; print at 100% and verify square size {args.square_mm} mm")


if __name__ == "__main__":
    main()
