#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "fixed": "#B58C5A",
    "left": "#B05F78",
    "right": "#A65F5B",
    "other": "#B8BBC4",
    "text": "#272727",
    "invalid": "#787B84",
}


def load_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def load_frames(video: Path, indices: set[int]) -> dict[int, np.ndarray]:
    capture = cv2.VideoCapture(str(video))
    frames: dict[int, np.ndarray] = {}
    index = 0
    while indices - frames.keys():
        ok, frame = capture.read()
        if not ok:
            break
        if index in indices:
            frames[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        index += 1
    capture.release()
    missing = indices - frames.keys()
    if missing:
        raise RuntimeError(f"could not decode frames: {sorted(missing)}")
    return frames


def pose_position(record: dict, side: str) -> np.ndarray | None:
    value = record["hands"][side].get("wrist_world_graph")
    return None if value is None else np.asarray(value["translation_m"], dtype=float)


def step_mm(records: list[dict], frame: int, side: str) -> float | None:
    if frame <= 0:
        return None
    current = pose_position(records[frame], side)
    previous = pose_position(records[frame - 1], side)
    if current is None or previous is None:
        return None
    return float(np.linalg.norm(current - previous) * 1000.0)


def crop_for_event(records: list[dict], indices: list[int], width: int, height: int) -> tuple[int, int, int, int]:
    corners = []
    for index in indices:
        corners.extend(records[index].get("detected_marker_corners", {}).values())
    points = np.concatenate([np.asarray(value, dtype=float) for value in corners])
    x0, y0 = np.floor(points.min(axis=0) - 90).astype(int)
    x1, y1 = np.ceil(points.max(axis=0) + 90).astype(int)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    target_ratio = 1.25
    crop_width, crop_height = x1 - x0, y1 - y0
    if crop_width / crop_height < target_ratio:
        extra = target_ratio * crop_height - crop_width
        x0 -= int(extra / 2)
        x1 += int(np.ceil(extra / 2))
    else:
        extra = crop_width / target_ratio - crop_height
        y0 -= int(extra / 2)
        y1 += int(np.ceil(extra / 2))
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if x1 > width:
        x0 -= x1 - width
        x1 = width
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if y1 > height:
        y0 -= y1 - height
        y1 = height
    return max(0, x0), max(0, y0), min(width, x1), min(height, y1)


def marker_color(marker_id: int, focus_side: str) -> str:
    if 20 <= marker_id <= 49:
        return COLORS["fixed"]
    if marker_id <= 5:
        return COLORS["left"] if focus_side == "strap_band_L" else COLORS["other"]
    if marker_id <= 11:
        return COLORS["right"] if focus_side == "strap_band_R" else COLORS["other"]
    return COLORS["other"]


def draw_panel(ax, image: np.ndarray, record: dict, crop: tuple[int, int, int, int], side: str) -> None:
    x0, y0, x1, y1 = crop
    ax.imshow(image[y0:y1, x0:x1])
    accepted = set(record["hands"][side].get("accepted_marker_ids") or [])
    for marker, values in record.get("detected_marker_corners", {}).items():
        marker_id = int(marker)
        points = np.asarray(values, dtype=float) - np.array([x0, y0])
        color = marker_color(marker_id, side)
        closed = np.vstack((points, points[0]))
        ax.plot(closed[:, 0], closed[:, 1], color=color, lw=1.2, solid_capstyle="round")
        ax.scatter(points[:, 0], points[:, 1], s=5, c=color, edgecolors="white", linewidths=.25)
        if marker_id in accepted or 20 <= marker_id <= 49:
            ax.text(
                points[:, 0].mean(), points[:, 1].mean(), str(marker_id),
                color="white", ha="center", va="center", fontsize=5.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=.16", facecolor=color, edgecolor="none", alpha=.92),
            )
    frame = record["frame"]
    delta = step_mm(RECORDS, frame, side)
    pose = record["hands"][side].get("wrist_world_graph")
    status = "identity only" if accepted and pose is None else ("6-DoF" if pose else "not observed")
    delta_text = "—" if delta is None else f"{delta:.1f} mm"
    short_side = "L" if side == "strap_band_L" else "R"
    ax.set_title(
        f"f{frame} · {record['timestamp_s']:.3f} s\n"
        f"{record['camera_world_source']}\n"
        f"{short_side}: {status} · Δ {delta_text}",
        fontsize=5.2, color=COLORS["text"], pad=2, linespacing=1.12,
    )
    ax.set_axis_off()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("actions", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    global RECORDS
    RECORDS = load_records(args.actions)
    events = [
        ("a", "Largest remaining left-wrist step", "strap_band_L", list(range(226, 231))),
        ("b", "Largest remaining right-wrist step", "strap_band_R", list(range(283, 288))),
        ("c", "Short single-marker segment withheld by the new gate", "strap_band_L", list(range(291, 296))),
    ]
    indices = {index for _, _, _, values in events for index in values}
    frames = load_frames(args.video, indices)
    height, width = next(iter(frames.values())).shape[:2]

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
    })
    fig, axes = plt.subplots(3, 5, figsize=(7.2, 6.0), facecolor="white")
    fig.subplots_adjust(left=.035, right=.995, top=.88, bottom=.07, wspace=.035, hspace=.42)
    fig.suptitle(
        "Static-wrist diagnostic: frames surrounding the largest residual-motion events",
        x=.035, y=.965, ha="left", fontsize=10, fontweight="bold", color=COLORS["text"],
    )
    fig.text(
        .035, .925,
        "Ochre: fixed anchors  ·  rose/red: focused wrist constellation  ·  Δ: consecutive published world-position step",
        ha="left", fontsize=6.5, color="#5E6068",
    )
    for row, (letter, title, side, event_indices) in enumerate(events):
        crop = crop_for_event(RECORDS, event_indices, width, height)
        for column, index in enumerate(event_indices):
            draw_panel(axes[row, column], frames[index], RECORDS[index], crop, side)
        axes[row, 0].text(
            -.02, 1.28, letter, transform=axes[row, 0].transAxes,
            ha="left", va="top", fontsize=8, fontweight="bold", color=COLORS["text"],
        )
        axes[row, 0].text(
            .08, 1.28, title, transform=axes[row, 0].transAxes,
            ha="left", va="top", fontsize=7, fontweight="bold", color=COLORS["text"],
        )
    fig.text(
        .035, .018,
        "Original RGB crops; identical crop within each row; no sharpening or local contrast adjustment. "
        "Decoded identity remains available when 6-DoF publication is withheld.",
        ha="left", fontsize=5.8, color="#6B6D74",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(args.output.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    RECORDS: list[dict] = []
    main()
