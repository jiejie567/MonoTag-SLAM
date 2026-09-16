#!/usr/bin/env python3
"""Render lightweight inspection views for the magnetic wrist-band STL pair."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


COLOR_A = "#7884B4"
COLOR_B = "#B58C5A"


def add_mesh(axis, mesh: trimesh.Trimesh, color: str, alpha: float = 1.0) -> None:
    collection = Poly3DCollection(
        mesh.triangles,
        facecolor=color,
        edgecolor="none",
        linewidth=0.0,
        alpha=alpha,
    )
    axis.add_collection3d(collection)


def equal_limits(axis, meshes: list[trimesh.Trimesh], pad: float = 0.07) -> None:
    bounds = np.array([mesh.bounds for mesh in meshes])
    lower = bounds[:, 0].min(axis=0)
    upper = bounds[:, 1].max(axis=0)
    center = (lower + upper) / 2.0
    radius = float((upper - lower).max()) * (0.5 + pad)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))


def style_axis(axis) -> None:
    axis.set_axis_off()
    axis.set_facecolor("#FAF9F6")


def draw_contact_layout(axis, mesh: trimesh.Trimesh, section_x: float, title: str) -> None:
    section = mesh.section(
        plane_origin=[section_x, 0.0, 0.0], plane_normal=[1.0, 0.0, 0.0]
    )
    for path in section.discrete:
        axis.plot(path[:, 1], path[:, 2], color="#272727", linewidth=1.15)
    axis.set_xlim(-45, 45)
    axis.set_ylim(-1, 57)
    axis.set_aspect("equal", adjustable="box")
    axis.set_axis_off()
    axis.set_title(title, fontsize=11, color="#272727", pad=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("half_a", type=Path)
    parser.add_argument("half_b", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    half_a = trimesh.load_mesh(args.half_a, process=True)
    half_b = trimesh.load_mesh(args.half_b, process=True)
    exploded_a = half_a.copy()
    exploded_b = half_b.copy()
    exploded_a.apply_translation([-3.5, 0.0, 0.0])
    exploded_b.apply_translation([3.5, 0.0, 0.0])

    fig = plt.figure(figsize=(12, 6.6), facecolor="#FAF9F6")
    views = (
        ("Assembled", half_a, half_b, 22, -48),
        ("Exploded seam", exploded_a, exploded_b, 14, -74),
    )
    for index, (title, mesh_a, mesh_b, elevation, azimuth) in enumerate(views, 1):
        axis = fig.add_subplot(2, 2, index, projection="3d")
        add_mesh(axis, mesh_a, COLOR_A)
        add_mesh(axis, mesh_b, COLOR_B)
        equal_limits(axis, [mesh_a, mesh_b])
        axis.view_init(elev=elevation, azim=azimuth)
        style_axis(axis)
        axis.set_title(title, fontsize=11, color="#272727", pad=2)

    axis = fig.add_subplot(2, 2, 3)
    draw_contact_layout(axis, half_a, -1.0, "Half A contact face")
    axis = fig.add_subplot(2, 2, 4)
    draw_contact_layout(axis, half_b, 1.0, "Half B contact face")

    fig.text(
        0.5,
        0.018,
        f"69 × 55 mm inner ellipse · {half_a.extents[2]:.0f} mm width · "
        "4.2 × 2.2 mm magnet pockets",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#272727",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1), pad=0.6)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
