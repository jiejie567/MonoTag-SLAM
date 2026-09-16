#!/usr/bin/env python3
"""Add reference-inspired 4 x 2 mm magnet pockets to the two-part wrist band."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh
from trimesh.transformations import rotation_matrix


MAGNET_DIAMETER_MM = 4.2
MAGNET_DEPTH_MM = 2.2
LUG_DEPTH_MM = 3.2
LUG_RADIAL_MM = 7.0
LUG_AXIAL_MM = 10.0
SEAM_Y_MM = (-36.0, 36.0)
VELCRO_CHANNEL_TOP_Z_MM = 16.0


def box(extents: tuple[float, float, float], center: tuple[float, float, float]):
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(center)
    return mesh


def pocket_x(center_x: float, center_y: float, center_z: float):
    mesh = trimesh.creation.cylinder(
        radius=MAGNET_DIAMETER_MM / 2.0,
        height=MAGNET_DEPTH_MM + 0.2,
        sections=64,
    )
    mesh.apply_transform(rotation_matrix(np.pi / 2.0, [0.0, 1.0, 0.0]))
    mesh.apply_translation([center_x, center_y, center_z])
    return mesh


def load_assembly(path: Path) -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
    mesh = trimesh.load_mesh(path, process=True)
    components = list(mesh.split(only_watertight=True))
    if len(components) != 2:
        raise ValueError(f"expected two watertight print components, got {len(components)}")
    # Boolean engines may mutate backing arrays; isolate the two components.
    negative = min(components, key=lambda item: item.bounds[0, 0]).copy()
    positive = max(components, key=lambda item: item.bounds[1, 0]).copy()
    # The source print layout places the negative half at x=-67 mm.
    seam_x = float(negative.bounds[1, 0])
    negative.apply_translation([-seam_x, 0.0, 0.0])
    return negative, positive


def remove_velcro_channel(mesh: trimesh.Trimesh, negative_side: bool) -> trimesh.Trimesh:
    """Remove the complete 16 mm-wide external hook-and-loop channel."""
    del negative_side  # Both halves use the same axial crop.
    retained_width = float(mesh.bounds[1, 2] - VELCRO_CHANNEL_TOP_Z_MM)
    upper_clip = box(
        (200.0, 200.0, retained_width),
        (0.0, 0.0, VELCRO_CHANNEL_TOP_Z_MM + retained_width / 2.0),
    )
    cropped = trimesh.boolean.intersection([mesh, upper_clip], engine="manifold")
    cropped.apply_translation([0.0, 0.0, -VELCRO_CHANNEL_TOP_Z_MM])
    return cropped


def add_join(
    negative: trimesh.Trimesh,
    positive: trimesh.Trimesh,
) -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
    width = float(min(negative.bounds[1, 2], positive.bounds[1, 2]))
    magnet_z_mm = (5.0, width - 5.0) if width <= 45.0 else (8.0, width - 8.0)
    # Remove the original tongue only beneath the magnet pads. Its central
    # portion remains as the mechanical anti-shear feature.
    tail_cutters = [
        # Extend 0.2 mm beyond the axial edge so no cropped tongue sliver
        # survives as a visible bump at either end.
        box((3.2, 8.0, 11.4), (-1.5, seam_y, magnet_z))
        for seam_y in SEAM_Y_MM
        for magnet_z in magnet_z_mm
    ]
    positive = trimesh.boolean.difference(
        [positive, *tail_cutters], engine="manifold"
    )

    negative_lugs = []
    positive_lugs = []
    for seam_y in SEAM_Y_MM:
        for magnet_z in magnet_z_mm:
            negative_lugs.append(
                box(
                    (LUG_DEPTH_MM, LUG_RADIAL_MM, LUG_AXIAL_MM),
                    (-LUG_DEPTH_MM / 2.0, seam_y, magnet_z),
                )
            )
            positive_lugs.append(
                box(
                    (LUG_DEPTH_MM, LUG_RADIAL_MM, LUG_AXIAL_MM),
                    (LUG_DEPTH_MM / 2.0, seam_y, magnet_z),
                )
            )
    negative = trimesh.boolean.union([negative, *negative_lugs], engine="manifold")
    positive = trimesh.boolean.union([positive, *positive_lugs], engine="manifold")

    negative_pockets = []
    positive_pockets = []
    for seam_y in SEAM_Y_MM:
        for magnet_z in magnet_z_mm:
            negative_pockets.append(
                pocket_x(-MAGNET_DEPTH_MM / 2.0 + 0.05, seam_y, magnet_z)
            )
            positive_pockets.append(
                pocket_x(MAGNET_DEPTH_MM / 2.0 - 0.05, seam_y, magnet_z)
            )
    negative = trimesh.boolean.difference(
        [negative, *negative_pockets], engine="manifold"
    )
    positive = trimesh.boolean.difference(
        [positive, *positive_pockets], engine="manifold"
    )
    return negative, positive


def verify(negative: trimesh.Trimesh, positive: trimesh.Trimesh) -> None:
    for name, mesh in (("half_A", negative), ("half_B", positive)):
        if not mesh.is_watertight or not mesh.is_winding_consistent:
            raise RuntimeError(f"{name} is not a closed, consistently wound mesh")
        if mesh.volume <= 0.0:
            raise RuntimeError(f"{name} has non-positive volume")
    collision = trimesh.boolean.intersection(
        [negative, positive], engine="manifold"
    )
    if collision.volume > 1e-3:
        raise RuntimeError(
            f"assembled halves overlap by {collision.volume:.6f} mm^3"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--remove-velcro-channel", action="store_true")
    args = parser.parse_args()
    negative, positive = load_assembly(args.input)
    if args.remove_velcro_channel:
        negative = remove_velcro_channel(negative, negative_side=True)
        positive = remove_velcro_channel(positive, negative_side=False)
    negative, positive = add_join(negative, positive)
    verify(negative, positive)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_magnetic_no_velcro" if args.remove_velcro_channel else "_magnetic"
    half_a_path = args.output_dir / f"strap_half_A{suffix}.stl"
    half_b_path = args.output_dir / f"strap_half_B{suffix}.stl"
    negative.export(half_a_path)
    positive.export(half_b_path)
    print_negative = negative.copy()
    print_negative.apply_translation([-67.0, 0.0, 0.0])
    combined = trimesh.util.concatenate((print_negative, positive))
    combined_path = args.output_dir / f"strap_band{suffix}_print.stl"
    combined.export(combined_path)
    print(f"wrote {half_a_path}")
    print(f"wrote {half_b_path}")
    print(f"wrote {combined_path}")
    print(
        f"magnet pockets: 8 per band, diameter={MAGNET_DIAMETER_MM} mm, "
        f"depth={MAGNET_DEPTH_MM} mm"
    )


if __name__ == "__main__":
    main()
