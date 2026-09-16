from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import json
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .bandsolve import solve_band_pose
from .models import BandLayout, Calibration, Pose
from .pipeline import compose_pose, inverse_pose, relative_pose
from .pose import reprojection_error, solve_square_pose, square_object_points
from .tag_graph import MarkerPoseTracker, TagPoseResult, optimize_tag_pose


@dataclass(frozen=True)
class CoVisibilityEdge:
    first_id: int
    second_id: int
    first_from_second: Pose
    observations: int
    translation_spread_m: float
    rotation_spread_deg: float
    reprojection_error_px: float | None = None
    viewpoint_span_deg: float = 0.0


@dataclass(frozen=True)
class AutoMarkerSubmap:
    submap_id: str
    anchor_marker_id: int
    marker_poses: dict[int, Pose]
    layout: BandLayout
    edges: tuple[CoVisibilityEdge, ...]
    reprojection_error_px: float | None


@dataclass(frozen=True)
class AutoMarkerMap:
    dictionary: str
    marker_size_m: float
    submaps: tuple[AutoMarkerSubmap, ...]
    mode: str = "reliable-covisibility-local-submaps"
    pending_marker_ids: tuple[int, ...] = ()

    @property
    def marker_to_submap(self) -> dict[int, str]:
        return {
            marker_id: submap.submap_id
            for submap in self.submaps
            for marker_id in submap.marker_poses
        }

    def save(self, path: str | Path) -> None:
        data = {
            "schema_version": 1,
            "mode": self.mode,
            "dictionary": self.dictionary,
            "marker_size_mm": 1000.0 * self.marker_size_m,
            "coordinate_policy": (
                "one recording uses one persistent session map; tracking loss "
                "does not create a new map and unregistered markers remain pending"
                if self.mode == "single-session-marker-map"
                else "each submap origin is its anchor marker; disconnected markers "
                "are not assigned a relative pose"
            ),
            "pending_marker_ids": list(self.pending_marker_ids),
            "submaps": [
                {
                    "submap_id": submap.submap_id,
                    "anchor_marker_id": submap.anchor_marker_id,
                    "marker_ids": sorted(submap.marker_poses),
                    "reprojection_error_px": submap.reprojection_error_px,
                    "markers": [
                        {
                            "id": marker_id,
                            "world_from_marker": {
                                "rotation_matrix": pose.rotation_matrix.tolist(),
                                "translation_m": pose.tvec.reshape(3).tolist(),
                            },
                            "object_points_m": submap.layout.markers[
                                marker_id
                            ].tolist(),
                        }
                        for marker_id, pose in sorted(submap.marker_poses.items())
                    ],
                    "covisibility_edges": [
                        {
                            "marker_ids": [edge.first_id, edge.second_id],
                            "observations": edge.observations,
                            "translation_spread_m": edge.translation_spread_m,
                            "rotation_spread_deg": edge.rotation_spread_deg,
                            "reprojection_error_px": edge.reprojection_error_px,
                            "viewpoint_span_deg": edge.viewpoint_span_deg,
                        }
                        for edge in submap.edges
                    ],
                }
                for submap in self.submaps
            ],
        }
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


@dataclass(frozen=True)
class AutoMarkerFrames:
    results: list[TagPoseResult | None]
    submap_ids: list[str | None]


def _marker_area(corners: np.ndarray) -> float:
    return abs(cv2.contourArea(np.asarray(corners, dtype=np.float32).reshape(4, 2)))


def _square_pose_candidates(
    corners: np.ndarray,
    marker_size_m: float,
    calibration: Calibration,
    maximum_error_px: float,
) -> list[Pose]:
    points = square_object_points(marker_size_m)
    solved = cv2.solvePnPGeneric(
        points,
        np.asarray(corners, dtype=np.float64).reshape(4, 2),
        calibration.camera_matrix,
        calibration.dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not solved[0]:
        return []
    output = []
    for rvec, tvec in zip(solved[1], solved[2]):
        rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
        if (
            not np.all(np.isfinite(rvec))
            or not np.all(np.isfinite(tvec))
            or tvec[2, 0] <= 0
            or np.linalg.norm(rvec) > 10.0
            or np.linalg.norm(tvec) > 5.0
        ):
            continue
        error = reprojection_error(
            points,
            corners,
            rvec,
            tvec,
            calibration.camera_matrix,
            calibration.dist_coeffs,
        )
        if error <= maximum_error_px:
            output.append(Pose(rvec, tvec, error))
    return output


def _rotation_distance(first: np.ndarray, second: np.ndarray) -> float:
    cosine = (float(np.trace(first.T @ second)) - 1.0) * 0.5
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _mean_rotation(rotations: list[np.ndarray]) -> np.ndarray:
    total = np.sum(rotations, axis=0)
    left, _, right = np.linalg.svd(total)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(left @ right)
    return left @ correction @ right


def _robust_edge(
    first_id: int,
    second_id: int,
    observations: list[tuple[int, Pose, np.ndarray]],
    minimum_observations: int,
    minimum_viewpoint_span_deg: float = 8.0,
    hypothesis_error: Callable[[Pose], float] | None = None,
) -> CoVisibilityEdge | None:
    if len({frame for frame, _, _ in observations}) < minimum_observations:
        return None
    # IPPE contributes up to four relative-pose combinations per frame.  Trying
    # every one as a cluster seed makes this robust selection quadratic in the
    # video length without adding information.  Score a deterministic uniform
    # hypothesis subset against all observations; all frames still contribute
    # to the selected cluster and the later bundle adjustment.
    maximum_seed_hypotheses = 64
    if len(observations) > maximum_seed_hypotheses:
        seed_indices = np.linspace(
            0, len(observations) - 1, maximum_seed_hypotheses, dtype=int
        )
        seed_observations = [observations[index] for index in seed_indices]
    else:
        seed_observations = observations
    prepared = [
        (frame, pose, view_direction, pose.rotation_matrix)
        for frame, pose, view_direction in observations
    ]
    clusters: list[tuple[list[tuple[Pose, np.ndarray]], Pose]] = []
    for _, seed, _ in seed_observations:
        seed_rotation = seed.rotation_matrix
        per_frame: dict[int, tuple[float, Pose, np.ndarray]] = {}
        for frame, pose, view_direction, pose_rotation in prepared:
            translation_error = float(np.linalg.norm(pose.tvec - seed.tvec))
            rotation_error = _rotation_distance(seed_rotation, pose_rotation)
            if translation_error > 0.05 or rotation_error > np.deg2rad(15.0):
                continue
            cost = translation_error + 0.1 * rotation_error
            if frame not in per_frame or cost < per_frame[frame][0]:
                per_frame[frame] = (cost, pose, view_direction)
        values = [(value[1], value[2]) for value in per_frame.values()]
        if len(values) < minimum_observations:
            continue
        if any(
            np.linalg.norm(seed.tvec - center.tvec) < 0.02
            and _rotation_distance(seed.rotation_matrix, center.rotation_matrix)
            < np.deg2rad(8.0)
            for _, center in clusters
        ):
            continue
        clusters.append((values, seed))
    if not clusters:
        return None
    clusters.sort(key=lambda item: len(item[0]), reverse=True)
    contenders = [item for item in clusters
                  if len(item[0]) >= max(minimum_observations, .8 * len(clusters[0][0]))]
    if hypothesis_error is None:
        # Preserve the established baseline unless the experimental gate is
        # explicitly requested; its real-clip coverage is not validated yet.
        values, _ = max(clusters, key=lambda item: (
            len(item[0]), -float(np.linalg.norm(cv2.Rodrigues(
                _mean_rotation([pose.rotation_matrix for pose, _ in item[0]]))[0]))))
    elif len(contenders) > 1:
        # Rotation magnitude depends on the chosen coordinates, not reliability.
        # Keep ambiguous layout edges pending unless independent pixel evidence
        # separates their hypotheses. Bound work rather than silently dropping
        # an untested competing branch.
        if len(contenders) > 8:
            return None
        ranked = sorted(((hypothesis_error(seed), i) for i, (_, seed)
                         in enumerate(contenders)), key=lambda item: item[0])
        if (not np.isfinite(ranked[0][0]) or ranked[0][0] > 3.0
                or ranked[1][0] - ranked[0][0] < .5):
            return None
        values, _ = contenders[ranked[0][1]]
    else:
        values, _ = contenders[0]
    poses = [pose for pose, _ in values]
    view_directions = np.asarray([direction for _, direction in values])
    if not np.all(np.isfinite(view_directions)):
        return None
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        view_cosines = np.clip(view_directions @ view_directions.T, -1.0, 1.0)
    if not np.all(np.isfinite(view_cosines)):
        return None
    viewpoint_span_deg = float(
        np.rad2deg(np.max(np.arccos(view_cosines)))
    )
    if viewpoint_span_deg < minimum_viewpoint_span_deg:
        return None
    translations = np.asarray([pose.tvec.reshape(3) for pose in poses])
    rotations = [pose.rotation_matrix for pose in poses]
    translation_center = np.median(translations, axis=0)
    medoid = min(
        range(len(rotations)),
        key=lambda index: sum(
            _rotation_distance(rotations[index], other) for other in rotations
        ),
    )
    rotation_center = rotations[medoid]
    translation_errors = np.linalg.norm(translations - translation_center, axis=1)
    rotation_errors = np.asarray(
        [_rotation_distance(rotation_center, rotation) for rotation in rotations]
    )
    translation_cutoff = min(
        0.06, max(0.01, 3.0 * float(np.median(translation_errors)) + 0.003)
    )
    rotation_cutoff = min(
        np.deg2rad(20.0),
        max(np.deg2rad(5.0), 3.0 * float(np.median(rotation_errors)) + np.deg2rad(1.0)),
    )
    inliers = (translation_errors <= translation_cutoff) & (
        rotation_errors <= rotation_cutoff
    )
    if int(np.count_nonzero(inliers)) < minimum_observations:
        return None
    translations = translations[inliers]
    rotations = [rotation for rotation, keep in zip(rotations, inliers) if keep]
    translation = np.median(translations, axis=0)
    rotation = _mean_rotation(rotations)
    translation_spread = float(
        np.sqrt(np.mean(np.sum((translations - translation) ** 2, axis=1)))
    )
    rotation_spread = float(
        np.sqrt(np.mean([_rotation_distance(rotation, value) ** 2 for value in rotations]))
    )
    if translation_spread > 0.04 or rotation_spread > np.deg2rad(12.0):
        return None
    return CoVisibilityEdge(
        first_id,
        second_id,
        Pose(cv2.Rodrigues(rotation)[0], translation.reshape(3, 1), 0.0),
        len(rotations),
        translation_spread,
        float(np.rad2deg(rotation_spread)),
        None,
        viewpoint_span_deg,
    )


def _components(marker_ids: set[int], edges: list[CoVisibilityEdge]) -> list[set[int]]:
    neighbors = {marker_id: set() for marker_id in marker_ids}
    for edge in edges:
        neighbors[edge.first_id].add(edge.second_id)
        neighbors[edge.second_id].add(edge.first_id)
    output: list[set[int]] = []
    unseen = set(marker_ids)
    while unseen:
        start = min(unseen)
        component = {start}
        pending = [start]
        unseen.remove(start)
        while pending:
            current = pending.pop()
            for neighbor in neighbors[current].intersection(unseen):
                unseen.remove(neighbor)
                component.add(neighbor)
                pending.append(neighbor)
        output.append(component)
    return output


def _initial_marker_poses(
    component: set[int],
    anchor: int,
    edges: list[CoVisibilityEdge],
) -> dict[int, Pose]:
    adjacency: dict[int, list[tuple[int, Pose, float]]] = {
        marker_id: [] for marker_id in component
    }
    for edge in edges:
        if edge.first_id not in component or edge.second_id not in component:
            continue
        weight = edge.observations / (
            1.0 + 100.0 * edge.translation_spread_m + edge.rotation_spread_deg
        )
        adjacency[edge.first_id].append(
            (edge.second_id, edge.first_from_second, weight)
        )
        adjacency[edge.second_id].append(
            (edge.first_id, inverse_pose(edge.first_from_second), weight)
        )
    poses = {anchor: Pose(np.zeros((3, 1)), np.zeros((3, 1)), 0.0)}
    while len(poses) < len(component):
        choices = [
            (weight, source, target, source_from_target)
            for source in poses
            for target, source_from_target, weight in adjacency[source]
            if target not in poses
        ]
        if not choices:
            break
        _, source, target, source_from_target = max(choices, key=lambda item: item[0])
        poses[target] = compose_pose(poses[source], source_from_target)
    return poses


def _layout_from_poses(
    name: str,
    dictionary: str,
    marker_size_m: float,
    marker_poses: dict[int, Pose],
) -> BandLayout:
    local = square_object_points(marker_size_m)
    markers = {
        marker_id: (
            pose.rotation_matrix @ local.T
        ).T + pose.tvec.reshape(1, 3)
        for marker_id, pose in marker_poses.items()
    }
    return BandLayout(name, dictionary, markers)


def _refine_submap(
    submap_id: str,
    anchor: int,
    marker_poses: dict[int, Pose],
    detections: list[dict[int, np.ndarray]],
    calibration: Calibration,
    dictionary: str,
    marker_size_m: float,
    maximum_frames: int = 120,
) -> tuple[dict[int, Pose], float | None]:
    if len(marker_poses) < 2:
        return marker_poses, None
    initial_layout = _layout_from_poses(
        submap_id, dictionary, marker_size_m, marker_poses
    )
    candidates = [
        (frame_index, {i: corners for i, corners in frame.items() if i in marker_poses})
        for frame_index, frame in enumerate(detections)
        if len(set(frame).intersection(marker_poses)) >= 2
    ]
    if len(candidates) > maximum_frames:
        selected = np.linspace(0, len(candidates) - 1, maximum_frames, dtype=int)
        candidates = [candidates[index] for index in selected]
    observations: list[tuple[dict[int, np.ndarray], Pose]] = []
    for _, frame in candidates:
        camera_pose = solve_band_pose(
            frame,
            initial_layout,
            calibration.camera_matrix,
            calibration.dist_coeffs,
            max_error_px=5.0,
        )
        if camera_pose is not None:
            observations.append((frame, camera_pose))
    if not observations:
        return marker_poses, None

    variable_markers = [marker_id for marker_id in sorted(marker_poses) if marker_id != anchor]
    marker_offsets = {marker_id: 6 * index for index, marker_id in enumerate(variable_markers)}
    camera_offset = 6 * len(variable_markers)
    values = [
        np.concatenate(
            (marker_poses[marker_id].rvec.reshape(3), marker_poses[marker_id].tvec.reshape(3))
        )
        for marker_id in variable_markers
    ] + [
        np.concatenate((pose.rvec.reshape(3), pose.tvec.reshape(3)))
        for _, pose in observations
    ]
    state = np.concatenate(values) if values else np.empty(0, dtype=np.float64)
    residual_count = sum(8 * len(frame) for frame, _ in observations)
    sparsity = lil_matrix((residual_count, len(state)), dtype=np.int8)
    row = 0
    for observation_index, (frame, _) in enumerate(observations):
        camera_columns = slice(
            camera_offset + 6 * observation_index,
            camera_offset + 6 * observation_index + 6,
        )
        for marker_id in sorted(frame):
            sparsity[row : row + 8, camera_columns] = 1
            if marker_id != anchor:
                offset = marker_offsets[marker_id]
                sparsity[row : row + 8, offset : offset + 6] = 1
            row += 8

    local_points = square_object_points(marker_size_m)

    def unpack(value: np.ndarray) -> tuple[dict[int, Pose], list[Pose]]:
        poses = {anchor: marker_poses[anchor]}
        for marker_id in variable_markers:
            offset = marker_offsets[marker_id]
            poses[marker_id] = Pose(
                value[offset : offset + 3].reshape(3, 1),
                value[offset + 3 : offset + 6].reshape(3, 1),
                0.0,
            )
        cameras = [
            Pose(
                value[camera_offset + 6 * index : camera_offset + 6 * index + 3].reshape(3, 1),
                value[camera_offset + 6 * index + 3 : camera_offset + 6 * index + 6].reshape(3, 1),
                0.0,
            )
            for index in range(len(observations))
        ]
        return poses, cameras

    def residual(value: np.ndarray) -> np.ndarray:
        poses, cameras = unpack(value)
        output: list[np.ndarray] = []
        for (frame, _), camera in zip(observations, cameras):
            for marker_id in sorted(frame):
                marker = poses[marker_id]
                world_points = (
                    marker.rotation_matrix @ local_points.T
                ).T + marker.tvec.reshape(1, 3)
                projected, _ = cv2.projectPoints(
                    world_points,
                    camera.rvec,
                    camera.tvec,
                    calibration.camera_matrix,
                    calibration.dist_coeffs,
                )
                output.append(
                    (projected.reshape(4, 2) - frame[marker_id].reshape(4, 2)).reshape(-1)
                )
        return np.concatenate(output)

    initial_error = float(np.sqrt(np.mean(residual(state) ** 2)))
    optimized = least_squares(
        residual,
        state,
        jac_sparsity=sparsity.tocsr(),
        loss="huber",
        f_scale=2.0,
        max_nfev=80,
    )
    optimized_error = float(np.sqrt(np.mean(residual(optimized.x) ** 2)))
    if not np.all(np.isfinite(optimized.x)) or optimized_error > min(5.0, 1.1 * initial_error):
        return marker_poses, initial_error
    return unpack(optimized.x)[0], optimized_error


def build_auto_marker_map(
    detections: list[dict[int, np.ndarray]],
    calibration: Calibration,
    marker_ids: set[int],
    marker_size_m: float,
    dictionary: str = "DICT_4X4_50",
    minimum_covisibility_frames: int = 6,
    minimum_marker_frames: int = 3,
    minimum_area_px2: float = 400.0,
    maximum_single_marker_error_px: float = 5.0,
    single_session: bool = False,
    verify_ambiguous_edges: bool = False,
) -> AutoMarkerMap:
    """Build layouts; pixel hypothesis verification is experimental and opt-in.

    The gate can reject useful co-visibility edges before joint refinement.
    It is intentionally not enabled by the production exporter.
    """
    if marker_size_m <= 0:
        raise ValueError("marker size must be positive")
    pair_observations: dict[
        tuple[int, int], list[tuple[int, Pose, np.ndarray]]
    ] = {}
    pair_pixels = {}
    pair_pixel_counts = {}
    pixel_rng = np.random.default_rng(0)
    marker_observations: dict[int, int] = {}
    usable_frames: list[dict[int, np.ndarray]] = []
    for frame_index, frame in enumerate(detections):
        candidates: dict[int, list[Pose]] = {}
        usable: dict[int, np.ndarray] = {}
        for marker_id, corners in frame.items():
            if marker_id not in marker_ids or _marker_area(corners) < minimum_area_px2:
                continue
            poses = _square_pose_candidates(
                corners,
                marker_size_m,
                calibration,
                maximum_single_marker_error_px,
            )
            if not poses:
                continue
            for pose in poses:
                pose.marker_ids = (marker_id,)
                pose.inlier_count = 4
            candidates[marker_id] = poses
            usable[marker_id] = corners
            marker_observations[marker_id] = marker_observations.get(marker_id, 0) + 1
        usable_frames.append(usable)
        for first_id, second_id in combinations(sorted(candidates), 2):
            pair = (first_id, second_id)
            # Deterministic reservoir: at most 12 pixel observations per pair,
            # spanning the recording without retaining a second full cache.
            if verify_ambiguous_edges:
                views = pair_pixels.setdefault(pair, [])
                count = pair_pixel_counts[pair] = pair_pixel_counts.get(pair, 0) + 1
                slot = len(views) if len(views) < 12 else int(pixel_rng.integers(count))
                if slot < 12:
                    value = (candidates[first_id], np.asarray(usable[second_id]).copy())
                    if slot == len(views):
                        views.append(value)
                    else:
                        views[slot] = value
            pair_observations.setdefault((first_id, second_id), []).extend(
                (
                    frame_index,
                    relative_pose(first_pose, second_pose),
                    (
                        -first_pose.rotation_matrix.T @ first_pose.tvec
                    ).reshape(3)
                    / max(
                        float(
                            np.linalg.norm(
                                -first_pose.rotation_matrix.T @ first_pose.tvec
                            )
                        ),
                        1e-12,
                    ),
                )
                for first_pose in candidates[first_id]
                for second_pose in candidates[second_id]
            )

    observed_ids = {
        marker_id
        for marker_id, count in marker_observations.items()
        if count >= minimum_marker_frames
    }
    def pixel_scorer(pair):
        views = pair_pixels[pair]
        selected = np.linspace(0, len(views) - 1, min(12, len(views)), dtype=int)
        points = square_object_points(marker_size_m)
        def score(first_from_second):
            errors = []
            for index in selected:
                cameras, pixels = views[index]
                costs = []
                for camera in cameras:
                    projected_pose = compose_pose(camera, first_from_second)
                    if np.any((projected_pose.rotation_matrix @ points.T
                               + projected_pose.tvec)[2] <= 0):
                        continue
                    error = reprojection_error(points, pixels, projected_pose.rvec,
                                               projected_pose.tvec, calibration.camera_matrix,
                                               calibration.dist_coeffs)
                    costs.append(np.hypot(camera.reprojection_error_px, error) / np.sqrt(2))
                errors.append(min(costs) if costs else np.inf)
            return float(np.median(errors))
        return score
    initial_edges = [
        edge
        for (first_id, second_id), values in pair_observations.items()
        if first_id in observed_ids and second_id in observed_ids
        for edge in [
            _robust_edge(
                first_id,
                second_id,
                values,
                minimum_covisibility_frames,
                minimum_viewpoint_span_deg=(0.0 if single_session else 8.0),
                hypothesis_error=(pixel_scorer((first_id, second_id))
                                  if verify_ambiguous_edges else None),
            )
        ]
        if edge is not None
    ]
    edges: list[CoVisibilityEdge] = []
    for edge in initial_edges:
        pair_poses = {
            edge.first_id: Pose(
                np.zeros((3, 1)), np.zeros((3, 1)), 0.0
            ),
            edge.second_id: edge.first_from_second,
        }
        refined, error = _refine_submap(
            f"pair_{edge.first_id}_{edge.second_id}",
            edge.first_id,
            pair_poses,
            usable_frames,
            calibration,
            dictionary,
            marker_size_m,
            maximum_frames=40,
        )
        if error is None or error > 3.0:
            continue
        edges.append(
            CoVisibilityEdge(
                edge.first_id,
                edge.second_id,
                refined[edge.second_id],
                edge.observations,
                edge.translation_spread_m,
                edge.rotation_spread_deg,
                error,
                edge.viewpoint_span_deg,
            )
        )
    components = _components(observed_ids, edges)
    pending_marker_ids: tuple[int, ...] = ()
    if single_session and components:
        component = max(
            components,
            key=lambda values: (
                sum(marker_observations[marker_id] for marker_id in values),
                len(values),
            ),
        )
        components = [component]
        pending_marker_ids = tuple(sorted(observed_ids - component))

    submaps: list[AutoMarkerSubmap] = []
    for component in components:
        anchor = min(
            component,
            key=lambda marker_id: (-marker_observations[marker_id], marker_id),
        )
        submap_id = "session_000" if single_session else f"submap_{anchor:03d}"
        initial = _initial_marker_poses(component, anchor, edges)
        refined, error = _refine_submap(
            submap_id,
            anchor,
            initial,
            usable_frames,
            calibration,
            dictionary,
            marker_size_m,
        )
        layout = _layout_from_poses(
            submap_id, dictionary, marker_size_m, refined
        )
        submaps.append(
            AutoMarkerSubmap(
                layout.name,
                anchor,
                refined,
                layout,
                tuple(
                    edge
                    for edge in edges
                    if edge.first_id in component and edge.second_id in component
                ),
                error,
            )
        )
    return AutoMarkerMap(
        dictionary,
        marker_size_m,
        tuple(submaps),
        "single-session-marker-map"
        if single_session
        else "reliable-covisibility-local-submaps",
        pending_marker_ids,
    )


def build_session_marker_map(
    detections: list[dict[int, np.ndarray]],
    calibration: Calibration,
    marker_ids: set[int],
    marker_size_m: float,
    dictionary: str = "DICT_4X4_50",
    minimum_covisibility_frames: int = 3,
    minimum_marker_frames: int = 3,
    minimum_area_px2: float = 400.0,
    maximum_single_marker_error_px: float = 5.0,
) -> AutoMarkerMap:
    """Build exactly one persistent map for a recording.

    Only markers connected by reliable co-visibility are registered. Other
    observed markers stay pending instead of creating additional maps.
    """
    return build_auto_marker_map(
        detections,
        calibration,
        marker_ids,
        marker_size_m,
        dictionary,
        minimum_covisibility_frames,
        minimum_marker_frames,
        minimum_area_px2,
        maximum_single_marker_error_px,
        single_session=True,
    )


def localize_auto_marker_frames(
    marker_map: AutoMarkerMap,
    detections: list[dict[int, np.ndarray]],
    calibration: Calibration,
    marker_weights: list[dict[int, float]] | None = None,
    fps: float = 30.0,
    assist_detections: list[dict[int, np.ndarray]] | None = None,
) -> AutoMarkerFrames:
    if marker_weights is not None and len(marker_weights) != len(detections):
        raise ValueError("marker weight and detection streams must have equal length")
    if assist_detections is not None and len(assist_detections) != len(detections):
        raise ValueError("marker assist and detection streams must have equal length")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("marker localization FPS must be positive and finite")
    trackers = {submap.submap_id: MarkerPoseTracker(submap.layout, calibration)
                for submap in marker_map.submaps}
    results: list[TagPoseResult | None] = []
    submap_ids: list[str | None] = []
    active_submap_id = None
    for frame_index, frame in enumerate(detections):
        candidates: list[tuple[tuple[int, float, float], AutoMarkerSubmap, TagPoseResult]] = []
        for submap in marker_map.submaps:
            if not set(frame).intersection(submap.marker_poses):
                continue
            result = trackers[submap.submap_id].update(
                frame, frame_index/fps,
                marker_weights=None if marker_weights is None else marker_weights[frame_index],
                assist_detections=(
                    None if assist_detections is None
                    else assist_detections[frame_index]
                ),
            )
            if result.pose is None:
                continue
            score = (
                len(result.accepted_marker_ids),
                result.confidence,
                -float(result.graph_reprojection_error_px or 0.0),
            )
            candidates.append((score, submap, result))
        if not candidates:
            results.append(None)
            submap_ids.append(None)
            continue
        active = [
            candidate
            for candidate in candidates
            if candidate[1].submap_id == active_submap_id
        ]
        _, submap, result = max(
            active or candidates, key=lambda item: item[0]
        )
        assert result.pose is not None
        active_submap_id = submap.submap_id
        results.append(result)
        submap_ids.append(submap.submap_id)
    return AutoMarkerFrames(results, submap_ids)
