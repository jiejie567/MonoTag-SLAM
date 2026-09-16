from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from .hands import HAND_CONNECTIONS, HandJointPose
from .models import Calibration, Pose
from .pipeline import FrameResult


LEFT_COLOR = (30, 220, 30)
RIGHT_COLOR = (0, 0, 255)
COLORS = [LEFT_COLOR, RIGHT_COLOR, (30, 160, 255), (255, 120, 30)]


def pose_color(name: str, fallback_index: int = 0) -> tuple[int, int, int]:
    if name.endswith("_L"):
        return LEFT_COLOR
    if name.endswith("_R"):
        return RIGHT_COLOR
    if name.startswith("marker-"):
        marker_id = int(name.removeprefix("marker-"))
        return LEFT_COLOR if marker_id < 6 else RIGHT_COLOR
    return COLORS[fallback_index % len(COLORS)]


class FadingTrajectory:
    def __init__(
        self,
        pose_names: list[str],
        max_points: int = 120,
        smoothing_alpha: float = 0.3,
        world_hold_frames: int = 5,
        world_only: bool = False,
        world_smoothing_alpha: float = 1.0,
        maximum_world_step_m: float | None = None,
    ):
        self.pose_names = tuple(pose_names)
        self._points = {name: deque(maxlen=max_points) for name in pose_names}
        self._world_points = {name: deque(maxlen=max_points) for name in pose_names}
        self._last_world_projection = {name: None for name in pose_names}
        self._smoothed = {name: None for name in pose_names}
        self._axis_points = {name: None for name in pose_names}
        self._axis_filter_points = {name: None for name in pose_names}
        self.mode: str | None = None
        self._world_gap_frames = 0
        self.smoothing_alpha = smoothing_alpha
        self.world_hold_frames = world_hold_frames
        self.world_only = world_only
        self.world_smoothing_alpha = world_smoothing_alpha
        self.maximum_world_step_m = maximum_world_step_m

    def reset(self) -> None:
        for histories in (self._points, self._world_points):
            for history in histories.values():
                history.clear()
        self._last_world_projection = {name: None for name in self.pose_names}
        self._smoothed = {name: None for name in self.pose_names}
        self._axis_points = {name: None for name in self.pose_names}
        self._axis_filter_points = {name: None for name in self.pose_names}
        self.mode = None
        self._world_gap_frames = 0

    def draw(self, output: np.ndarray, result: FrameResult, calibration: Calibration) -> None:
        if result.world_reference is not None:
            if self.mode == "camera":
                self._append_break(self._points)
            self._world_gap_frames = 0
            self.mode = "world"
            self._draw_world(output, result, calibration)
            return

        self._world_gap_frames += 1
        if (
            self.mode in {"world", "world_hold"}
            and self._world_gap_frames <= self.world_hold_frames
        ):
            self.mode = "world_hold"
            self._draw_world_hold(output)
            return

        if self.world_only:
            if self.mode != "world_unavailable":
                self._append_break(self._world_points)
            self.mode = "world_unavailable"
            return

        if self.mode != "camera":
            self._append_break(self._world_points)
            self._append_break(self._points)
            self._smoothed = {name: None for name in self._smoothed}
        self.mode = "camera"
        self._draw_camera(output, result, calibration)

    @staticmethod
    def _append_break(histories: dict[str, deque]) -> None:
        for history in histories.values():
            if history and history[-1] is not None:
                history.append(None)

    def _draw_camera(
        self, output: np.ndarray, result: FrameResult, calibration: Calibration
    ) -> None:
        for index, (name, history) in enumerate(self._points.items()):
            pose = result.poses.get(name)
            point = None
            if pose is not None:
                projected, _ = cv2.projectPoints(
                    np.zeros((1, 3)),
                    pose.rvec * 0.0,
                    pose.tvec,
                    calibration.camera_matrix,
                    calibration.dist_coeffs,
                )
                projected = projected.reshape(2)
                previous = self._smoothed[name]
                if previous is not None:
                    projected = (
                        (1.0 - self.smoothing_alpha) * previous
                        + self.smoothing_alpha * projected
                    )
                self._smoothed[name] = projected
                point = tuple(np.rint(projected).astype(int))
            elif history and history[-1] is not None:
                point = history[-1]
            history.append(point)
            display_history = deque(history, maxlen=history.maxlen)
            raw_pose = result.raw_poses.get(name)
            if raw_pose is not None:
                display_history[-1] = self._project_pose_origin(
                    raw_pose, calibration
                )
            self._draw_history(output, display_history, pose_color(name, index))

    def _draw_world(
        self, output: np.ndarray, result: FrameResult, calibration: Calibration
    ) -> None:
        reference = result.world_reference
        if reference is None:
            return
        for name, history in self._world_points.items():
            pose = result.world_poses.get(name)
            point = pose.tvec.reshape(3).copy() if pose is not None else None
            previous = history[-1] if history and history[-1] is not None else None
            if point is not None and previous is not None:
                step = float(np.linalg.norm(point - previous))
                if (
                    self.maximum_world_step_m is not None
                    and step > self.maximum_world_step_m
                ):
                    point = previous.copy()
                else:
                    point = (
                        (1.0 - self.world_smoothing_alpha) * previous
                        + self.world_smoothing_alpha * point
                    )
            if point is None and history and history[-1] is not None:
                point = history[-1].copy()
            history.append(point)

        if self.world_only:
            self._draw_world_inset(output)
            return

        reference_rotation = reference.rotation_matrix
        reference_translation = reference.tvec.reshape(3)
        for index, (name, history) in enumerate(self._world_points.items()):
            projected_points: list[tuple[int, int] | None] = [None] * len(history)
            valid = [
                (point_index, world_point)
                for point_index, world_point in enumerate(history)
                if world_point is not None
                and np.all(np.isfinite(world_point))
                and (
                    reference_rotation @ world_point.reshape(3)
                    + reference_translation
                )[2]
                > 0.02
            ]
            if valid:
                projected, _ = cv2.projectPoints(
                    np.stack([world_point for _, world_point in valid]),
                    reference.rvec,
                    reference.tvec,
                    calibration.camera_matrix,
                    calibration.dist_coeffs,
                )
                for (point_index, _), image_point in zip(valid, projected.reshape(-1, 2)):
                    if np.all(np.isfinite(image_point)) and np.max(np.abs(image_point)) < 1e6:
                        projected_points[point_index] = tuple(
                            int(value) for value in np.rint(image_point)
                        )
            projected_history = deque(projected_points, maxlen=history.maxlen)
            raw_pose = result.raw_poses.get(name)
            if raw_pose is not None:
                projected_history[-1] = self._project_pose_origin(
                    raw_pose, calibration
                )
            self._last_world_projection[name] = projected_history
            self._draw_history(output, projected_history, pose_color(name, index))

    @staticmethod
    def _project_pose_origin(
        pose: Pose, calibration: Calibration
    ) -> tuple[int, int]:
        projected, _ = cv2.projectPoints(
            np.zeros((1, 3)),
            np.zeros((3, 1)),
            pose.tvec,
            calibration.camera_matrix,
            calibration.dist_coeffs,
        )
        return tuple(np.rint(projected.reshape(2)).astype(int))

    def _draw_world_hold(self, output: np.ndarray) -> None:
        if self.world_only:
            self._draw_world_inset(output)
            return
        for index, name in enumerate(self.pose_names):
            projected_history = self._last_world_projection[name]
            if projected_history is not None:
                self._draw_history(output, projected_history, pose_color(name, index))

    def _draw_world_inset(self, output: np.ndarray) -> None:
        height, width = output.shape[:2]
        panel_width = min(280, max(180, width // 4))
        panel_height = min(240, max(160, height // 4))
        x0 = width - panel_width - 16
        y0 = 52
        x1 = x0 + panel_width
        y1 = y0 + panel_height
        overlay = output.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (18, 18, 18), -1)
        output[:] = cv2.addWeighted(overlay, 0.82, output, 0.18, 0.0)
        cv2.rectangle(output, (x0, y0), (x1, y1), (180, 180, 180), 1)

        origin = (x0 + panel_width // 2, y0 + panel_height // 2 + 8)
        half_range_m = 0.30
        scale = 0.43 * min(panel_width, panel_height) / half_range_m
        cv2.line(output, (x0 + 8, origin[1]), (x1 - 8, origin[1]), (90, 90, 90), 1)
        cv2.line(output, (origin[0], y0 + 27), (origin[0], y1 - 8), (90, 90, 90), 1)
        cv2.circle(output, origin, 3, (230, 230, 230), -1, cv2.LINE_AA)
        cv2.putText(
            output,
            "WORLD XY  fixed  +/-0.30 m",
            (x0 + 8, y0 + 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(output, "+X", (x1 - 30, origin[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 170, 170), 1)
        cv2.putText(output, "+Y", (origin[0] + 5, y0 + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 170, 170), 1)

        for index, (name, history) in enumerate(self._world_points.items()):
            screen_points = deque(maxlen=history.maxlen)
            for point in history:
                if point is None:
                    screen_points.append(None)
                    continue
                screen_points.append(
                    (
                        int(round(origin[0] + point[0] * scale)),
                        int(round(origin[1] - point[1] * scale)),
                    )
                )
            self._draw_history(output, screen_points, pose_color(name, index))

    def draw_world_inset(self, output: np.ndarray) -> None:
        self._draw_world_inset(output)

    def draw_pose_axes(
        self,
        output: np.ndarray,
        name: str,
        calibration: Calibration,
        pose: Pose | None,
        raw_pose: Pose | None,
        length: float,
        thickness: int,
    ) -> bool:
        if pose is None:
            self._axis_points[name] = None
            self._axis_filter_points[name] = None
            return False

        filtered_candidate = project_frame_axes(
            calibration, pose.rvec, pose.tvec, length
        )
        candidate = project_frame_axes(
            calibration,
            pose.rvec,
            raw_pose.tvec if raw_pose is not None else pose.tvec,
            length,
        )
        if candidate is None:
            return False
        previous = self._axis_filter_points[name]
        if (
            previous is not None
            and raw_pose is not None
            and raw_pose.ambiguous
            and filtered_candidate is not None
        ):
            threshold = 0.08 * output.shape[1]
            raw_filter_delta = np.max(
                np.linalg.norm(candidate - filtered_candidate, axis=1)
            )
            raw_previous_delta = np.max(
                np.linalg.norm(candidate - previous, axis=1)
            )
            if raw_filter_delta > threshold and raw_previous_delta > threshold:
                self._axis_points[name] = None
                return False
        if previous is None:
            displayed = candidate
        else:
            candidate_vectors = candidate[1:] - candidate[0]
            previous_vectors = previous[1:] - previous[0]
            motion = float(
                np.median(
                    np.linalg.norm(candidate_vectors - previous_vectors, axis=1)
                )
            )
            motion_scale = 0.03125 * output.shape[1]
            alpha = float(np.clip(0.18 + 0.72 * motion / motion_scale, 0.18, 0.90))
            displayed = candidate.copy()
            displayed[1:] = candidate[0] + (
                (1.0 - alpha) * previous_vectors + alpha * candidate_vectors
            )
        self._axis_filter_points[name] = displayed
        self._axis_points[name] = displayed
        draw_projected_axes(output, displayed, thickness)
        return True

    @staticmethod
    def _draw_history(
        output: np.ndarray, history: deque[tuple[int, int] | None], color: tuple[int, int, int]
    ) -> None:
        points = list(history)
        if len(points) < 2:
            return
        denominator = max(1, len(points) - 1)
        height, width = output.shape[:2]
        clipped_segments: list[
            tuple[tuple[int, int], tuple[int, int], float]
        ] = []
        for segment in range(1, len(points)):
            start, end = points[segment - 1], points[segment]
            if start is None or end is None:
                continue
            age = segment / denominator
            alpha = 0.05 + 0.80 * age * age
            visible, clipped_start, clipped_end = cv2.clipLine(
                (0, 0, width, height), start, end
            )
            if visible:
                clipped_segments.append((clipped_start, clipped_end, alpha))
        if not clipped_segments:
            if points[-1] is not None:
                cv2.circle(output, points[-1], 4, color, -1, cv2.LINE_AA)
            return
        segment_points = np.asarray(
            [point for start, end, _ in clipped_segments for point in (start, end)]
        )
        padding = 5
        x0 = max(0, int(np.min(segment_points[:, 0])) - padding)
        y0 = max(0, int(np.min(segment_points[:, 1])) - padding)
        x1 = min(width, int(np.max(segment_points[:, 0])) + padding + 1)
        y1 = min(height, int(np.max(segment_points[:, 1])) + padding + 1)
        layer = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8)
        alpha_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        for start, end, alpha in clipped_segments:
            local_start = (start[0] - x0, start[1] - y0)
            local_end = (end[0] - x0, end[1] - y0)
            cv2.line(layer, local_start, local_end, color, 3, cv2.LINE_AA)
            cv2.line(
                alpha_mask,
                local_start,
                local_end,
                round(255 * alpha),
                3,
                cv2.LINE_AA,
            )
        alpha = alpha_mask.astype(np.float32)[..., None] / 255.0
        region = output[y0:y1, x0:x1]
        region[:] = np.rint(region * (1.0 - alpha) + layer * alpha).astype(np.uint8)
        if points[-1] is not None:
            cv2.circle(output, points[-1], 4, color, -1, cv2.LINE_AA)


def project_frame_axes(
    calibration: Calibration,
    rvec: np.ndarray,
    tvec: np.ndarray,
    length: float,
) -> np.ndarray | None:
    axes = np.array(
        [[0.0, 0.0, 0.0], [length, 0.0, 0.0], [0.0, length, 0.0], [0.0, 0.0, length]]
    )
    projected, _ = cv2.projectPoints(
        axes,
        rvec,
        tvec,
        calibration.camera_matrix,
        calibration.dist_coeffs,
    )
    projected = projected.reshape(-1, 2)
    if not np.all(np.isfinite(projected)):
        return None
    return np.clip(projected, -1e6, 1e6)


def draw_projected_axes(
    output: np.ndarray, projected: np.ndarray, thickness: int
) -> None:
    projected = np.rint(projected).astype(int)
    origin = tuple(projected[0])
    rectangle = (0, 0, output.shape[1], output.shape[0])
    colors = ((0, 0, 255), (0, 255, 0), (255, 0, 0))
    for endpoint, color in zip(projected[1:], colors):
        visible, start, end = cv2.clipLine(rectangle, origin, tuple(endpoint))
        if visible:
            cv2.line(output, start, end, color, thickness, cv2.LINE_AA)


def draw_frame_axes(
    output: np.ndarray,
    calibration: Calibration,
    rvec: np.ndarray,
    tvec: np.ndarray,
    length: float,
    thickness: int,
) -> None:
    projected = project_frame_axes(calibration, rvec, tvec, length)
    if projected is not None:
        draw_projected_axes(output, projected, thickness)


def draw_hand_joints(
    output: np.ndarray, hand_joints: dict[str, HandJointPose]
) -> None:
    height, width = output.shape[:2]
    for index, (name, joints) in enumerate(hand_joints.items()):
        if name.startswith("unassigned_"):
            continue
        color = pose_color(name, index)
        points = joints.image_landmarks_normalized[:, :2] * (width, height)
        points = np.rint(points).astype(np.int32)
        for start, end in HAND_CONNECTIONS:
            visible, clipped_start, clipped_end = cv2.clipLine(
                (0, 0, width, height), tuple(points[start]), tuple(points[end])
            )
            if visible:
                cv2.line(output, clipped_start, clipped_end, color, 2, cv2.LINE_AA)
        for point in points:
            if 0 <= point[0] < width and 0 <= point[1] < height:
                cv2.circle(output, tuple(point), 3, color, -1, cv2.LINE_AA)
        wrist = points[0]
        anchor = "3D" if joints.band_landmarks_m is not None else "2D"
        cv2.putText(
            output,
            f"{name} JOINTS {anchor} " + (f"{joints.handedness_score:.2f}"
                if joints.handedness_score is not None else "score n/a"),
            (int(wrist[0]) + 8, int(wrist[1]) + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )


def draw_rejected_marker_boundaries(
    image: np.ndarray, rejected: dict[int, np.ndarray], reasons: dict[int, str] | None = None,
) -> None:
    for marker_id, corners in rejected.items():
        polygon = np.rint(corners).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(image, [polygon], True, (0, 140, 255), 2, cv2.LINE_AA)
        x, y = polygon[0, 0]
        reason = (reasons or {}).get(marker_id, "boundary").replace("_", " ").upper()
        cv2.putText(
            image, f"ID {marker_id} REJECT: {reason}", (int(x), int(y) - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1, cv2.LINE_AA,
        )


def draw_soft_marker(image: np.ndarray, marker_id: int, corners: np.ndarray, weight: float) -> None:
    polygon = np.rint(corners).astype(np.int32).reshape(-1, 1, 2)
    color = (255, 220, 0)  # Cyan means retained with reduced information, not rejected.
    cv2.polylines(image, [polygon], True, color, 2, cv2.LINE_AA)
    x, y = polygon[0, 0]
    cv2.putText(image, f"ID {marker_id} SOFT {weight:.0%}", (int(x), int(y) - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def draw_result(
    frame: np.ndarray,
    result: FrameResult,
    calibration: Calibration,
    trajectory: FadingTrajectory | None = None,
    hand_joints: dict[str, HandJointPose] | None = None,
) -> np.ndarray:
    output = frame.copy()
    draw_rejected_marker_boundaries(output, result.rejected_detections,
                                    {mid: quality.reason for mid, quality in result.boundary_quality.items()})
    recovered_ids = set(result.recovered_ids)
    tracked_ids = set(result.tracked_ids)
    for marker_id, corners in result.detections.items():
        polygon = np.rint(corners).astype(np.int32).reshape(-1, 1, 2)
        recovered = marker_id in recovered_ids
        tracked = marker_id in tracked_ids
        marker_color = (
            (255, 220, 0)
            if tracked
            else (0, 220, 255) if recovered else (LEFT_COLOR if marker_id < 6 else RIGHT_COLOR)
        )
        cv2.polylines(output, [polygon], True, marker_color, 2 if recovered or tracked else 1, cv2.LINE_AA)
        x, y = polygon[0, 0]
        label = f"{marker_id} T" if tracked else f"{marker_id} R" if recovered else str(marker_id)
        cv2.putText(output, label, (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, marker_color, 1)
        quality = result.boundary_quality.get(marker_id)
        if quality is not None and quality.reason == "soft_grid":
            draw_soft_marker(output, marker_id, corners, quality.information_weight)
    if trajectory is not None:
        trajectory.draw(output, result, calibration)
    if result.world_reference is not None:
        draw_frame_axes(
            output,
            calibration,
            result.world_reference.rvec,
            result.world_reference.tvec,
            0.04,
            2,
        )
    rendered_trajectory_axes: set[str] = set()
    for index, (name, pose) in enumerate(result.poses.items()):
        color = pose_color(name, index)
        is_marker = name.startswith("marker-")
        world_pose = result.world_poses.get(name)
        # The hand axes are an image overlay, so use the directly solved
        # camera-frame hand pose.  Re-composing a filtered world pose with a
        # noisy current board pose makes the axes drift away from the wrist.
        display_pose = pose
        is_trajectory_pose = trajectory is not None and name in trajectory.pose_names
        if is_trajectory_pose:
            trajectory.draw_pose_axes(
                output,
                name,
                calibration,
                display_pose,
                result.raw_poses.get(name),
                0.012,
                2,
            )
            rendered_trajectory_axes.add(name)
        else:
            draw_frame_axes(
                output,
                calibration,
                display_pose.rvec,
                display_pose.tvec,
                0.006 if is_marker else 0.012,
                1 if is_marker else 2,
            )
        t = display_pose.tvec.reshape(3)
        warning = " AMBIGUOUS" if pose.ambiguous else ""
        world_label = ""
        if world_pose is not None:
            world_t = world_pose.tvec.reshape(3)
            world_label = (
                f"  world=({world_t[0]:+.3f}, {world_t[1]:+.3f}, "
                f"{world_t[2]:+.3f}) m"
            )
        label = (
            f"{name}: camera=({t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f}) m"
            f"{world_label}  err={pose.reprojection_error_px:.2f}px{warning}"
        )
        cv2.putText(output, label, (16, 30 + 26 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)
    if trajectory is not None:
        for name in trajectory.pose_names:
            if name not in rendered_trajectory_axes:
                trajectory.draw_pose_axes(
                    output, name, calibration, None, None, 0.012, 2
                )
    if hand_joints:
        draw_hand_joints(output, hand_joints)
    if result.world_reference is not None:
        reference = result.world_reference
        label = (
            f"WORLD LOCK  ids={len(reference.marker_ids)}  "
            f"err={reference.reprojection_error_px:.2f}px"
        )
        color = (0, 220, 255)
    else:
        label = "WORLD UNAVAILABLE"
        color = (0, 0, 255)
    cv2.putText(
        output,
        label,
        (16, output.shape[0] - 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        color,
        2,
        cv2.LINE_AA,
    )
    if trajectory is not None:
        if trajectory.mode == "world":
            trajectory_label = (
                "TRAJECTORY: WORLD XY INSET"
                if trajectory.world_only
                else "TRAJECTORY: WORLD FRAME"
            )
            trajectory_color = (0, 220, 255)
        elif trajectory.mode == "world_hold":
            trajectory_label = "TRAJECTORY: WORLD HOLD"
            trajectory_color = (0, 200, 255)
        elif trajectory.mode == "world_unavailable":
            trajectory_label = "TRAJECTORY: WORLD FRAME (UNAVAILABLE)"
            trajectory_color = (0, 0, 255)
        else:
            trajectory_label = "TRAJECTORY: CAMERA FRAME"
            trajectory_color = (0, 180, 255)
        cv2.putText(
            output,
            trajectory_label,
            (16, output.shape[0] - 74),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            trajectory_color,
            2,
            cv2.LINE_AA,
        )
    return output
