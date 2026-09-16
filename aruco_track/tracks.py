from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from .models import Pose


def _matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    m = matrix
    w = np.sqrt(max(0.0, 1.0 + np.trace(m))) / 2.0
    x = np.copysign(np.sqrt(max(0.0, 1.0 + m[0, 0] - m[1, 1] - m[2, 2])) / 2.0, m[2, 1] - m[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1.0 - m[0, 0] + m[1, 1] - m[2, 2])) / 2.0, m[0, 2] - m[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1.0 - m[0, 0] - m[1, 1] + m[2, 2])) / 2.0, m[1, 0] - m[0, 1])
    return np.array([w, x, y, z], dtype=np.float64)


def _quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


class PoseSmoother:
    def __init__(self, translation_alpha: float = 0.35, rotation_alpha: float = 0.3):
        self.translation_alpha = translation_alpha
        self.rotation_alpha = rotation_alpha
        self._pose: Pose | None = None

    def update(self, pose: Pose) -> Pose:
        if self._pose is None:
            self._pose = pose
            return pose
        tvec = (1.0 - self.translation_alpha) * self._pose.tvec + self.translation_alpha * pose.tvec
        old_q = _matrix_to_quaternion(cv2.Rodrigues(self._pose.rvec)[0])
        new_q = _matrix_to_quaternion(cv2.Rodrigues(pose.rvec)[0])
        if np.dot(old_q, new_q) < 0:
            new_q = -new_q
        q = (1.0 - self.rotation_alpha) * old_q + self.rotation_alpha * new_q
        q /= np.linalg.norm(q)
        rvec = cv2.Rodrigues(_quaternion_to_matrix(q))[0]
        self._pose = Pose(rvec, tvec, pose.reprojection_error_px, pose.marker_ids, pose.inlier_count, pose.ambiguous)
        return self._pose


class AdaptivePoseSmoother(PoseSmoother):
    def __init__(self):
        super().__init__()
        self._previous_raw: Pose | None = None
        self._translation_velocity = np.zeros(3, dtype=np.float64)
        self._rotation_speed = 0.0

    @staticmethod
    def _alpha(cutoff_hz: float) -> float:
        sample_period = 1.0 / 30.0
        return 1.0 / (1.0 + 1.0 / (2.0 * np.pi * cutoff_hz * sample_period))

    def update(self, pose: Pose) -> Pose:
        if self._previous_raw is None:
            self._previous_raw = pose
            return super().update(pose)
        sample_period = 1.0 / 30.0
        derivative_alpha = self._alpha(1.0)
        translation_velocity = (
            pose.tvec.reshape(3) - self._previous_raw.tvec.reshape(3)
        ) / sample_period
        self._translation_velocity = (
            (1.0 - derivative_alpha) * self._translation_velocity
            + derivative_alpha * translation_velocity
        )
        previous_rotation = cv2.Rodrigues(self._previous_raw.rvec)[0]
        current_rotation = cv2.Rodrigues(pose.rvec)[0]
        cosine = np.clip((np.trace(previous_rotation.T @ current_rotation) - 1.0) / 2.0, -1.0, 1.0)
        rotation_speed = np.arccos(cosine) / sample_period
        self._rotation_speed = (
            (1.0 - derivative_alpha) * self._rotation_speed
            + derivative_alpha * rotation_speed
        )
        confidence = 0.58 if pose.ambiguous else 1.0
        confidence *= np.clip(2.0 / max(pose.reprojection_error_px, 1e-6), 0.5, 1.0)
        confidence_gain = 0.7 + 0.3 * confidence
        translation_cutoff = 0.6 + 30.0 * np.linalg.norm(self._translation_velocity)
        rotation_cutoff = 0.3 + 1.5 * self._rotation_speed
        self.translation_alpha = self._alpha(translation_cutoff) * confidence_gain
        self.rotation_alpha = self._alpha(rotation_cutoff) * confidence_gain
        self._previous_raw = pose
        return super().update(pose)


class WorldPoseSmoother(PoseSmoother):
    def __init__(self):
        super().__init__()
        self._measurements: deque[Pose] = deque(maxlen=5)

    def update(self, pose: Pose) -> Pose:
        self._measurements.append(pose)
        translations = np.stack(
            [measurement.tvec.reshape(3) for measurement in self._measurements]
        )
        translation = np.median(translations, axis=0).reshape(3, 1)
        quaternions = np.stack(
            [
                _matrix_to_quaternion(measurement.rotation_matrix)
                for measurement in self._measurements
            ]
        )
        quaternions[np.sum(quaternions * quaternions[0], axis=1) < 0] *= -1
        quaternion = np.mean(quaternions, axis=0)
        quaternion /= np.linalg.norm(quaternion)
        candidate = Pose(
            cv2.Rodrigues(_quaternion_to_matrix(quaternion))[0],
            translation,
            pose.reprojection_error_px,
            pose.marker_ids,
            pose.inlier_count,
            pose.ambiguous,
        )
        if self._pose is None:
            return super().update(candidate)

        translation_delta = np.linalg.norm(candidate.tvec - self._pose.tvec)
        rotation_delta = self._pose.rotation_matrix.T @ candidate.rotation_matrix
        rotation_angle = np.linalg.norm(cv2.Rodrigues(rotation_delta)[0])
        confidence = 0.5 if pose.ambiguous else 1.0
        confidence *= np.clip(
            2.0 / max(pose.reprojection_error_px, 1e-6), 0.5, 1.0
        )
        self.translation_alpha = (
            0.0
            if translation_delta <= 0.002
            else min(0.65, 0.10 + 25.0 * (translation_delta - 0.002))
            * confidence
        )
        rotation_deadband = np.deg2rad(0.75)
        self.rotation_alpha = (
            0.0
            if rotation_angle <= rotation_deadband
            else min(0.55, 0.08 + 4.0 * (rotation_angle - rotation_deadband))
            * confidence
        )
        return super().update(candidate)
