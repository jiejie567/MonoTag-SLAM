from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .models import Calibration, Pose


LANDMARK_NAMES = (
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
)

HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)

_BEND_TRIPLETS = {
    "thumb_cmc": (0, 1, 2),
    "thumb_mcp": (1, 2, 3),
    "thumb_ip": (2, 3, 4),
    "index_mcp": (0, 5, 6),
    "index_pip": (5, 6, 7),
    "index_dip": (6, 7, 8),
    "middle_mcp": (0, 9, 10),
    "middle_pip": (9, 10, 11),
    "middle_dip": (10, 11, 12),
    "ring_mcp": (0, 13, 14),
    "ring_pip": (13, 14, 15),
    "ring_dip": (14, 15, 16),
    "pinky_mcp": (0, 17, 18),
    "pinky_pip": (17, 18, 19),
    "pinky_dip": (18, 19, 20),
}

_HAND_AXIS_LENGTH_M = 0.08
_MAX_WRIST_ANCHOR_AXIS_RATIO = 0.75
_MAX_HAND_AXIS_ERROR_DEG = 70.0


@dataclass(frozen=True)
class RawHandJoints:
    handedness: str
    handedness_score: float | None
    image_landmarks_normalized: np.ndarray
    model_landmarks_m: np.ndarray
    detection_source: str = "full_frame"


def raw_hand_from_dict(joints: dict[str, object]) -> RawHandJoints:
    """Restore an actual cached image/model observation, never world coordinates."""
    image = np.asarray(joints["image_landmarks_normalized"], dtype=np.float64)
    model = np.asarray(joints["model_landmarks_m"], dtype=np.float64)
    if image.shape != (21, 3) or model.shape != (21, 3):
        raise ValueError("cached hand landmarks must have shape (21, 3)")
    if not np.all(np.isfinite(image)) or not np.all(np.isfinite(model)):
        raise ValueError("cached hand landmarks must be finite")
    return RawHandJoints(
        str(joints["handedness"]), (float(joints["handedness_score"])
                                    if joints.get("handedness_score") is not None else None),
        image, model, str(joints.get("detection_source", "full_frame")),
    )


@dataclass(frozen=True)
class HandJointPose:
    name: str
    handedness: str
    handedness_score: float | None
    image_landmarks_normalized: np.ndarray
    model_landmarks_m: np.ndarray
    bend_angles_rad: dict[str, float]
    camera_landmarks_m: np.ndarray | None = None
    band_landmarks_m: np.ndarray | None = None
    world_landmarks_m: np.ndarray | None = None
    wrist_anchor_error_px: float | None = None
    wrist_anchor_axis_ratio: float | None = None
    wrist_axis_error_deg: float | None = None
    wrist_association_confidence: float | None = 0.0
    association_status: str = "unassigned"
    temporal_confirmation_frames: int = 0
    detection_source: str = "full_frame"
    prediction_backend: str = "mediapipe"
    detector_confidence: float | None = None
    temporal_future_frames: int = 0
    image_supported: bool = True
    keypoint_2d_source: str = "landmark_network"


class TemporalHandAssignmentGate:
    """Reject isolated hand/band matches without filling missing measurements."""

    def __init__(self, band_names: list[str], min_consecutive_frames: int = 2):
        self.band_names = tuple(band_names)
        self.min_consecutive_frames = max(1, int(min_consecutive_frames))
        self._streaks = {name: 0 for name in self.band_names}

    def update(self, assignments: dict[str, object]) -> dict[str, object]:
        output = {
            name: value
            for name, value in assignments.items()
            if name not in self._streaks
        }
        for name in self.band_names:
            value = assignments.get(name)
            if value is None:
                self._streaks[name] = 0
                continue
            self._streaks[name] += 1
            if self._streaks[name] >= self.min_consecutive_frames:
                output[name] = value
            else:
                output[f"unassigned_temporal_{name}"] = value
        return output

    def streak(self, name: str) -> int:
        return self._streaks.get(name, 0)


def bend_angles(landmarks: np.ndarray) -> dict[str, float]:
    landmarks = np.asarray(landmarks, dtype=np.float64)
    if landmarks.shape != (21, 3):
        raise ValueError("hand landmarks must have shape (21, 3)")
    result: dict[str, float] = {}
    for name, (parent, joint, child) in _BEND_TRIPLETS.items():
        before = landmarks[joint] - landmarks[parent]
        after = landmarks[child] - landmarks[joint]
        denominator = np.linalg.norm(before) * np.linalg.norm(after)
        if denominator <= 1e-12:
            result[name] = float("nan")
            continue
        cosine = float(np.clip(np.dot(before, after) / denominator, -1.0, 1.0))
        result[name] = float(np.arccos(cosine))
    return result


def bind_landmarks_to_wrist(
    model_landmarks_m: np.ndarray,
    band_pose: Pose | None,
    world_reference: Pose | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if band_pose is None:
        return None, None, None
    landmarks = np.asarray(model_landmarks_m, dtype=np.float64)
    if landmarks.shape != (21, 3):
        raise ValueError("hand landmarks must have shape (21, 3)")

    # MediaPipe world landmarks are metric, camera-axis-aligned vectors around
    # the hand. Anchor landmark 0 to the measured wrist-band origin. A constant
    # anatomical wrist offset can be calibrated later without changing schema.
    camera_vectors = landmarks - landmarks[0]
    camera_landmarks = camera_vectors + band_pose.tvec.reshape(1, 3)
    band_landmarks = (band_pose.rotation_matrix.T @ camera_vectors.T).T
    world_landmarks = None
    if world_reference is not None:
        world_landmarks = (
            world_reference.rotation_matrix.T
            @ (camera_landmarks - world_reference.tvec.reshape(1, 3)).T
        ).T
    return camera_landmarks, band_landmarks, world_landmarks


def band_side(name: str) -> str | None:
    lowered = name.lower()
    if lowered.endswith(("_l", "-l", " left")) or "left" in lowered:
        return "Left"
    if lowered.endswith(("_r", "-r", " right")) or "right" in lowered:
        return "Right"
    return None


def _project_band_origins(
    band_poses: dict[str, Pose], calibration: Calibration
) -> dict[str, np.ndarray]:
    origins: dict[str, np.ndarray] = {}
    for name, pose in band_poses.items():
        projected, _ = cv2.projectPoints(
            np.zeros((1, 3)),
            np.zeros((3, 1)),
            pose.tvec,
            calibration.camera_matrix,
            calibration.dist_coeffs,
        )
        origins[name] = projected.reshape(2)
    return origins


def _hand_band_geometry(
    hand: RawHandJoints,
    band_pose: Pose,
    calibration: Calibration,
) -> tuple[float, float | None, float]:
    """Return wrist distance, palm/band-axis-line angle and projected axis length."""
    projected, _ = cv2.projectPoints(
        np.array([[0.0, 0.0, 0.0], [0.0, -_HAND_AXIS_LENGTH_M, 0.0]]),
        band_pose.rvec,
        band_pose.tvec,
        calibration.camera_matrix,
        calibration.dist_coeffs,
    )
    origin, hand_axis_end = projected.reshape(2, 2)
    wrist = hand.image_landmarks_normalized[0, :2] * calibration.image_size
    palm = (
        hand.image_landmarks_normalized[[5, 9, 13, 17], :2].mean(axis=0)
        * calibration.image_size
    )
    expected = hand_axis_end - origin
    observed = palm - wrist
    projected_axis_length = float(np.linalg.norm(expected))
    denominator = projected_axis_length * float(np.linalg.norm(observed))
    angle = None
    if denominator > 1e-6:
        cosine = float(np.clip(np.dot(expected, observed) / denominator, -1.0, 1.0))
        directed_angle = float(np.degrees(np.arccos(cosine)))
        # The current passive cuff has no keyed proximal/distal mounting: it can
        # be worn with either end toward the hand.  Its projected Y *line* is a
        # useful association cue, but treating one sign as mandatory rejects a
        # geometrically correct nearby hand after the cuff is flipped.
        angle = min(directed_angle, 180.0 - directed_angle)
    return float(np.linalg.norm(wrist - origin)), angle, projected_axis_length


def wrist_anchor_error_px(
    image_landmarks_normalized: np.ndarray,
    band_pose: Pose,
    calibration: Calibration,
) -> float:
    origin = _project_band_origins({"band": band_pose}, calibration)["band"]
    wrist = np.asarray(image_landmarks_normalized, dtype=np.float64)[0, :2].copy()
    wrist *= calibration.image_size
    return float(np.linalg.norm(wrist - origin))


def assign_hands_to_bands(
    hands: list[RawHandJoints],
    band_names: list[str],
    band_poses: dict[str, Pose],
    calibration: Calibration,
) -> dict[str, RawHandJoints]:
    assigned: dict[str, RawHandJoints] = {}
    remaining = set(range(len(hands)))

    # A visible wrist band is a stronger identity cue than model handedness.
    origins = _project_band_origins(band_poses, calibration)
    maximum_distance = 0.12 * float(np.hypot(*calibration.image_size))
    pairs: list[tuple[float, int, str]] = []
    for index in remaining:
        for name in band_names:
            if name not in origins:
                continue
            distance, axis_error, projected_axis_length = _hand_band_geometry(
                hands[index], band_poses[name], calibration
            )
            if projected_axis_length > 1e-6:
                if distance > _MAX_WRIST_ANCHOR_AXIS_RATIO * projected_axis_length:
                    continue
                if axis_error is not None and axis_error > _MAX_HAND_AXIS_ERROR_DEG:
                    continue
            pairs.append((distance, index, name))
    for distance, index, name in sorted(pairs):
        if distance > maximum_distance:
            break
        if index not in remaining or name in assigned:
            continue
        assigned[name] = hands[index]
        remaining.remove(index)

    for index in sorted(remaining):
        assigned[f"unassigned_{index}"] = hands[index]
    return assigned


def joint_pose_to_dict(joints: HandJointPose) -> dict[str, object]:
    def points(value: np.ndarray | None) -> list[list[float]] | None:
        return value.tolist() if value is not None else None

    angles = {
        name: value if np.isfinite(value) else None
        for name, value in joints.bend_angles_rad.items()
    }
    return {
        "valid": True,
        "handedness": joints.handedness,
        "handedness_score": joints.handedness_score,
        "landmark_names": list(LANDMARK_NAMES),
        "image_landmarks_normalized": points(joints.image_landmarks_normalized),
        "model_landmarks_m": points(joints.model_landmarks_m),
        "camera_landmarks_m": points(joints.camera_landmarks_m),
        "band_landmarks_m": points(joints.band_landmarks_m),
        "world_landmarks_m": points(joints.world_landmarks_m),
        "bend_angles_rad": angles,
        "wrist_anchor_valid": joints.band_landmarks_m is not None,
        "wrist_anchor_error_px": joints.wrist_anchor_error_px,
        "wrist_anchor_axis_ratio": joints.wrist_anchor_axis_ratio,
        "wrist_axis_error_deg": joints.wrist_axis_error_deg,
        "wrist_association_confidence": joints.wrist_association_confidence,
        "association_status": joints.association_status,
        "temporal_confirmation_frames": joints.temporal_confirmation_frames,
        "detection_source": joints.detection_source,
        "prediction_backend": joints.prediction_backend,
        "detector_confidence": joints.detector_confidence,
        "temporal_future_frames": joints.temporal_future_frames,
        "image_supported": joints.image_supported,
        "keypoint_2d_source": joints.keypoint_2d_source,
    }


class HandJointTracker:
    def __init__(
        self,
        model_path: str | Path,
        calibration: Calibration,
        band_names: list[str],
        min_confidence: float = 0.4,
        enable_recovery: bool = True,
        initialize_full_frame_detector: bool = True,
    ):
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError("hand joints require: pip install mediapipe") from exc
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"hand landmark model not found: {model_path}")
        self._mp = mp
        self.calibration = calibration
        self.band_names = list(band_names)
        self.max_wrist_anchor_distance_px = 0.12 * float(
            np.hypot(*calibration.image_size)
        )
        self._assignment_gate = TemporalHandAssignmentGate(self.band_names)
        self._last_timestamp_ms = -1
        self._landmarker = None
        if initialize_full_frame_detector:
            options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=min_confidence,
                min_hand_presence_confidence=min_confidence,
                min_tracking_confidence=min_confidence,
            )
            self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        from .hand_recovery import ContinuityAssociation, WristCropDetector, hand_recovery_policy

        self.recovery_policy = hand_recovery_policy(enable_recovery, min_confidence)
        self._continuity = ContinuityAssociation(self.band_names, calibration) if enable_recovery else None
        self._crop_detector = (
            WristCropDetector(model_path, calibration, self.band_names, min_confidence)
            if enable_recovery else None
        )

    def process(
        self,
        frame: np.ndarray,
        timestamp_ms: int,
        band_poses: dict[str, Pose],
        world_reference: Pose | None = None,
    ) -> dict[str, HandJointPose]:
        if self._landmarker is None:
            raise RuntimeError("full-frame detector disabled; use process_observations")
        timestamp_ms = max(int(timestamp_ms), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        hands: list[RawHandJoints] = []
        for handedness, image_points, model_points in zip(
            result.handedness,
            result.hand_landmarks,
            result.hand_world_landmarks,
        ):
            category = handedness[0]
            hands.append(
                RawHandJoints(
                    handedness=category.category_name,
                    handedness_score=float(category.score),
                    image_landmarks_normalized=np.asarray(
                        [[point.x, point.y, point.z] for point in image_points],
                        dtype=np.float64,
                    ),
                    model_landmarks_m=np.asarray(
                        [[point.x, point.y, point.z] for point in model_points],
                        dtype=np.float64,
                    ),
                )
            )
        return self._bind_observations(frame, timestamp_ms, hands, band_poses, world_reference)

    def process_observations(
        self,
        frame: np.ndarray | None,
        timestamp_ms: int,
        raw_hands: list[RawHandJoints],
        band_poses: dict[str, Pose],
        world_reference: Pose | None = None,
        protected_assignments: dict[str, RawHandJoints] | None = None,
    ) -> dict[str, HandJointPose]:
        """Recover cached current-frame detections without whole-frame inference.

        ``protected_assignments`` is exclusively for already confirmed cached
        identities. It preserves their observations, not their old world poses.
        With ``frame=None``, only cached candidates and continuity are used;
        no ROI detector is initialized or run.
        """
        timestamp_ms = max(int(timestamp_ms), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        return self._bind_observations(
            frame, timestamp_ms, list(raw_hands), band_poses, world_reference,
            protected_assignments,
        )

    def _bind_observations(
        self, frame, timestamp_ms, hands, band_poses, world_reference,
        protected_assignments=None,
    ) -> dict[str, HandJointPose]:
        protected_assignments = protected_assignments or {}
        recovered_names = set()
        if self._continuity is not None:
            from .hand_recovery import supplement_assignments

            geometric_assignments = self._continuity.assign(
                hands, band_poses, timestamp_ms / 1000.0, protected_assignments
            )
            recovered_names = self._continuity.recovered
            # Continuity is cheap and runs first; ROI only fills remaining slots.
            extra = (
                self._crop_detector.detect(frame, hands, band_poses, geometric_assignments)
                if frame is not None else []
            )
            geometric_assignments = supplement_assignments(
                geometric_assignments, hands, extra, self.band_names, band_poses, self.calibration
            )
        else:
            from .hand_recovery import ContinuityAssociation

            # Keep the cache-protection contract even when recovery is disabled.
            geometric_assignments = ContinuityAssociation(self.band_names, self.calibration).assign(
                hands, band_poses, timestamp_ms / 1000.0, protected_assignments
            )
        band_by_hand = {
            id(hand): name
            for name, hand in geometric_assignments.items()
            if name in self.band_names
        }
        assigned = self._assignment_gate.update(geometric_assignments)
        if protected_assignments:
            protected_ids = {id(hand) for hand in protected_assignments.values()}
            assigned = {
                name: hand for name, hand in assigned.items()
                if name not in protected_assignments and id(hand) not in protected_ids
            }
            assigned.update(protected_assignments)
        if self._continuity is not None:
            self._continuity.observe(assigned, band_poses, timestamp_ms / 1000.0)
        output: dict[str, HandJointPose] = {}
        for name, hand in assigned.items():
            matched_band = band_by_hand.get(id(hand))
            accepted = name in self.band_names
            band_pose = band_poses.get(name) if accepted else None
            anchor_error = None
            anchor_axis_ratio = None
            axis_error = None
            association_confidence = 0.0
            if matched_band is not None and matched_band in band_poses:
                distance, axis_error, projected_axis_length = _hand_band_geometry(
                    hand, band_poses[matched_band], self.calibration
                )
                anchor_error = distance
                if projected_axis_length > 1e-6:
                    anchor_axis_ratio = distance / projected_axis_length
                    distance_score = np.clip(
                        1.0 - anchor_axis_ratio / _MAX_WRIST_ANCHOR_AXIS_RATIO,
                        0.0,
                        1.0,
                    )
                    direction_score = (
                        0.5
                        if axis_error is None
                        else np.clip(1.0 - axis_error / _MAX_HAND_AXIS_ERROR_DEG, 0.0, 1.0)
                    )
                    association_confidence = float(
                        (hand.handedness_score if hand.handedness_score is not None else 0.0)
                        * 0.5 * (distance_score + direction_score)
                    )
            if band_pose is not None:
                if anchor_error > self.max_wrist_anchor_distance_px:
                    band_pose = None
            camera_points, band_points, world_points = bind_landmarks_to_wrist(
                hand.model_landmarks_m,
                band_pose,
                world_reference,
            )
            output[name] = HandJointPose(
                name=name,
                handedness=hand.handedness,
                handedness_score=hand.handedness_score,
                image_landmarks_normalized=hand.image_landmarks_normalized,
                model_landmarks_m=hand.model_landmarks_m,
                bend_angles_rad=bend_angles(hand.model_landmarks_m),
                camera_landmarks_m=camera_points,
                band_landmarks_m=band_points,
                world_landmarks_m=world_points,
                wrist_anchor_error_px=anchor_error,
                wrist_anchor_axis_ratio=anchor_axis_ratio,
                wrist_axis_error_deg=axis_error,
                wrist_association_confidence=association_confidence,
                association_status=(
                    "confirmed_roi"
                    if accepted and hand.detection_source == "wrist_roi"
                    else "confirmed_temporal"
                    if accepted and name in recovered_names
                    else "confirmed"
                    if accepted
                    else "temporal_unconfirmed"
                    if matched_band is not None
                    else "geometry_rejected"
                ),
                temporal_confirmation_frames=(
                    max(self._assignment_gate.streak(matched_band), 2 if name in protected_assignments else 0)
                    if matched_band is not None
                    else 0
                ),
                detection_source=hand.detection_source,
            )
        return output

    @property
    def recovery_diagnostics(self) -> dict[str, object]:
        return {
            "policy": dict(self.recovery_policy),
            "roi_inference_calls": self._crop_detector.calls if self._crop_detector is not None else 0,
        }

    def close(self) -> None:
        try:
            if self._landmarker is not None:
                self._landmarker.close()
        finally:
            if self._crop_detector is not None:
                self._crop_detector.close()

    def __enter__(self) -> "HandJointTracker":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
