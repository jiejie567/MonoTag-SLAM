"""Conservative current-frame hand recovery; never synthesize missing joints."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .hands import RawHandJoints, _project_band_origins, assign_hands_to_bands


HAND_RECOVERY_VERSION = "wrist-roi-continuity-v1"
PALM = [0, 5, 9, 13, 17]


def hand_recovery_policy(enabled: bool = True, min_confidence: float = 0.4) -> dict:
    return {
        "version": HAND_RECOVERY_VERSION,
        "enabled": bool(enabled),
        "priority": ["original_geometry", "confirmed_continuity", "missing_wrist_roi"],
        "continuity_max_age_s": 0.15,
        "temporal_confirmation_frames": 2,
        "min_confidence": float(min_confidence),
        "current_frame_measurements_only": True,
        "interpolated_training_labels": False,
    }


def supplement_assignments(assigned, hands, extra, names, poses, calibration):
    """Only fill empty identities, keeping the original measurement objects."""
    kept = {name: hand for name, hand in assigned.items() if name in names}
    missing = [name for name in names if name not in kept]
    used = {id(hand) for hand in kept.values()}
    proposed = assign_hands_to_bands(
        [hand for hand in extra if id(hand) not in used], missing, poses, calibration
    )
    for name in missing:
        hand = proposed.get(name)
        if hand is not None and id(hand) not in used:
            kept[name] = hand
            used.add(id(hand))
    unassigned = {
        f"unassigned_{index}": hand
        for index, hand in enumerate(hands + extra)
        if id(hand) not in used
    }
    return {**kept, **unassigned}


class ContinuityAssociation:
    """Relax only the axis association for a recently confirmed current hand."""

    def __init__(self, names, calibration, max_age_s=0.15):
        self.names = tuple(names)
        self.calibration = calibration
        self.max_age_s = float(max_age_s)
        self.previous = {}
        self.recovered = set()

    def assign(self, hands, poses, timestamp, protected_assignments=None):
        assigned = assign_hands_to_bands(hands, self.names, poses, self.calibration)
        if protected_assignments:
            protected_ids = {id(hand) for hand in protected_assignments.values()}
            known_ids = {id(hand) for hand in hands}
            if (not set(protected_assignments).issubset(self.names)
                    or not protected_ids.issubset(known_ids)
                    or len(protected_ids) != len(protected_assignments)):
                raise ValueError("protected hand assignments must be unique current observations")
            assigned = {
                name: hand for name, hand in assigned.items()
                if name not in protected_assignments and id(hand) not in protected_ids
            }
            assigned.update(protected_assignments)
        self.recovered = set()
        used = {id(hand) for name, hand in assigned.items() if name in self.names}
        remaining = [hand for hand in hands if id(hand) not in used]
        origins = _project_band_origins(poses, self.calibration)
        options = []
        for name in self.names:
            if name in assigned or name not in poses or name not in self.previous:
                continue
            old = self.previous[name]
            if not np.isfinite(timestamp) or not 0 <= timestamp - old["time"] <= self.max_age_s:
                continue
            z = float(poses[name].tvec[2, 0])
            if not np.isfinite(z) or z <= 0:
                continue
            scale = old["z"] / z
            if not 0.7 <= scale <= 1.4:
                continue
            predicted = (old["points"] - old["origin"]) * scale + origins[name]
            length = max(20.0, float(np.linalg.norm(predicted[0] - predicted[1:].mean(axis=0))))
            for hand in remaining:
                pixels = hand.image_landmarks_normalized[PALM, :2] * self.calibration.image_size
                if not np.all(np.isfinite(pixels)):
                    continue
                distance = np.linalg.norm(pixels[0] - origins[name])
                bound = min(
                    0.12 * np.hypot(*self.calibration.image_size),
                    0.12 * self.calibration.camera_matrix[0, 0] / z,
                )
                error = float(np.max(np.linalg.norm(pixels - predicted, axis=1)) / length)
                if distance <= bound and error <= 0.35:
                    options.append((error, name, hand))
        options.sort(key=lambda item: item[0])
        for error, name, hand in options:
            if name in assigned or id(hand) in used:
                continue
            competitors = [
                value for value, other_name, other_hand in options
                if (other_name == name or other_hand is hand)
                and (other_name != name or other_hand is not hand)
            ]
            if competitors and min(competitors) - error < 0.15:
                continue
            assigned[name] = hand
            used.add(id(hand))
            self.recovered.add(name)
        return {
            **{name: hand for name, hand in assigned.items() if name in self.names},
            **{f"unassigned_{index}": hand for index, hand in enumerate(hands) if id(hand) not in used},
        }

    def observe(self, confirmed, poses, timestamp):
        origins = _project_band_origins(poses, self.calibration)
        for name in self.names:
            hand = confirmed.get(name)
            if hand is not None and name in poses:
                z = float(poses[name].tvec[2, 0])
                pixels = hand.image_landmarks_normalized[PALM, :2] * self.calibration.image_size
                if np.isfinite(timestamp) and np.isfinite(z) and z > 0 and np.all(np.isfinite(pixels)):
                    self.previous[name] = {
                        "time": timestamp, "z": z, "origin": origins[name].copy(),
                        "points": pixels.copy(),
                    }
            elif name in self.previous and not 0 <= timestamp - self.previous[name]["time"] <= self.max_age_s:
                del self.previous[name]


class WristCropDetector:
    """Lazy image-mode detector, invoked only for unassigned visible wrists."""

    def __init__(self, model_path, calibration, names, min_confidence=0.4):
        self.model_path = Path(model_path)
        self.calibration = calibration
        self.names = tuple(names)
        self.min_confidence = float(min_confidence)
        self.model = None
        self.mp = None
        self.calls = 0

    def _detect_crop(self, crop):
        if self.model is None:
            import mediapipe as mp

            self.mp = mp
            self.model = mp.tasks.vision.HandLandmarker.create_from_options(
                mp.tasks.vision.HandLandmarkerOptions(
                    base_options=mp.tasks.BaseOptions(model_asset_path=str(self.model_path)),
                    running_mode=mp.tasks.vision.RunningMode.IMAGE,
                    num_hands=2,
                    min_hand_detection_confidence=self.min_confidence,
                    min_hand_presence_confidence=self.min_confidence,
                )
            )
        self.calls += 1
        return self.model.detect(self.mp.Image(
            image_format=self.mp.ImageFormat.SRGB,
            data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
        ))

    def detect(self, frame, hands, poses, assigned=None):
        if assigned is None:
            assigned = assign_hands_to_bands(hands, self.names, poses, self.calibration)
        missing = [name for name in self.names if name not in assigned and name in poses]
        if not missing:
            return []
        if frame is None:
            raise ValueError("current RGB frame required for wrist ROI hand recovery")
        origins = _project_band_origins(poses, self.calibration)
        height, width = frame.shape[:2]
        extra = []
        for name in missing:
            z = float(poses[name].tvec[2, 0])
            cx, cy = origins[name]
            if not np.isfinite(z) or z <= 0 or not np.all(np.isfinite([cx, cy])):
                continue
            if not 0 <= cx < width or not 0 <= cy < height:
                continue
            side = int(np.clip(
                0.40 * self.calibration.camera_matrix[0, 0] / z,
                min(128, width, height), min(width, height),
            ))
            x0 = max(0, min(width - side, int(cx - side / 2)))
            y0 = max(0, min(height - side, int(cy - side / 2)))
            result = self._detect_crop(frame[y0:y0 + side, x0:x0 + side])
            for handedness, landmarks, world in zip(
                result.handedness, result.hand_landmarks, result.hand_world_landmarks
            ):
                image = np.asarray([[point.x, point.y, point.z] for point in landmarks], dtype=float)
                model = np.asarray([[point.x, point.y, point.z] for point in world], dtype=float)
                if image.shape != (21, 3) or model.shape != (21, 3):
                    continue
                if not np.all(np.isfinite(image)) or not np.all(np.isfinite(model)):
                    continue
                image[:, 0] = (image[:, 0] * side + x0) / width
                image[:, 1] = (image[:, 1] * side + y0) / height
                image[:, 2] *= side / width
                category = handedness[0]
                hand = RawHandJoints(
                    category.category_name, float(category.score), image, model,
                    detection_source="wrist_roi",
                )
                selected = assign_hands_to_bands([hand], [name], poses, self.calibration)
                if selected.get(name) is not hand:
                    continue
                if any(np.median(np.linalg.norm(
                    (image[PALM, :2] - old.image_landmarks_normalized[PALM, :2]) * [width, height], axis=1
                )) < 25 for old in hands + extra):
                    continue
                extra.append(hand)
                break
        return extra

    def close(self):
        if self.model is not None:
            self.model.close()
            self.model = None
