"""Short-lived tracking of decoded fixed-tag corners, not a background VO.

Only a complete, strong detection seeds identities. Optical flow carries those
identities across a brief decoding gap; it never invents a hidden fourth corner.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
import cv2
import numpy as np

from .models import BandLayout, Calibration, Pose


@dataclass
class TrackedMarkerObservation:
    pose: Pose | None = None
    confidence: float = 0.
    partial_only: bool = False
    world_points: list = field(default_factory=list)
    image_points: list = field(default_factory=list)
    marker_ids: list = field(default_factory=list)
    corner_indices: list = field(default_factory=list)
    reason: str = 'no_tracks'
    age_s: float = 0.
    point_weights: list = field(default_factory=list)


class MarkerCornerTracker:
    """Conservative measured-corner bridge, capped at 0.30 s since decoding."""
    def __init__(self, calibration: Calibration, layout: BandLayout, max_age_s=.30):
        self.calibration, self.layout, self.max_age_s = calibration, layout, max_age_s
        self.previous_gray = None
        self.previous_time = None
        self.last_pose = None
        self.last_pose_time = None
        self.tracks = {}  # (decoded marker id, corner index) -> pixel/seed time
        self.weak_streaks = {}  # rejected decoded corner -> (consecutive frames, time)

    @staticmethod
    def _patch(gray, xy):
        x, y = xy
        if not (7 <= x < gray.shape[1]-7 and 7 <= y < gray.shape[0]-7):
            return None
        patch = cv2.getRectSubPix(gray, (11, 11), (float(x), float(y))).astype(float)
        patch -= patch.mean()
        norm = np.linalg.norm(patch)
        return patch / norm if norm > 60. else None

    def _project(self, world, camera):
        rotation = camera.rotation_matrix.T
        translation = -rotation @ camera.tvec.reshape(3)
        pixels = cv2.projectPoints(np.asarray(world, float), cv2.Rodrigues(rotation)[0],
                                  translation, self.calibration.camera_matrix,
                                  self.calibration.dist_coeffs)[0].reshape(-1, 2)
        return pixels, (np.asarray(world) @ rotation.T + translation)[:, 2]

    def _partial_pose(self, world, image, marker_ids, timestamp):
        if self.last_pose is None or timestamp-self.last_pose_time > .10:
            return None, 'no_recent_pose', None
        world = np.asarray(world, float)
        image = np.asarray(image, float)
        if len(image) < 3 or cv2.contourArea(cv2.convexHull(image.astype(np.float32))) < 40:
            return None, 'insufficient_geometry', None
        # A hidden corner can flow onto a different black/white edge even with
        # good forward/backward error. Test bounded, identity-preserving triples
        # as well as the whole set; retain only measured geometric inliers.
        subsets = [list(range(len(world)))]
        if len(world) > 3:
            for mid in sorted(set(marker_ids)):
                indices = [i for i, value in enumerate(marker_ids) if value == mid]
                subsets.extend(list(triple) for triple in combinations(indices, 3))
        candidates = []
        three_corner_rejected = False
        for indices in subsets:
            try:
                if len(indices) == 3 and self.last_pose is not None:
                    # SQPNP is multi-valued for a three-point planar sample;
                    # on OpenCV 4.6 it can return a positive-depth branch far
                    # from the last measured pose even when the image points
                    # are exact.  Use the recent pose only as an extrinsic
                    # initial guess for the iterative solver, preserving the
                    # existing motion gate rather than accepting that branch.
                    guess_rotation = self.last_pose.rotation_matrix.T
                    guess_translation = -guess_rotation @ self.last_pose.tvec
                    guess_rvec = cv2.Rodrigues(guess_rotation)[0]
                    ok, rvec, tvec = cv2.solvePnP(
                        world[indices], image[indices],
                        self.calibration.camera_matrix, self.calibration.dist_coeffs,
                        guess_rvec, guess_translation, True,
                        flags=cv2.SOLVEPNP_ITERATIVE)
                    if not ok:
                        continue
                    rotations, translations = [rvec], [tvec]
                else:
                    _, rotations, translations, _ = cv2.solvePnPGeneric(
                        world[indices], image[indices], self.calibration.camera_matrix,
                        self.calibration.dist_coeffs, flags=cv2.SOLVEPNP_SQPNP)
            except cv2.error:
                continue
            for rvec, tvec in zip(rotations, translations):
                rotation = cv2.Rodrigues(rvec)[0].T
                pose = Pose(cv2.Rodrigues(rotation)[0], -rotation @ tvec.reshape(3, 1), 0.)
                predicted, depth = self._project(world, pose)
                residual = np.linalg.norm(predicted-image, axis=1)
                inliers = (depth > .02) & (residual <= 1.5)
                if np.count_nonzero(inliers) < 3:
                    continue
                error = float(np.sqrt(np.mean(residual[inliers]**2)))
                distance = float(np.linalg.norm(pose.tvec-self.last_pose.tvec))
                angle = float(np.linalg.norm(cv2.Rodrigues(
                    pose.rotation_matrix @ self.last_pose.rotation_matrix.T)[0]))
                # Three 2D corners have no residual redundancy: even a wrong
                # pose can fit them exactly. Only carry small measured steps;
                # fast motion must be supported by a full tag or background ORB.
                # Keep this gate in sync with native SetExternalTagObservation.
                if inliers.sum() == 3 and (distance > .010 or angle > np.deg2rad(3)):
                    three_corner_rejected = True
                    continue
                if distance > .03 or angle > np.deg2rad(12):
                    continue
                score = distance/.015 + angle/np.deg2rad(6) + error
                # Different triples often recover the same solution; those are
                # corroboration, not independent planar pose ambiguities.
                if any(np.linalg.norm(pose.tvec-p.tvec) < .001 and
                       np.linalg.norm(cv2.Rodrigues(pose.rotation_matrix @ p.rotation_matrix.T)[0]) < np.deg2rad(.1)
                       for _, _, p, _ in candidates):
                    continue
                candidates.append((-int(inliers.sum()), score, Pose(pose.rvec, pose.tvec, error), inliers))
        candidates.sort(key=lambda item: item[:2])
        if not candidates:
            return None, ('three_corner_motion_gate' if three_corner_rejected
                          else 'prediction_inconsistent'), None
        if (len(candidates)>1 and candidates[1][0] == candidates[0][0]
                and candidates[1][1]-candidates[0][1] < .5):
            return None, 'ambiguous_pose', None
        return candidates[0][2], 'tracked', candidates[0][3]

    def update(
        self, frame, timestamp, detections, accepted_ids, pose, confidence,
        weights=None, weak_detections=None, weak_corner_weights=None,
    ):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        weights = weights or {}
        reliable = pose is not None and confidence >= .35 and pose.reprojection_error_px <= 2.5
        decoded = set(accepted_ids) if reliable else set()
        result = TrackedMarkerObservation()
        live = {}
        quality = []
        keys = [key for key, (_, seen) in self.tracks.items()
                if timestamp-seen <= self.max_age_s and key[0] not in decoded]
        if (keys and self.previous_gray is not None and self.previous_time is not None
                and 0 < timestamp-self.previous_time <= .10):
            old = np.array([self.tracks[key][0] for key in keys], np.float32).reshape(-1, 1, 2)
            new, good, errors = cv2.calcOpticalFlowPyrLK(self.previous_gray, gray, old, None,
                winSize=(15, 15), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, .01))
            if new is not None:
                back, back_good, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous_gray, new, None,
                    winSize=(15, 15), maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, .01))
                # First apply the inexpensive forward/backward and patch
                # checks.  A second, marker-local affine check then rejects a
                # corner that has flowed onto a nearby edge or background: the
                # other three corners predict its position without inventing a
                # fourth measurement.  This is especially useful for a single
                # occluded corner and costs only a few 2-D operations.
                flow_candidates = {}
                for i, key in enumerate(keys):
                    if (back is None or not good[i, 0] or not back_good[i, 0]
                            or errors[i, 0] > 12 or np.linalg.norm(old[i]-back[i]) > .65):
                        continue
                    previous_patch = self._patch(self.previous_gray, old[i, 0])
                    patch = self._patch(gray, new[i, 0])
                    if previous_patch is None or patch is None:
                        continue
                    similarity = float(np.sum(previous_patch*patch))
                    if similarity < .85:
                        continue
                    flow_candidates[i] = (new[i, 0].copy(), self.tracks[key][1], similarity)
                geometric_inliers = set(flow_candidates)
                for marker_id in sorted({key[0] for key in keys}):
                    group = [i for i, key in enumerate(keys)
                             if key[0] == marker_id and i in flow_candidates]
                    if len(group) < 4:
                        continue
                    displacements = np.asarray(
                        [new[j, 0]-old[j, 0] for j in group], np.float32)
                    median_displacement = np.median(displacements, axis=0)
                    motion_inliers = {
                        j for j, displacement in zip(
                            group, displacements
                        ) if np.linalg.norm(displacement-median_displacement) <= 6.0
                    }
                    # A hidden corner often follows a nearby edge and still
                    # passes forward/backward flow.  Its displacement is,
                    # however, an outlier relative to the other three corners.
                    # Prefer this robust median test when it identifies exactly
                    # one outlier; avoid discarding an entire marker if fewer
                    # than three corners have trustworthy motion.
                    if len(motion_inliers) >= 3 and len(motion_inliers) < len(group):
                        geometric_inliers.intersection_update(motion_inliers)
                        continue
                    # Exact three-point affine hypotheses are sufficient for a
                    # small marker patch.  Enumerating the four triples gives
                    # a tiny RANSAC-like test: retain the largest consistent
                    # subset and discard a corner that flowed to an unrelated
                    # edge/background.  This avoids trusting the outlier while
                    # keeping all four corners when the marker is intact.
                    best_inliers, best_score = set(group), (0, float('inf'))
                    for triple in combinations(group, 3):
                        try:
                            affine = cv2.getAffineTransform(
                                np.asarray([old[j, 0] for j in triple], np.float32),
                                np.asarray([flow_candidates[j][0] for j in triple], np.float32),
                            )
                        except cv2.error:
                            continue
                        residuals = {}
                        for j in group:
                            predicted = affine @ np.array(
                                [old[j, 0, 0], old[j, 0, 1], 1.], np.float32)
                            residuals[j] = float(np.linalg.norm(predicted-flow_candidates[j][0]))
                        inliers = {j for j, residual in residuals.items() if residual <= 4.0}
                        score = (-len(inliers), sum(residuals[j] for j in inliers))
                        if score < best_score:
                            best_inliers, best_score = inliers, score
                    if len(best_inliers) >= 3:
                        geometric_inliers.intersection_update(best_inliers)
                for i, key in enumerate(keys):
                    candidate = flow_candidates.get(i)
                    if candidate is None or i not in geometric_inliers:
                        continue
                    pixel, seen, similarity = candidate
                    world = self.layout.markers[key[0]][key[1]]
                    if reliable:
                        prediction, depth = self._project([world], pose)
                        if depth[0] <= 0 or np.linalg.norm(prediction[0]-pixel) > 2.:
                            continue
                    live[key] = (pixel, seen)
                    result.world_points.append(world.tolist())
                    result.image_points.append(pixel.tolist())
                    result.marker_ids.append(key[0]); result.corner_indices.append(key[1])
                    result.point_weights.append(.25)
                    quality.append(similarity)
                    result.age_s = max(result.age_s, timestamp-seen)
        current_weak_streaks = {}
        if reliable and weak_detections and weak_corner_weights:
            for marker_id, corners in weak_detections.items():
                if marker_id in decoded or marker_id not in self.layout.markers:
                    continue
                corners = np.asarray(corners, float).reshape(-1, 2)
                corner_weights = weak_corner_weights.get(marker_id, ())
                if len(corners) != 4 or len(corner_weights) != 4:
                    continue
                predicted, depth = self._project(self.layout.markers[marker_id], pose)
                for corner_index, (pixel, expected, z, point_weight) in enumerate(
                    zip(corners, predicted, depth, corner_weights)
                ):
                    if point_weight <= 0 or z <= 0 or np.linalg.norm(pixel-expected) > 2.0:
                        continue
                    key = (marker_id, corner_index)
                    previous = self.weak_streaks.get(key)
                    consecutive = (
                        previous[0] + 1
                        if previous is not None and 0 < timestamp-previous[1] <= .12
                        else 1
                    )
                    current_weak_streaks[key] = (consecutive, timestamp)
                    if consecutive < 3:
                        continue
                    result.world_points.append(
                        self.layout.markers[marker_id][corner_index].tolist()
                    )
                    result.image_points.append(pixel.tolist())
                    result.marker_ids.append(marker_id)
                    result.corner_indices.append(corner_index)
                    result.point_weights.append(float(point_weight))
        self.weak_streaks = current_weak_streaks
        if result.world_points:
            if reliable:
                result.pose, result.confidence, result.reason = pose, confidence, 'with_decoded_marker'
            else:
                result.pose, result.reason, inliers = self._partial_pose(
                    result.world_points, result.image_points, result.marker_ids, timestamp)
                result.partial_only = True
                # Tiny per-frame flow errors accumulate without residual
                # redundancy. Three-point poses expire sooner than 4+ points.
                if result.pose is not None and inliers.sum() == 3 and result.age_s > .10+1e-9:
                    result.pose, result.reason = None, 'three_corner_age_gate'
                if result.pose is not None:
                    for name in (
                        'world_points', 'image_points', 'marker_ids',
                        'corner_indices', 'point_weights',
                    ):
                        setattr(result, name, [value for value, keep in zip(getattr(result, name), inliers) if keep])
                    quality = [value for value, keep in zip(quality, inliers) if keep]
                    live = {key: value for key, value in live.items()
                            if key in set(zip(result.marker_ids, result.corner_indices))}
                    result.confidence = float(.45*np.mean(quality)*np.exp(-result.age_s/self.max_age_s))
        if reliable:
            for mid in decoded:
                if mid not in self.layout.markers or mid not in detections or weights.get(mid, 1.) < .99:
                    continue
                corners = np.asarray(detections[mid], float).reshape(-1, 2)
                if len(corners) != 4:
                    continue
                for corner, pixel in enumerate(corners):
                    live[(mid, corner)] = (pixel.astype(np.float32), timestamp)
            self.last_pose, self.last_pose_time = pose, timestamp
        elif result.pose is not None:
            self.last_pose, self.last_pose_time = result.pose, timestamp
        self.tracks, self.previous_gray, self.previous_time = live, gray.copy(), timestamp
        return result
