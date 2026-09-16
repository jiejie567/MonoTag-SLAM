"""Current-measurement-only regression checks; no real network or SLAM run."""
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from aruco_track.hand_recovery import (
    ContinuityAssociation,
    WristCropDetector,
    hand_recovery_policy,
    supplement_assignments,
)
from aruco_track.hands import (
    HandJointTracker, RawHandJoints, assign_hands_to_bands,
    joint_pose_to_dict, raw_hand_from_dict,
)
from aruco_track.models import Calibration, Pose


class HandRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.names = ["strap_band_L", "strap_band_R"]
        self.cal = Calibration(
            np.array([[500., 0., 320.], [0., 500., 240.], [0., 0., 1.]]),
            np.zeros(5), (640, 480),
        )
        self.poses = {
            name: Pose(np.zeros((3, 1)), np.array([[x], [0.], [.5]]), .1)
            for name, x in zip(self.names, [-.1, .1])
        }
        self.history = ContinuityAssociation(self.names, self.cal)
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def hand(self, angle=0., x=220., shift=(0., 0.), source="full_frame"):
        wrist = np.array([x, 210.]) + shift
        palm = wrist + 40 * np.array([np.sin(np.deg2rad(angle)), -np.cos(np.deg2rad(angle))])
        image = np.zeros((21, 3))
        image[:, :2] = palm
        image[0, :2] = wrist
        image[:, :2] /= [640, 480]
        return RawHandJoints("Left", .9, image, np.zeros((21, 3)), source)

    def seed(self):
        old = self.hand(65.)
        selected = assign_hands_to_bands([old], self.names, self.poses, self.cal)
        self.assertIs(selected[self.names[0]], old)
        self.history.observe(selected, self.poses, 0.)

    def tracker(self, enabled=True):
        # Image-mode network remains lazy; no model content is needed here.
        with patch.dict("sys.modules", {"mediapipe": types.SimpleNamespace()}):
            return HandJointTracker(
                Path(__file__), self.cal, self.names,
                enable_recovery=enabled, initialize_full_frame_detector=False,
            )

    def roi_result(self, hand, name):
        side = 400
        x0 = 20 if name == self.names[0] else 220
        y0 = 40
        image = hand.image_landmarks_normalized.copy()
        image[:, 0] = (image[:, 0] * 640 - x0) / side
        image[:, 1] = (image[:, 1] * 480 - y0) / side
        points = lambda values: [types.SimpleNamespace(x=x, y=y, z=z) for x, y, z in values]
        return types.SimpleNamespace(
            handedness=[[types.SimpleNamespace(category_name="Left", score=.9)]],
            hand_landmarks=[points(image)], hand_world_landmarks=[points(hand.model_landmarks_m)],
        )

    def test_gradual_bend_recovers_actual_current_measurement(self):
        self.seed()
        hand = self.hand(75.)
        self.assertNotIn(self.names[0], assign_hands_to_bands([hand], self.names, self.poses, self.cal))
        self.assertIs(self.history.assign([hand], self.poses, .01)[self.names[0]], hand)

    def test_no_history_does_not_relax_geometry(self):
        self.assertNotIn(self.names[0], self.history.assign([self.hand(75.)], self.poses, .01))

    def test_no_current_hand_never_copies_history(self):
        self.seed()
        self.assertEqual(self.history.assign([], self.poses, .01), {})

    def test_no_current_wrist_never_recovers_identity(self):
        self.seed()
        self.assertNotIn(self.names[0], self.history.assign([self.hand(75.)], {}, .01))

    def test_history_expiry_and_backward_time(self):
        self.seed()
        for timestamp in [.151, -.01, float("nan")]:
            self.assertNotIn(self.names[0], self.history.assign([self.hand(75.)], self.poses, timestamp))

    def test_large_palm_motion_is_rejected(self):
        self.seed()
        self.assertNotIn(self.names[0], self.history.assign([self.hand(75., shift=(40., 0.))], self.poses, .01))

    def test_competing_candidates_are_rejected(self):
        self.seed()
        self.assertNotIn(self.names[0], self.history.assign([self.hand(75.), self.hand(75.)], self.poses, .01))

    def test_original_strict_match_is_preserved(self):
        self.seed()
        hand = self.hand(10.)
        self.assertIs(self.history.assign([hand], self.poses, .01)[self.names[0]], hand)

    def test_roi_cannot_replace_continuity_assignment(self):
        self.seed()
        hand, extra = self.hand(75.), self.hand(0., source="wrist_roi")
        assigned = self.history.assign([hand], self.poses, .01)
        combined = supplement_assignments(assigned, [hand], [extra], self.names, self.poses, self.cal)
        self.assertIs(combined[self.names[0]], hand)

    def test_roi_fills_only_empty_wrist(self):
        left, right = self.hand(), self.hand(x=420., source="wrist_roi")
        assigned = supplement_assignments(
            {self.names[0]: left}, [left], [right], self.names, self.poses, self.cal
        )
        self.assertIs(assigned[self.names[0]], left)
        self.assertIs(assigned[self.names[1]], right)

    def test_roi_missing_identity_not_raw_candidate_count(self):
        detector = WristCropDetector(Path(__file__), self.cal, self.names)
        bad = [self.hand(90.), self.hand(90., x=420.)]
        detector._detect_crop = Mock(side_effect=[
            self.roi_result(self.hand(), self.names[0]),
            self.roi_result(self.hand(x=420.), self.names[1]),
        ])
        extra = detector.detect(self.frame, bad, self.poses)
        self.assertEqual(detector._detect_crop.call_count, 2)
        self.assertEqual(len(extra), 2)
        self.assertTrue(all(hand.detection_source == "wrist_roi" for hand in extra))

    def test_roi_no_visible_wrist_has_no_network_cost(self):
        detector = WristCropDetector(Path(__file__), self.cal, self.names)
        detector._detect_crop = Mock()
        self.assertEqual(detector.detect(self.frame, [], {}), [])
        detector._detect_crop.assert_not_called()
        self.assertIsNone(detector.model)

    def test_roi_respects_original_geometry(self):
        detector = WristCropDetector(Path(__file__), self.cal, self.names[:1])
        detector._detect_crop = Mock(return_value=self.roi_result(self.hand(90.), self.names[0]))
        self.assertEqual(detector.detect(self.frame, [], self.poses), [])

    def test_roi_rejects_duplicate_existing_hand(self):
        detector = WristCropDetector(Path(__file__), self.cal, self.names[:1])
        current = self.hand()
        detector._detect_crop = Mock(return_value=self.roi_result(current, self.names[0]))
        self.assertEqual(detector.detect(self.frame, [current], self.poses, assigned={}), [])

    def test_continuity_precedes_roi_and_skips_inference(self):
        tracker = self.tracker()
        self.seed()
        tracker._continuity = self.history
        tracker._crop_detector._detect_crop = Mock()
        poses = {self.names[0]: self.poses[self.names[0]]}
        tracker.process_observations(self.frame, 10, [self.hand(75.)], poses)
        tracker._crop_detector._detect_crop.assert_not_called()

    def test_frame_none_only_reuses_cached_candidates(self):
        tracker = self.tracker()
        tracker._crop_detector.detect = Mock()
        self.assertEqual(tracker.process_observations(None, 10, [], self.poses), {})
        tracker._crop_detector.detect.assert_not_called()

    def test_two_frame_confirmation_is_not_removed(self):
        tracker = self.tracker()
        first = tracker.process_observations(None, 0, [self.hand()], self.poses)
        second = tracker.process_observations(None, 11, [self.hand()], self.poses)
        self.assertNotIn(self.names[0], first)
        self.assertIn(self.names[0], second)
        self.assertEqual(second[self.names[0]].temporal_confirmation_frames, 2)

    def test_unconfirmed_first_frame_does_not_seed_recovery(self):
        tracker = self.tracker()
        tracker.process_observations(None, 0, [self.hand(65.)], self.poses)
        second = tracker.process_observations(None, 11, [self.hand(75.)], self.poses)
        self.assertNotIn(self.names[0], second)

    def test_confirmed_history_recovers_and_marks_provenance(self):
        tracker = self.tracker()
        tracker.process_observations(None, 0, [self.hand(65.)], self.poses)
        tracker.process_observations(None, 11, [self.hand(65.)], self.poses)
        result = tracker.process_observations(None, 22, [self.hand(75.)], self.poses)
        self.assertEqual(result[self.names[0]].association_status, "confirmed_temporal")

    def test_recovery_can_be_disabled(self):
        tracker = self.tracker(enabled=False)
        tracker.process_observations(None, 0, [self.hand(65.)], self.poses)
        tracker.process_observations(None, 11, [self.hand(65.)], self.poses)
        result = tracker.process_observations(None, 22, [self.hand(75.)], self.poses)
        self.assertNotIn(self.names[0], result)
        self.assertFalse(tracker.recovery_policy["enabled"])

    def test_protected_cache_match_is_not_rejected_or_replaced(self):
        tracker = self.tracker()
        old = self.hand(90., source="wrist_roi")
        raw = old.image_landmarks_normalized.copy()
        result = tracker.process_observations(None, 0, [old], self.poses,
            protected_assignments={self.names[0]: old})
        self.assertIn(self.names[0], result)
        self.assertEqual(result[self.names[0]].temporal_confirmation_frames, 2)
        self.assertEqual(result[self.names[0]].detection_source, "wrist_roi")
        np.testing.assert_array_equal(result[self.names[0]].image_landmarks_normalized, raw)
        np.testing.assert_array_equal(old.image_landmarks_normalized, raw)

    def test_protected_cache_without_final_wrist_has_no_world_pose(self):
        tracker = self.tracker()
        hand = self.hand()
        result = tracker.process_observations(None, 0, [hand], {},
            self.poses[self.names[0]], {self.names[0]: hand})
        self.assertIn(self.names[0], result)
        self.assertIsNone(result[self.names[0]].camera_landmarks_m)
        self.assertIsNone(result[self.names[0]].world_landmarks_m)

    def test_protected_assignments_must_be_unique_current_objects(self):
        hand = self.hand()
        for protected in [
            {self.names[0]: self.hand()},
            {self.names[0]: hand, self.names[1]: hand},
            {"unknown": hand},
        ]:
            with self.assertRaises(ValueError):
                self.history.assign([hand], self.poses, 0., protected)

    def test_raw_cache_restoration_preserves_source_and_checks_shape(self):
        tracker = self.tracker()
        hand = self.hand(source="wrist_roi")
        result = tracker.process_observations(None, 0, [hand], self.poses,
            protected_assignments={self.names[0]: hand})
        encoded = joint_pose_to_dict(result[self.names[0]])
        restored = raw_hand_from_dict(encoded)
        self.assertEqual(restored.detection_source, "wrist_roi")
        encoded["model_landmarks_m"] = [[0., 0., 0.]]
        with self.assertRaises(ValueError):
            raw_hand_from_dict(encoded)

    def test_policy_and_lazy_crop_use_configured_threshold(self):
        created = []
        result = types.SimpleNamespace(handedness=[], hand_landmarks=[], hand_world_landmarks=[])
        model = types.SimpleNamespace(detect=Mock(return_value=result), close=Mock())
        def create(options):
            created.append(options)
            return model
        mp = types.SimpleNamespace(
            Image=lambda **options: options,
            ImageFormat=types.SimpleNamespace(SRGB="SRGB"),
            tasks=types.SimpleNamespace(
                BaseOptions=lambda **options: options,
                vision=types.SimpleNamespace(
                    RunningMode=types.SimpleNamespace(IMAGE="IMAGE", VIDEO="VIDEO"),
                    HandLandmarkerOptions=lambda **options: options,
                    HandLandmarker=types.SimpleNamespace(create_from_options=create),
                ),
            ),
        )
        with patch.dict("sys.modules", {"mediapipe": mp}):
            tracker = HandJointTracker(Path(__file__), self.cal, self.names,
                min_confidence=.62, initialize_full_frame_detector=False)
            self.assertEqual(created, [])
            tracker.process_observations(self.frame, 0, [], {self.names[0]: self.poses[self.names[0]]})
            self.assertEqual(created[0]["min_hand_detection_confidence"], .62)
            self.assertEqual(created[0]["min_hand_presence_confidence"], .62)
            self.assertEqual(tracker.recovery_diagnostics["roi_inference_calls"], 1)
            tracker.close()
            model.close.assert_called_once()
        self.assertEqual(hand_recovery_policy(min_confidence=.62)["min_confidence"], .62)
        self.assertFalse(hand_recovery_policy()["interpolated_training_labels"])


if __name__ == "__main__":
    unittest.main()
