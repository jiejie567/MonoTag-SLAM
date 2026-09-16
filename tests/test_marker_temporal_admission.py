"""Offline admission reviews observations; it never repairs or averages corners."""
import copy
import unittest

import cv2
import numpy as np

from aruco_track.marker_temporal_admission import review_static_marker_sequence
from aruco_track.models import Calibration


# Explicit source observations reproduce the 5689 lower-left corner anomaly,
# without depending on local output videos, caches, or a running SLAM backend.
BAD_CORNERS = np.array([[1463.8392333984375, 698.2258911132812],
                        [1543.851806640625, 704.509765625],
                        [1557.425048828125, 786.2421875],
                        [1480.3048095703125, 780.1654052734375]])
CLEAN_CORNERS = np.array([[1463.4970703125, 698.4363403320312],
                          [1543.4014892578125, 704.6444702148438],
                          [1556.9224853515625, 786.4943237304688],
                          [1474.3648681640625, 780.101318359375]])


class MarkerTemporalAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.calibration = Calibration(
            np.array([[741.6085732738395, 0., 969.1962180620636],
                      [0., 741.2205496694032, 557.5812743710624], [0., 0., 1.]]),
            np.array([-.0064470047900432054, -.03410240116622422,
                      .0004951538562097184, -.0004894565369588001, .010000663211677745]),
            (1920, 1080))

    @staticmethod
    def clean(marker=27):
        return {marker: CLEAN_CORNERS.copy()}

    @staticmethod
    def bad(marker=27):
        return {marker: BAD_CORNERS.copy()}

    def review(self, frames, timestamps=None, **kwargs):
        if timestamps is None:
            timestamps = [i / 90. for i in range(len(frames))]
        return review_static_marker_sequence(frames, timestamps, self.calibration,
                                             static_marker_ids={24, 27}, **kwargs)

    def test_clean_single_frame_initialization_is_not_unconditionally_delayed(self):
        result = self.review([self.clean()])
        self.assertEqual(result.excluded_ids, [set()])
        np.testing.assert_array_equal(result.detections[0][27], CLEAN_CORNERS)

    def test_static_clean_sequence_is_retained_from_its_first_frame(self):
        result = self.review([self.clean() for _ in range(8)])
        self.assertEqual(result.excluded_ids, [set() for _ in range(8)])

    def test_camera_translation_rotation_and_perspective_are_not_corner_motion_outliers(self):
        local = np.array([[-.024,.024,0], [.024,.024,0], [.024,-.024,0], [-.024,-.024,0]])
        frames = []
        for i in range(5):
            corners = cv2.projectPoints(local, np.array([2.6 + .08*i, .2 - .025*i, .1]),
                                         np.array([-.02 + .008*i, .01 - .003*i, .48 - .01*i]),
                                         self.calibration.camera_matrix, self.calibration.dist_coeffs)[0].reshape(4, 2)
            frames.append({27: corners})
        self.assertGreater(np.linalg.norm(frames[-1][27] - frames[0][27]), 50.)
        result = self.review(frames)
        self.assertEqual(result.excluded_ids, [set() for _ in frames])

    def test_single_displaced_corner_is_excluded_with_two_clean_temporal_references(self):
        frames = [self.clean(), self.clean(), self.bad(), self.clean(), self.clean()]
        result = self.review(frames)
        self.assertEqual(result.excluded_ids, [set(), set(), {27}, set(), set()])
        self.assertNotIn(27, result.detections[2])
        self.assertEqual(result.diagnostics[2][27]['reason'], 'temporal_geometry_outlier')

    def test_bad_reappearance_never_gets_repaired_or_backfilled_by_later_clean_frames(self):
        frames = [self.bad(), self.clean(), self.clean()]
        result = self.review(frames)
        self.assertEqual(result.excluded_ids, [{27}, set(), set()])
        self.assertNotIn(27, result.detections[0])
        for index in (1, 2):
            np.testing.assert_array_equal(result.detections[index][27], frames[index][27])
        diagnostic = result.diagnostics[0][27]
        self.assertTrue(diagnostic['uses_future_evidence'])
        self.assertEqual(diagnostic['evidence_frame_indices'], [1, 2])
        self.assertEqual(diagnostic['evidence_timestamps_s'], [1/90., 2/90.])

    def test_past_only_evidence_is_not_mislabeled_as_lookahead(self):
        result = self.review([self.clean(), self.clean(), self.bad()])
        self.assertEqual(result.diagnostics[2][27]['reason'], 'temporal_geometry_outlier')
        self.assertFalse(result.diagnostics[2][27]['uses_future_evidence'])

    def test_mature_continuous_stream_still_reviews_a_new_bad_corner(self):
        frames = [self.clean() for _ in range(5)] + [self.bad(), self.clean(), self.clean()]
        result = self.review(frames)
        self.assertEqual(result.excluded_ids[5], {27})
        self.assertEqual(result.diagnostics[5][27]['reason'], 'temporal_geometry_outlier')

    def test_mature_stream_without_independent_evidence_is_inconclusive_not_deleted(self):
        frames = [self.clean(), self.clean(), self.clean(), {}, {}, self.clean(), self.bad(), {}, {}]
        result = self.review(frames)
        self.assertIn(27, result.detections[6])
        self.assertEqual(result.excluded_ids[6], set())
        self.assertEqual(result.diagnostics[6][27]['state'], 'inconclusive')

    def test_mature_residual_must_also_exceed_reference_ratio_not_just_absolute_gate(self):
        reference = .7 * CLEAN_CORNERS + .3 * BAD_CORNERS
        suspicious = .2 * CLEAN_CORNERS + .8 * BAD_CORNERS
        frames = [self.clean() for _ in range(3)]
        frames += [{27: reference.copy()}, {27: reference.copy()}, {27: suspicious},
                   {27: reference.copy()}, {27: reference.copy()}]
        result = self.review(frames)
        diagnostic = result.diagnostics[5][27]
        self.assertGreater(diagnostic['square_rms_px'], .75)
        self.assertLessEqual(diagnostic['neighbor_median_rms_px'], .5)
        self.assertLess(diagnostic['square_rms_px'], 4 * diagnostic['neighbor_median_rms_px'])
        self.assertEqual(diagnostic['state'], 'inconclusive')
        self.assertIn(27, result.detections[5])

    def test_single_suspicious_frame_cannot_initialize_as_strong(self):
        result = self.review([self.bad()])
        self.assertEqual(result.excluded_ids, [{27}])
        self.assertEqual(result.diagnostics[0][27]['reason'], 'pending_geometry_confirmation')

    def test_short_flash_has_insufficient_evidence_but_clean_frame_is_not_delayed(self):
        result = self.review([{}, self.bad(), self.clean(), {}])
        self.assertEqual(result.excluded_ids[1], {27})
        self.assertEqual(result.diagnostics[1][27]['reason'], 'pending_geometry_confirmation')
        self.assertIn(27, result.detections[2])

    def test_suspicious_end_of_sequence_does_not_invent_future_confirmation(self):
        result = self.review([{}, {}, self.bad()])
        self.assertEqual(result.excluded_ids[-1], {27})
        self.assertEqual(result.diagnostics[-1][27]['reason'], 'pending_geometry_confirmation')

    def test_eight_bad_first_observations_cannot_mature_by_persistence(self):
        result = self.review([self.bad() for _ in range(8)])
        self.assertEqual(result.excluded_ids, [{27} for _ in range(8)])
        self.assertTrue(all(row[27]['state'] == 'pending' for row in result.diagnostics))
        self.assertTrue(all(row[27]['reason'] == 'pending_geometry_confirmation'
                            for row in result.diagnostics))

    def test_one_real_clean_observation_confirms_episode_but_does_not_backfill_bad_history(self):
        frames = [self.bad() for _ in range(8)] + [self.clean()] + [self.bad() for _ in range(3)]
        result = self.review(frames)
        self.assertEqual(result.excluded_ids[:8], [{27} for _ in range(8)])
        self.assertEqual(result.diagnostics[8][27]['state'], 'accepted')
        self.assertEqual(result.excluded_ids[8:], [set() for _ in range(4)])
        self.assertTrue(all(row[27]['state'] == 'inconclusive' for row in result.diagnostics[9:]))

    def test_half_second_gap_resets_confirmation_even_after_previous_clean_episode(self):
        frames = [self.clean(), self.bad(), {}] + [self.bad() for _ in range(8)]
        times = [0., .01, .5] + [.51 + i/90. for i in range(8)]
        result = self.review(frames, times)
        self.assertEqual(result.diagnostics[0][27]['state'], 'accepted')
        self.assertEqual(result.diagnostics[1][27]['state'], 'inconclusive')
        self.assertEqual(result.excluded_ids[3:], [{27} for _ in range(8)])
        self.assertTrue(all(row[27]['state'] == 'pending' for row in result.diagnostics[3:]))

    def test_long_absence_starts_a_new_geometry_confirmation_episode(self):
        frames = [self.clean(), self.clean(), self.clean(), {}, self.bad()]
        result = self.review(frames, [0., .01, .02, .60, .61])
        self.assertEqual(result.excluded_ids[-1], {27})
        self.assertEqual(result.diagnostics[-1][27]['reason'], 'pending_geometry_confirmation')

    def test_clean_reappearance_after_bad_group_can_immediately_join(self):
        frames = [self.clean(), self.clean(), self.clean(), {}, self.bad(), self.clean(), self.clean()]
        result = self.review(frames, [0., .01, .02, .60, .61, .62, .63])
        self.assertEqual(result.excluded_ids[4], {27})
        self.assertEqual(result.diagnostics[4][27]['reason'], 'temporal_geometry_outlier')
        self.assertEqual(result.excluded_ids[5:], [set(), set()])

    def test_references_outside_two_source_frames_are_not_used(self):
        result = self.review([self.bad(), {}, {}, self.clean(), self.clean()])
        self.assertEqual(result.diagnostics[0][27]['reason'], 'pending_geometry_confirmation')

    def test_neighboring_indices_outside_time_window_are_not_used(self):
        result = self.review([self.clean(), self.clean(), self.bad(), self.clean(), self.clean()],
                             [0., .01, .4, .8, .81])
        # The earlier clean observation already confirmed the current episode;
        # its age prevents voting, not the existing inconclusive fallback.
        self.assertEqual(result.diagnostics[2][27]['evidence_frame_indices'], [])
        self.assertEqual(result.diagnostics[2][27]['state'], 'inconclusive')
        self.assertEqual(result.excluded_ids[2], set())

    def test_weak_observations_do_not_supply_clean_temporal_evidence(self):
        frames = [self.clean(), self.clean(), self.bad(), self.clean(), self.clean()]
        weights = [{27: .25}, {27: .25}, {27: 1.}, {27: .25}, {27: .25}]
        result = self.review(frames, marker_weights=weights)
        self.assertEqual(result.diagnostics[2][27]['reason'], 'pending_geometry_confirmation')

    def test_flow_or_recovered_corners_do_not_supply_fresh_decode_evidence(self):
        frames = [self.clean(), self.clean(), self.bad(), self.clean(), self.clean()]
        nondecoded = [{27}, {27}, set(), {27}, {27}]
        result = self.review(frames, nondecoded_ids=nondecoded)
        self.assertEqual(result.diagnostics[2][27]['reason'], 'pending_geometry_confirmation')

    def test_weak_current_observation_keeps_its_existing_weak_path_without_a_vote(self):
        result = self.review([self.bad()], marker_weights=[{27: .25}])
        self.assertEqual(result.excluded_ids, [set()])
        self.assertEqual(result.diagnostics, [{}])
        np.testing.assert_array_equal(result.detections[0][27], BAD_CORNERS)

    def test_nondecoded_current_observation_is_not_promoted_or_temporally_reviewed(self):
        result = self.review([self.bad()], nondecoded_ids=[{27}])
        self.assertEqual(result.excluded_ids, [set()])
        self.assertEqual(result.diagnostics, [{}])
        np.testing.assert_array_equal(result.detections[0][27], BAD_CORNERS)

    def test_other_static_marker_does_not_confirm_the_target_marker(self):
        frames = [self.clean(24), self.clean(24), self.bad(27), self.clean(24), self.clean(24)]
        result = self.review(frames)
        self.assertEqual(result.diagnostics[2][27]['reason'], 'pending_geometry_confirmation')
        # Even absent target IDs remain blocked to clear native flow/weak state.
        self.assertEqual(result.excluded_ids, [set(), set(), {27}, {27}, {27}])
        self.assertIn(24, result.detections[3])
        self.assertIn(24, result.detections[4])

    def test_raw_flow_or_board_recovery_successor_stays_blocked_until_clean_decode(self):
        result = self.review([self.bad(), self.clean(), self.clean()],
                             nondecoded_ids=[set(), {27}, set()])
        self.assertEqual(result.excluded_ids, [{27}, {27}, set()])
        self.assertNotIn(27, result.detections[1])
        self.assertEqual(result.diagnostics[1][27]['reason'], 'unverified_recovery_from_rejected_quad')
        self.assertEqual(result.diagnostics[2][27]['state'], 'accepted')
        np.testing.assert_array_equal(result.detections[2][27], CLEAN_CORNERS)

    def test_weak_successor_cannot_remove_quarantine_even_with_clean_looking_corners(self):
        result = self.review([self.bad(), self.clean(), self.clean()],
                             marker_weights=[{27: 1.}, {27: .25}, {27: 1.}])
        self.assertEqual(result.excluded_ids, [{27}, {27}, set()])
        self.assertEqual(result.diagnostics[1][27]['reason'], 'unverified_recovery_from_rejected_quad')
        self.assertEqual(result.diagnostics[2][27]['state'], 'accepted')

    def test_missing_frames_keep_tracker_guard_without_inventing_pixel_observations(self):
        result = self.review([self.bad(), {}, {}, {}, self.clean()])
        self.assertEqual(result.excluded_ids, [{27}, {27}, {27}, {27}, set()])
        for index in (1, 2, 3):
            self.assertEqual(result.detections[index], {})
            self.assertEqual(result.diagnostics[index], {})
        self.assertEqual(result.diagnostics[4][27]['state'], 'accepted')
        self.assertEqual(result.summary['excluded_observations'], 1)
        self.assertEqual(result.summary['excluded_frames'], 1)
        self.assertEqual(result.summary['blocked_tracker_frames'], 4)

    def test_mature_stream_rejected_quad_also_blocks_flow_and_weak_successors(self):
        frames = [self.clean(), self.clean(), self.clean(), self.bad(),
                  self.clean(), self.clean(), self.clean()]
        result = self.review(frames, marker_weights=[{}, {}, {}, {}, {}, {27: .25}, {}],
                             nondecoded_ids=[set(), set(), set(), set(), {27}, set(), set()])
        self.assertEqual(result.diagnostics[3][27]['reason'], 'temporal_geometry_outlier')
        self.assertEqual(result.excluded_ids, [set(), set(), set(), {27}, {27}, {27}, set()])
        self.assertEqual(result.diagnostics[6][27]['state'], 'accepted')

    def test_fresh_but_inconclusive_successor_cannot_clear_a_rejection(self):
        frames = [self.clean(), self.clean(), self.bad(), self.bad(), self.bad(), {}, {}, self.clean()]
        result = self.review(frames)
        self.assertEqual(result.diagnostics[2][27]['reason'], 'temporal_geometry_outlier')
        for index in (3, 4):
            self.assertEqual(result.diagnostics[index][27]['reason'], 'awaiting_clean_decode_after_rejection')
            self.assertEqual(result.diagnostics[index][27]['state'], 'pending')
            self.assertNotIn(27, result.detections[index])
        self.assertEqual(result.excluded_ids[5:7], [{27}, {27}])
        self.assertEqual(result.diagnostics[7][27]['state'], 'accepted')

    def test_quarantining_static_id_does_not_touch_neighboring_wrist_observations(self):
        frames = [{**self.bad(), **self.bad(6)}, self.bad(6), {**self.clean(), **self.bad(6)}]
        result = self.review(frames)
        self.assertEqual(result.excluded_ids, [{27}, {27}, set()])
        for index in range(3):
            np.testing.assert_array_equal(result.detections[index][6], BAD_CORNERS)
            self.assertNotIn(6, result.diagnostics[index])

    def test_unconfigured_wrist_marker_is_untouched_even_with_bad_square_geometry(self):
        frames = [self.bad(6), self.bad(6), self.bad(6)]
        result = self.review(frames)
        self.assertEqual(result.excluded_ids, [set(), set(), set()])
        for index in range(3):
            np.testing.assert_array_equal(result.detections[index][6], BAD_CORNERS)
            self.assertNotIn(6, result.diagnostics[index])

    def test_invalid_static_geometry_is_rejected_without_changing_its_input(self):
        for corners in (np.full((4, 2), np.nan), np.zeros((4, 2)), np.zeros((3, 2))):
            with self.subTest(corners=corners.tolist()):
                original = corners.copy()
                result = self.review([{27: corners}])
                self.assertEqual(result.excluded_ids, [{27}])
                np.testing.assert_array_equal(corners, original)

    def test_raw_input_and_retained_corner_arrays_are_independent(self):
        frames = [self.clean(), self.bad(), self.clean()]
        before = copy.deepcopy(frames)
        result = self.review(frames)
        for original, unchanged in zip(frames, before):
            np.testing.assert_array_equal(original[27], unchanged[27])
        self.assertIsNot(result.detections, frames)
        self.assertIsNot(result.detections[0], frames[0])
        self.assertIsNot(result.detections[0][27], frames[0][27])
        result.detections[0][27][0, 0] += 1.
        np.testing.assert_array_equal(frames[0][27], before[0][27])

    def test_duplicate_time_cannot_supply_two_independent_frames(self):
        with self.assertRaises(ValueError):
            self.review([self.clean(), self.bad(), self.clean()], [0., .01, .01])

    def test_invalid_time_or_unmatched_side_inputs_fail_explicitly(self):
        frames = [self.clean(), self.clean()]
        for timestamps in ([0.], [0., np.nan], [.1, 0.]):
            with self.subTest(timestamps=timestamps), self.assertRaises(ValueError):
                self.review(frames, timestamps)
        with self.assertRaises(ValueError):
            self.review(frames, marker_weights=[{}])
        with self.assertRaises(ValueError):
            self.review(frames, nondecoded_ids=[set()])

    def test_empty_sequence_has_no_invented_measurements(self):
        result = self.review([])
        self.assertEqual(result.detections, [])
        self.assertEqual(result.excluded_ids, [])
        self.assertEqual(result.diagnostics, [])


if __name__ == '__main__':
    unittest.main()
