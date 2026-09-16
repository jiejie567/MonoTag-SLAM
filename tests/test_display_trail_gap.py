"""Display-only short-gap joins must never promote missing action labels."""
import copy
import unittest

import numpy as np

import aruco_track.slam_replay as replay


def snapshot(index=0, state=2):
    return {
        'timestamp': index / 30., 'final': False, 'state': state,
        'active_map': 0,
        'pose': [0., 0., 0., 0., 0., 0., 1.] if state == 2 else None,
        'reference': 10, 'relative': [0., 0., 0., 0., 0., 0., 1.],
        'references': [[10, 0, [0., 0., 0., 0., 0., 0., 1.]]],
        'maps': [{'id': 0, 'metric': True, 'scale': 1., 'seed': True,
                  'background': True, 'revision': 0, 'points': [],
                  'keyframes': [], 'markers': {}, 'loops': [], 'merges': []}],
    }


def records(count=5):
    return [
        {'frame': index, 'timestamp_s': index / 30.,
         'hands': {'right': {'wrist_camera_graph': (
             {'translation_m': [1., 0., 1.]} if index in (0, count - 1)
             else None)}}}
        for index in range(count)
    ]


class DisplayTrailGapTests(unittest.TestCase):
    def test_short_gap_uses_actual_elapsed_time_not_uniform_frame_spacing(self):
        values = [[0., 0., 1.], None, None, [1., 2., 3.]]
        filled, indices = replay._bridge_display_trail(
            values, [2., 2.05, 2.3, 2.4], [False] * 4)
        self.assertEqual(indices, [1, 2])
        np.testing.assert_allclose(filled, [values[0], [.125, .25, 1.25],
                                           [.75, 1.5, 2.5], values[-1]])

    def test_inclusive_half_second_boundary_and_longer_gap(self):
        values = [[0., 0., 1.], None, [1., 0., 1.]]
        for duration in (.499, .5, .5 + .5e-9):
            with self.subTest(duration=duration):
                filled, indices = replay._bridge_display_trail(
                    values, [0., duration / 2, duration], [False] * 3)
                self.assertEqual(indices, [1])
                np.testing.assert_allclose(filled[1], [.5, 0., 1.])
        filled, indices = replay._bridge_display_trail(
            values, [0., .25, .50001], [False] * 3)
        self.assertEqual(filled, values)
        self.assertEqual(indices, [])

    def test_leading_and_trailing_missing_values_are_never_extrapolated(self):
        values = [None, [0., 0., 1.], None, [1., 0., 1.], None, None]
        filled, indices = replay._bridge_display_trail(
            values, np.arange(6) / 30., [False] * 6)
        self.assertEqual(indices, [2])
        self.assertIsNone(filled[0])
        self.assertEqual(filled[4:], [None, None])

    def test_no_connection_before_real_right_endpoint_arrives(self):
        values = [[0., 0., 1.], None, None, [1., 0., 1.]]
        timestamps = np.arange(4) / 30.
        for count in (1, 2, 3):
            filled, indices = replay._bridge_display_trail(
                values[:count], timestamps[:count], [False] * count)
            self.assertEqual(filled, values[:count])
            self.assertEqual(indices, [])
        filled, indices = replay._bridge_display_trail(
            values, timestamps, [False] * 4)
        self.assertEqual(indices, [1, 2])
        self.assertTrue(all(value is not None for value in filled))

    def test_short_separate_gaps_require_their_own_real_endpoints(self):
        values = [[0., 0., 1.], None, [.2, 0., 1.], None, [.4, 0., 1.]]
        filled, indices = replay._bridge_display_trail(
            values, np.arange(5) / 10., [False] * 5)
        self.assertEqual(indices, [1, 3])
        np.testing.assert_allclose(filled, [[i / 10., 0., 1.] for i in range(5)])

    def test_any_intervening_hard_break_prevents_bridge(self):
        values = [[0., 0., 1.], None, None, [1., 0., 1.]]
        for blocked in (1, 2):
            flags = [False] * 4
            flags[blocked] = True
            filled, indices = replay._bridge_display_trail(
                values, np.arange(4) / 30., flags)
            self.assertEqual(filled[1:3], [None, None])
            self.assertEqual(indices, [])

    def test_nonfinite_point_cannot_be_an_endpoint_or_bridge_interior(self):
        for bad in (float('nan'), float('inf'), -float('inf')):
            values = [[0., 0., 1.], None, [bad, 0., 1.], None, [1., 0., 1.]]
            filled, indices = replay._bridge_display_trail(
                values, np.arange(5) / 30., [False] * 5)
            self.assertEqual(indices, [])
            self.assertIsNone(filled[1])
            self.assertIsNone(filled[3])
            self.assertTrue(all(p is None or np.all(np.isfinite(p)) for p in filled))

    def test_invalid_or_non_increasing_timestamps_prevent_bridging(self):
        values = [[0., 0., 1.], None, None, [1., 0., 1.]]
        for timestamps in ([0., .1, float('nan'), .3],
                           [0., .1, float('inf'), .3],
                           [0., .1, .1, .3],
                           [0., .2, .1, .3],
                           [0., .1, .2, .2]):
            with self.subTest(timestamps=timestamps):
                filled, indices = replay._bridge_display_trail(
                    values, timestamps, [False] * 4)
                self.assertEqual(indices, [])
                self.assertEqual(filled[1:3], [None, None])

    def test_input_values_timestamps_and_flags_are_unchanged(self):
        values = [[0., 0., 1.], None, [1., 0., 1.]]
        timestamps, flags = [0., .1, .2], [False] * 3
        original = copy.deepcopy((values, timestamps, flags))
        filled, indices = replay._bridge_display_trail(values, timestamps, flags)
        self.assertEqual((values, timestamps, flags), original)
        self.assertIsNot(filled, values)
        self.assertEqual(indices, [1])

    def test_empty_or_singleton_trails_do_not_need_special_caller_handling(self):
        self.assertEqual(replay._bridge_display_trail([], [], []), ([], []))
        self.assertEqual(replay._bridge_display_trail([None], [0.], [False]),
                         ([None], []))


class DisplayTrailReplayIntegrationTests(unittest.TestCase):
    def test_wrist_only_gap_is_joined_identically_by_cached_and_direct_replay(self):
        history, actions = [snapshot(i) for i in range(5)], records()
        original_actions = copy.deepcopy(actions)
        cache = replay._TrailReplayCache(history, actions, 30.)
        before, _, _ = replay.trails_at_revision(3, history, actions, history[3], 30.)
        self.assertEqual(before['right'][1:], [None] * 3)
        direct, _, _ = replay.trails_at_revision(4, history, actions, history[4], 30.)
        cached = cache.resolve(4, history[4])[1]
        self.assertEqual(cached, direct)
        np.testing.assert_allclose(direct['right'], [[1., 0., 1.]] * 5)
        # Seeking backwards must not expose the connection that appeared later.
        self.assertEqual(cache.resolve(3, history[3])[1], before)
        self.assertEqual(actions, original_actions)

    def test_camera_failure_is_a_hard_break_even_with_short_wrist_gap(self):
        history, actions = [snapshot(i) for i in range(5)], records()
        history[2] = snapshot(2, state=4)
        direct, _, _ = replay.trails_at_revision(4, history, actions, history[4], 30.)
        cached = replay._TrailReplayCache(history, actions, 30.).resolve(4, history[4])[1]
        self.assertEqual(cached, direct)
        self.assertEqual(direct['right'][1:4], [None] * 3)

    def test_unrelated_historical_map_prevents_a_short_gap_join(self):
        history, actions = [snapshot(i) for i in range(5)], records()
        history[2]['reference'] = 99
        history[-1]['references'].append([99, 1, [0., 0., 0., 0., 0., 0., 1.]])
        direct, _, _ = replay.trails_at_revision(4, history, actions, history[4], 30.)
        cached = replay._TrailReplayCache(history, actions, 30.).resolve(4, history[4])[1]
        self.assertEqual(cached, direct)
        self.assertEqual(direct['right'][1:4], [None] * 3)

    def test_absent_hand_entry_does_not_compress_the_frame_clock(self):
        history, actions = [snapshot(i) for i in range(5)], records()
        actions[2]['hands'] = {}
        direct, _, _ = replay.trails_at_revision(4, history, actions, history[4], 30.)
        cache = replay._TrailReplayCache(history, actions, 30.)
        cached = cache.resolve(4, history[4])[1]
        self.assertEqual(cached, direct)
        self.assertEqual(len(direct['right']), 5)
        np.testing.assert_allclose(direct['right'], [[1., 0., 1.]] * 5)

    def test_bridged_history_is_recomputed_in_current_committed_revision(self):
        history, actions = [snapshot(i) for i in range(5)], records()
        cache = replay._TrailReplayCache(history, actions, 30.)
        original = cache.resolve(4, history[4])[1]['right']
        revision = copy.deepcopy(history[-1])
        revision['references'][0][2][0] = .5
        revision['maps'][0]['revision'] += 1
        corrected = cache.resolve(4, revision)[1]['right']
        direct = replay.trails_at_revision(4, history, actions, revision, 30.)[0]['right']
        self.assertEqual(corrected, direct)
        np.testing.assert_allclose(np.asarray(corrected) - original, [[.5, 0., 0.]] * 5)


if __name__ == '__main__':
    unittest.main()
