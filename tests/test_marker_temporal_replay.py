import unittest

from aruco_track.slam_replay import _temporally_excluded_marker_ids


class MarkerTemporalReplayTests(unittest.TestCase):
    def test_only_published_rejected_or_pending_states_are_excluded(self):
        action = {'marker_temporal_admission': {
            '24': {'state': 'accepted'}, '27': {'state': 'rejected'},
            '28': {'state': 'pending'}, '29': {'state': 'inconclusive'},
        }}
        self.assertEqual(_temporally_excluded_marker_ids(action), {27, 28})

    def test_legacy_raw_detections_are_not_reinterpreted_during_replay(self):
        self.assertEqual(_temporally_excluded_marker_ids({
            'detected_marker_corners': {'27': [[0, 0]] * 4}
        }), set())


if __name__ == '__main__':
    unittest.main()
