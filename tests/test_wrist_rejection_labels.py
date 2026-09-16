import unittest
from aruco_track.slam_replay import _rejected_wrist_label


class WristRejectionLabelsTests(unittest.TestCase):
    def test_explains_grid_border_and_candidate_without_claiming_acceptance(self):
        action = {'hands': {'right': {'assist_only_marker_ids': [7]}},
                  'marker_boundary_quality': {
                      '6': {'reason': 'wrist_boundary'},
                      '7': {'reason': 'wrist_grid_mismatch'},
                      '8': {'reason': 'wrist_grid_mismatch'}}}
        self.assertEqual(_rejected_wrist_label(action, '6'), 'BORDER REJECTED')
        self.assertEqual(_rejected_wrist_label(action, 7), 'ASSIST CANDIDATE')
        self.assertEqual(_rejected_wrist_label(action, '8'), 'GRID REJECTED')
        self.assertEqual(_rejected_wrist_label({}, 7), 'REJECTED')


if __name__ == '__main__':
    unittest.main()
