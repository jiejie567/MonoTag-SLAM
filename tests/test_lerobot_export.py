import unittest

import numpy as np

from tools.export_lerobot_dataset import (
    ACTION_NAMES, STATE_NAMES, sampled_frame_indices, training_rows,
)


def pose(x=0.0, y=0.0, z=0.0):
    return {
        "translation_m": [x, y, z],
        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
    }


def record(frame, left=None, right=None, camera=None):
    def hand(value, segment):
        return {
            "wrist_world_graph": value,
            "world_submap_id": "atlas_0",
            "trajectory_segment_graph": segment,
            "confidence": 0.9 if value else 0.0,
            "joints": {"world_landmarks_graph_m": None},
        }
    return {
        "frame": frame,
        "camera_world_pose_fused": camera,
        "camera_world_source": "head-slam" if camera else "invalid",
        "camera_world_confidence": 1.0 if camera else 0.0,
        "world_frame_id": "atlas_0" if camera else None,
        "hands": {
            "strap_band_L": hand(left, 1 if left else None),
            "strap_band_R": hand(right, 2 if right else None),
        },
    }


class LeRobotExportTest(unittest.TestCase):
    def test_sampling_is_monotonic_and_does_not_upsample(self):
        self.assertEqual(sampled_frame_indices(7, 60.0, 20), [0, 3, 6])
        with self.assertRaises(ValueError):
            sampled_frame_indices(7, 30.0, 60)

    def test_rows_have_fixed_shapes_masks_and_local_wrist_actions(self):
        records = [
            record(0, left=pose(), camera=pose()),
            record(1, left=pose(0.1, 0.0, 0.0), camera=pose(0.01, 0.0, 0.0)),
        ]
        rows = training_rows(records, [0, 1])
        self.assertEqual(rows[0]["observation.state"].shape, (len(STATE_NAMES),))
        self.assertEqual(rows[0]["action"].shape, (len(ACTION_NAMES),))
        np.testing.assert_allclose(rows[0]["action"][:3], [0.1, 0.0, 0.0], atol=1e-6)
        np.testing.assert_array_equal(rows[0]["observation.valid_mask"],
                                      [True, True, False, False, False])
        np.testing.assert_array_equal(rows[0]["action.valid_mask"], [True, False])
        np.testing.assert_array_equal(rows[-1]["action.valid_mask"], [False, False])
        np.testing.assert_allclose(rows[-1]["action"], 0.0)

    def test_segment_change_does_not_create_false_action(self):
        records = [record(0, left=pose(), camera=pose()),
                   record(1, left=pose(1.0, 0.0, 0.0), camera=pose())]
        records[1]["hands"]["strap_band_L"]["trajectory_segment_graph"] = 9
        rows = training_rows(records, [0, 1])
        self.assertFalse(rows[0]["action.valid_mask"][0])
        np.testing.assert_allclose(rows[0]["action"][:6], 0.0)


if __name__ == "__main__":
    unittest.main()
