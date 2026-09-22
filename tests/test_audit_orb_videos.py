import unittest

import numpy as np

from tools.audit_orb_videos import _single_frame_spikes


class OrbAuditDiagnosticsTests(unittest.TestCase):
    @staticmethod
    def pose(x):
        return np.array([x, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])

    def test_constant_velocity_is_not_called_a_spike(self):
        poses = [self.pose(value) for value in (0.0, 0.05, 0.10)]
        self.assertEqual(_single_frame_spikes(poses, [0.0, 0.1, 0.2], ['m'] * 3), [])

    def test_isolated_outlier_is_reported(self):
        poses = [self.pose(value) for value in (0.0, 2.0, 0.02)]
        spikes = _single_frame_spikes(
            poses, [0.0, 0.1, 0.2], ['m'] * 3, sources=['slam'] * 3
        )
        self.assertEqual(len(spikes), 1)
        self.assertGreater(spikes[0]['interpolation_residual_mm'], 1900.0)
        self.assertEqual(spikes[0]['source'], 'slam')

    def test_map_boundary_is_not_compared(self):
        poses = [self.pose(value) for value in (0.0, 2.0, 0.02)]
        self.assertEqual(
            _single_frame_spikes(poses, [0.0, 0.1, 0.2], ['a', 'b', 'b']), []
        )


if __name__ == '__main__':
    unittest.main()
