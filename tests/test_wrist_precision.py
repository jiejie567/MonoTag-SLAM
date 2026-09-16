import argparse
import copy
import contextlib
import json
import io
import unittest
from unittest.mock import patch

from aruco_track.wrist_precision import (
    add_wrist_precision_arguments, precision_fields_from_quality,
    wrist_precision_fields,
)


class WristPrecisionTests(unittest.TestCase):
    def quality(self, sigma):
        return {'status': 'ok', 'joint': {'translation_sigma_max_m': sigma}}

    def test_budget_and_boundary(self):
        for sigma, expected in ((.009, True), (.010, True), (.011, False)):
            result = precision_fields_from_quality(self.quality(sigma), world_valid=True)
            self.assertIs(result['wrist_precision_qualified'], expected)

    def test_unknown_not_false_or_true(self):
        for quality in ({'status': 'unavailable'}, self.quality(float('nan')),
                        self.quality(float('inf')), self.quality(-1)):
            result = precision_fields_from_quality(quality, world_valid=True)
            self.assertIsNone(result['wrist_precision_qualified'])
            json.dumps(result, allow_nan=False)
        result = precision_fields_from_quality(self.quality(.001), world_valid=False)
        self.assertIsNone(result['wrist_precision_qualified'])

    def test_configurable_not_frame_specific(self):
        self.assertTrue(precision_fields_from_quality(self.quality(.0179),
                        world_valid=True, budget_mm=20)['wrist_precision_qualified'])

    def test_preserves_quality_and_existing_labels(self):
        quality = self.quality(.018)
        saved = copy.deepcopy(quality)
        original = {'valid': True, 'wrist_world_graph': {'translation_m': [.039, 0, 0]}}
        output = copy.deepcopy(original)
        output.update(precision_fields_from_quality(quality, world_valid=True))
        self.assertEqual(quality, saved)
        self.assertEqual({k: output[k] for k in original}, original)
        self.assertFalse(output['wrist_precision_qualified'])

    def test_explicit_model_limitations(self):
        detail = precision_fields_from_quality(self.quality(.001), world_valid=True)['wrist_precision']
        self.assertEqual(detail['calibration_status'], 'assumed_noise_not_empirically_calibrated')
        self.assertFalse(detail['changes_pose_or_validity'])

    def test_no_world_skips_fitting(self):
        with patch('aruco_track.wrist_precision.wrist_geometry_quality') as fit:
            result = wrist_precision_fields({}, (), None, None, None, world_valid=False)
            fit.assert_not_called()
        self.assertIsNone(result['wrist_precision_qualified'])

    def test_passes_only_accepted_ids_and_noise(self):
        with patch('aruco_track.wrist_precision.wrist_geometry_quality',
                   return_value=self.quality(.003)) as fit:
            result = wrist_precision_fields({1: 'retained', 2: 'rejected'}, (1,),
                                           'layout', 'cal', 'seed', world_valid=True,
                                           corner_sigma_px=.7)
            self.assertEqual(fit.call_args.args[1], (1,))
            self.assertEqual(fit.call_args.args[-1], .7)
            self.assertTrue(result['wrist_precision_qualified'])

    def test_cli_defaults_and_invalid(self):
        parser = argparse.ArgumentParser()
        add_wrist_precision_arguments(parser)
        args = parser.parse_args([])
        self.assertEqual(args.wrist_precision_budget_mm, 10)
        self.assertEqual(args.wrist_corner_sigma_px, .5)
        for option in ('--wrist-precision-budget-mm', '--wrist-corner-sigma-px'):
            for value in ('0', '-1', 'nan', 'inf'):
                with self.assertRaises(SystemExit):
                    # argparse is expected to reject these values; keep the
                    # contract suite quiet so real diagnostics remain visible.
                    with contextlib.redirect_stderr(io.StringIO()):
                        parser.parse_args([option, value])


if __name__ == '__main__':
    unittest.main()
