import unittest
import numpy as np

from scripts.audit_reanchor_metric_consistency import square, fit_square, evaluate, classify_event


def synthetic_views(scale=1., stationary=False, translation=None):
    template = square(.048) + [0, 0, 1]
    shift = np.zeros(3) if translation is None else np.asarray(translation)
    views = []
    for i, x in enumerate([-.2, -.1, 0, .1, .2]):
        c = np.array([0 if stationary else x, .01 * i if not stationary else 0, 0])
        relative = template - c
        pixels = relative[:, :2] / relative[:, 2:]
        center = scale * c + shift
        views.append({'id': i, 'center': center, 'pixels': pixels,
                      'projection': np.column_stack((np.eye(3), -center))})
    return views


class MetricSizeAuditTests(unittest.TestCase):
    def test_known_scale_and_direction(self):
        for scale in [1., .8, 1.2, .98, 1.07]:
            result = fit_square(synthetic_views(scale), .048, np.array([1000., 1000.]))
            self.assertAlmostEqual(result['metric_per_reconstructed_unit'], 1/scale, places=8)
            self.assertAlmostEqual(result['reconstructed_edge_mm'], 48*scale, places=7)

    def test_world_translation_does_not_change_size(self):
        result = fit_square(synthetic_views(1.2, translation=[1000, -2000, 300]), .048, np.array([1000., 1000.]))
        self.assertAlmostEqual(result['metric_per_reconstructed_unit'], 1/1.2, places=7)

    def test_metric_geometry_and_residual_drift_classified_separately(self):
        self.assertEqual(evaluate(synthetic_views(1.01), .048, np.array([1000., 1000.]))['status'], 'consistent')
        self.assertEqual(evaluate(synthetic_views(1.07), .048, np.array([1000., 1000.]))['status'], 'residual_scale_error')

    def test_improvement_alone_is_not_success(self):
        before = evaluate(synthetic_views(1.10), .048, np.array([1000., 1000.]))
        after = evaluate(synthetic_views(1.06), .048, np.array([1000., 1000.]))
        self.assertLess(abs(after['size_error_pct']), abs(before['size_error_pct']))
        self.assertEqual(after['status'], 'residual_scale_error')

    def test_stationary_not_verified(self):
        self.assertEqual(evaluate(synthetic_views(stationary=True), .048, np.array([1000., 1000.]))['status'], 'unverified')

    def test_two_views_not_verified(self):
        self.assertEqual(evaluate(synthetic_views()[:2], .048, np.array([1000., 1000.]))['status'], 'unverified')

    def test_duplicate_views_do_not_count_as_new_evidence(self):
        views = synthetic_views()[:2]
        self.assertEqual(evaluate(views * 3, .048, np.array([1000., 1000.]))['status'], 'unverified')

    def test_translation_without_sufficient_parallax_not_verified(self):
        views = synthetic_views()
        for view in views:
            center = view['center'] * .125  # 5 cm baseline but only 1 m depth
            xyz = square(.048) + [0, 0, 5]
            ray = xyz - center
            view.update(center=center, projection=np.column_stack((np.eye(3), -center)), pixels=ray[:, :2]/ray[:, 2:])
        result = evaluate(views, .048, np.array([1000., 1000.]))
        self.assertEqual(result['status'], 'unverified')
        self.assertEqual(result['reason'], 'insufficient_corner_parallax')

    def test_eightfold_error_is_detected_not_silently_passed(self):
        for scale in (8., .125):
            result = evaluate(synthetic_views(scale), .048, np.array([1000., 1000.]))
            self.assertEqual(result['status'], 'residual_scale_error')
            self.assertAlmostEqual(result['metric_per_reconstructed_unit'], 1/scale, places=7)

    def test_noisy_large_drift_is_not_certified(self):
        rng = np.random.default_rng(903)
        for scale in (.8, 1.08, 1.2, 8.):
            for _ in range(25):
                views = synthetic_views(scale)
                for view in views:
                    view['pixels'] += rng.normal(0, .5/1000, (4, 2))
                self.assertNotEqual(evaluate(views, .048, np.array([1000., 1000.]))['status'], 'consistent')

    def test_deformed_corner_not_verified(self):
        views = synthetic_views()
        for view in views:
            view['pixels'][0, 0] += .03
        self.assertEqual(evaluate(views, .048, np.array([1000., 1000.]))['status'], 'unverified')


class MetricEventClassificationTests(unittest.TestCase):
    @staticmethod
    def pair(before, after):
        return {'before': {'status': before}, 'after': {'status': after}}

    def test_already_metric_is_confirmation_not_recovery(self):
        self.assertEqual(classify_event({'28': self.pair('consistent', 'consistent')}, [28]), 'metric_consistency_confirmed')

    def test_verified_error_to_metric_is_recovery(self):
        self.assertEqual(classify_event({'28': self.pair('residual_scale_error', 'consistent')}, [28]), 'scale_restored')

    def test_recovery_label_from_actual_triangulation(self):
        before = evaluate(synthetic_views(1.10), .048, np.array([1000., 1000.]))
        after = evaluate(synthetic_views(1.01), .048, np.array([1000., 1000.]))
        self.assertEqual(classify_event({'28': {'before': before, 'after': after}}, [28]), 'scale_restored')

    def test_commit_with_remaining_error_is_not_recovery(self):
        self.assertEqual(classify_event({'28': self.pair('residual_scale_error', 'residual_scale_error')}, [28]), 'residual_scale_error')

    def test_one_bad_marker_not_averaged_away(self):
        markers = {'28': self.pair('consistent', 'consistent'), '29': self.pair('consistent', 'residual_scale_error')}
        self.assertEqual(classify_event(markers, [28, 29]), 'residual_scale_error')

    def test_missing_or_uncertain_evidence_not_certified(self):
        self.assertEqual(classify_event({}, [28]), 'scale_unverified')
        self.assertEqual(classify_event({'28': self.pair('consistent', 'consistent')}, [28, 29]), 'scale_unverified')
        self.assertEqual(classify_event({'28': self.pair('consistent', 'unverified')}, [28]), 'scale_unverified')

    def test_unknown_before_does_not_prove_recovery(self):
        self.assertEqual(classify_event({'28': self.pair('unverified', 'consistent')}, [28]), 'metric_consistency_verified')


if __name__ == '__main__':
    unittest.main()
