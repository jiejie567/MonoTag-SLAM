import copy
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np
from aruco_track.models import Calibration
from aruco_track.offline_gap_display import supplement_gap_display

ROOT = Path(__file__).resolve().parents[1]
GAPS = Path(__file__).parent


class GapDisplayTests(unittest.TestCase):
    def test_verified_recovery_visible_without_rewriting_history(self):
        spec = importlib.util.spec_from_file_location('gap_fixture', GAPS / 'test_offline_gap_recovery.py')
        fixture_module = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(GAPS))
        try:
            spec.loader.exec_module(fixture_module)
        finally:
            sys.path.remove(str(GAPS))
        r = fixture_module.fixture()
        q = fixture_module.request(r)
        recovered = fixture_module.apply_gap_candidates(r, q, fixture_module.rows()).frames[1]
        final = dict(r.history[-1], maps=[r.maps['atlas_0']])
        history = r.history[:-1] + [final]
        action = dict(frame=1, timestamp_s=.1, camera_submap_id='atlas_0', world_frame_id='atlas_0',
                      scale_status='metric', map_revision=42, camera_world_source='head-slam',
                      camera_world_confidence=.4, camera_localization_recovery=recovered.localization_recovery,
                      camera_world_pose_fused=dict(translation_m=[0, 0, 1], quaternion_wxyz=[1, 0, 0, 0]))
        cal = Calibration(np.asarray(q['camera_matrix'], float), np.zeros(5), (640, 480))
        before = copy.deepcopy(history)
        display = supplement_gap_display(history, [action], cal, {})
        self.assertEqual(len(display[1]['matched_features']), 40)
        self.assertEqual(display[1]['correspondence_origin'], 'validated-short-gap-pnp')
        self.assertEqual(history, before)
        self.assertIsNone(history[1]['pose'])
        for change in ('pose', 'revision', 'time', 'marker', 'nan'):
            bad = copy.deepcopy(action)
            if change == 'pose': bad['camera_world_pose_fused']['translation_m'][0] = 1.
            if change == 'revision': bad['map_revision'] = 43
            if change == 'time': bad['timestamp_s'] = 1.
            if change == 'marker': bad['camera_submap_id'] = 'atlas_1'
            if change == 'nan': bad['camera_localization_recovery']['matched_features'][0][0] = float('nan')
            self.assertEqual(supplement_gap_display(history, [bad], cal, {}), {}, change)
