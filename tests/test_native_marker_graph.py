"""Production native marker graph regression (RUN_NATIVE_SLAM_TESTS=1)."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import cv2
import numpy as np

from aruco_track.models import BandLayout, Calibration, Pose
from aruco_track.orbslam3_backend import (
    run_orbslam3_sequence, write_orbslam3_settings, write_tag_observation_hints,
)


PROJECT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.environ.get('RUN_NATIVE_SLAM_TESTS') == '1', 'explicit native integration run')
class NativeMarkerGraphTests(unittest.TestCase):
    def run_regression(self, name, *arguments):
        binary = PROJECT / 'third_party/ORB_SLAM3/Examples/Monocular' / name
        self.assertTrue(binary.is_file(), f'build target {name} first')
        result = subprocess.run([str(binary), *(str(argument) for argument in arguments)],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        print(result.stdout.strip())

    def test_interval_scale_raw_corner_joint_ba(self):
        self.run_regression('marker_graph_regression')

    def test_graph_order_does_not_depend_on_inputs_or_allocations(self):
        self.run_regression('marker_graph_regression', '--graph-order-only')

    def test_new_station_corner_scale_preserves_old_anchor(self):
        self.run_regression('marker_graph_regression', '--new-station')

    def test_extreme_new_station_requires_validated_physical_scale(self):
        self.run_regression('marker_graph_regression', '--extreme-new-station')

    def test_extreme_scale_rejects_weak_geometry(self):
        self.run_regression('marker_graph_regression', '--corner-scale-only')

    def test_provisional_marker_state_reaches_final_and_loop_ba(self):
        self.run_regression('marker_graph_regression', '--provisional-marker-only')

    def test_triangulation_cannot_claim_one_target_twice(self):
        self.run_regression('marker_graph_regression', '--triangulation-unique-only')

    def test_final_ba_cannot_drop_unsupported_revisited_marker_evidence(self):
        self.run_regression('marker_graph_coordinator_regression', '--marker-retry-only')

    def test_large_known_marker_loop_requires_independent_scale_evidence(self):
        self.run_regression('marker_graph_regression', '--known-marker-loop-only')

    def test_cheirality_repair_preserves_raw_observations_and_rejects_bad_geometry(self):
        self.run_regression('marker_graph_regression', '--cheirality-repair-only')

    def test_refine_preserves_fixed_gauge_without_pinning_new_markers(self):
        self.run_regression('marker_graph_coordinator_regression', '--refine-gauge-only')

    def test_common_marker_merge_raw_corner_joint_ba(self):
        self.run_regression('marker_map_merge_regression')

    def test_commit_history_ownership_rejection_and_serialization(self):
        self.run_regression('marker_graph_coordinator_regression')

    def test_scale_reanchor_through_native_tracking_scheduler(self):
        with tempfile.TemporaryDirectory() as temporary:
            settings = Path(temporary) / 'camera.yaml'
            camera = np.array([[500., 0, 320], [0, 500., 240], [0, 0, 1]])
            calibration = Calibration(camera, np.zeros(5), (640, 480))
            write_orbslam3_settings(settings, calibration, 30)
            self.run_regression('marker_graph_coordinator_regression', settings)

    def test_loss_new_map_then_common_marker_merge_through_real_tracking(self):
        # Two independent static textures prevent a visual-overlap shortcut.
        # The second map starts from a different marker; only AFTER it has a
        # background and multiple KFs does marker 20 reappear. This exercises
        # actual OnFrameEnd scheduling, mapper stop, proposal and commit.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sequence, output = root / 'sequence', root / 'output'
            sequence.mkdir(); output.mkdir()
            camera = np.array([[500., 0, 320], [0, 500., 240], [0, 0, 1]])
            calibration = Calibration(camera, np.zeros(5), (640, 480))
            square = np.array([[-.024, -.024, 0], [.024, -.024, 0],
                               [.024, .024, 0], [-.024, .024, 0]])
            layout = BandLayout('world', 'DICT_4X4_50', {20: square, 21: square + [.15, 0, 0]})
            textures = [np.random.default_rng(seed).integers(0, 256, (480, 2000), dtype=np.uint8)
                        for seed in (753, 951)]
            stages = ([(0, x, 20) for x in np.linspace(0, .18, 76)]
                      + [(-1, 0., None)] * 220
                      + [(1, x, 21) for x in np.linspace(0, .18, 76)]
                      + [(1, x, 20) for x in np.linspace(.18, 0., 100)])
            poses, confidences, detections, accepted, lines = [], [], [], [], []
            for index, (texture_id, x, marker_id) in enumerate(stages):
                left = 700 + round(x * 1000)
                frame = (np.zeros((480, 640), np.uint8) if texture_id < 0 else
                         textures[texture_id][:, left:left+640])
                name = f'{index:04d}.png'
                cv2.imwrite(str(sequence / name), frame)
                lines.append(f'{index/30:.9f} {name}')
                pose = Pose(np.zeros((3, 1)), np.array([[x], [0.], [-.5]]), 0.)
                if marker_id is None:
                    poses.append(None); confidences.append(0.); detections.append({}); accepted.append(())
                else:
                    pixels = cv2.projectPoints(layout.markers[marker_id], np.zeros(3), np.array([-x, 0., .5]),
                                               camera, np.zeros(5))[0].reshape(4, 2)
                    self.assertTrue(np.all((pixels[:, 0] > 0) & (pixels[:, 0] < 640)))
                    poses.append(pose); confidences.append(.9)
                    detections.append({marker_id: pixels}); accepted.append((marker_id,))
            (sequence / 'rgb.txt').write_text('\n'.join(lines) + '\n')
            write_tag_observation_hints(root / 'tags.txt', poses, confidences, detections, accepted,
                                        layout, calibration, 30, 0, include_ids=True)
            write_orbslam3_settings(root / 'camera.yaml', calibration, 30, save_atlas=root / 'atlas.osa')
            run_orbslam3_sequence(PROJECT, sequence, root / 'camera.yaml', output, root / 'tags.txt')
            history = [json.loads(line) for line in (output / 'frames.txt.history.jsonl').read_text().splitlines()]
            events = history[-1].get('marker_graph_events', [])
            merged = [event for event in events if event['type'] == 'marker_map_merge' and event['status'] == 'accepted']
            details = {'events': events, 'map_keyframes': [(m['id'], len(m['keyframes']), len(m['points']))
                                                        for m in history[-1]['maps']]}
            self.assertTrue(merged, details)
            event = merged[0]
            commit = next(h for h in history if any(e['sequence'] == event['sequence']
                                                  for e in h.get('marker_graph_events', [])))
            self.assertGreaterEqual(commit['timestamp'], 372/30)
            self.assertTrue(any(len([m for m in h['maps'] if m['background']]) >= 2
                                for h in history if h['timestamp'] < commit['timestamp']), details)
            self.assertEqual(commit['active_map'], event['target_map_id'])
            target = next(m for m in commit['maps'] if m['id'] == event['target_map_id'])
            self.assertEqual(set(target['markers']), {'20', '21'})
            self.assertGreater(target.get('point_count', len(target['points'])), 50)
            self.assertFalse(any(m['id'] == event['source_map_id'] for m in commit['maps']))
            for marker_id, corners in layout.markers.items():
                optimized = np.asarray(target['markers'][str(marker_id)]).reshape(4, 3)
                # Marker poses are now graph variables, so a valid BA may move the
                # whole quad slightly.  Its physical geometry must remain rigid.
                expected_lengths = sorted(np.linalg.norm(corners[i] - corners[j])
                                          for i in range(4) for j in range(i + 1, 4))
                actual_lengths = sorted(np.linalg.norm(optimized[i] - optimized[j])
                                        for i in range(4) for j in range(i + 1, 4))
                np.testing.assert_allclose(actual_lengths, expected_lengths, atol=1e-6)
                self.assertLess(np.linalg.norm(optimized.mean(axis=0) - corners.mean(axis=0)), 1e-3)
            # LOST/no-observation frames remain invalid; no interpolation.
            self.assertTrue(any(h['pose'] is None for h in history[90:290]))
            for h in history:
                if h['state'] not in (2, 6): self.assertIsNone(h['pose'])
            self.assertTrue((root / 'atlas.osa').is_file())
            self.assertTrue(history[-1]['marker_graph_capabilities']['marker_map_merge'])
            print('NATIVE_MARKER_MAP_MERGE_TRACKING_OK ' + json.dumps(details, sort_keys=True))


if __name__ == '__main__':
    unittest.main()
