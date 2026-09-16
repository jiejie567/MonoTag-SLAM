#!/usr/bin/env python3
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from aruco_track.auto_marker_map import (
    build_auto_marker_map,
    build_session_marker_map,
    localize_auto_marker_frames,
    _robust_edge,
)
from aruco_track.models import Calibration, Pose
from aruco_track.pose import square_object_points


class AutoMarkerMapTests(unittest.TestCase):
    def setUp(self):
        self.calibration = Calibration(
            np.array(
                [[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]]
            ),
            np.zeros(5),
            (1280, 720),
        )
        self.marker_size_m = 0.05
        self.local = square_object_points(self.marker_size_m)

    def _project(
        self, camera_from_world: Pose, world_from_marker: Pose
    ) -> np.ndarray:
        world_points = (
            world_from_marker.rotation_matrix @ self.local.T
        ).T + world_from_marker.tvec.reshape(1, 3)
        image, _ = cv2.projectPoints(
            world_points,
            camera_from_world.rvec,
            camera_from_world.tvec,
            self.calibration.camera_matrix,
            self.calibration.dist_coeffs,
        )
        return image.reshape(4, 2)

    def _detections(self) -> list[dict[int, np.ndarray]]:
        first = Pose(np.zeros((3, 1)), np.zeros((3, 1)), 0.0)
        second = Pose(
            np.array([[0.0], [0.08], [0.02]]),
            np.array([[0.13], [0.015], [0.0]]),
            0.0,
        )
        isolated = Pose(np.zeros((3, 1)), np.zeros((3, 1)), 0.0)
        frames = []
        for index in range(12):
            camera = Pose(
                np.array([[0.04 * np.sin(index / 4.0)], [0.10 * np.cos(index / 5.0)], [0.0]]),
                np.array([[-0.10 + 0.018 * index], [-0.01], [0.62]]),
                0.0,
            )
            frames.append(
                {
                    20: self._project(camera, first),
                    21: self._project(camera, second),
                }
            )
        for index in range(6):
            camera = Pose(
                np.zeros((3, 1)),
                np.array([[0.002 * index], [0.0], [0.55]]),
                0.0,
            )
            frames.append({30: self._project(camera, isolated)})
        return frames

    def test_covisible_markers_share_submap_and_isolated_marker_does_not(self):
        marker_map = build_auto_marker_map(
            self._detections(),
            self.calibration,
            {20, 21, 30},
            self.marker_size_m,
        )

        self.assertEqual(len(marker_map.submaps), 2)
        mapping = marker_map.marker_to_submap
        self.assertEqual(mapping[20], mapping[21])
        self.assertNotEqual(mapping[20], mapping[30])
        first = next(submap for submap in marker_map.submaps if 20 in submap.marker_poses)
        self.assertEqual(first.anchor_marker_id, 20)
        np.testing.assert_allclose(
            first.marker_poses[21].tvec.reshape(3),
            [0.13, 0.015, 0.0],
            atol=2e-3,
        )
        self.assertGreaterEqual(first.edges[0].observations, 6)
        self.assertGreaterEqual(first.edges[0].viewpoint_span_deg, 8.0)

    def test_localization_and_saved_map_keep_submap_identity(self):
        detections = self._detections()
        marker_map = build_auto_marker_map(
            detections,
            self.calibration,
            {20, 21, 30},
            self.marker_size_m,
        )
        localized = localize_auto_marker_frames(
            marker_map, detections, self.calibration
        )

        self.assertTrue(
            all(
                value == marker_map.marker_to_submap[20]
                for value in localized.submap_ids[:12]
            )
        )
        self.assertTrue(
            all(
                value == marker_map.marker_to_submap[30]
                for value in localized.submap_ids[12:]
            )
        )
        self.assertTrue(all(result is not None for result in localized.results))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "scene_marker_map.json"
            marker_map.save(output)
            data = json.loads(output.read_text())
        self.assertEqual(data["mode"], "reliable-covisibility-local-submaps")
        self.assertEqual(len(data["submaps"]), 2)

    def test_repeated_covisibility_without_viewpoint_change_stays_disconnected(self):
        first = Pose(np.zeros((3, 1)), np.zeros((3, 1)), 0.0)
        second = Pose(
            np.zeros((3, 1)), np.array([[0.13], [0.0], [0.0]]), 0.0
        )
        camera = Pose(
            np.zeros((3, 1)), np.array([[0.0], [0.0], [0.62]]), 0.0
        )
        detections = [
            {20: self._project(camera, first), 21: self._project(camera, second)}
            for _ in range(10)
        ]

        marker_map = build_auto_marker_map(
            detections,
            self.calibration,
            {20, 21},
            self.marker_size_m,
        )

        self.assertEqual(len(marker_map.submaps), 2)
        self.assertNotEqual(
            marker_map.marker_to_submap[20], marker_map.marker_to_submap[21]
        )

    def test_session_map_never_turns_disconnected_markers_into_extra_maps(self):
        marker_map = build_session_marker_map(
            self._detections(),
            self.calibration,
            {20, 21, 30},
            self.marker_size_m,
        )

        self.assertEqual(len(marker_map.submaps), 1)
        self.assertEqual(marker_map.submaps[0].submap_id, "session_000")
        self.assertEqual(set(marker_map.submaps[0].marker_poses), {20, 21})
        self.assertEqual(marker_map.pending_marker_ids, (30,))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "session_map.json"
            marker_map.save(output)
            data = json.loads(output.read_text())
        self.assertEqual(data["mode"], "single-session-marker-map")
        self.assertEqual(data["pending_marker_ids"], [30])
        self.assertEqual(len(data["submaps"]), 1)

    def test_session_map_can_register_repeated_covisible_markers(self):
        first = Pose(np.zeros((3, 1)), np.zeros((3, 1)), 0.0)
        second = Pose(
            np.zeros((3, 1)), np.array([[0.13], [0.0], [0.0]]), 0.0
        )
        camera = Pose(
            np.zeros((3, 1)), np.array([[0.0], [0.0], [0.62]]), 0.0
        )
        detections = [
            {20: self._project(camera, first), 21: self._project(camera, second)}
            for _ in range(10)
        ]

        marker_map = build_session_marker_map(
            detections,
            self.calibration,
            {20, 21},
            self.marker_size_m,
        )

        self.assertEqual(len(marker_map.submaps), 1)
        self.assertEqual(set(marker_map.submaps[0].marker_poses), {20, 21})
        self.assertEqual(marker_map.pending_marker_ids, ())
        experimental = build_auto_marker_map(
            detections, self.calibration, {20, 21}, self.marker_size_m,
            single_session=True, verify_ambiguous_edges=True)
        self.assertEqual(set(experimental.submaps[0].marker_poses), {20})
        self.assertEqual(experimental.pending_marker_ids, (21,))

    def test_multiview_pixel_evidence_not_small_rotation_selects_branch(self):
        low_rotation = Pose(np.zeros((3, 1)), np.array([[.13], [0], [0]]), 0.)
        supported = Pose(np.array([[0.], [.5], [0.]]), np.array([[.13], [0], [0]]), 0.)
        observations = [(i, pose, np.array([0., 0., 1.]))
                        for i in range(8) for pose in (low_rotation, supported)]
        for values in (observations, list(reversed(observations))):
            edge = _robust_edge(20, 21, values, 6, 0.,
                                lambda p: 2. if np.linalg.norm(p.rvec) < .1 else .1)
            self.assertIsNotNone(edge)
            np.testing.assert_allclose(edge.first_from_second.rotation_matrix,
                                       supported.rotation_matrix, atol=1e-8)
            self.assertIsNone(_robust_edge(20, 21, values, 6, 0., lambda p: .2))

    def test_unresolved_hypotheses_wait_for_later_evidence(self):
        poses = [Pose(np.array([[0.], [angle], [0.]]),
                      np.array([[.13], [0], [0]]), 0.) for angle in (0., .5)]
        values = [(i, pose, np.array([0., 0., 1.]))
                  for i in range(8) for pose in poses]
        self.assertIsNone(_robust_edge(20, 21, values, 6, 0., lambda p: .2))
        self.assertIsNone(_robust_edge(20, 21, values, 6, 0., lambda p: float('inf')))
        self.assertIsNotNone(_robust_edge(20, 21, values, 6, 0.,
                                         lambda p: .1 if np.linalg.norm(p.rvec) > .1 else 1.))


if __name__ == "__main__":
    unittest.main()
