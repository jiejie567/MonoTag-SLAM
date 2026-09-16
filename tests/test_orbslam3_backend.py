import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from aruco_track.models import BandLayout, Calibration, Pose
from aruco_track.orbslam3_backend import (
    MetricOrbSlamResult,
    OrbSlamObservation,
    first_reliable_marker_frame,
    refine_final_frame_poses,
    run_orbslam3_sequence,
    write_orbslam3_settings,
    write_tag_observation_hints,
)
from aruco_track.camera_state import FusedCameraFrame


def pose(rotation: np.ndarray, translation: np.ndarray, error: float = 0.5) -> Pose:
    return Pose(
        cv2.Rodrigues(rotation)[0],
        np.asarray(translation, dtype=np.float64).reshape(3, 1),
        error,
    )


class OrbSlam3RunnerEnvironmentTests(unittest.TestCase):
    def assert_flow_recovery_environment(self, inherited, expected):
        with TemporaryDirectory() as directory:
            project = Path(directory)
            binary = project / "third_party/ORB_SLAM3/Examples/Monocular/mono_tum_headless"
            binary.parent.mkdir(parents=True)
            binary.touch()
            (project / "rgb.txt").write_text("0.0 rgb/000000.png\n")
            with patch.dict(os.environ, inherited, clear=True), patch(
                "aruco_track.orbslam3_backend.subprocess.run",
                return_value=SimpleNamespace(returncode=0, stdout="mock native runner\n"),
            ) as run:
                run_orbslam3_sequence(project, project, project / "camera.yaml", project)
                run.assert_called_once()
                environment = run.call_args.kwargs["env"]
                self.assertEqual(environment["ORB_SLAM3_FLOW_RECOVERY"], expected)
                self.assertNotIn("ORB_SLAM3_TEMPORAL_FLOW", environment)
                self.assertEqual(dict(os.environ), inherited)

    def test_offline_runner_defaults_flow_recovery_to_enabled(self):
        self.assert_flow_recovery_environment({}, "1")

    def test_offline_runner_preserves_explicit_flow_recovery_disable(self):
        self.assert_flow_recovery_environment({"ORB_SLAM3_FLOW_RECOVERY": "0"}, "0")


class OrbSlam3MetricFusionTests(unittest.TestCase):
    def test_dense_refinement_gates_exported_world_camera_centre(self):
        calibration = Calibration(
            np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]),
            np.zeros(5), (640, 480),
        )
        world = np.array([
            [x, y, z]
            for z in (2.5, 3.0, 3.5)
            for x, y in ((-0.4, -0.3), (0.4, -0.3), (-0.4, 0.3), (0.4, 0.3))
        ])
        initial_twc = pose(np.eye(3), [1.0, 0.0, 0.0], 0.0)
        candidate_rvec = np.array([0.0, 0.0, 0.04])
        candidate_tvec = np.array([-1.0, 0.0, 0.0])
        pixels = cv2.projectPoints(
            world, candidate_rvec, candidate_tvec,
            calibration.camera_matrix, np.zeros(5),
        )[0].reshape(-1, 2)
        observation = OrbSlamObservation(
            len(world), pixels, np.empty((0, 2)), 2, np.arange(len(world)),
        )
        result = MetricOrbSlamResult(
            [FusedCameraFrame(initial_twc, "head-slam", 1.0, len(world), None,
                              "atlas_0", 0, True, "marker", True)],
            world, (), [observation], 0, 1.0, 1, None, None, {}, [],
            {"atlas_0": {"points": [[i, *point] for i, point in enumerate(world)],
                         "markers": {}}},
        )
        optimized = np.r_[candidate_rvec, candidate_tvec]
        with patch(
            "aruco_track.orbslam3_backend.least_squares",
            return_value=SimpleNamespace(x=optimized, success=True),
        ):
            refined = refine_final_frame_poses(
                result, calibration, [{}], [()], [{}],
            )
        self.assertIs(refined.frames[0].pose, initial_twc)
        self.assertEqual(refined.timing["dense_pose_accepted"], 0.0)

    def test_settings_enable_metric_tag_fusion_and_sparse_keyframes(self):
        calibration = Calibration(
            np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]),
            np.zeros(5),
            (640, 480),
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "camera.yaml"
            write_orbslam3_settings(path, calibration, 60.0)
            settings = path.read_text()
        self.assertIn("TagFusion.enabled: 1", settings)
        self.assertIn("TagFusion.minimumTranslationM: 0.03", settings)
        self.assertIn("TagFusion.minimumRotationDeg: 7.0", settings)
        self.assertIn("ORBextractor.dynamicGeometry: 0", settings)
        self.assertIn("Offline.synchronousMapping: 1", settings)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "camera.yaml"
            write_orbslam3_settings(
                path, calibration, 60., dynamic_geometry=True,
            )
            enabled = path.read_text()
            self.assertIn("ORBextractor.dynamicGeometry: 1", enabled)

    def test_tag_sidecar_contains_metric_corners_and_invalid_frames(self):
        calibration = Calibration(
            np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]),
            np.zeros(5),
            (640, 480),
        )
        world_points = np.array(
            [
                [-0.02, -0.02, 0.0],
                [0.02, -0.02, 0.0],
                [0.02, 0.02, 0.0],
                [-0.02, 0.02, 0.0],
            ]
        )
        image_points = np.array(
            [[300.0, 220.0], [340.0, 220.0], [340.0, 260.0], [300.0, 260.0]]
        )
        layout = BandLayout("world", "DICT_4X4_50", {20: world_points})
        camera_pose = pose(np.eye(3), [0.0, 0.0, 0.5])
        with TemporaryDirectory() as directory:
            path = Path(directory) / "tags.txt"
            write_tag_observation_hints(
                path,
                [camera_pose, None],
                [0.9, 0.0],
                [{20: image_points}, {}],
                [(20,), ()],
                layout,
                calibration,
                60.0,
                0,
            )
            lines = path.read_text().splitlines()
        valid = lines[1].split()
        self.assertEqual(valid[1], "1")
        self.assertEqual(valid[10], "4")
        self.assertEqual(len(valid), 11 + 4 * 5)
        np.testing.assert_allclose(
            np.asarray(valid[11:], dtype=float).reshape(4, 5)[:, :3], world_points
        )
        self.assertEqual(lines[2].split()[1], "0")


    def test_sidecar_transmits_each_markers_corner_weights(self):
        calibration = Calibration(np.eye(3), np.zeros(5), (640, 480))
        points = np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)
        layout = BandLayout("world", "DICT_4X4_50", {20: points, 21: points + [2, 0, 0]})
        detections = {20: points[:, :2], 21: points[:, :2] + [2, 0]}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "tags.txt"
            write_tag_observation_hints(
                path, [pose(np.eye(3), [0, 0, 0.5])], [0.9], [detections], [(20, 21)],
                layout, calibration, 60.0, 0, marker_weights=[{20: 1.0, 21: 0.25}],
            )
            values = path.read_text().splitlines()[1].split()
        self.assertEqual(values[10], "8")
        split = values.index("weights")
        self.assertEqual(split, 11 + 8 * 5)
        np.testing.assert_allclose(np.array(values[split + 1:], dtype=float),
                                   [1.0] * 4 + [0.25] * 4)

    def test_sidecar_uses_each_disconnected_marker_components_local_layout(self):
        calibration = Calibration(np.eye(3), np.zeros(5), (640, 480))
        square = np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)
        layouts = {
            "room_a": BandLayout("room_a", "DICT_4X4_50", {48: square}),
            "room_b": BandLayout("room_b", "DICT_4X4_50", {20: square + [5, 0, 0]}),
        }
        detections = [{48: square[:, :2]}, {20: square[:, :2] + [5, 0]}]
        camera_pose = pose(np.eye(3), [0, 0, 0.5])
        with TemporaryDirectory() as directory:
            path = Path(directory) / "tags.txt"
            write_tag_observation_hints(
                path, [camera_pose, camera_pose], [0.9, 0.9], detections,
                [(48,), (20,)], BandLayout("empty", "DICT_4X4_50", {}),
                calibration, 60.0, 0, include_ids=True,
                marker_layouts=layouts, marker_component_ids=["room_a", "room_b"],
            )
            lines = [line.split() for line in path.read_text().splitlines()[1:]]
        for values, component, expected in zip(lines, ("room_a", "room_b"),
                                                (square, square + [5, 0, 0])):
            count = int(values[10])
            np.testing.assert_allclose(
                np.asarray(values[11:11 + count * 5], dtype=float).reshape(count, 5)[:, :3],
                expected,
            )
            self.assertEqual(values[-2:], ["component", component])




    def test_unreliable_early_marker_does_not_open_initialization_gate(self):
        markers = [pose(np.eye(3), [0.0, 0.0, 0.5]) for _ in range(4)]
        self.assertEqual(
            first_reliable_marker_frame(markers, [0.1, 0.2, 0.9, 1.0]), 2
        )


if __name__ == "__main__":
    unittest.main()
