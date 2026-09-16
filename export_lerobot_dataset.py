#!/usr/bin/env python3
"""Optional LeRobot v3 export for finalized ego demonstration labels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


POSE_NAMES = ("x", "y", "z", "qw", "qx", "qy", "qz")
SIDES = ("left", "right")
STATE_NAMES = tuple(
    [f"camera_{name}" for name in POSE_NAMES]
    + [f"{side}_wrist_{name}" for side in SIDES for name in POSE_NAMES]
    + [f"{side}_hand_{joint}_{axis}" for side in SIDES for joint in range(21)
       for axis in ("x", "y", "z")]
)
ACTION_NAMES = tuple(
    f"{side}_{name}" for side in SIDES
    for name in ("local_dx", "local_dy", "local_dz", "drx", "dry", "drz")
)
VALIDITY_NAMES = ("camera", "left_wrist", "right_wrist", "left_hand", "right_hand")


def _pose_vector(pose: dict[str, Any] | None) -> np.ndarray:
    if not pose:
        return np.zeros(7, dtype=np.float32)
    values = [*pose["translation_m"], *pose["quaternion_wxyz"]]
    result = np.asarray(values, dtype=np.float32)
    return result if result.shape == (7,) and np.isfinite(result).all() else np.zeros(7, dtype=np.float32)


def _pose_valid(pose: dict[str, Any] | None) -> bool:
    return bool(pose) and bool(np.any(_pose_vector(pose)))


def _hand(record: dict[str, Any], side: str) -> dict[str, Any]:
    hands = record.get("hands") or {}
    tokens = ("left", "_l") if side == "left" else ("right", "_r")
    for name, value in hands.items():
        lowered = name.lower()
        if any(token in lowered for token in tokens):
            return value or {}
    return {}


def _landmarks(hand: dict[str, Any]) -> tuple[np.ndarray, bool]:
    points = (hand.get("joints") or {}).get("world_landmarks_graph_m")
    if points is None:
        return np.zeros(63, dtype=np.float32), False
    values = np.asarray(points, dtype=np.float32)
    valid = values.shape == (21, 3) and np.isfinite(values).all()
    return (values.reshape(-1) if valid else np.zeros(63, dtype=np.float32)), valid


def _same_wrist_segment(current: dict[str, Any], following: dict[str, Any]) -> bool:
    segment = current.get("trajectory_segment_graph")
    return (
        segment is not None
        and segment == following.get("trajectory_segment_graph")
        and current.get("world_submap_id") == following.get("world_submap_id")
    )


def _local_pose_delta(current: dict[str, Any] | None,
                      following: dict[str, Any] | None) -> np.ndarray:
    if not (_pose_valid(current) and _pose_valid(following)):
        return np.zeros(6, dtype=np.float32)
    a, b = _pose_vector(current), _pose_vector(following)
    rotation_a = Rotation.from_quat([a[4], a[5], a[6], a[3]])
    rotation_b = Rotation.from_quat([b[4], b[5], b[6], b[3]])
    translation = rotation_a.inv().apply(b[:3] - a[:3])
    rotation = (rotation_a.inv() * rotation_b).as_rotvec()
    return np.asarray([*translation, *rotation], dtype=np.float32)


def sampled_frame_indices(frame_count: int, source_fps: float,
                          output_fps: int) -> list[int]:
    if frame_count <= 0 or source_fps <= 0 or output_fps <= 0:
        return []
    if output_fps > source_fps + 1e-6:
        raise ValueError("training FPS cannot exceed source FPS")
    count = int(np.floor((frame_count - 1) * output_fps / source_fps)) + 1
    indices = [min(frame_count - 1, round(i * source_fps / output_fps))
               for i in range(count)]
    return list(dict.fromkeys(indices))


def training_rows(records: list[dict[str, Any]], indices: list[int]) -> list[dict[str, np.ndarray]]:
    rows: list[dict[str, np.ndarray]] = []
    map_indices: dict[str, int] = {}
    for position, index in enumerate(indices):
        record = records[index]
        hands = [_hand(record, side) for side in SIDES]
        wrists = [hand.get("wrist_world_graph") for hand in hands]
        landmarks = [_landmarks(hand) for hand in hands]
        camera = record.get("camera_world_pose_fused")
        camera_valid = _pose_valid(camera) and record.get("camera_world_source") != "invalid"
        wrist_valid = [_pose_valid(pose) for pose in wrists]
        validity = np.asarray(
            [camera_valid, *wrist_valid, landmarks[0][1], landmarks[1][1]],
            dtype=np.bool_,
        )
        state = np.concatenate([
            _pose_vector(camera), *(_pose_vector(pose) for pose in wrists),
            landmarks[0][0], landmarks[1][0],
        ]).astype(np.float32)
        next_record = records[indices[position + 1]] if position + 1 < len(indices) else None
        action_parts, action_valid = [], []
        for side, hand, pose in zip(SIDES, hands, wrists):
            following = _hand(next_record, side) if next_record is not None else {}
            valid = bool(
                next_record is not None
                and _pose_valid(pose)
                and _pose_valid(following.get("wrist_world_graph"))
                and _same_wrist_segment(hand, following)
            )
            action_valid.append(valid)
            action_parts.append(_local_pose_delta(
                pose if valid else None,
                following.get("wrist_world_graph") if valid else None,
            ))
        world = str(record.get("world_frame_id") or record.get("camera_submap_id") or "invalid")
        if world != "invalid" and world not in map_indices:
            map_indices[world] = len(map_indices)
        rows.append({
            "observation.state": state,
            "observation.valid_mask": validity,
            "observation.confidence": np.asarray([
                float(record.get("camera_world_confidence") or 0.0),
                float(hands[0].get("confidence") or 0.0),
                float(hands[1].get("confidence") or 0.0),
            ], dtype=np.float32),
            "observation.map_context": np.asarray(
                [map_indices.get(world, -1), int(record.get("map_revision") or -1)],
                dtype=np.int64,
            ),
            "action": np.concatenate(action_parts).astype(np.float32),
            "action.valid_mask": np.asarray(action_valid, dtype=np.bool_),
        })
    return rows


def _features(height: int, width: int) -> dict[str, dict[str, Any]]:
    return {
        "observation.images.ego": {
            "dtype": "video", "shape": (3, height, width),
            "names": ["channels", "height", "width"],
        },
        "observation.state": {
            "dtype": "float32", "shape": (len(STATE_NAMES),), "names": list(STATE_NAMES),
        },
        "observation.valid_mask": {
            "dtype": "bool", "shape": (len(VALIDITY_NAMES),), "names": list(VALIDITY_NAMES),
        },
        "observation.confidence": {
            "dtype": "float32", "shape": (3,), "names": ["camera", "left_wrist", "right_wrist"],
        },
        "observation.map_context": {
            "dtype": "int64", "shape": (2,), "names": ["map_index", "map_revision"],
        },
        "action": {
            "dtype": "float32", "shape": (len(ACTION_NAMES),), "names": list(ACTION_NAMES),
        },
        "action.valid_mask": {
            "dtype": "bool", "shape": (2,), "names": ["left_wrist", "right_wrist"],
        },
    }


def export_lerobot(actions: Path, output: Path, task: str, output_fps: int = 20,
                   repo_id: str | None = None, dataset_class=None) -> Path:
    actions, output = Path(actions), Path(output)
    metadata_path = actions.with_suffix(".meta.json")
    if output.exists():
        raise FileExistsError(f"LeRobot output already exists: {output}")
    metadata = json.loads(metadata_path.read_text())
    video_path = Path(metadata["video"])
    records = [json.loads(line) for line in actions.open() if line.strip()]
    source_fps = float(metadata["fps"])
    indices = sampled_frame_indices(len(records), source_fps, output_fps)
    rows = training_rows(records, indices)
    if dataset_class is None:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise RuntimeError(
                "LeRobot export is optional; run "
                "`scripts/setup_vla_export.sh` once"
            ) from exc
        dataset_class = LeRobotDataset
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open source video: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dataset = dataset_class.create(
        repo_id=repo_id or f"local/{output.name}", root=output,
        fps=output_fps, robot_type="passive-ego-demonstration",
        features=_features(height, width), use_videos=True,
        image_writer_threads=4,
    )
    wanted = iter(zip(indices, rows))
    target = next(wanted, None)
    decoded = 0
    try:
        while target is not None:
            ok, image = capture.read()
            if not ok:
                break
            if decoded == target[0]:
                frame = dict(target[1])
                frame["observation.images.ego"] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                frame["task"] = task
                dataset.add_frame(frame)
                target = next(wanted, None)
            decoded += 1
    finally:
        capture.release()
    if target is not None:
        raise RuntimeError(f"video ended before source frame {target[0]}")
    dataset.save_episode()
    dataset.finalize()
    provenance = {
        "schema": "passive-ego-to-lerobot/v1",
        "source_actions": str(actions.resolve()),
        "source_metadata": str(metadata_path.resolve()),
        "source_video": str(video_path.resolve()),
        "source_actions_sha256": hashlib.sha256(actions.read_bytes()).hexdigest(),
        "task": task,
        "source_fps": source_fps,
        "training_fps": output_fps,
        "source_frames": len(records),
        "training_frames": len(rows),
        "action_semantics": "next-sampled-frame wrist-local SE(3) delta; zero when invalid",
        "missing_data": "zero-filled numeric tensors plus explicit validity masks",
        "direct_robot_control": False,
    }
    (output / "source_provenance.json").write_text(
        json.dumps(provenance, indent=2, allow_nan=False) + "\n"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("actions", type=Path, help="finalized *_actions.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", required=True, help="natural-language episode instruction")
    parser.add_argument("--fps", type=int, default=20, help="training FPS (default: 20)")
    parser.add_argument("--repo-id", help="LeRobot repository ID; defaults to local/<output name>")
    args = parser.parse_args()
    export_lerobot(args.actions, args.output, args.task, args.fps, args.repo_id)
    print(f"wrote optional LeRobot dataset to {args.output}")


if __name__ == "__main__":
    main()
