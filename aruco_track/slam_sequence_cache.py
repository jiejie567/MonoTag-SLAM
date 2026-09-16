from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import cv2

from .models import BandLayout, Calibration, Pose


SEQUENCE_CACHE_VERSION = "native-input-v5"


def sequence_cache_root(project_dir: Path) -> Path:
    """Allow fast local scratch storage without changing cache contents."""
    override = os.environ.get("SLAM_SEQUENCE_CACHE_DIR", "").strip()
    return Path(override).expanduser().resolve() if override else project_dir / "output" / ".slam_sequence_cache"


def snapshot_sequence(source: Path, destination: Path) -> None:
    """Pin immutable inputs; avoid thousands of guaranteed EXDEV failures."""
    if source.stat().st_dev != destination.parent.stat().st_dev:
        shutil.copytree(source, destination)
        return
    try:
        shutil.copytree(source, destination, copy_function=os.link)
    except OSError:
        shutil.rmtree(destination, ignore_errors=True)
        shutil.copytree(source, destination)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepared_observation_digest(path: Path) -> str:
    """Hash only raw image measurements that affect prepared pixels/masks.

    Final SLAM/action fields differ between a fresh run and a reuse run. They
    must not prevent the first reuse from hitting the exact same prepared
    sequence. Hand identity also does not affect the union mask, so raw hand
    landmark sets are sorted independently of their old left/right binding.
    """
    digest = hashlib.sha256()
    with path.open() as stream:
        for line_number, line in enumerate(stream):
            if not line.strip():
                continue
            record = json.loads(line)
            hand_landmarks = []
            for hand in record.get("hands", {}).values():
                joints = hand.get("joints", {})
                if joints.get("valid"):
                    hand_landmarks.append(joints.get("image_landmarks_normalized"))
            for joints in record.get("unassigned_hands", []):
                if joints.get("valid"):
                    hand_landmarks.append(joints.get("image_landmarks_normalized"))
            hand_landmarks.sort(
                key=lambda value: json.dumps(value, separators=(",", ":"))
            )
            payload = {
                "line": line_number,
                "frame": record.get("frame"),
                "detected_marker_corners": record.get(
                    "detected_marker_corners", {}
                ),
                "boundary_rejected_marker_corners": record.get(
                    "boundary_rejected_marker_corners", {}
                ),
                "marker_mask_corners": record.get("marker_mask_corners", {}),
                "marker_boundary_quality": record.get(
                    "marker_boundary_quality", {}
                ),
                "hand_image_landmarks": hand_landmarks,
            }
            digest.update(
                json.dumps(
                    payload, sort_keys=True, separators=(",", ":"), allow_nan=False
                ).encode()
            )
            digest.update(b"\n")
    return digest.hexdigest()


def sequence_cache_key(
    video_path: Path,
    observations_path: Path,
    calibration: Calibration,
    marker_layout: BandLayout | None,
    fps: float,
    frame_count: int,
    marker_poses: list[Pose | None] | None = None,
    marker_confidences: list[float] | None = None,
    accepted_marker_ids: list[tuple[int, ...]] | None = None,
    marker_weights: list[dict[int, float]] | None = None,
    detections: list[dict[int, object]] | None = None,
    marker_layouts: dict[str, BandLayout] | None = None,
    active_submap_ids: list[str | None] | None = None,
    weak_marker_corners: bool = False,
    excluded_marker_ids: list[set[int]] | None = None,
) -> str:
    """Fingerprint every input that can change the prepared native sequence."""
    video = video_path.stat()
    implementation_files = [
        Path(__file__),
        Path(__file__).with_name("camera_state.py"),
        Path(__file__).with_name("marker_corners.py"),
        Path(__file__).with_name("marker_temporal_admission.py"),
        Path(__file__).with_name("orbslam3_backend.py"),
        Path(__file__).resolve().parent.parent / "export_action_labels.py",
    ]
    optional_lengths = {
        len(values)
        for values in (
            marker_poses,
            marker_confidences,
            accepted_marker_ids,
            marker_weights,
            detections,
            active_submap_ids,
            excluded_marker_ids,
        )
        if values is not None
    }
    core_marker_streams = (
        marker_poses,
        marker_confidences,
        accepted_marker_ids,
    )
    if any(value is not None for value in core_marker_streams) and not all(
        value is not None for value in core_marker_streams
    ):
        raise ValueError(
            "marker poses, confidences and accepted IDs must be cached together"
        )
    if marker_weights is not None and marker_poses is None:
        raise ValueError("marker weights require cached marker poses")
    if optional_lengths and optional_lengths != {frame_count}:
        raise ValueError("marker hint streams must match the prepared frame count")
    marker_hints = None
    if marker_poses is not None:
        marker_hints = []
        for index, pose in enumerate(marker_poses):
            marker_hints.append(
                {
                    "pose": (
                        {
                            "rvec": pose.rvec.reshape(3).tolist(),
                            "tvec": pose.tvec.reshape(3).tolist(),
                            "error_px": float(pose.reprojection_error_px),
                            "marker_ids": list(pose.marker_ids),
                            "inliers": int(pose.inlier_count),
                            "ambiguous": bool(pose.ambiguous),
                        }
                        if pose is not None
                        else None
                    ),
                    "confidence": (
                        float(marker_confidences[index])
                        if marker_confidences is not None
                        else None
                    ),
                    "accepted_ids": (
                        list(accepted_marker_ids[index])
                        if accepted_marker_ids is not None
                        else None
                    ),
                    "weights": (
                        {
                            str(marker_id): float(weight)
                            for marker_id, weight in sorted(marker_weights[index].items())
                        }
                        if marker_weights is not None
                        else None
                    ),
                }
            )
    payload = {
        "version": SEQUENCE_CACHE_VERSION,
        "opencv": cv2.__version__,
        # Prepared pixels and marker hints are immutable cache outputs. Include
        # the small implementation sources so an algorithm edit cannot silently
        # reuse pixels generated by an older build.
        "implementation_sha256": {
            path.name: _file_digest(path) for path in implementation_files
        },
        "video": {
            "path": str(video_path.resolve()),
            "size": video.st_size,
            "mtime_ns": video.st_mtime_ns,
            "ctime_ns": video.st_ctime_ns,
        },
        "prepared_observations_sha256": _prepared_observation_digest(
            observations_path
        ),
        "camera_matrix": calibration.camera_matrix.tolist(),
        "dist_coeffs": calibration.dist_coeffs.reshape(-1).tolist(),
        "image_size": list(calibration.image_size),
        "marker_layout": (
            {
                "name": marker_layout.name,
                "dictionary": marker_layout.dictionary,
                "markers": {
                    str(marker_id): marker_layout.markers[marker_id].tolist()
                    for marker_id in sorted(marker_layout.markers)
                },
            }
            if marker_layout is not None
            else None
        ),
        "marker_layouts": (
            {
                submap_id: {
                    "name": layout.name,
                    "dictionary": layout.dictionary,
                    "markers": {
                        str(marker_id): layout.markers[marker_id].tolist()
                        for marker_id in sorted(layout.markers)
                    },
                }
                for submap_id, layout in sorted(marker_layouts.items())
            }
            if marker_layouts is not None
            else None
        ),
        "active_submap_ids": active_submap_ids,
        "weak_marker_corners": bool(weak_marker_corners),
        "excluded_marker_ids": (
            [sorted(ids) for ids in excluded_marker_ids]
            if excluded_marker_ids is not None else None
        ),
        "fps": float(fps),
        "frame_count": int(frame_count),
        "marker_hints": marker_hints,
        # This is an independent runtime argument used by marker corner
        # tracking, tag-hint generation and exclusion masks. Do not rely on a
        # caller convention that it happens to match the observation JSONL.
        "detections": (
            [
                {
                    str(marker_id): corners.tolist()
                    if hasattr(corners, "tolist")
                    else corners
                    for marker_id, corners in sorted(frame.items())
                }
                for frame in detections
            ]
            if detections is not None
            else None
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _tree_digest(path: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    count = 0
    total_bytes = 0
    for item in sorted(value for value in path.rglob("*") if value.is_file()):
        if item.name == "manifest.json":
            continue
        relative = item.relative_to(path).as_posix().encode()
        size = item.stat().st_size
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        digest.update(size.to_bytes(8, "little"))
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        count += 1
        total_bytes += size
    return digest.hexdigest(), count, total_bytes


@dataclass(frozen=True)
class SequenceCacheEntry:
    path: Path
    hit: bool


class SlamSequenceCache:
    """Small, verified LRU for the expensive JPEG/mask native input tree."""

    def __init__(
        self,
        root: Path,
        max_bytes: int = 2_500_000_000,
        max_entries: int = 4,
        active_grace_s: float = 10 * 60,
    ):
        self.root = root
        self.max_bytes = max(0, int(max_bytes))
        self.max_entries = max(1, int(max_entries))
        self.active_grace_s = max(0.0, float(active_grace_s))

    def lookup(self, key: str) -> SequenceCacheEntry | None:
        entry = self.root / key
        manifest_path = entry / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("version") != SEQUENCE_CACHE_VERSION or manifest.get("key") != key:
                return None
            digest, count, total_bytes = _tree_digest(entry)
            if (
                digest != manifest.get("tree_sha256")
                or count != manifest.get("file_count")
                or total_bytes != manifest.get("total_bytes")
            ):
                return None
            if not (entry / "rgb.txt").is_file() or not (entry / "tag_observations.txt").is_file():
                return None
        except (OSError, ValueError, TypeError):
            return None
        os.utime(entry, None)
        return SequenceCacheEntry(entry, True)

    def staging_directory(self, key: str) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        self.prune()
        destination = self.root / key
        if destination.exists() and self.lookup(key) is None:
            # Entries are immutable and content addressed. A failed digest means
            # this exact directory is unusable, so replace only that entry.
            self.discard(destination)
        return Path(tempfile.mkdtemp(prefix=f".{key}.", dir=self.root))

    def publish(self, key: str, staging: Path) -> SequenceCacheEntry:
        digest, count, total_bytes = _tree_digest(staging)
        manifest = {
            "version": SEQUENCE_CACHE_VERSION,
            "key": key,
            "tree_sha256": digest,
            "file_count": count,
            "total_bytes": total_bytes,
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
        destination = self.root / key
        try:
            staging.rename(destination)
        except FileExistsError:
            # Another process published the same immutable entry first.
            existing = self.lookup(key)
            if existing is None:
                shutil.rmtree(staging)
                raise RuntimeError("concurrent SLAM sequence cache entry is invalid")
            shutil.rmtree(staging)
            return existing
        os.utime(destination, None)
        self.prune(keep=destination)
        return SequenceCacheEntry(destination, False)

    def discard(self, staging: Path) -> None:
        try:
            staging.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("refusing to remove a staging directory outside the cache") from exc
        if staging.exists():
            shutil.rmtree(staging)

    def prune(self, keep: Path | None = None) -> None:
        if not self.root.is_dir():
            return
        entries = []
        for path in self.root.iterdir():
            manifest_path = path / "manifest.json"
            if not path.is_dir():
                continue
            if not manifest_path.is_file():
                # Interrupted publications have no manifest. Never touch a
                # currently-running staging tree; reap only abandoned ones.
                try:
                    age_s = time.time() - path.stat().st_mtime
                    is_staging = path.name.startswith(".")
                    if is_staging and age_s > 24 * 60 * 60:
                        shutil.rmtree(path)
                except OSError:
                    pass
                continue
            try:
                manifest = json.loads(manifest_path.read_text())
                size = int(manifest.get("total_bytes", 0))
            except (OSError, ValueError, TypeError):
                continue
            entries.append((path.stat().st_mtime_ns, size, path))
        entries.sort(reverse=True)
        kept_bytes = 0
        kept_count = 0
        now_ns = time.time_ns()
        for modified_ns, size, path in entries:
            # lookup() touches an entry immediately before use.  Keep recently
            # touched immutable trees even if the nominal LRU budget is
            # temporarily exceeded, so another video process cannot prune a
            # sequence while the native runner is reading it.
            recently_used = (
                self.active_grace_s > 0
                and (now_ns - modified_ns) / 1e9 < self.active_grace_s
            )
            retain = path == keep or recently_used or (
                kept_count < self.max_entries and kept_bytes + size <= self.max_bytes
            )
            if retain:
                kept_count += 1
                kept_bytes += size
            else:
                shutil.rmtree(path)
