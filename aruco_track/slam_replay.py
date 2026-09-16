from __future__ import annotations

import gzip
import json
from pathlib import Path
import shutil
import struct
import subprocess
import time

import cv2
import numpy as np

from .hands import HAND_CONNECTIONS
from .camera_state import FusedCameraFrame
from .models import BandLayout, Calibration, Pose
from .orbslam3_backend import annotate_anchor_consistency, camera_at_revision, pose_from_native, select_camera_frame
from .pipeline import inverse_pose
from .render import draw_projected_axes, project_frame_axes
from .replay_browser import write_browser_timeline


WORLD_AXIS_LENGTH_M = .05
FINAL_REPLAY_HOLD_SECONDS = 4.0
DISPLAY_TRAIL_MAX_GAP_SECONDS = 0.5


def _world_axis_pixels(camera, calibration, metric):
    """Project the map's fixed origin through the CURRENT world camera pose.

    Use the selected camera's committed Atlas reference; never substitute a
    differently framed PnP pose, screen filter, or last pose. ``marker_world``
    remains readable only for historical replay packages.
    """
    if camera is None or not metric:
        return None
    world_to_camera = inverse_pose(camera)
    points = np.vstack((np.zeros(3), np.eye(3) * WORLD_AXIS_LENGTH_M))
    depths = ((world_to_camera.rotation_matrix @ points.T).T
              + world_to_camera.tvec.reshape(1, 3))[:, 2]
    if not np.all(np.isfinite(depths)) or np.any(depths <= 1e-6):
        return None
    return project_frame_axes(calibration, world_to_camera.rvec,
                              world_to_camera.tvec, WORLD_AXIS_LENGTH_M)


def _draw_world_axes(image, camera, calibration, metric):
    pixels = _world_axis_pixels(camera, calibration, metric)
    if pixels is None:
        return False
    height, width = image.shape[:2]
    origin = tuple(np.rint(pixels[0]).astype(int))
    if not (0 <= origin[0] < width and 0 <= origin[1] < height):
        return False
    draw_projected_axes(image, pixels, 4)
    cv2.circle(image, origin, 5, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(image, origin, 3, (255, 255, 255), -1, cv2.LINE_AA)
    labels = [('WORLD (50 mm)', pixels[0], (255, 255, 255)),
              ('X', pixels[1], (0, 0, 255)),
              ('Y', pixels[2], (0, 255, 0)),
              ('Z', pixels[3], (255, 0, 0))]
    for label, point, color in labels:
        if 0 <= point[0] < width and 0 <= point[1] < height:
            at = tuple(np.rint(point + [8, -8]).astype(int))
            cv2.putText(image, label, at, cv2.FONT_HERSHEY_SIMPLEX, .65,
                        (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(image, label, at, cv2.FONT_HERSHEY_SIMPLEX, .65,
                        color, 2, cv2.LINE_AA)
    return True


def _native_feature_data(snapshot):
    """Return the extracted count and compact tracked native features."""
    legacy = snapshot.get('features', [])
    total = int(snapshot.get('feature_count', len(legacy)))
    matched = snapshot.get('matched_features')
    if matched is None:
        matched = [feature for feature in legacy
                   if len(feature) >= 3 and feature[2] is not None]
    return total, matched


def _draw_orb_features(image, snapshot):
    """Draw only native tracked matches; keep all detections in the counts."""
    total, native_matches = _native_feature_data(snapshot)
    matched = ([(round(feature[0]), round(feature[1])) for feature in native_matches]
               if snapshot['state'] == 2 else [])
    # Do not retain a feature after its association disappears.
    for x, y in matched:
        cv2.rectangle(image, (x-5, y-5), (x+5, y+5), (0, 220, 0), 2, cv2.LINE_AA)
        cv2.circle(image, (x, y), 2, (0, 220, 0), -1, cv2.LINE_AA)
    return total, len(matched)


def _draw_offline_orb_features(image, features):
    """Measured image locations, cyan rings to distinguish late map matching."""
    for u, v, _, _ in features:
        at = (round(u), round(v))
        cv2.circle(image, at, 5, (255, 210, 0), 2, cv2.LINE_AA)
        cv2.circle(image, at, 1, (255, 210, 0), -1, cv2.LINE_AA)


def _world_points(points, camera: Pose):
    values = np.asarray(points, float).reshape(-1, 3)
    return (camera.rotation_matrix @ values.T).T + camera.tvec.reshape(1, 3)


def _pose_from_action(value):
    if not value:
        return None
    try:
        quaternion = value.get('quaternion_wxyz', [1.0, 0.0, 0.0, 0.0])
        pose = pose_from_native([
            *value['translation_m'],
            *quaternion[1:],
            quaternion[0],
        ])
        pose.reprojection_error_px = float(value.get('reprojection_error_px', 0.0))
        pose.marker_ids = tuple(int(mid) for mid in value.get('marker_ids', ()))
        pose.inlier_count = int(value.get('inlier_count', 0))
        pose.ambiguous = bool(value.get('ambiguous', False))
        return pose
    except (KeyError, TypeError, ValueError, IndexError):
        return None


def _replay_wrist_camera(action, hand, final_labels=False):
    """Use the offline wrist fit while retaining the replay map revision."""
    optimized_world = _pose_from_action(hand.get('wrist_world_graph'))
    final_camera = _pose_from_action(action.get('camera_world_pose_fused'))
    same_map = (
        hand.get('world_submap_id') is not None
        and hand.get('world_submap_id') == action.get('camera_submap_id')
    )
    if optimized_world is not None and final_camera is not None and same_map:
        camera_rotation = final_camera.rotation_matrix.T
        return Pose(
            cv2.Rodrigues(camera_rotation @ optimized_world.rotation_matrix)[0],
            camera_rotation @ (optimized_world.tvec - final_camera.tvec),
            optimized_world.reprojection_error_px,
            optimized_world.marker_ids,
            optimized_world.inlier_count,
            optimized_world.ambiguous,
        )
    return None if final_labels else _pose_from_action(hand.get('wrist_camera_graph'))


def final_label_camera_frame(action, revision):
    """Read frozen final labels; never mix them with process-map coordinates.

    Unlike process replay, this may show a validated final-only prefix recovery.
    No interpolation, alias renaming or second scale transform is performed.
    """
    map_id = action.get('camera_submap_id')
    mapping = next((m for m in revision.get('maps', [])
                    if f"atlas_{m['id']}" == map_id), None)
    invalid = FusedCameraFrame(None, 'invalid', 0., 0, None,
                               map_id if mapping else None,
                               (mapping or {}).get('revision', 0))
    source = action.get('camera_world_source')
    if (not mapping or not mapping.get('metric') or action.get('scale_status') != 'metric'
            or action.get('map_revision') != mapping.get('revision')
            or action.get('world_frame_id') not in (None, map_id)
            or source not in ('marker', 'marker+slam', 'head-slam')):
        return invalid
    value = action.get('camera_world_pose_fused')
    if not isinstance(value, dict):
        return invalid
    try:
        t = np.asarray(value.get('translation_m'), float)
        q = np.asarray(value.get('quaternion_wxyz'), float)
        confidence = float(action.get('camera_world_confidence', 0.))
        if (t.shape != (3,) or q.shape != (4,) or not np.all(np.isfinite(t))
                or not np.all(np.isfinite(q)) or np.linalg.norm(q) < 1e-12
                or not np.isfinite(confidence)):
            return invalid
        pose = _pose_from_action(value)
    except (TypeError, ValueError):
        return invalid
    if pose is None:
        return invalid
    return FusedCameraFrame(
        pose, source, confidence, int(action.get('slam_inliers', 0)),
        action.get('graph_reprojection_error_px'), map_id, mapping['revision'], True,
        action.get('initialization_source'), bool(mapping.get('background')),
        bool(action.get('camera_metric_recovered_later')),
        action.get('camera_anchor_consistency'), action.get('camera_localization_recovery'))


def _smooth_world_trail(values, fps):
    """Causal, speed-adaptive display filter in ONE committed world revision.

    No deadband, future samples, screen-space filter or bridging invalid poses.
    Observations/action labels are unchanged; callers reproject history into
    the current revision before filtering, including after map corrections.
    """
    output = []
    filtered = previous = None
    velocity = np.zeros(3)
    derivative_alpha = 1.0 - np.exp(-2.0 * np.pi / fps)
    for value in values:
        if value is None:
            output.append(None)
            filtered = previous = None
            velocity = np.zeros(3)
            continue
        point = np.asarray(value, dtype=float)
        if previous is None:
            filtered = point.copy()
        else:
            velocity += derivative_alpha * ((point - previous) * fps - velocity)
            cutoff_hz = 6.0 + 30.0 * float(np.linalg.norm(velocity))
            alpha = 1.0 - np.exp(-2.0 * np.pi * cutoff_hz / fps)
            filtered = filtered + alpha * (point - filtered)
        output.append(filtered.tolist())
        previous = point
    return output


def _bridge_display_trail(values, timestamps, hard_breaks,
                          max_gap_s=DISPLAY_TRAIL_MAX_GAP_SECONDS):
    """Join short, bracketed wrist-only gaps for DISPLAY, never action labels.

    Callers supply only history up to the displayed frame, already expressed
    in its committed world revision. Unknown cameras/worlds are hard breaks.
    Linear world-space interpolation cannot overshoot its measured endpoints;
    the existing display filter and rounded curves smooth the joined path.
    """
    if not (len(values) == len(timestamps) == len(hard_breaks)):
        raise ValueError('Trail values, timestamps and break mask must align')
    output = [None if p is None else list(p) for p in values]
    inserted = []
    previous = None
    for index, value in enumerate(values):
        timestamp = timestamps[index]
        invalid_clock = (not np.isfinite(timestamp) or
                         (index > 0 and (not np.isfinite(timestamps[index-1]) or
                                        timestamp <= timestamps[index-1])))
        if hard_breaks[index] or invalid_clock:
            previous = None
            output[index] = None
            continue
        if value is None:
            continue
        point = np.asarray(value, dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            previous = None
            output[index] = None
            continue
        if previous is not None and index > previous + 1:
            duration = timestamp - timestamps[previous]
            if 0 < duration <= max_gap_s + 1e-9:
                start = np.asarray(values[previous], dtype=float)
                for missing in range(previous + 1, index):
                    fraction = (timestamps[missing] - timestamps[previous]) / duration
                    output[missing] = (start + fraction * (point - start)).tolist()
                    inserted.append(missing)
        previous = index
    return output, inserted


def _display_trails(trails, timestamps, hard_breaks, fps):
    displayed, bridges = {}, {}
    for name, values in trails.items():
        filled, indices = _bridge_display_trail(values, timestamps, hard_breaks)
        displayed[name] = _smooth_world_trail(filled, fps)
        bridges[name] = indices
    return displayed, bridges


def _trail_timestamp(action, snapshot, index, fps):
    return float(action.get('timestamp_s', snapshot.get('timestamp', index / fps)))


def _temporally_excluded_marker_ids(action):
    return {int(mid) for mid, diagnostic in action.get('marker_temporal_admission', {}).items()
            if diagnostic.get('state') in {'rejected', 'pending'}}


def _rejected_wrist_label(action, marker_id):
    # Exported assist IDs are candidates, not necessarily committed factors.
    if any(int(marker_id) in hand.get('assist_only_marker_ids', [])
           for hand in action.get('hands', {}).values()):
        return 'ASSIST CANDIDATE'
    reason = action.get('marker_boundary_quality', {}).get(str(marker_id), {}).get('reason')
    return {'wrist_grid_mismatch': 'GRID REJECTED',
            'wrist_boundary': 'BORDER REJECTED'}.get(reason, 'REJECTED')


def replay_camera_frame(snapshot, revision, action, slam_inliers=None,
                        revision_references=None):
    """Use only the original marker observation, with the shared label policy."""
    raw = action.get('marker_camera_pose_observed')
    marker = None
    if raw is not None:
        try:
            q = raw['quaternion_wxyz']
            marker = pose_from_native([*raw['translation_m'], *q[1:], q[0]])
            marker.reprojection_error_px = float(raw['reprojection_error_px'])
            marker.marker_ids = tuple(int(mid) for mid in raw.get('marker_ids', []))
        except (KeyError, TypeError, ValueError, IndexError):
            marker = None
    accepted = action.get('accepted_marker_ids')
    rejected = set(action.get('rejected_marker_ids', []))
    rejected.update(_temporally_excluded_marker_ids(action))
    rejected.update(int(mid) for mid in action.get('boundary_rejected_marker_corners', {}))
    if accepted is not None:
        accepted = tuple(int(mid) for mid in accepted if int(mid) not in rejected)
    elif marker is not None and rejected:
        accepted = tuple(mid for mid in marker.marker_ids if mid not in rejected)
    quality = action.get('marker_boundary_quality')
    weights = ({int(mid): value.get('information_weight', 1.0) for mid, value in quality.items()}
               if quality is not None else None)
    if slam_inliers is None:
        slam_inliers = (len(_native_feature_data(snapshot)[1])
                        if snapshot['state'] == 2 else 0)
    return select_camera_frame(snapshot, revision, marker, action.get('marker_camera_confidence', 0.0),
                               slam_inliers, accepted, weights, revision_references)


def _marker_world_view(selected, layout: BandLayout | None):
    """Independent fixed reference, not a new Atlas or transformed ORB cloud."""
    if selected.map_id != 'marker_world' or selected.pose is None:
        return None
    markers = ({str(mid): layout.markers[mid].reshape(-1).tolist()
                for mid in selected.pose.marker_ids if mid in layout.markers}
               if layout is not None else {})
    return {'id': 'marker_world', 'metric': True, 'revision': 0,
            'seed': True, 'background': False, 'point_count': 0, 'keyframes': [],
            'markers': markers, 'geometry_source': 'fixed_layout' if layout is not None else 'unavailable'}


def trails_at_revision(frame_index, snapshots, actions, revision, fps, seconds=4):
    """Reproject past measurements in one committed world, not camera trails."""
    selected = replay_camera_frame(snapshots[frame_index], revision, actions[frame_index])
    camera, map_id = selected.pose, selected.map_id
    if camera is None or not selected.metric:
        # The 3-D viewer may still inspect a native arbitrary-unit map/camera;
        # these are never used for metric wrist trails or the fixed-world axes.
        native_camera, native_map = camera_at_revision(snapshots[frame_index], revision)
        return {}, native_camera, native_map
    first = max(0, frame_index - round(seconds * fps))
    names = {name for action in actions[first:frame_index+1] for name in action['hands']}
    trails = {name: [] for name in sorted(names)}
    timestamps, hard_breaks = [], []
    for index in range(first, frame_index + 1):
        previous = replay_camera_frame(snapshots[index], revision, actions[index])
        hard_break = previous.pose is None or not previous.metric or previous.map_id != map_id
        hard_breaks.append(hard_break)
        timestamps.append(_trail_timestamp(actions[index], snapshots[index], index, fps))
        for name, trail in trails.items():
            hand = actions[index]['hands'].get(name, {})
            wrist = _replay_wrist_camera(actions[index], hand)
            if hard_break or wrist is None:
                trail.append(None)
            else:
                trail.append(_world_points([wrist.tvec.reshape(3)], previous.pose)[0].tolist())
    displayed, _ = _display_trails(trails, timestamps, hard_breaks, fps)
    return displayed, camera, map_id


def _freeze_revision_value(value):
    if isinstance(value, dict):
        return tuple((key, _freeze_revision_value(item)) for key, item in sorted(value.items()))
    if isinstance(value, list):
        return tuple(_freeze_revision_value(item) for item in value)
    return value


class _TrailReplayCache:
    """Cache camera reprojections while one committed graph transform is unchanged."""
    def __init__(self, snapshots, actions, fps, seconds=4, final_labels=False):
        self.snapshots = snapshots
        self.actions = actions
        self.fps = fps
        self.seconds = seconds
        self.final_labels = final_labels
        self.inliers = [len(_native_feature_data(frame)[1]) if frame.get('state') in (2, 7) else 0
                        for frame in snapshots]
        self.signature = None
        self.references = {}
        self.selected = {}
        self.last_request = None
        self.last_result = None
        self.display_bridges = {}
        self.trail_timestamps = []

    @staticmethod
    def _context(revision):
        references = {value[0]: value for value in revision.get('references', [])}
        maps = tuple((mapping.get('id'), bool(mapping.get('metric')),
                      bool(mapping.get('seed')), bool(mapping.get('background')))
                     for mapping in revision.get('maps', []))
        signature = (_freeze_revision_value(revision.get('references', [])), maps,
                     revision.get('active_map'),
                     _freeze_revision_value(revision.get('marker_map_aliases', {})),
                     'reference_marker_gauge' in revision,
                     _freeze_revision_value(revision.get('reference_marker_gauge')))
        return signature, references

    def _historical(self, index, revision):
        frame = self.snapshots[index]
        same = (frame is revision or
                (frame.get('timestamp') == revision.get('timestamp') and frame == revision))
        key = index, same
        if key not in self.selected:
            self.selected[key] = (final_label_camera_frame(self.actions[index], revision)
                                  if self.final_labels else replay_camera_frame(
                                      frame, revision, self.actions[index], self.inliers[index], self.references))
        return self.selected[key]

    def resolve(self, frame_index, revision):
        signature, references = self._context(revision)
        if signature != self.signature:
            self.signature, self.references = signature, references
            self.selected.clear()
            self.last_request = self.last_result = None
        map_revisions = tuple((mapping.get('id'), mapping.get('revision'))
                              for mapping in revision.get('maps', []))
        request = (frame_index, signature, revision.get('timestamp'),
                   bool(revision.get('final')), map_revisions)
        if request == self.last_request:
            return self.last_result

        self.display_bridges = {}
        self.trail_timestamps = []
        selected = self._historical(frame_index, revision)
        camera, map_id = selected.pose, selected.map_id
        if camera is None or not selected.metric:
            if not self.final_labels:
                camera, map_id = camera_at_revision(
                    self.snapshots[frame_index], revision, self.references)
            result = selected, {}, camera, map_id
        else:
            first = max(0, frame_index - round(self.seconds * self.fps))
            names = {name for action in self.actions[first:frame_index+1] for name in action['hands']}
            trails = {name: [] for name in sorted(names)}
            hard_breaks = []
            for index in range(first, frame_index + 1):
                previous = selected if index == frame_index else self._historical(index, revision)
                hard_break = previous.pose is None or not previous.metric or previous.map_id != map_id
                hard_breaks.append(hard_break)
                self.trail_timestamps.append(_trail_timestamp(
                    self.actions[index], self.snapshots[index], index, self.fps))
                for name, trail in trails.items():
                    hand = self.actions[index]['hands'].get(name, {})
                    wrist = _replay_wrist_camera(self.actions[index], hand, self.final_labels)
                    if hard_break or wrist is None:
                        trail.append(None)
                    else:
                        trail.append(_world_points(
                            [wrist.tvec.reshape(3)], previous.pose)[0].tolist())
            displayed, self.display_bridges = _display_trails(
                trails, self.trail_timestamps, hard_breaks, self.fps)
            result = selected, displayed, camera, map_id
        self.last_request, self.last_result = request, result
        return result


def _draw_trails(image, trails, camera, calibration):
    if camera is None:
        return
    inverse = inverse_pose(camera)
    height, width = image.shape[:2]
    for hand_index, values in enumerate(trails.values()):
        color = np.array((70, 220, 70) if hand_index == 0 else (60, 60, 240))
        previous = None
        for index, point in enumerate(values):
            if point is None:
                previous = None
                continue
            point = np.asarray(point, float)
            if (inverse.rotation_matrix @ point + inverse.tvec.reshape(3))[2] <= 0:
                previous = None
                continue
            pixel = cv2.projectPoints(point.reshape(1, 3), inverse.rvec, inverse.tvec,
                                      calibration.camera_matrix, calibration.dist_coeffs)[0].reshape(2)
            if not np.all(np.isfinite(pixel)) or np.max(np.abs(pixel)) > 1e6:
                previous = None
                continue
            current = tuple(np.rint(pixel).astype(int))
            if previous is not None:
                visible, start, stop = cv2.clipLine((0, 0, width, height), previous, current)
                if visible:
                    fade = .15 + .85 * (index + 1) / len(values)
                    cv2.line(image, start, stop, tuple((color * fade).astype(int).tolist()), 3, cv2.LINE_AA)
            previous = current


def _events(previous, snapshot):
    result = []
    marker_event = snapshot.get('marker_keyframe_event')
    if marker_event and not snapshot.get('final'):
        labels = (marker_event.replace('first_seen:', '首次可靠观测 marker ')
                  .replace('anchor_reobserved:', '固定 marker 重新可见：保留角点关键帧 ')
                  .replace('relocalized:', '持续失跟后由 marker 恢复定位 ')
                  .replace('reobserved:', '遮挡后重现 marker '))  # historical recordings
        result.append(f"{labels} → 关键帧 {snapshot['marker_event_keyframe_id']}（固定码信息触发）")
    tracked = snapshot.get('marker_tracking', {})
    old_tracked = (previous or {}).get('marker_tracking', {})
    if tracked.get('partial') and tracked.get('accepted') and not old_tracked.get('partial'):
        result.append(f"部分 marker 角点辅助定位：{tracked['corners']} 点，弱权重（非完整码观测）")
    if (tracked.get('pose_constraint_reason') == 'deferred_to_marker_graph' and
            old_tracked.get('pose_constraint_reason') != 'deferred_to_marker_graph'):
        position_mm = 1000.0 * max(0.0, float(tracked.get('pose_position_residual_m', 0.0)))
        rotation_deg = max(0.0, float(tracked.get('pose_rotation_residual_deg', 0.0)))
        result.append(f"Marker/ORB 单帧差异 {position_mm:.1f} mm / {rotation_deg:.1f}°："
                      "不跳变当前位姿，保留观测供多帧图优化")
    reason = tracked.get('pose_constraint_reason')
    old_reason = old_tracked.get('pose_constraint_reason')
    if reason == 'deferred_to_marker_graph_background_conflict' and old_reason != reason:
        result.append("单帧 marker 融合与已跟踪背景点冲突：撤销本次位姿更新，保留角点供多帧图优化")
    elif reason == 'single_marker_graph_only' and old_reason != reason:
        result.append("单个完整固定码：角点进入关键帧/图优化，不单帧拉动 Atlas 位姿")
    elif reason == 'partial_marker_graph_only' and old_reason != reason:
        result.append("不完整固定码：弱权重角点进入图优化，不直接修改 Atlas 位姿")
    elif reason == 'partial_marker_observation_only' and old_reason != reason:
        result.append("不完整固定码：仅作短时 marker 位姿连续观测，不写入静态 marker 图因子")
    elif reason == 'marker_set_unconfirmed' and old_reason != reason:
        result.append("固定码组合发生变化：等待连续 3 帧位姿一致后再融合")
    elif reason == 'fused_low_visual_support' and old_reason != reason:
        result.append("ORB 有效匹配偏少：使用至少 2 个完整固定码角点稳定当前位姿")
    elif reason == 'fused_marker_recovery' and old_reason != reason:
        result.append("Marker → ORB 背景恢复：保留完整固定码角点约束，避免两个世界系产生接缝")
    elif reason == 'fused_three_marker_support' and old_reason != reason:
        result.append("至少 3 个完整固定码：12+ 强角点直接进入当前联合位姿优化")
    bootstrap = snapshot.get('marker_bootstrap', {})
    if bootstrap.get('reference_changed'):
        reason = {'stronger_reference': '原参考图像特征不足',
                  'reference_view_changed': '观察方向变化',
                  'insufficient_matches': '旧参考图像匹配不足',
                  'insufficient_geometry': '旧参考图像三角化不足',
                  'waiting_baseline': '建立初始参考'}.get(bootstrap['reason'], bootstrap['reason'])
        result.append(f"背景初始化参考 {bootstrap['reference_frame']} → "
                      f"{bootstrap.get('next_reference_frame', bootstrap['reference_frame'])}：{reason}；"
                      f"匹配 {bootstrap['matches']} / 有效三角化 {bootstrap['triangulated']}；世界原点不变")
    old_maps = {m['id']: m for m in previous.get('maps', [])} if previous else {}
    if previous and previous['state'] != snapshot['state']:
        states = {0: '等待图像', 1: '等待单目视差初始化', 2: 'ORB 跟踪有效',
                  3: '失跟，尝试重定位', 4: '定位无效',
                  6: 'Marker 定位有效，背景地图等待建立/恢复'}
        result.append(states.get(snapshot['state'], '跟踪状态改变'))
    for mapping in snapshot['maps']:
        old = old_maps.get(mapping['id'])
        if old is None:
            result.append(f"新地图 {mapping['id']}：" + ('米制 marker 种子' if mapping['seed'] else '单目任意尺度'))
        elif mapping['metric'] and not old['metric']:
            result.append(f"地图 {mapping['id']} " + (
                "marker 直接米制初始化（不需要运动）" if mapping['seed'] else
                f"多视角首次尺度锚定：{mapping['scale']:.5g} m/unit"))
        if mapping['background'] and (old is None or not old['background']):
            result.append(f"地图 {mapping['id']} 背景三角化成功")
        for field, title in [('loops', '原生视觉回环'), ('merges', '原生地图合并')]:
            before = set(tuple(v) for v in (old or {}).get(field, []))
            for pair in mapping.get(field, []):
                if tuple(pair) not in before:
                    result.append(f'{title}：关键帧 {pair[0]} ↔ {pair[1]}')
    seen_rejections = {event['sequence'] for event in (previous or {}).get('correction_rejections', [])}
    for event in snapshot.get('correction_rejections', []):
        if event['sequence'] in seen_rejections:
            continue
        seen_rejections.add(event['sequence'])
        kind = event['kind']
        title = {'loop': '视觉回环', 'merge': '视觉合并', 'global_ba': '全局BA'}.get(kind, kind)
        reason = event['reason']
        if reason == 'fixed_marker_geometry_requires_joint_optimization':
            reason = ('优化期间 marker 坐标/尺度已变更' if kind == 'global_ba'
                      else '当前固定 marker 几何尚不支持共同优化')
        elif reason == 'map_unavailable':
            reason = '目标地图不可用'
        elif reason == 'current_marker_factor_requires_marker_graph':
            reason = '当前帧已有 marker 角点因子，应由 marker 图优化而非纯视觉回环处理'
        # Emit at first publication, not retroactively at the candidate KF's
        # timestamp. Repeated cumulative native snapshots must not repeat it.
        result.append(f"{title}未提交：{reason}；图 {event['map_id']} ↔ {event['other_map_id']}，"
                      f"候选关键帧 {event['keyframe_id']} @ {event['timestamp']:.3f}s")
    seen_graph_events = {event['sequence'] for event in (previous or {}).get('marker_graph_events', [])}
    for event in snapshot.get('marker_graph_events', []):
        if event['sequence'] in seen_graph_events:
            continue
        seen_graph_events.add(event['sequence'])
        title = {'scale_reanchor': '尺度再锚定', 'marker_map_merge': '标记辅助合图',
                 'marker_global_ba': 'Marker–ORB 全局 BA'}.get(
            event['type'], event['type'])
        maps = (f"图 {event['source_map_id']} → {event['target_map_id']}"
                if event['type'] == 'marker_map_merge' else f"图 {event['map_id']}")
        if event['status'] != 'accepted':
            result.append(f"{title}未提交：{event['reason']}；{maps}")
            continue
        keyframes = event['affected_keyframes']
        count = len(keyframes) if isinstance(keyframes, list) else keyframes
        markers = ', '.join(str(mid) for mid in event['marker_ids'])
        # A separate cumulative journal: a rejected visual candidate and a
        # committed marker correction may legitimately share sequence numbers.
        # pack_history stamps the FIRST publication, never an earlier anchor.
        result.append(f"{title}已提交：{maps}；marker {markers}；"
                      f"尺度 ×{event['scale']:.5g}；{count} 个关键帧")
    if snapshot.get('final'):
        result.append('视频结束：显示线程收尾后提交的地图版本（此停帧不生成训练标签）')
    return result


def _accepted_loop_events(history, fps):
    """Return compact, playback-clocked records for committed loop edges.

    Native loop commits can be published at shutdown, while the edge belongs
    to an earlier keyframe.  The replay must therefore keep both clocks: the
    candidate keyframe time (when the correction is shown) and the publication
    time (when the backend actually committed it).  This is metadata only; it
    never invents a causal pose or changes the native history.
    """
    if not history or not np.isfinite(float(fps)) or float(fps) <= 0:
        return []
    final = history[-1]
    graph_events = [event for event in final.get('marker_graph_events', [])
                    if event.get('status') == 'accepted']

    def pair_key(pair):
        try:
            values = tuple(sorted(int(value) for value in pair[:2]))
        except (TypeError, ValueError, IndexError):
            return None
        return values if len(values) == 2 else None

    def keyframe_times(mapping):
        result = {}
        for keyframe in mapping.get('keyframes', []):
            try:
                result[int(keyframe[0])] = float(keyframe[1])
            except (TypeError, ValueError, IndexError):
                continue
        return result

    def first_commit(map_id, pair):
        for sequence, snapshot in enumerate(history):
            for mapping in snapshot.get('maps', []):
                try:
                    same_map = int(mapping.get('id')) == map_id
                except (TypeError, ValueError):
                    same_map = False
                if not same_map:
                    continue
                if any(pair_key(candidate) == pair
                       for candidate in mapping.get('loops', [])):
                    return sequence, float(snapshot.get('timestamp', 0.0)), bool(snapshot.get('final'))
        last = len(history) - 1
        return last, float(history[last].get('timestamp', 0.0)), bool(history[last].get('final'))

    records, seen = [], set()
    for mapping in final.get('maps', []):
        try:
            map_id = int(mapping.get('id'))
        except (TypeError, ValueError):
            continue
        times = keyframe_times(mapping)
        for raw_pair in mapping.get('loops', []):
            pair = pair_key(raw_pair)
            seen_key = (map_id, pair) if pair is not None else None
            if pair is None or seen_key in seen or pair[0] not in times or pair[1] not in times:
                continue
            seen.add(seen_key)
            candidate_timestamp = max(times[pair[0]], times[pair[1]])
            candidate_frame = int(round(candidate_timestamp * float(fps)))

            # Prefer the explicit marker/loop graph event when it carries the
            # same candidate KF/time.  Generic ORB loops still get a record.
            matched = None
            best_score = float('inf')
            for event in graph_events:
                try:
                    if int(event.get('map_id')) != map_id:
                        continue
                except (TypeError, ValueError):
                    continue
                event_frame = event.get('candidate_frame')
                event_timestamp = event.get('candidate_timestamp', event.get('timestamp'))
                try:
                    frame_delta = (abs(int(event_frame) - candidate_frame)
                                   if event_frame is not None else float('inf'))
                    time_delta = (abs(float(event_timestamp) - candidate_timestamp)
                                  if event_timestamp is not None else float('inf'))
                except (TypeError, ValueError):
                    continue
                # A candidate KF match is stronger than a loose timestamp
                # match; reject unrelated marker BA events.
                score = min(frame_delta, time_delta * float(fps))
                if frame_delta <= 2 or time_delta <= 2.0 / float(fps):
                    if score < best_score:
                        matched, best_score = event, score

            commit_sequence, commit_timestamp, commit_final = first_commit(map_id, pair)
            event_timestamp = (float(matched.get('candidate_timestamp', candidate_timestamp))
                               if matched and matched.get('candidate_timestamp') is not None
                               else candidate_timestamp)
            event_frame = (int(matched.get('candidate_frame'))
                           if matched and matched.get('candidate_frame') is not None
                           else int(round(event_timestamp * float(fps))))
            affected = matched.get('affected_keyframes', []) if matched else []
            records.append({
                'id': f'loop-{map_id}-{pair[0]}-{pair[1]}',
                'map_id': map_id,
                'keyframe_a': pair[0],
                'keyframe_b': pair[1],
                'source_frame': event_frame,
                'timestamp_s': event_timestamp,
                'candidate_frame': event_frame,
                'candidate_timestamp_s': event_timestamp,
                'commit_sequence': commit_sequence,
                'commit_timestamp_s': commit_timestamp,
                'commit_source': 'offline-finalize' if commit_final else 'online',
                'event_sequence': matched.get('sequence') if matched else None,
                'affected_keyframes': (len(affected) if isinstance(affected, list)
                                       else int(affected or 0)),
                'correction_kind': matched.get('type', 'visual_loop') if matched else 'visual_loop',
            })
    return sorted(records, key=lambda event: (
        event['timestamp_s'], event['commit_sequence'], event['id']))


def _loop_event_banner(image, event):
    """Annotate the encoded video at the candidate frame, not at shutdown."""
    height, width = image.shape[:2]
    y = 126
    box_width = min(width - 20, 760)
    cv2.rectangle(image, (10, y), (box_width, y + 48), (20, 20, 20), -1)
    title = (f"LOOP CORRECTION | BEFORE -> AFTER | KF {event['keyframe_a']} "
             f"<-> {event['keyframe_b']}")
    detail = (f"candidate {event['timestamp_s']:.2f}s | "
              f"commit {event['commit_source']} at {event['commit_timestamp_s']:.2f}s | "
              f"affected KFs {event['affected_keyframes']}")
    cv2.putText(image, title, (20, y + 20), cv2.FONT_HERSHEY_SIMPLEX,
                .58, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, detail, (20, y + 40), cv2.FONT_HERSHEY_SIMPLEX,
                .50, (170, 220, 255), 1, cv2.LINE_AA)


def pack_history(history, directory: Path, fps: float):
    """Pack legacy full maps or native deltas into one viewer point journal."""
    rows, point_state, known_maps, previous = [], {}, set(), None
    offset = 0
    with gzip.open(directory / 'points.bin.gz', 'wb', compresslevel=6) as binary:
        with (directory / 'events.jsonl').open('w') as events:
            last_checkpoint_time = -float('inf')
            for sequence, snapshot in enumerate(history):
                before = point_state.copy()
                current_map_ids = {mapping['id'] for mapping in snapshot['maps']}
                for key in [key for key in point_state if key[0] not in current_map_ids]:
                    del point_state[key]
                known_maps.intersection_update(current_map_ids)

                compact = any('points_mode' in mapping for mapping in snapshot['maps'])
                if compact:
                    for mapping in snapshot['maps']:
                        map_id = mapping['id']
                        mode = mapping.get('points_mode')
                        if mode not in ('full', 'delta'):
                            raise ValueError(f'map {map_id} has invalid compact points mode')
                        if mode == 'delta' and map_id not in known_maps:
                            raise ValueError(f'map {map_id} delta precedes a full publication')
                        if mode == 'full':
                            for key in [key for key in point_state if key[0] == map_id]:
                                del point_state[key]
                            known_maps.add(map_id)
                        for point in mapping.get('points', []):
                            point_state[(map_id, int(point[0]))] = tuple(point[1:])
                        for point_id in mapping.get('deleted_points', []):
                            point_state.pop((map_id, int(point_id)), None)
                        count = sum(key[0] == map_id for key in point_state)
                        if count != mapping.get('point_count'):
                            raise ValueError(f'map {map_id} compact point count mismatch')
                else:
                    point_state = {(mapping['id'], int(point[0])): tuple(point[1:])
                                   for mapping in snapshot['maps'] for point in mapping['points']}
                    known_maps = current_map_ids

                checkpoint = (snapshot.get('final') or sequence == 0 or
                              snapshot['timestamp'] - last_checkpoint_time >= 5.0)
                comparison = {} if checkpoint else before
                if checkpoint:
                    last_checkpoint_time = snapshot['timestamp']
                changed = [(key, point) for key, point in point_state.items()
                           if comparison.get(key) != point]
                removed = [list(key) for key in comparison if key not in point_state]
                for (map_id, point_id), point in changed:
                    binary.write(struct.pack('<QQfff', map_id, point_id, *point))
                row = {k: v for k, v in snapshot.items()
                       if k not in ('maps', 'features', 'matched_features', 'references')}
                row.update(sequence=sequence, checkpoint=checkpoint, offset=offset,
                           count=len(changed), deleted=removed,
                           maps=[{k: v for k, v in mapping.items()
                                  if k not in ('points', 'deleted_points', 'points_mode')} |
                                 {'point_count': mapping.get(
                                     'point_count', len(mapping.get('points', [])))}
                                 for mapping in snapshot['maps']],
                           events=_events(previous, snapshot))
                for description in row['events']:
                    events.write(json.dumps({'sequence': sequence, 'source_frame': round(snapshot['timestamp'] * fps),
                                             'timestamp_s': snapshot['timestamp'], 'description': description},
                                            ensure_ascii=False) + '\n')
                rows.append(row)
                offset += len(changed) * 28
                previous = snapshot
    with gzip.open(directory / 'timeline.json.gz', 'wt', encoding='utf-8') as stream:
        json.dump(rows, stream, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
    return rows


def _marker_graph_capabilities(history):
    """Only native's explicit wired-feature declaration enables these flags."""
    return {name: bool(history) and all(
        snapshot.get('marker_graph_capabilities', {}).get(name) is True for snapshot in history)
        for name in ('interval_scale_reanchor', 'marker_map_merge',
                     'rigid_marker_pose_optimization', 'metric_tag_global_ba')}


def write_slam_replay(video_path: Path, actions_path: Path, directory: Path,
                      history: list[dict], calibration: Calibration,
                      detections: list[dict], accepted_ids: list[tuple], fps: float,
                      output_path: Path | None = None,
                      marker_layout: BandLayout | None = None,
                      actions: list[dict] | None = None,
                      final_map: bool = False,
                      offline_features: dict | None = None,
                      hybrid: bool = False) -> tuple[Path, Path]:
    started = time.perf_counter()
    if final_map and hybrid:
        raise ValueError('final-map and hybrid replay modes are mutually exclusive')
    offline = final_map or hybrid
    if offline and (not history or not history[-1].get('final')):
        raise ValueError('offline replay requires a completed final native publication')
    directory.mkdir(parents=True, exist_ok=True)
    rows = pack_history(history, directory, fps)
    # Keep the canonical journal for existing validators; the browser loads
    # exact shared map/event versions instead of a repeated >512 MiB string.
    write_browser_timeline(rows, directory)
    if actions is None:
        with actions_path.open() as stream:
            actions = [json.loads(line) for line in stream if line.strip()]
    loop_events = _accepted_loop_events(history, fps)
    by_index = {round(h['timestamp'] * fps): h for h in history if not h.get('final')}
    snapshots = [by_index.get(i, {'state': 4, 'pose': None, 'active_map': -1, 'maps': []})
                 for i in range(len(actions))]
    history_indices = {round(h['timestamp'] * fps): i for i, h in enumerate(history) if not h.get('final')}
    output_fps = 30.0
    count = int(np.ceil(len(actions) * output_fps / fps))
    sampling = [(min(len(actions) - 1, int(i * fps / output_fps)), False) for i in range(count)]
    if not final_map:
        sampling += [(len(actions) - 1, True)] * round(
            output_fps * FINAL_REPLAY_HOLD_SECONDS)
    process_path = directory / 'process.mp4'
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise RuntimeError('ffmpeg is required for browser-compatible H.264 replay')
    command = [ffmpeg, '-v', 'error', '-y', '-f', 'rawvideo', '-pixel_format', 'bgr24',
               '-video_size', '1280x720', '-framerate', str(output_fps), '-i', '-',
               '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '22',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(process_path)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        encoder.stdin.close()
        encoder.wait()
        raise RuntimeError(f'cannot read {video_path}')
    current_index, raw = -1, None
    video_frames = []
    trail_cache = _TrailReplayCache(snapshots, actions, fps, final_labels=offline)
    from .wrist_gap_export import complete_wrists, CompletedReplayTrails
    completed_trails = CompletedReplayTrails(complete_wrists(actions)) if offline else None
    try:
        for out_index, (source_index, final) in enumerate(sampling):
            while current_index < source_index:
                valid, raw = capture.read()
                if not valid:
                    raise RuntimeError('source video ended before cached action observations')
                current_index += 1
            image = raw.copy()
            snapshot = snapshots[source_index]
            revision = history[-1] if final or offline else snapshot
            selected, trails, camera, map_id = trail_cache.resolve(source_index, revision)
            completion_frame = None
            if completed_trails is not None:
                completion_frame = completed_trails.apply(dict(source_frame=source_index,
                    trail_timestamps_s=trail_cache.trail_timestamps, camera=camera,
                    metric=selected.metric, map_id=map_id, map_revision=selected.revision))
                trails = completion_frame['trails']
            marker_view = _marker_world_view(selected, marker_layout)
            _draw_trails(image, trails, camera, calibration)
            if completion_frame is not None:
                count = sum(len(v) for v in completion_frame['trail_display_interpolated'].values())
                cv2.putText(image, f'EXPORTED WRIST TRAIL | {count} interpolated samples (estimates)',
                            (12, image.shape[0]-18), cv2.FONT_HERSHEY_SIMPLEX, .6,
                            (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(image, f'EXPORTED WRIST TRAIL | {count} interpolated samples (estimates)',
                            (12, image.shape[0]-18), cv2.FONT_HERSHEY_SIMPLEX, .6,
                            (255, 255, 255), 1, cv2.LINE_AA)
            width, height = calibration.image_size
            action = actions[source_index]
            landmarks = {}
            for hand_index, (name, hand) in enumerate(action['hands'].items()):
                color = (70, 220, 70) if hand_index == 0 else (60, 60, 240)
                joints = hand['joints']
                pixels = joints.get('image_landmarks_normalized')
                if joints.get('valid') and pixels is not None:
                    pixels = np.rint(np.asarray(pixels)[:, :2] * [width, height]).astype(int)
                    for first, second in HAND_CONNECTIONS:
                        cv2.line(image, tuple(pixels[first]), tuple(pixels[second]), color, 2, cv2.LINE_AA)
                if offline:
                    world = joints.get('world_landmarks_graph_m')
                    if (selected.metric and camera is not None and joints.get('valid')
                            and hand.get('world_submap_id') == map_id and world is not None):
                        values = np.asarray(world, float)
                        if values.shape == (21, 3) and np.all(np.isfinite(values)):
                            landmarks[name] = values.tolist()
                elif camera is not None and trails and joints.get('camera_landmarks_m') is not None:
                    landmarks[name] = _world_points(joints['camera_landmarks_m'], camera).tolist()
                wrist = _replay_wrist_camera(action, hand, offline)
                if trails and camera is not None and wrist:
                    cv2.drawFrameAxes(image, calibration.camera_matrix, calibration.dist_coeffs,
                                      wrist.rvec, wrist.tvec, .025, 2)
            extracted, matched = _draw_orb_features(image, snapshot)
            _, native_matches = _native_feature_data(snapshot)
            local_map_point_ids = (
                sorted({int(feature[2]) for feature in native_matches
                        if len(feature) >= 3 and feature[2] is not None})
                if snapshot.get('state') == 2 else []
            )
            process_map_point_ids = local_map_point_ids
            if offline and map_id != f"atlas_{snapshot.get('active_map')}":
                # Old-map point IDs are not a cross-map correspondence.
                local_map_point_ids = []
            recovered_features = (offline_features or {}).get(source_index, {}) if offline else {}
            final_mapping = next((m for m in revision['maps'] if f"atlas_{m['id']}" == map_id), {})
            display_mapping = next((m for m in revision['maps']
                if m['id'] == recovered_features.get('map_id')), {})
            display_only = (offline and recovered_features.get('display_only') is True
                and recovered_features.get('source') == 'offline-prefix-display-correspondence'
                and recovered_features.get('accepted') is True
                and recovered_features.get('frame') == source_index
                and display_mapping.get('metric') is True
                and recovered_features.get('map_revision') == display_mapping.get('revision'))
            if not display_only and (camera is None or not selected.metric or recovered_features.get('accepted') is not True
                    or recovered_features.get('source') != 'offline-final-map-correspondence'
                    or recovered_features.get('frame') != source_index
                    or recovered_features.get('map_id') != final_mapping.get('id')
                    or recovered_features.get('map_revision') != selected.revision):
                recovered_features = {}
            offline_matches = recovered_features.get('matched_features', [])
            if offline_matches:
                _draw_offline_orb_features(image, offline_matches)
                if display_only:
                    label = f"LATE DISPLAY MATCHES: map {recovered_features['map_id']} | original pose labels unchanged"
                    cv2.putText(image, label, (12, 128), cv2.FONT_HERSHEY_SIMPLEX, .65,
                                (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, label, (12, 128), cv2.FONT_HERSHEY_SIMPLEX, .65,
                                (255, 210, 0), 2, cv2.LINE_AA)
                if not display_only or recovered_features.get('map_id') == final_mapping.get('id'):
                    local_map_point_ids = sorted(set(local_map_point_ids) | {int(f[2]) for f in offline_matches})
            tracked = snapshot.get('marker_tracking', {})
            for u, v, _ in tracked.get('points_undistorted', []):
                ray = np.linalg.solve(calibration.camera_matrix, [u, v, 1.])
                pixel = cv2.projectPoints(np.asarray([ray]), np.zeros(3), np.zeros(3),
                    calibration.camera_matrix, calibration.dist_coeffs)[0].reshape(2)
                cv2.circle(image, tuple(np.rint(pixel).astype(int)), 5, (220, 80, 220), 2, cv2.LINE_AA)
            wrist_ids = {mid for hand in action["hands"].values()
                         for mid in hand.get("accepted_marker_ids", [])}
            temporal_excluded = _temporally_excluded_marker_ids(action)
            admitted_detections = {mid: corners for mid, corners in detections[source_index].items()
                                   if mid not in temporal_excluded}
            for marker_id, corners in admitted_detections.items():
                quality = action.get("marker_boundary_quality", {}).get(str(marker_id), {})
                soft = quality.get("reason") == "soft_grid"
                color = (255, 210, 0) if soft else (
                    (0, 220, 0) if marker_id in accepted_ids[source_index] or marker_id in wrist_ids
                    else (0, 160, 255))
                polygon = np.rint(corners).astype(np.int32)
                cv2.polylines(image, [polygon], True, color, 2, cv2.LINE_AA)
                cv2.putText(image, str(marker_id) + (' SOFT 25%' if soft else ''), tuple(polygon[0]), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
            for marker_id in sorted(temporal_excluded):
                corners = action.get('detected_marker_corners', {}).get(str(marker_id))
                if corners is None:
                    continue
                polygon = np.rint(corners).astype(np.int32)
                state = action['marker_temporal_admission'][str(marker_id)]['state']
                label = 'PENDING GEOMETRY' if state == 'pending' else 'TEMPORAL REJECTED'
                cv2.polylines(image, [polygon], True, (0, 160, 255), 2, cv2.LINE_AA)
                cv2.putText(image, f'{marker_id} {label}', tuple(polygon[0]),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 160, 255), 2)
            for marker_id, corners in action.get('boundary_rejected_marker_corners', {}).items():
                quality = action.get('marker_boundary_quality', {}).get(str(marker_id), {})
                if not quality.get('reason', '').startswith('wrist_'):
                    continue
                polygon = np.rint(corners).astype(np.int32)
                reason_label = _rejected_wrist_label(action, marker_id)
                color = (255, 210, 0) if reason_label == 'ASSIST CANDIDATE' else (0, 165, 255)
                cv2.polylines(image, [polygon], True, color, 2, cv2.LINE_AA)
                cv2.putText(image, f'{marker_id} {reason_label}', tuple(polygon[0]),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, color, 2)
            mapping = marker_view or next((m for m in revision['maps'] if f"atlas_{m['id']}" == map_id), {})
            status_mapping = mapping or next((m for m in revision['maps']
                                             if m['id'] == revision['active_map']), {})
            selected = annotate_anchor_consistency(
                selected, mapping, calibration, admitted_detections,
                tuple(accepted_ids[source_index]),
                {int(mid): q.get('information_weight', 1.) for mid, q in
                 action.get('marker_boundary_quality', {}).items()},
            )
            world_axes_visible = _draw_world_axes(image, camera, calibration,
                                                  mapping.get('metric', False))
            source = selected.source
            map_label = map_id or f"atlas_{revision['active_map']} LOST"
            label = (f"{'OFFLINE FINAL MAP' if offline else 'NATIVE ORB'} | frame {source_index} | {source} | {map_label} | "
                     f"{'METRIC / ORB UNALIGNED' if marker_view else 'METRIC' if status_mapping.get('metric') else 'ARBITRARY UNITS'} | "
                     f"MP {status_mapping.get('point_count', len(status_mapping.get('points', [])))} "
                     f"KF {len(status_mapping.get('keyframes', []))}")
            if selected.metric_recovered_later:
                label += " | METRIC RECOVERED LATER"
            if offline and selected.localization_recovery:
                method = selected.localization_recovery.get('method')
                label += (" | POSE REFINED OFFLINE" if method == 'final-map-rematched-pose-refinement' else
                          " | SHORT GAP RECOVERED OFFLINE" if method == 'native-orb-final-atlas-short-gap-pnp' else
                          " | PREFIX RECOVERED OFFLINE")
            elif offline and camera is None:
                label += " | NO VALID OFFLINE POSE"
            if action.get('marker_pose_offline_evidence'):
                label += " | OFFLINE MULTIVIEW DISAMBIGUATION"
            if selected.anchor_consistency['status'] == 'conflict':
                label += " | ANCHOR CONFLICT"
            axis_status = ('VISIBLE' if world_axes_visible else 'OUT OF VIEW'
                           if camera is not None and mapping.get('metric') else 'UNAVAILABLE')
            bootstrap = None if offline else snapshot.get('marker_bootstrap')
            cv2.rectangle(image, (0, 0), (width, 116 if bootstrap else 82), (0, 0, 0), -1)
            cv2.putText(image, label, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)
            cv2.putText(image, f'ORB detected {extracted} / tracked {matched} | '
                        + (f'OFFLINE matched {len(offline_matches)} (cyan) | ' if offline_matches else '') +
                        f'WORLD axes: {axis_status} | tracked tag corners {tracked.get("corners", 0)} | 50 mm',
                        (12, 65), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)
            footer = ('OFFLINE VIDEO / GLOBAL / HANDS | LOCAL VIEW: recorded mapping process'
                      if hybrid else 'OFFLINE RECONSTRUCTION: final map shown from start; not online mapping'
                      if final_map else 'DISPLAY TRAIL: wrist gaps <= 0.5 s joined; labels unchanged')
            cv2.putText(image, footer,
                        (12, height-18), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(image, footer,
                        (12, height-18), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2, cv2.LINE_AA)
            if bootstrap:
                cv2.putText(image, f"BACKGROUND INIT: {bootstrap['reason']} | "
                            f"matches {bootstrap['matches']} / 3D {bootstrap['triangulated']} | "
                            f"baseline {bootstrap['baseline_m']*1000:.1f} mm",
                            (12, 100), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 220, 255), 2)
            if final:
                cv2.putText(image, 'END: committed map state; no new action label',
                            (12, 146 if bootstrap else 112), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 220, 255), 2)
            for event in loop_events:
                if event['source_frame'] == source_index:
                    _loop_event_banner(image, event)
            encoder.stdin.write(cv2.resize(image, (1280, 720), interpolation=cv2.INTER_AREA).tobytes())
            video_frames.append({'source_frame': source_index, 'sequence': len(history) - 1 if final or final_map else history_indices[source_index],
                                 'observation_sequence': history_indices[source_index],
                                 'localization_recovery': selected.localization_recovery if offline else None,
                                 'tail': final, 'source': source, 'trails': trails, 'hands': landmarks,
                                 'trail_display_interpolated': trail_cache.display_bridges,
                                 'trail_timestamps_s': trail_cache.trail_timestamps,
                                 'orb_detected': extracted, 'orb_tracked': matched,
                                 'orb_offline_matched': len(offline_matches),
                                 'orb_offline_match_rms_px': recovered_features.get('rms_px'),
                                 'orb_offline_map_point_ids': [int(f[2]) for f in offline_matches],
                                 'orb_offline_match_source': recovered_features.get('source'),
                                 'orb_offline_display_only': bool(display_only),
                                 'orb_offline_match_map_id': recovered_features.get('map_id'),
                                 'orb_map_point_ids': local_map_point_ids,
                                 'visible_marker_ids': sorted(
                                     int(mid) for mid in accepted_ids[source_index]
                                     if int(mid) not in temporal_excluded
                                 ),
                                 'world_axes_visible': world_axes_visible,
                                 'metric': selected.metric, 'map_revision': selected.revision,
                                 'anchor_consistency': selected.anchor_consistency,
                                 'metric_recovered_later': selected.metric_recovered_later,
                                 'marker_world_view': marker_view,
                                 'loop_event_ids_reached': [event['id'] for event in loop_events
                                                            if event['source_frame'] <= source_index],
                                 'loop_event_ids_active': [event['id'] for event in loop_events
                                                           if event['source_frame'] == source_index],
                                 'camera': {'rotation': camera.rotation_matrix.tolist(), 'translation': camera.tvec.reshape(3).tolist()}
                                 if camera else None, 'map_id': map_id})
            if completion_frame is not None:
                for key in ('trail_sample_sources', 'trail_display_interpolated', 'wrist_completion_source'):
                    video_frames[-1][key] = completion_frame[key]
            if hybrid:
                # These fields are deliberately in the historical map gauge.
                # Never project final hand/world labels into that local map.
                process_revision = history[-1] if final else snapshot
                process_selected = replay_camera_frame(snapshot, process_revision, action)
                process_camera, process_map_id = process_selected.pose, process_selected.map_id
                if process_camera is None or not process_selected.metric:
                    process_camera, process_map_id = camera_at_revision(snapshot, process_revision)
                process_mapping = next((m for m in process_revision['maps']
                                        if f"atlas_{m['id']}" == process_map_id), {})
                video_frames[-1].update(
                    global_sequence=len(history) - 1,
                    process_camera=({'rotation': process_camera.rotation_matrix.tolist(),
                                     'translation': process_camera.tvec.reshape(3).tolist()}
                                    if process_camera else None),
                    process_map_id=process_map_id,
                    process_map_revision=process_mapping.get('revision', 0),
                    process_metric=bool(process_camera is not None and process_selected.metric),
                    process_source=process_selected.source,
                    process_orb_map_point_ids=process_map_point_ids,
                    process_marker_world_view=_marker_world_view(process_selected, marker_layout),
                )
    finally:
        capture.release()
        encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError('H.264 encoding failed')
    with gzip.open(directory / 'video_frames.json.gz', 'wt') as stream:
        json.dump(video_frames, stream, separators=(',', ':'), allow_nan=False)
    marker_graph_capabilities = _marker_graph_capabilities(history)
    metric_visual_loop = bool(history) and all(
        snapshot.get('marker_graph_capabilities', {}).get('metric_tag_visual_loop') is True
        for snapshot in history)
    metric_tag_pose_graph_factors = bool(history) and all(
        snapshot.get('marker_graph_capabilities', {}).get('metric_tag_pose_graph_factors') is True
        for snapshot in history)
    metric_visual_merge = bool(history) and all(
        snapshot.get('marker_graph_capabilities', {}).get('metric_tag_visual_merge') is True
        for snapshot in history)
    metric_loop_policy = (
        'same-map metric visual loops fix one gauge keyframe and jointly optimize the other '
        'keyframes with robust original marker-corner projection factors; '
        if metric_tag_pose_graph_factors else
        'same-map metric visual-loop behavior follows the recorded legacy native backend; '
    )
    unsupported = ['intra-frame point lifecycle events']
    if not marker_graph_capabilities['marker_map_merge']:
        unsupported.append('marker-assisted map merge')
    if not marker_graph_capabilities['interval_scale_reanchor']:
        unsupported.append('interval scale re-anchoring')
    if not marker_graph_capabilities['metric_tag_global_ba']:
        unsupported.append('metric/tag-map global BA without joint marker optimization')
    if not metric_visual_loop:
        unsupported.append('metric/tag-map same-map pure-visual loop with a fixed marker gauge')
    if not metric_visual_merge:
        unsupported.append('metric/tag-map cross-map pure-visual merge without a common marker')
    manifest = {'schema': 'native-orb-frame-publications/v1', 'fps': output_fps, 'source_fps': fps,
                'replay_mode': 'hybrid' if hybrid else 'final-map' if final_map else 'process',
                'final_labels_source': str(actions_path.resolve()) if offline else None,
                'display_semantics': ('hybrid: final Atlas from frame zero and final-label video/hands; '
                                      'local view uses historical native publications in their own gauge; '
                                      'accepted loop edges and a before-to-after correction overlay are revealed '
                                      'at the candidate keyframe time; publication time remains explicit; '
                                      'invalid final poses stay invalid; no final hands in process-map coordinates'
                                      if hybrid else 'offline final reconstruction: final point cloud from frame zero; '
                                      'camera, wrist and hand labels share the final map revision; '
                                      'historical native events retained, not an online mapping process'
                                      if final_map else 'actual native frame-publication process'),
                'loop_events': loop_events,
                'loop_replay': {
                    'enabled': bool(loop_events),
                    'clock': 'candidate keyframe timestamp; commit/publication timestamp retained separately',
                    'overlay': 'LOOP CORRECTION | BEFORE -> AFTER',
                    'local_view_policy': ('show the causal process map plus a red-before/green-after '
                                          'correction overlay when each accepted edge is reached'),
                },
                'browser_timeline': 'timeline.browser.json.gz',
                'rendered_at_s': time.time(),
                'video_overlays': {'orb_features': 'green native tracked box+dot; cyan verified offline final-map matches; unmatched detections hidden',
                                   'offline_feature_matching': bool(offline and offline_features),
                                   'world_trails': ('final optimized world wrist/hand labels in the final map revision; '
                                                    'display-only smoothing and bracketed <= 0.5 s wrist gaps; '
                                                    'invalid cameras and cross-map gaps remain broken; labels unchanged'
                                                    if offline else 'offline optimized wrist measurements, mapped through the '
                                                    'current committed revision, then causal display filtering; '
                                                    'display-only interpolation of bracketed wrist gaps <= 0.5 s '
                                                    'with valid same-world cameras, then causal display filtering; '
                                                    'camera/world failures and longer gaps remain broken; labels unchanged'),
                                   'trail_gap_display_max_s': DISPLAY_TRAIL_MAX_GAP_SECONDS,
                                   'world_axes_length_m': WORLD_AXIS_LENGTH_M,
                                   'world_axes_pose': ('inverse final optimized metric camera world pose' if offline
                                                       else 'inverse current committed metric camera world pose')},
                'local_view': 'process',
                'video': 'process.mp4', 'frames': len(video_frames), 'analysis_frames': len(actions),
                'final_hold_seconds': 0. if final_map else FINAL_REPLAY_HOLD_SECONDS,
                'checkpoint_interval_s': 5, 'point_record_bytes': 28,
                'history_semantics': 'native state published after each input frame; final shutdown publication is a labelled tail',
                'offline_preprocessing': {
                    'marker_bootstrap_observations': sum(bool(a.get('marker_pose_offline_evidence')) for a in actions),
                    'marker_bootstrap_available_after_s': max(
                        (a['marker_pose_offline_evidence']['available_after_s'] for a in actions
                         if a.get('marker_pose_offline_evidence')), default=None),
                    'prefix_relocalized_final_labels': sum((a.get('camera_localization_recovery') or {}).get('method') == 'native-orb-final-atlas-prefix-pnp' for a in actions),
                    'short_gap_relocalized_final_labels': sum((a.get('camera_localization_recovery') or {}).get('method') == 'native-orb-final-atlas-short-gap-pnp' for a in actions),
                    'policy': ('short marker-free ORB prefix analysis may disambiguate measured marker hints '
                               'before the native replayed pass; this is an offline, noncausal algorithm. '
                               'Separate final-Atlas prefix relocalizations are final labels only, '
                               'not retroactively valid process frames.'),
                },
                'native_history_storage': ('feature_count plus publication-valid matched features; '
                                           'per-map point full/delta records with a complete final checkpoint'
                                           if all(h.get('point_protocol') == 'map-delta-v1' for h in history)
                                           else 'legacy full features and full map points'),
                'camera_revision_policy': 'new tag-constrained measurements follow only explicit rigid marker-world merge gauge deltas, '
                                          'never local scale re-anchoring or generic reference-keyframe BA; '
                                          'missing new-protocol historical gauges are invalid; '
                                          'legacy graph-only/absolute logs retain their recorded correction policy; '
                                          'propagate reference-keyframe pose and local-unit corrections to tag-free frames; '
                                          'arbitrary-scale historical frames with a resolvable reference are republished in metres after a committed metricization and explicitly flagged; '
                                          'LOST or unresolved historical frames remain invalid and are never interpolated; '
                                          'metric camera-local wrist/hand geometry is never rescaled; '
                                          'raw marker PnP never bypasses native Atlas marker tracking as an output-pose fallback; '
                                          'not a new dense all-frame bundle adjustment',
                'map_correction_policy': 'pure-visual loop/merge remain enabled for non-metric tag-free maps; '
                                         + metric_loop_policy +
                                         'marker interval corrections and common-marker merges require explicitly declared native capabilities '
                                         'and a validated accepted commit, independent of rejected visual candidates; '
                                         'cross-map metric pure-visual merge remains unsupported without a common marker; '
                                         'in-flight pure-visual global BA must not commit after marker units/gauge change',
                'rejection_event_clock': 'first native snapshot publication; event timestamp identifies the candidate keyframe, not commit time',
                'marker_graph_event_clock': ('native publication timestamp is retained as the commit/publication clock; '
                                             'candidate_frame/candidate_timestamp retain the observation clock; '
                                             'offline final/hybrid overview may reveal accepted corrections at the '
                                             'candidate frame, while the causal process view stays on publication time'),
                'not_yet_supported': unsupported,
                'capabilities': {'native_orb': True, 'marker_seed': True, 'final_atlas': True,
                                 **marker_graph_capabilities,
                                 'visual_loop_merge_unscaled': True,
                                 'metric_tag_visual_loop': metric_visual_loop,
                                 'metric_tag_pose_graph_factors': metric_tag_pose_graph_factors,
                                 'metric_tag_visual_merge': metric_visual_merge,
                                 'metric_tag_visual_loop_merge': metric_visual_loop and metric_visual_merge,
                                 'correction_rejection_events': all('correction_rejections' in h for h in history),
                                 'marker_graph_events': all('marker_graph_events' in h for h in history),
                                 'compact_native_history': bool(history) and all(
                                     h.get('point_protocol') == 'map-delta-v1' for h in history),
                                 'full_commit_event_journal': False},
                'generation_seconds': time.perf_counter() - started}
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    shutil.copy2(Path(__file__).with_name('slam_replay.html'), directory / 'index.html')
    shutil.copy2(Path(__file__).with_name('replay_event_overlay.js'), directory / 'replay_event_overlay.js')
    if output_path and output_path.resolve() != process_path.resolve():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # One encode; requested path is an alias of the same replay video.
        if output_path.exists():
            raise FileExistsError(output_path)
        output_path.symlink_to(process_path.resolve())
    return process_path, directory / 'index.html'
