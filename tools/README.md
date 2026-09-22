# Command-line utilities

Run commands from the repository root after activating the environment.
The production entry remains `python process_monotag.py ...`.
Only organization changed; algorithms and default parameters are unchanged.

| Purpose | Commands |
|---|---|
| Camera capture | `list_cameras.py`, `record.py`, `track.py`, `compare_camera_fps.py` |
| Calibration | `calibrate.py`, `calibrate_band_layout.py`, `visualize_calibration.py` |
| Printable assets | `make_band.py`, `make_markers.py`, `make_fixed_marker_pack.py`, `make_world_board.py` |
| Standalone replay/export | `replay.py`, `export_world_trajectory.py`, `prepare_orbslam3_sequence.py`, `preview_marker_cover.py` |
| Diagnostics | `audit_orb_videos.py`, `compare_marker_corner_policy.py` |

Examples:

```bash
python tools/record.py --help
python tools/calibrate.py --help
python tools/make_band.py --help
```

Update old commands from `python record.py ...` to
`python tools/record.py ...`. Imports now use
`from tools.make_band import band_layout`, for example.
Relative input/output paths still resolve against the working directory.
Inspect generation scripts without argument parsers before execution.

Build/deployment and batch-experiment scripts remain under `scripts/`.
The only root Python entry is `process_monotag.py`. Processing/export/replay
commands also live here: `export_action_labels.py`, `export_lerobot_dataset.py`,
`refresh_hand_labels.py`, `render_slam_replay.py`, `replay_orb_slam.py`,
`validate_native_sequence.py`, and `verify_slam_replay.py`.
Python callers must import these modules through `tools`.

## 中文说明

原根目录的采集、标定、打印文件生成及独立诊断工具已移入本目录。
旧命令只需在脚本名前加 `tools/`；请从仓库根目录运行。
本次不改变算法和参数，也不删除实验代码。
