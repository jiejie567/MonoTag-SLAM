#!/usr/bin/env python3
"""Build a local catalog for every complete interactive SLAM replay."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import html
import json
from pathlib import Path
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "replay_catalog" / "index.html"
MIN_SOURCE_FPS = 50.0
MIN_DURATION_S = 10.0
REQUIRED = (
    "index.html",
    "manifest.json",
    "timeline.json.gz",
    "video_frames.json.gz",
    "points.bin.gz",
    "process.mp4",
)


def _replay_directory(value: str) -> Path:
    path = Path(value)
    return path.parent if path.suffix else path


def _url(path: Path) -> str:
    return "/" + quote(str(path.resolve().relative_to(ROOT)), safe="/")


def _read_items() -> list[dict]:
    items = []
    for metadata_path in ROOT.rglob("*.meta.json"):
        try:
            metadata = json.loads(metadata_path.read_text())
            calibration_name = Path(str(metadata.get("calibration", ""))).name
            if not calibration_name.startswith("camera_usb_"):
                continue
            replay_value = metadata.get("replay")
            if not replay_value:
                continue
            directory = _replay_directory(replay_value).resolve()
            directory.relative_to(ROOT)
            if not all((directory / name).is_file() for name in REQUIRED):
                continue
            manifest = json.loads((directory / "manifest.json").read_text())
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        video = str(Path(metadata.get("video", "unknown")).resolve())
        capabilities = manifest.get("capabilities", {})
        metric_yield = metadata.get("valid_metric_label_yield", {}).get("camera_fraction")
        source_fps = float(manifest.get("source_fps") or metadata.get("fps") or 0)
        if source_fps < MIN_SOURCE_FPS:
            continue
        analysis_frames = int(manifest.get("analysis_frames") or 0)
        seconds = analysis_frames / source_fps if source_fps else 0
        if seconds <= MIN_DURATION_S:
            continue
        items.append({
            "video": video,
            "dataset": Path(video).name,
            "directory": directory,
            "relative": str(directory.relative_to(ROOT)),
            "url": _url(directory / "index.html"),
            "analysis_frames": analysis_frames,
            "source_fps": source_fps,
            "seconds": seconds,
            "metric_yield": metric_yield,
            "capabilities": sum(bool(value) for value in capabilities.values()),
            "keyframes": metadata.get("head_slam", {}).get("keyframes"),
            "points": metadata.get("head_slam", {}).get("map_points"),
            "modified": metadata_path.stat().st_mtime,
        })
    return items


def _recommended(group: list[dict]) -> dict:
    maximum_frames = max(item["analysis_frames"] for item in group)
    complete = [item for item in group
                if item["analysis_frames"] >= maximum_frames * 0.98]
    return max(complete, key=lambda item: (item["capabilities"], item["modified"]))


def _metric(value) -> str:
    return "—" if value is None else f"{100 * float(value):.1f}%"


def _count(value) -> str:
    return "—" if value is None else f"{int(value):,}"


def _row(item: dict, recommended: bool = False) -> str:
    badge = '<span class="badge">推荐</span>' if recommended else ""
    return (
        "<tr>"
        f'<td><a href="{item["url"]}" target="_blank">{html.escape(item["dataset"])}</a>{badge}'
        f'<small>{html.escape(item["relative"])}</small></td>'
        f'<td>{item["seconds"]:.1f} s<small>{item["analysis_frames"]:,} 帧</small></td>'
        f'<td>{item["source_fps"]:.1f}</td>'
        f'<td>{_metric(item["metric_yield"])}</td>'
        f'<td>{_count(item["keyframes"])}</td>'
        f'<td>{_count(item["points"])}</td>'
        f'<td>{item["capabilities"]}</td>'
        "</tr>"
    )


def main() -> None:
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in _read_items():
        groups[item["video"]].append(item)
    recommended = [_recommended(group) for group in groups.values()]
    recommended.sort(key=lambda item: item["modified"], reverse=True)
    all_items = sorted((item for group in groups.values() for item in group),
                       key=lambda item: (item["dataset"], -item["modified"]))
    generated = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>全局快门 Marker–ORB 数据集回放</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#f5f6f8;color:#1c2734;font:14px system-ui,sans-serif}}
main{{max-width:1240px;margin:auto;padding:28px}}h1{{margin:0 0 6px;font-size:26px}}p{{color:#647085;margin:0 0 20px}}
.summary{{display:flex;gap:12px;margin:16px 0 22px}}.stat{{background:#fff;border:1px solid #dce0e6;border-radius:10px;padding:12px 18px}}
.stat strong{{display:block;font-size:22px;color:#344a78}}table{{width:100%;border-collapse:collapse;background:#fff;border:1px solid #dce0e6}}
th,td{{padding:10px 12px;text-align:left;border-bottom:1px solid #e7e9ed}}th{{position:sticky;top:0;background:#eef1f5;color:#445064}}
tr:hover{{background:#f7f9fc}}a{{color:#245fae;font-weight:650;text-decoration:none}}a:hover{{text-decoration:underline}}
small{{display:block;color:#7a8492;font-weight:400;margin-top:3px}}.badge{{margin-left:7px;padding:2px 6px;border-radius:10px;background:#e1eee4;color:#3f6f4a;font-size:11px}}
details{{margin-top:22px}}summary{{cursor:pointer;font-weight:650;padding:10px 0}}footer{{color:#788391;margin:18px 0}}
</style></head><body><main>
<h1>全局快门 Marker–ORB 交互回放</h1>
<p>仅保留使用 USB/UVC 全局快门相机标定、源帧率不低于 {MIN_SOURCE_FPS:.0f} FPS 且时长超过 {MIN_DURATION_S:.0f} 秒的数据。</p>
<div class="summary"><div class="stat"><strong>{len(groups)}</strong>个原始数据集</div>
<div class="stat"><strong>{len(all_items)}</strong>个可用历史版本</div></div>
<h2>推荐完整版本</h2><table><thead><tr><th>数据集 / 回放目录</th><th>分析时长</th><th>源FPS</th><th>米制相机有效率</th><th>关键帧</th><th>地图点</th><th>能力项</th></tr></thead>
<tbody>{''.join(_row(item, True) for item in recommended)}</tbody></table>
<details><summary>展开全部 {len(all_items)} 个当前最佳版本</summary>
<table><thead><tr><th>数据集 / 回放目录</th><th>分析时长</th><th>源FPS</th><th>米制相机有效率</th><th>关键帧</th><th>地图点</th><th>能力项</th></tr></thead>
<tbody>{''.join(_row(item) for item in all_items)}</tbody></table></details>
<footer>生成时间：{generated}。全部条目均由当前最佳流程重新生成。</footer>
</main></body></html>"""
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(document)
    print(OUTPUT)
    print(json.dumps({"datasets": len(groups), "replays": len(all_items)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
