"""Compare two completed runs, ignoring address-dependent serialization order.

Reports reproducibility, not external accuracy. Point IDs are comparable only
until their creation/lifecycle diverges; no later point-ID displacement is used.
"""
import argparse
import io
import json
from itertools import zip_longest
from pathlib import Path

import numpy as np
import zstandard


def history(root):
    path = root / "actions_replay/native_history.jsonl.zst"
    with path.open("rb") as source, zstandard.ZstdDecompressor().stream_reader(source) as stream:
        for line in io.TextIOWrapper(stream):
            yield json.loads(line)


def update_points(states, row):
    for m in row["maps"]:
        if m["points_mode"] == "full" or m["id"] not in states:
            states[m["id"]] = {}
        points = states[m["id"]]
        for p in m["points"]:
            points[p[0]] = p[1:]
        for identity in m.get("deleted_points", []):
            points.pop(identity, None)
    return {m["id"]: states[m["id"]] for m in row["maps"]}


def compare(a, b):
    first = {}
    states = [{}, {}]
    rows = 0
    point_identity_diverged = False
    for index, pair in enumerate(zip_longest(history(a), history(b))):
        if None in pair:
            first["history_length"] = {"record": index}
            break
        left, right = pair
        rows += 1
        at = {"record": index, "time_s": left["timestamp"]}

        def note(name, extra=None):
            first.setdefault(name, dict(at, **(extra or {})))

        if left["timestamp"] != right["timestamp"]:
            note("timestamp")
            break
        for key in ("state", "active_map", "feature_count", "map_tracking_inliers", "pose",
                    "matched_features", "reference", "reference_scale", "relative", "visual_relative"):
            if left.get(key) != right.get(key):
                note(key)
        lm, rm = ({m["id"]: m for m in r["maps"]} for r in pair)
        if lm.keys() != rm.keys():
            note("map_ids")
        for mid in lm.keys() & rm.keys():
            x, y = lm[mid], rm[mid]
            for key in ("keyframe_count", "point_count", "markers"):
                if x[key] != y[key]:
                    note(key)
            kx, ky = ({k[0]: k[1:] for k in m["keyframes"]} for m in (x, y))
            if kx.keys() != ky.keys():
                note("keyframe_ids")
            if kx != ky:
                note("keyframes")
        # Revision counters and pointer-based serialization order are not
        # geometry. Compare actual event constraints/results after removing
        # presentation-only ordering of ID sets.
        def event_signature(row):
            events = []
            for event in row.get("marker_graph_events", []):
                e = {k: v for k, v in event.items() if k != "revision"}
                for key in ("marker_ids", "affected_keyframes", "excluded_tag_keyframes",
                            "excluded_tag_groups"):
                    if key in e:
                        e[key] = sorted(e[key])
                events.append(e)
            return events
        if event_signature(left) != event_signature(right):
            note("marker_graph_events")
        px, py = (update_points(s, r) for s, r in zip(states, pair))
        if not point_identity_diverged:
            point_identity_diverged = (px.keys() != py.keys() or any(
                px[m].keys() != py[m].keys() for m in px.keys() & py.keys()))
            if point_identity_diverged:
                note("point_id_lifecycle")
            elif px != py and "point_coordinates" not in first:
                distances = [np.linalg.norm(np.subtract(p, py[m][identity]))
                             for m in px for identity, p in px[m].items()]
                note("point_coordinates", {"max_difference_m": float(max(distances, default=0))})

    def poses(root):
        result = {}
        with (root / "actions.jsonl").open() as source:
            for line in source:
                row = json.loads(line)
                p = row.get("camera_world_pose_fused")
                if p and row.get("camera_world_source") != "invalid":
                    result[row["frame"]] = (row["camera_submap_id"], p["translation_m"])
        return result

    x, y = poses(a), poses(b)
    frames = sorted(i for i in x.keys() & y.keys() if x[i][0] == y[i][0])
    d = np.array([np.linalg.norm(np.subtract(x[i][1], y[i][1])) for i in frames])
    return {"run_a": str(a), "run_b": str(b), "history_records_compared": rows,
            "first_difference": first, "valid_frames": [len(x), len(y)],
            "final_same_map_common_frames": len(frames),
            "final_camera_difference_m": None if not len(d) else {
                "rms": float(np.sqrt(np.mean(d*d))), "max": float(d.max()),
                "p95": float(np.percentile(d, 95))},
            "note": "Same-ID point comparison stops at first lifecycle divergence; no truth inferred."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(compare(args.run_a, args.run_b), indent=2)
    if args.output:
        args.output.write_text(report + "\n")
    print(report)
