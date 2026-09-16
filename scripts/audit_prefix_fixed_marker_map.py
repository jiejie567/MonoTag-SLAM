#!/usr/bin/env python3
"""Run a prefix with a saved independent-marker layout for controlled audits.

The saved layout is only the initialization geometry. Native marker BA remains
enabled. Remaining arguments are passed unchanged to export_action_labels.
"""
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import export_action_labels as exporter
from aruco_track.auto_marker_map import AutoMarkerMap, AutoMarkerSubmap
from aruco_track.models import BandLayout, Pose


def main():
    source = Path(sys.argv[1])
    data = json.loads(source.read_text())
    submaps = []
    for group in data['submaps']:
        poses, points = {}, {}
        for marker in group['markers']:
            mid = marker['id']
            value = marker['world_from_marker']
            poses[mid] = Pose(cv2.Rodrigues(np.asarray(value['rotation_matrix']))[0],
                              np.asarray(value['translation_m']).reshape(3, 1), 0.)
            points[mid] = np.asarray(marker['object_points_m'])
        submaps.append(AutoMarkerSubmap(group['submap_id'], group['anchor_marker_id'],
            poses, BandLayout(group['submap_id'], data['dictionary'], points), (),
            group.get('reprojection_error_px')))
    layout = AutoMarkerMap(data['dictionary'], data['marker_size_mm']/1000.,
        tuple(submaps), data['mode'], tuple(data.get('pending_marker_ids', [])))
    exporter.build_auto_marker_map = lambda *args, **kwargs: layout
    sys.argv = ['export_action_labels.py'] + sys.argv[2:]
    print('Controlled audit: fixed initialization layout from', source, flush=True)
    exporter.main()


if __name__ == '__main__':
    main()
