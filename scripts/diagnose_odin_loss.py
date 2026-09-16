"""Read-only diagnostic: can pre-loss map correspondences survive with LK/PnP?"""
import io
import json
from pathlib import Path
import cv2
import numpy as np
import zstandard

BASE = Path(__file__).resolve().parents[1] / 'output/odin_raw_20260909_094004_local'
points = {}
history = {}
with open(BASE / 'actions_replay/native_history.jsonl.zst', 'rb') as src:
    stream = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(src))
    for index, line in enumerate(stream):
        row = json.loads(line)
        for m in row['maps']:
            if m['id'] != 0:
                continue
            for p in m.get('points', []):
                points[int(p[0])] = p[1:4]
            for pid in m.get('deleted_points', []):
                points.pop(int(pid), None)
        if 380 <= index <= 383:
            history[index] = (row, dict(points))
        if index >= 383:
            break

cap = cv2.VideoCapture(str(BASE / 'rectified_lossless.mp4'))
images = {}
for i in range(395):
    ok, im = cap.read()
    if not ok:
        raise RuntimeError(i)
    if i >= 380:
        images[i] = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
cap.release()
K = np.array([[733.88714582321018, 0, 799.5], [0, 733.68907130221851, 647.5], [0, 0, 1.]])
results = []
for start, (row, cloud) in history.items():
    matches = [p for p in row['matched_features'] if int(p[2]) in cloud]
    xy = np.float32([p[:2] for p in matches]).reshape(-1, 1, 2)
    xyz = np.float64([cloud[int(p[2])] for p in matches])
    last = images[start]
    for end in range(start + 1, 395):
        curr = images[end]
        nxt, good, _ = cv2.calcOpticalFlowPyrLK(last, curr, xy, None, winSize=(31, 31), maxLevel=4)
        back, good_back, _ = cv2.calcOpticalFlowPyrLK(curr, last, nxt, None, winSize=(31, 31), maxLevel=4)
        valid = (good.ravel() > 0) & (good_back.ravel() > 0) & (np.linalg.norm(back-xy, axis=2).ravel() < 1.)
        xy, xyz = nxt[valid], xyz[valid]
        if len(xy) < 6:
            break
        cv2.setRNGSeed(42)
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(xyz, xy, K, None, iterationsCount=300, reprojectionError=3., confidence=.999, flags=cv2.SOLVEPNP_EPNP)
        result = {'seed_frame': start, 'frame': end, 'fb_valid': len(xy), 'pnp_inliers': 0 if inliers is None else len(inliers)}
        if ok and inliers is not None and len(inliers) >= 6:
            inds = inliers.ravel()
            rvec, tvec = cv2.solvePnPRefineLM(xyz[inds], xy[inds], K, None, rvec, tvec)
            projected, _ = cv2.projectPoints(xyz[inds], rvec, tvec, K, None)
            result['rms_px'] = float(np.sqrt(np.mean(np.sum((projected-xy[inds])**2, axis=2))))
            rot = cv2.Rodrigues(rvec)[0]
            result['camera_position_m'] = (-rot.T @ tvec).ravel().tolist()
            result['positive_depth_fraction'] = float(np.mean((xyz[inds] @ rot.T + tvec.ravel())[:, 2] > 0))
        results.append(result)
        last = curr
print(json.dumps(results, indent=2))
