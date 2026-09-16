#!/usr/bin/env python3
"""Offline HaWoR hand candidates. Never estimates SLAM or changes measured wrists.

Heavy dependencies are imported only when running, so --help and protocol tests
work without Torch. The model has 16-frame bidirectional context, not online pose
tracking. Missing detector boxes can be interpolated, but those predictions are
explicitly unsupported and must not be promoted to measurements by consumers.
"""
import argparse
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time


LANDMARK_NAMES = ("wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
                  "index_mcp", "index_pip", "index_dip", "index_tip",
                  "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
                  "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
                  "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip")


def select_detections(rows, start, end):
    """Keep the highest score per detector class; do not invent confidence."""
    selected = {"Left": {}, "Right": {}}
    by_frame = {}
    for row in rows:
        frame = int(row["frame"])
        if frame in by_frame:
            raise ValueError(f"Duplicate detection frame {frame}")
        by_frame[frame] = row
    for frame in range(start, end):
        if frame not in by_frame:
            raise ValueError(f"Missing detection frame {frame}")
        for cls, side in enumerate(selected):
            candidates = [(i, box) for i, box in enumerate(by_frame[frame]["boxes"])
                          if int(box["handedness_class_id"]) == cls]
            if candidates:
                i, box = max(candidates, key=lambda item: float(item[1]["detection_score"]))
                selected[side][frame] = {**box, "candidate_index": i,
                                         "same_side_candidate_count": len(candidates)}
    return selected


def choose_device(torch, requested):
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "mps" if torch.backends.mps.is_available() else "cpu"
    available = {"cpu": True, "cuda": torch.cuda.is_available(),
                 "mps": torch.backends.mps.is_available()}
    if not available[requested]:
        raise RuntimeError(f"Requested device {requested} is unavailable; no silent CPU fallback")
    return requested


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New predictions JSONL file; no overwrite")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, help="Exclusive; default video frame count")
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--repo", type=Path, required=True, help="Official HaWoR source checkout")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, help="Default: checkpoint directory/model_config.yaml")
    parser.add_argument("--mano-dir", type=Path, required=True, help="User-authorized directory containing MANO_RIGHT.pkl")
    parser.add_argument("--detector", type=Path, help="Official WiLoR YOLO weights; required without --detections")
    parser.add_argument("--detections", type=Path, help="Reuse frame-indexed detector JSONL (skip detector timing)")
    parser.add_argument("--detection-confidence", type=float, default=0.3)
    parser.add_argument("--calibration", type=Path, help="Use mean fx/fy as native scalar focal; distortion not modeled")
    parser.add_argument("--focal-length", type=float, help="Override native scalar focal in pixels; default 600 without calibration")
    args = parser.parse_args(argv)
    if args.start_frame < 0 or (args.end_frame is not None and args.end_frame <= args.start_frame):
        parser.error("Require 0 <= start-frame < end-frame")
    if args.detections is None and args.detector is None:
        parser.error("Provide --detector or --detections")
    return args


def main(argv=None):
    args = parse_args(argv)
    args.output = args.output.resolve()
    metrics_path = args.output.with_suffix(".metrics.json")
    if args.output.exists() or metrics_path.exists():
        raise FileExistsError("Output or metrics file already exists; choose a new output path")
    for path in (args.video, args.checkpoint, args.mano_dir / "MANO_RIGHT.pkl"):
        if not path.is_file():
            raise FileNotFoundError(path)
    import cv2
    import numpy as np
    import torch
    device = choose_device(torch, args.device)
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
        return time.perf_counter()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.video}")
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps, total = float(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start, end = args.start_frame, total if args.end_frame is None else args.end_frame
    if not (fps > 0 and start < end <= total):
        raise ValueError("Invalid frame range or video FPS")
    cap.release()
    focal = args.focal_length
    if focal is None and args.calibration:
        calibration = json.loads(args.calibration.read_text())
        if tuple(calibration["image_size"]) != (width, height):
            raise ValueError("Calibration image size differs from video")
        k = calibration["camera_matrix"]
        focal = (k[0][0] + k[1][1]) / 2
    focal = float(600 if focal is None else focal)
    if focal <= 0:
        raise ValueError("Focal length must be positive")
    center = np.array([width / 2, height / 2], dtype=np.float32)
    metrics = {"model": "HaWoR hand-only", "device": device, "requested_device": args.device,
               "torch": torch.__version__, "dtype": "float32", "batch_size": 1, "sequence_length": 16,
               "mps_cpu_fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0"),
               "video": str(args.video.resolve()), "start_frame": start, "end_frame_exclusive": end,
               "frames": end-start, "fps": fps, "image_size": [width, height],
               "focal_px": focal, "principal_point_px": center.tolist(), "distortion_modeled": False,
               "checkpoint": str(args.checkpoint.resolve()), "repo": str(args.repo.resolve()),
               "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "marker_camera_wrist_modified": False, "slam_used": False, "infiller_used": False,
               "bbox_policy": "highest-score per class; linear interpolation only between detected endpoints",
               "temporal_policy": "16-frame bidirectional chunks, repeated-last tail padding; not causal",
               "supported_policy": "detector-supported only; interpolated boxes are predictions, not measurements"}
    if args.detections:
        rows = [json.loads(line) for line in args.detections.read_text().splitlines()]
        metrics["detector_reused"] = True
        metrics["detector_shared_sha256"] = hashlib.sha256(args.detections.read_bytes()).hexdigest()
        metrics["detector_forward_s"] = None
    else:
        from ultralytics import YOLO
        t = sync()
        detector = YOLO(str(args.detector))
        metrics["detector_load_s"] = sync() - t
        names = {int(k): str(v).lower() for k, v in detector.names.items()}
        if names.get(0) != "left" or names.get(1) != "right":
            raise ValueError(f"Expected official detector classes 0:left,1:right, got {names}")
        cap = cv2.VideoCapture(str(args.video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        rows, decode_s, detector_s = [], 0.0, 0.0
        for frame in range(start, end):
            t = time.perf_counter()
            ok, bgr = cap.read()
            decode_s += time.perf_counter()-t
            if not ok:
                raise RuntimeError(f"Decode failed at {frame}")
            if frame == start:
                t = sync()
                detector.predict(bgr, conf=args.detection_confidence, device=device, verbose=False)
                metrics["detector_warmup_s"] = sync()-t
            t = sync()
            prediction = detector.predict(bgr, conf=args.detection_confidence, device=device, verbose=False)[0]
            elapsed = sync()-t
            detector_s += elapsed
            boxes = [{"bbox_xyxy": xy.tolist(), "detection_score": float(score),
                      "handedness_class_id": int(cls), "handedness": ("Left", "Right")[int(cls)]}
                     for xy, score, cls in zip(prediction.boxes.xyxy.cpu().numpy(),
                                                prediction.boxes.conf.cpu().numpy(),
                                                prediction.boxes.cls.cpu().numpy())]
            rows.append({"frame": frame, "timestamp_s": frame/fps, "boxes": boxes})
        cap.release()
        metrics.update(detector_reused=False, detector_forward_s=detector_s, detector_decode_s=decode_s)
        del detector, prediction
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()
    selected = select_detections(rows, start, end)
    # Compatibility aliases affect this process only, not MANO assets or libraries.
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec
    for name, val in {"bool": bool, "int": int, "float": float, "complex": complex,
                      "object": object, "unicode": str, "str": str}.items():
        if name not in np.__dict__:
            setattr(np, name, val)
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    from hawor.configs import get_config
    from lib.models.hawor import HAWOR
    from lib.utils.imutils import crop, boxes_2_cs
    cfg = get_config(str(args.model_config or args.checkpoint.parent / "model_config.yaml"))
    cfg.defrost()
    cfg.MANO.MODEL_PATH = str(args.mano_dir.resolve())
    cfg.MANO.MEAN_PARAMS = str(repo / "_DATA/data/mano_mean_params.npz")
    cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS = None
    cfg.MODEL.BACKBONE.TORCH_COMPILE = 0
    cfg.freeze()
    t = sync()
    model = HAWOR(cfg)
    state = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False, mmap=True)
    mismatch = model.load_state_dict(state["state_dict"], strict=False)
    if mismatch.missing_keys or mismatch.unexpected_keys:
        raise RuntimeError(f"Checkpoint mismatch: {mismatch}")
    del state
    model = model.to(device).eval()
    metrics["model_load_s"] = sync()-t
    boxes_by_side, centers, scales, bounds = {}, {}, {}, {}
    for side, known_dict in selected.items():
        if not known_dict:
            continue
        known = np.array(sorted(known_dict))
        span = np.arange(known[0], known[-1]+1)
        b = np.array([known_dict[int(f)]["bbox_xyxy"] for f in known])
        boxes_by_side[side] = np.stack([np.interp(span, known, b[:, j]) for j in range(4)], axis=1)
        centers[side], scales[side] = boxes_2_cs(boxes_by_side[side])
        scales[side] *= 1.2
        bounds[side] = int(span[0]), int(span[-1])+1
    output = {f: {"frame": f, "timestamp_s": f/fps, "hands": []} for f in range(start, end)}
    mean = torch.tensor([.485, .456, .406], device=device)[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device=device)[None, :, None, None]
    metrics.update(decode_s=0., preprocessing_s=0., normalize_transfer_s=0., forward_s=0., postprocess_s=0.,
                   warmup_s=0., tail_padding_hand_frames=0, chunks=[])
    warmed = False

    def run_chunk(side, items):
        nonlocal warmed
        n = len(items)
        padded = items + [items[-1]]*(16-n)
        lo, _ = bounds[side]
        indices = [f-lo for f, _ in padded]
        t = sync()
        image = torch.from_numpy(np.stack([im for _, im in padded])).to(device).permute(0, 3, 1, 2).float()/255
        ctr = centers[side][indices].copy()
        if side == "Left":
            ctr[:, 0] = width-ctr[:, 0]-1
        batch = {"img": ((image-mean)/std)[None],
                 "center": torch.tensor(ctr, device=device)[None].float(),
                 "scale": torch.tensor(scales[side][indices], device=device)[None].float(),
                 "img_focal": torch.full((1, 16), focal, device=device),
                 "img_center": torch.tensor(center, device=device).expand(1, 16, 2).clone()}
        if side == "Left":
            batch["do_flip"] = torch.ones(1, 16, device=device)
        metrics["normalize_transfer_s"] += sync()-t
        with torch.inference_mode():
            if not warmed:
                t = sync()
                # Native HaWoR mutates the left-hand center tensor in forward.
                # Keep actual inference inputs untouched by the warmup pass.
                model({key: value.clone() for key, value in batch.items()})
                metrics["warmup_s"] = sync()-t
                warmed = True
            t = sync()
            prediction = model(batch)
            elapsed = sync()-t
        metrics["forward_s"] += elapsed
        metrics["tail_padding_hand_frames"] += 16-n
        t = time.perf_counter()
        joints = prediction["pred_keypoints_3d"][:n].cpu().numpy().copy()
        if side == "Left":
            joints[:, :, 0] *= -1
        trans = prediction["out"]["trans_full"][:n].cpu().numpy()
        relative = joints-joints[:, :1]
        cam = joints+trans
        uv = cam[:, :, :2]/cam[:, :, 2:]*focal+center
        if not (np.isfinite(uv).all() and np.isfinite(relative).all()):
            raise RuntimeError("Nonfinite HaWoR prediction")
        for j, (frame, _) in enumerate(items):
            det = selected[side].get(frame)
            normalized = np.c_[uv[j]/[width, height], np.zeros(21)]
            output[frame]["hands"].append({
                "handedness": side, "handedness_score": 0.0, "handedness_score_valid": False,
                "detection_score": None if det is None else float(det["detection_score"]),
                "detector_supported": det is not None, "image_supported": det is not None,
                "input_frame_available": True, "detection_source": "hawor_temporal" if det else "hawor_interpolated_bbox",
                "status": "detected_temporal_prediction" if det else "interpolated_bbox_temporal_prediction",
                "landmark_names": LANDMARK_NAMES, "image_landmarks_normalized": normalized.tolist(),
                "model_landmarks_m": relative[j].tolist(), "bbox_xyxy": boxes_by_side[side][frame-lo].tolist(),
                "context_start_frame": items[0][0], "context_end_frame_inclusive": items[-1][0],
                "future_frames_used": items[-1][0]-frame, "chunk_repeat_padding_frames": 16-n})
        metrics["postprocess_s"] += time.perf_counter()-t
        metrics["chunks"].append({"side": side, "start_frame": items[0][0], "end_frame_exclusive": items[-1][0]+1,
                                  "forward_s": elapsed, "tail_padding_hand_frames": 16-n})
        print(f"{device} {side} {items[0][0]}:{items[-1][0]+1} forward {elapsed:.3f}s", file=sys.stderr, flush=True)

    cap = cv2.VideoCapture(str(args.video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    pending = {side: [] for side in bounds}
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    mps_samples = []
    for frame in range(start, end):
        t = time.perf_counter()
        ok, bgr = cap.read()
        metrics["decode_s"] += time.perf_counter()-t
        if not ok:
            raise RuntimeError(f"Decode failed at {frame}")
        for side, (lo, hi) in bounds.items():
            if not lo <= frame < hi:
                continue
            t = time.perf_counter()
            rgb, ctr = bgr[:, :, ::-1], centers[side][frame-lo].copy()
            if side == "Left":
                rgb = rgb[:, ::-1]
                ctr[0] = width-ctr[0]-1
            image = crop(rgb, ctr, scales[side][frame-lo], [256, 256], rot=0).astype(np.uint8)
            metrics["preprocessing_s"] += time.perf_counter()-t
            pending[side].append((frame, image))
            if len(pending[side]) == 16 or frame == hi-1:
                run_chunk(side, pending[side])
                pending[side] = []
                if device == "mps":
                    mps_samples.append(torch.mps.driver_allocated_memory())
    cap.release()
    metrics["mps_driver_memory_sampled_max_bytes"] = max(mps_samples, default=None)
    metrics["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated() if device == "cuda" else None
    metrics["supported_hand_frames"] = sum(h["detector_supported"] for r in output.values() for h in r["hands"])
    metrics["inferred_hand_frames"] = sum(not h["detector_supported"] for r in output.values() for h in r["hands"])
    metrics["status"] = "complete"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        for row in output.values():
            f.write(json.dumps(row, separators=(",", ":"))+"\n")
    metrics_path.write_text(json.dumps(metrics, indent=2)+"\n")
    print(json.dumps({k: v for k, v in metrics.items() if k != "chunks"}, indent=2))


if __name__ == "__main__":
    main()
