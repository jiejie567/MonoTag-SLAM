"""Independent USB camera preview with hardware controls; Q/Esc exits."""
import sys
import argparse
import time
import threading
import tempfile
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aruco_track.camera_controls import UVCController
from aruco_track.detector import ArucoDetector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-controls', action='store_true')
    parser.add_argument('--fps', type=float, default=90)
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error('--fps must be positive')
    controller = UVCController("1bcf:28c4")
    caps = controller.capabilities()
    detector = ArucoDetector()
    capture = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not capture.isOpened():
        raise RuntimeError("Cannot open camera 0")
    controls_visible = not args.no_controls
    window = "Camera tuning - Q to exit" if controls_visible else "Camera preview - Q to exit"
    try:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        capture.set(cv2.CAP_PROP_FPS, args.fps)
        print(f'Requested {args.fps:g} FPS; camera reports {capture.get(cv2.CAP_PROP_FPS):g}', flush=True)
        if controls_visible:
            controller._set_verified("exposure_auto", 1)
            controller._set_verified("exposure_time_absolute", 80)
            controller._set_verified("gamma", 160)
            controller._set_verified("brightness", 8)
        caps = controller.capabilities()
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 1100, 850 if controls_visible else 619)
        specs = [
            ("Auto exposure", "exposure_auto", 0, 1, 0),
            ("Exposure x0.1ms", "exposure_time_absolute", 3, 333, 80),
            ("Auto WB", "white_balance_temperature_auto", 0, 1, caps["white_balance_temperature_auto"]["value"]),
            ("WB Kelvin", "white_balance_temperature", 2800, 6500, caps["white_balance_temperature"]["value"]),
            ("Gamma", "gamma", 100, 300, caps["gamma"]["value"]),
            ("Brightness +64", "brightness", 0, 128, caps["brightness"]["value"] + 64),
        ]
        if not controls_visible:
            specs = []
        applied = {}
        pending = {}
        for label, name, lower, upper, initial in specs:
            cv2.createTrackbar(label, window, int(initial), upper, lambda _: None)
            cv2.setTrackbarMin(label, window, lower)
            applied[name] = int(initial)
        stop = threading.Event()
        latest = {"frame": None, "fps": 0.0, "error": None}
        def acquire():
            start, count = time.monotonic(), 0
            while not stop.is_set():
                ok, frame = capture.read()
                if not ok:
                    latest["error"] = "Camera stopped delivering frames"
                    break
                latest["frame"] = frame
                count += 1
                now = time.monotonic()
                if now - start >= 2:
                    latest["fps"] = count / (now - start)
                    print(f"Capture measured: {latest['fps']:.1f} FPS", flush=True)
                    start, count = now, 0
        worker = threading.Thread(target=acquire, daemon=True)
        worker.start()
        snapshot_at = time.monotonic() + 5
        snapshot_saved = False
        status = "Hardware controls | manual exposure 8 ms | no ISO control" if controls_visible else "Marker IDs only | new lens: calibration required for pose"
        print(status, flush=True)
        while True:
            if latest["error"]:
                raise RuntimeError(latest["error"])
            frame = latest["frame"]
            if frame is None:
                cv2.waitKey(1)
                continue
            now = time.monotonic()
            if not snapshot_saved and now >= snapshot_at:
                path = Path(tempfile.mkdtemp(prefix="camera_single_stream_")) / "preview.png"
                cv2.imwrite(str(path), frame)
                print(f"Snapshot: {path}", flush=True)
                snapshot_saved = True
            for label, name, lower, upper, initial in specs:
                value = cv2.getTrackbarPos(label, window)
                if value == applied[name]:
                    pending.pop(name, None)
                    continue
                if name not in pending or pending[name][0] != value:
                    pending[name] = (value, now)
                if now - pending[name][1] < 0.25:
                    continue
                try:
                    if name == "exposure_time_absolute":
                        controller._set_verified("exposure_auto", 1)
                        cv2.setTrackbarPos("Auto exposure", window, 0)
                        applied["exposure_auto"] = 0
                    if name == "white_balance_temperature":
                        controller._set_verified("white_balance_temperature_auto", 0)
                        cv2.setTrackbarPos("Auto WB", window, 0)
                        applied["white_balance_temperature_auto"] = 0
                    native = (8 if value else 1) if name == "exposure_auto" else value
                    if name == "brightness":
                        native -= 64
                    actual = controller._set_verified(name, native)
                    status = f"{name} = {actual}"
                    applied[name] = value
                    print(status, flush=True)
                except Exception as exc:
                    status = str(exc)
                    cv2.setTrackbarPos(label, window, applied[name])
                pending.pop(name, None)
            preview = cv2.resize(frame, (1100, 619))
            detections = detector.detect(frame)
            for marker_id, corners in detections.items():
                points = corners.copy()
                points[:, 0] *= 1100 / frame.shape[1]
                points[:, 1] *= 619 / frame.shape[0]
                points = points.astype("int32")
                cv2.polylines(preview, [points], True, (0, 220, 0), 2)
                cv2.putText(preview, f"ID {marker_id}", tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 220, 0), 2)
            cv2.putText(preview, f"Capture {latest['fps']:.1f} FPS | display <=30 FPS | Q: exit", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
            cv2.putText(preview, status[:110], (15, 52), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)
            cv2.putText(preview, f"Markers: {len(detections)}", (15, 76), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)
            cv2.imshow(window, preview)
            if cv2.waitKey(1) & 255 in (27, ord("q")):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            time.sleep(max(0, 1 / 30 - (time.monotonic() - now)))
    finally:
        if "stop" in locals():
            stop.set()
            worker.join(timeout=2)
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
