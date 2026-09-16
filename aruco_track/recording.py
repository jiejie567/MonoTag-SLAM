from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
from queue import Full, Queue
from threading import Condition, Event, Lock, Thread
import time
from typing import Any

import cv2


@dataclass(frozen=True)
class RawRecordingResult:
    path: Path
    metadata_path: Path
    frames: int
    dropped_frames: int
    fps: float
    duration_s: float


class RawVideoRecorder:
    """Asynchronously save untouched camera frames without blocking capture."""

    def __init__(
        self,
        image_size: tuple[int, int],
        fps: float,
        output_dir: Path = Path("recordings"),
        queue_size: int = 240,
        metadata: dict | None = None,
    ):
        self.image_size = image_size
        self.fps = max(1.0, float(fps))
        self.output_dir = output_dir
        self.queue_size = max(1, int(queue_size))
        self.metadata = dict(metadata or {})
        self.path: Path | None = None
        self.frames = 0
        self.dropped_frames = 0
        self._writer: cv2.VideoWriter | None = None
        self._queue: Queue[Any] | None = None
        self._thread: Thread | None = None
        self._lock = Lock()
        self._written_frames = 0
        self._first_timestamp_s: float | None = None
        self._last_timestamp_s: float | None = None

    @property
    def active(self) -> bool:
        with self._lock:
            return self._writer is not None

    def start(self, output: Path | None = None) -> Path:
        with self._lock:
            if self._writer is not None:
                assert self.path is not None
                return self.path
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = output or self.output_dir / f"raw_{datetime.now():%Y%m%d_%H%M%S_%f}.avi"
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix.lower() != ".avi":
                raise ValueError("raw recording output must use the .avi extension")
            writer = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                self.fps,
                self.image_size,
            )
            if not writer.isOpened():
                raise RuntimeError(f"cannot create raw recording: {path}")
            queue: Queue[Any] = Queue(maxsize=self.queue_size)
            self.path = path
            self.frames = 0
            self.dropped_frames = 0
            self._written_frames = 0
            self._first_timestamp_s = None
            self._last_timestamp_s = None
            self._writer = writer
            self._queue = queue
            self._thread = Thread(
                target=self._write_frames,
                args=(queue, writer),
                daemon=True,
                name="raw-video-writer",
            )
            self._thread.start()
            return path

    def write(self, frame, timestamp_s: float | None = None) -> bool:
        if (frame.shape[1], frame.shape[0]) != self.image_size:
            with self._lock:
                self.dropped_frames += 1
            return False
        timestamp_s = time.monotonic() if timestamp_s is None else float(timestamp_s)
        with self._lock:
            if self._queue is None:
                return False
            try:
                self._queue.put_nowait((frame, timestamp_s))
            except Full:
                self.dropped_frames += 1
                return False
            self.frames += 1
            return True

    def _write_frames(self, queue: Queue[Any], writer: cv2.VideoWriter) -> None:
        while True:
            item = queue.get()
            if item is None:
                return
            frame, timestamp_s = item
            writer.write(frame)
            self._written_frames += 1
            if self._first_timestamp_s is None:
                self._first_timestamp_s = timestamp_s
            self._last_timestamp_s = timestamp_s

    def stop(self) -> RawRecordingResult | None:
        with self._lock:
            if (
                self._writer is None
                or self._queue is None
                or self._thread is None
                or self.path is None
            ):
                return None
            writer = self._writer
            queue = self._queue
            thread = self._thread
            path = self.path
            dropped_frames = self.dropped_frames
            self._writer = None
            self._queue = None
            self._thread = None
            self.path = None
        queue.put(None)
        thread.join()
        writer.release()
        frames = self._written_frames
        duration_s = 0.0
        if self._first_timestamp_s is not None and self._last_timestamp_s is not None:
            duration_s = max(0.0, self._last_timestamp_s - self._first_timestamp_s)
        metadata_path = path.with_suffix(".meta.json")
        result = RawRecordingResult(
            path=path,
            metadata_path=metadata_path,
            frames=frames,
            dropped_frames=dropped_frames,
            fps=self.fps,
            duration_s=duration_s,
        )
        payload = asdict(result)
        payload["path"] = str(path)
        payload["metadata_path"] = str(metadata_path)
        payload["image_size"] = list(self.image_size)
        payload.update(self.metadata)
        metadata_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return result


class LatestFrameCapture:
    """Continuously capture frames while consumers process only the newest one."""

    def __init__(
        self,
        capture: cv2.VideoCapture,
        recorder: RawVideoRecorder | None = None,
    ):
        self.capture = capture
        self.recorder = recorder
        self._condition = Condition()
        self._stop = Event()
        self._thread: Thread | None = None
        self._latest = None
        self._latest_timestamp_s = 0.0
        self._sequence = -1
        self._capture_started_s = 0.0
        self._captured_frames = 0
        self._ended = False

    @property
    def measured_fps(self) -> float:
        if self._capture_started_s <= 0.0:
            return 0.0
        elapsed = time.monotonic() - self._capture_started_s
        return self._captured_frames / elapsed if elapsed > 0.0 else 0.0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._capture_started_s = time.monotonic()
        self._thread = Thread(target=self._capture_loop, daemon=True, name="camera-capture")
        self._thread.start()

    def _capture_loop(self) -> None:
        try:
            while not self._stop.is_set():
                ok, frame = self.capture.read()
                if not ok:
                    return
                timestamp_s = time.monotonic()
                self._captured_frames += 1
                if self.recorder is not None:
                    self.recorder.write(frame, timestamp_s)
                with self._condition:
                    self._latest = frame
                    self._latest_timestamp_s = timestamp_s
                    self._sequence += 1
                    self._condition.notify_all()
        finally:
            with self._condition:
                self._ended = True
                self._condition.notify_all()

    def read_latest(
        self, after_sequence: int, timeout_s: float = 1.0
    ) -> tuple[int, float, Any] | None:
        with self._condition:
            self._condition.wait_for(
                lambda: self._sequence > after_sequence or self._ended,
                timeout=max(0.0, timeout_s),
            )
            if self._sequence <= after_sequence:
                return None
            return self._sequence, self._latest_timestamp_s, self._latest

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.capture.release()
