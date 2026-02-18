import threading
import time
from dataclasses import dataclass, asdict
from typing import Optional, Dict

import cv2
import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from ultralytics import YOLO

from ballscope.config import AiConfig

BALL_CLASS_ID = 0


@dataclass
class AiStatus:
    running: bool = False
    fps: float = 0.0
    last_conf: float = 0.0
    last_seen_sec: float = 0.0
    state: str = "idle"
    source: str = "camL"


class AiWorker:
    def __init__(self, config: AiConfig):
        self.config = config
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._latest_jpeg: Optional[bytes] = None
        self._jpeg_seq = 0

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._stop = threading.Event()
        self._enabled = threading.Event()

        self._model: Optional[YOLO] = None
        self._source_frame_fn = None

        self._center = None
        self._last_seen_ts: Optional[float] = None
        self.status = AiStatus()
        self._frame_times = []

    def attach_source(self, source_name: str, frame_fn):
        self.status.source = source_name
        self._source_frame_fn = frame_fn

    def start(self):
        if not self._thread.is_alive():
            self._thread.start()
        self._enabled.set()
        self.status.running = True

    def stop(self):
        self._enabled.clear()
        self.status.running = False
        placeholder = self._make_placeholder()
        if placeholder is not None:
            with self._lock:
                self._latest_jpeg = placeholder
                self._jpeg_seq += 1
                self._cond.notify_all()

    def shutdown(self):
        self._stop.set()
        self._enabled.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def get_latest_jpeg_and_seq(self):
        with self._lock:
            return self._latest_jpeg, self._jpeg_seq

    def wait_for_new_frame(self, last_seq: int, timeout: float = 1.0) -> bool:
        with self._lock:
            if self._jpeg_seq != last_seq:
                return True
            self._cond.wait(timeout=timeout)
            return self._jpeg_seq != last_seq

    def status_dict(self) -> Dict[str, object]:
        return asdict(self.status)

    def _load_model(self):
        if self._model is not None:
            return
        self._model = YOLO(self.config.model_path)
        if torch is not None and torch.cuda.is_available() and self.config.device:
            try:
                self._model.to(self.config.device)
            except Exception:
                pass
        try:
            self._model.fuse()
        except Exception:
            pass

    def _update_fps(self, now: float):
        self._frame_times.append(now)
        if len(self._frame_times) > 30:
            self._frame_times.pop(0)
        if len(self._frame_times) >= 2:
            dt = self._frame_times[-1] - self._frame_times[0]
            if dt > 0:
                self.status.fps = (len(self._frame_times) - 1) / dt

    def _run(self):
        while not self._stop.is_set():
            if not self._enabled.is_set():
                time.sleep(0.05)
                continue

            if self._source_frame_fn is None:
                time.sleep(0.05)
                continue

            frame = self._source_frame_fn()
            if frame is None:
                time.sleep(0.01)
                continue

            self._load_model()

            now = time.time()
            self._update_fps(now)

            conf = self.config.conf
            iou = self.config.iou

            results = self._model.predict(
                frame,
                imgsz=self.config.imgsz,
                conf=conf,
                iou=iou,
                device=self.config.device,
                verbose=False,
            )

            best_conf = 0.0
            best_box = None

            if results and results[0].boxes is not None:
                for b in results[0].boxes:
                    if int(b.cls[0]) != BALL_CLASS_ID:
                        continue
                    score = float(b.conf[0])
                    if score > best_conf:
                        best_conf = score
                        best_box = tuple(map(int, b.xyxy[0]))

            h, w = frame.shape[:2]

            if best_box:
                x1, y1, x2, y2 = best_box
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                self._last_seen_ts = now
                if self._center is None:
                    self._center = (cx, cy)
                else:
                    smooth = self.config.smooth
                    self._center = (
                        self._center[0] * smooth + cx * (1.0 - smooth),
                        self._center[1] * smooth + cy * (1.0 - smooth),
                    )
                self.status.state = "tracking"
            else:
                if self._last_seen_ts is not None and (now - self._last_seen_ts) <= self.config.lost_hold_sec:
                    self.status.state = "hold"
                else:
                    self.status.state = "search"
                    if self._center is not None:
                        smooth = min(self.config.smooth + 0.1, 0.95)
                        self._center = (
                            self._center[0] * smooth + (w / 2.0) * (1.0 - smooth),
                            self._center[1] * smooth + (h / 2.0) * (1.0 - smooth),
                        )

            self.status.last_conf = best_conf
            if self._last_seen_ts is not None:
                self.status.last_seen_sec = now - self._last_seen_ts
            else:
                self.status.last_seen_sec = 0.0

            output = frame
            if best_box:
                x1, y1, x2, y2 = best_box
                cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    output,
                    f"BALL {best_conf:.2f}",
                    (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            zoom = max(self.config.zoom, 1.0)
            if self._center and zoom > 1.0:
                roi_w = int(w / zoom)
                roi_h = int(h / zoom)
                cx, cy = self._center
                x1 = int(max(0, min(cx - roi_w / 2, w - roi_w)))
                y1 = int(max(0, min(cy - roi_h / 2, h - roi_h)))
                x2 = x1 + roi_w
                y2 = y1 + roi_h
                roi = output[y1:y2, x1:x2]
                if roi.size > 0:
                    output = cv2.resize(roi, (w, h), interpolation=cv2.INTER_LINEAR)

            ok, jpeg = cv2.imencode(".jpg", output, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ok:
                with self._lock:
                    self._latest_jpeg = jpeg.tobytes()
                    self._jpeg_seq += 1
                    self._cond.notify_all()

            time.sleep(0.001)

    def _make_placeholder(self) -> Optional[bytes]:
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(
            frame,
            "AI idle",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )
        ok, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ok:
            return jpeg.tobytes()
        return None


def ai_mjpeg_stream(ai_worker: AiWorker):
    boundary = "frame"
    jpeg, seq = ai_worker.get_latest_jpeg_and_seq()

    while True:
        ai_worker.wait_for_new_frame(seq, timeout=1.0)
        jpeg, new_seq = ai_worker.get_latest_jpeg_and_seq()

        if new_seq == seq:
            continue
        seq = new_seq

        if jpeg is None:
            continue

        yield (
            b"--" + boundary.encode() + b"\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" +
            jpeg + b"\r\n"
        )
