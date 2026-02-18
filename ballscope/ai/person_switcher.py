import threading
import time
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Callable, Tuple

import cv2
import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from ultralytics import YOLO

from ballscope.config import AiConfig


@dataclass
class AiStatus:
    running: bool = False
    fps: float = 0.0
    last_conf: float = 0.0
    last_seen_sec: float = 0.0
    state: str = "idle"
    active_camera: str = "camL"
    mode: str = "auto"
    manual_camera: Optional[str] = None


class PersonSwitcherWorker:
    def __init__(self, config: AiConfig):
        self.config = config
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._latest_jpeg: Optional[bytes] = None
        self._latest_frame = None
        self._jpeg_seq = 0
        self._frame_idx = 0

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._stop = threading.Event()
        self._enabled = threading.Event()

        self._model: Optional[YOLO] = None
        self._model_path_loaded: Optional[str] = None
        self._source_frame_fns: Dict[str, Callable[[], Optional[np.ndarray]]] = {}

        self._centers: Dict[str, Tuple[float, float]] = {}
        self._last_seen_ts: Dict[str, float] = {}
        self._last_dets: Dict[str, dict] = {}
        self.status = AiStatus()
        self._frame_times = []
        self._recorder = None

    def attach_sources(self, sources: Dict[str, Callable[[], Optional[np.ndarray]]]):
        self._source_frame_fns = sources

    def set_manual_camera(self, camera_id: Optional[str]):
        self.status.manual_camera = camera_id
        if camera_id:
            self.status.mode = "manual"
        else:
            self.status.mode = "auto"

    def set_recorder(self, recorder):
        self._recorder = recorder

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

    def get_latest_frame(self):
        with self._lock:
            return self._latest_frame

    def wait_for_new_frame(self, last_seq: int, timeout: float = 1.0) -> bool:
        with self._lock:
            if self._jpeg_seq != last_seq:
                return True
            self._cond.wait(timeout=timeout)
            return self._jpeg_seq != last_seq

    def status_dict(self) -> Dict[str, object]:
        return asdict(self.status)

    def _load_model(self):
        if self._model is not None and self._model_path_loaded == self.config.model_path:
            return
        self._model = YOLO(self.config.model_path)
        self._model_path_loaded = self.config.model_path
        device = self._resolve_device()
        if torch is not None and device != "cpu":
            try:
                self._model.to(device)
            except Exception:
                pass
        try:
            self._model.fuse()
        except Exception:
            pass

    def _resolve_device(self) -> str:
        dev = (self.config.device or "").strip().lower()
        if dev in ("", "auto"):
            if torch is not None and torch.cuda.is_available():
                return "cuda"
            return "cpu"
        if dev.startswith("cuda") or dev.isdigit():
            if torch is not None and torch.cuda.is_available():
                return dev
            raise RuntimeError("CUDA requested but not available. Install CUDA-enabled PyTorch or set BALLSCOPE_AI_DEVICE=cpu.")
        return dev

    def _update_fps(self, now: float):
        self._frame_times.append(now)
        if len(self._frame_times) > 30:
            self._frame_times.pop(0)
        if len(self._frame_times) >= 2:
            dt = self._frame_times[-1] - self._frame_times[0]
            if dt > 0:
                self.status.fps = (len(self._frame_times) - 1) / dt

    def _detect_best_ball(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        imgsz = int(self.config.imgsz)
        small = cv2.resize(frame, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        device = self._resolve_device()
        results = self._model.predict(
            small,
            imgsz=imgsz,
            conf=self.config.conf,
            iou=self.config.iou,
            device=device,
            verbose=False,
        )
        best_conf = 0.0
        best_box = None
        best_score = 0.0

        if results and results[0].boxes is not None:
            for b in results[0].boxes:
                if int(b.cls[0]) != int(self.config.class_id):
                    continue
                score = float(b.conf[0])
                if score < self.config.conf:
                    continue
                x1s, y1s, x2s, y2s = map(float, b.xyxy[0])
                x1 = int(x1s * (w / imgsz))
                y1 = int(y1s * (h / imgsz))
                x2 = int(x2s * (w / imgsz))
                y2 = int(y2s * (h / imgsz))
                area = max(0, x2 - x1) * max(0, y2 - y1)
                norm_area = area / float(max(1, w * h))
                cam_score = score * norm_area
                if cam_score > best_score:
                    best_score = cam_score
                    best_conf = score
                    best_box = (x1, y1, x2, y2)

        return best_conf, best_box, best_score

    def _select_active_camera(self, candidates: Dict[str, dict], now: float) -> str:
        manual = self.status.manual_camera
        if manual and manual in candidates:
            return manual

        best_cam = None
        best_score = 0.0
        for cid, info in candidates.items():
            if info["score"] > best_score:
                best_score = info["score"]
                best_cam = cid

        if best_cam is not None and best_score > 0:
            return best_cam

        last_active = self.status.active_camera
        last_seen = self._last_seen_ts.get(last_active)
        if last_seen is not None and (now - last_seen) <= self.config.lost_hold_sec:
            return last_active

        return next(iter(candidates.keys()))

    def _run(self):
        while not self._stop.is_set():
            if not self._enabled.is_set():
                time.sleep(0.05)
                continue

            if not self._source_frame_fns:
                time.sleep(0.05)
                continue

            frames: Dict[str, np.ndarray] = {}
            for cid, fn in self._source_frame_fns.items():
                frame = fn()
                if frame is not None:
                    frames[cid] = frame

            if not frames:
                time.sleep(0.01)
                continue

            self._load_model()

            now = time.time()
            self._update_fps(now)

            candidates: Dict[str, dict] = {}
            for cid, frame in frames.items():
                detect_every = max(1, int(self.config.detect_every))
                do_detect = (self._frame_idx % detect_every) == 0
                if do_detect:
                    best_conf, best_box, best_score = self._detect_best_ball(frame)
                    self._last_dets[cid] = {
                        "conf": best_conf,
                        "box": best_box,
                        "score": best_score,
                    }
                else:
                    last = self._last_dets.get(cid, {})
                    best_conf = last.get("conf", 0.0)
                    best_box = last.get("box")
                    best_score = last.get("score", 0.0)
                candidates[cid] = {
                    "conf": best_conf,
                    "box": best_box,
                    "score": best_score,
                    "frame": frame,
                }
                if best_box:
                    x1, y1, x2, y2 = best_box
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    prev = self._centers.get(cid)
                    if prev:
                        smooth = self.config.smooth
                        cx = prev[0] * smooth + cx * (1.0 - smooth)
                        cy = prev[1] * smooth + cy * (1.0 - smooth)
                    self._centers[cid] = (cx, cy)
                    self._last_seen_ts[cid] = now

            active_cam = self._select_active_camera(candidates, now)
            self.status.active_camera = active_cam

            active_info = candidates.get(active_cam)
            if active_info is None:
                time.sleep(0.01)
                continue

            frame = active_info["frame"]
            h, w = frame.shape[:2]
            center = self._centers.get(active_cam)
            last_seen = self._last_seen_ts.get(active_cam)

            if active_info["box"]:
                self.status.state = "tracking"
                self.status.last_conf = active_info["conf"]
                self.status.last_seen_sec = 0.0
            else:
                if last_seen is not None and (now - last_seen) <= self.config.lost_hold_sec:
                    self.status.state = "hold"
                    self.status.last_seen_sec = now - last_seen
                else:
                    self.status.state = "search"
                    self.status.last_seen_sec = 0.0
                    center = None
                self.status.last_conf = active_info["conf"]

            if center is None:
                center = (w / 2.0, h / 2.0)

            zoom = max(self.config.zoom, 1.0)
            output = frame
            if zoom > 1.0:
                roi_w = int(w / zoom)
                roi_h = int(h / zoom)
                cx, cy = center
                x1 = int(max(0, min(cx - roi_w / 2, w - roi_w)))
                y1 = int(max(0, min(cy - roi_h / 2, h - roi_h)))
                x2 = x1 + roi_w
                y2 = y1 + roi_h
                roi = output[y1:y2, x1:x2]
                if roi.size > 0:
                    output = cv2.resize(roi, (self.config.output_w, self.config.output_h), interpolation=cv2.INTER_LINEAR)
            else:
                if (w, h) != (self.config.output_w, self.config.output_h):
                    output = cv2.resize(frame, (self.config.output_w, self.config.output_h), interpolation=cv2.INTER_LINEAR)

            record_frame = output
            if not self.config.record_use_zoom:
                if (w, h) != (self.config.output_w, self.config.output_h):
                    record_frame = cv2.resize(frame, (self.config.output_w, self.config.output_h), interpolation=cv2.INTER_LINEAR)
                else:
                    record_frame = frame

            with self._lock:
                self._latest_frame = output

            ok, jpeg = cv2.imencode(".jpg", output, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ok:
                with self._lock:
                    self._latest_jpeg = jpeg.tobytes()
                    self._jpeg_seq += 1
                    self._cond.notify_all()

            if self._recorder is not None and self._recorder.is_running():
                self._recorder.write(record_frame)

            self._frame_idx += 1
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


def ai_mjpeg_stream(ai_worker: PersonSwitcherWorker):
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
