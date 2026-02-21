import os
import time
import threading
import subprocess
import re
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple
from collections import deque

import cv2

from ballscope.config import QUALITY_PRESETS, DEFAULT_PRESET, MJPEG_JPEG_QUALITY, STREAM_MAX_FPS, PREFERRED_PIXFMT, V4L2_BUFFER_SIZE


def _is_linux() -> bool:
    try:
        return os.name == "posix" and os.uname().sysname.lower() == "linux"
    except Exception:
        return False


def _is_macos() -> bool:
    try:
        return os.name == "posix" and os.uname().sysname.lower() == "darwin"
    except Exception:
        return False


def resolve_dev_path(src: str) -> Optional[str]:
    if not _is_linux():
        return None
    if src.isdigit():
        return f"/dev/video{int(src)}"
    if src.startswith("/dev/video"):
        return src
    return None


def normalize_linux_source(src: str) -> Optional[str]:
    s = (src or "").strip()
    if s.isdigit():
        return s
    if s.startswith("/dev/video"):
        return s
    m = re.search(r"/dev/video(\d+)", s)
    if m:
        return f"/dev/video{int(m.group(1))}"
    m = re.fullmatch(r"video(\d+)", s, flags=re.IGNORECASE)
    if m:
        return f"/dev/video{int(m.group(1))}"
    return None


def run_v4l2(args: list[str]) -> Tuple[bool, str]:
    try:
        p = subprocess.run(
            ["v4l2-ctl", *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if p.returncode == 0:
            out = (p.stdout or "").strip()
            return True, out
        err = (p.stderr or p.stdout or "").strip()
        return False, err or f"v4l2-ctl failed rc={p.returncode}"
    except FileNotFoundError:
        return False, "v4l2-ctl not found (sudo apt install v4l-utils)"
    except Exception as exc:
        return False, f"v4l2-ctl error: {exc}"


def driver_set_format(dev: str, w: int, h: int, fps: int, pixfmt: str) -> Tuple[bool, str]:
    ok1, msg1 = run_v4l2(["-d", dev, f"--set-fmt-video=width={w},height={h},pixelformat={pixfmt}"])
    ok2, msg2 = run_v4l2(["-d", dev, f"--set-parm={fps}"])
    ok = ok1 and ok2
    msg = "; ".join([m for m in [msg1, msg2] if m]) or ("ok" if ok else "failed")
    return ok, msg


def driver_set_controls(dev: str, settings: "CameraSettings") -> Tuple[bool, str]:
    parts = []
    ok_all = True

    def set_ctrl(expr: str):
        nonlocal ok_all
        ok, msg = run_v4l2(["-d", dev, f"--set-ctrl={expr}"])
        ok_all = ok_all and ok
        if msg:
            parts.append(msg)

    if settings.brightness is not None:
        set_ctrl(f"brightness={int(settings.brightness)}")
    if settings.contrast is not None:
        set_ctrl(f"contrast={int(settings.contrast)}")
    if settings.saturation is not None:
        set_ctrl(f"saturation={int(settings.saturation)}")
    if settings.gain is not None:
        set_ctrl(f"gain={int(settings.gain)}")

    if settings.auto_wb is not None:
        if settings.auto_wb:
            set_ctrl("white_balance_temperature_auto=1")
        else:
            set_ctrl("white_balance_temperature_auto=0")

    if settings.auto_exposure is not None:
        if settings.auto_exposure:
            set_ctrl("exposure_auto=3")
        else:
            set_ctrl("exposure_auto=1")

    msg = "; ".join([m for m in parts if m]) or ("ok" if ok_all else "controls may be unsupported")
    return ok_all, msg


@dataclass
class CameraSettings:
    preset: str = DEFAULT_PRESET
    brightness: Optional[int] = None
    contrast: Optional[int] = None
    saturation: Optional[int] = None
    gain: Optional[int] = None
    auto_wb: Optional[bool] = True
    auto_exposure: Optional[bool] = True


@dataclass
class CameraState:
    name: str
    src: str
    settings: CameraSettings
    is_open: bool = False
    last_frame_ts: float = 0.0
    last_error: Optional[str] = None
    last_frame_shape: Optional[Tuple[int, int, int]] = None
    fps: float = 0.0


class CameraWorker:
    def __init__(self, state: CameraState):
        self.state = state
        self._cap: Optional[cv2.VideoCapture] = None

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

        self._latest_jpeg: Optional[bytes] = None
        self._latest_frame_bgr = None
        self._jpeg_seq = 0

        self._stop = threading.Event()
        self._reopen = threading.Event()
        self._suspend = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

        self._last_preset_key: Optional[str] = None
        self._frame_times = deque(maxlen=30)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._reopen.set()
        self._suspend.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        with self._lock:
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception:
                    pass
            self._cap = None
            self._latest_jpeg = None
            self._latest_frame_bgr = None
            self._jpeg_seq += 1
            self._cond.notify_all()

    def request_reopen(self):
        self._reopen.set()
        with self._lock:
            self._cond.notify_all()

    def suspend(self):
        self._suspend.set()
        self.request_reopen()

    def resume(self):
        if self._suspend.is_set():
            self._suspend.clear()
            self.request_reopen()

    def get_latest_jpeg_and_seq(self) -> Tuple[Optional[bytes], int]:
        with self._lock:
            return self._latest_jpeg, self._jpeg_seq

    def wait_for_new_frame(self, last_seq: int, timeout: float = 1.0) -> bool:
        with self._lock:
            if self._jpeg_seq != last_seq:
                return True
            self._cond.wait(timeout=timeout)
            return self._jpeg_seq != last_seq

    def get_latest_frame_bgr(self):
        with self._lock:
            return self._latest_frame_bgr

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        src = self.state.src
        preset = QUALITY_PRESETS.get(self.state.settings.preset, QUALITY_PRESETS[DEFAULT_PRESET])
        w, h, fps = preset["w"], preset["h"], preset["fps"]

        if _is_linux():
            normalized = normalize_linux_source(src)
            if normalized is None:
                self.state.last_error = (
                    f"Invalid Linux camera source: '{src}'. Use '/dev/videoX' or numeric index like '0'."
                )
                return None
            if normalized != src:
                self.state.src = normalized
                src = normalized

        dev = resolve_dev_path(src)
        if dev is not None:
            ok, msg = driver_set_format(dev, w, h, fps, PREFERRED_PIXFMT)
            if not ok:
                self.state.last_error = f"Driver fmt warn: {msg}"
            else:
                self.state.last_error = None

            okc, msgc = driver_set_controls(dev, self.state.settings)
            if not okc:
                self.state.last_error = (
                    (self.state.last_error + " | " if self.state.last_error else "") + f"Ctrl warn: {msgc}"
                )

        src_to_open = src
        if not _is_linux() and src.startswith("/dev/video"):
            idx = src.replace("/dev/video", "", 1)
            if idx.isdigit():
                src_to_open = idx

        if _is_linux():
            backend = cv2.CAP_V4L2
        elif _is_macos() and hasattr(cv2, "CAP_AVFOUNDATION"):
            backend = cv2.CAP_AVFOUNDATION
        else:
            backend = 0
        try_sources = [src_to_open]
        if _is_macos() and src_to_open.isdigit() and src_to_open != "0":
            try_sources.append("0")

        cap = None
        open_err = None
        used_src = src_to_open
        for candidate in try_sources:
            try:
                cur = cv2.VideoCapture(int(candidate), backend) if candidate.isdigit() else cv2.VideoCapture(candidate, backend)
            except Exception as exc:
                open_err = str(exc)
                continue
            if cur is not None and cur.isOpened():
                cap = cur
                used_src = candidate
                break
            try:
                if cur is not None:
                    cur.release()
            except Exception:
                pass

        if cap is None:
            if open_err:
                self.state.last_error = f"Open error: {open_err}"
            else:
                self.state.last_error = f"Could not open camera source: {src_to_open}"
            return None
        self.state.src = used_src

        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, float(V4L2_BUFFER_SIZE))
        except Exception:
            pass

        return cap

    def _safe_close_cap(self):
        with self._lock:
            cap = self._cap
            self._cap = None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

    def _update_fps(self, now: float):
        self._frame_times.append(now)
        if len(self._frame_times) >= 2:
            dt = self._frame_times[-1] - self._frame_times[0]
            if dt > 0:
                self.state.fps = (len(self._frame_times) - 1) / dt

    def _run(self):
        try:
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        except Exception:
            pass

        while not self._stop.is_set():
            if self._suspend.is_set():
                self.state.is_open = False
                self._safe_close_cap()
                time.sleep(0.1)
                continue

            if self._reopen.is_set():
                self.state.is_open = False
                self._safe_close_cap()
                self._reopen.clear()
                time.sleep(0.2)

            if self._cap is None:
                cap = self._open_capture()
                if cap is None:
                    self.state.is_open = False
                    # Avoid terminal spam on unavailable sources; keep retry loop responsive enough.
                    time.sleep(1.0)
                    continue
                with self._lock:
                    self._cap = cap
                self.state.is_open = True
                self._last_preset_key = self.state.settings.preset

            if self._last_preset_key != self.state.settings.preset:
                self._last_preset_key = self.state.settings.preset
                self.request_reopen()
                continue

            with self._lock:
                cap = self._cap

            ok, frame = False, None
            try:
                ok, frame = cap.read()
            except Exception as exc:
                self.state.last_error = f"Read error: {exc}"
                ok = False

            if not ok or frame is None:
                self.state.last_error = self.state.last_error or "Frame read failed."
                self.state.is_open = False
                self._safe_close_cap()
                time.sleep(0.15)
                continue

            now = time.time()
            self._update_fps(now)

            with self._lock:
                self._latest_frame_bgr = frame
                self.state.last_frame_shape = frame.shape

            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), MJPEG_JPEG_QUALITY]
            ok2, jpeg = cv2.imencode(".jpg", frame, encode_params)
            if ok2:
                with self._lock:
                    self._latest_jpeg = jpeg.tobytes()
                    self._jpeg_seq += 1
                    self._cond.notify_all()
                self.state.last_frame_ts = now
                self.state.last_error = None

            time.sleep(0.001)

    def apply_driver_controls_now(self):
        dev = resolve_dev_path(self.state.src)
        if dev is None:
            return
        ok, msg = driver_set_controls(dev, self.state.settings)
        if not ok:
            self.state.last_error = (
                (self.state.last_error + " | " if self.state.last_error else "") + f"Ctrl warn: {msg}"
            )


def mjpeg_stream(worker: CameraWorker):
    boundary = "frame"
    min_dt = (1.0 / STREAM_MAX_FPS) if STREAM_MAX_FPS and STREAM_MAX_FPS > 0 else 0.0
    last_sent = 0.0

    jpeg, seq = worker.get_latest_jpeg_and_seq()

    while True:
        worker.wait_for_new_frame(seq, timeout=1.0)
        jpeg, new_seq = worker.get_latest_jpeg_and_seq()

        if new_seq == seq:
            continue
        seq = new_seq

        if jpeg is None:
            continue

        if min_dt > 0:
            now = time.time()
            dt = now - last_sent
            if dt < min_dt:
                time.sleep(min_dt - dt)
            last_sent = time.time()

        yield (
            b"--" + boundary.encode() + b"\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" +
            jpeg + b"\r\n"
        )


def camera_state_dict(states: Dict[str, CameraState]) -> Dict[str, dict]:
    return {cid: {**asdict(state), "settings": asdict(state.settings)} for cid, state in states.items()}
