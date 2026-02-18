import os
import time
import threading
import subprocess
import queue
import sys
import site
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple
from contextlib import asynccontextmanager

import cv2
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates

# -----------------------------
# Config / Quality presets
# -----------------------------

QUALITY_PRESETS: Dict[str, Dict[str, int]] = {
    "2160p20": {"w": 3840, "h": 2160, "fps": 20},
    "1080p60": {"w": 1920, "h": 1080, "fps": 60},
    "720p120": {"w": 1280, "h": 720, "fps": 120},
    "1520p50": {"w": 2016, "h": 1520, "fps": 50},
    "3040p10": {"w": 4056, "h": 3040, "fps": 10},
}
DEFAULT_PRESET = os.getenv("BALLSCOPE_DEFAULT_PRESET", "1080p60")

# On your Jetson with 2x Arducam B0459:
# Cam 1: /dev/video0 + /dev/video1
# Cam 2: /dev/video2 + /dev/video3
# Use the FIRST node per camera for capture:
# Fixed defaults (no manual env exports needed)
DEFAULT_CAM_A = "/dev/video0"
DEFAULT_CAM_B = "/dev/video2"

CAM_A_SRC = DEFAULT_CAM_A
CAM_B_SRC = DEFAULT_CAM_B

# MJPEG stream JPEG quality (0-100)
MJPEG_JPEG_QUALITY = int(os.getenv("BALLSCOPE_MJPEG_JPEG_QUALITY", "85"))

# Prefer MJPEG from UVC cams (usually fastest on Jetson)
PREFERRED_PIXFMT = "MJPG"  # MJPG or YUYV

# Low-latency buffer (best effort)
V4L2_BUFFER_SIZE = 1

# Optional: limit outgoing stream fps (0 = unlimited)
STREAM_MAX_FPS = float(os.getenv("BALLSCOPE_STREAM_MAX_FPS", "0"))

# -----------------------------
# Audio (USB mic) - pure Python
# -----------------------------

# PortAudio uses device names/indices (not ALSA hw: strings).
# Use "auto" to pick the USB mic by name.
MIC_DEVICE = "auto"
MIC_SAMPLE_RATE = 48000
MIC_CHANNELS = 2
MIC_BLOCKSIZE = 480  # 10ms @ 48kHz

MIC_DEVICE_HINTS = ["comica", "vm10", "usb audio"]

# -----------------------------
# Data model
# -----------------------------

@dataclass
class CameraSettings:
    preset: str = DEFAULT_PRESET
    brightness: Optional[int] = None
    contrast: Optional[int] = None
    saturation: Optional[int] = None
    gain: Optional[int] = None
    auto_wb: Optional[bool] = None
    auto_exposure: Optional[bool] = None

@dataclass
class CameraState:
    name: str
    src: str
    settings: CameraSettings
    is_open: bool = False
    last_frame_ts: float = 0.0
    last_error: Optional[str] = None
    # AI-ready info
    last_frame_shape: Optional[Tuple[int, int, int]] = None


# -----------------------------
# Driver / V4L2 helpers
# -----------------------------

def _is_linux() -> bool:
    try:
        return os.name == "posix" and os.uname().sysname.lower() == "linux"
    except Exception:
        return False

def resolve_dev_path(src: str) -> Optional[str]:
    """
    Convert "0" -> "/dev/video0" on Linux, or pass through "/dev/videoX".
    If not Linux, returns None (driver controls not supported).
    """
    if not _is_linux():
        return None
    if src.isdigit():
        return f"/dev/video{int(src)}"
    if src.startswith("/dev/video"):
        return src
    return None

def run_v4l2(args: list[str]) -> Tuple[bool, str]:
    """
    Run v4l2-ctl safely. Returns (ok, message).
    Never throws (we don't want crashes because a control isn't supported).
    """
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
    except Exception as e:
        return False, f"v4l2-ctl error: {e}"

def driver_set_format(dev: str, w: int, h: int, fps: int, pixfmt: str) -> Tuple[bool, str]:
    """
    Set format + fps via driver. Best-effort.
    This is the stable way to change res/fps: apply -> reopen capture.
    """
    # Set pixel format + size
    ok1, msg1 = run_v4l2(["-d", dev, f"--set-fmt-video=width={w},height={h},pixelformat={pixfmt}"])
    # Set fps
    ok2, msg2 = run_v4l2(["-d", dev, f"--set-parm={fps}"])
    ok = ok1 and ok2
    msg = "; ".join([m for m in [msg1, msg2] if m]) or ("ok" if ok else "failed")
    return ok, msg

def driver_set_controls(dev: str, settings: CameraSettings) -> Tuple[bool, str]:
    """
    Apply UVC controls via driver (v4l2-ctl). Best-effort.
    Exposure/WB control names vary by device; we attempt common ones.
    """
    parts = []
    ok_all = True

    def set_ctrl(expr: str):
        nonlocal ok_all
        ok, msg = run_v4l2(["-d", dev, f"--set-ctrl={expr}"])
        ok_all = ok_all and ok
        if msg:
            parts.append(msg)

    # Common numeric controls (often supported)
    if settings.brightness is not None:
        set_ctrl(f"brightness={int(settings.brightness)}")
    if settings.contrast is not None:
        set_ctrl(f"contrast={int(settings.contrast)}")
    if settings.saturation is not None:
        set_ctrl(f"saturation={int(settings.saturation)}")
    if settings.gain is not None:
        set_ctrl(f"gain={int(settings.gain)}")

    # Auto WB: very common UVC control name
    if settings.auto_wb is not None:
        # try common names
        if settings.auto_wb:
            set_ctrl("white_balance_temperature_auto=1")
        else:
            set_ctrl("white_balance_temperature_auto=0")

    # Auto exposure: UVC usually uses exposure_auto values:
    # 1=manual, 3=aperture priority (auto) (varies).
    if settings.auto_exposure is not None:
        if settings.auto_exposure:
            set_ctrl("exposure_auto=3")
        else:
            set_ctrl("exposure_auto=1")

    msg = "; ".join([m for m in parts if m]) or ("ok" if ok_all else "controls may be unsupported")
    return ok_all, msg


# -----------------------------
# Camera worker (threaded grab)
# -----------------------------

class CameraWorker:
    """
    - Reads frames in a background thread.
    - Keeps latest JPEG for MJPEG streaming.
    - Keeps latest BGR frame for AI (YOLO) later.
    - Preset change (res/fps/pixfmt) is handled by a SAFE REOPEN to avoid segfault.
    - Brightness/contrast/etc. are applied via v4l2-ctl driver controls (safe while streaming).
    """

    def __init__(self, state: CameraState):
        self.state = state

        self._cap: Optional[cv2.VideoCapture] = None

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

        self._latest_jpeg: Optional[bytes] = None
        self._latest_frame_bgr = None  # numpy array
        self._jpeg_seq = 0

        self._stop = threading.Event()
        self._reopen = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

        # Track the last applied preset key to detect changes
        self._last_preset_key: Optional[str] = None

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._reopen.set()
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
        """AI-ready: returns latest BGR frame (copy-free reference). Keep usage read-only."""
        with self._lock:
            return self._latest_frame_bgr

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        src = self.state.src

        # Determine preset
        preset = QUALITY_PRESETS.get(self.state.settings.preset, QUALITY_PRESETS[DEFAULT_PRESET])
        w, h, fps = preset["w"], preset["h"], preset["fps"]

        # If on Linux and device path available, set driver format BEFORE opening OpenCV
        dev = resolve_dev_path(src)
        if dev is not None:
            ok, msg = driver_set_format(dev, w, h, fps, PREFERRED_PIXFMT)
            if not ok:
                # Don't fail hard; some cameras refuse certain combos.
                # We'll still try to open; state.last_error will show driver msg.
                self.state.last_error = f"Driver fmt warn: {msg}"
            else:
                self.state.last_error = None

            # Apply driver controls too (safe anytime)
            okc, msgc = driver_set_controls(dev, self.state.settings)
            if not okc:
                # Not fatal
                self.state.last_error = (self.state.last_error + " | " if self.state.last_error else "") + f"Ctrl warn: {msgc}"

        # Open with V4L2 backend on Linux for stability
        backend = cv2.CAP_V4L2 if _is_linux() else 0

        try:
            if src.isdigit():
                cap = cv2.VideoCapture(int(src), backend)
            else:
                cap = cv2.VideoCapture(src, backend)
        except Exception as e:
            self.state.last_error = f"Open error: {e}"
            return None

        if cap is None or not cap.isOpened():
            self.state.last_error = f"Could not open camera source: {src}"
            return None

        # Low latency buffers (best-effort)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, float(V4L2_BUFFER_SIZE))
        except Exception:
            pass

        # IMPORTANT:
        # Do NOT set width/height/fps/fourcc via cap.set while reading (Jetson/UVC can segfault).
        # Format is already set via driver_set_format (above). If ignored, camera will run its mode.

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

    def _run(self):
        # Reduce OpenCV logging spam (safe)
        try:
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        except Exception:
            pass

        while not self._stop.is_set():

            # SAFE REOPEN request (used for preset/res/fps changes)
            if self._reopen.is_set():
                self.state.is_open = False
                self._safe_close_cap()
                self._reopen.clear()
                # Small pause to let driver settle
                time.sleep(0.2)

            # Ensure capture open
            if self._cap is None:
                cap = self._open_capture()
                if cap is None:
                    self.state.is_open = False
                    time.sleep(0.25)
                    continue
                with self._lock:
                    self._cap = cap
                self.state.is_open = True
                self._last_preset_key = self.state.settings.preset

            # Detect preset change (res/fps/pixfmt) -> request safe reopen
            if self._last_preset_key != self.state.settings.preset:
                self._last_preset_key = self.state.settings.preset
                self.request_reopen()
                continue

            # Read frame
            with self._lock:
                cap = self._cap

            ok, frame = False, None
            try:
                ok, frame = cap.read()
            except Exception as e:
                self.state.last_error = f"Read error: {e}"
                ok = False

            if not ok or frame is None:
                self.state.last_error = self.state.last_error or "Frame read failed."
                self.state.is_open = False
                self._safe_close_cap()
                time.sleep(0.15)
                continue

            now = time.time()

            # Store latest BGR frame for AI
            with self._lock:
                self._latest_frame_bgr = frame
                self.state.last_frame_shape = frame.shape

            # Encode JPEG for MJPEG
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), MJPEG_JPEG_QUALITY]
            ok2, jpeg = cv2.imencode(".jpg", frame, encode_params)
            if ok2:
                with self._lock:
                    self._latest_jpeg = jpeg.tobytes()
                    self._jpeg_seq += 1
                    self._cond.notify_all()
                self.state.last_frame_ts = now
                self.state.last_error = None

            # Tiny sleep so we don’t max CPU if fps is ignored
            time.sleep(0.001)

    def apply_driver_controls_now(self):
        """Apply brightness/contrast/etc via driver. Safe while streaming."""
        dev = resolve_dev_path(self.state.src)
        if dev is None:
            return
        ok, msg = driver_set_controls(dev, self.state.settings)
        if not ok:
            self.state.last_error = (self.state.last_error + " | " if self.state.last_error else "") + f"Ctrl warn: {msg}"


# -----------------------------
# FastAPI app
# -----------------------------

templates = Jinja2Templates(directory="templates")

cam_states: Dict[str, CameraState] = {
    "camA": CameraState(name="Webcam A", src=CAM_A_SRC, settings=CameraSettings()),
    "camB": CameraState(name="Webcam B", src=CAM_B_SRC, settings=CameraSettings()),
}

workers: Dict[str, CameraWorker] = {
    cid: CameraWorker(state) for cid, state in cam_states.items()
}

MAIN_SOURCE = os.getenv("BALLSCOPE_MAIN_SOURCE", "camA")


@asynccontextmanager
async def lifespan(app: FastAPI):
    for w in workers.values():
        w.start()
    yield
    for w in workers.values():
        w.stop()

app = FastAPI(title="BallScope – Live Preview", lifespan=lifespan)

# -----------------------------
# MJPEG generator
# -----------------------------

def mjpeg_stream(camera_id: str):
    boundary = "frame"
    min_dt = (1.0 / STREAM_MAX_FPS) if STREAM_MAX_FPS and STREAM_MAX_FPS > 0 else 0.0
    last_sent = 0.0

    jpeg, seq = workers[camera_id].get_latest_jpeg_and_seq()

    while True:
        workers[camera_id].wait_for_new_frame(seq, timeout=1.0)
        jpeg, new_seq = workers[camera_id].get_latest_jpeg_and_seq()

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

def _wav_header(sample_rate: int, channels: int, bits_per_sample: int) -> bytes:
    byte_rate = sample_rate * channels * (bits_per_sample // 8)
    block_align = channels * (bits_per_sample // 8)
    # Use 0xFFFFFFFF for streaming (unknown length)
    riff_size = 0xFFFFFFFF
    data_size = 0xFFFFFFFF
    return (
        b"RIFF" +
        riff_size.to_bytes(4, "little") +
        b"WAVE" +
        b"fmt " +
        (16).to_bytes(4, "little") +
        (1).to_bytes(2, "little") +  # PCM
        channels.to_bytes(2, "little") +
        sample_rate.to_bytes(4, "little") +
        byte_rate.to_bytes(4, "little") +
        block_align.to_bytes(2, "little") +
        bits_per_sample.to_bytes(2, "little") +
        b"data" +
        data_size.to_bytes(4, "little")
    )

def _import_sounddevice():
    try:
        import sounddevice as sd  # type: ignore
        return sd
    except Exception:
        # venv may not include user-site; try to add it
        try:
            user_site = site.getusersitepackages()
            if user_site and user_site not in sys.path:
                sys.path.append(user_site)
            import sounddevice as sd  # type: ignore
            return sd
        except Exception:
            return None

def _resolve_audio_device(sd):
    if isinstance(MIC_DEVICE, int):
        return MIC_DEVICE
    if isinstance(MIC_DEVICE, str) and MIC_DEVICE.isdigit():
        return int(MIC_DEVICE)
    if isinstance(MIC_DEVICE, str) and MIC_DEVICE.lower() != "auto":
        # Try to match by substring
        needle = MIC_DEVICE.lower()
        for idx, dev in enumerate(sd.query_devices()):
            if dev.get("max_input_channels", 0) > 0 and needle in dev.get("name", "").lower():
                return idx
        return MIC_DEVICE  # let PortAudio try it as a name

    # Auto-pick a USB mic by name
    for idx, dev in enumerate(sd.query_devices()):
        name = dev.get("name", "").lower()
        if dev.get("max_input_channels", 0) > 0 and any(h in name for h in MIC_DEVICE_HINTS):
            return idx
    return None  # default input device

def audio_stream():
    sd = _import_sounddevice()
    if sd is None:
        return

    device = _resolve_audio_device(sd)

    q: "queue.Queue[bytes]" = queue.Queue(maxsize=8)

    def callback(indata, frames, time_info, status):
        if status:
            return
        try:
            q.put_nowait(bytes(indata))
        except queue.Full:
            # Drop to keep latency low
            pass

    stream = sd.RawInputStream(
        samplerate=MIC_SAMPLE_RATE,
        channels=MIC_CHANNELS,
        dtype="int16",
        blocksize=MIC_BLOCKSIZE,
        device=device,
        latency="low",
        callback=callback,
    )

    stream.start()
    yield _wav_header(MIC_SAMPLE_RATE, MIC_CHANNELS, 16)

    try:
        while True:
            chunk = q.get()
            yield chunk
    except GeneratorExit:
        pass
    finally:
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass

# -----------------------------
# Routes
# -----------------------------

@app.get("/video/{camera_id}.mjpg")
def video_mjpg(camera_id: str):
    if camera_id not in workers:
        raise HTTPException(status_code=404, detail="Unknown camera_id")
    return StreamingResponse(
        mjpeg_stream(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/audio/mic")
def audio_mic():
    sd = _import_sounddevice()
    if sd is None:
        raise HTTPException(status_code=500, detail="sounddevice not installed or not visible in venv")
    return StreamingResponse(
        audio_stream(),
        media_type="audio/wav"
    )

@app.get("/api/state")
def api_state():
    return JSONResponse({
        "main_source": MAIN_SOURCE,
        "cameras": {cid: asdict(state) for cid, state in cam_states.items()},
        "presets": QUALITY_PRESETS,
        "pixfmt": PREFERRED_PIXFMT,
        "v4l2_buffer": V4L2_BUFFER_SIZE,
        "mjpeg_quality": MJPEG_JPEG_QUALITY,
        "stream_max_fps": STREAM_MAX_FPS,
    })

@app.post("/api/main/{camera_id}")
def set_main(camera_id: str):
    global MAIN_SOURCE
    if camera_id not in cam_states:
        raise HTTPException(status_code=404, detail="Unknown camera_id")
    MAIN_SOURCE = camera_id
    return {"ok": True, "main_source": MAIN_SOURCE}

@app.post("/api/settings/{camera_id}")
async def set_settings(camera_id: str, request: Request):
    if camera_id not in cam_states:
        raise HTTPException(status_code=404, detail="Unknown camera_id")
    body = await request.json()

    st = cam_states[camera_id].settings

    # Track if preset changed (requires safe reopen)
    preset_changed = False

    preset = body.get("preset")
    if preset is not None:
        if preset not in QUALITY_PRESETS:
            raise HTTPException(status_code=400, detail="Unknown preset")
        if st.preset != preset:
            st.preset = preset
            preset_changed = True

    # numeric controls
    for key in ["brightness", "contrast", "saturation", "gain"]:
        if key in body:
            val = body[key]
            if val is None:
                setattr(st, key, None)
            else:
                try:
                    setattr(st, key, int(val))
                except Exception:
                    raise HTTPException(status_code=400, detail=f"Invalid {key}")

    # boolean controls
    for key in ["auto_wb", "auto_exposure"]:
        if key in body:
            val = body[key]
            if val is None:
                setattr(st, key, None)
            else:
                setattr(st, key, bool(val))

    # Apply changes safely:
    # - preset (res/fps) => REOPEN (prevents segfault)
    # - controls => apply via driver immediately
    if preset_changed:
        workers[camera_id].request_reopen()

    # Controls can be applied anytime safely via v4l2-ctl
    workers[camera_id].apply_driver_controls_now()

    return {"ok": True, "camera_id": camera_id, "settings": asdict(st)}

# -----------------------------
# UI (single-file HTML)
# -----------------------------

UI_HTML = r"""
<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>BallScope – Live Preview</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body { background: #0b0f17; color: #e9eefb; }
    .card { background: #0f1626; border: 1px solid rgba(255,255,255,0.08); border-radius: 16px; }
    .soft { color: rgba(233,238,251,0.75); }
    .player-wrap { position: relative; width: 100%; aspect-ratio: 16/9; border-radius: 18px; overflow: hidden; background: #060912; }
    .player-wrap img { width: 100%; height: 100%; object-fit: cover; display:block; }
    .overlay-top { position:absolute; left:12px; top:10px; right:12px; display:flex; align-items:center; justify-content:space-between; gap:8px; }
    .pill { padding: 6px 10px; border-radius: 999px; background: rgba(255,255,255,0.08); font-size: 12px; }
    .overlay-bottom { position:absolute; left:12px; bottom:10px; right:12px; display:flex; align-items:center; justify-content:space-between; gap:8px; }
    .mini-wrap { width: 100%; aspect-ratio: 16/9; border-radius: 14px; overflow: hidden; background: #060912; border: 1px solid rgba(255,255,255,0.08); }
    .mini-wrap img { width:100%; height:100%; object-fit:cover; display:block; }
    .btn-ghost { background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.12); color: #e9eefb; }
    .btn-ghost:hover { background: rgba(255,255,255,0.14); }
    .select-dark, .form-control, .form-select { background:#0b1222; color:#e9eefb; border: 1px solid rgba(255,255,255,0.12); }
    .form-control:focus, .form-select:focus { box-shadow:none; border-color: rgba(255,255,255,0.22); }
    .smalllabel { font-size: 12px; color: rgba(233,238,251,0.7); }
    .badge-dot { width:10px; height:10px; border-radius:999px; display:inline-block; margin-right:8px; }
  </style>
</head>
<body>
  <div class="container py-4">
    <div class="d-flex align-items-center justify-content-between mb-3">
      <div>
        <h3 class="mb-0">BallScope</h3>
        <div class="soft">Live Preview (ohne AI) – Main Stream + 2 Webcams</div>
      </div>
      <div class="d-flex gap-2">
        <button class="btn btn-ghost" onclick="refreshState()">Refresh</button>
        <button class="btn btn-ghost" data-bs-toggle="modal" data-bs-target="#settingsModal">Settings</button>
      </div>
    </div>

    <div class="row g-3">
      <!-- Main player -->
      <div class="col-12 col-lg-8">
        <div class="card p-3 h-100">
          <div class="d-flex align-items-center justify-content-between mb-2">
            <div>
              <div class="smalllabel">MAIN STREAM</div>
              <div id="mainTitle" class="fw-semibold">–</div>
            </div>
            <div class="d-flex gap-2">
              <select id="mainSelect" class="form-select select-dark" style="min-width: 170px;" onchange="setMain(this.value)"></select>
            </div>
          </div>

          <div class="player-wrap">
            <img id="mainImg" src="" alt="Main Stream"/>
            <div class="overlay-top">
              <div class="pill" id="mainPresetPill">Preset: –</div>
              <div class="pill" id="mainStatusPill">Status: –</div>
            </div>
            <div class="overlay-bottom">
              <div class="pill">BallScope Preview (Zoom/Tracking kommt später)</div>
              <div class="d-flex gap-2 align-items-center">
                <button id="muteBtn" class="btn btn-ghost btn-sm" onclick="toggleMute()">Mic: muted</button>
                <div class="pill" id="clockPill">–</div>
              </div>
            </div>
          </div>

          <div class="mt-2 soft" style="font-size: 13px;">
            Tipp: Für später (AI): Worker hält bereits den neuesten BGR-Frame im RAM (perfekt für YOLO).
          </div>
        </div>
      </div>

      <!-- Side previews -->
      <div class="col-12 col-lg-4">
        <div class="card p-3">
          <div class="d-flex align-items-center justify-content-between">
            <div>
              <div class="smalllabel">WEBCAM A</div>
              <div class="fw-semibold" id="camATitle">–</div>
            </div>
            <button class="btn btn-ghost btn-sm" onclick="setMain('camA')">To Main</button>
          </div>
          <div class="mini-wrap mt-2">
            <img id="camAImg" src="" alt="Webcam A"/>
          </div>
          <div class="mt-2 soft" style="font-size: 12px;" id="camAStatus">–</div>
        </div>

        <div class="card p-3 mt-3">
          <div class="d-flex align-items-center justify-content-between">
            <div>
              <div class="smalllabel">WEBCAM B</div>
              <div class="fw-semibold" id="camBTitle">–</div>
            </div>
            <button class="btn btn-ghost btn-sm" onclick="setMain('camB')">To Main</button>
          </div>
          <div class="mini-wrap mt-2">
            <img id="camBImg" src="" alt="Webcam B"/>
          </div>
          <div class="mt-2 soft" style="font-size: 12px;" id="camBStatus">–</div>
        </div>
      </div>
    </div>

    <div class="mt-3 soft" style="font-size: 12px;">
      Wenn du Lag hast: Preset 720p120 oder JPEG Quality runter via ENV <code>BALLSCOPE_MJPEG_JPEG_QUALITY</code>.
    </div>
  </div>

  <audio id="micAudio" src="/audio/mic" muted preload="none" playsinline></audio>

  <!-- Settings Modal -->
  <div class="modal fade" id="settingsModal" tabindex="-1" aria-hidden="true">
    <div class="modal-dialog modal-lg modal-dialog-centered">
      <div class="modal-content" style="background:#0f1626; border:1px solid rgba(255,255,255,0.10); border-radius:16px;">
        <div class="modal-header" style="border-bottom: 1px solid rgba(255,255,255,0.08);">
          <h5 class="modal-title">Settings (YouTube-Style)</h5>
          <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal" aria-label="Close"></button>
        </div>

        <div class="modal-body">
          <div class="row g-3">
            <div class="col-12 col-md-4">
              <label class="smalllabel mb-1">Camera</label>
              <select id="settingsCamSelect" class="form-select select-dark" onchange="loadCamSettingsToUI()"></select>
            </div>

            <div class="col-12 col-md-8">
              <label class="smalllabel mb-1">Quality Preset</label>
              <select id="presetSelect" class="form-select select-dark"></select>
            </div>

            <div class="col-6 col-md-3">
              <label class="smalllabel mb-1">Brightness</label>
              <input id="brightness" class="form-control" type="number" placeholder="(auto/ignore)"/>
            </div>
            <div class="col-6 col-md-3">
              <label class="smalllabel mb-1">Contrast</label>
              <input id="contrast" class="form-control" type="number" placeholder="(auto/ignore)"/>
            </div>
            <div class="col-6 col-md-3">
              <label class="smalllabel mb-1">Saturation</label>
              <input id="saturation" class="form-control" type="number" placeholder="(auto/ignore)"/>
            </div>
            <div class="col-6 col-md-3">
              <label class="smalllabel mb-1">Gain</label>
              <input id="gain" class="form-control" type="number" placeholder="(auto/ignore)"/>
            </div>

            <div class="col-12 col-md-6">
              <div class="form-check">
                <input class="form-check-input" type="checkbox" id="autoWb">
                <label class="form-check-label" for="autoWb">Auto White Balance</label>
              </div>
              <div class="form-check mt-1">
                <input class="form-check-input" type="checkbox" id="autoExposure">
                <label class="form-check-label" for="autoExposure">Auto Exposure</label>
              </div>
              <div class="soft mt-2" style="font-size:12px;">
                Hinweis: Controls werden über den Treiber (v4l2-ctl) gesetzt. Falls ein Control nicht unterstützt wird, wird es einfach ignoriert (kein Crash).
              </div>
            </div>

            <div class="col-12 col-md-6">
              <div class="soft" style="font-size:12px;">
                Empfohlen für BallScope:
                <ul class="mb-0">
                  <li><b>720p120</b> für schnelle Bewegungen (Ball)</li>
                  <li><b>1080p60</b> als Standard</li>
                  <li><b>2160p20</b> wenn du Details willst (mehr Load)</li>
                </ul>
              </div>
            </div>
          </div>

          <div class="alert mt-3" id="settingsResult" style="display:none; background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.10); color:#e9eefb;"></div>
        </div>

        <div class="modal-footer" style="border-top: 1px solid rgba(255,255,255,0.08);">
          <button class="btn btn-ghost" onclick="resetSettingInputs()">Clear numeric (ignore)</button>
          <button class="btn btn-primary" onclick="applySettings()">Apply</button>
        </div>
      </div>
    </div>
  </div>

  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
  <script>
    let APP_STATE = null;

    function nowClock() {
      const d = new Date();
      return d.toLocaleTimeString();
    }

    function badgeStatus(isOpen, lastErr) {
      if (!isOpen) return '<span class="badge-dot" style="background:#ff4d4f;"></span>offline';
      if (lastErr) return '<span class="badge-dot" style="background:#faad14;"></span>warning';
      return '<span class="badge-dot" style="background:#52c41a;"></span>live';
    }

    async function refreshState() {
      const res = await fetch('/api/state');
      APP_STATE = await res.json();

      const mainSelect = document.getElementById('mainSelect');
      mainSelect.innerHTML = '';
      for (const camId of Object.keys(APP_STATE.cameras)) {
        const opt = document.createElement('option');
        opt.value = camId;
        opt.textContent = APP_STATE.cameras[camId].name;
        if (APP_STATE.main_source === camId) opt.selected = true;
        mainSelect.appendChild(opt);
      }

      const settingsCamSelect = document.getElementById('settingsCamSelect');
      settingsCamSelect.innerHTML = '';
      for (const camId of Object.keys(APP_STATE.cameras)) {
        const opt = document.createElement('option');
        opt.value = camId;
        opt.textContent = APP_STATE.cameras[camId].name + ' (' + APP_STATE.cameras[camId].src + ')';
        settingsCamSelect.appendChild(opt);
      }

      const presetSelect = document.getElementById('presetSelect');
      presetSelect.innerHTML = '';
      for (const key of Object.keys(APP_STATE.presets)) {
        const p = APP_STATE.presets[key];
        const opt = document.createElement('option');
        opt.value = key;
        opt.textContent = `${key}  (${p.w}×${p.h} @ ${p.fps}fps)`;
        presetSelect.appendChild(opt);
      }

      document.getElementById('camAImg').src = '/video/camA.mjpg?ts=' + Date.now();
      document.getElementById('camBImg').src = '/video/camB.mjpg?ts=' + Date.now();
      setMainUI(APP_STATE.main_source);

      const camA = APP_STATE.cameras.camA;
      const camB = APP_STATE.cameras.camB;

      document.getElementById('camATitle').textContent = camA.name;
      document.getElementById('camBTitle').textContent = camB.name;

      document.getElementById('camAStatus').innerHTML = `Status: ${badgeStatus(camA.is_open, camA.last_error)} &nbsp; · preset: <b>${camA.settings.preset}</b>`;
      document.getElementById('camBStatus').innerHTML = `Status: ${badgeStatus(camB.is_open, camB.last_error)} &nbsp; · preset: <b>${camB.settings.preset}</b>`;

      loadCamSettingsToUI();
    }

    function setMainUI(camId) {
      const cam = APP_STATE.cameras[camId];
      document.getElementById('mainTitle').textContent = cam ? cam.name : '–';
      document.getElementById('mainImg').src = `/video/${camId}.mjpg?ts=` + Date.now();
      document.getElementById('mainPresetPill').textContent = 'Preset: ' + (cam?.settings?.preset ?? '–');
      document.getElementById('mainStatusPill').innerHTML = 'Status: ' + badgeStatus(cam?.is_open, cam?.last_error);
    }

    async function setMain(camId) {
      await fetch('/api/main/' + camId, { method: 'POST' });
      await refreshState();
    }

    function resetSettingInputs() {
      for (const id of ["brightness","contrast","saturation","gain"]) {
        document.getElementById(id).value = '';
      }
    }

    async function toggleMute() {
      const audio = document.getElementById('micAudio');
      if (audio.muted) {
        audio.src = '/audio/mic?ts=' + Date.now();
        audio.muted = false;
        try {
          await audio.play();
        } catch (e) {}
      } else {
        audio.muted = true;
        audio.pause();
      }
      updateMuteBtn();
    }

    function updateMuteBtn() {
      const audio = document.getElementById('micAudio');
      const btn = document.getElementById('muteBtn');
      btn.textContent = audio.muted ? 'Mic: muted' : 'Mic: live';
    }

    function loadCamSettingsToUI() {
      if (!APP_STATE) return;
      const camId = document.getElementById('settingsCamSelect').value;
      const cam = APP_STATE.cameras[camId];
      if (!cam) return;

      document.getElementById('presetSelect').value = cam.settings.preset;

      document.getElementById('brightness').value = (cam.settings.brightness ?? '');
      document.getElementById('contrast').value = (cam.settings.contrast ?? '');
      document.getElementById('saturation').value = (cam.settings.saturation ?? '');
      document.getElementById('gain').value = (cam.settings.gain ?? '');

      document.getElementById('autoWb').checked = !!cam.settings.auto_wb;
      document.getElementById('autoExposure').checked = !!cam.settings.auto_exposure;
    }

    async function applySettings() {
      const camId = document.getElementById('settingsCamSelect').value;

      const payload = {
        preset: document.getElementById('presetSelect').value,
        brightness: document.getElementById('brightness').value === '' ? null : Number(document.getElementById('brightness').value),
        contrast: document.getElementById('contrast').value === '' ? null : Number(document.getElementById('contrast').value),
        saturation: document.getElementById('saturation').value === '' ? null : Number(document.getElementById('saturation').value),
        gain: document.getElementById('gain').value === '' ? null : Number(document.getElementById('gain').value),
        auto_wb: document.getElementById('autoWb').checked,
        auto_exposure: document.getElementById('autoExposure').checked
      };

      const res = await fetch('/api/settings/' + camId, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      const out = await res.json();
      const box = document.getElementById('settingsResult');
      box.style.display = 'block';
      box.textContent = res.ok ? ('Applied to ' + camId + '. (Preset-Wechsel macht einen Safe-Reopen, kein Crash)') : ('Error: ' + (out.detail ?? 'unknown'));
      await refreshState();
    }

    setInterval(() => {
      const el = document.getElementById('clockPill');
      if (el) el.textContent = nowClock();
    }, 500);

    refreshState();
    updateMuteBtn();
  </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(UI_HTML)

# -----------------------------
# Run
# -----------------------------
# uvicorn main:app --host 0.0.0.0 --port 8000

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
