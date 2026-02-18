import os
import platform
from dataclasses import dataclass
from typing import Dict

QUALITY_PRESETS: Dict[str, Dict[str, int]] = {
    "2160p20": {"w": 3840, "h": 2160, "fps": 20},
    "1080p30": {"w": 1920, "h": 1080, "fps": 30},
    "1520p50": {"w": 2016, "h": 1520, "fps": 50},
    "1080p60": {"w": 1920, "h": 1080, "fps": 60},
    "720p120": {"w": 1280, "h": 720, "fps": 120},
}

DEFAULT_PRESET = os.getenv("BALLSCOPE_DEFAULT_PRESET", "1080p60")


def _is_apple_silicon_mac() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _is_jetson_linux() -> bool:
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        return False
    if os.path.exists("/etc/nv_tegra_release"):
        return True
    model = "/proc/device-tree/model"
    if os.path.exists(model):
        try:
            with open(model, "rb") as f:
                raw = f.read().lower()
            return b"jetson" in raw
        except Exception:
            return False
    return False


if _is_apple_silicon_mac():
    _default_cam_left = "0"
    _default_cam_right = "1"
elif _is_jetson_linux():
    _default_cam_left = "/dev/video0"
    _default_cam_right = "/dev/video2"
else:
    _default_cam_left = "0"
    _default_cam_right = "1"

DEFAULT_CAM_LEFT = os.getenv("BALLSCOPE_CAM_LEFT", _default_cam_left)
DEFAULT_CAM_RIGHT = os.getenv("BALLSCOPE_CAM_RIGHT", _default_cam_right)

DEFAULT_VIDEO_DIR = os.getenv("BALLSCOPE_VIDEO_DIR", "recordings")
DEFAULT_RECORD_MODE = os.getenv("BALLSCOPE_RECORD_MODE", "mjpg")
DEFAULT_RECORD_FPS = float(os.getenv("BALLSCOPE_RECORD_FPS", "60"))
DEFAULT_RECORD_JPEG_QUALITY = int(os.getenv("BALLSCOPE_RECORD_JPEG_QUALITY", "90"))

MJPEG_JPEG_QUALITY = int(os.getenv("BALLSCOPE_MJPEG_JPEG_QUALITY", "85"))
STREAM_MAX_FPS = float(os.getenv("BALLSCOPE_STREAM_MAX_FPS", "0"))
PREFERRED_PIXFMT = os.getenv("BALLSCOPE_PIXFMT", "MJPG")
V4L2_BUFFER_SIZE = int(os.getenv("BALLSCOPE_V4L2_BUFFER_SIZE", "1"))

AI_MODEL_PATH = os.getenv("BALLSCOPE_MODEL_PATH", "models/football-ball-detection.pt")
AI_DEVICE = os.getenv("BALLSCOPE_AI_DEVICE", "auto")
AI_IMG_SIZE = int(os.getenv("BALLSCOPE_AI_IMG_SIZE", "640"))
AI_CONF = float(os.getenv("BALLSCOPE_AI_CONF", "0.35"))
AI_IOU = float(os.getenv("BALLSCOPE_AI_IOU", "0.5"))
AI_ZOOM = float(os.getenv("BALLSCOPE_AI_ZOOM", "1.6"))
AI_SMOOTH = float(os.getenv("BALLSCOPE_AI_SMOOTH", "0.85"))
AI_LOST_HOLD_SEC = float(os.getenv("BALLSCOPE_AI_LOST_HOLD_SEC", "1.5"))
AI_DETECT_EVERY = int(os.getenv("BALLSCOPE_AI_DETECT_EVERY", "1"))
AI_OUTPUT_W = int(os.getenv("BALLSCOPE_AI_OUTPUT_W", "1920"))
AI_OUTPUT_H = int(os.getenv("BALLSCOPE_AI_OUTPUT_H", "1080"))
AI_CLASS_ID = int(os.getenv("BALLSCOPE_AI_CLASS_ID", "0"))
AI_RECORD_USE_ZOOM = os.getenv("BALLSCOPE_AI_RECORD_USE_ZOOM", "true").strip().lower() in ("1", "true", "yes", "on")

@dataclass
class RecordingConfig:
    preset: str = DEFAULT_PRESET
    output_dir: str = DEFAULT_VIDEO_DIR
    mode: str = DEFAULT_RECORD_MODE
    fps: float = DEFAULT_RECORD_FPS
    jpeg_quality: int = DEFAULT_RECORD_JPEG_QUALITY

@dataclass
class AiConfig:
    model_path: str = AI_MODEL_PATH
    device: str = AI_DEVICE
    imgsz: int = AI_IMG_SIZE
    conf: float = AI_CONF
    iou: float = AI_IOU
    zoom: float = AI_ZOOM
    smooth: float = AI_SMOOTH
    lost_hold_sec: float = AI_LOST_HOLD_SEC
    detect_every: int = AI_DETECT_EVERY
    output_w: int = AI_OUTPUT_W
    output_h: int = AI_OUTPUT_H
    class_id: int = AI_CLASS_ID
    record_use_zoom: bool = AI_RECORD_USE_ZOOM
