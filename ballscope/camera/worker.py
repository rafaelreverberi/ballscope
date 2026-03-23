import json
import os
import shutil
import time
import threading
import subprocess
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple
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


CONTROL_SPECS = {
    "hdr": {
        "label": "HDR / Backlight",
        "kind": "bool",
        "default": False,
        "help": "Backlight compensation. For BRIO this is used as the HDR-style toggle.",
    },
    "auto_exposure": {
        "label": "Auto Exposure",
        "kind": "bool",
        "default": True,
    },
    "exposure_time": {
        "label": "Exposure",
        "kind": "int",
        "default": 12,
        "min": 1,
        "max": 250,
        "step": 1,
    },
    "exposure_priority": {
        "label": "Exposure Priority",
        "kind": "bool",
        "default": False,
        "help": "Allows dynamic frame rate / exposure priority when supported by the camera backend.",
    },
    "brightness": {
        "label": "Brightness",
        "kind": "int",
        "default": 128,
        "min": 0,
        "max": 255,
        "step": 1,
    },
    "contrast": {
        "label": "Contrast",
        "kind": "int",
        "default": 128,
        "min": 0,
        "max": 255,
        "step": 1,
    },
    "saturation": {
        "label": "Saturation",
        "kind": "int",
        "default": 128,
        "min": 0,
        "max": 255,
        "step": 1,
    },
    "sharpness": {
        "label": "Sharpness",
        "kind": "int",
        "default": 128,
        "min": 0,
        "max": 255,
        "step": 1,
    },
    "gain": {
        "label": "Gain",
        "kind": "int",
        "default": 0,
        "min": 0,
        "max": 255,
        "step": 1,
    },
    "power_line_frequency": {
        "label": "Power Line Frequency",
        "kind": "menu",
        "default": 1,
        "options": [
            {"value": 0, "label": "Disabled"},
            {"value": 1, "label": "50 Hz"},
            {"value": 2, "label": "60 Hz"},
        ],
    },
    "auto_wb": {
        "label": "Auto White Balance",
        "kind": "bool",
        "default": True,
    },
    "white_balance_temperature": {
        "label": "White Balance",
        "kind": "int",
        "default": 4500,
        "min": 2000,
        "max": 6500,
        "step": 1,
    },
    "auto_focus": {
        "label": "Auto Focus",
        "kind": "bool",
        "default": True,
    },
    "focus": {
        "label": "Focus",
        "kind": "int",
        "default": 15,
        "min": 0,
        "max": 255,
        "step": 1,
    },
    "zoom": {
        "label": "Zoom",
        "kind": "int",
        "default": 100,
        "min": 100,
        "max": 500,
        "step": 1,
    },
    "pan": {
        "label": "Pan",
        "kind": "int",
        "default": 0,
        "min": -36000,
        "max": 36000,
        "step": 3600,
    },
    "tilt": {
        "label": "Tilt",
        "kind": "int",
        "default": 0,
        "min": -36000,
        "max": 36000,
        "step": 3600,
    },
}

LINUX_CTRL_ALIASES = {
    "hdr": ["backlight_compensation"],
    "auto_exposure": ["auto_exposure"],
    "exposure_time": ["exposure_time_absolute"],
    "exposure_priority": ["exposure_dynamic_framerate", "auto_exposure_bias"],
    "brightness": ["brightness"],
    "contrast": ["contrast"],
    "saturation": ["saturation"],
    "sharpness": ["sharpness"],
    "gain": ["gain"],
    "power_line_frequency": ["power_line_frequency"],
    "auto_wb": ["white_balance_automatic", "white_balance_temperature_auto"],
    "white_balance_temperature": ["white_balance_temperature"],
    "auto_focus": ["focus_automatic_continuous"],
    "focus": ["focus_absolute"],
    "zoom": ["zoom_absolute"],
    "pan": ["pan_absolute"],
    "tilt": ["tilt_absolute"],
}

MAC_CTRL_NAMES = {
    "hdr": "backlight_compensation",
    "auto_exposure": "auto_exposure_mode",
    "exposure_time": "absolute_exposure_time",
    "exposure_priority": "auto_exposure_priority",
    "brightness": "brightness",
    "contrast": "contrast",
    "saturation": "saturation",
    "sharpness": "sharpness",
    "gain": "gain",
    "power_line_frequency": "power_line_frequency",
    "auto_wb": "auto_white_balance_temperature",
    "white_balance_temperature": "white_balance_temperature",
    "auto_focus": "auto_focus",
    "focus": "absolute_focus",
    "zoom": "absolute_zoom",
    "pan_tilt": "absolute_pan_tilt",
}

V4L2_CTRL_RE = re.compile(r"^\s*([a-zA-Z0-9_]+)\s+0x[0-9a-f]+ \(([^)]+)\)\s*:\s*(.*)$")
V4L2_KV_RE = re.compile(r"(min|max|step|default|value)=([^\s]+)")


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


_NO_CAMERA_NOTICE_LOCK = threading.Lock()
_NO_CAMERA_NOTICE_SHOWN = False
_MACOS_CAMERA_AUTH_OK: Optional[bool] = None
_MACOS_CAMERA_AUTH_MESSAGE: Optional[str] = None

try:
    from AVFoundation import AVCaptureDevice, AVMediaTypeVideo  # type: ignore
    _HAS_AVFOUNDATION_AUTH = True
except Exception:
    AVCaptureDevice = None  # type: ignore
    AVMediaTypeVideo = "vide"  # type: ignore
    _HAS_AVFOUNDATION_AUTH = False


def log_no_camera_notice_once() -> None:
    global _NO_CAMERA_NOTICE_SHOWN
    with _NO_CAMERA_NOTICE_LOCK:
        if _NO_CAMERA_NOTICE_SHOWN:
            return
        _NO_CAMERA_NOTICE_SHOWN = True
    print("[WARN] No camera detected. Please connect a camera (Linux: /dev/videoX, macOS: 0/1).")


def prepare_macos_camera_access(sources: list[str]) -> Tuple[bool, str]:
    global _MACOS_CAMERA_AUTH_OK, _MACOS_CAMERA_AUTH_MESSAGE
    if not _is_macos():
        return True, "not-macos"

    if _MACOS_CAMERA_AUTH_OK is True:
        os.environ["OPENCV_AVFOUNDATION_SKIP_AUTH"] = "1"
        return True, _MACOS_CAMERA_AUTH_MESSAGE or "authorized"

    if not _HAS_AVFOUNDATION_AUTH or AVCaptureDevice is None:
        os.environ["OPENCV_AVFOUNDATION_SKIP_AUTH"] = "1"
        _MACOS_CAMERA_AUTH_OK = False
        _MACOS_CAMERA_AUTH_MESSAGE = (
            "macOS camera permission check is unavailable. Allow camera access for the app or Terminal "
            "in System Settings > Privacy & Security > Camera, then restart BallScope."
        )
        return False, _MACOS_CAMERA_AUTH_MESSAGE

    try:
        status = int(AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeVideo))
    except Exception as exc:
        os.environ["OPENCV_AVFOUNDATION_SKIP_AUTH"] = "1"
        _MACOS_CAMERA_AUTH_OK = False
        _MACOS_CAMERA_AUTH_MESSAGE = f"Could not query macOS camera permission state: {exc}"
        return False, _MACOS_CAMERA_AUTH_MESSAGE

    if status == 3:
        os.environ["OPENCV_AVFOUNDATION_SKIP_AUTH"] = "1"
        _MACOS_CAMERA_AUTH_OK = True
        _MACOS_CAMERA_AUTH_MESSAGE = "Camera access ready on macOS."
        return True, _MACOS_CAMERA_AUTH_MESSAGE

    if status == 0:
        done = threading.Event()
        granted_box = {"granted": False}

        def _completion(granted):
            try:
                granted_box["granted"] = bool(granted)
            finally:
                done.set()

        try:
            AVCaptureDevice.requestAccessForMediaType_completionHandler_(AVMediaTypeVideo, _completion)
        except Exception as exc:
            os.environ["OPENCV_AVFOUNDATION_SKIP_AUTH"] = "1"
            _MACOS_CAMERA_AUTH_OK = False
            _MACOS_CAMERA_AUTH_MESSAGE = f"Could not request macOS camera access: {exc}"
            return False, _MACOS_CAMERA_AUTH_MESSAGE

        done.wait(timeout=12.0)
        os.environ["OPENCV_AVFOUNDATION_SKIP_AUTH"] = "1"
        if granted_box["granted"]:
            _MACOS_CAMERA_AUTH_OK = True
            _MACOS_CAMERA_AUTH_MESSAGE = "Camera access ready on macOS."
            return True, _MACOS_CAMERA_AUTH_MESSAGE

    os.environ["OPENCV_AVFOUNDATION_SKIP_AUTH"] = "1"
    _MACOS_CAMERA_AUTH_OK = False
    _MACOS_CAMERA_AUTH_MESSAGE = (
        "macOS camera access is not granted. Allow camera access for the app or Terminal in "
        "System Settings > Privacy & Security > Camera, then restart BallScope."
    )
    return False, _MACOS_CAMERA_AUTH_MESSAGE


def macos_camera_auth_message() -> Optional[str]:
    if not _is_macos():
        return None
    return _MACOS_CAMERA_AUTH_MESSAGE


def list_linux_video_devices() -> list[str]:
    if not _is_linux():
        return []
    try:
        entries = os.listdir("/dev")
    except Exception:
        return []
    devices = [f"/dev/{name}" for name in entries if re.fullmatch(r"video\d+", name)]
    devices.sort()
    return devices


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


def run_uvcc(args: list[str]) -> Tuple[bool, str]:
    try:
        p = subprocess.run(
            ["uvcc", *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if p.returncode == 0:
            return True, (p.stdout or "").strip()
        err = (p.stderr or p.stdout or "").strip()
        return False, err or f"uvcc failed rc={p.returncode}"
    except FileNotFoundError:
        return False, "uvcc not found (install Node.js/npm, then npm install --global uvcc)"
    except Exception as exc:
        return False, f"uvcc error: {exc}"


def driver_set_format(dev: str, w: int, h: int, fps: int, pixfmt: str) -> Tuple[bool, str]:
    ok1, msg1 = run_v4l2(["-d", dev, f"--set-fmt-video=width={w},height={h},pixelformat={pixfmt}"])
    ok2, msg2 = run_v4l2(["-d", dev, f"--set-parm={fps}"])
    ok = ok1 and ok2
    msg = "; ".join([m for m in [msg1, msg2] if m]) or ("ok" if ok else "failed")
    return ok, msg


def _parse_control_value(raw: str) -> Any:
    s = str(raw).strip().strip(",")
    if s.lower() in {"true", "false"}:
        return s.lower() == "true"
    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        return tuple(_parse_control_value(p) for p in parts)
    if re.fullmatch(r"-?\d+", s):
        try:
            return int(s)
        except Exception:
            return s
    return s


def _linux_dev_for_source(src: str) -> Optional[str]:
    dev = resolve_dev_path(src)
    if dev is not None:
        return dev
    normalized = normalize_linux_source(src)
    if normalized is None:
        return None
    return resolve_dev_path(normalized)


def _parse_v4l2_controls(output: str) -> Dict[str, Dict[str, Any]]:
    controls: Dict[str, Dict[str, Any]] = {}
    current_name: Optional[str] = None
    for line in (output or "").splitlines():
        m = V4L2_CTRL_RE.match(line)
        if m:
            name = m.group(1)
            kind = m.group(2)
            rest = m.group(3)
            entry: Dict[str, Any] = {"type": kind}
            for key, raw in V4L2_KV_RE.findall(rest):
                entry[key] = _parse_control_value(raw)
            controls[name] = entry
            current_name = name
            continue
        if current_name and ":" in line and line.lstrip().startswith(tuple(str(i) for i in range(10))):
            idx_part, label = line.split(":", 1)
            try:
                idx = int(idx_part.strip(), 10)
            except Exception:
                continue
            controls.setdefault(current_name, {}).setdefault("options", []).append(
                {"value": idx, "label": label.strip()}
            )
    return controls


def _read_linux_control_snapshot(src: str) -> Dict[str, Dict[str, Any]]:
    dev = _linux_dev_for_source(src)
    if dev is None:
        return {}
    ok, out = run_v4l2(["-d", dev, "--list-ctrls-menus"])
    if not ok:
        return {}
    return _parse_v4l2_controls(out)


def _uvcc_selector_args_for_source(src: str) -> list[str]:
    s = (src or "").strip()
    if not s.isdigit():
        return []
    idx = int(s)
    ok, out = run_uvcc(["devices"])
    if not ok or not out:
        return []
    devices = []
    current: Dict[str, str] = {}
    for raw_line in out.splitlines():
        line = raw_line.strip()
        if not line:
            if current:
                devices.append(current)
                current = {}
            continue
        low = line.lower()
        vendor_match = re.search(r"(?:vendor|vid|idvendor)[^0-9a-f]*(0x[0-9a-f]+|\d+)", low)
        product_match = re.search(r"(?:product|pid|idproduct)[^0-9a-f]*(0x[0-9a-f]+|\d+)", low)
        address_match = re.search(r"(?:address|device address)[^0-9]*(\d+)", low)
        if vendor_match:
            current["vendor"] = vendor_match.group(1)
        if product_match:
            current["product"] = product_match.group(1)
        if address_match:
            current["address"] = address_match.group(1)
    if current:
        devices.append(current)
    if idx < 0 or idx >= len(devices):
        return []
    chosen = devices[idx]
    args = []
    if chosen.get("vendor"):
        args += ["--vendor", chosen["vendor"]]
    if chosen.get("product"):
        args += ["--product", chosen["product"]]
    if chosen.get("address"):
        args += ["--address", chosen["address"]]
    return args


def _read_uvcc_json(args: list[str]) -> Optional[Dict[str, Any]]:
    ok, out = run_uvcc(args)
    if not ok or not out:
        return None
    try:
        loaded = json.loads(out)
        if isinstance(loaded, dict):
            return loaded
    except Exception:
        return None
    return None


def _read_macos_control_snapshot(src: str) -> Dict[str, Any]:
    selector = _uvcc_selector_args_for_source(src)
    export_data = _read_uvcc_json([*selector, "export"]) or {}
    range_data = _read_uvcc_json([*selector, "ranges"]) or {}
    return {
        "values": export_data,
        "ranges": range_data,
        "selector": selector,
    }


def _coerce_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return None


def _read_saved_setting(settings: "CameraSettings", key: str) -> Any:
    return getattr(settings, key, None)


def camera_control_metadata(src: str, settings: "CameraSettings") -> Dict[str, Any]:
    platform_name = "linux" if _is_linux() else "macos" if _is_macos() else "other"
    backend = "basic"
    backend_available = False
    backend_hint = None
    controls_out = []

    linux_snapshot: Dict[str, Dict[str, Any]] = {}
    mac_snapshot: Dict[str, Any] = {}
    if _is_linux():
        backend = "v4l2"
        linux_snapshot = _read_linux_control_snapshot(src)
        backend_available = bool(linux_snapshot) or shutil.which("v4l2-ctl") is not None
        if not backend_available:
            backend_hint = "Install v4l-utils to enable advanced camera controls."
    elif _is_macos():
        backend = "uvcc"
        mac_snapshot = _read_macos_control_snapshot(src)
        backend_available = bool(mac_snapshot.get("values")) or shutil.which("uvcc") is not None
        if not backend_available:
            backend_hint = "Install Node.js/npm and uvcc to enable advanced BRIO controls."

    for key, spec in CONTROL_SPECS.items():
        item = {
            "id": key,
            "label": spec["label"],
            "kind": spec["kind"],
            "help": spec.get("help"),
            "min": spec.get("min"),
            "max": spec.get("max"),
            "step": spec.get("step", 1),
            "options": spec.get("options"),
            "current": _read_saved_setting(settings, key),
            "default": spec.get("default"),
        }

        if _is_linux():
            control_name = next((alias for alias in LINUX_CTRL_ALIASES.get(key, []) if alias in linux_snapshot), None)
            if control_name:
                snap = linux_snapshot[control_name]
                item["backend_control"] = control_name
                item["supported"] = True
                item["min"] = snap.get("min", item["min"])
                item["max"] = snap.get("max", item["max"])
                item["step"] = snap.get("step", item["step"])
                if snap.get("options"):
                    item["options"] = snap["options"]
                value = snap.get("value")
                if key == "hdr":
                    item["current"] = bool(int(value or 0) > 0)
                elif key == "auto_exposure":
                    item["current"] = int(value or 3) != 1
                elif key == "auto_wb":
                    item["current"] = bool(int(value or 0))
                elif key == "auto_focus":
                    item["current"] = bool(int(value or 0))
                elif key == "exposure_priority":
                    item["current"] = bool(int(value or 0))
                else:
                    item["current"] = value
            else:
                item["supported"] = False
        elif _is_macos():
            values = mac_snapshot.get("values") or {}
            ranges = mac_snapshot.get("ranges") or {}
            mac_key = MAC_CTRL_NAMES.get(key)
            if key in {"pan", "tilt"}:
                pan_tilt = values.get(MAC_CTRL_NAMES["pan_tilt"])
                range_info = ranges.get(MAC_CTRL_NAMES["pan_tilt"], {})
                item["supported"] = pan_tilt is not None or bool(range_info)
                if isinstance(pan_tilt, list) and len(pan_tilt) >= 2:
                    item["current"] = pan_tilt[0] if key == "pan" else pan_tilt[1]
                if isinstance(range_info, dict):
                    item["min"] = range_info.get("min", item["min"])
                    item["max"] = range_info.get("max", item["max"])
                    item["step"] = range_info.get("step", item["step"])
            elif mac_key:
                item["backend_control"] = mac_key
                item["supported"] = mac_key in values or mac_key in ranges
                value = values.get(mac_key)
                range_info = ranges.get(mac_key, {})
                if isinstance(range_info, dict):
                    item["min"] = range_info.get("min", item["min"])
                    item["max"] = range_info.get("max", item["max"])
                    item["step"] = range_info.get("step", item["step"])
                if key == "hdr":
                    item["current"] = bool(int(value or 0) > 0)
                elif key == "auto_exposure":
                    item["current"] = int(value or 8) != 1 if value is not None else item["current"]
                else:
                    item["current"] = value if value is not None else item["current"]
            else:
                item["supported"] = False
        else:
            item["supported"] = False

        if item["current"] is None:
            item["current"] = item.get("default")
        controls_out.append(item)

    return {
        "platform": platform_name,
        "backend": backend,
        "backend_available": backend_available,
        "backend_hint": backend_hint,
        "controls": controls_out,
    }


def apply_camera_controls(src: str, settings: "CameraSettings") -> Tuple[bool, str]:
    parts = []
    ok_all = True

    def record(result_ok: bool, msg: str):
        nonlocal ok_all
        ok_all = ok_all and result_ok
        if msg:
            parts.append(msg)

    if _is_linux():
        dev = _linux_dev_for_source(src)
        if dev is None:
            return False, "Invalid Linux camera source."
        snapshot = _read_linux_control_snapshot(src)

        def set_ctrl(name: str, value: Any):
            record(*run_v4l2(["-d", dev, f"--set-ctrl={name}={value}"]))

        def first_existing(key: str, fallback: Optional[str] = None) -> Optional[str]:
            for alias in LINUX_CTRL_ALIASES.get(key, []):
                if alias in snapshot:
                    return alias
            return fallback

        if settings.hdr is not None:
            ctrl = first_existing("hdr", "backlight_compensation")
            if ctrl:
                set_ctrl(ctrl, 1 if settings.hdr else 0)
        if settings.auto_exposure is not None:
            ctrl = first_existing("auto_exposure", "auto_exposure")
            if ctrl:
                set_ctrl(ctrl, 3 if settings.auto_exposure else 1)
        if settings.exposure_priority is not None:
            ctrl = first_existing("exposure_priority")
            if ctrl:
                set_ctrl(ctrl, 1 if settings.exposure_priority else 0)
        if settings.exposure_time is not None and settings.auto_exposure is False:
            ctrl = first_existing("exposure_time", "exposure_time_absolute")
            if ctrl:
                set_ctrl(ctrl, int(settings.exposure_time))

        for key in ["brightness", "contrast", "saturation", "sharpness", "gain", "power_line_frequency", "zoom", "pan", "tilt"]:
            value = getattr(settings, key)
            if value is None:
                continue
            linux_name = first_existing(key, LINUX_CTRL_ALIASES[key][0])
            if linux_name:
                set_ctrl(linux_name, int(value))

        if settings.auto_wb is not None:
            ctrl = first_existing("auto_wb")
            if ctrl:
                set_ctrl(ctrl, 1 if settings.auto_wb else 0)
        if settings.white_balance_temperature is not None and settings.auto_wb is False:
            ctrl = first_existing("white_balance_temperature", "white_balance_temperature")
            if ctrl:
                set_ctrl(ctrl, int(settings.white_balance_temperature))

        if settings.auto_focus is not None:
            ctrl = first_existing("auto_focus", "focus_automatic_continuous")
            if ctrl:
                set_ctrl(ctrl, 1 if settings.auto_focus else 0)
        if settings.focus is not None and settings.auto_focus is False:
            ctrl = first_existing("focus", "focus_absolute")
            if ctrl:
                set_ctrl(ctrl, int(settings.focus))

        msg = "; ".join([m for m in parts if m]) or ("ok" if ok_all else "controls may be unsupported")
        return ok_all, msg

    if _is_macos():
        selector = _uvcc_selector_args_for_source(src)

        def set_uvcc(name: str, *values: Any):
            args = [*selector, "set", name, *[str(v) for v in values]]
            record(*run_uvcc(args))

        if settings.hdr is not None:
            set_uvcc("backlight_compensation", 1 if settings.hdr else 0)
        if settings.auto_exposure is not None:
            set_uvcc("auto_exposure_mode", 8 if settings.auto_exposure else 1)
        if settings.exposure_priority is not None:
            set_uvcc("auto_exposure_priority", 1 if settings.exposure_priority else 0)
        if settings.exposure_time is not None and settings.auto_exposure is False:
            set_uvcc("absolute_exposure_time", int(settings.exposure_time))

        for key in ["brightness", "contrast", "saturation", "sharpness", "gain", "power_line_frequency", "white_balance_temperature", "zoom"]:
            value = getattr(settings, key)
            if value is None:
                continue
            if key == "white_balance_temperature" and settings.auto_wb is not False:
                continue
            mac_name = MAC_CTRL_NAMES[key]
            set_uvcc(mac_name, int(value))

        if settings.auto_wb is not None:
            set_uvcc("auto_white_balance_temperature", 1 if settings.auto_wb else 0)
        if settings.auto_focus is not None:
            set_uvcc("auto_focus", 1 if settings.auto_focus else 0)
        if settings.focus is not None and settings.auto_focus is False:
            set_uvcc("absolute_focus", int(settings.focus))

        pan_value = settings.pan
        tilt_value = settings.tilt
        if pan_value is not None or tilt_value is not None:
            current = (_read_macos_control_snapshot(src).get("values") or {}).get("absolute_pan_tilt") or [0, 0]
            cur_pan = int(current[0]) if isinstance(current, list) and len(current) >= 2 else 0
            cur_tilt = int(current[1]) if isinstance(current, list) and len(current) >= 2 else 0
            set_uvcc("absolute_pan_tilt", int(pan_value if pan_value is not None else cur_pan), int(tilt_value if tilt_value is not None else cur_tilt))

        msg = "; ".join([m for m in parts if m]) or ("ok" if ok_all else "controls may be unsupported")
        return ok_all, msg

    return False, "Advanced camera controls are not available on this platform."


@dataclass
class CameraSettings:
    preset: str = DEFAULT_PRESET
    brightness: Optional[int] = None
    contrast: Optional[int] = None
    saturation: Optional[int] = None
    sharpness: Optional[int] = None
    gain: Optional[int] = None
    auto_wb: Optional[bool] = True
    white_balance_temperature: Optional[int] = None
    auto_exposure: Optional[bool] = True
    exposure_time: Optional[int] = None
    exposure_priority: Optional[bool] = None
    hdr: Optional[bool] = None
    auto_focus: Optional[bool] = None
    focus: Optional[int] = None
    zoom: Optional[int] = None
    pan: Optional[int] = None
    tilt: Optional[int] = None


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
            video_devices = list_linux_video_devices()
            if not video_devices:
                self.state.last_error = "No Linux camera device found. Please connect a camera."
                log_no_camera_notice_once()
                return None

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
        if dev is not None and _is_linux() and not os.path.exists(dev):
            self.state.last_error = f"Camera source not present: {dev}. Please connect a camera."
            log_no_camera_notice_once()
            return None
        if dev is not None:
            ok, msg = driver_set_format(dev, w, h, fps, PREFERRED_PIXFMT)
            if not ok:
                self.state.last_error = f"Driver fmt warn: {msg}"
            else:
                self.state.last_error = None

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
            if _is_macos() and macos_camera_auth_message():
                self.state.last_error = macos_camera_auth_message()
            elif open_err:
                self.state.last_error = f"Open error: {open_err}"
            else:
                self.state.last_error = f"Could not open camera source: {src_to_open}"
            return None
        self.state.src = used_src

        okc, msgc = apply_camera_controls(self.state.src, self.state.settings)
        if not okc and msgc:
            self.state.last_error = (
                (self.state.last_error + " | " if self.state.last_error else "") + f"Ctrl warn: {msgc}"
            )

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
        ok, msg = apply_camera_controls(self.state.src, self.state.settings)
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
