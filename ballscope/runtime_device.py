import os
import platform
from typing import Any, Dict, List


def is_apple_silicon_mac() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def is_jetson() -> bool:
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        return False
    if os.path.exists("/etc/nv_tegra_release"):
        return True
    model_path = "/proc/device-tree/model"
    if os.path.exists(model_path):
        try:
            with open(model_path, "rb") as f:
                raw = f.read()
            if b"jetson" in raw.lower():
                return True
        except Exception:
            pass
    return False


def _cuda_available(torch_mod: Any) -> bool:
    try:
        return bool(torch_mod is not None and torch_mod.cuda.is_available())
    except Exception:
        return False


def _mps_available(torch_mod: Any) -> bool:
    try:
        return bool(
            torch_mod is not None
            and hasattr(torch_mod, "backends")
            and hasattr(torch_mod.backends, "mps")
            and torch_mod.backends.mps.is_available()
        )
    except Exception:
        return False


def resolve_torch_device(preferred: str = "auto", torch_mod: Any = None) -> str:
    pref = (preferred or "auto").strip().lower()
    cuda_ok = _cuda_available(torch_mod)
    mps_ok = _mps_available(torch_mod)

    if pref in ("", "auto"):
        if is_jetson() and cuda_ok:
            return "cuda:0"
        if cuda_ok:
            return "cuda:0"
        if is_apple_silicon_mac() and mps_ok:
            return "mps"
        if mps_ok:
            return "mps"
        return "cpu"

    if pref == "cuda":
        if cuda_ok:
            return "cuda:0"
        raise RuntimeError("CUDA requested but not available.")

    if pref.startswith("cuda:") or pref.isdigit():
        if cuda_ok:
            if pref.isdigit():
                return f"cuda:{pref}"
            return pref
        raise RuntimeError("CUDA requested but not available.")

    if pref == "mps":
        if mps_ok:
            return "mps"
        raise RuntimeError("MPS requested but not available.")

    if pref == "cpu":
        return "cpu"

    return pref


def runtime_device_options(torch_mod: Any = None) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = [{"id": "auto", "label": "Auto (recommended)"}]

    if _cuda_available(torch_mod):
        try:
            count = int(torch_mod.cuda.device_count())
        except Exception:
            count = 0
        if count <= 0:
            out.append({"id": "cuda:0", "label": "CUDA GPU"})
        else:
            for i in range(count):
                try:
                    name = str(torch_mod.cuda.get_device_name(i))
                except Exception:
                    name = f"GPU {i}"
                label = f"CUDA {i} ({name})"
                if is_jetson() and i == 0:
                    label += " [Jetson]"
                out.append({"id": f"cuda:{i}", "label": label})

    if _mps_available(torch_mod):
        out.append({"id": "mps", "label": "Apple Silicon GPU (MPS)"})

    out.append({"id": "cpu", "label": "CPU"})
    return out
