import os
import platform
from typing import Optional, Tuple


def _linux_camera_device(device: str) -> str:
    dev = str(device).strip()
    if dev.isdigit():
        return f"/dev/video{int(dev)}"
    return dev


def gst_camera_source_elements(device: str, w: int, h: int, fps: int) -> Tuple[Optional[list[str]], Optional[str]]:
    """
    Return a GStreamer camera source fragment ending with '!' and JPEG output.

    Linux/Jetson:
      v4l2src -> image/jpeg -> jpegparse
    macOS:
      avfvideosrc -> raw convert/scale/rate -> jpegenc
    """
    system = platform.system()
    dev = str(device).strip()

    if system == "Darwin":
      if not dev.isdigit():
          return None, f"macOS camera source must be a numeric index (got: {dev})"
      idx = int(dev)
      return [
          "avfvideosrc",
          f"device-index={idx}",
          "do-timestamp=true",
          "!",
          "videoconvert",
          "!",
          "videoscale",
          "!",
          "videorate",
          "!",
          f"video/x-raw,width={int(w)},height={int(h)},framerate={int(fps)}/1",
          "!",
          "jpegenc",
          "quality=90",
          "!",
      ], None

    # Default Linux/Jetson behavior (unchanged)
    dev_path = _linux_camera_device(dev)
    if not os.path.exists(dev_path):
        return None, f"Device not found: {dev_path}"
    return [
        "v4l2src",
        f"device={dev_path}",
        "do-timestamp=true",
        "!",
        f"image/jpeg,width={int(w)},height={int(h)},framerate={int(fps)}/1",
        "!",
        "jpegparse",
        "!",
    ], None

