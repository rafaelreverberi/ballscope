import platform
from typing import Optional


def gst_audio_source_elements(audio_device: Optional[str]) -> list[str]:
    """
    Return a GStreamer source fragment for the selected audio device.

    Linux/Jetson keeps ALSA behavior.
    macOS prefers AVFoundation indices (from ffmpeg device listing); "default" uses autoaudiosrc.
    """
    if not audio_device:
        return []

    dev = str(audio_device).strip()
    if not dev:
        return []

    if platform.system() == "Darwin":
        if dev == "default":
            return ["autoaudiosrc", "do-timestamp=true", "!"]
        if dev.isdigit():
            return ["avfaudiosrc", f"device-index={int(dev)}", "do-timestamp=true", "!"]
        # Fallback for advanced/manual values on macOS.
        return ["osxaudiosrc", f"device={dev}", "do-timestamp=true", "!"]

    return ["alsasrc", f"device={dev}", "do-timestamp=true", "!"]

