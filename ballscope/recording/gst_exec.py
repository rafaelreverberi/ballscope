import os
import shutil


def gst_launch_bin() -> str:
    """
    Resolve gst-launch across Linux and macOS (including Homebrew paths not in PATH).
    Returns the command name if not found so callers keep existing FileNotFound handling.
    """
    found = shutil.which("gst-launch-1.0")
    if found:
        return found

    candidates = [
        "/opt/homebrew/bin/gst-launch-1.0",  # Apple Silicon Homebrew
        "/usr/local/bin/gst-launch-1.0",     # Intel Homebrew / custom installs
    ]
    for path in candidates:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return "gst-launch-1.0"

