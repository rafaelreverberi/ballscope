import os
import time
import signal
import subprocess
import threading
from dataclasses import dataclass, asdict
from typing import Optional

from .audio_source import gst_audio_source_elements
from .gst_exec import gst_launch_bin
from .video_source import gst_camera_source_elements

@dataclass
class GstRecordingStatus:
    running: bool = False
    output_path: Optional[str] = None
    last_error: Optional[str] = None
    started_ts: float = 0.0

    def to_dict(self):
        return asdict(self)


class GstDeviceRecorder:
    def __init__(self, name: str):
        self._name = name
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._status = GstRecordingStatus()
        self._stderr_thread: Optional[threading.Thread] = None

    def status_dict(self):
        with self._lock:
            return self._status.to_dict()

    def is_running(self) -> bool:
        with self._lock:
            return self._status.running

    def start(
        self,
        device: str,
        output_dir: str,
        w: int,
        h: int,
        fps: int,
        audio_device: Optional[str] = None,
        audio_bitrate: int = 64000,
        ts: Optional[str] = None,
    ):
        self.stop()
        os.makedirs(output_dir, exist_ok=True)
        ts = ts or time.strftime("%Y%m%d_%H%M%S")
        ext = "mkv" if audio_device else "avi"
        path = os.path.join(output_dir, f"{self._name}_{ts}.{ext}")
        cam_src, cam_err = gst_camera_source_elements(device, w, h, fps)
        if cam_src is None:
            with self._lock:
                self._status.last_error = cam_err or "Camera source error"
            return self.status_dict()

        if audio_device:
            pipeline = [
                gst_launch_bin(),
                "-e",
                "matroskamux",
                "name=mux",
                "!",
                "filesink",
                f"location={path}",
                *cam_src,
                "queue",
                "!",
                "mux.",
                *gst_audio_source_elements(audio_device),
                "audioconvert",
                "!",
                "audioresample",
                "!",
                "opusenc",
                f"bitrate={int(audio_bitrate)}",
                "!",
                "queue",
                "!",
                "mux.",
            ]
        else:
            pipeline = [
                gst_launch_bin(),
                "-e",
                *cam_src,
                "avimux",
                "!",
                "filesink",
                f"location={path}",
            ]

        with self._lock:
            self._status = GstRecordingStatus(
                running=False,
                output_path=path,
                last_error=None,
                started_ts=time.time(),
            )

        try:
            proc = subprocess.Popen(
                pipeline,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            with self._lock:
                self._status.last_error = "gst-launch-1.0 not found."
            return self.status_dict()
        except Exception as exc:
            with self._lock:
                self._status.last_error = f"GStreamer start error: {exc}"
            return self.status_dict()

        def _read_stderr(p: subprocess.Popen):
            try:
                if p.stderr is None:
                    return
                lines = []
                for _ in range(20):
                    line = p.stderr.readline()
                    if not line:
                        break
                    try:
                        lines.append(line.decode(errors="ignore").strip())
                    except Exception:
                        pass
                if lines:
                    with self._lock:
                        self._status.last_error = "; ".join(lines[-3:])
            except Exception:
                pass

        self._stderr_thread = threading.Thread(target=_read_stderr, args=(proc,), daemon=True)
        self._stderr_thread.start()

        time.sleep(0.2)
        if proc.poll() is not None:
            with self._lock:
                self._status.last_error = self._status.last_error or "GStreamer exited early."
                self._status.running = False
            return self.status_dict()

        with self._lock:
            self._proc = proc
            self._status.running = True
        return self.status_dict()

    def stop(self):
        with self._lock:
            proc = self._proc
            self._proc = None
            self._status.running = False

        if proc is None:
            return

        try:
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=2.0)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass
