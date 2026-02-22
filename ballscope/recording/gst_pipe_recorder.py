import os
import time
import signal
import subprocess
import threading
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import cv2
import numpy as np
from .audio_source import gst_audio_source_elements
from .gst_exec import gst_launch_bin

@dataclass
class RecordingStatus:
    running: bool = False
    output_path: Optional[str] = None
    fps: float = 60.0
    frame_count: int = 0
    last_error: Optional[str] = None
    started_ts: float = 0.0

    def to_dict(self):
        return asdict(self)


class GstPipeRecorder:
    def __init__(self):
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._status = RecordingStatus()
        self._frame_size: Optional[Tuple[int, int]] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._stopping = False

    def status_dict(self):
        with self._lock:
            return self._status.to_dict()

    def is_running(self) -> bool:
        with self._lock:
            return self._status.running

    def start(
        self,
        output_dir: str,
        fps: float,
        frame_size: Tuple[int, int],
        audio_device: Optional[str] = None,
        audio_bitrate: int = 64000,
        jpeg_quality: int = 75,
        ts: Optional[str] = None,
    ):
        self.stop()
        os.makedirs(output_dir, exist_ok=True)
        ts = ts or time.strftime("%Y%m%d_%H%M%S")
        use_mkv = bool(audio_device)
        path = os.path.join(output_dir, f"final_{ts}.{'mkv' if use_mkv else 'avi'}")

        w, h = frame_size
        if use_mkv:
            pipeline = [
                gst_launch_bin(),
                "-e",
                "matroskamux",
                "name=mux",
                "!",
                "filesink",
                f"location={path}",
                "fdsrc",
                "do-timestamp=true",
                "!",
                f"video/x-raw,format=BGR,width={w},height={h},framerate={int(fps)}/1",
                "!",
                "videoconvert",
                "!",
                "jpegenc",
                f"quality={int(jpeg_quality)}",
                "!",
                "queue",
                "!",
                "mux.",
            ]
        else:
            pipeline = [
                gst_launch_bin(),
                "-e",
                "fdsrc",
                "do-timestamp=true",
                "!",
                f"video/x-raw,format=BGR,width={w},height={h},framerate={int(fps)}/1",
                "!",
                "videoconvert",
                "!",
                "jpegenc",
                f"quality={int(jpeg_quality)}",
                "!",
                "avimux",
                "!",
                "filesink",
                f"location={path}",
            ]

        if audio_device:
            pipeline += [
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

        with self._lock:
            self._status = RecordingStatus(
                running=False,
                output_path=path,
                fps=fps,
                frame_count=0,
                last_error=None,
                started_ts=time.time(),
            )
            self._frame_size = frame_size

        try:
            proc = subprocess.Popen(
                pipeline,
                stdin=subprocess.PIPE,
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
        if proc.poll() is not None or proc.stdin is None:
            with self._lock:
                self._status.last_error = self._status.last_error or "GStreamer exited early."
            return self.status_dict()

        with self._lock:
            self._proc = proc
            self._status.running = True
        return self.status_dict()

    def write(self, frame):
        with self._lock:
            if not self._status.running:
                return
            proc = self._proc
            frame_size = self._frame_size
            if proc is None or proc.stdin is None:
                return

        if frame_size and (frame.shape[1], frame.shape[0]) != frame_size:
            frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_LINEAR)

        try:
            data = np.ascontiguousarray(frame).tobytes()
            proc.stdin.write(data)
        except Exception:
            with self._lock:
                if self._stopping or not self._status.running:
                    return
                if proc.poll() is not None:
                    self._status.last_error = "GStreamer exited."
                else:
                    self._status.last_error = "GStreamer write failed."
                self._status.running = False
            return

        with self._lock:
            self._status.frame_count += 1

    def stop(self):
        with self._lock:
            proc = self._proc
            self._proc = None
            self._status.running = False
            self._stopping = True

        if proc is None:
            return

        try:
            if proc.stdin:
                proc.stdin.close()
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=2.0)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass
        with self._lock:
            self._stopping = False
