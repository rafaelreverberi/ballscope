import os
import time
import signal
import subprocess
import threading
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class GstDualStatus:
    running: bool = False
    output_left: Optional[str] = None
    output_right: Optional[str] = None
    last_error: Optional[str] = None
    started_ts: float = 0.0

    def to_dict(self):
        return asdict(self)


class GstDualRecorder:
    def __init__(self):
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._status = GstDualStatus()
        self._stderr_thread: Optional[threading.Thread] = None

    def status_dict(self):
        with self._lock:
            return self._status.to_dict()

    def is_running(self) -> bool:
        with self._lock:
            return self._status.running

    def start(
        self,
        device_left: str,
        device_right: str,
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
        both_path = os.path.join(output_dir, f"both_{ts}.mkv")

        devL = device_left
        devR = device_right
        if str(devL).isdigit():
            devL = f"/dev/video{int(devL)}"
        if str(devR).isdigit():
            devR = f"/dev/video{int(devR)}"

        if not os.path.exists(devL) or not os.path.exists(devR):
            with self._lock:
                self._status.last_error = f"Device not found: {devL if not os.path.exists(devL) else devR}"
            return self.status_dict()

        pipeline = [
            "gst-launch-1.0",
            "-e",
            "matroskamux",
            "name=mux",
            "!",
            "filesink",
            f"location={both_path}",
            "v4l2src",
            f"device={devL}",
            "do-timestamp=true",
            "!",
            f"image/jpeg,width={w},height={h},framerate={int(fps)}/1",
            "!",
            "jpegparse",
            "!",
            "queue",
            "!",
            "mux.",
            "v4l2src",
            f"device={devR}",
            "do-timestamp=true",
            "!",
            f"image/jpeg,width={w},height={h},framerate={int(fps)}/1",
            "!",
            "jpegparse",
            "!",
            "queue",
            "!",
            "mux.",
        ]

        if audio_device:
            pipeline += [
                "alsasrc",
                f"device={audio_device}",
                "do-timestamp=true",
                "!",
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
            self._status = GstDualStatus(
                running=False,
                output_left=both_path,
                output_right=None,
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
