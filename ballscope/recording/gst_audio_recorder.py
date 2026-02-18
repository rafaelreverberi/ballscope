import os
import time
import signal
import subprocess
import threading
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class AudioRecordingStatus:
    running: bool = False
    output_path: Optional[str] = None
    last_error: Optional[str] = None
    started_ts: float = 0.0

    def to_dict(self):
        return asdict(self)


class GstAudioRecorder:
    def __init__(self, name: str = "audio"):
        self._name = name
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._status = AudioRecordingStatus()
        self._stderr_thread: Optional[threading.Thread] = None

    def status_dict(self):
        with self._lock:
            return self._status.to_dict()

    def is_running(self) -> bool:
        with self._lock:
            return self._status.running

    def start(self, output_dir: str, audio_device: str, audio_bitrate: int = 128000, ts: Optional[str] = None):
        self.stop()
        os.makedirs(output_dir, exist_ok=True)
        ts = ts or time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(output_dir, f"{self._name}_{ts}.mp3")

        pipeline = [
            "gst-launch-1.0",
            "-e",
            "alsasrc",
            f"device={audio_device}",
            "do-timestamp=true",
            "!",
            "audioconvert",
            "!",
            "audioresample",
            "!",
            "lamemp3enc",
            f"bitrate={int(audio_bitrate)}",
            "!",
            "filesink",
            f"location={path}",
        ]

        with self._lock:
            self._status = AudioRecordingStatus(
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
