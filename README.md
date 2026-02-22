# BallScope

BallScope is an AI-assisted multi-camera football tracking project. It captures live video from two cameras, runs YOLO-based detection, selects the best camera view, and serves preview + recording controls through a web interface.

## Project Context
- School project (Grade 9, upper secondary / Sek II)
- Region: Wasseramt Ost
- Authors: Rafael Reverberi, Benjamin Flury

## What BallScope Can Do
- Run two live camera feeds in parallel.
- Track the ball (or any configured class) with YOLO.
- Auto-switch to the camera with the better detection.
- Apply smoothed crop/zoom on the selected camera.
- Stream live MJPEG preview in the browser.
- Record processed output and raw camera streams.
- Analyze uploaded videos in the web UI.

## Supported Platforms
- Apple Silicon macOS (`arm64`, M1/M2/M3/M4)
- NVIDIA Jetson (`Linux aarch64`, e.g. Orin Nano)

Not supported:
- Intel macOS (`x86_64`)

## Project Setup (from `git clone`)
### 1) Clone repository
```bash
git clone https://github.com/rafaelreverberi/ballscope
cd ballscope
```

### 2) Run platform installer
```bash
./setup.sh
```
Jetson note:
```bash
sudo ./setup.sh
```
(`setup.sh` normalizes repository ownership back to your user at the end when run via `sudo`.)

### 3) Activate virtual environment and start app
```bash
source .venv/bin/activate
python main.py
```
Or from project root:
```bash
./start.sh
```
Jetson note (if device permissions require it):
```bash
sudo ./start.sh
```

Open:
- Local: `http://localhost:8000`
- Jetson on LAN: `http://<jetson-ip>:8000`

## Quick Start
```bash
git clone https://github.com/rafaelreverberi/ballscope && cd ballscope
./setup.sh
./start.sh
```

## Installation Model
Dependencies are split by role:
- `requirements.txt`: shared Python dependencies
- `requirements-mac-apple-silicon.txt`: shared + Apple Silicon PyTorch (MPS)
- `requirements-jetson.txt`: shared dependencies for Jetson (installed after PyTorch is ready)

`setup.sh` does all of this:
- Detects platform (Apple Silicon Mac vs Jetson)
- Creates/uses `.venv`
- Installs platform-specific dependencies
  - macOS: Homebrew packages including `ffmpeg`, `gstreamer` and common GStreamer plugins (used for recording/audio device support)
- Downloads model files (`models/*.pt`) from Hugging Face:
  - `https://huggingface.co/RafaelReverberi/ballscope-assets/tree/main/models`
- On Jetson: ensures PyTorch CUDA is installed before other Python packages
- At the end: optional autostart setup (`systemd` on Jetson, `launchd` on macOS)
- If run with `sudo`: resets repository ownership back to the invoking user
- Verifies key imports and runtime device resolution
- Writes install logs to `logs/setup_*.log`

## Jetson PyTorch Flow
On Jetson, `setup.sh` asks how to provide PyTorch first:
- Manual mode: setup stops after `.venv` so you can install CUDA PyTorch manually.
- Hugging Face wheel mode: downloads `.whl` files from
  `https://huggingface.co/RafaelReverberi/ballscope-jetson-wheels/tree/main/wheels`
  into local `wheels/` and installs them.
- Preinstalled mode: verifies existing PyTorch in `.venv` has CUDA enabled and continues.

## Optional Autostart
At the end of `setup.sh`, you can enable autostart:
- Jetson: installs/overwrites `ballscope.service` (`systemd`) and enables it
- macOS: installs/overwrites `com.ballscope.start.plist` (`launchd`)

## Runtime Device Behavior
Default is `BALLSCOPE_AI_DEVICE=auto`.

When `auto` is used, BallScope resolves devices as follows:
- Jetson: prefer `cuda:0`
- Apple Silicon Mac: prefer `mps`
- fallback: `cpu`

You can override manually:
```bash
export BALLSCOPE_AI_DEVICE=cpu
python main.py
```

## Camera Source Defaults
- Jetson default sources: `/dev/video0`, `/dev/video2`
- Mac default sources: `0`, `1`

In the UI (Camera Tuning), you can change source values at runtime.

## Main Components
- `main.py`: app entrypoint
- `ballscope/web/app.py`: FastAPI app + frontend UI
- `ballscope/camera/`: camera workers / capture loop
- `ballscope/ai/`: detection + camera switching logic
- `ballscope/recording/`: recording pipelines
- `ballscope/runtime_device.py`: platform/device resolution helpers

## Documentation
- `docs/setup.md`: full setup instructions
- `docs/jetson_notes.md`: Jetson-specific notes
- `docs/architecture.md`: architecture and data flow

## Troubleshooting
- If setup fails, inspect the latest installer log in `logs/`.
- If a camera cannot be opened on Mac, use numeric sources (`0`, `1`) instead of `/dev/videoX`.
- If Jetson reports no CUDA, verify your Torch wheel matches your JetPack/L4T version.
- If Jetson shows NumPy ABI warnings at startup, rerun `./setup.sh` so Jetson dependencies are re-resolved with `numpy<2`.
- In the analysis page, enable `Speed Up` for faster processing (reduced preview overhead, lower inference size, and detection every N frames).

## License
MIT License (`LICENSE`)
