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

### 2) Install Git LFS (required for model weights)
This project uses Git LFS for large model files (`*.pt`), including files in `models/`.

If `git lfs` is not installed:

```bash
# macOS (Homebrew)
brew install git-lfs

# Ubuntu/Jetson
sudo apt update && sudo apt install -y git-lfs
```

Then enable it and fetch model files:

```bash
git lfs install
git lfs pull
```

### 3) Run platform installer
```bash
./setup.sh
```

### 4) Activate virtual environment and start app
```bash
source .venv/bin/activate
python main.py
```

Open:
- Local: `http://localhost:8000`
- Jetson on LAN: `http://<jetson-ip>:8000`

## Quick Start
```bash
git clone https://github.com/rafaelreverberi/ballscope && cd ballscope
git lfs install && git lfs pull
./setup.sh
source .venv/bin/activate
python main.py
```

## Installation Model
Dependencies are split by role:
- `requirements.txt`: shared Python dependencies
- `requirements-mac-apple-silicon.txt`: shared + Apple Silicon PyTorch (MPS)
- `requirements-jetson.txt`: shared dependencies for Jetson (Torch wheel is installed separately)

`setup.sh` does all of this:
- Detects platform (Apple Silicon Mac vs Jetson)
- Creates/uses `.venv`
- Installs platform-specific dependencies
- Verifies key imports and runtime device resolution
- Writes install logs to `logs/setup_*.log`

## Jetson Torch Wheel Requirement
Jetson requires CUDA-enabled Torch wheels that are **not stored in this repository**.

1. Copy placeholder config:
```bash
cp jetson_torch_wheels.example.env jetson_torch_wheels.env
```
2. Fill wheel URLs or local paths in `jetson_torch_wheels.env`
3. Run setup:
```bash
./setup.sh
```

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

## License
MIT License (`LICENSE`)
