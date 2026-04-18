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
- Run post-analysis on separate left/right camera videos in the web UI.
- Tune both cameras in a dedicated Camera Settings workspace, keep those settings for the current session, and save named camera presets for later reuse.

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
  - macOS: Homebrew packages including `ffmpeg`, `gstreamer`, common GStreamer plugins, `node`, and `uvcc` (used for recording/audio support and BRIO camera controls)
- Downloads model files (`models/*.pt`, including `models/ballscope-ai.pt`) from Hugging Face:
  - `https://huggingface.co/RafaelReverberi/ballscope-assets/tree/main/models`
- On Jetson: ensures PyTorch CUDA is installed before other Python packages
- Installs both YOLO and RF-DETR analysis dependencies into the local `.venv`
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
  (also installs a `sudoers` rule for BallScope power-control API endpoints: reboot/shutdown)
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

In the `Camera Settings` workspace, you can change source values and save BRIO camera controls for the current app session. You can also store the full left/right camera setup as a named preset and reload it later with one click. Session settings and saved presets are reused by recording and live previews.

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
- `docs/architecture_video_analysis_2026-04-17.md`: detailed redesign plan for the dual-camera offline analysis pipeline

## Camera Settings Workspace
- Open `Camera Settings` from the home screen to tune left and right cameras side by side.
- The page shows a live result preview at the top and both raw camera previews below it.
- BRIO-relevant controls are exposed on both macOS and Jetson, including HDR/backlight, auto exposure, manual exposure, white balance, focus, zoom, pan, and tilt.
- On macOS, advanced controls use `uvcc`.
- On Jetson, advanced controls use `v4l2-ctl`.
- Named camera presets save the full current left/right camera configuration, including source, quality preset, and manual control values.
- Saved camera presets are stored in `camera_presets.json` by default. You can override the path with `BALLSCOPE_CAMERA_PRESET_FILE`.
- In the Camera Settings page, you can enable automatic loading of the last used preset on app startup.

## Troubleshooting
- If setup fails, inspect the latest installer log in `logs/`.
- If a camera cannot be opened on Mac, use numeric sources (`0`, `1`) instead of `/dev/videoX`.
- If Jetson reports no CUDA, verify your Torch wheel matches your JetPack/L4T version.
- If Jetson shows NumPy ABI warnings at startup, rerun `./setup.sh` so Jetson dependencies are re-resolved with `numpy<2`.
- The analysis page now always uses the high-quality dual-camera offline pipeline by default.

## Analysis Workspace
- The `Analysis` page accepts separate `Left Camera` and `Right Camera` uploads.
- Analysis can optionally be limited to the first N minutes of the uploaded files for fast debugging on long recordings.
- BallScope detects whether a model is a YOLO checkpoint or an RF-DETR checkpoint and uses the matching runtime automatically.
- `models/ballscope-ai.pt` is detected as RF-DETR and is the default analysis model.
- Dual-camera analysis is per-camera first: left/right detections stay separate, are fused in master-canvas space, and the final broadcast crop is rendered from the master canvas.
- Full-frame global reacquire scans keep original frame resolution so small-ball detectability is not destroyed by downscaling.
- Short ball losses are handled with bounded prediction and phased camera behavior (`TRACKED`, `HOLD_SHORT`, `LOST_SHORT`, `LOST_LONG`, `UNKNOWN`) so the view widens gradually instead of snapping away.

## License
MIT License (`LICENSE`)
