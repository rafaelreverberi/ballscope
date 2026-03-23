# Setup Guide

This guide covers installation on Apple Silicon macOS and NVIDIA Jetson.

## 1) Clone Repository
```bash
git clone <repo-url>
cd ballscope
```

## 2) Run Installer

### Apple Silicon Mac
```bash
./setup.sh
```
`setup.sh` installs Homebrew dependencies for BallScope on macOS, including `ffmpeg`, `gstreamer`, common GStreamer plugins, `node`, and `uvcc`.
That means the macOS camera-control stack is prepared automatically by the installer:
- `node` / `npm`
- `uvcc` via `npm install --global uvcc`

`uvcc` is used by the `Camera Settings` workspace for advanced BRIO controls such as:
- HDR / backlight compensation
- auto exposure + manual exposure time
- white balance
- focus
- zoom / pan / tilt

### NVIDIA Jetson (Linux aarch64)
Run:
```bash
sudo ./setup.sh
```
The installer asks for a PyTorch mode before Jetson Python dependencies:
1. Manual install mode: creates/uses `.venv`, then exits so you can install CUDA PyTorch manually.
2. Hugging Face wheel mode: downloads `.whl` files from
   `https://huggingface.co/RafaelReverberi/ballscope-jetson-wheels/tree/main/wheels`
   into `wheels/` and installs them.
3. Preinstalled mode: verifies existing `torch` in `.venv` has CUDA and continues.

If CUDA is unavailable, setup fails with guidance.
When run with `sudo`, setup resets repository ownership back to the invoking user at the end.
The Linux camera-control stack is also installed automatically by `setup.sh`, including `v4l-utils` / `v4l2-ctl`.

### Model Download (both Mac and Jetson)
`setup.sh` downloads all `.pt` files from:
- `https://huggingface.co/RafaelReverberi/ballscope-assets/tree/main/models`
into local:
- `models/`

## 3) Start BallScope
```bash
source .venv/bin/activate
python main.py
```
Or from project root:
```bash
./start.sh
```
Jetson (if camera/device permission issues occur):
```bash
sudo ./start.sh
```

Web UI:
- Local: `http://localhost:8000`
- Jetson on LAN: `http://<jetson-ip>:8000`

## Camera Settings Workspace
Open `Camera Settings` from the home screen to tune both cameras side by side.

Notes:
- macOS uses `uvcc` for advanced UVC controls.
- Jetson uses `v4l2-ctl` (`v4l-utils`), which is installed by `setup.sh`.
- Saved camera settings remain active for the full current BallScope session and are used by recording and live preview pages.

## 4) Optional Device Override
```bash
export BALLSCOPE_AI_DEVICE=cuda
python main.py
```

Default (`BALLSCOPE_AI_DEVICE=auto`) selects:
- Jetson -> `cuda:0`
- Apple Silicon -> `mps`
- fallback -> `cpu`

## 5) Installer Logs
Every setup run writes a log file:
- `logs/setup_YYYYMMDD_HHMMSS.log`

Use this file for debugging failed installs.

## 6) Optional Autostart
At the end of setup, you can choose autostart:
- Jetson: creates/overwrites `ballscope.service` and enables it (`systemd`)
  and installs a `sudoers` rule so BallScope can call the reboot/shutdown API endpoints without running the whole app as root
- macOS: creates/overwrites `com.ballscope.start.plist` (`launchd`)
