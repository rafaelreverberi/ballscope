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

### NVIDIA Jetson (Linux aarch64)
1. Prepare wheel config:
```bash
cp jetson_torch_wheels.example.env jetson_torch_wheels.env
```
2. Fill `TORCH_WHEEL_URL` (or `TORCH_WHEEL_PATH`) in `jetson_torch_wheels.env`
3. Optional: fill torchvision wheel values
4. Run installer:
```bash
./setup.sh
```

Note: CUDA Torch wheels are not bundled in this repository.

## 3) Start BallScope
```bash
source .venv/bin/activate
python main.py
```

Web UI:
- Local: `http://localhost:8000`
- Jetson on LAN: `http://<jetson-ip>:8000`

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
