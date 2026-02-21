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
Run:
```bash
./setup.sh
```
The installer asks for a PyTorch mode before Jetson Python dependencies:
1. Manual install mode: creates/uses `.venv`, then exits so you can install CUDA PyTorch manually.
2. Hugging Face wheel mode: downloads `.whl` files from
   `https://huggingface.co/RafaelReverberi/ballscope-jetson-wheels/tree/main/wheels`
   into `wheels/` and installs them.
3. Preinstalled mode: verifies existing `torch` in `.venv` has CUDA and continues.

If CUDA is unavailable, setup fails with guidance.

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
