# Jetson Orin Nano Notes

Quick operational notes for BallScope on Jetson.

## Installation
From repo root:
```bash
sudo ./setup.sh
```

The installer will:
- create/use `.venv`
- ask how to provide PyTorch first (manual, Hugging Face wheels, or preinstalled)
- verify CUDA availability when continuing
- install shared Python dependencies
- install Jetson system dependencies (`ffmpeg`, `v4l-utils`, `gstreamer`)
- download model files to `models/` from Hugging Face assets
- validate imports and report runtime device status
- optionally install/overwrite a `systemd` autostart service at the end

When setup is run with `sudo`, it normalizes repository ownership back to your user at the end.

## Camera Mapping
List V4L2 devices:
```bash
v4l2-ctl --list-devices
```

Typical mapping example:
- Left camera: `/dev/video0`
- Right camera: `/dev/video2` (or `/dev/video4` depending on hardware)

Update sources in Web UI:
- `Camera Tuning` -> `Source (Device)`

## Recommended Defaults
- Preset: `1080p60`
- Record FPS: `60`
- AI model: `models/football-ball-detection.pt`
- Class ID: `0` (ball)
- Detect every: `2-5` if you need higher FPS

## Run
```bash
source .venv/bin/activate
python main.py
```
Or from project root:
```bash
./start.sh
```

Open in browser:
- `http://<jetson-ip>:8000`

## Recording Notes
- Final output: OpenCV MJPG/frame pipeline
- Raw camera capture: GStreamer MJPEG (AVI)

## Troubleshooting
- If CUDA is not used, verify Torch wheel compatibility with your JetPack/L4T version.
- If you see NumPy ABI warnings on startup, recreate the venv with `./setup.sh` and keep Jetson on `numpy<2` (already enforced by `requirements-jetson.txt`).
- If cameras fail to open, confirm source mapping with `v4l2-ctl`.
- Check installer logs in `logs/`.
