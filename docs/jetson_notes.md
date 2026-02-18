# Jetson Orin Nano Notes

Quick operational notes for BallScope on Jetson.

## Installation
From repo root:
```bash
cp jetson_torch_wheels.example.env jetson_torch_wheels.env
# Fill wheel URLs/paths in the file
./setup.sh
```

The installer will:
- create/use `.venv`
- install shared Python dependencies
- install Jetson system dependencies (`ffmpeg`, `v4l-utils`, `gstreamer`)
- validate imports and report runtime device status

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

Open in browser:
- `http://<jetson-ip>:8000`

## Recording Notes
- Final output: OpenCV MJPG/frame pipeline
- Raw camera capture: GStreamer MJPEG (AVI)

## Troubleshooting
- If CUDA is not used, verify Torch wheel compatibility with your JetPack/L4T version.
- If cameras fail to open, confirm source mapping with `v4l2-ctl`.
- Check installer logs in `logs/`.
