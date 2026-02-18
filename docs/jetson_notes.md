# Jetson Orin Nano Notes

Quick setup tips for dual Logitech Brio capture and ball tracking.

## Packages
- `sudo apt-get install -y v4l-utils gstreamer1.0-tools gstreamer1.0-plugins-good`

## Device Mapping
- Check devices: `v4l2-ctl --list-devices`
- Example: left `/dev/video0`, right `/dev/video4`
- Update in Web UI -> Camera Tuning -> Source (Device)

## Recommended Defaults
- Preset: `1080p60`
- Record FPS: `60`
- AI model: `models/football-ball-detection.pt`
- Class ID: `0` (ball)
- Detect every: `2-5` for higher FPS if needed

## Run
```
python main.py
```
Then open `http://<jetson-ip>:8000` in the browser.

## Recording
- Final output uses OpenCV MJPG/frames.
- Raw camera capture uses GStreamer MJPEG (AVI).
