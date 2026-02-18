# BallScope Architecture

## Scope
BallScope is designed to run the same application logic on:
- Apple Silicon Mac (development and analysis)
- NVIDIA Jetson (capture and deployment)

## Goals
- Low-latency dual-camera capture.
- Reliable object detection on live streams.
- Automatic camera selection based on detection confidence.
- Stable zoom/crop output for preview and recording.
- Safe recovery from camera disconnects and runtime failures.

## Core Modules
- `ballscope/camera/`
  - Camera workers for capture, buffering, and stream state.
- `ballscope/ai/`
  - Detection + switching logic (`PersonSwitcherWorker`, YOLO-driven scoring).
- `ballscope/recording/`
  - Processed output recording and raw stream recorders.
- `ballscope/web/`
  - FastAPI endpoints, frontend views, and analysis APIs.
- `ballscope/runtime_device.py`
  - Platform and accelerator resolution (`cuda`, `mps`, `cpu`).

## Data Flow
1. Camera workers acquire frames from two camera sources.
2. AI worker processes downscaled frames for detection.
3. Worker scores detections and selects active camera.
4. Crop/zoom is applied on full-resolution selected frame.
5. Final frame is streamed (MJPEG) and optionally recorded.

## Runtime Compatibility Strategy
BallScope keeps behavior aligned across Mac and Jetson by:
- using one shared application codebase
- applying platform-specific capture backends when needed
- resolving compute device automatically via `BALLSCOPE_AI_DEVICE=auto`

Auto device resolution:
- Jetson: `cuda:0` when available
- Apple Silicon: `mps` when available
- fallback: `cpu`

## Operational Reliability
- Camera reopen on source or preset changes.
- Graceful worker start/stop lifecycle in app lifespan.
- Recording start/stop endpoints are idempotent.
- Setup and environment diagnostics are logged by installer.

## Project Context
- School project (Grade 9 upper secondary, Wasseramt Ost)
- Authors: Rafael Reverberi, Benjamin Flury
