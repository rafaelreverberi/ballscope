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
  - FastAPI endpoints, frontend views, recording/live workspaces, camera settings workspace, and analysis APIs.
- `ballscope/runtime_device.py`
  - Platform and accelerator resolution (`cuda`, `mps`, `cpu`).

## Data Flow
1. Camera workers acquire frames from two camera sources.
2. AI worker processes downscaled frames for detection.
3. Worker scores detections and selects active camera.
4. Crop/zoom is applied on full-resolution selected frame.
5. Final frame is streamed (MJPEG) and optionally recorded.

## Analysis Flow
- The `Analysis` workspace can ingest either one video file or separate left/right camera files.
- Offline analysis uses the same shared runtime-device resolution strategy (`auto` -> Jetson CUDA, Apple Silicon MPS, fallback CPU).
- Model loading is backend-aware: YOLO checkpoints use `ultralytics`, RF-DETR checkpoints use `rfdetr`.
- Dual-camera post-analysis now uses a per-camera-first pipeline:
  1. left/right videos are detected independently
  2. detections are mapped into master-canvas coordinates
  3. hypotheses are fused into one shared ball state
  4. a virtual broadcast camera renders the final output from the master canvas
- The Analysis UI can now store session-scoped manual stitch overrides for overlap, seam blend width, and left/right top crop alignment; those values feed the same shared master-canvas backend instead of a separate render path.
- Full-frame reacquire scans preserve original frame resolution. Only ROI follow-up detection is allowed to use a reduced inference size.
- Missing-ball handling is stateful: `UNKNOWN` -> `TRACKED` -> `HOLD_SHORT` -> `LOST_SHORT` -> `LOST_LONG`, with the virtual camera widening gradually instead of snapping to center.
- The detailed design direction and validation notes stay documented in `docs/architecture_video_analysis_2026-04-17.md`.

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
- Session-wide camera settings are applied through one shared state object for live preview and recording.
- Named camera presets persist complete left/right camera configurations in a shared JSON store and can be reapplied through the same settings API on macOS and Jetson.
- When enabled by the user, the last used camera preset is applied during app startup before camera workers begin opening devices.
- Graceful worker start/stop lifecycle in app lifespan.
- Recording start/stop endpoints are idempotent.
- Setup and environment diagnostics are logged by installer.

## Project Context
- School project (Grade 9 upper secondary, Wasseramt Ost)
- Authors: Rafael Reverberi, Benjamin Flury
