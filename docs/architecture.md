BallScope Architecture (Jetson Orin Nano)

Goals
- Low-latency dual-camera capture (RAW frames) with AI-driven camera selection.
- YOLO person detection on downscaled frames, crop/zoom on full-resolution frames.
- Single final stream for preview and recording (MJPG/frames).
- Robust handling of disconnects and safe restart of capture pipelines.

Modules
- camera/: CameraWorker handles V4L2 capture, low-latency buffering, and MJPEG preview. It exposes the latest BGR frame for AI and status metrics (fps, resolution, errors).
- ai/: PersonSwitcherWorker runs YOLO inference on both camera frames (downscaled), selects the active camera, applies ROI smoothing and zoom on the original frame, and publishes a final MJPEG stream.
- recording/: SimpleRecorder writes only the final output stream (MJPG AVI or JPG frame sequence).
- web/: FastAPI app with REST endpoints for state, camera settings, recording control, and AI control. A single-page UI displays the final preview plus camera thumbnails and exposes configuration.

Data Flow
1) CameraWorker threads grab frames from /dev/video* using V4L2.
2) AI worker reads both frames, downsamples for YOLO, then selects the most relevant camera.
3) Crop/zoom happens on the original frame from the chosen camera.
4) Final output is streamed via MJPEG and optionally saved via MJPG AVI or JPG frames.

Recording Pipeline (MJPG / Frames)
- MJPG: OpenCV VideoWriter, AVI container, CPU-friendly for Jetson Orin Nano (no NVENC).
- Frames: individual JPGs for debug-friendly post-processing (ffmpeg assembly offline).

Error Handling
- CameraWorker reopens the device safely if read fails or settings change.
- Recording control is idempotent (start/stop returns status).
- AI worker can be started/stopped without restarting the web server.

Configuration
- Environment variables control camera devices, presets, AI model, and recording defaults.
- Runtime configuration exposed via REST; changes take effect with safe reopen where needed.
