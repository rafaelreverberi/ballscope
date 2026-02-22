import os
import time
import re
import json
import shutil
import platform
import subprocess
import threading
import uuid
from collections import defaultdict, deque
from pathlib import Path
from dataclasses import asdict
from typing import Dict, Optional, List
from contextlib import asynccontextmanager

import cv2
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse

from ballscope.config import (
    QUALITY_PRESETS,
    DEFAULT_PRESET,
    DEFAULT_CAM_LEFT,
    DEFAULT_CAM_RIGHT,
    DEFAULT_VIDEO_DIR,
    DEFAULT_RECORD_FPS,
    AiConfig,
)
from ballscope.camera import CameraState, CameraSettings, CameraWorker, mjpeg_stream, camera_state_dict
from ballscope.ai import PersonSwitcherWorker, ai_mjpeg_stream
from ballscope.recording import GstPipeRecorder, GstDeviceRecorder, GstDualRecorder, GstAudioRecorder
from ballscope.runtime_device import resolve_torch_device, runtime_device_options


@asynccontextmanager
async def lifespan(app: FastAPI):
    for worker in CAMERA_WORKERS.values():
        worker.start()
    AI_WORKER.start()
    AI_WORKER.stop()
    yield
    for worker in CAMERA_WORKERS.values():
        worker.stop()
    AI_WORKER.shutdown()
    RECORDER.stop()
    for recorder in RAW_RECORDERS.values():
        recorder.stop()
    RAW_DUAL.stop()
    AUDIO_RECORDER.stop()


app = FastAPI(title="BallScope", lifespan=lifespan)

CAMERA_STATES: Dict[str, CameraState] = {
    "camL": CameraState(name="Cam Left", src=DEFAULT_CAM_LEFT, settings=CameraSettings()),
    "camR": CameraState(name="Cam Right", src=DEFAULT_CAM_RIGHT, settings=CameraSettings()),
}

CAMERA_WORKERS: Dict[str, CameraWorker] = {cid: CameraWorker(state) for cid, state in CAMERA_STATES.items()}

AI_WORKER = PersonSwitcherWorker(AiConfig())
AI_WORKER.attach_sources(
    {
        "camL": CAMERA_WORKERS["camL"].get_latest_frame_bgr,
        "camR": CAMERA_WORKERS["camR"].get_latest_frame_bgr,
    }
)

RECORDER = GstPipeRecorder()
AI_WORKER.set_recorder(RECORDER)

RAW_RECORDERS: Dict[str, GstDeviceRecorder] = {
    "camL": GstDeviceRecorder("camL"),
    "camR": GstDeviceRecorder("camR"),
}

RAW_DUAL = GstDualRecorder()
AUDIO_RECORDER = GstAudioRecorder("audio")
RAW_SYNC = None

ANALYSIS_UPLOAD_DIR = "uploads"
ANALYSIS_RESULT_DIR = "recordings"
ANALYSIS_SESSIONS: Dict[str, dict] = {}
ANALYSIS_LOCK = threading.Lock()
ANALYSIS_MODELS: Dict[str, object] = {}
ANALYSIS_MODEL_LOCK = threading.Lock()
ANALYSIS_DEFAULT_MODEL = "models/football-ball-detection.pt"
ANALYSIS_TARGET_FPS = 30
ANALYSIS_ENCODE_CRF = 14
ANALYSIS_ENCODE_PRESET = "slow"
ANALYSIS_ENCODE_PRESET_SPEED_UP = "medium"


def _analysis_load_model(model_path: str):
    with ANALYSIS_MODEL_LOCK:
        if model_path in ANALYSIS_MODELS:
            return ANALYSIS_MODELS[model_path]
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError(f"Ultralytics import failed: {exc}")
        ANALYSIS_MODELS[model_path] = YOLO(model_path)
        return ANALYSIS_MODELS[model_path]


def _analysis_clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _analysis_parse_fps(raw: str) -> float:
    if not raw:
        return 0.0
    if "/" in raw:
        try:
            a, b = raw.split("/", 1)
            den = float(b)
            if den == 0:
                return 0.0
            return float(a) / den
        except Exception:
            return 0.0
    try:
        return float(raw)
    except Exception:
        return 0.0


def _analysis_ffprobe_video_streams(path: str) -> List[dict]:
    if not shutil.which("ffprobe"):
        return []
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        path,
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if res.returncode != 0:
            return []
        data = json.loads(res.stdout or "{}")
        streams = []
        for st in data.get("streams", []):
            if st.get("codec_type") != "video":
                continue
            streams.append(
                {
                    "index": int(st.get("index", 0)),
                    "width": int(st.get("width") or 0),
                    "height": int(st.get("height") or 0),
                    "avg_fps": _analysis_parse_fps(st.get("avg_frame_rate") or "0/1"),
                }
            )
        return streams
    except Exception:
        return []


def _analysis_extract_video_stream(input_path: str, stream_idx: int, output_path: str) -> bool:
    if not shutil.which("ffmpeg"):
        return False
    fast_copy = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-map",
        f"0:{stream_idx}",
        "-an",
        "-c:v",
        "copy",
        output_path,
    ]
    res = subprocess.run(fast_copy, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    if res.returncode == 0 and os.path.exists(output_path):
        return True
    fallback = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-map",
        f"0:{stream_idx}",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        output_path,
    ]
    res2 = subprocess.run(fallback, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    return res2.returncode == 0 and os.path.exists(output_path)


def _analysis_list_model_files() -> List[str]:
    root = Path("models")
    if not root.exists():
        return [ANALYSIS_DEFAULT_MODEL]
    exts = {".pt", ".onnx", ".engine"}
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(str(p.as_posix()))
    files = sorted(set(files))
    if ANALYSIS_DEFAULT_MODEL not in files:
        files.insert(0, ANALYSIS_DEFAULT_MODEL)
    return files


def _analysis_model_classes(model_path: str) -> List[dict]:
    try:
        model = _analysis_load_model(model_path)
    except Exception:
        return []
    names = getattr(model, "names", None)
    if names is None and hasattr(model, "model"):
        names = getattr(model.model, "names", None)
    out = []
    if isinstance(names, dict):
        for k in sorted(names.keys()):
            out.append({"id": int(k), "label": str(names[k])})
    elif isinstance(names, list):
        for i, n in enumerate(names):
            out.append({"id": int(i), "label": str(n)})
    return out


def _analysis_detect_runtime_devices() -> List[dict]:
    devices = [{"id": "auto", "label": "Auto"}]
    try:
        import torch  # type: ignore
        devices = runtime_device_options(torch)
        return devices
    except Exception:
        devices = [{"id": "auto", "label": "Auto"}, {"id": "cpu", "label": "CPU"}]
        return devices


def _analysis_resolve_device(device_pref: str) -> str:
    pref = (device_pref or "auto")
    try:
        import torch  # type: ignore
        return resolve_torch_device(pref, torch)
    except Exception:
        return "cpu"
    return "cpu"


def _clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _parse_preview_profile(request: Request) -> dict:
    q = request.query_params
    profile = (q.get("profile") or "balanced").strip().lower()
    presets = {
        "high": {"max_w": 0, "fps": 24, "jpeg_q": 84},
        "balanced": {"max_w": 960, "fps": 15, "jpeg_q": 76},
        "fast": {"max_w": 640, "fps": 10, "jpeg_q": 66},
    }
    cfg = dict(presets.get(profile, presets["balanced"]))
    if q.get("max_w"):
        try:
            cfg["max_w"] = _clamp_int(int(q.get("max_w")), 0, 1920)
        except Exception:
            pass
    if q.get("fps"):
        try:
            cfg["fps"] = _clamp_int(int(float(q.get("fps"))), 1, 60)
        except Exception:
            pass
    if q.get("q"):
        try:
            cfg["jpeg_q"] = _clamp_int(int(q.get("q")), 30, 95)
        except Exception:
            pass
    return cfg


def _mjpeg_from_frame_getter(frame_getter, seq_getter, wait_for_new, max_w: int, fps: int, jpeg_q: int):
    last_seq = -1
    min_dt = 1.0 / float(max(1, fps))
    last_emit = 0.0
    while True:
        try:
            wait_for_new(last_seq, timeout=1.0)
        except Exception:
            time.sleep(0.05)
            continue

        now = time.time()
        dt = now - last_emit
        if dt < min_dt:
            time.sleep(min_dt - dt)

        frame = None
        try:
            frame = frame_getter()
        except Exception:
            frame = None

        if frame is None:
            time.sleep(0.02)
            continue

        frame_seq = None
        try:
            _jpeg_unused, frame_seq = seq_getter()
        except Exception:
            frame_seq = None

        try:
            h, w = frame.shape[:2]
            if max_w and w > max_w:
                scale = max_w / float(max(1, w))
                nh = max(1, int(h * scale))
                frame = cv2.resize(frame, (max_w, nh), interpolation=cv2.INTER_LINEAR)
            ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_q)])
            if not ok:
                time.sleep(0.02)
                continue
            last_emit = time.time()
            if frame_seq is not None:
                last_seq = frame_seq
            chunk = (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
            )
            yield chunk
        except GeneratorExit:
            return
        except Exception:
            time.sleep(0.02)


def _analysis_concat_segments(segment_paths: List[str], stitched_path: str) -> bool:
    if not segment_paths:
        return False
    if len(segment_paths) == 1:
        try:
            shutil.copyfile(segment_paths[0], stitched_path)
            return True
        except Exception:
            return False
    if not shutil.which("ffmpeg"):
        return False
    list_file = stitched_path + ".txt"
    try:
        with open(list_file, "w", encoding="utf-8") as f:
            for p in segment_paths:
                f.write(f"file '{os.path.abspath(p)}'\n")
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_file,
            "-c",
            "copy",
            stitched_path,
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return res.returncode == 0 and os.path.exists(stitched_path)
    finally:
        try:
            os.remove(list_file)
        except Exception:
            pass


def _analysis_process_video(session_id: str, video_path: str):
    with ANALYSIS_LOCK:
        session = ANALYSIS_SESSIONS.get(session_id)
    if not session:
        return

    model_path = session.get("model_path") or ANALYSIS_DEFAULT_MODEL
    class_id = session.get("class_id")
    conf = float(session.get("conf", 0.32))
    iou = float(session.get("iou", 0.35))
    zoom = float(session.get("zoom", 1.6))
    speed_up = bool(session.get("speed_up", False))
    # Keep normal mode unchanged; fast mode trades some temporal density for throughput.
    detect_every = max(1, int(session.get("detect_every", 2 if speed_up else 1)))
    infer_imgsz = max(224, min(1280, int(session.get("imgsz", 384 if speed_up else 640))))
    preview_stride = max(1, int(session.get("preview_stride", 2 if speed_up else 1)))
    allow_switch = bool(session.get("allow_switch", False))
    device_pref = str(session.get("device", "auto"))
    device_use = _analysis_resolve_device(device_pref)
    crop = session.get("crop") or {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}

    with ANALYSIS_LOCK:
        ANALYSIS_SESSIONS[session_id]["running"] = True
        ANALYSIS_SESSIONS[session_id]["state"] = "loading-model"

    try:
        model = _analysis_load_model(model_path)
    except Exception as exc:
        with ANALYSIS_LOCK:
            ANALYSIS_SESSIONS[session_id]["running"] = False
            ANALYSIS_SESSIONS[session_id]["state"] = "error"
            ANALYSIS_SESSIONS[session_id]["last_error"] = str(exc)
        return

    os.makedirs(ANALYSIS_RESULT_DIR, exist_ok=True)
    source_paths: List[str] = [video_path]
    tmp_sources: List[str] = []
    streams = _analysis_ffprobe_video_streams(video_path)
    if len(streams) > 1:
        source_paths = []
        for idx, st in enumerate(streams):
            extracted = os.path.join(ANALYSIS_UPLOAD_DIR, f"{session_id}_stream{idx}.mp4")
            if _analysis_extract_video_stream(video_path, int(st.get("index", idx)), extracted):
                source_paths.append(extracted)
                tmp_sources.append(extracted)
        if not source_paths:
            source_paths = [video_path]

    caps: List[cv2.VideoCapture] = []
    for p in source_paths:
        cap = cv2.VideoCapture(p)
        if cap.isOpened():
            caps.append(cap)
        else:
            cap.release()

    if not caps:
        with ANALYSIS_LOCK:
            ANALYSIS_SESSIONS[session_id]["running"] = False
            ANALYSIS_SESSIONS[session_id]["state"] = "error"
            ANALYSIS_SESSIONS[session_id]["last_error"] = "Could not open uploaded video."
        return

    fps_list = [(cap.get(cv2.CAP_PROP_FPS) or ANALYSIS_TARGET_FPS) for cap in caps]
    frame_count_list = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) for cap in caps]
    total_frames = max([n for n in frame_count_list if n > 0] or [0])
    src_fps = max([f for f in fps_list if f > 0] or [ANALYSIS_TARGET_FPS])
    proc_times = deque(maxlen=50)
    last_seen_ts = None
    lost_hold_sec = 1.8
    focus_stream = 0
    smooth_alpha = 0.90
    smooth_cx = None
    smooth_cy = None
    smooth_vx = 0.0
    smooth_vy = 0.0
    max_pan_ratio = 0.05
    out_size = None
    writer = None
    segment_paths: List[str] = []
    segment_frames = max(120, int(src_fps * 8))
    segment_frame_count = 0
    segment_index = 0
    frame_idx = 0
    stream_best_cache: Dict[int, tuple] = {}
    stream_last_seen_ts: Dict[int, float] = {}
    fast_cache_hold_sec = 0.35
    fast_track_hold_sec = 0.9
    fast_search_expand = 2.6
    segment_dir = os.path.join(ANALYSIS_RESULT_DIR, f"analysis_{session_id}_parts")
    os.makedirs(segment_dir, exist_ok=True)
    stitched_avi = os.path.join(segment_dir, f"{session_id}_stitched.avi")
    stem = Path(video_path).stem
    ts = time.strftime("%Y%m%d_%H%M%S")
    final_mp4 = os.path.join(ANALYSIS_RESULT_DIR, f"{stem}_analysis_{ts}_{session_id}.mp4")

    with ANALYSIS_LOCK:
        ANALYSIS_SESSIONS[session_id]["state"] = "running"
        ANALYSIS_SESSIONS[session_id]["device_used"] = device_use
        ANALYSIS_SESSIONS[session_id]["input_fps"] = float(src_fps)
        ANALYSIS_SESSIONS[session_id]["total_frames"] = float(total_frames)
        ANALYSIS_SESSIONS[session_id]["progress_pct"] = 0.0
        ANALYSIS_SESSIONS[session_id]["eta_sec"] = None
        ANALYSIS_SESSIONS[session_id]["stream_count"] = len(caps)
        ANALYSIS_SESSIONS[session_id]["speed_up"] = speed_up
        ANALYSIS_SESSIONS[session_id]["detect_every"] = detect_every
        ANALYSIS_SESSIONS[session_id]["imgsz"] = infer_imgsz
        ANALYSIS_SESSIONS[session_id]["output_path"] = None
        ANALYSIS_SESSIONS[session_id]["recovery_dir"] = segment_dir

    def _detect_best_box(roi_img, off_x: int, off_y: int, imgsz_local: int):
        if roi_img is None or roi_img.size == 0:
            return None
        results = model(
            roi_img,
            imgsz=imgsz_local,
            conf=conf,
            iou=iou,
            verbose=False,
            device=predict_device,
            half=use_half,
        )
        best = None
        if results and results[0].boxes:
            for box in results[0].boxes:
                if class_id is not None and int(box.cls[0]) != int(class_id):
                    continue
                score = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                gx1 = off_x + x1
                gy1 = off_y + y1
                gx2 = off_x + x2
                gy2 = off_y + y2
                gcx = (gx1 + gx2) // 2
                gcy = (gy1 + gy2) // 2
                cand = (score, gx1, gy1, gx2, gy2, gcx, gcy)
                if best is None or score > best[0]:
                    best = cand
        return best

    if device_use.startswith("cuda"):
        if device_use in ("cuda", "cuda:0"):
            predict_device = "0"
        elif device_use.startswith("cuda:") and device_use.split(":", 1)[1].isdigit():
            predict_device = device_use.split(":", 1)[1]
        else:
            predict_device = device_use
    else:
        predict_device = device_use
    use_half = device_use.startswith("cuda")

    try:
        while True:
            t0 = time.time()
            frame_idx += 1
            frames = []
            any_open = False
            for cap in caps:
                ok, frame = cap.read()
                if ok and frame is not None:
                    any_open = True
                    frames.append(frame)
                else:
                    frames.append(None)
            if not any_open:
                break

            detected = False
            last_conf = 0.0
            stream_best = {}  # idx -> (score, x1, y1, x2, y2, cx, cy)

            now_ts = time.time()
            for idx, frame in enumerate(frames):
                if frame is None:
                    continue
                h, w = frame.shape[:2]
                rx = int(float(crop.get("x", 0.0)) * w)
                ry = int(float(crop.get("y", 0.0)) * h)
                rw = int(float(crop.get("w", 1.0)) * w)
                rh = int(float(crop.get("h", 1.0)) * h)
                rx = _analysis_clamp(rx, 0, w - 1)
                ry = _analysis_clamp(ry, 0, h - 1)
                rw = _analysis_clamp(rw, 1, w - rx)
                rh = _analysis_clamp(rh, 1, h - ry)
                roi = frame[ry:ry + rh, rx:rx + rw]
                if roi.size == 0:
                    continue

                do_full_scan = (frame_idx % detect_every) == 0
                best_for_stream = None

                if speed_up:
                    # Smart fast-tracking: first search around last known ball area, then periodic full scan.
                    cached = stream_best_cache.get(idx)
                    cached_ts = stream_last_seen_ts.get(idx, 0.0)
                    if cached is not None and (now_ts - cached_ts) <= fast_track_hold_sec:
                        _, bx1, by1, bx2, by2, bcx, bcy = cached
                        bw = max(8, bx2 - bx1)
                        bh = max(8, by2 - by1)
                        sw = max(120, int(bw * fast_search_expand))
                        sh = max(120, int(bh * fast_search_expand))
                        sx1 = _analysis_clamp(int(bcx - sw // 2), rx, rx + rw - 2)
                        sy1 = _analysis_clamp(int(bcy - sh // 2), ry, ry + rh - 2)
                        sx2 = _analysis_clamp(sx1 + sw, sx1 + 1, rx + rw)
                        sy2 = _analysis_clamp(sy1 + sh, sy1 + 1, ry + rh)
                        search_roi = frame[sy1:sy2, sx1:sx2]
                        best_for_stream = _detect_best_box(search_roi, sx1, sy1, max(256, infer_imgsz - 64))

                    if best_for_stream is None and do_full_scan:
                        best_for_stream = _detect_best_box(roi, rx, ry, infer_imgsz)
                else:
                    best_for_stream = _detect_best_box(roi, rx, ry, infer_imgsz)

                if best_for_stream is not None:
                    stream_best_cache[idx] = best_for_stream
                    stream_last_seen_ts[idx] = now_ts
                    stream_best[idx] = best_for_stream
                else:
                    # Short grace reuse only; prevents stale lock-on while still smoothing misses.
                    cached = stream_best_cache.get(idx)
                    cached_ts = stream_last_seen_ts.get(idx, 0.0)
                    if cached is not None and (now_ts - cached_ts) <= fast_cache_hold_sec:
                        stream_best[idx] = cached
                    else:
                        stream_best_cache.pop(idx, None)
                        stream_last_seen_ts.pop(idx, None)

            chosen = None
            if stream_best:
                cur_best = stream_best.get(focus_stream)
                if allow_switch:
                    global_best_stream = max(stream_best.keys(), key=lambda k: stream_best[k][0])
                    global_best = stream_best[global_best_stream]
                    switch_margin = 0.08
                    if cur_best is None:
                        focus_stream = global_best_stream
                        chosen = global_best
                    else:
                        if global_best_stream != focus_stream and global_best[0] > (cur_best[0] + switch_margin):
                            focus_stream = global_best_stream
                            chosen = global_best
                        else:
                            chosen = cur_best
                else:
                    chosen = cur_best

            if chosen is not None:
                detected = True
                last_conf = chosen[0]
                target_cx = chosen[5]
                target_cy = chosen[6]
                if smooth_cx is None:
                    smooth_cx, smooth_cy = target_cx, target_cy
                else:
                    # Damped follow: smooth + limited per-frame pan to avoid hard jumps.
                    want_x = smooth_alpha * smooth_cx + (1.0 - smooth_alpha) * target_cx
                    want_y = smooth_alpha * smooth_cy + (1.0 - smooth_alpha) * target_cy
                    dx = want_x - smooth_cx
                    dy = want_y - smooth_cy
                    # Deadzone avoids tiny left-right jitter when the ball is near center.
                    deadzone = max(2.0, min(fw if 'fw' in locals() else 1920, fh if 'fh' in locals() else 1080) * 0.012)
                    if abs(dx) < deadzone:
                        dx = 0.0
                    if abs(dy) < deadzone:
                        dy = 0.0
                    max_pan = max(8.0, min(fw if 'fw' in locals() else 1920, fh if 'fh' in locals() else 1080) * max_pan_ratio)
                    if dx > max_pan:
                        dx = max_pan
                    elif dx < -max_pan:
                        dx = -max_pan
                    if dy > max_pan:
                        dy = max_pan
                    elif dy < -max_pan:
                        dy = -max_pan
                    smooth_vx = 0.75 * smooth_vx + 0.25 * dx
                    smooth_vy = 0.75 * smooth_vy + 0.25 * dy
                    smooth_cx = int(smooth_cx + smooth_vx)
                    smooth_cy = int(smooth_cy + smooth_vy)
                last_seen_ts = time.time()

            source_frame = None
            if focus_stream < len(frames):
                source_frame = frames[focus_stream]
            if source_frame is None:
                for f in frames:
                    if f is not None:
                        source_frame = f
                        break
            if source_frame is None:
                continue

            fh, fw = source_frame.shape[:2]
            rx = int(float(crop.get("x", 0.0)) * fw)
            ry = int(float(crop.get("y", 0.0)) * fh)
            rw = int(float(crop.get("w", 1.0)) * fw)
            rh = int(float(crop.get("h", 1.0)) * fh)
            rx = _analysis_clamp(rx, 0, fw - 1)
            ry = _analysis_clamp(ry, 0, fh - 1)
            rw = _analysis_clamp(rw, 1, fw - rx)
            rh = _analysis_clamp(rh, 1, fh - ry)
            vis = None
            if not speed_up:
                vis = source_frame.copy()
                cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh), (255, 190, 40), 1)
                if chosen is not None:
                    x1, y1, x2, y2 = chosen[1], chosen[2], chosen[3], chosen[4]
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 0), 2)
                    cv2.putText(vis, f"BALL {last_conf:.2f}", (x1, max(16, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)
                elif last_seen_ts and (time.time() - last_seen_ts) <= lost_hold_sec:
                    cv2.putText(vis, "BALL remembered", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                else:
                    cv2.putText(vis, "BALL lost", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(vis, f"Stream {focus_stream + 1}/{len(caps)}", (12, fh - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 220, 255), 2)

            if out_size is None:
                out_size = (fw, fh)
            out_w, out_h = out_size
            if smooth_cx is None:
                smooth_cx = fw // 2
                smooth_cy = fh // 2
            zoom_use = max(1.0, min(4.0, zoom))
            win_w = int(out_w / zoom_use)
            win_h = int(out_h / zoom_use)
            left = _analysis_clamp(int(smooth_cx - win_w // 2), 0, max(0, fw - win_w))
            top = _analysis_clamp(int(smooth_cy - win_h // 2), 0, max(0, fh - win_h))
            crop_frame = source_frame[top:top + win_h, left:left + win_w]
            if crop_frame.size == 0:
                crop_frame = source_frame
            if speed_up:
                interp = cv2.INTER_LINEAR
            else:
                interp = cv2.INTER_LANCZOS4 if zoom_use > 1.0 else cv2.INTER_AREA
            render = cv2.resize(crop_frame, (out_w, out_h), interpolation=interp)
            if not speed_up:
                # Mild unsharp mask to reduce softness from digital zoom upscaling.
                blur = cv2.GaussianBlur(render, (0, 0), 1.0)
                render = cv2.addWeighted(render, 1.18, blur, -0.18, 0)

            if speed_up:
                # Fast path: publish final render directly as preview (skip expensive debug-overlay rendering).
                preview_frame = render
            else:
                # Preview mirrors final zoom framing but keeps debug overlays.
                vis_zoom_src = vis[top:top + win_h, left:left + win_w]
                if vis_zoom_src.size == 0:
                    vis_zoom_src = vis
                preview_frame = cv2.resize(vis_zoom_src, (out_w, out_h), interpolation=interp)
                cv2.putText(preview_frame, "Preview: final zoom + tracking overlay", (12, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 255), 2)

            if writer is None:
                segment_path = os.path.join(segment_dir, f"part_{segment_index:04d}.avi")
                writer = cv2.VideoWriter(
                    segment_path,
                    cv2.VideoWriter_fourcc(*"MJPG"),
                    float(src_fps),
                    (out_w, out_h),
                )
                if writer is not None:
                    segment_paths.append(segment_path)
                    segment_frame_count = 0
                    segment_index += 1
            if writer is not None:
                writer.write(render)
                segment_frame_count += 1
                if segment_frame_count >= segment_frames:
                    writer.release()
                    writer = None

            proc_times.append(time.time() - t0)
            fps_proc = 0.0
            if proc_times:
                fps_proc = 1.0 / max(1e-6, (sum(proc_times) / len(proc_times)))

            with ANALYSIS_LOCK:
                sess = ANALYSIS_SESSIONS.get(session_id)
                if not sess:
                    break
                sess["frames"] = float(sess.get("frames", 0.0) + 1.0)
                sess["detections"] = float(sess.get("detections", 0.0) + (1.0 if detected else 0.0))
                sess["fps"] = fps_proc
                sess["last_conf"] = last_conf
                sess["state"] = "BALL" if detected else ("REMEMBERED" if last_seen_ts and (time.time() - last_seen_ts) <= lost_hold_sec else "LOST")
                sess["last_seen_sec"] = (time.time() - last_seen_ts) if last_seen_ts else None
                if total_frames > 0:
                    done = float(sess["frames"])
                    pct = min(100.0, (done / float(total_frames)) * 100.0)
                    sess["progress_pct"] = pct
                    sess["eta_sec"] = ((float(total_frames) - done) / fps_proc) if fps_proc > 0 and done < total_frames else 0.0
                if (frame_idx % preview_stride) == 0 or sess.get("frame") is None:
                    sess["frame"] = preview_frame

            # Offline analysis: run at full speed, no realtime pacing delay.
    except Exception as exc:
        with ANALYSIS_LOCK:
            sess = ANALYSIS_SESSIONS.get(session_id)
            if sess is not None:
                sess["state"] = "error"
                sess["last_error"] = str(exc)
    finally:
        for cap in caps:
            cap.release()
        if writer is not None:
            writer.release()
        stitched_ok = _analysis_concat_segments(segment_paths, stitched_avi)
        if stitched_ok and shutil.which("ffmpeg"):
            encode_preset = ANALYSIS_ENCODE_PRESET_SPEED_UP if speed_up else ANALYSIS_ENCODE_PRESET
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                stitched_avi,
                "-i",
                video_path,
                "-map",
                "0:v:0",
                "-map",
                "1:a:0?",
                "-c:v",
                "libx264",
                "-preset",
                encode_preset,
                "-crf",
                str(ANALYSIS_ENCODE_CRF),
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-shortest",
                final_mp4,
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        if not os.path.exists(final_mp4) and stitched_ok:
            try:
                shutil.copyfile(stitched_avi, final_mp4)
            except Exception:
                pass
        for p in tmp_sources:
            try:
                os.remove(p)
            except Exception:
                pass

        with ANALYSIS_LOCK:
            sess = ANALYSIS_SESSIONS.get(session_id)
            if sess is not None:
                sess["running"] = False
                if sess.get("state") != "error":
                    sess["state"] = "done"
                sess["progress_pct"] = 100.0 if os.path.exists(final_mp4) else sess.get("progress_pct", 0.0)
                sess["eta_sec"] = 0.0 if os.path.exists(final_mp4) else sess.get("eta_sec")
                sess["output_path"] = final_mp4 if os.path.exists(final_mp4) else None
                sess["segments_saved"] = len(segment_paths)
        print(f"[analysis] job finished sid={session_id} output={final_mp4 if os.path.exists(final_mp4) else 'none'}")


def raw_status_dict() -> Dict[str, dict]:
    if RAW_DUAL.is_running():
        st = RAW_DUAL.status_dict()
        return {
            "camL": {"running": True, "output_path": st.get("output_left"), "last_error": st.get("last_error")},
            "camR": {"running": True, "output_path": st.get("output_left"), "last_error": st.get("last_error")},
        }
    return {cid: rec.status_dict() for cid, rec in RAW_RECORDERS.items()}


def _ffprobe_duration(path: str) -> Optional[float]:
    if not shutil.which("ffprobe"):
        return None
    try:
        res = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nokey=1:noprint_wrappers=1", path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if res.returncode != 0:
            return None
        return float((res.stdout or "").strip())
    except Exception:
        return None


def _trim_to_duration(path: str, duration: float) -> bool:
    if not shutil.which("ffmpeg"):
        return False
    try:
        tmp = path + ".trim"
        res = subprocess.run(
            ["ffmpeg", "-y", "-i", path, "-t", f"{duration}", "-c", "copy", tmp],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if res.returncode != 0:
            return False
        os.replace(tmp, path)
        return True
    except Exception:
        return False


def list_alsa_devices() -> list[dict]:
    devices = [{"id": "", "label": "None"}, {"id": "default", "label": "default"}]
    try:
        res = subprocess.run(["arecord", "-L"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        text = res.stdout or ""
        lines = text.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i].rstrip()
            name = line.strip()
            if not name or name.startswith("#"):
                i += 1
                continue
            if any(ch.isspace() for ch in name):
                i += 1
                continue
            desc = ""
            if i + 1 < len(lines) and lines[i + 1].startswith(" "):
                desc = lines[i + 1].strip()
                i += 1
            if name in ("null", "default"):
                i += 1
                continue
            label = f"{desc} ({name})" if desc else name
            devices.append({"id": name, "label": label})
            i += 1
    except Exception:
        pass
    return devices


def list_macos_audio_devices() -> list[dict]:
    devices = [{"id": "", "label": "None"}, {"id": "default", "label": "default"}]
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return devices

    # AVFoundation device listing is printed to stderr and usually exits with a non-zero code.
    cmd = [ffmpeg, "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        text = (res.stderr or "") + "\n" + (res.stdout or "")
    except Exception:
        return devices

    in_audio_section = False
    seen_ids = set()
    for raw in text.splitlines():
        line = raw.strip()
        low = line.lower()
        if "avfoundation video devices" in low:
            in_audio_section = False
            continue
        if "avfoundation audio devices" in low:
            in_audio_section = True
            continue
        if not in_audio_section:
            continue

        m = re.search(r"\[(\d+)\]\s+(.+)$", line)
        if not m:
            continue
        dev_id = str(int(m.group(1)))
        name = m.group(2).strip()
        if not name or dev_id in seen_ids:
            continue
        seen_ids.add(dev_id)
        devices.append({"id": dev_id, "label": f"{name} (avfoundation:{dev_id})"})
    return devices


def list_audio_devices() -> list[dict]:
    system = platform.system()
    if system == "Darwin":
        return list_macos_audio_devices()
    return list_alsa_devices()


@app.get("/video/cam/{camera_id}.mjpg")
def video_mjpg(camera_id: str):
    if camera_id not in CAMERA_WORKERS:
        raise HTTPException(status_code=404, detail="Unknown camera_id")
    return StreamingResponse(
        mjpeg_stream(CAMERA_WORKERS[camera_id]),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/video/final.mjpg")
def final_preview():
    return StreamingResponse(
        ai_mjpeg_stream(AI_WORKER),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/video/live/cam/{camera_id}.mjpg")
def live_video_mjpg(camera_id: str, request: Request):
    if camera_id not in CAMERA_WORKERS:
        raise HTTPException(status_code=404, detail="Unknown camera_id")
    cfg = _parse_preview_profile(request)
    worker = CAMERA_WORKERS[camera_id]
    return StreamingResponse(
        _mjpeg_from_frame_getter(
            worker.get_latest_frame_bgr,
            worker.get_latest_jpeg_and_seq,
            worker.wait_for_new_frame,
            max_w=cfg["max_w"],
            fps=cfg["fps"],
            jpeg_q=cfg["jpeg_q"],
        ),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/video/live/final.mjpg")
def live_final_preview(request: Request):
    cfg = _parse_preview_profile(request)
    return StreamingResponse(
        _mjpeg_from_frame_getter(
            AI_WORKER.get_latest_frame,
            AI_WORKER.get_latest_jpeg_and_seq,
            AI_WORKER.wait_for_new_frame,
            max_w=cfg["max_w"],
            fps=cfg["fps"],
            jpeg_q=cfg["jpeg_q"],
        ),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )

@app.get("/api/state")
def api_state():
    return JSONResponse(
        {
            "cameras": camera_state_dict(CAMERA_STATES),
            "presets": QUALITY_PRESETS,
            "recording": RECORDER.status_dict(),
            "recording_raw": raw_status_dict(),
            "recording_audio": AUDIO_RECORDER.status_dict(),
            "ai": AI_WORKER.status_dict(),
            "ai_config": asdict(AI_WORKER.config),
            "defaults": {
                "preset": DEFAULT_PRESET,
                "video_dir": DEFAULT_VIDEO_DIR,
                "record_fps": DEFAULT_RECORD_FPS,
            },
        }
    )


@app.get("/api/audio/devices")
def api_audio_devices():
    return {"devices": list_audio_devices()}


@app.post("/api/settings/{camera_id}")
async def set_settings(camera_id: str, request: Request):
    if camera_id not in CAMERA_STATES:
        raise HTTPException(status_code=404, detail="Unknown camera_id")
    body = await request.json()
    st = CAMERA_STATES[camera_id].settings

    preset_changed = False
    src_changed = False
    src = body.get("src")
    if src is not None:
        if str(src) != CAMERA_STATES[camera_id].src:
            CAMERA_STATES[camera_id].src = str(src)
            src_changed = True
    preset = body.get("preset")
    if preset is not None:
        if preset not in QUALITY_PRESETS:
            raise HTTPException(status_code=400, detail="Unknown preset")
        if st.preset != preset:
            st.preset = preset
            preset_changed = True

    for key in ["brightness", "contrast", "saturation", "gain"]:
        if key in body:
            val = body[key]
            if val is None:
                setattr(st, key, None)
            else:
                try:
                    setattr(st, key, int(val))
                except Exception:
                    raise HTTPException(status_code=400, detail=f"Invalid {key}")

    for key in ["auto_wb", "auto_exposure"]:
        if key in body:
            val = body[key]
            if val is None:
                setattr(st, key, None)
            else:
                setattr(st, key, bool(val))

    if preset_changed or src_changed:
        CAMERA_WORKERS[camera_id].request_reopen()
    CAMERA_WORKERS[camera_id].apply_driver_controls_now()

    return {"ok": True, "camera_id": camera_id, "settings": asdict(st)}


@app.post("/api/record/start")
async def start_recording(request: Request):
    body = await request.json()
    output_dir = body.get("output_dir", DEFAULT_VIDEO_DIR)
    fps = float(DEFAULT_RECORD_FPS)
    audio_device = body.get("audio_device") or None
    audio_bitrate = int(body.get("audio_bitrate", 64000))
    container = body.get("container", "mkv")
    jpeg_quality = 75
    ts = time.strftime("%Y%m%d_%H%M%S")

    AI_WORKER.config.record_use_zoom = True

    RECORDER.stop()
    AUDIO_RECORDER.stop()
    if container == "avi" and audio_device:
        AUDIO_RECORDER.start(output_dir=output_dir, audio_device=audio_device, audio_bitrate=audio_bitrate, ts=ts)
        audio_device = None
    status = RECORDER.start(
        output_dir=output_dir,
        fps=fps,
        frame_size=(AI_WORKER.config.output_w, AI_WORKER.config.output_h),
        audio_device=audio_device,
        audio_bitrate=audio_bitrate,
        jpeg_quality=jpeg_quality,
        ts=ts,
    )
    return status


@app.post("/api/record/stop")
def stop_recording():
    RECORDER.stop()
    AUDIO_RECORDER.stop()
    return RECORDER.status_dict()


@app.post("/api/raw/start")
async def start_raw(request: Request):
    body = await request.json()
    output_dir = body.get("output_dir", DEFAULT_VIDEO_DIR)
    which = body.get("which", "camL")
    audio_device = body.get("audio_device") or None
    audio_bitrate = int(body.get("audio_bitrate", 64000))
    container = body.get("container", "mkv")
    ts = time.strftime("%Y%m%d_%H%M%S")

    AI_WORKER.stop()
    RECORDER.stop()
    AUDIO_RECORDER.stop()

    for recorder in RAW_RECORDERS.values():
        recorder.stop()
    RAW_DUAL.stop()

    if which in ("camL", "both"):
        CAMERA_WORKERS["camL"].suspend()
    if which in ("camR", "both"):
        CAMERA_WORKERS["camR"].suspend()

    global RAW_SYNC
    if which == "both":
        stL = CAMERA_STATES["camL"]
        stR = CAMERA_STATES["camR"]
        presetL = QUALITY_PRESETS.get(stL.settings.preset, QUALITY_PRESETS[DEFAULT_PRESET])
        presetR = QUALITY_PRESETS.get(stR.settings.preset, QUALITY_PRESETS[DEFAULT_PRESET])
        w = presetL["w"]
        h = presetL["h"]
        fps = presetL["fps"]
        if (presetR["w"], presetR["h"], presetR["fps"]) != (w, h, fps):
            return {"error": "Both cams must use same preset for sync recording."}
        RAW_SYNC = {"ts": ts, "paths": []}
        if container == "avi" and audio_device:
            AUDIO_RECORDER.start(output_dir=output_dir, audio_device=audio_device, audio_bitrate=audio_bitrate, ts=ts)
            audio_device = None
        status = RAW_DUAL.start(
            device_left=stL.src,
            device_right=stR.src,
            output_dir=output_dir,
            w=w,
            h=h,
            fps=fps,
            audio_device=audio_device,
            audio_bitrate=audio_bitrate,
            ts=ts,
        )
        if status.get("output_left") and status.get("output_right"):
            RAW_SYNC["paths"] = [status["output_left"], status["output_right"]]
        return status

    status = {}
    if which == "camL":
        st = CAMERA_STATES["camL"]
        preset = QUALITY_PRESETS.get(st.settings.preset, QUALITY_PRESETS[DEFAULT_PRESET])
        RAW_SYNC = {"ts": ts, "paths": []}
        if container == "avi" and audio_device:
            AUDIO_RECORDER.start(output_dir=output_dir, audio_device=audio_device, audio_bitrate=audio_bitrate, ts=ts)
            audio_device = None
        status["camL"] = RAW_RECORDERS["camL"].start(
            device=st.src,
            output_dir=output_dir,
            w=preset["w"],
            h=preset["h"],
            fps=preset["fps"],
            audio_device=audio_device,
            audio_bitrate=audio_bitrate,
            ts=ts,
        )
        if status["camL"].get("output_path"):
            RAW_SYNC["paths"].append(status["camL"]["output_path"])
    if which == "camR":
        st = CAMERA_STATES["camR"]
        preset = QUALITY_PRESETS.get(st.settings.preset, QUALITY_PRESETS[DEFAULT_PRESET])
        RAW_SYNC = {"ts": ts, "paths": []}
        if container == "avi" and audio_device:
            AUDIO_RECORDER.start(output_dir=output_dir, audio_device=audio_device, audio_bitrate=audio_bitrate, ts=ts)
            audio_device = None
        status["camR"] = RAW_RECORDERS["camR"].start(
            device=st.src,
            output_dir=output_dir,
            w=preset["w"],
            h=preset["h"],
            fps=preset["fps"],
            audio_device=audio_device,
            audio_bitrate=audio_bitrate,
            ts=ts,
        )
        if status["camR"].get("output_path"):
            RAW_SYNC["paths"].append(status["camR"]["output_path"])

    return status


@app.post("/api/raw/stop")
def stop_raw():
    for recorder in RAW_RECORDERS.values():
        recorder.stop()
    RAW_DUAL.stop()
    AUDIO_RECORDER.stop()
    global RAW_SYNC
    if RAW_SYNC and isinstance(RAW_SYNC, dict):
        paths = RAW_SYNC.get("paths") or []
        if len(paths) >= 2:
            durs = [(p, _ffprobe_duration(p)) for p in paths]
            durs = [(p, d) for p, d in durs if d is not None]
            if len(durs) >= 2:
                min_dur = min(d for _, d in durs)
                for p, _ in durs:
                    _trim_to_duration(p, min_dur)
    RAW_SYNC = None
    for worker in CAMERA_WORKERS.values():
        worker.resume()
    return raw_status_dict()


@app.post("/api/ai/start")
async def start_ai(request: Request):
    body = await request.json()
    active_camera = body.get("active_camera")
    if active_camera == "auto":
        active_camera = None
    if active_camera is not None and active_camera not in CAMERA_WORKERS:
        raise HTTPException(status_code=404, detail="Unknown active_camera")

    for worker in CAMERA_WORKERS.values():
        worker.resume()

    AI_WORKER.set_manual_camera(active_camera)
    AI_WORKER.config.conf = float(body.get("conf", AI_WORKER.config.conf))
    AI_WORKER.config.iou = float(body.get("iou", AI_WORKER.config.iou))
    AI_WORKER.config.device = str(body.get("device", AI_WORKER.config.device))
    AI_WORKER.config.imgsz = int(body.get("imgsz", AI_WORKER.config.imgsz))
    AI_WORKER.config.zoom = float(body.get("zoom", AI_WORKER.config.zoom))
    AI_WORKER.config.smooth = float(body.get("smooth", AI_WORKER.config.smooth))
    AI_WORKER.config.lost_hold_sec = float(body.get("lost_hold_sec", AI_WORKER.config.lost_hold_sec))
    AI_WORKER.config.detect_every = int(body.get("detect_every", AI_WORKER.config.detect_every))
    AI_WORKER.config.model_path = str(body.get("model_path", AI_WORKER.config.model_path))
    AI_WORKER.config.class_id = int(body.get("class_id", AI_WORKER.config.class_id))
    AI_WORKER.config.record_use_zoom = True

    AI_WORKER.start()
    return AI_WORKER.status_dict()


@app.post("/api/ai/stop")
def stop_ai():
    AI_WORKER.stop()
    return AI_WORKER.status_dict()


@app.post("/api/system/start")
async def start_system(request: Request):
    body = await request.json()
    active_camera = body.get("active_camera")
    if active_camera == "auto":
        active_camera = None
    if active_camera is not None and active_camera not in CAMERA_WORKERS:
        raise HTTPException(status_code=404, detail="Unknown active_camera")

    for worker in CAMERA_WORKERS.values():
        worker.resume()

    AI_WORKER.set_manual_camera(active_camera)
    AI_WORKER.config.conf = float(body.get("conf", AI_WORKER.config.conf))
    AI_WORKER.config.iou = float(body.get("iou", AI_WORKER.config.iou))
    AI_WORKER.config.device = str(body.get("device", AI_WORKER.config.device))
    AI_WORKER.config.imgsz = int(body.get("imgsz", AI_WORKER.config.imgsz))
    AI_WORKER.config.zoom = float(body.get("zoom", AI_WORKER.config.zoom))
    AI_WORKER.config.smooth = float(body.get("smooth", AI_WORKER.config.smooth))
    AI_WORKER.config.lost_hold_sec = float(body.get("lost_hold_sec", AI_WORKER.config.lost_hold_sec))
    AI_WORKER.config.detect_every = int(body.get("detect_every", AI_WORKER.config.detect_every))
    AI_WORKER.config.model_path = str(body.get("model_path", AI_WORKER.config.model_path))
    AI_WORKER.config.class_id = int(body.get("class_id", AI_WORKER.config.class_id))
    AI_WORKER.config.record_use_zoom = True
    AI_WORKER.start()

    output_dir = body.get("output_dir", DEFAULT_VIDEO_DIR)
    fps = float(DEFAULT_RECORD_FPS)
    audio_device = body.get("audio_device") or None
    audio_bitrate = int(body.get("audio_bitrate", 64000))
    container = body.get("container", "mkv")
    jpeg_quality = 75
    ts = time.strftime("%Y%m%d_%H%M%S")
    RECORDER.stop()
    AUDIO_RECORDER.stop()
    if container == "avi" and audio_device:
        AUDIO_RECORDER.start(output_dir=output_dir, audio_device=audio_device, audio_bitrate=audio_bitrate, ts=ts)
        audio_device = None

    rec_status = RECORDER.start(
        output_dir=output_dir,
        fps=fps,
        frame_size=(AI_WORKER.config.output_w, AI_WORKER.config.output_h),
        audio_device=audio_device,
        audio_bitrate=audio_bitrate,
        jpeg_quality=jpeg_quality,
        ts=ts,
    )
    return {"ai": AI_WORKER.status_dict(), "recording": rec_status}


@app.post("/api/system/stop")
def stop_system():
    AI_WORKER.stop()
    RECORDER.stop()
    AUDIO_RECORDER.stop()
    for recorder in RAW_RECORDERS.values():
        recorder.stop()
    RAW_DUAL.stop()
    for worker in CAMERA_WORKERS.values():
        worker.resume()
    return {"ai": AI_WORKER.status_dict(), "recording": RECORDER.status_dict()}


def _power_command(action: str) -> List[str]:
    system = platform.system()
    if system == "Darwin":
        if action == "reboot":
            return ["/sbin/shutdown", "-r", "now"]
        if action == "shutdown":
            return ["/sbin/shutdown", "-h", "now"]
    if system == "Linux":
        if shutil.which("systemctl"):
            if action == "reboot":
                return ["systemctl", "reboot"]
            if action == "shutdown":
                return ["systemctl", "poweroff"]
        if action == "reboot":
            return ["/sbin/shutdown", "-r", "now"]
        if action == "shutdown":
            return ["/sbin/shutdown", "-h", "now"]
    raise RuntimeError(f"Unsupported platform/action: {system}/{action}")


def _schedule_power_action(action: str) -> dict:
    try:
        cmd = _power_command(action)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    # Stop runtime components first to reduce file/stream corruption risk.
    try:
        AI_WORKER.stop()
        RECORDER.stop()
        AUDIO_RECORDER.stop()
        for recorder in RAW_RECORDERS.values():
            recorder.stop()
        RAW_DUAL.stop()
    except Exception:
        pass

    def _run():
        time.sleep(1.0)
        try:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    threading.Thread(target=_run, daemon=True).start()
    return {
        "ok": True,
        "action": action,
        "scheduled": True,
        "delay_sec": 1.0,
        "requires_privileges": True,
        "note": "If the process lacks permission, the OS power action may fail.",
    }


@app.post("/api/system/reboot")
def api_system_reboot():
    return _schedule_power_action("reboot")


@app.post("/api/system/shutdown")
def api_system_shutdown():
    return _schedule_power_action("shutdown")


@app.get("/health")
def health():
    return {"ok": True, "ts": time.time()}


HOME_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>BallScope - Start</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Rubik:wght@400;600;700&family=JetBrains+Mono:wght@500&display=swap');
    :root {
      color-scheme: dark;
      --bg: #0c1117;
      --bg-2: #101925;
      --panel: rgba(17, 24, 38, 0.78);
      --panel-2: rgba(10, 15, 24, 0.72);
      --stroke: rgba(255,255,255,0.10);
      --stroke-strong: rgba(255,255,255,0.18);
      --text: #f7f8ff;
      --muted: rgba(247,248,255,0.68);
      --accent: #ffb454;
      --accent2: #67d6ff;
      --accent-ink: #271303;
      --shadow: 0 22px 60px rgba(0,0,0,0.36);
      --disabled: #627188;
      --pill-bg: rgba(255,255,255,0.03);
    }
    html[data-theme="light"] {
      color-scheme: light;
      --bg: #eef4fb;
      --bg-2: #dfe9f7;
      --panel: rgba(255,255,255,0.86);
      --panel-2: rgba(245,249,255,0.86);
      --stroke: rgba(20, 33, 53, 0.12);
      --stroke-strong: rgba(20, 33, 53, 0.22);
      --text: #0f1724;
      --muted: rgba(15, 23, 36, 0.66);
      --accent: #df7b17;
      --accent2: #0d84cc;
      --accent-ink: #ffffff;
      --shadow: 0 24px 60px rgba(23, 34, 56, 0.14);
      --disabled: #8492a5;
      --pill-bg: rgba(255,255,255,0.7);
    }
    * { box-sizing: border-box; }
    html, body { transition: background-color .25s ease, color .25s ease; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: "Rubik", sans-serif;
      color: var(--text);
      background:
        radial-gradient(900px 560px at -8% -18%, rgba(103,214,255,0.22), transparent),
        radial-gradient(800px 520px at 108% -8%, rgba(255,180,84,0.20), transparent),
        linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0)),
        var(--bg);
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 20px;
    }
    .shell {
      width: min(1020px, 100%);
      background:
        linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.01)),
        var(--panel);
      border: 1px solid var(--stroke);
      border-radius: 24px;
      backdrop-filter: blur(18px) saturate(140%);
      padding: 22px;
      box-shadow: var(--shadow);
      position: relative;
      overflow: hidden;
    }
    .shell::before {
      content: "";
      position: absolute;
      inset: -30% auto auto -10%;
      width: 260px;
      height: 260px;
      border-radius: 999px;
      background: radial-gradient(circle, rgba(103,214,255,0.18), transparent 70%);
      pointer-events: none;
    }
    .topbar {
      display: flex;
      justify-content: flex-start;
      align-items: flex-start;
      gap: 12px;
      position: relative;
      z-index: 1;
    }
    .global-settings {
      position: fixed;
      top: 14px;
      right: 14px;
      z-index: 40;
    }
    .icon-btn {
      appearance: none;
      border: 1px solid var(--stroke);
      background:
        linear-gradient(180deg, rgba(255,255,255,0.14), rgba(255,255,255,0.02)),
        var(--pill-bg);
      color: var(--text);
      width: 40px;
      height: 40px;
      border-radius: 14px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.08), 0 8px 18px rgba(0,0,0,0.08);
      transition: transform .16s ease, border-color .16s ease, box-shadow .16s ease;
      font-size: 18px;
      line-height: 1;
    }
    .icon-btn:hover {
      transform: translateY(-1px);
      border-color: var(--stroke-strong);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.10), 0 10px 20px rgba(0,0,0,0.12);
    }
    .icon-btn:focus-visible {
      outline: 2px solid var(--accent2);
      outline-offset: 2px;
    }
    .settings-menu {
      position: relative;
    }
    .settings-pop {
      position: absolute;
      right: 0;
      top: calc(100% + 10px);
      width: min(340px, calc(100vw - 40px));
      border: 1px solid var(--stroke);
      border-radius: 16px;
      background:
        linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.02)),
        var(--panel-2);
      backdrop-filter: blur(18px) saturate(140%);
      box-shadow: 0 18px 40px rgba(0,0,0,0.18);
      padding: 12px;
      display: none;
    }
    .settings-pop[open] { display: block; }
    .eyebrow {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border: 1px solid var(--stroke);
      background: var(--pill-bg);
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 12px;
      color: var(--muted);
      font-family: "JetBrains Mono", monospace;
      letter-spacing: .03em;
      text-transform: uppercase;
    }
    .eyebrow::before {
      content: "";
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: linear-gradient(180deg, var(--accent2), var(--accent));
      box-shadow: 0 0 0 4px rgba(255,255,255,0.04);
    }
    .hero {
      margin-top: 18px;
      position: relative;
      z-index: 1;
    }
    h1 {
      margin: 0;
      font-size: clamp(28px, 3.8vw, 38px);
      letter-spacing: -0.03em;
      line-height: 1.05;
    }
    .hero p {
      margin: 10px 0 0;
      color: var(--muted);
      max-width: 62ch;
    }
    .cards {
      margin-top: 22px;
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
      position: relative;
      z-index: 1;
    }
    .card {
      border-radius: 18px;
      border: 1px solid var(--stroke);
      padding: 16px;
      background:
        linear-gradient(180deg, rgba(255,255,255,0.09), rgba(255,255,255,0.00) 40%),
        linear-gradient(120deg, rgba(103,214,255,0.05), rgba(255,180,84,0.03)),
        var(--panel-2);
      text-decoration: none;
      color: var(--text);
      transition: transform .2s ease, border-color .2s ease, box-shadow .2s ease, background .2s ease;
      display: block;
      backdrop-filter: blur(14px) saturate(130%);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.07);
    }
    .card:hover {
      transform: translateY(-4px);
      border-color: color-mix(in srgb, var(--accent) 45%, var(--stroke));
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.1), 0 16px 28px rgba(0,0,0,0.14);
    }
    .card:focus-visible {
      outline: 2px solid var(--accent2);
      outline-offset: 2px;
    }
    .card h2 { margin: 0 0 8px; font-size: 20px; }
    .card small {
      color: var(--accent2);
      font-family: "JetBrains Mono", monospace;
      letter-spacing: .03em;
    }
    .card p { margin-top: 10px; min-height: 46px; }
    .card.live {
      cursor: not-allowed;
      opacity: .7;
      border-color: rgba(98, 113, 136, 0.6);
    }
    .pill {
      display: inline-block;
      margin-top: 12px;
      border-radius: 999px;
      padding: 5px 10px;
      font-size: 12px;
      border: 1px solid var(--stroke);
      color: var(--muted);
      background: var(--pill-bg);
    }
    .pill.coming {
      color: var(--text);
      border-color: rgba(98, 113, 136, 0.9);
      background: rgba(98, 113, 136, 0.22);
    }
    .settings-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      margin-bottom: 10px;
    }
    .settings-head h3 {
      margin: 0;
      font-size: 13px;
      letter-spacing: .14em;
      text-transform: uppercase;
      color: var(--muted);
    }
    .settings-grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 12px;
      align-items: start;
    }
    .field label {
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 6px;
    }
    .field select {
      width: 100%;
      border-radius: 12px;
      border: 1px solid var(--stroke);
      background:
        linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.02)),
        var(--panel);
      color: var(--text);
      padding: 10px 12px;
      font: 600 13px/1.2 "Rubik", sans-serif;
      transition: border-color .16s ease, box-shadow .16s ease;
    }
    .field select:focus {
      outline: none;
      border-color: color-mix(in srgb, var(--accent2) 55%, var(--stroke));
      box-shadow: 0 0 0 4px rgba(103,214,255,0.16);
    }
    html[data-theme="light"] .field select:focus {
      box-shadow: 0 0 0 4px rgba(13,132,204,0.14);
    }
    .settings-copy {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.35;
    }
    .settings-copy strong {
      color: var(--text);
      font-weight: 600;
    }
    .settings-copy {
      padding-top: 2px;
    }
    @media (max-width: 860px) {
      .cards { grid-template-columns: 1fr; }
      .shell { padding: 18px; }
      .topbar { align-items: center; }
      .global-settings {
        top: 10px;
        right: 10px;
      }
      .settings-pop {
        right: -2px;
        width: min(320px, calc(100vw - 30px));
      }
    }
  </style>
</head>
<body>
  <div class="global-settings">
    <div class="settings-menu">
      <button type="button" class="icon-btn" id="settingsBtn" aria-label="Open settings" aria-haspopup="dialog" aria-expanded="false">⚙</button>
      <section class="settings-pop" id="settingsPop" aria-label="Interface settings">
        <div class="settings-head">
          <h3>Settings</h3>
        </div>
        <div class="settings-grid">
          <div class="field">
            <label for="themePref">Theme</label>
            <select id="themePref">
              <option value="system">System</option>
              <option value="light">Light</option>
              <option value="dark">Dark</option>
            </select>
          </div>
          <div class="settings-copy">
            <strong>System</strong> follows your browser/device color scheme automatically. You can override it here any time.
          </div>
        </div>
      </section>
    </div>
  </div>
  <main class="shell">
    <div class="topbar">
      <div class="eyebrow">BallScope Control</div>
    </div>
    <div class="hero">
      <h1>BallScope System</h1>
      <p>Choose a workspace: recording, post-analysis, or Live later. The app behavior stays the same, only the design has been refined.</p>
    </div>
    <section class="cards">
      <a class="card" href="/record">
        <small>01 / RECORD</small>
        <h2>Record</h2>
        <p>Use the existing recording and control interface with the new visual theme.</p>
        <span class="pill">Open</span>
      </a>
      <a class="card" href="/analysis">
        <small>02 / AI ANALYSIS</small>
        <h2>Analysis</h2>
        <p>Upload recorded videos, define crop/ROI, and run post-match ball tracking.</p>
        <span class="pill">Open</span>
      </a>
      <a class="card" href="/live">
        <small>03 / LIVE</small>
        <h2>Live</h2>
        <p>Live dual-camera ball tracking in the browser with auto camera selection and zoom.</p>
        <span class="pill">Open</span>
      </a>
    </section>
  </main>
  <script>
    (() => {
      const KEY = "ballscope-theme";
      const root = document.documentElement;
      const select = document.getElementById("themePref");
      const settingsBtn = document.getElementById("settingsBtn");
      const settingsPop = document.getElementById("settingsPop");
      const media = window.matchMedia ? window.matchMedia("(prefers-color-scheme: light)") : null;
      const normalize = (value) => (value === "light" || value === "dark" || value === "system") ? value : "system";
      const getPref = () => normalize(localStorage.getItem(KEY));
      const resolve = (pref) => pref === "system" ? ((media && media.matches) ? "light" : "dark") : pref;
      const apply = () => {
        const pref = getPref();
        const actual = resolve(pref);
        root.setAttribute("data-theme", actual);
        root.setAttribute("data-theme-pref", pref);
        if (select) select.value = pref;
      };
      if (!localStorage.getItem(KEY)) localStorage.setItem(KEY, "system");
      apply();
      select?.addEventListener("change", (ev) => {
        localStorage.setItem(KEY, normalize(ev.target.value));
        apply();
      });
      if (media) {
        const onMediaChange = () => {
          if (getPref() === "system") apply();
        };
        if (typeof media.addEventListener === "function") media.addEventListener("change", onMediaChange);
        else if (typeof media.addListener === "function") media.addListener(onMediaChange);
      }
      const closeSettings = () => {
        settingsPop?.removeAttribute("open");
        settingsBtn?.setAttribute("aria-expanded", "false");
      };
      const openSettings = () => {
        settingsPop?.setAttribute("open", "");
        settingsBtn?.setAttribute("aria-expanded", "true");
      };
      settingsBtn?.addEventListener("click", (ev) => {
        ev.stopPropagation();
        if (settingsPop?.hasAttribute("open")) closeSettings();
        else openSettings();
      });
      settingsPop?.addEventListener("click", (ev) => ev.stopPropagation());
      document.addEventListener("click", closeSettings);
      document.addEventListener("keydown", (ev) => {
        if (ev.key === "Escape") closeSettings();
      });
    })();
  </script>
</body>
</html>
"""


ANALYSIS_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>BallScope - AI Analysis</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Rubik:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
    :root {
      color-scheme: dark;
      --bg:#0d131a;
      --panel:rgba(16,24,36,0.78);
      --panel-2:rgba(10,16,24,0.72);
      --stroke:rgba(255,255,255,.09);
      --stroke-strong:rgba(255,255,255,.18);
      --text:#f4f7ff;
      --muted:rgba(244,247,255,.66);
      --accent:#ffb454;
      --accent-2:#67d6ff;
      --ring: rgba(103,214,255,.35);
      --shadow: 0 18px 34px rgba(0,0,0,.22);
      --input-bg: rgba(9,14,22,.95);
      --chip-bg: rgba(255,255,255,.02);
      --btn-text: #fffaf1;
    }
    html[data-theme="light"] {
      color-scheme: light;
      --bg:#edf3fb;
      --panel:rgba(255,255,255,0.88);
      --panel-2:rgba(249,252,255,0.95);
      --stroke:rgba(17,29,48,.12);
      --stroke-strong:rgba(17,29,48,.22);
      --text:#101826;
      --muted:rgba(16,24,38,.66);
      --accent:#dd7b1b;
      --accent-2:#0b84c8;
      --ring: rgba(11,132,200,.25);
      --shadow: 0 14px 30px rgba(20,31,55,.10);
      --input-bg: rgba(255,255,255,.96);
      --chip-bg: rgba(255,255,255,.7);
      --btn-text: #fff;
    }
    * { box-sizing: border-box; }
    body {
      margin:0; min-height:100vh; color:var(--text); font-family:"Rubik", sans-serif;
      background:
        radial-gradient(700px 420px at 0% -10%, rgba(103,214,255,0.2), transparent),
        radial-gradient(700px 420px at 100% -10%, rgba(255,180,84,0.18), transparent),
        var(--bg);
    }
    .wrap { max-width:1180px; margin:0 auto; padding:22px 16px 40px; }
    .top { display:flex; justify-content:space-between; align-items:center; gap:12px; margin-bottom:14px; }
    .top-actions { display:flex; align-items:center; gap:10px; flex-wrap: wrap; }
    .nav-btn, .top a {
      appearance:none;
      color:var(--text);
      text-decoration:none;
      border:1px solid var(--stroke);
      border-radius:12px;
      padding:9px 12px;
      font-size:13px;
      font-weight:600;
      background:
        linear-gradient(180deg, rgba(255,255,255,.16), rgba(255,255,255,.03)),
        var(--panel-2);
      cursor: pointer;
      transition: transform .16s ease, border-color .16s ease, box-shadow .16s ease;
      display:inline-flex;
      align-items:center;
      justify-content:center;
      gap:8px;
      backdrop-filter: blur(12px) saturate(135%);
      box-shadow: inset 0 1px 0 rgba(255,255,255,.08);
    }
    .nav-btn:hover, .top a:hover {
      transform: translateY(-1px);
      border-color: var(--stroke-strong);
      box-shadow: 0 8px 16px rgba(0,0,0,.10);
    }
    .nav-btn:focus-visible, .top a:focus-visible {
      outline: 2px solid var(--accent-2);
      outline-offset: 2px;
    }
    .grid { display:grid; gap:14px; grid-template-columns: 360px 1fr; }
    .card {
      background:
        linear-gradient(180deg, rgba(255,255,255,.10), rgba(255,255,255,.00) 38%),
        linear-gradient(120deg, rgba(103,214,255,.05), rgba(255,180,84,.03)),
        var(--panel);
      border:1px solid var(--stroke);
      border-radius:18px;
      padding:14px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(16px) saturate(135%);
      box-shadow: inset 0 1px 0 rgba(255,255,255,.06), var(--shadow);
    }
    .card h3 { margin:0 0 10px; font-size:13px; letter-spacing:.12em; text-transform:uppercase; color:var(--muted); }
    label { display:block; font-size:12px; color:var(--muted); margin:10px 0 6px; }
    input, select {
      width:100%;
      border-radius:12px;
      border:1px solid var(--stroke);
      background:var(--input-bg);
      color:var(--text);
      padding:9px 11px;
      transition: border-color .16s ease, box-shadow .16s ease, background-color .16s ease;
    }
    input:focus, select:focus {
      outline:none;
      border-color: color-mix(in srgb, var(--accent-2) 55%, var(--stroke));
      box-shadow: 0 0 0 4px var(--ring);
    }
    .row { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
    button {
      width:100%;
      margin-top:12px;
      padding:11px 12px;
      border-radius:12px;
      border:1px solid rgba(255,180,84,.52);
      background:
        linear-gradient(180deg, rgba(255,255,255,.20), rgba(255,255,255,0)),
        linear-gradient(180deg, rgba(255,180,84,.80), rgba(255,180,84,.30));
      color:var(--btn-text);
      font-weight:700;
      letter-spacing:.01em;
      cursor:pointer;
      box-shadow: 0 8px 18px rgba(255,180,84,.16);
      transition: transform .16s ease, box-shadow .16s ease, border-color .16s ease;
      backdrop-filter: blur(12px) saturate(130%);
    }
    button:hover {
      transform: translateY(-1px);
      box-shadow: 0 10px 22px rgba(255,180,84,.22);
      border-color: rgba(255,180,84,.75);
    }
    button:focus-visible {
      outline: 2px solid var(--accent-2);
      outline-offset: 2px;
    }
    .status { font-family:"JetBrains Mono", monospace; font-size:12px; color:var(--muted); margin-top:10px; min-height:16px; }
    .preview {
      border-radius:16px; overflow:hidden; border:1px solid var(--stroke); background:
      linear-gradient(180deg, rgba(255,255,255,.05), rgba(255,255,255,0)),
      var(--panel-2); aspect-ratio:16/9;
      display:flex; align-items:center; justify-content:center;
    }
    .preview img { width:100%; height:100%; object-fit:contain; display:none; }
    .preview span { color:var(--muted); font-size:14px; }
    .stats { margin-top:10px; display:grid; grid-template-columns:repeat(2,1fr); gap:8px; }
    .chip {
      border:1px solid var(--stroke);
      background:
        linear-gradient(180deg, rgba(255,255,255,.08), rgba(255,255,255,0)),
        var(--chip-bg);
      border-radius:999px;
      padding:6px 8px;
      font-size:12px;
      text-align:center;
      font-family:"JetBrains Mono", monospace;
    }
    .result-link {
      display:none;
      color: var(--accent-2);
      text-decoration: none;
      font-weight: 600;
      border: 1px solid var(--stroke);
      background:
        linear-gradient(180deg, rgba(255,255,255,.14), rgba(255,255,255,.03)),
        var(--panel-2);
      padding: 8px 10px;
      border-radius: 12px;
      backdrop-filter: blur(12px) saturate(135%);
    }
    .result-link:hover { border-color: var(--stroke-strong); }
    @media (max-width: 980px) {
      .grid { grid-template-columns:1fr; }
      .top { align-items:flex-start; flex-direction: column; }
    }
    @media (max-width: 680px) {
      .row { grid-template-columns: 1fr; }
      .stats { grid-template-columns: 1fr; }
      .card { padding: 12px; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="top">
      <div>
        <h1 style="margin:0;font-size:26px;">AI Analysis</h1>
        <div style="color:var(--muted);font-size:13px;">Upload a video, define crop/ROI, and run ball tracking after recording.</div>
      </div>
      <div class="top-actions">
        <a class="nav-btn" href="/">Back to Home</a>
      </div>
    </div>
    <section class="grid">
      <div class="card">
        <h3>Upload + Crop</h3>
        <label>Video File</label>
        <input id="videoFile" type="file" accept="video/*"/>
        <div class="row">
          <div>
            <label>Class</label>
            <select id="classId">
              <option value="">Auto (no filter)</option>
            </select>
          </div>
          <div>
            <label>Confidence</label>
            <input id="conf" type="number" value="0.32" step="0.01" min="0" max="1"/>
          </div>
        </div>
        <div class="row">
          <div>
            <label>IOU</label>
            <input id="iou" type="number" value="0.35" step="0.01" min="0" max="1"/>
          </div>
          <div>
            <label>Model</label>
            <select id="modelPath"></select>
          </div>
        </div>
        <div class="row">
          <div>
            <label>Device</label>
            <select id="deviceSel">
              <option value="auto">Auto</option>
            </select>
          </div>
          <div></div>
        </div>
        <div class="row">
          <div>
            <label>Zoom Output</label>
            <input id="zoom" type="number" value="1.6" step="0.1" min="1.0" max="4.0"/>
          </div>
          <div>
            <label>Speed Up</label>
            <select id="speedUp">
              <option value="false">Off (best quality preview)</option>
              <option value="true">On (faster analysis)</option>
            </select>
          </div>
        </div>
        <label>Crop X/Y/W/H (0.0 - 1.0, relative to frame)</label>
        <div class="row">
          <input id="cropX" type="number" value="0.00" step="0.01" min="0" max="1"/>
          <input id="cropY" type="number" value="0.00" step="0.01" min="0" max="1"/>
        </div>
        <div class="row">
          <input id="cropW" type="number" value="1.00" step="0.01" min="0.05" max="1"/>
          <input id="cropH" type="number" value="1.00" step="0.01" min="0.05" max="1"/>
        </div>
        <button id="startBtn">Start Analysis</button>
        <div class="status" id="line">idle</div>
      </div>
      <div class="card">
        <h3>Tracking Preview</h3>
        <div class="preview"><img id="stream" alt="analysis stream"/><span id="streamHint">No session started yet</span></div>
        <div class="stats">
          <div class="chip" id="stState">state:-</div>
          <div class="chip" id="stProg">progress:0%</div>
          <div class="chip" id="stFps">fps:0</div>
          <div class="chip" id="stEta">eta:-</div>
          <div class="chip" id="stFrames">frames:0</div>
          <div class="chip" id="stTotal">total:0</div>
          <div class="chip" id="stDet">detections:0</div>
          <div class="chip" id="stSeg">segments:0</div>
          <div class="chip" id="stConf">conf:0</div>
          <div class="chip" id="stSeen">last seen:-</div>
        </div>
        <div style="margin-top:12px;">
          <a id="resultLink" class="result-link" href="#">Download Result MP4</a>
        </div>
      </div>
    </section>
  </div>
  <script>
    (() => {
      const KEY = "ballscope-theme";
      const root = document.documentElement;
      const media = window.matchMedia ? window.matchMedia("(prefers-color-scheme: light)") : null;
      const normalize = (value) => (value === "light" || value === "dark" || value === "system") ? value : "system";
      const getPref = () => normalize(localStorage.getItem(KEY));
      const resolve = (pref) => pref === "system" ? ((media && media.matches) ? "light" : "dark") : pref;
      const apply = () => {
        const pref = getPref();
        root.setAttribute("data-theme", resolve(pref));
        root.setAttribute("data-theme-pref", pref);
      };
      if (!localStorage.getItem(KEY)) localStorage.setItem(KEY, "system");
      apply();
      if (media) {
        const onMediaChange = () => { if (getPref() === "system") apply(); };
        if (typeof media.addEventListener === "function") media.addEventListener("change", onMediaChange);
        else if (typeof media.addListener === "function") media.addListener(onMediaChange);
      }
    })();

    const $ = (id) => document.getElementById(id);
    let sid = null;
    let timer = null;

    const loadClasses = async () => {
      const modelPath = $("modelPath").value || "models/football-ball-detection.pt";
      try {
        const r = await fetch(`/api/analysis/classes?model_path=${encodeURIComponent(modelPath)}`);
        if (!r.ok) return;
        const j = await r.json();
        const sel = $("classId");
        const current = sel.value;
        sel.innerHTML = "";
        const autoOpt = document.createElement("option");
        autoOpt.value = "";
        autoOpt.textContent = "Auto (no filter)";
        sel.appendChild(autoOpt);
        const classes = j.classes || [];
        let sportsBallId = "";
        for (const c of classes) {
          const opt = document.createElement("option");
          opt.value = String(c.id);
          opt.textContent = `${c.id}: ${c.label}`;
          if ((c.label || "").toLowerCase() === "sports ball") sportsBallId = String(c.id);
          sel.appendChild(opt);
        }
        if ([...sel.options].some(o => o.value === current)) sel.value = current;
        else if (sportsBallId) sel.value = sportsBallId;
      } catch (e) {}
    };

    const loadModels = async () => {
      try {
        const r = await fetch("/api/analysis/models");
        if (!r.ok) return;
        const j = await r.json();
        const sel = $("modelPath");
        sel.innerHTML = "";
        for (const m of (j.models || [])) {
          const opt = document.createElement("option");
          opt.value = m;
          opt.textContent = m;
          sel.appendChild(opt);
        }
        if (j.default && [...sel.options].some(o => o.value === j.default)) {
          sel.value = j.default;
        }
        await loadClasses();
      } catch (e) {}
    };

    const loadDevices = async () => {
      try {
        const r = await fetch("/api/analysis/devices");
        if (!r.ok) return;
        const j = await r.json();
        const sel = $("deviceSel");
        const cur = sel.value;
        sel.innerHTML = "";
        for (const d of (j.devices || [])) {
          const opt = document.createElement("option");
          opt.value = d.id;
          opt.textContent = d.label;
          sel.appendChild(opt);
        }
        if ([...sel.options].some(o => o.value === cur)) sel.value = cur;
        else if ([...sel.options].some(o => o.value === (j.default || "auto"))) sel.value = j.default || "auto";
      } catch (e) {}
    };

    const poll = async () => {
      if (!sid) return;
      try {
        const r = await fetch(`/api/analysis/status/${sid}`);
        if (!r.ok) return;
        const j = await r.json();
        $("stState").textContent = `state:${j.state || "-"}`;
        $("stProg").textContent = `progress:${(j.progress_pct || 0).toFixed(1)}%`;
        $("stFps").textContent = `fps:${(j.fps || 0).toFixed(1)}`;
        $("stEta").textContent = `eta:${j.eta_sec == null ? "-" : Math.max(0, j.eta_sec).toFixed(1) + "s"}`;
        $("stFrames").textContent = `frames:${(j.frames || 0).toFixed(0)}`;
        $("stTotal").textContent = `total:${(j.total_frames || 0).toFixed(0)}`;
        $("stDet").textContent = `detections:${(j.detections || 0).toFixed(0)}`;
        $("stSeg").textContent = `segments:${(j.segments_saved || 0).toFixed(0)}`;
        $("stConf").textContent = `conf:${(j.last_conf || 0).toFixed(2)}`;
        $("stSeen").textContent = `last seen:${j.last_seen_sec == null ? "-" : j.last_seen_sec.toFixed(2) + "s"}`;
        if (j.running) {
          const speedTxt = j.speed_up ? "speed-up:on" : "speed-up:off";
          const tuneTxt = `imgsz:${j.imgsz || "-"}, detectEvery:${j.detect_every || "-"}`;
          $("line").textContent = `Analysis running ${((j.progress_pct || 0)).toFixed(1)}% (${j.stream_count || 1} stream(s), ${j.device_used || "auto"}, ${speedTxt}, ${tuneTxt})`;
        }
        if (j.output_url && j.state === "done") {
          $("resultLink").href = j.output_url;
          $("resultLink").style.display = "inline-block";
          $("line").textContent = "Analysis finished - MP4 ready";
        }
        if (j.last_error) $("line").textContent = `error: ${j.last_error}`;
      } catch (e) {}
    };

    $("startBtn").onclick = async () => {
      const file = $("videoFile").files[0];
      if (!file) {
        $("line").textContent = "Please select a video first.";
        return;
      }
      const params = new URLSearchParams({
        conf: $("conf").value,
        iou: $("iou").value,
        model_path: $("modelPath").value,
        device: $("deviceSel").value,
        zoom: $("zoom").value,
        speed_up: $("speedUp").value,
        crop_x: $("cropX").value,
        crop_y: $("cropY").value,
        crop_w: $("cropW").value,
        crop_h: $("cropH").value,
      });
      if ($("classId").value !== "") params.set("class_id", $("classId").value);
      $("line").textContent = `Uploading ... (0 / ${Math.round(file.size / (1024 * 1024))} MB)`;
      try {
        const j = await new Promise((resolve, reject) => {
          const xhr = new XMLHttpRequest();
          xhr.open("POST", `/api/analysis/upload?${params.toString()}`);
          xhr.setRequestHeader("Content-Type", "application/octet-stream");
          xhr.setRequestHeader("X-Filename", file.name || "upload.mp4");
          xhr.responseType = "json";
          xhr.upload.onprogress = (ev) => {
            if (!ev.lengthComputable) return;
            const up = Math.round(ev.loaded / (1024 * 1024));
            const total = Math.max(1, Math.round(ev.total / (1024 * 1024)));
            $("line").textContent = `Uploading ... (${up} / ${total} MB)`;
          };
          xhr.onerror = () => reject(new Error("network error"));
          xhr.onload = () => {
            if (xhr.status < 200 || xhr.status >= 300) {
              reject(new Error(typeof xhr.response === "string" ? xhr.response : `HTTP ${xhr.status}`));
              return;
            }
            resolve(xhr.response || JSON.parse(xhr.responseText || "{}"));
          };
          xhr.send(file);
        });
        sid = j.session_id;
        $("resultLink").style.display = "none";
        $("stream").src = `/video/analysis/${sid}.mjpg`;
        $("stream").style.display = "block";
        $("streamHint").style.display = "none";
        $("line").textContent = `Analysis started (${sid})`;
        if (timer) clearInterval(timer);
        timer = setInterval(poll, 350);
      } catch (e) {
        $("line").textContent = "Upload/start failed.";
      }
    };

    $("modelPath").onchange = loadClasses;
    loadModels();
    loadDevices();
  </script>
</body>
</html>
"""


LIVE_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>BallScope - Live</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Rubik:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
    :root {
      color-scheme: dark;
      --bg: #0c1218;
      --panel: rgba(17,24,39,0.72);
      --panel-2: rgba(10,15,24,0.78);
      --stroke: rgba(255,255,255,0.09);
      --stroke-strong: rgba(255,255,255,0.18);
      --text: #f6f8ff;
      --muted: rgba(246,248,255,0.66);
      --accent: #ffb454;
      --accent-2: #68d7ff;
      --ok: #7dffb3;
      --warn: #ffcf5d;
      --ring: rgba(104,215,255,0.28);
      --shadow: 0 16px 34px rgba(0,0,0,0.2);
      --input-bg: rgba(8,12,18,0.92);
      --chip-bg: rgba(255,255,255,0.03);
    }
    html[data-theme="light"] {
      color-scheme: light;
      --bg: #ecf3fb;
      --panel: rgba(255,255,255,0.88);
      --panel-2: rgba(247,251,255,0.95);
      --stroke: rgba(20,32,52,0.12);
      --stroke-strong: rgba(20,32,52,0.22);
      --text: #101826;
      --muted: rgba(16,24,38,0.66);
      --accent: #dd7c1c;
      --accent-2: #0b86cc;
      --ok: #149c62;
      --warn: #c58b16;
      --ring: rgba(11,134,204,0.22);
      --shadow: 0 12px 28px rgba(20,31,55,0.10);
      --input-bg: rgba(255,255,255,0.96);
      --chip-bg: rgba(255,255,255,0.8);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: "Rubik", sans-serif;
      color: var(--text);
      background:
        radial-gradient(900px 520px at 10% -12%, rgba(104,215,255,0.18), transparent),
        radial-gradient(800px 560px at 92% -20%, rgba(255,180,84,0.14), transparent),
        linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0)),
        var(--bg);
    }
    .wrap { max-width: 1320px; margin: 0 auto; padding: 22px 14px 40px; }
    .top {
      display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 14px;
    }
    .brand h1 { margin: 0; font-size: 28px; letter-spacing: -0.02em; }
    .brand .sub { color: var(--muted); font-size: 13px; margin-top: 4px; }
    .top-actions { display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
    .nav-btn {
      appearance:none; border:1px solid var(--stroke); color:var(--text); text-decoration:none; cursor:pointer;
      border-radius:12px; padding:9px 12px; font-size:13px; font-weight:600;
      background:
        linear-gradient(180deg, rgba(255,255,255,0.16), rgba(255,255,255,0.03)),
        var(--panel-2);
      backdrop-filter: blur(12px) saturate(135%);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
      transition: transform .16s ease, border-color .16s ease, box-shadow .16s ease;
      display:inline-flex; align-items:center; gap:8px;
    }
    .nav-btn:hover { transform: translateY(-1px); border-color: var(--stroke-strong); box-shadow: inset 0 1px 0 rgba(255,255,255,0.1), 0 8px 16px rgba(0,0,0,0.10); }
    .nav-btn:focus-visible { outline:2px solid var(--accent-2); outline-offset:2px; }
    .badge { display:inline-flex; align-items:center; gap:8px; color:var(--muted); font-size:12px; }
    .dot { width:10px; height:10px; border-radius:50%; background:var(--warn); }
    .dot.ok { background: var(--ok); }

    .layout { display:grid; grid-template-columns: minmax(0, 1.6fr) minmax(340px, .9fr); gap: 14px; }
    .stack { display:grid; gap:14px; }
    .card {
      border:1px solid var(--stroke);
      border-radius:18px;
      padding:14px;
      background:
        linear-gradient(180deg, rgba(255,255,255,0.10), rgba(255,255,255,0) 38%),
        linear-gradient(120deg, rgba(104,215,255,0.05), rgba(255,180,84,0.03)),
        var(--panel);
      backdrop-filter: blur(16px) saturate(135%);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.06), var(--shadow);
    }
    .card h3 {
      margin: 0 0 10px; font-size: 12px; letter-spacing: .18em; text-transform: uppercase; color: var(--muted);
    }
    .stage-head {
      display:flex; justify-content:space-between; align-items:center; gap:10px; margin-bottom:10px; flex-wrap:wrap;
    }
    .btn-row { display:flex; gap:8px; flex-wrap:wrap; }
    .btn {
      appearance:none; cursor:pointer; border-radius:12px; border:1px solid var(--stroke); padding:10px 12px;
      color: var(--text);
      background:
        linear-gradient(180deg, rgba(255,255,255,.14), rgba(255,255,255,.03)),
        var(--panel-2);
      backdrop-filter: blur(12px) saturate(130%);
      box-shadow: inset 0 1px 0 rgba(255,255,255,.07);
      font: 600 13px/1 "Rubik", sans-serif;
      transition: transform .16s ease, border-color .16s ease, box-shadow .16s ease;
    }
    .btn:hover { transform: translateY(-1px); border-color: var(--stroke-strong); box-shadow: inset 0 1px 0 rgba(255,255,255,.1), 0 8px 16px rgba(0,0,0,.08); }
    .btn.primary {
      border-color: rgba(255,180,84,.52);
      background:
        linear-gradient(180deg, rgba(255,255,255,.22), rgba(255,255,255,0)),
        linear-gradient(180deg, rgba(255,180,84,.80), rgba(255,180,84,.30));
      color: #fffaf1;
      box-shadow: 0 10px 20px rgba(255,180,84,.16);
    }
    .btn.primary:hover { border-color: rgba(255,180,84,.75); box-shadow: 0 12px 24px rgba(255,180,84,.22); }
    .btn:focus-visible { outline:2px solid var(--accent-2); outline-offset:2px; }

    .stage {
      border-radius: 16px;
      border: 1px solid var(--stroke);
      background:
        linear-gradient(180deg, rgba(255,255,255,.05), rgba(255,255,255,0)),
        var(--panel-2);
      padding: 10px;
    }
    .frame {
      border-radius: 14px; overflow:hidden; border:1px solid var(--stroke);
      background: linear-gradient(180deg, rgba(255,255,255,.05), rgba(255,255,255,0)), var(--panel-2);
      position: relative;
    }
    .frame.main { aspect-ratio: 16/9; }
    .frame.side { aspect-ratio: 16/9; }
    .frame img { width:100%; height:100%; object-fit: cover; display:block; }
    .frame img.fit { object-fit: contain; }
    .label-chip {
      position:absolute; top:8px; left:8px; z-index:2;
      border-radius:999px; padding:5px 8px; font-size:11px;
      border:1px solid var(--stroke);
      background: linear-gradient(180deg, rgba(255,255,255,.14), rgba(255,255,255,.03)), var(--chip-bg);
      backdrop-filter: blur(10px) saturate(130%);
      color: var(--text);
      font-family: "JetBrains Mono", monospace;
    }
    .meta { display:flex; gap:8px; flex-wrap:wrap; margin-top:10px; }
    .chip {
      border:1px solid var(--stroke);
      background: linear-gradient(180deg, rgba(255,255,255,.08), rgba(255,255,255,0)), var(--chip-bg);
      border-radius:999px; padding:5px 9px; font-size:11px; color:var(--muted);
      font-family: "JetBrains Mono", monospace;
    }
    .statusline { margin-top: 10px; color: var(--muted); font-size: 12px; min-height: 18px; font-family: "JetBrains Mono", monospace; }

    label { display:block; font-size:12px; color:var(--muted); margin-bottom:6px; }
    input, select {
      width:100%; padding:9px 11px; border-radius:12px; border:1px solid var(--stroke);
      background: var(--input-bg); color:var(--text); font-size:13px;
      transition: border-color .16s ease, box-shadow .16s ease;
    }
    input:focus, select:focus { outline:none; border-color: color-mix(in srgb, var(--accent-2) 55%, var(--stroke)); box-shadow: 0 0 0 4px var(--ring); }
    .row { display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap:10px; }
    .row3 { display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap:10px; }
    .hint { color: var(--muted); font-size: 12px; margin-top: 8px; }

    @media (max-width: 1120px) { .layout { grid-template-columns: 1fr; } }
    @media (max-width: 720px) {
      .wrap { padding: 14px 10px 30px; }
      .top { align-items: flex-start; flex-direction: column; }
      .row, .row3 { grid-template-columns: 1fr; }
      .btn-row .btn { width: 100%; justify-content: center; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="top">
      <div class="brand">
        <h1>BallScope Live</h1>
        <div class="sub">Live browser preview with ball tracking, auto camera selection, and zoom. No recording.</div>
      </div>
      <div class="top-actions">
        <a class="nav-btn" href="/">Back to Home</a>
        <div class="badge"><span class="dot" id="systemDot"></span><span id="systemStatus">Booting</span></div>
      </div>
    </div>

    <section class="layout">
      <div class="stack">
        <div class="card">
          <div class="stage-head">
            <h3 style="margin:0;">Live Preview</h3>
            <span class="chip" id="previewModeChip">preview:auto-zoom</span>
          </div>

          <div class="stage" id="liveStage">
            <div class="frame main">
              <span class="label-chip" id="previewLabel">Final / Auto Zoom</span>
              <img id="liveMainImg" class="fit" src="" alt="Live preview">
            </div>
          </div>

          <div class="btn-row" style="margin-top:10px;">
            <button type="button" class="btn primary" id="liveStart">Start Live Tracking</button>
            <button type="button" class="btn" id="liveStop">Stop Live</button>
            <button type="button" class="btn" id="fsBtn">Fullscreen</button>
          </div>
          <div class="statusline" id="liveStatusLine">idle</div>
          <div class="meta">
            <div class="chip" id="aiState">state:-</div>
            <div class="chip" id="aiMode">mode:-</div>
            <div class="chip" id="aiActive">active:-</div>
            <div class="chip" id="aiFps">fps:0.0</div>
            <div class="chip" id="aiConfStat">conf:0.00</div>
            <div class="chip" id="aiSeen">last_seen:-</div>
            <div class="chip" id="camLStatus">camL:-</div>
            <div class="chip" id="camRStatus">camR:-</div>
          </div>
        </div>
      </div>

      <div class="stack">
        <div class="card">
          <h3>Live AI Controls</h3>
          <div class="row">
            <div>
              <label>Camera Selection</label>
              <select id="aiActiveInput">
                <option value="auto">Auto (pick best)</option>
                <option value="camL">Force Left</option>
                <option value="camR">Force Right</option>
              </select>
            </div>
            <div>
              <label>Model</label>
              <select id="modelPath"></select>
            </div>
          </div>
          <div class="row" style="margin-top:10px;">
            <div>
              <label>Class</label>
              <select id="classId">
                <option value="">Auto (no filter)</option>
              </select>
            </div>
            <div>
              <label>Device</label>
              <select id="deviceSel"><option value="auto">Auto</option></select>
            </div>
          </div>
          <div class="row3" style="margin-top:10px;">
            <div>
              <label>Confidence</label>
              <input id="aiConfInput" type="number" step="0.01" min="0" max="1" value="0.35"/>
            </div>
            <div>
              <label>IOU</label>
              <input id="aiIouInput" type="number" step="0.01" min="0" max="1" value="0.5"/>
            </div>
            <div>
              <label>Image Size</label>
              <input id="aiImgsz" type="number" step="32" min="256" max="1280" value="640"/>
            </div>
          </div>
          <div class="row3" style="margin-top:10px;">
            <div>
              <label>Zoom</label>
              <input id="aiZoomInput" type="number" step="0.1" min="1" max="5" value="1.6"/>
            </div>
            <div>
              <label>Smooth</label>
              <input id="aiSmoothInput" type="number" step="0.05" min="0" max="0.99" value="0.85"/>
            </div>
            <div>
              <label>Lost Hold (sec)</label>
              <input id="aiLostHold" type="number" step="0.1" min="0" max="10" value="1.5"/>
            </div>
          </div>
          <div class="row" style="margin-top:10px;">
            <div>
              <label>Detect Every N Frames</label>
              <input id="aiDetectEvery" type="number" min="1" max="30" value="1"/>
            </div>
            <div>
              <label>View Mode</label>
              <select id="viewModeSelect">
                <option value="auto">Auto Zoom</option>
                <option value="left">Left Camera</option>
                <option value="right">Right Camera</option>
              </select>
            </div>
          </div>
          <div class="row" style="margin-top:10px;">
            <div>
              <label>Preview Quality</label>
              <select id="previewQuality">
                <option value="high">High (sharper)</option>
                <option value="balanced" selected>Balanced</option>
                <option value="fast">Fast (higher UI FPS)</option>
              </select>
            </div>
            <div>
              <label>Preview Source</label>
              <div class="hint" style="margin-top:8px;">Affects browser preview only</div>
            </div>
          </div>
        </div>

        <div class="card">
          <h3>Camera Tuning</h3>
          <div class="row">
            <div>
              <label>Camera</label>
              <select id="camSelect">
                <option value="camL">Camera Left</option>
                <option value="camR">Camera Right</option>
              </select>
            </div>
            <div>
              <label>Preset</label>
              <select id="camPreset"></select>
            </div>
          </div>
          <div class="row" style="margin-top:10px;">
            <div>
              <label>Source (Device)</label>
              <input id="camSource" type="text" placeholder="/dev/video0 or 0"/>
            </div>
            <div>
              <label>Apply</label>
              <div class="hint" style="margin-top:8px;">Use `0/1` on Mac, `/dev/video*` on Jetson</div>
            </div>
          </div>
          <div class="row" style="margin-top:10px;">
            <div>
              <label>Brightness</label>
              <input id="camBrightness" type="number"/>
            </div>
            <div>
              <label>Contrast</label>
              <input id="camContrast" type="number"/>
            </div>
          </div>
          <div class="row" style="margin-top:10px;">
            <div>
              <label>Saturation</label>
              <input id="camSaturation" type="number"/>
            </div>
            <div>
              <label>Gain</label>
              <input id="camGain" type="number"/>
            </div>
          </div>
          <div class="row" style="margin-top:10px;">
            <div>
              <label>Auto WB</label>
              <select id="camAutoWb">
                <option value="">Keep</option>
                <option value="true">On</option>
                <option value="false">Off</option>
              </select>
            </div>
            <div>
              <label>Auto Exposure</label>
              <select id="camAutoExp">
                <option value="">Keep</option>
                <option value="true">On</option>
                <option value="false">Off</option>
              </select>
            </div>
          </div>
          <div class="btn-row" style="margin-top:10px;">
            <button type="button" class="btn" id="camApply">Apply Camera Settings</button>
          </div>
          <div class="statusline" id="camStatusLine">idle</div>
        </div>
      </div>
    </section>
  </div>

  <script>
    (() => {
      const KEY = "ballscope-theme";
      const root = document.documentElement;
      const media = window.matchMedia ? window.matchMedia("(prefers-color-scheme: light)") : null;
      const normalize = (value) => (value === "light" || value === "dark" || value === "system") ? value : "system";
      const getPref = () => normalize(localStorage.getItem(KEY));
      const resolve = (pref) => pref === "system" ? ((media && media.matches) ? "light" : "dark") : pref;
      const apply = () => {
        const pref = getPref();
        root.setAttribute("data-theme", resolve(pref));
        root.setAttribute("data-theme-pref", pref);
      };
      if (!localStorage.getItem(KEY)) localStorage.setItem(KEY, "system");
      apply();
      if (media) {
        const onMediaChange = () => { if (getPref() === "system") apply(); };
        if (typeof media.addEventListener === "function") media.addEventListener("change", onMediaChange);
        else if (typeof media.addListener === "function") media.addListener(onMediaChange);
      }
    })();

    const $ = (id) => document.getElementById(id);
    const fetchJSON = async (url, opts = {}) => {
      const res = await fetch(url, Object.assign({ headers: { "Content-Type": "application/json" } }, opts));
      if (!res.ok) throw new Error(await res.text());
      return await res.json();
    };

    let lastState = null;
    let pollTimer = null;
    let viewMode = "auto";
    let currentPreviewKey = "";
    let streamNonce = 0;

    const streamForView = (mode) => {
      const profile = $("previewQuality")?.value || "balanced";
      const qp = `profile=${encodeURIComponent(profile)}`;
      const aiRunning = !!(lastState && lastState.ai && lastState.ai.running);

      if (mode === "left") {
        return {
          url: `/video/live/cam/camL.mjpg?${qp}`,
          label: "Camera Left",
          fit: false,
          chip: `preview:left/${profile}`,
        };
      }
      if (mode === "right") {
        return {
          url: `/video/live/cam/camR.mjpg?${qp}`,
          label: "Camera Right",
          fit: false,
          chip: `preview:right/${profile}`,
        };
      }

      if (aiRunning) {
        return {
          url: `/video/live/final.mjpg?${qp}`,
          label: "Final / Auto Zoom",
          fit: true,
          chip: `preview:auto-zoom/${profile}`,
        };
      }

      const fallbackCam = ($("aiActiveInput")?.value === "camR") ? "camR" : "camL";
      return {
        url: `/video/live/cam/${fallbackCam}.mjpg?${qp}`,
        label: `Preview (${fallbackCam === "camR" ? "Camera Right" : "Camera Left"}) / AI Off`,
        fit: false,
        chip: `preview:camera/${profile}`,
      };
    };

    const refreshMainStream = (force = false) => {
      const img = $("liveMainImg");
      if (!img) return;
      const spec = streamForView(viewMode);
      $("previewLabel").textContent = spec.label;
      $("previewModeChip").textContent = spec.chip;
      img.classList.toggle("fit", !!spec.fit);
      const nextKey = `${spec.url}|${spec.fit ? "fit" : "cover"}`;
      if (!force && nextKey === currentPreviewKey) return;
      currentPreviewKey = nextKey;
      img.src = "";
      streamNonce += 1;
      const sep = spec.url.includes("?") ? "&" : "?";
      img.src = `${spec.url}${sep}v=${streamNonce}`;
    };

    const applyViewMode = (mode) => {
      viewMode = (mode === "left" || mode === "right") ? mode : "auto";
      $("viewModeSelect").value = viewMode;
      refreshMainStream();
    };

    const loadModels = async () => {
      try {
        const r = await fetch("/api/analysis/models");
        if (!r.ok) return;
        const j = await r.json();
        const sel = $("modelPath");
        sel.innerHTML = "";
        for (const m of (j.models || [])) {
          const opt = document.createElement("option");
          opt.value = m;
          opt.textContent = m;
          sel.appendChild(opt);
        }
        if (j.default && [...sel.options].some(o => o.value === j.default)) sel.value = j.default;
        await loadClasses();
      } catch (e) {}
    };

    const loadClasses = async () => {
      const modelPath = $("modelPath").value || "models/football-ball-detection.pt";
      try {
        const r = await fetch(`/api/analysis/classes?model_path=${encodeURIComponent(modelPath)}`);
        if (!r.ok) return;
        const j = await r.json();
        const sel = $("classId");
        const cur = sel.value;
        sel.innerHTML = "";
        const autoOpt = document.createElement("option");
        autoOpt.value = "";
        autoOpt.textContent = "Auto (no filter)";
        sel.appendChild(autoOpt);
        let sportsBallId = "";
        for (const c of (j.classes || [])) {
          const opt = document.createElement("option");
          opt.value = String(c.id);
          opt.textContent = `${c.id}: ${c.label}`;
          if ((c.label || "").toLowerCase() === "sports ball") sportsBallId = String(c.id);
          sel.appendChild(opt);
        }
        if ([...sel.options].some(o => o.value === cur)) sel.value = cur;
        else if (sportsBallId) sel.value = sportsBallId;
      } catch (e) {}
    };

    const loadDevices = async () => {
      try {
        const r = await fetch("/api/analysis/devices");
        if (!r.ok) return;
        const j = await r.json();
        const sel = $("deviceSel");
        const cur = sel.value;
        sel.innerHTML = "";
        for (const d of (j.devices || [])) {
          const opt = document.createElement("option");
          opt.value = d.id;
          opt.textContent = d.label;
          sel.appendChild(opt);
        }
        if ([...sel.options].some(o => o.value === cur)) sel.value = cur;
        else if ([...sel.options].some(o => o.value === (j.default || "auto"))) sel.value = (j.default || "auto");
      } catch (e) {}
    };

    const updateState = (st) => {
      lastState = st;
      $("systemDot").className = "dot ok";
      $("systemStatus").textContent = "Online";

      if ($("camPreset").options.length === 0) {
        for (const k of Object.keys(st.presets || {})) {
          const opt = document.createElement("option");
          opt.value = k;
          opt.textContent = k;
          $("camPreset").appendChild(opt);
        }
        $("camPreset").value = st.defaults.preset || "1080p60";
      }

      const camL = st.cameras?.camL;
      const camR = st.cameras?.camR;
      if (camL) $("camLStatus").textContent = camL.is_open ? `camL:${camL.fps.toFixed(1)}fps` : "camL:offline";
      if (camR) $("camRStatus").textContent = camR.is_open ? `camR:${camR.fps.toFixed(1)}fps` : "camR:offline";

      const ai = st.ai || {};
      $("aiState").textContent = `state:${ai.state || "idle"}`;
      $("aiMode").textContent = `mode:${ai.mode || "auto"}`;
      $("aiActive").textContent = `active:${ai.active_camera || "-"}`;
      $("aiFps").textContent = `fps:${(ai.fps || 0).toFixed(1)}`;
      $("aiConfStat").textContent = `conf:${(ai.last_conf || 0).toFixed(2)}`;
      $("aiSeen").textContent = `last_seen:${ai.last_seen_sec == null ? "-" : Number(ai.last_seen_sec).toFixed(2) + "s"}`;

      const aiCfg = st.ai_config || {};
      const setIfInit = (id, value) => {
        const el = $(id);
        if (!el || el.dataset.init) return;
        if (value != null && value !== "") el.value = value;
        el.dataset.init = "1";
      };
      setIfInit("aiConfInput", aiCfg.conf);
      setIfInit("aiIouInput", aiCfg.iou);
      setIfInit("aiZoomInput", aiCfg.zoom);
      setIfInit("aiSmoothInput", aiCfg.smooth);
      setIfInit("aiLostHold", aiCfg.lost_hold_sec);
      setIfInit("aiDetectEvery", aiCfg.detect_every);
      setIfInit("aiImgsz", aiCfg.imgsz);
      setIfInit("deviceSel", aiCfg.device || "auto");
      if (!$("camSource").dataset.init) {
        const selCam = $("camSelect").value || "camL";
        const camState = st.cameras[selCam];
        if (camState) $("camSource").value = camState.src || "";
        $("camSource").dataset.init = "1";
      }

      const running = !!ai.running;
      $("liveStatusLine").textContent = running
        ? `live tracking running | active=${ai.active_camera || "-"} | state=${ai.state || "-"} | fps=${(ai.fps || 0).toFixed(1)}`
        : "live stopped";
      refreshMainStream();
    };

    const pollState = async () => {
      try {
        const st = await fetchJSON("/api/state");
        updateState(st);
      } catch (e) {}
    };

    const startLoop = () => {
      if (pollTimer) clearInterval(pollTimer);
      pollState();
      pollTimer = setInterval(pollState, 1000);
    };

    $("liveStart").onclick = async () => {
      const payload = {
        active_camera: $("aiActiveInput").value,
        conf: $("aiConfInput").value,
        iou: $("aiIouInput").value,
        device: $("deviceSel").value,
        imgsz: $("aiImgsz").value,
        zoom: $("aiZoomInput").value,
        smooth: $("aiSmoothInput").value,
        lost_hold_sec: $("aiLostHold").value,
        detect_every: $("aiDetectEvery").value,
        model_path: $("modelPath").value,
      };
      if ($("classId").value !== "") payload.class_id = $("classId").value;
      try {
        await fetchJSON("/api/ai/start", { method: "POST", body: JSON.stringify(payload) });
        $("liveStatusLine").textContent = "starting live tracking ...";
        await pollState();
      } catch (e) {
        $("liveStatusLine").textContent = "live start failed";
      }
    };

    $("liveStop").onclick = async () => {
      try {
        await fetchJSON("/api/ai/stop", { method: "POST" });
        $("liveStatusLine").textContent = "live stopped";
      } catch (e) {
        $("liveStatusLine").textContent = "live stop failed";
      }
    };

    $("camApply").onclick = async () => {
      const camId = $("camSelect").value;
      const payload = {
        src: $("camSource").value,
        preset: $("camPreset").value,
        brightness: $("camBrightness").value ? parseInt($("camBrightness").value) : null,
        contrast: $("camContrast").value ? parseInt($("camContrast").value) : null,
        saturation: $("camSaturation").value ? parseInt($("camSaturation").value) : null,
        gain: $("camGain").value ? parseInt($("camGain").value) : null,
      };
      if ($("camAutoWb").value !== "") payload.auto_wb = $("camAutoWb").value === "true";
      if ($("camAutoExp").value !== "") payload.auto_exposure = $("camAutoExp").value === "true";
      try {
        await fetchJSON(`/api/settings/${camId}`, { method: "POST", body: JSON.stringify(payload) });
        $("camStatusLine").textContent = "camera settings applied";
      } catch (e) {
        $("camStatusLine").textContent = "camera settings failed";
      }
    };

    $("camSelect").onchange = () => {
      if (!lastState) return;
      const camState = lastState.cameras?.[$("camSelect").value];
      if (camState) $("camSource").value = camState.src || "";
    };

    $("modelPath").onchange = loadClasses;

    $("viewModeSelect").onchange = (ev) => applyViewMode(ev.target.value);
    $("previewQuality").onchange = () => refreshMainStream(true);
    $("aiActiveInput").onchange = () => {
      if (!(lastState && lastState.ai && lastState.ai.running) && viewMode === "auto") refreshMainStream(true);
    };

    $("fsBtn").onclick = async () => {
      const el = $("liveStage");
      try {
        if (document.fullscreenElement) await document.exitFullscreen();
        else if (el.requestFullscreen) await el.requestFullscreen();
      } catch (e) {}
    };

    applyViewMode("auto");
    loadModels();
    loadDevices();
    startLoop();
  </script>
</body>
</html>
"""


RECORD_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>BallScope - Dual Cam Ball Tracker</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Rubik:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
    :root {
      color-scheme: dark;
      --bg: #0c1217;
      --bg-2: #0f1821;
      --panel: rgba(17, 24, 39, 0.7);
      --panel-solid: rgba(12, 17, 26, 0.82);
      --stroke: rgba(255,255,255,0.08);
      --stroke-strong: rgba(255,255,255,0.16);
      --accent: #ffb454;
      --accent-2: #68d7ff;
      --text: #f5f7ff;
      --muted: rgba(245,247,255,0.65);
      --ok: #7dffb3;
      --warn: #ffcf5d;
      --danger: #ff8f8f;
      --shadow: 0 12px 30px rgba(0,0,0,0.22);
      --ring: rgba(104,215,255,0.32);
      --input-bg: rgba(9, 12, 18, 0.9);
      --chip-bg: rgba(255,255,255,0.02);
      --btn-text: #fffaf1;
    }
    html[data-theme="light"] {
      color-scheme: light;
      --bg: #edf4fb;
      --bg-2: #dfeaf8;
      --panel: rgba(255, 255, 255, 0.88);
      --panel-solid: rgba(251, 253, 255, 0.95);
      --stroke: rgba(20, 32, 52, 0.11);
      --stroke-strong: rgba(20, 32, 52, 0.22);
      --accent: #dd7c1c;
      --accent-2: #0b86cc;
      --text: #101826;
      --muted: rgba(16, 24, 38, 0.64);
      --ok: #149c62;
      --warn: #c58b16;
      --danger: #d14b4b;
      --shadow: 0 10px 26px rgba(20,31,55,0.10);
      --ring: rgba(11, 134, 204, 0.22);
      --input-bg: rgba(255,255,255,0.96);
      --chip-bg: rgba(255,255,255,0.8);
      --btn-text: #ffffff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Rubik", sans-serif;
      color: var(--text);
      background: radial-gradient(900px 500px at 15% -10%, rgba(104,215,255,0.2), transparent),
                  radial-gradient(800px 600px at 90% -20%, rgba(255,180,84,0.15), transparent),
                  linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0)),
                  var(--bg);
      min-height: 100vh;
    }
    .wrap {
      max-width: 1200px;
      margin: 0 auto;
      padding: 28px 18px 60px;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 14px;
      margin-bottom: 18px;
    }
    .header-actions {
      display:flex;
      align-items:center;
      gap:10px;
      flex-wrap: wrap;
    }
    .nav-btn {
      appearance:none;
      border: 1px solid var(--stroke);
      background:
        linear-gradient(180deg, rgba(255,255,255,0.16), rgba(255,255,255,0.03)),
        var(--panel-solid);
      color: var(--text);
      text-decoration: none;
      border-radius: 12px;
      padding: 9px 12px;
      font-size: 13px;
      font-weight: 600;
      display:inline-flex;
      align-items:center;
      justify-content:center;
      cursor:pointer;
      transition: transform .16s ease, border-color .16s ease, box-shadow .16s ease;
      backdrop-filter: blur(12px) saturate(135%);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
    }
    .nav-btn:hover {
      transform: translateY(-1px);
      border-color: var(--stroke-strong);
      box-shadow: 0 8px 16px rgba(0,0,0,0.10);
    }
    .nav-btn:focus-visible {
      outline: 2px solid var(--accent-2);
      outline-offset: 2px;
    }
    h1 {
      margin: 0;
      font-size: 26px;
      letter-spacing: -0.02em;
    }
    .sub { color: var(--muted); font-size: 13px; }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-size: 12px;
      color: var(--muted);
    }
    .dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--warn);
    }
    .dot.ok { background: var(--ok); }
    .grid {
      display: grid;
      grid-template-columns: repeat(12, 1fr);
      gap: 16px;
    }
    .card {
      background:
        linear-gradient(180deg, rgba(255,255,255,0.10), rgba(255,255,255,0) 38%),
        linear-gradient(120deg, rgba(104,215,255,0.05), rgba(255,180,84,0.03)),
        var(--panel);
      border: 1px solid var(--stroke);
      border-radius: 18px;
      padding: 16px;
      backdrop-filter: blur(16px) saturate(135%);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.06), var(--shadow);
    }
    .card h3 {
      margin: 0 0 10px;
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.2em;
      color: var(--muted);
    }
    .preview {
      border-radius: 14px;
      overflow: hidden;
      border: 1px solid rgba(255,255,255,0.06);
      background:
        linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0)),
        var(--panel-solid);
      aspect-ratio: 16/9;
    }
    .preview img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }
    .meta {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 10px;
      font-size: 12px;
      color: var(--muted);
    }
    .chip {
      border: 1px solid var(--stroke);
      background:
        linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0)),
        var(--chip-bg);
      border-radius: 999px;
      padding: 4px 10px;
      font-family: "JetBrains Mono", monospace;
      font-size: 11px;
    }
    label {
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 6px;
    }
    input, select {
      width: 100%;
      padding: 9px 11px;
      border-radius: 12px;
      border: 1px solid var(--stroke);
      background: var(--input-bg);
      color: var(--text);
      font-size: 13px;
      transition: border-color .16s ease, box-shadow .16s ease, background-color .16s ease;
    }
    input:focus, select:focus {
      outline: none;
      border-color: color-mix(in srgb, var(--accent-2) 55%, var(--stroke));
      box-shadow: 0 0 0 4px var(--ring);
    }
    input:disabled, select:disabled {
      opacity: .7;
      cursor: not-allowed;
    }
    .row {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 10px;
    }
    .btns {
      display: flex;
      gap: 10px;
      margin-top: 10px;
      flex-wrap: wrap;
    }
    button {
      padding: 10px 14px;
      border-radius: 12px;
      border: 1px solid var(--stroke);
      background:
        linear-gradient(180deg, rgba(255,255,255,0.16), rgba(255,255,255,0.03)),
        var(--panel-solid);
      color: var(--text);
      font-weight: 600;
      cursor: pointer;
      transition: transform .16s ease, border-color .16s ease, box-shadow .16s ease, background .16s ease;
      backdrop-filter: blur(12px) saturate(130%);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.07);
    }
    button:hover {
      transform: translateY(-1px);
      border-color: var(--stroke-strong);
      box-shadow: 0 8px 16px rgba(0,0,0,0.10);
    }
    button:focus-visible {
      outline: 2px solid var(--accent-2);
      outline-offset: 2px;
    }
    button.primary {
      background:
        linear-gradient(180deg, rgba(255,255,255,.22), rgba(255,255,255,0)),
        linear-gradient(180deg, rgba(255,180,84,0.78), rgba(255,180,84,0.28));
      border-color: rgba(255,180,84,0.56);
      color: var(--btn-text);
      box-shadow: 0 10px 22px rgba(255,180,84,0.18);
    }
    button.primary:hover {
      border-color: rgba(255,180,84,0.76);
      box-shadow: 0 12px 24px rgba(255,180,84,0.24);
    }
    button.ghost { background: transparent; }
    .status {
      margin-top: 8px;
      font-size: 12px;
      color: var(--muted);
      min-height: 18px;
    }
    .span-8 { grid-column: span 8; }
    .span-4 { grid-column: span 4; }
    .span-6 { grid-column: span 6; }
    .span-12 { grid-column: span 12; }
    @media (max-width: 1000px) {
      .span-8, .span-4, .span-6 { grid-column: span 12; }
      header { flex-direction: column; align-items: flex-start; }
      .header-actions { width: 100%; }
    }
    @media (max-width: 680px) {
      .wrap { padding: 18px 12px 40px; }
      .row { grid-template-columns: 1fr; }
      .btns > button { width: 100%; }
      .meta { gap: 8px; }
      .card { padding: 12px; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <div>
        <h1>BallScope Recording</h1>
        <div class="sub">Dual-camera recording and camera tuning. AI controls are available in Live and Analysis.</div>
      </div>
      <div class="header-actions">
        <a class="nav-btn" href="/">Back to Home</a>
        <div class="badge"><span class="dot" id="systemDot"></span><span id="systemStatus">Booting</span></div>
      </div>
    </header>

    <section class="grid">
      <div class="card span-12">
        <h3>Cameras</h3>
        <div class="row">
          <div class="preview" style="aspect-ratio: 4/3;"><img id="camLImg" src="" alt="Cam Left"></div>
          <div class="preview" style="aspect-ratio: 4/3;"><img id="camRImg" src="" alt="Cam Right"></div>
        </div>
        <div class="meta">
          <div class="chip" id="camLStatus">camL</div>
          <div class="chip" id="camRStatus">camR</div>
        </div>
      </div>
    </section>

    <section class="grid" style="margin-top: 16px;">
      <div class="card span-12">
        <h3>Recording</h3>
        <div class="row">
          <div>
            <label>Camera</label>
            <select id="rawWhich">
              <option value="camL">Left</option>
              <option value="camR">Right</option>
              <option value="both">Both</option>
            </select>
          </div>
          <div>
            <label>Output Dir</label>
            <input id="rawOutputDir" type="text" placeholder="recordings" />
          </div>
        </div>
        <div class="row" style="margin-top: 10px;">
          <div>
            <label>Container</label>
            <select id="rawContainer">
              <option value="mkv">MKV (Audio in same file)</option>
              <option value="avi">AVI (Audio separate MP3)</option>
            </select>
          </div>
        </div>
        <div class="row" style="margin-top: 10px;">
          <div>
            <label>Mic (Audio)</label>
            <select id="rawAudioDevice"></select>
          </div>
          <div>
            <label>Audio Bitrate</label>
            <select id="rawAudioBitrate">
              <option value="32000">32 kbps (low)</option>
              <option value="64000" selected>64 kbps</option>
              <option value="96000">96 kbps</option>
            </select>
          </div>
        </div>
        <div class="btns">
          <button class="primary" id="rawStart">Start Recording</button>
          <button class="ghost" id="rawStop">Stop</button>
        </div>
        <div class="status" id="rawStatusLine">idle</div>
      </div>

      <div class="card span-12">
        <h3>Camera Tuning</h3>
        <div class="row">
          <div>
            <label>Camera</label>
            <select id="camSelect">
              <option value="camL">Camera Left</option>
              <option value="camR">Camera Right</option>
            </select>
          </div>
          <div>
            <label>Preset</label>
            <select id="camPreset"></select>
          </div>
        </div>
        <div class="row" style="margin-top: 10px;">
          <div>
            <label>Source (Device)</label>
            <input id="camSource" type="text" placeholder="/dev/video0 or 0" />
          </div>
          <div>
            <label>Apply Source Change</label>
            <div class="status" style="margin-top: 6px; color: var(--muted);">Tip: Jetson: /dev/video0, Mac: 0 or 1</div>
          </div>
        </div>
        <div class="row" style="margin-top: 10px;">
          <div>
            <label>Brightness</label>
            <input id="camBrightness" type="number" />
          </div>
          <div>
            <label>Contrast</label>
            <input id="camContrast" type="number" />
          </div>
        </div>
        <div class="row" style="margin-top: 10px;">
          <div>
            <label>Saturation</label>
            <input id="camSaturation" type="number" />
          </div>
          <div>
            <label>Gain</label>
            <input id="camGain" type="number" />
          </div>
        </div>
        <div class="row" style="margin-top: 10px;">
          <div>
            <label>Auto WB</label>
            <select id="camAutoWb">
              <option value="">Keep</option>
              <option value="true">On</option>
              <option value="false">Off</option>
            </select>
          </div>
          <div>
            <label>Auto Exposure</label>
            <select id="camAutoExp">
              <option value="">Keep</option>
              <option value="true">On</option>
              <option value="false">Off</option>
            </select>
          </div>
        </div>
        <div class="btns">
          <button class="primary" id="camApply">Apply</button>
        </div>
        <div class="status" id="camStatusLine">idle</div>
      </div>
    </section>
  </div>

  <script>
    (() => {
      const KEY = "ballscope-theme";
      const root = document.documentElement;
      const media = window.matchMedia ? window.matchMedia("(prefers-color-scheme: light)") : null;
      const normalize = (value) => (value === "light" || value === "dark" || value === "system") ? value : "system";
      const getPref = () => normalize(localStorage.getItem(KEY));
      const resolve = (pref) => pref === "system" ? ((media && media.matches) ? "light" : "dark") : pref;
      const apply = () => {
        const pref = getPref();
        root.setAttribute("data-theme", resolve(pref));
        root.setAttribute("data-theme-pref", pref);
      };
      if (!localStorage.getItem(KEY)) localStorage.setItem(KEY, "system");
      apply();
      if (media) {
        const onMediaChange = () => { if (getPref() === "system") apply(); };
        if (typeof media.addEventListener === "function") media.addEventListener("change", onMediaChange);
        else if (typeof media.addListener === "function") media.addListener(onMediaChange);
      }
    })();

    const fetchJSON = async (url, opts = {}) => {
      const res = await fetch(url, Object.assign({ headers: { "Content-Type": "application/json" } }, opts));
      if (!res.ok) throw new Error(await res.text());
      return await res.json();
    };

    const $ = (id) => document.getElementById(id);

    let lastState = null;
    const updateState = (st) => {
      lastState = st;
      $("systemDot").className = "dot ok";
      $("systemStatus").textContent = "Online";

    if ($("camPreset").options.length === 0) {
        for (const k of Object.keys(st.presets || {})) {
          const opt = document.createElement("option");
          opt.value = k;
          opt.textContent = k;
          $("camPreset").appendChild(opt);
        }
        $("camPreset").value = st.defaults.preset || "1080p60";
      }

      $("rawOutputDir").value = $("rawOutputDir").value || st.defaults.video_dir;

      const camL = st.cameras.camL;
      const camR = st.cameras.camR;
      if (camL) $("camLStatus").textContent = camL.is_open ? `camL ${camL.fps.toFixed(1)}fps` : "camL offline";
      if (camR) $("camRStatus").textContent = camR.is_open ? `camR ${camR.fps.toFixed(1)}fps` : "camR offline";

      const raw = st.recording_raw || {};
      const rawL = raw.camL && raw.camL.running ? `Raw L -> ${raw.camL.output_path}` : (raw.camL && raw.camL.last_error ? `Raw L err: ${raw.camL.last_error}` : "");
      const rawR = raw.camR && raw.camR.running ? `Raw R -> ${raw.camR.output_path}` : (raw.camR && raw.camR.last_error ? `Raw R err: ${raw.camR.last_error}` : "");
      const rawLine = [rawL, rawR].filter(Boolean).join(" | ");
      $("rawStatusLine").textContent = rawLine || "idle";

      if (!$("camSource").dataset.init) {
        const selCam = $("camSelect").value || "camL";
        const camState = st.cameras[selCam];
        if (camState) {
          $("camSource").value = camState.src || "";
        }
        $("camSource").dataset.init = "1";
      }
    };

    const startLoop = () => {
      setInterval(async () => {
        try {
          const st = await fetchJSON("/api/state");
          updateState(st);
        } catch (e) {}
      }, 1000);
    };

    const initAudioDevices = async () => {
      try {
        const res = await fetchJSON("/api/audio/devices");
        const devices = res.devices || [];
        const sels = [$("rawAudioDevice")];
        for (const sel of sels) {
          if (sel.options.length === 0) {
            for (const d of devices) {
              const opt = document.createElement("option");
              opt.value = d.id;
              opt.textContent = d.label;
              sel.appendChild(opt.cloneNode(true));
            }
          }
        }
      } catch (e) {}
    };

    $("camLImg").src = "/video/cam/camL.mjpg";
    $("camRImg").src = "/video/cam/camR.mjpg";

    $("camApply").onclick = async () => {
      const camId = $("camSelect").value;
      const payload = {
        src: $("camSource").value,
        preset: $("camPreset").value,
        brightness: $("camBrightness").value ? parseInt($("camBrightness").value) : null,
        contrast: $("camContrast").value ? parseInt($("camContrast").value) : null,
        saturation: $("camSaturation").value ? parseInt($("camSaturation").value) : null,
        gain: $("camGain").value ? parseInt($("camGain").value) : null
      };
      if ($("camAutoWb").value !== "") payload.auto_wb = $("camAutoWb").value === "true";
      if ($("camAutoExp").value !== "") payload.auto_exposure = $("camAutoExp").value === "true";
      try {
        await fetchJSON(`/api/settings/${camId}`, { method: "POST", body: JSON.stringify(payload) });
        $("camStatusLine").textContent = "applied";
      } catch (e) {
        $("camStatusLine").textContent = "apply failed";
      }
    };
    $("rawStart").onclick = async () => {
      const payload = {
        which: $("rawWhich").value,
        output_dir: $("rawOutputDir").value,
        audio_device: $("rawAudioDevice").value,
        audio_bitrate: $("rawAudioBitrate").value,
        container: $("rawContainer").value
      };
      try {
        await fetchJSON("/api/raw/start", { method: "POST", body: JSON.stringify(payload) });
        $("rawStatusLine").textContent = "recording started";
      } catch (e) {}
    };

    $("rawStop").onclick = async () => {
      try {
        await fetchJSON("/api/raw/stop", { method: "POST" });
        $("rawStatusLine").textContent = "recording stopped";
      } catch (e) {}
    };

    $("camSelect").onchange = () => {
      if (!lastState) return;
      const sel = $("camSelect").value;
      const camState = lastState.cameras[sel];
      if (camState) {
        $("camSource").value = camState.src || "";
      }
    };

    startLoop();
    initAudioDevices();
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(HOME_HTML)


@app.get("/record", response_class=HTMLResponse)
def record_page():
    return HTMLResponse(RECORD_HTML)


@app.get("/live", response_class=HTMLResponse)
def live_page():
    return HTMLResponse(LIVE_HTML)


@app.get("/analysis", response_class=HTMLResponse)
def analysis_page():
    return HTMLResponse(ANALYSIS_HTML)


@app.get("/api/analysis/models")
def analysis_models():
    models = _analysis_list_model_files()
    return {"models": models, "default": ANALYSIS_DEFAULT_MODEL}


@app.get("/api/analysis/classes")
def analysis_classes(model_path: str = ANALYSIS_DEFAULT_MODEL):
    classes = _analysis_model_classes(model_path)
    return {"model_path": model_path, "classes": classes}


@app.get("/api/analysis/devices")
def analysis_devices():
    devices = _analysis_detect_runtime_devices()
    return {"devices": devices, "default": "auto"}


@app.post("/api/analysis/upload")
async def analysis_upload(
    request: Request,
    class_id: Optional[int] = None,
    conf: float = 0.32,
    iou: float = 0.35,
    model_path: str = ANALYSIS_DEFAULT_MODEL,
    device: str = "auto",
    zoom: float = 1.6,
    speed_up: bool = False,
    crop_x: float = 0.0,
    crop_y: float = 0.0,
    crop_w: float = 1.0,
    crop_h: float = 1.0,
):
    os.makedirs(ANALYSIS_UPLOAD_DIR, exist_ok=True)
    available_models = set(_analysis_list_model_files())
    if model_path not in available_models:
        model_path = ANALYSIS_DEFAULT_MODEL
    upload_filename = request.headers.get("x-filename", "upload.mp4")
    ext = Path(upload_filename).suffix.lower() or ".mp4"
    if ext not in {".mp4", ".mov", ".mkv", ".avi", ".m4v"}:
        ext = ".mp4"
    session_id = uuid.uuid4().hex[:10]
    target_path = os.path.join(ANALYSIS_UPLOAD_DIR, f"{session_id}{ext}")
    max_bytes = 8 * 1024 * 1024 * 1024  # 8 GB guardrail for Jetson sessions
    written = 0
    try:
        with open(target_path, "wb") as out:
            async for chunk in request.stream():
                if not chunk:
                    continue
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(status_code=413, detail="Upload too large.")
                out.write(chunk)
    except HTTPException:
        try:
            os.remove(target_path)
        except Exception:
            pass
        raise
    except Exception as exc:
        try:
            os.remove(target_path)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}")

    if written <= 0:
        try:
            os.remove(target_path)
        except Exception:
            pass
        raise HTTPException(status_code=400, detail="Empty upload body.")
    print(f"[analysis] upload complete sid={session_id} file={target_path} bytes={written}")

    crop = {
        "x": max(0.0, min(1.0, float(crop_x))),
        "y": max(0.0, min(1.0, float(crop_y))),
        "w": max(0.05, min(1.0, float(crop_w))),
        "h": max(0.05, min(1.0, float(crop_h))),
    }
    if crop["x"] + crop["w"] > 1.0:
        crop["w"] = max(0.05, 1.0 - crop["x"])
    if crop["y"] + crop["h"] > 1.0:
        crop["h"] = max(0.05, 1.0 - crop["y"])

    with ANALYSIS_LOCK:
        ANALYSIS_SESSIONS[session_id] = {
            "session_id": session_id,
            "video_path": target_path,
            "model_path": model_path,
            "class_id": class_id,
            "conf": max(0.01, min(1.0, float(conf))),
            "iou": max(0.01, min(1.0, float(iou))),
            "device": device,
            "zoom": max(1.0, min(4.0, float(zoom))),
            "speed_up": bool(speed_up),
            "crop": crop,
            "running": False,
            "state": "queued",
            "frames": 0.0,
            "total_frames": 0.0,
            "progress_pct": 0.0,
            "eta_sec": None,
            "detections": 0.0,
            "fps": 0.0,
            "stream_count": 0,
            "device_used": None,
            "segments_saved": 0,
            "last_conf": 0.0,
            "output_path": None,
            "recovery_dir": None,
            "last_error": None,
        }

    thread = threading.Thread(target=_analysis_process_video, args=(session_id, target_path), daemon=True)
    thread.start()
    print(
        f"[analysis] job started sid={session_id} model={model_path} class_id={class_id} "
        f"device={device} zoom={zoom} speed_up={bool(speed_up)}"
    )
    return {"ok": True, "session_id": session_id}


@app.get("/api/analysis/status/{session_id}")
def analysis_status(session_id: str):
    with ANALYSIS_LOCK:
        sess = ANALYSIS_SESSIONS.get(session_id)
        if not sess:
            raise HTTPException(status_code=404, detail="Unknown session_id")
        return {
            "session_id": session_id,
            "state": sess.get("state"),
            "running": sess.get("running"),
            "frames": sess.get("frames", 0.0),
            "total_frames": sess.get("total_frames", 0.0),
            "progress_pct": sess.get("progress_pct", 0.0),
            "eta_sec": sess.get("eta_sec"),
            "detections": sess.get("detections", 0.0),
            "fps": sess.get("fps", 0.0),
            "input_fps": sess.get("input_fps", 0.0),
            "stream_count": sess.get("stream_count", 0),
            "device_used": sess.get("device_used"),
            "speed_up": bool(sess.get("speed_up", False)),
            "detect_every": sess.get("detect_every"),
            "imgsz": sess.get("imgsz"),
            "segments_saved": sess.get("segments_saved", 0),
            "last_conf": sess.get("last_conf", 0.0),
            "last_seen_sec": sess.get("last_seen_sec"),
            "output_path": sess.get("output_path"),
            "output_url": f"/api/analysis/result/{session_id}" if sess.get("output_path") else None,
            "last_error": sess.get("last_error"),
            "crop": sess.get("crop"),
        }


@app.get("/api/analysis/result/{session_id}")
def analysis_result(session_id: str):
    with ANALYSIS_LOCK:
        sess = ANALYSIS_SESSIONS.get(session_id)
        if not sess:
            raise HTTPException(status_code=404, detail="Unknown session_id")
        output_path = sess.get("output_path")
    if not output_path or not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Result not available yet")
    return FileResponse(output_path, media_type="video/mp4", filename=os.path.basename(output_path))


@app.get("/video/analysis/{session_id}.mjpg")
def analysis_stream(session_id: str):
    with ANALYSIS_LOCK:
        if session_id not in ANALYSIS_SESSIONS:
            raise HTTPException(status_code=404, detail="Unknown session_id")

    def gen():
        while True:
            with ANALYSIS_LOCK:
                sess = ANALYSIS_SESSIONS.get(session_id)
                if not sess:
                    break
                frame = sess.get("frame")
                running = bool(sess.get("running"))
            if frame is None:
                if not running:
                    break
                time.sleep(0.03)
                continue
            ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if not ok:
                time.sleep(0.02)
                continue
            yield b"--frame\r\nContent-Type:image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
            time.sleep(0.01)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")
