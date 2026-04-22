from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import cv2
import numpy as np

from .analysis_models import AnalysisDetection, BaseAnalysisModel
from .master_canvas import (
    MasterCanvasConfig,
    MasterCanvasLayout,
    assemble_master_canvas,
    estimate_master_canvas_layout,
    map_box_to_master,
    map_point_to_master,
)


BallTrackPhase = Literal["UNKNOWN", "TRACKED", "HOLD_SHORT", "LOST_SHORT", "LOST_LONG"]
DetectionKind = Literal["detected", "predicted"]


@dataclass
class StreamHypothesis:
    stream_idx: int
    stream_label: str
    detection: AnalysisDetection
    kind: DetectionKind
    source_box: tuple[int, int, int, int]
    master_box: tuple[int, int, int, int]
    master_center: tuple[float, float]
    velocity: tuple[float, float]
    quality: float
    used_roi: bool


@dataclass
class StreamTrackerState:
    label: str
    center: Optional[tuple[float, float]] = None
    velocity: tuple[float, float] = (0.0, 0.0)
    box: Optional[tuple[int, int, int, int]] = None
    last_confidence: float = 0.0
    last_detected_frame: Optional[int] = None
    last_real_detection_frame: Optional[int] = None
    last_full_scan_frame: Optional[int] = None
    last_roi_scan_frame: Optional[int] = None
    full_scan_hits: int = 0
    roi_hits: int = 0
    misses: int = 0

    def has_recent_real_detection(self, frame_idx: int, keep_frames: int) -> bool:
        return self.last_real_detection_frame is not None and (frame_idx - self.last_real_detection_frame) <= keep_frames

    def predicted_center(self) -> Optional[tuple[float, float]]:
        if self.center is None:
            return None
        return (self.center[0] + self.velocity[0], self.center[1] + self.velocity[1])


@dataclass
class FusedBallState:
    phase: BallTrackPhase = "UNKNOWN"
    center: Optional[tuple[float, float]] = None
    velocity: tuple[float, float] = (0.0, 0.0)
    confidence: float = 0.0
    last_seen_frame: Optional[int] = None
    last_real_center: Optional[tuple[float, float]] = None
    last_safe_center: Optional[tuple[float, float]] = None
    last_safe_box: Optional[tuple[int, int, int, int]] = None
    last_sources: tuple[str, ...] = ()
    pending_center: Optional[tuple[float, float]] = None
    pending_box: Optional[tuple[int, int, int, int]] = None
    pending_frames: int = 0


@dataclass
class CameraViewState:
    center: Optional[tuple[float, float]] = None
    velocity: tuple[float, float] = (0.0, 0.0)
    zoom: float = 1.0
    shot_target_center: Optional[tuple[float, float]] = None
    shot_target_zoom: float = 1.0
    last_shot_update_frame: int = 0


@dataclass
class OfflineAnalysisEngineConfig:
    model: BaseAnalysisModel
    model_backend: str
    model_resolution: Optional[int]
    predict_device: str
    source_labels: Sequence[str]
    class_id: Optional[int]
    conf: float
    iou: float
    crop: Dict[str, float]
    default_zoom: float
    roi_imgsz: int
    speed_up: bool
    target_output_size: Optional[tuple[int, int]]
    master_canvas_config: MasterCanvasConfig = field(default_factory=MasterCanvasConfig)
    target_fps: float = 30.0


@dataclass
class AnalysisFrameResult:
    render_frame: np.ndarray
    preview_frame: np.ndarray
    master_debug_frame: Optional[np.ndarray]
    phase: BallTrackPhase
    detected: bool
    remembered: bool
    confidence: float
    zoom: float
    focus_stream_label: str
    view_half: str
    field_side: str
    fusion_sources: tuple[str, ...]


def _clamp_int(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def _crop_bounds(frame: np.ndarray, crop: Dict[str, float]) -> tuple[int, int, int, int]:
    h, w = frame.shape[:2]
    x = _clamp_int(int(float(crop.get("x", 0.0)) * w), 0, max(0, w - 1))
    y = _clamp_int(int(float(crop.get("y", 0.0)) * h), 0, max(0, h - 1))
    width = _clamp_int(int(float(crop.get("w", 1.0)) * w), 1, max(1, w - x))
    height = _clamp_int(int(float(crop.get("h", 1.0)) * h), 1, max(1, h - y))
    return x, y, width, height


def _center_from_box(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _area_ratio(box: tuple[int, int, int, int], frame_shape: tuple[int, int]) -> float:
    x1, y1, x2, y2 = box
    area = max(1, x2 - x1) * max(1, y2 - y1)
    return float(area) / float(max(1, frame_shape[0] * frame_shape[1]))


def _score_ball_scale(area_ratio: float) -> float:
    target = 0.00018
    spread = 0.8
    return max(0.25, 1.0 - min(1.0, abs(math.log(max(area_ratio, 1e-6) / target)) / spread))


def _round_up_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 1:
        return max(1, int(value))
    return int(((max(1, int(value)) + multiple - 1) // multiple) * multiple)


def _native_imgsz_for_frame(frame: np.ndarray) -> tuple[int, int]:
    h, w = frame.shape[:2]
    # Full-frame reacquire must preserve ball detail, so we do not downscale here.
    # Some backends require input dimensions aligned to a model stride/window size.
    return _round_up_to_multiple(h, 32), _round_up_to_multiple(w, 32)


def _tile_starts(length: int, tile_size: int, stride: int) -> List[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, max(1, length - tile_size + 1), max(1, stride)))
    end_start = max(0, length - tile_size)
    if not starts or starts[-1] != end_start:
        starts.append(end_start)
    return starts


class OfflineAnalysisEngine:
    def __init__(self, config: OfflineAnalysisEngineConfig):
        self.config = config
        self.streams = [StreamTrackerState(label=str(label)) for label in config.source_labels]
        self.fused = FusedBallState()
        self.camera = CameraViewState(zoom=max(1.0, float(config.default_zoom)))
        self.output_size = config.target_output_size
        self.last_focus_stream = str(config.source_labels[0]) if config.source_labels else "main"
        self.full_scan_phase = {idx: idx % 2 for idx in range(len(self.streams))}
        self.hold_frames = max(3, int(round(config.target_fps * 0.45)))
        self.lost_short_frames = max(self.hold_frames + 1, int(round(config.target_fps * 1.4)))
        self.lost_long_frames = max(self.lost_short_frames + 1, int(round(config.target_fps * 3.0)))
        self.roi_interval_tracked = max(8, int(round(config.target_fps * 0.70)))
        self.roi_interval_hold = max(10, int(round(config.target_fps * 0.90)))
        self.full_interval_unknown = max(8, int(round(config.target_fps * 1.0)))
        self.full_interval_lost_short = max(6, int(round(config.target_fps * 0.7)))
        self.full_interval_hold = max(10, int(round(config.target_fps * 1.2)))
        self.full_interval_tracked = max(16, int(round(config.target_fps * 2.0)))
        self.max_ball_speed_px_per_sec = 0.42 * 1920.0
        self.large_jump_confirm_frames = max(2, int(round(config.target_fps * 0.18)))
        self.shot_hold_frames = max(24, int(round(config.target_fps * 2.2)))
        self.shot_break_distance_px = 220.0
        self.shot_break_distance_y_px = 140.0

    def process_frame(
        self,
        frame_idx: int,
        frames: Sequence[Optional[np.ndarray]],
        master_frame: Optional[np.ndarray],
        master_layout: Optional[MasterCanvasLayout],
    ) -> Optional[AnalysisFrameResult]:
        render_base = master_frame if master_frame is not None else next((frame for frame in frames if frame is not None), None)
        if render_base is None:
            return None

        hypotheses: List[StreamHypothesis] = []
        for idx, frame in enumerate(frames):
            if frame is None or idx >= len(self.streams):
                continue
            hypothesis = self._process_stream(frame_idx, idx, frame, master_layout)
            if hypothesis is not None:
                hypotheses.append(hypothesis)

        fused_hypothesis = self._fuse_hypotheses(frame_idx, hypotheses, render_base.shape[:2], master_layout)
        self._update_camera_controller(frame_idx, fused_hypothesis, render_base.shape[:2])

        render = self._render_virtual_camera(render_base)
        preview_frame = render
        debug_frame = None if self.config.speed_up else self._build_debug_frame(render_base, master_layout, hypotheses, fused_hypothesis)

        if self.fused.phase == "UNKNOWN":
            focus_sources = ("wide",)
            focus_label = "wide"
        else:
            focus_sources = self.fused.last_sources or ((fused_hypothesis.stream_label,) if fused_hypothesis is not None else (self.last_focus_stream,))
            focus_label = focus_sources[0] if len(focus_sources) == 1 else "fused"
        self.last_focus_stream = focus_label
        focus_x = self.fused.center[0] if self.fused.center is not None else render_base.shape[1] / 2.0
        view_half = "left" if focus_x < (render_base.shape[1] / 2.0) else "right"
        field_side = "center" if self.fused.phase == "UNKNOWN" else (focus_label if focus_label in {"left", "right"} else view_half)
        return AnalysisFrameResult(
            render_frame=render,
            preview_frame=preview_frame,
            master_debug_frame=debug_frame,
            phase=self.fused.phase,
            detected=bool(fused_hypothesis and fused_hypothesis.kind == "detected"),
            remembered=bool(fused_hypothesis and fused_hypothesis.kind == "predicted"),
            confidence=self.fused.confidence,
            zoom=self.camera.zoom,
            focus_stream_label=focus_label,
            view_half=view_half,
            field_side=field_side,
            fusion_sources=focus_sources,
        )

    def _process_stream(
        self,
        frame_idx: int,
        stream_idx: int,
        frame: np.ndarray,
        master_layout: Optional[MasterCanvasLayout],
    ) -> Optional[StreamHypothesis]:
        tracker = self.streams[stream_idx]
        crop_x, crop_y, crop_w, crop_h = _crop_bounds(frame, self.config.crop)
        crop_frame = frame[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
        if crop_frame.size == 0:
            return None

        should_try_roi = (
            tracker.box is not None
            and tracker.has_recent_real_detection(frame_idx, self.lost_short_frames)
            and self._should_run_roi_scan(frame_idx, tracker)
        )
        best_detection = None
        used_roi = False
        if should_try_roi:
            roi = self._roi_search_region(frame, tracker, crop_x, crop_y, crop_w, crop_h)
            if roi is not None:
                tracker.last_roi_scan_frame = frame_idx
                rx1, ry1, rx2, ry2 = roi
                roi_frame = frame[ry1:ry2, rx1:rx2]
                best_detection = self._detect_best(
                    image=roi_frame,
                    off_x=rx1,
                    off_y=ry1,
                    tracker=tracker,
                    frame_shape=frame.shape[:2],
                    preserve_resolution=False,
                )
                used_roi = best_detection is not None

        if best_detection is None and self._should_run_full_scan(frame_idx, stream_idx, tracker):
            tracker.last_full_scan_frame = frame_idx
            best_detection = self._detect_best(
                image=frame,
                off_x=0,
                off_y=0,
                tracker=tracker,
                frame_shape=frame.shape[:2],
                preserve_resolution=True,
            )

        if best_detection is not None:
            det = best_detection
            self._apply_detection(tracker, det, frame_idx, used_roi=used_roi)
            return self._build_hypothesis(stream_idx, tracker.label, det, "detected", tracker.velocity, master_layout, frame.shape[:2], used_roi)

        tracker.misses += 1
        if tracker.box is None or not tracker.has_recent_real_detection(frame_idx, self.hold_frames):
            return None
        predicted = self._predicted_detection(tracker, crop_x, crop_y, crop_w, crop_h)
        if predicted is None:
            return None
        tracker.center = (predicted.cx, predicted.cy)
        tracker.box = (predicted.x1, predicted.y1, predicted.x2, predicted.y2)
        tracker.velocity = (tracker.velocity[0] * 0.84, tracker.velocity[1] * 0.84)
        tracker.last_confidence = predicted.confidence
        return self._build_hypothesis(stream_idx, tracker.label, predicted, "predicted", tracker.velocity, master_layout, frame.shape[:2], used_roi=False)

    def _should_run_roi_scan(self, frame_idx: int, tracker: StreamTrackerState) -> bool:
        if tracker.last_roi_scan_frame is None:
            return True
        if self.fused.phase == "TRACKED":
            interval = self.roi_interval_tracked
        else:
            interval = self.roi_interval_hold
        if tracker.misses >= 1:
            interval = min(interval, max(3, int(round(self.config.target_fps * 0.25))))
        return (frame_idx - tracker.last_roi_scan_frame) >= max(1, interval)

    def _should_run_full_scan(self, frame_idx: int, stream_idx: int, tracker: StreamTrackerState) -> bool:
        if tracker.last_full_scan_frame is None:
            return frame_idx >= self.full_scan_phase.get(stream_idx, 0)
        if tracker.last_real_detection_frame is None:
            interval = self.full_interval_unknown
        elif self.fused.phase in {"UNKNOWN", "LOST_LONG"}:
            interval = self.full_interval_unknown
        elif self.fused.phase == "LOST_SHORT":
            interval = self.full_interval_lost_short
        elif self.fused.phase == "HOLD_SHORT":
            interval = self.full_interval_hold
        else:
            interval = self.full_interval_tracked
        if tracker.misses >= 2:
            interval = min(interval, self.full_interval_lost_short)
        if self.fused.phase in {"UNKNOWN", "LOST_LONG"}:
            interval = max(interval, self.full_interval_unknown)
        return (frame_idx - tracker.last_full_scan_frame) >= max(1, interval)

    def _roi_search_region(
        self,
        frame: np.ndarray,
        tracker: StreamTrackerState,
        crop_x: int,
        crop_y: int,
        crop_w: int,
        crop_h: int,
    ) -> Optional[tuple[int, int, int, int]]:
        if tracker.box is None:
            return None
        pred = tracker.predicted_center() or tracker.center
        if pred is None:
            pred = _center_from_box(tracker.box)
        x1, y1, x2, y2 = tracker.box
        bw = max(10, x2 - x1)
        bh = max(10, y2 - y1)
        expand = 4.6 if self.config.model_backend == "rfdetr" else 3.8
        sw = max(144, int(round(bw * expand)))
        sh = max(144, int(round(bh * expand)))
        rx1 = _clamp_int(int(round(pred[0] - sw / 2)), crop_x, crop_x + crop_w - 2)
        ry1 = _clamp_int(int(round(pred[1] - sh / 2)), crop_y, crop_y + crop_h - 2)
        rx2 = _clamp_int(rx1 + sw, rx1 + 1, crop_x + crop_w)
        ry2 = _clamp_int(ry1 + sh, ry1 + 1, crop_y + crop_h)
        if (rx2 - rx1) < 8 or (ry2 - ry1) < 8:
            return None
        return rx1, ry1, rx2, ry2

    def _detect_best(
        self,
        image: np.ndarray,
        off_x: int,
        off_y: int,
        tracker: StreamTrackerState,
        frame_shape: tuple[int, int],
        preserve_resolution: bool,
    ) -> Optional[AnalysisDetection]:
        if image.size == 0:
            return None
        if preserve_resolution and self.config.model_backend == "rfdetr" and self.config.model_resolution:
            return self._detect_best_tiled_rfdetr(image=image, off_x=off_x, off_y=off_y, tracker=tracker, frame_shape=frame_shape)
        imgsz: int | tuple[int, int] | None
        if preserve_resolution:
            imgsz = _native_imgsz_for_frame(image)
        else:
            imgsz = max(256, int(self.config.roi_imgsz))
        detections = self.config.model.predict(
            image=image,
            conf=self.config.conf,
            iou=self.config.iou,
            imgsz=imgsz,
            class_id=self.config.class_id,
            device=self.config.predict_device,
        )
        best_score = -1e9
        best_detection = None
        ref_center = tracker.predicted_center() or tracker.center
        for det in detections:
            candidate = AnalysisDetection(
                class_id=int(det.class_id),
                confidence=float(det.confidence),
                x1=off_x + int(det.x1),
                y1=off_y + int(det.y1),
                x2=off_x + int(det.x2),
                y2=off_y + int(det.y2),
                label=det.label,
            )
            score = self._score_stream_detection(candidate, ref_center, frame_shape)
            if score > best_score:
                best_score = score
                best_detection = candidate
        return best_detection

    def _detect_best_tiled_rfdetr(
        self,
        image: np.ndarray,
        off_x: int,
        off_y: int,
        tracker: StreamTrackerState,
        frame_shape: tuple[int, int],
    ) -> Optional[AnalysisDetection]:
        tile_size = _round_up_to_multiple(int(self.config.model_resolution or 1024), 32)
        stride = max(tile_size // 2, int(round(tile_size * 0.65)))
        h, w = image.shape[:2]
        best_score = -1e9
        best_detection = None
        ref_center = tracker.predicted_center() or tracker.center
        for ty in _tile_starts(h, tile_size, stride):
            for tx in _tile_starts(w, tile_size, stride):
                tile = image[ty:min(h, ty + tile_size), tx:min(w, tx + tile_size)]
                if tile.size == 0:
                    continue
                detections = self.config.model.predict(
                    image=tile,
                    conf=self.config.conf,
                    iou=self.config.iou,
                    imgsz=(tile_size, tile_size),
                    class_id=self.config.class_id,
                    device=self.config.predict_device,
                )
                for det in detections:
                    candidate = AnalysisDetection(
                        class_id=int(det.class_id),
                        confidence=float(det.confidence),
                        x1=off_x + tx + int(det.x1),
                        y1=off_y + ty + int(det.y1),
                        x2=off_x + tx + int(det.x2),
                        y2=off_y + ty + int(det.y2),
                        label=det.label,
                    )
                    score = self._score_stream_detection(candidate, ref_center, frame_shape)
                    if score > best_score:
                        best_score = score
                        best_detection = candidate
        return best_detection

    def _score_stream_detection(
        self,
        detection: AnalysisDetection,
        ref_center: Optional[tuple[float, float]],
        frame_shape: tuple[int, int],
    ) -> float:
        score = float(detection.confidence) * 1.7
        area_ratio = _area_ratio((detection.x1, detection.y1, detection.x2, detection.y2), frame_shape)
        score *= _score_ball_scale(area_ratio)
        if ref_center is not None:
            dist = math.hypot(detection.cx - ref_center[0], detection.cy - ref_center[1])
            ref_span = max(90.0, min(frame_shape) * 0.22)
            score -= min(0.75, dist / ref_span * 0.30)
        return score

    def _apply_detection(self, tracker: StreamTrackerState, detection: AnalysisDetection, frame_idx: int, used_roi: bool) -> None:
        prev_center = tracker.center
        if prev_center is not None:
            raw_vx = detection.cx - prev_center[0]
            raw_vy = detection.cy - prev_center[1]
            tracker.velocity = (
                tracker.velocity[0] * 0.55 + raw_vx * 0.45,
                tracker.velocity[1] * 0.55 + raw_vy * 0.45,
            )
        tracker.center = (detection.cx, detection.cy)
        tracker.box = (detection.x1, detection.y1, detection.x2, detection.y2)
        tracker.last_confidence = float(detection.confidence)
        tracker.last_detected_frame = frame_idx
        tracker.last_real_detection_frame = frame_idx
        tracker.misses = 0
        if used_roi:
            tracker.roi_hits += 1
        else:
            tracker.full_scan_hits += 1

    def _predicted_detection(
        self,
        tracker: StreamTrackerState,
        crop_x: int,
        crop_y: int,
        crop_w: int,
        crop_h: int,
    ) -> Optional[AnalysisDetection]:
        if tracker.box is None or tracker.center is None:
            return None
        x1, y1, x2, y2 = tracker.box
        bw = max(10, x2 - x1)
        bh = max(10, y2 - y1)
        pred = tracker.predicted_center() or tracker.center
        px1 = _clamp_int(int(round(pred[0] - bw / 2)), crop_x, crop_x + crop_w - 2)
        py1 = _clamp_int(int(round(pred[1] - bh / 2)), crop_y, crop_y + crop_h - 2)
        px2 = _clamp_int(px1 + bw, px1 + 1, crop_x + crop_w)
        py2 = _clamp_int(py1 + bh, py1 + 1, crop_y + crop_h)
        conf = max(0.02, tracker.last_confidence * (0.78 ** max(1, tracker.misses)))
        return AnalysisDetection(
            class_id=int(self.config.class_id or 0),
            confidence=conf,
            x1=px1,
            y1=py1,
            x2=px2,
            y2=py2,
            label="predicted",
        )

    def _build_hypothesis(
        self,
        stream_idx: int,
        stream_label: str,
        detection: AnalysisDetection,
        kind: DetectionKind,
        velocity: tuple[float, float],
        master_layout: Optional[MasterCanvasLayout],
        frame_shape: tuple[int, int],
        used_roi: bool,
    ) -> StreamHypothesis:
        source_box = (detection.x1, detection.y1, detection.x2, detection.y2)
        if master_layout is not None and stream_label in {"left", "right"}:
            master_box = map_box_to_master(stream_label, source_box, master_layout)
            master_center = map_point_to_master(stream_label, detection.cx, detection.cy, master_layout)
        else:
            master_box = source_box
            master_center = (detection.cx, detection.cy)
        quality = float(detection.confidence) * _score_ball_scale(_area_ratio(source_box, frame_shape))
        if kind == "predicted":
            quality *= 0.72
        return StreamHypothesis(
            stream_idx=stream_idx,
            stream_label=stream_label,
            detection=detection,
            kind=kind,
            source_box=source_box,
            master_box=master_box,
            master_center=master_center,
            velocity=velocity,
            quality=quality,
            used_roi=used_roi,
        )

    def _fuse_hypotheses(
        self,
        frame_idx: int,
        hypotheses: Sequence[StreamHypothesis],
        render_shape: tuple[int, int],
        master_layout: Optional[MasterCanvasLayout],
    ) -> Optional[StreamHypothesis]:
        if not hypotheses:
            self._advance_lost_state(frame_idx)
            return None

        predicted_center = None
        if self.fused.center is not None:
            predicted_center = (
                self.fused.center[0] + self.fused.velocity[0],
                self.fused.center[1] + self.fused.velocity[1],
            )

        best = None
        best_score = -1e9
        for hypothesis in hypotheses:
            score = float(hypothesis.quality)
            if predicted_center is not None:
                dist = math.hypot(
                    hypothesis.master_center[0] - predicted_center[0],
                    hypothesis.master_center[1] - predicted_center[1],
                )
                continuity_span = max(120.0, min(render_shape) * 0.22)
                score -= min(0.90, dist / continuity_span * 0.45)
            if master_layout is not None:
                seam_margin = max(18.0, master_layout.overlap_px * 0.22)
                seam_dist = abs(hypothesis.master_center[0] - master_layout.seam_x)
                if seam_dist < seam_margin and hypothesis.kind == "predicted":
                    score -= 0.16
            if score > best_score:
                best_score = score
                best = hypothesis

        detections = [item for item in hypotheses if item.kind == "detected"]
        if len(detections) >= 2:
            pair = self._best_pair_agreement(detections)
            if pair is not None:
                best = pair

        if best is None:
            self._advance_lost_state(frame_idx)
            return None

        if not self._accept_fused_candidate(frame_idx, best, detections):
            self._advance_lost_state(frame_idx)
            return None

        self._update_fused_state(frame_idx, best, hypotheses)
        return best

    def _accept_fused_candidate(
        self,
        frame_idx: int,
        best: StreamHypothesis,
        detections: Sequence[StreamHypothesis],
    ) -> bool:
        if self.fused.last_safe_center is None or self.fused.last_seen_frame is None:
            self.fused.pending_center = None
            self.fused.pending_box = None
            self.fused.pending_frames = 0
            return True
        if len(detections) >= 2:
            self.fused.pending_center = None
            self.fused.pending_box = None
            self.fused.pending_frames = 0
            return True

        frames_since_seen = max(1, frame_idx - self.fused.last_seen_frame)
        max_jump = max(140.0, (self.max_ball_speed_px_per_sec / max(1.0, self.config.target_fps)) * frames_since_seen)
        jump = math.hypot(
            best.master_center[0] - self.fused.last_safe_center[0],
            best.master_center[1] - self.fused.last_safe_center[1],
        )
        if jump <= max_jump or best.detection.confidence >= 0.86:
            self.fused.pending_center = None
            self.fused.pending_box = None
            self.fused.pending_frames = 0
            return True

        if self.fused.pending_center is not None:
            pending_jump = math.hypot(
                best.master_center[0] - self.fused.pending_center[0],
                best.master_center[1] - self.fused.pending_center[1],
            )
            if pending_jump <= max(40.0, max_jump * 0.35):
                self.fused.pending_frames += 1
            else:
                self.fused.pending_center = best.master_center
                self.fused.pending_box = best.master_box
                self.fused.pending_frames = 1
        else:
            self.fused.pending_center = best.master_center
            self.fused.pending_box = best.master_box
            self.fused.pending_frames = 1
        return self.fused.pending_frames >= self.large_jump_confirm_frames

    def _best_pair_agreement(self, detections: Sequence[StreamHypothesis]) -> Optional[StreamHypothesis]:
        best_pair = None
        best_pair_score = -1e9
        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                a = detections[i]
                b = detections[j]
                dist = math.hypot(a.master_center[0] - b.master_center[0], a.master_center[1] - b.master_center[1])
                agreement = (a.quality + b.quality) - min(0.75, dist / 220.0)
                if agreement > best_pair_score:
                    best_pair_score = agreement
                    best_pair = (a, b)
        if best_pair is None:
            return None
        a, b = best_pair
        if best_pair_score < max(a.quality, b.quality) + 0.15:
            return max(best_pair, key=lambda item: item.quality)

        avg_box = (
            int(round((a.master_box[0] + b.master_box[0]) / 2.0)),
            int(round((a.master_box[1] + b.master_box[1]) / 2.0)),
            int(round((a.master_box[2] + b.master_box[2]) / 2.0)),
            int(round((a.master_box[3] + b.master_box[3]) / 2.0)),
        )
        avg_center = _center_from_box(avg_box)
        avg_det = AnalysisDetection(
            class_id=a.detection.class_id,
            confidence=max(a.detection.confidence, b.detection.confidence),
            x1=avg_box[0],
            y1=avg_box[1],
            x2=avg_box[2],
            y2=avg_box[3],
            label=a.detection.label or b.detection.label,
        )
        return StreamHypothesis(
            stream_idx=a.stream_idx,
            stream_label="fused",
            detection=avg_det,
            kind="detected",
            source_box=a.source_box,
            master_box=avg_box,
            master_center=avg_center,
            velocity=((a.velocity[0] + b.velocity[0]) / 2.0, (a.velocity[1] + b.velocity[1]) / 2.0),
            quality=max(a.quality, b.quality) + 0.22,
            used_roi=a.used_roi and b.used_roi,
        )

    def _update_fused_state(
        self,
        frame_idx: int,
        best: StreamHypothesis,
        all_hypotheses: Sequence[StreamHypothesis],
    ) -> None:
        prev_center = self.fused.center
        if prev_center is not None:
            raw_vx = best.master_center[0] - prev_center[0]
            raw_vy = best.master_center[1] - prev_center[1]
            self.fused.velocity = (
                self.fused.velocity[0] * 0.55 + raw_vx * 0.45,
                self.fused.velocity[1] * 0.55 + raw_vy * 0.45,
            )
        else:
            self.fused.velocity = best.velocity
        self.fused.center = best.master_center
        self.fused.confidence = float(best.detection.confidence if best.kind == "detected" else max(0.02, best.detection.confidence * 0.72))
        if best.kind == "detected":
            self.fused.phase = "TRACKED"
            self.fused.last_seen_frame = frame_idx
            self.fused.last_real_center = best.master_center
            self.fused.last_safe_center = best.master_center
            self.fused.last_safe_box = best.master_box
            self.fused.pending_center = None
            self.fused.pending_box = None
            self.fused.pending_frames = 0
        else:
            self._advance_lost_state(frame_idx)
            if self.fused.center is None:
                self.fused.center = best.master_center
        sources = sorted({item.stream_label for item in all_hypotheses if item.kind == "detected"})
        if not sources:
            sources = [best.stream_label]
        self.fused.last_sources = tuple(sources)

    def _advance_lost_state(self, frame_idx: int) -> None:
        last_seen = self.fused.last_seen_frame
        if last_seen is None:
            self.fused.phase = "UNKNOWN"
            self.fused.confidence = 0.0
            return
        missing_frames = frame_idx - last_seen
        if missing_frames <= self.hold_frames:
            self.fused.phase = "HOLD_SHORT"
            self.fused.confidence *= 0.92
        elif missing_frames <= self.lost_short_frames:
            self.fused.phase = "LOST_SHORT"
            self.fused.confidence *= 0.82
        elif missing_frames <= self.lost_long_frames:
            self.fused.phase = "LOST_LONG"
            self.fused.confidence *= 0.68
        else:
            self.fused.phase = "UNKNOWN"
            self.fused.confidence = 0.0
            self.fused.center = None
            self.fused.velocity = (0.0, 0.0)

    def _update_camera_controller(
        self,
        frame_idx: int,
        best: Optional[StreamHypothesis],
        render_shape: tuple[int, int],
    ) -> None:
        h, w = render_shape
        tactical_center = (w / 2.0, h / 2.0)
        if self.camera.center is None:
            self.camera.center = tactical_center
        if self.camera.shot_target_center is None:
            self.camera.shot_target_center = tactical_center
            self.camera.shot_target_zoom = self.camera.zoom
            self.camera.last_shot_update_frame = frame_idx

        proposed_center = tactical_center
        proposed_zoom = 1.0
        if self.fused.phase == "TRACKED" and best is not None:
            lead = 0.04
            proposed_center = self._comfortable_tracking_center(best, render_shape, lead)
            proposed_zoom = self._tracked_zoom(best, render_shape)
        elif self.fused.phase == "HOLD_SHORT":
            proposed_center = self.fused.last_safe_center or self.camera.center
            proposed_zoom = max(1.08, self.camera.zoom * 0.995)
        elif self.fused.phase == "LOST_SHORT":
            safe = self.fused.last_safe_center or self.camera.center
            proposed_center = (
                safe[0] * 0.78 + tactical_center[0] * 0.22,
                safe[1] * 0.78 + tactical_center[1] * 0.22,
            )
            proposed_zoom = max(1.0, self.camera.zoom * 0.985)
        elif self.fused.phase in {"LOST_LONG", "UNKNOWN"}:
            proposed_center = tactical_center
            proposed_zoom = 1.0

        self._update_shot_targets(frame_idx, proposed_center, proposed_zoom)
        target_center = self.camera.shot_target_center or tactical_center
        target_zoom = self.camera.shot_target_zoom

        cx, cy = self.camera.center
        deadzone = max(18.0, min(w, h) * 0.055)
        dx = target_center[0] - cx
        dy = target_center[1] - cy
        if abs(dx) < deadzone:
            dx = 0.0
        if abs(dy) < deadzone:
            dy = 0.0
        accel = max(2.5, min(w, h) * 0.005)
        self.camera.velocity = (
            max(-accel, min(accel, self.camera.velocity[0] * 0.94 + dx * 0.028)),
            max(-accel, min(accel, self.camera.velocity[1] * 0.94 + dy * 0.028)),
        )
        if self.fused.phase in {"LOST_LONG", "UNKNOWN"}:
            self.camera.velocity = (self.camera.velocity[0] * 0.84, self.camera.velocity[1] * 0.84)
        self.camera.center = (
            float(cx + self.camera.velocity[0]),
            float(cy + self.camera.velocity[1]),
        )
        zoom_alpha = 0.008 if self.fused.phase == "TRACKED" else 0.012
        self.camera.zoom = float(max(1.0, min(4.0, self.camera.zoom * (1.0 - zoom_alpha) + target_zoom * zoom_alpha)))
        self.camera.center = (
            float(max(0.0, min(w, self.camera.center[0]))),
            float(max(0.0, min(h, self.camera.center[1]))),
        )

    def _update_shot_targets(
        self,
        frame_idx: int,
        proposed_center: tuple[float, float],
        proposed_zoom: float,
    ) -> None:
        current_target = self.camera.shot_target_center or proposed_center
        dx = proposed_center[0] - current_target[0]
        dy = proposed_center[1] - current_target[1]
        frames_since = frame_idx - self.camera.last_shot_update_frame
        can_refresh = frames_since >= self.shot_hold_frames
        must_refresh = abs(dx) > self.shot_break_distance_px or abs(dy) > self.shot_break_distance_y_px
        if can_refresh or must_refresh or self.fused.phase in {"LOST_LONG", "UNKNOWN"}:
            self.camera.shot_target_center = proposed_center
            if self.fused.phase == "TRACKED":
                self.camera.shot_target_zoom = proposed_zoom
            else:
                self.camera.shot_target_zoom = min(self.camera.shot_target_zoom, proposed_zoom)
            self.camera.last_shot_update_frame = frame_idx

    def _comfortable_tracking_center(
        self,
        best: StreamHypothesis,
        render_shape: tuple[int, int],
        lead: float,
    ) -> tuple[float, float]:
        w = render_shape[1]
        h = render_shape[0]
        out_w, out_h = self.output_size or (w, h)
        zoom = max(1.0, self.camera.zoom)
        view_w = max(1.0, out_w / zoom)
        view_h = max(1.0, out_h / zoom)
        current_center = self.camera.center or (w / 2.0, h / 2.0)
        predicted_ball = (
            best.master_center[0] + self.fused.velocity[0] * lead,
            best.master_center[1] + self.fused.velocity[1] * lead,
        )
        safe_half_w = view_w * 0.14
        safe_half_h = view_h * 0.12
        edge_half_w = view_w * 0.36
        edge_half_h = view_h * 0.32
        dx = predicted_ball[0] - current_center[0]
        dy = predicted_ball[1] - current_center[1]
        target_x = current_center[0]
        target_y = current_center[1]

        if abs(dx) > edge_half_w:
            overflow_x = dx - math.copysign(safe_half_w, dx)
            target_x += overflow_x
        if abs(dy) > edge_half_h:
            overflow_y = dy - math.copysign(safe_half_h, dy)
            target_y += overflow_y

        return (target_x, target_y)

    def _tracked_zoom(self, best: StreamHypothesis, render_shape: tuple[int, int]) -> float:
        box = best.master_box
        area_ratio = _area_ratio(box, render_shape)
        zoom = 1.18
        if area_ratio < 0.00014:
            zoom += 0.35
        elif area_ratio < 0.00028:
            zoom += 0.22
        elif area_ratio < 0.00055:
            zoom += 0.10
        if best.detection.confidence < 0.55:
            zoom -= 0.18
        return max(1.0, min(1.55, zoom))

    def _render_virtual_camera(self, render_base: np.ndarray) -> np.ndarray:
        h, w = render_base.shape[:2]
        if self.output_size is None:
            self.output_size = (w, h)
        out_w, out_h = self.output_size
        center = self.camera.center or (w / 2.0, h / 2.0)
        zoom = max(1.0, self.camera.zoom)
        win_w = max(1, int(round(out_w / zoom)))
        win_h = max(1, int(round(out_h / zoom)))
        left = _clamp_int(int(round(center[0] - win_w / 2.0)), 0, max(0, w - win_w))
        top = _clamp_int(int(round(center[1] - win_h / 2.0)), 0, max(0, h - win_h))
        crop = render_base[top:top + win_h, left:left + win_w]
        if crop.size == 0:
            crop = render_base
        interpolation = cv2.INTER_LINEAR if self.config.speed_up else (cv2.INTER_LANCZOS4 if zoom > 1.0 else cv2.INTER_AREA)
        render = cv2.resize(crop, (out_w, out_h), interpolation=interpolation)
        if not self.config.speed_up:
            blur = cv2.GaussianBlur(render, (0, 0), 1.0)
            render = cv2.addWeighted(render, 1.15, blur, -0.15, 0)
        return render

    def _build_debug_frame(
        self,
        render_base: np.ndarray,
        master_layout: Optional[MasterCanvasLayout],
        hypotheses: Sequence[StreamHypothesis],
        best: Optional[StreamHypothesis],
    ) -> np.ndarray:
        vis = render_base.copy()
        for hypothesis in hypotheses:
            x1, y1, x2, y2 = hypothesis.master_box
            color = (0, 220, 0) if hypothesis.kind == "detected" else (0, 165, 255)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                vis,
                f"{hypothesis.stream_label}:{hypothesis.kind[:4]} {hypothesis.detection.confidence:.2f}",
                (x1, max(16, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                1,
            )
        if best is not None and self.fused.center is not None:
            cx, cy = int(round(self.fused.center[0])), int(round(self.fused.center[1]))
            cv2.circle(vis, (cx, cy), 7, (40, 40, 255), -1)
        if self.fused.last_safe_center is not None and self.fused.phase != "TRACKED":
            sx, sy = int(round(self.fused.last_safe_center[0])), int(round(self.fused.last_safe_center[1]))
            cv2.circle(vis, (sx, sy), 10, (255, 220, 60), 2)
        if master_layout is not None:
            cv2.line(vis, (master_layout.seam_x, 0), (master_layout.seam_x, vis.shape[0]), (220, 200, 60), 1)
        cx, cy = self.camera.center or (vis.shape[1] / 2.0, vis.shape[0] / 2.0)
        win_w = max(1, int(round(self.output_size[0] / max(1.0, self.camera.zoom)))) if self.output_size else vis.shape[1]
        win_h = max(1, int(round(self.output_size[1] / max(1.0, self.camera.zoom)))) if self.output_size else vis.shape[0]
        left = _clamp_int(int(round(cx - win_w / 2.0)), 0, max(0, vis.shape[1] - win_w))
        top = _clamp_int(int(round(cy - win_h / 2.0)), 0, max(0, vis.shape[0] - win_h))
        cv2.rectangle(vis, (left, top), (left + win_w, top + win_h), (255, 255, 255), 2)
        cv2.putText(
            vis,
            f"{self.fused.phase} zoom={self.camera.zoom:.2f}",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (180, 220, 255),
            2,
        )
        return vis


def calibrate_master_layout(
    left_path: str,
    right_path: str,
    left_start_sec: float,
    right_start_sec: float,
    media_duration_sec: float,
    config: Optional[MasterCanvasConfig] = None,
) -> Optional[MasterCanvasLayout]:
    cfg = config or MasterCanvasConfig()
    left_cap = cv2.VideoCapture(left_path)
    right_cap = cv2.VideoCapture(right_path)
    if not left_cap.isOpened() or not right_cap.isOpened():
        left_cap.release()
        right_cap.release()
        return None
    try:
        samples: List[MasterCanvasLayout] = []
        left_shape = None
        right_shape = None
        for offset_sec in (0.0, 1.2, 2.6, 4.0):
            left_ts = max(0.0, left_start_sec) + offset_sec
            right_ts = max(0.0, right_start_sec) + offset_sec
            if media_duration_sec > 0:
                left_ts = min(max(0.0, media_duration_sec - 0.35), left_ts)
                right_ts = min(max(0.0, media_duration_sec - 0.35), right_ts)
            left_cap.set(cv2.CAP_PROP_POS_MSEC, left_ts * 1000.0)
            right_cap.set(cv2.CAP_PROP_POS_MSEC, right_ts * 1000.0)
            ok_left, left_frame = left_cap.read()
            ok_right, right_frame = right_cap.read()
            if not ok_left or left_frame is None or not ok_right or right_frame is None:
                continue
            left_shape = left_frame.shape[:2]
            right_shape = right_frame.shape[:2]
            samples.append(estimate_master_canvas_layout(left_frame, right_frame, cfg))
        if not samples or left_shape is None or right_shape is None:
            return None
        right_offset_x = int(round(statistics.median(layout.right_offset_x for layout in samples)))
        left_crop_y = int(round(statistics.median(layout.left_crop_y for layout in samples)))
        right_crop_y = int(round(statistics.median(layout.right_crop_y for layout in samples)))
        left_h, left_w = int(left_shape[0]), int(left_shape[1])
        right_h, right_w = int(right_shape[0]), int(right_shape[1])
        right_offset_x = max(0, min(left_w - 32, right_offset_x))
        overlap_px = max(32, min(left_w - right_offset_x, right_w))
        width = max(left_w, right_offset_x + right_w)
        height = max(1, min(left_h - left_crop_y, right_h - right_crop_y))
        seam_x = right_offset_x + (overlap_px // 2)
        blend_width = int(round(overlap_px * cfg.seam_blend_ratio))
        blend_width = max(cfg.min_blend_px, min(cfg.max_blend_px, max(1, blend_width)))
        blend_start_x = max(right_offset_x, seam_x - (blend_width // 2))
        blend_end_x = min(right_offset_x + overlap_px, blend_start_x + blend_width)
        return MasterCanvasLayout(
            width=width,
            height=height,
            overlap_px=overlap_px,
            left_offset_x=0,
            left_crop_y=max(0, left_crop_y),
            right_offset_x=right_offset_x,
            right_crop_y=max(0, right_crop_y),
            left_width=left_w,
            right_width=right_w,
            seam_x=seam_x,
            blend_start_x=blend_start_x,
            blend_end_x=blend_end_x,
        )
    finally:
        left_cap.release()
        right_cap.release()


def build_master_frame(
    frames: Sequence[Optional[np.ndarray]],
    source_labels: Sequence[str],
    layout: Optional[MasterCanvasLayout],
    config: Optional[MasterCanvasConfig] = None,
) -> tuple[Optional[np.ndarray], Optional[MasterCanvasLayout]]:
    if "left" not in source_labels or "right" not in source_labels:
        return None, layout
    left_idx = source_labels.index("left")
    right_idx = source_labels.index("right")
    left_frame = frames[left_idx] if left_idx < len(frames) else None
    right_frame = frames[right_idx] if right_idx < len(frames) else None
    if left_frame is None or right_frame is None:
        return None, layout
    master_frame, master_layout = assemble_master_canvas(left_frame, right_frame, config=config, layout=layout)
    return master_frame, master_layout
