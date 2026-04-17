from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class MasterCanvasConfig:
    overlap_ratio: float = 0.44
    min_overlap_ratio: float = 0.22
    max_overlap_ratio: float = 0.68
    seam_blend_ratio: float = 0.02
    search_width: int = 960
    max_vertical_shift_ratio: float = 0.08
    color_gain_min: float = 0.88
    color_gain_max: float = 1.14
    min_blend_px: int = 4
    max_blend_px: int = 16
    orb_features: int = 3000
    base_top_crop_ratio: float = 0.06
    base_bottom_crop_ratio: float = 0.12


@dataclass(frozen=True)
class MasterCanvasLayout:
    width: int
    height: int
    overlap_px: int
    left_offset_x: int
    left_crop_y: int
    right_offset_x: int
    right_crop_y: int
    left_width: int
    right_width: int
    seam_x: int
    blend_start_x: int
    blend_end_x: int


def compute_master_canvas_layout(
    left_shape: Tuple[int, int],
    right_shape: Tuple[int, int],
    config: MasterCanvasConfig,
) -> MasterCanvasLayout:
    left_h, left_w = int(left_shape[0]), int(left_shape[1])
    right_h, right_w = int(right_shape[0]), int(right_shape[1])
    overlap_px = max(32, int(min(left_w, right_w) * config.overlap_ratio))
    overlap_px = min(overlap_px, min(left_w - 1, right_w - 1))
    right_offset_x = max(0, left_w - overlap_px)
    width = max(left_w, right_offset_x + right_w)
    height = min(left_h, right_h)
    blend_width = int(round(overlap_px * config.seam_blend_ratio))
    blend_width = max(config.min_blend_px, min(config.max_blend_px, max(1, blend_width)))
    seam_x = right_offset_x + (overlap_px // 2)
    blend_start_x = max(right_offset_x, seam_x - (blend_width // 2))
    blend_end_x = min(right_offset_x + overlap_px, blend_start_x + blend_width)
    return MasterCanvasLayout(
        width=width,
        height=max(1, height),
        overlap_px=overlap_px,
        left_offset_x=0,
        left_crop_y=0,
        right_offset_x=right_offset_x,
        right_crop_y=0,
        left_width=left_w,
        right_width=right_w,
        seam_x=seam_x,
        blend_start_x=blend_start_x,
        blend_end_x=blend_end_x,
    )


def map_point_to_master(
    stream_label: str,
    x: float,
    y: float,
    layout: MasterCanvasLayout,
) -> Tuple[float, float]:
    label = (stream_label or "").strip().lower()
    if label == "right":
        return float(layout.right_offset_x + x), float(y - layout.right_crop_y)
    return float(layout.left_offset_x + x), float(y - layout.left_crop_y)


def map_box_to_master(
    stream_label: str,
    box: Tuple[int, int, int, int],
    layout: MasterCanvasLayout,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(v) for v in box]
    mx1, my1 = map_point_to_master(stream_label, x1, y1, layout)
    mx2, my2 = map_point_to_master(stream_label, x2, y2, layout)
    return int(mx1), int(my1), int(mx2), int(my2)


def _resize_to_height(frame: np.ndarray, target_h: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if h == target_h:
        return frame
    scale = target_h / float(max(1, h))
    target_w = max(1, int(round(w * scale)))
    return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def _prepare_feature_frame(frame: np.ndarray, cfg: MasterCanvasConfig) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    y0 = int(round(h * 0.18))
    y1 = int(round(h * 0.90))
    x0 = int(round(w * 0.06))
    x1 = int(round(w * 0.94))
    gray = gray[y0:y1, x0:x1]
    if gray.shape[1] > cfg.search_width:
        scale = cfg.search_width / float(gray.shape[1])
        target_h = max(1, int(round(gray.shape[0] * scale)))
        gray = cv2.resize(gray, (cfg.search_width, target_h), interpolation=cv2.INTER_AREA)
    return gray


def _feature_crop_bounds(shape: Tuple[int, int], cfg: MasterCanvasConfig) -> Tuple[int, int, int, int]:
    h, w = int(shape[0]), int(shape[1])
    y0 = int(round(h * 0.18))
    y1 = int(round(h * 0.90))
    x0 = int(round(w * 0.06))
    x1 = int(round(w * 0.94))
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def _estimate_vertical_anchor_y(frame: np.ndarray) -> int:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, _w = gray.shape[:2]
    y0 = int(round(h * 0.20))
    y1 = int(round(h * 0.72))
    roi = gray[y0:y1, :]
    if roi.size == 0:
        return h // 2
    grad_y = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
    row_score = np.mean(np.abs(grad_y), axis=1)
    if row_score.size == 0:
        return h // 2
    row_score = cv2.GaussianBlur(row_score.reshape(-1, 1), (1, 0), sigmaX=7.0).reshape(-1)
    return y0 + int(np.argmax(row_score))


def _estimate_feature_translation(
    left_frame: np.ndarray,
    right_frame: np.ndarray,
    cfg: MasterCanvasConfig,
) -> Optional[Tuple[float, float]]:
    left_feat = _prepare_feature_frame(left_frame, cfg)
    right_feat = _prepare_feature_frame(right_frame, cfg)
    _lx0, _ly0, left_crop_w, left_crop_h = _feature_crop_bounds(left_frame.shape[:2], cfg)
    _rx0, _ry0, right_crop_w, right_crop_h = _feature_crop_bounds(right_frame.shape[:2], cfg)
    if left_feat.size == 0 or right_feat.size == 0:
        return None

    orb = cv2.ORB_create(nfeatures=cfg.orb_features)
    kp_left, desc_left = orb.detectAndCompute(left_feat, None)
    kp_right, desc_right = orb.detectAndCompute(right_feat, None)
    if desc_left is None or desc_right is None or len(kp_left) < 12 or len(kp_right) < 12:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = matcher.knnMatch(desc_left, desc_right, k=2)
    good = []
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.72 * n.distance:
            good.append(m)
    if len(good) < 16:
        return None

    shifts = []
    for m in good:
        p_left = kp_left[m.queryIdx].pt
        p_right = kp_right[m.trainIdx].pt
        shifts.append((p_left[0] - p_right[0], p_left[1] - p_right[1]))
    arr = np.asarray(shifts, dtype=np.float32)
    if arr.shape[0] < 16:
        return None
    median = np.median(arr, axis=0)
    mad = np.median(np.abs(arr - median), axis=0) + 1e-6
    keep = (
        (np.abs(arr[:, 0] - median[0]) < (2.5 * mad[0])) &
        (np.abs(arr[:, 1] - median[1]) < (2.5 * mad[1]))
    )
    filtered = arr[keep]
    if filtered.shape[0] < 12:
        filtered = arr
    median = np.median(filtered, axis=0)

    # The keypoint shift is measured inside the cropped feature region, not across
    # the full frame width/height. Scaling against the full image overestimates the
    # translation and pushes the seam too far sideways.
    scale_x = left_crop_w / float(max(1, left_feat.shape[1]))
    scale_y = left_crop_h / float(max(1, left_feat.shape[0]))
    dx = float(median[0] * scale_x)
    dy = float(median[1] * scale_y)
    return dx, dy


def estimate_master_canvas_layout(
    left_frame: np.ndarray,
    right_frame: np.ndarray,
    config: Optional[MasterCanvasConfig] = None,
) -> MasterCanvasLayout:
    cfg = config or MasterCanvasConfig()
    target_h = max(left_frame.shape[0], right_frame.shape[0])
    left = _resize_to_height(left_frame, target_h)
    right = _resize_to_height(right_frame, target_h)
    default_layout = compute_master_canvas_layout(left.shape[:2], right.shape[:2], cfg)

    shift = _estimate_feature_translation(left, right, cfg)
    if shift is None:
        return default_layout

    dx, dy = shift
    min_overlap = int(round(min(left.shape[1], right.shape[1]) * cfg.min_overlap_ratio))
    max_overlap = int(round(min(left.shape[1], right.shape[1]) * cfg.max_overlap_ratio))
    right_offset_x = int(round(dx))
    right_offset_x = max(0, min(left.shape[1] - min_overlap, right_offset_x))
    overlap_px = min(left.shape[1] - right_offset_x, right.shape[1])
    overlap_px = max(min_overlap, min(max_overlap, overlap_px))
    right_offset_x = left.shape[1] - overlap_px

    max_vertical_shift = int(round(left.shape[0] * cfg.max_vertical_shift_ratio))
    base_top = int(round(left.shape[0] * cfg.base_top_crop_ratio))
    base_bottom = int(round(left.shape[0] * cfg.base_bottom_crop_ratio))
    left_anchor_y = _estimate_vertical_anchor_y(left)
    right_anchor_y = _estimate_vertical_anchor_y(right)
    anchor_delta = int(round(right_anchor_y - left_anchor_y))
    feature_delta = int(round(dy))
    if abs(anchor_delta) >= 8:
        dy_px = anchor_delta
    else:
        dy_px = feature_delta
    dy_px = max(-max_vertical_shift, min(max_vertical_shift, dy_px))
    left_crop_y = max(0, base_top + max(0, -dy_px))
    right_crop_y = max(0, base_top + max(0, dy_px))
    left_bottom_crop = base_bottom + max(0, dy_px)
    right_bottom_crop = base_bottom + max(0, -dy_px)
    height = min(
        left.shape[0] - left_crop_y - left_bottom_crop,
        right.shape[0] - right_crop_y - right_bottom_crop,
    )
    height = max(1, height)

    width = max(left.shape[1], right_offset_x + right.shape[1])
    blend_width = int(round(overlap_px * cfg.seam_blend_ratio))
    blend_width = max(cfg.min_blend_px, min(cfg.max_blend_px, max(1, blend_width)))
    seam_x = right_offset_x + (overlap_px // 2)
    blend_start_x = max(right_offset_x, seam_x - (blend_width // 2))
    blend_end_x = min(right_offset_x + overlap_px, blend_start_x + blend_width)
    return MasterCanvasLayout(
        width=width,
        height=height,
        overlap_px=overlap_px,
        left_offset_x=0,
        left_crop_y=left_crop_y,
        right_offset_x=right_offset_x,
        right_crop_y=right_crop_y,
        left_width=left.shape[1],
        right_width=right.shape[1],
        seam_x=seam_x,
        blend_start_x=blend_start_x,
        blend_end_x=blend_end_x,
    )


def assemble_master_canvas(
    left_frame: Optional[np.ndarray],
    right_frame: Optional[np.ndarray],
    config: Optional[MasterCanvasConfig] = None,
    layout: Optional[MasterCanvasLayout] = None,
) -> tuple[np.ndarray, MasterCanvasLayout]:
    cfg = config or MasterCanvasConfig()
    if left_frame is None and right_frame is None:
        raise ValueError("At least one frame is required.")
    if left_frame is None:
        frame = right_frame.copy()
        h, w = frame.shape[:2]
        single_layout = MasterCanvasLayout(w, h, 0, 0, 0, 0, 0, w, w, w, w, w)
        return frame, single_layout
    if right_frame is None:
        frame = left_frame.copy()
        h, w = frame.shape[:2]
        single_layout = MasterCanvasLayout(w, h, 0, 0, 0, 0, 0, w, w, w, w, w)
        return frame, single_layout

    target_h = max(left_frame.shape[0], right_frame.shape[0])
    left = _resize_to_height(left_frame, target_h)
    right = _resize_to_height(right_frame, target_h)
    use_layout = layout or estimate_master_canvas_layout(left, right, cfg)

    left_crop_y = max(0, min(use_layout.left_crop_y, max(0, left.shape[0] - 1)))
    right_crop_y = max(0, min(use_layout.right_crop_y, max(0, right.shape[0] - 1)))
    common_h = min(left.shape[0] - left_crop_y, right.shape[0] - right_crop_y, use_layout.height)
    common_h = max(1, common_h)
    left = left[left_crop_y:left_crop_y + common_h, :]
    right = right[right_crop_y:right_crop_y + common_h, :]

    overlap_x1 = max(0, use_layout.right_offset_x)
    overlap_x2 = min(use_layout.left_width, use_layout.right_offset_x + use_layout.right_width)
    overlap_px = max(0, overlap_x2 - overlap_x1)
    width = max(use_layout.left_width, use_layout.right_offset_x + use_layout.right_width)

    canvas = np.zeros((common_h, width, 3), dtype=left.dtype)
    canvas[:, :use_layout.left_width] = left

    if overlap_px <= 0:
        canvas[:, use_layout.right_offset_x:use_layout.right_offset_x + use_layout.right_width] = right
        final_layout = MasterCanvasLayout(
            width=width,
            height=common_h,
            overlap_px=0,
            left_offset_x=0,
            left_crop_y=left_crop_y,
            right_offset_x=use_layout.right_offset_x,
            right_crop_y=right_crop_y,
            left_width=use_layout.left_width,
            right_width=use_layout.right_width,
            seam_x=use_layout.right_offset_x,
            blend_start_x=use_layout.right_offset_x,
            blend_end_x=use_layout.right_offset_x,
        )
        return canvas, final_layout

    right_overlap_start = overlap_x1 - use_layout.right_offset_x
    right_overlap_end = right_overlap_start + overlap_px

    overlap_left = left[:, overlap_x1:overlap_x2]
    overlap_right = right[:, right_overlap_start:right_overlap_end]
    left_mean = overlap_left.reshape(-1, 3).mean(axis=0)
    right_mean = overlap_right.reshape(-1, 3).mean(axis=0)
    gain = left_mean / np.maximum(right_mean, 1.0)
    gain = np.clip(gain, cfg.color_gain_min, cfg.color_gain_max).astype(np.float32)
    right = np.clip(right.astype(np.float32) * gain.reshape(1, 1, 3), 0, 255).astype(left.dtype)
    overlap_right = right[:, right_overlap_start:right_overlap_end]

    if use_layout.right_offset_x > 0:
        canvas[:, :use_layout.right_offset_x] = left[:, :use_layout.right_offset_x]

    seam_x = use_layout.seam_x
    blend_start = max(overlap_x1, min(use_layout.blend_start_x, overlap_x2))
    blend_end = max(blend_start, min(use_layout.blend_end_x, overlap_x2))

    left_only_rel = max(0, blend_start - overlap_x1)
    blend_start_rel = max(0, blend_start - overlap_x1)
    blend_end_rel = max(blend_start_rel, min(overlap_px, blend_end - overlap_x1))

    overlap_canvas = overlap_left.copy()
    if blend_end_rel < overlap_px:
        overlap_canvas[:, blend_end_rel:] = overlap_right[:, blend_end_rel:]
    if blend_end_rel > blend_start_rel:
        alpha = np.linspace(0.0, 1.0, blend_end_rel - blend_start_rel, dtype=np.float32).reshape(1, -1, 1)
        left_blend = overlap_left[:, blend_start_rel:blend_end_rel].astype(np.float32)
        right_blend = overlap_right[:, blend_start_rel:blend_end_rel].astype(np.float32)
        overlap_canvas[:, blend_start_rel:blend_end_rel] = np.clip(
            left_blend * (1.0 - alpha) + right_blend * alpha,
            0,
            255,
        ).astype(left.dtype)
    if left_only_rel > 0:
        overlap_canvas[:, :left_only_rel] = overlap_left[:, :left_only_rel]

    canvas[:, overlap_x1:overlap_x2] = overlap_canvas
    if overlap_x2 < width:
        canvas[:, overlap_x2:use_layout.right_offset_x + use_layout.right_width] = right[:, right_overlap_end:]

    final_layout = MasterCanvasLayout(
        width=width,
        height=common_h,
        overlap_px=overlap_px,
        left_offset_x=0,
        left_crop_y=left_crop_y,
        right_offset_x=use_layout.right_offset_x,
        right_crop_y=right_crop_y,
        left_width=use_layout.left_width,
        right_width=use_layout.right_width,
        seam_x=seam_x,
        blend_start_x=blend_start,
        blend_end_x=blend_end,
    )
    return canvas, final_layout
