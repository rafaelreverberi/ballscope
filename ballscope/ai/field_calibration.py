from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from .master_canvas import MasterCanvasConfig, MasterCanvasLayout


def _field_line_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (30, 35, 35), (95, 255, 255))
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    field = cv2.dilate(green, np.ones((21, 21), np.uint8), iterations=1)

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_chan = lab[:, :, 0]
    white = cv2.inRange(l_chan, 150, 255)
    low_sat = cv2.inRange(hsv[:, :, 1], 0, 95)
    mask = cv2.bitwise_and(white, low_sat)
    mask = cv2.bitwise_and(mask, field)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
    return mask


def _feature_image(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lines = cv2.bitwise_and(gray, gray, mask=mask)
    boosted = cv2.addWeighted(gray, 0.35, lines, 1.65, 0)
    return cv2.GaussianBlur(boosted, (3, 3), 0)


def _feature_matches(
    left_frame: np.ndarray,
    right_frame: np.ndarray,
    cfg: Optional[MasterCanvasConfig] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float, int]:
    config = cfg or MasterCanvasConfig()
    max_h = 720
    scale = 1.0
    if max(left_frame.shape[0], right_frame.shape[0]) > max_h:
        scale = max_h / float(max(left_frame.shape[0], right_frame.shape[0]))
        left_small = cv2.resize(left_frame, (int(round(left_frame.shape[1] * scale)), int(round(left_frame.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        right_small = cv2.resize(right_frame, (int(round(right_frame.shape[1] * scale)), int(round(right_frame.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    else:
        left_small = left_frame
        right_small = right_frame

    left_mask = _field_line_mask(left_small)
    right_mask = _field_line_mask(right_small)
    left_feat = _feature_image(left_small, left_mask)
    right_feat = _feature_image(right_small, right_mask)

    orb = cv2.ORB_create(nfeatures=max(1500, config.orb_features), edgeThreshold=12, patchSize=31)
    kp_left, desc_left = orb.detectAndCompute(left_feat, left_mask)
    kp_right, desc_right = orb.detectAndCompute(right_feat, right_mask)
    if desc_left is None or desc_right is None or len(kp_left) < 24 or len(kp_right) < 24:
        return None, None, 0.0, 0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw = matcher.knnMatch(desc_right, desc_left, k=2)
    good = []
    for pair in raw:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.74 * n.distance:
            good.append(m)
    if len(good) < 18:
        return None, None, 0.0, len(good)

    src = np.float32([kp_right[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_left[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    if scale != 1.0:
        src = src / np.float32(scale)
        dst = dst / np.float32(scale)
    return src, dst, 1.0, len(good)


def estimate_field_homography(
    left_frame: np.ndarray,
    right_frame: np.ndarray,
    cfg: Optional[MasterCanvasConfig] = None,
) -> Tuple[Optional[np.ndarray], float, int]:
    src, dst, _scale, match_count = _feature_matches(left_frame, right_frame, cfg)
    if src is None or dst is None:
        return None, 0.0, match_count
    h_small, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if h_small is None or inlier_mask is None:
        return None, 0.0, match_count
    inliers = int(inlier_mask.sum())
    confidence = inliers / float(max(1, match_count))
    if inliers < 14 or confidence < 0.35:
        return None, confidence, inliers

    if not np.all(np.isfinite(h_small)):
        return None, 0.0, inliers
    return h_small.astype(np.float64), confidence, inliers


def estimate_field_affine(
    left_frame: np.ndarray,
    right_frame: np.ndarray,
    cfg: Optional[MasterCanvasConfig] = None,
) -> Tuple[Optional[np.ndarray], float, int]:
    src, dst, _scale, match_count = _feature_matches(left_frame, right_frame, cfg)
    if src is None or dst is None:
        return None, 0.0, match_count
    affine, inlier_mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    if affine is None or inlier_mask is None:
        return None, 0.0, match_count
    inliers = int(inlier_mask.sum())
    confidence = inliers / float(max(1, match_count))
    if inliers < 14 or confidence < 0.35:
        return None, confidence, inliers
    h = np.eye(3, dtype=np.float64)
    h[:2, :] = affine.astype(np.float64)
    if not np.all(np.isfinite(h)):
        return None, 0.0, inliers
    return h, confidence, inliers


def assemble_field_calibrated_canvas(
    left_frame: np.ndarray,
    right_frame: np.ndarray,
    cfg: Optional[MasterCanvasConfig] = None,
) -> Optional[Tuple[np.ndarray, MasterCanvasLayout, float, int]]:
    config = cfg or MasterCanvasConfig()
    h_right_to_left, confidence, inliers = estimate_field_homography(left_frame, right_frame, config)
    if h_right_to_left is None:
        h_right_to_left, confidence, inliers = estimate_field_affine(left_frame, right_frame, config)
        if h_right_to_left is None:
            return None

    left_h, left_w = left_frame.shape[:2]
    right_h, right_w = right_frame.shape[:2]
    left_corners = np.float32([[0, 0], [left_w, 0], [left_w, left_h], [0, left_h]]).reshape(-1, 1, 2)
    right_corners = np.float32([[0, 0], [right_w, 0], [right_w, right_h], [0, right_h]]).reshape(-1, 1, 2)
    warped_right_corners = cv2.perspectiveTransform(right_corners, h_right_to_left)
    all_corners = np.vstack([left_corners, warped_right_corners]).reshape(-1, 2)
    min_x, min_y = np.floor(all_corners.min(axis=0)).astype(int)
    max_x, max_y = np.ceil(all_corners.max(axis=0)).astype(int)
    max_dim = 5200
    width = int(max(1, max_x - min_x))
    height = int(max(1, max_y - min_y))
    if width > max_dim or height > max_dim or width * height > 12_000_000:
        h_right_to_left, confidence, inliers = estimate_field_affine(left_frame, right_frame, config)
        if h_right_to_left is None:
            return None
        warped_right_corners = cv2.perspectiveTransform(right_corners, h_right_to_left)
        all_corners = np.vstack([left_corners, warped_right_corners]).reshape(-1, 2)
        min_x, min_y = np.floor(all_corners.min(axis=0)).astype(int)
        max_x, max_y = np.ceil(all_corners.max(axis=0)).astype(int)
        width = int(max(1, max_x - min_x))
        height = int(max(1, max_y - min_y))
        if width > max_dim or height > max_dim or width * height > 12_000_000:
            return None
    if height > int(round(max(left_h, right_h) * 1.03)):
        return None

    translate = np.array([[1.0, 0.0, -float(min_x)], [0.0, 1.0, -float(min_y)], [0.0, 0.0, 1.0]], dtype=np.float64)
    left_warp = cv2.warpPerspective(left_frame, translate, (width, height))
    right_h_canvas = translate @ h_right_to_left
    right_warp = cv2.warpPerspective(right_frame, right_h_canvas, (width, height))
    left_mask = cv2.warpPerspective(np.full((left_h, left_w), 255, dtype=np.uint8), translate, (width, height))
    right_mask = cv2.warpPerspective(np.full((right_h, right_w), 255, dtype=np.uint8), right_h_canvas, (width, height))
    union = (left_mask > 0) | (right_mask > 0)
    coverage = float(union.sum()) / float(max(1, width * height))
    if coverage < 0.82:
        return None

    overlap = (left_mask > 0) & (right_mask > 0)
    canvas = left_warp.copy()
    right_only = (right_mask > 0) & ~overlap
    canvas[right_only] = right_warp[right_only]
    if overlap.any():
        left_float = left_warp.astype(np.float32)
        right_float = right_warp.astype(np.float32)
        dist_l = cv2.distanceTransform(left_mask, cv2.DIST_L2, 3).astype(np.float32)
        dist_r = cv2.distanceTransform(right_mask, cv2.DIST_L2, 3).astype(np.float32)
        alpha = dist_r / np.maximum(dist_l + dist_r, 1.0)
        alpha = np.clip(alpha, 0.0, 1.0)
        blended = (left_float * (1.0 - alpha[..., None])) + (right_float * alpha[..., None])
        canvas[overlap] = np.clip(blended[overlap], 0, 255).astype(np.uint8)

    transformed_left = cv2.perspectiveTransform(np.float32([[[0.0, 0.0]]]), translate)[0, 0]
    seam_x = int(round(width * 0.5))
    layout = MasterCanvasLayout(
        width=width,
        height=height,
        overlap_px=int(overlap.sum() > 0),
        left_offset_x=int(round(transformed_left[0])),
        left_crop_y=int(round(transformed_left[1])),
        right_offset_x=0,
        right_crop_y=0,
        left_width=left_w,
        right_width=right_w,
        seam_x=seam_x,
        blend_start_x=seam_x,
        blend_end_x=seam_x,
        left_homography=tuple(float(v) for v in translate.reshape(-1)),
        right_homography=tuple(float(v) for v in right_h_canvas.reshape(-1)),
    )
    return canvas, layout, confidence, inliers


def assemble_manual_field_calibrated_canvas(
    left_frame: np.ndarray,
    right_frame: np.ndarray,
    cfg: MasterCanvasConfig,
) -> Optional[Tuple[np.ndarray, MasterCanvasLayout, float, int]]:
    left_points = cfg.manual_left_points or ()
    right_points = cfg.manual_right_points or ()
    count = min(len(left_points), len(right_points))
    if count < 4:
        return None
    left_h, left_w = left_frame.shape[:2]
    right_h, right_w = right_frame.shape[:2]
    dst = np.float32([
        [float(x) * float(left_w), float(y) * float(left_h)]
        for x, y in left_points[:count]
    ]).reshape(-1, 1, 2)
    src = np.float32([
        [float(x) * float(right_w), float(y) * float(right_h)]
        for x, y in right_points[:count]
    ]).reshape(-1, 1, 2)
    if count == 4:
        h_right_to_left = cv2.getPerspectiveTransform(src.reshape(4, 2), dst.reshape(4, 2))
        inliers = 4
        confidence = 1.0
    else:
        h_right_to_left, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if h_right_to_left is None or inlier_mask is None:
            return None
        inliers = int(inlier_mask.sum())
        confidence = inliers / float(max(1, count))
        if inliers < 4:
            return None
    return _assemble_with_homography(left_frame, right_frame, h_right_to_left, confidence, inliers, reject_unsafe=False)


def _assemble_with_homography(
    left_frame: np.ndarray,
    right_frame: np.ndarray,
    h_right_to_left: np.ndarray,
    confidence: float,
    inliers: int,
    reject_unsafe: bool,
) -> Optional[Tuple[np.ndarray, MasterCanvasLayout, float, int]]:
    left_h, left_w = left_frame.shape[:2]
    right_h, right_w = right_frame.shape[:2]
    left_corners = np.float32([[0, 0], [left_w, 0], [left_w, left_h], [0, left_h]]).reshape(-1, 1, 2)
    right_corners = np.float32([[0, 0], [right_w, 0], [right_w, right_h], [0, right_h]]).reshape(-1, 1, 2)
    warped_right_corners = cv2.perspectiveTransform(right_corners, h_right_to_left)
    all_corners = np.vstack([left_corners, warped_right_corners]).reshape(-1, 2)
    min_x, min_y = np.floor(all_corners.min(axis=0)).astype(int)
    max_x, max_y = np.ceil(all_corners.max(axis=0)).astype(int)
    max_dim = 5200
    width = int(max(1, max_x - min_x))
    height = int(max(1, max_y - min_y))
    if width > max_dim or height > max_dim or width * height > 12_000_000:
        return None
    if reject_unsafe and height > int(round(max(left_h, right_h) * 1.03)):
        return None

    translate = np.array([[1.0, 0.0, -float(min_x)], [0.0, 1.0, -float(min_y)], [0.0, 0.0, 1.0]], dtype=np.float64)
    left_warp = cv2.warpPerspective(left_frame, translate, (width, height))
    right_h_canvas = translate @ h_right_to_left
    right_warp = cv2.warpPerspective(right_frame, right_h_canvas, (width, height))
    left_mask = cv2.warpPerspective(np.full((left_h, left_w), 255, dtype=np.uint8), translate, (width, height))
    right_mask = cv2.warpPerspective(np.full((right_h, right_w), 255, dtype=np.uint8), right_h_canvas, (width, height))
    union = (left_mask > 0) | (right_mask > 0)
    coverage = float(union.sum()) / float(max(1, width * height))
    if reject_unsafe and coverage < 0.82:
        return None

    overlap = (left_mask > 0) & (right_mask > 0)
    canvas = left_warp.copy()
    right_only = (right_mask > 0) & ~overlap
    canvas[right_only] = right_warp[right_only]
    if overlap.any():
        left_float = left_warp.astype(np.float32)
        right_float = right_warp.astype(np.float32)
        dist_l = cv2.distanceTransform(left_mask, cv2.DIST_L2, 3).astype(np.float32)
        dist_r = cv2.distanceTransform(right_mask, cv2.DIST_L2, 3).astype(np.float32)
        alpha = dist_r / np.maximum(dist_l + dist_r, 1.0)
        alpha = np.clip(alpha, 0.0, 1.0)
        blended = (left_float * (1.0 - alpha[..., None])) + (right_float * alpha[..., None])
        canvas[overlap] = np.clip(blended[overlap], 0, 255).astype(np.uint8)

    transformed_left = cv2.perspectiveTransform(np.float32([[[0.0, 0.0]]]), translate)[0, 0]
    seam_x = int(round(width * 0.5))
    layout = MasterCanvasLayout(
        width=width,
        height=height,
        overlap_px=int(overlap.sum() > 0),
        left_offset_x=int(round(transformed_left[0])),
        left_crop_y=int(round(transformed_left[1])),
        right_offset_x=0,
        right_crop_y=0,
        left_width=left_w,
        right_width=right_w,
        seam_x=seam_x,
        blend_start_x=seam_x,
        blend_end_x=seam_x,
        left_homography=tuple(float(v) for v in translate.reshape(-1)),
        right_homography=tuple(float(v) for v in right_h_canvas.reshape(-1)),
    )
    return canvas, layout, confidence, inliers


def assemble_field_canvas_with_layout(
    left_frame: np.ndarray,
    right_frame: np.ndarray,
    layout: MasterCanvasLayout,
) -> Optional[np.ndarray]:
    if not layout.left_homography or not layout.right_homography:
        return None
    width = int(layout.width)
    height = int(layout.height)
    if width <= 0 or height <= 0:
        return None
    left_h = np.asarray(layout.left_homography, dtype=np.float64).reshape(3, 3)
    right_h = np.asarray(layout.right_homography, dtype=np.float64).reshape(3, 3)
    left_warp = cv2.warpPerspective(left_frame, left_h, (width, height))
    right_warp = cv2.warpPerspective(right_frame, right_h, (width, height))
    left_mask = cv2.warpPerspective(np.full(left_frame.shape[:2], 255, dtype=np.uint8), left_h, (width, height))
    right_mask = cv2.warpPerspective(np.full(right_frame.shape[:2], 255, dtype=np.uint8), right_h, (width, height))
    overlap = (left_mask > 0) & (right_mask > 0)
    canvas = left_warp.copy()
    right_only = (right_mask > 0) & ~overlap
    canvas[right_only] = right_warp[right_only]
    if overlap.any():
        dist_l = cv2.distanceTransform(left_mask, cv2.DIST_L2, 3).astype(np.float32)
        dist_r = cv2.distanceTransform(right_mask, cv2.DIST_L2, 3).astype(np.float32)
        alpha = dist_r / np.maximum(dist_l + dist_r, 1.0)
        blended = (left_warp.astype(np.float32) * (1.0 - alpha[..., None])) + (right_warp.astype(np.float32) * alpha[..., None])
        canvas[overlap] = np.clip(blended[overlap], 0, 255).astype(np.uint8)
    return canvas
