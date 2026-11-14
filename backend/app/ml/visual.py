"""
visual.py — Robust handwriting feature extraction for real-world photos.

Handles:
1. Paper boundary detection: Automatically crops out patterned bedsheets, desks, and backgrounds.
2. Ruled and margin line removal: Isolates handwriting strokes from notebook lines.
3. Illumination correction: Morphological background division removes phone shadows.
4. Scale normalization: Normalizes handwriting to canonical letter scale (~32px).
5. Robust physical handwriting features: Stroke width, pressure variation, slant angle,
   slant consistency, letter aspect ratio, and stroke-to-height ratio.
"""
from __future__ import annotations

import cv2
import numpy as np


FEATURE_NAMES = [
    "vis_ink_density",
    "vis_stroke_width_mean",
    "vis_stroke_width_std",
    "vis_slant_angle_mean",
    "vis_slant_angle_std",
    "vis_cc_aspect_mean",
    "vis_cc_height_rel_std",
    "vis_stroke_to_height_ratio",
]

FEATURE_LABELS: dict[str, str] = {
    "vis_ink_density":            "Ink Packing Density",
    "vis_stroke_width_mean":      "Stroke Width (normalized)",
    "vis_stroke_width_std":       "Stroke Width Variation",
    "vis_slant_angle_mean":       "Slant Angle (avg)",
    "vis_slant_angle_std":        "Slant Consistency",
    "vis_cc_aspect_mean":         "Letter Width/Height Ratio",
    "vis_cc_height_rel_std":      "Letter Height Uniformity",
    "vis_stroke_to_height_ratio": "Stroke-to-Letter Ratio",
}


def decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode uploaded image bytes to a BGR NumPy array."""
    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Ensure the file is a valid PNG/JPEG/WEBP.")
    return img


def extract_paper_region(image: np.ndarray) -> np.ndarray:
    """
    Detects the sheet of paper in the image and crops away patterned bedsheets,
    fabrics, and desk backgrounds.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    blurred = cv2.GaussianBlur(gray, (25, 25), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if cnts:
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area > (image.shape[0] * image.shape[1] * 0.22):
            x, y, w, h = cv2.boundingRect(c)
            # Inset slightly (2.5%) to avoid paper border / bedsheet overlap
            px = int(w * 0.025)
            py = int(h * 0.025)
            x1 = max(0, x + px)
            y1 = max(0, y + py)
            x2 = min(image.shape[1], x + w - px)
            y2 = min(image.shape[0], y + h - py)
            if (x2 - x1) > 64 and (y2 - y1) > 64:
                return image[y1:y2, x1:x2]
    return image


def remove_ruled_and_margin_lines(binary: np.ndarray) -> np.ndarray:
    """
    Removes full-width horizontal notebook lines and full-height vertical margin
    lines so they don't distort letter connected components.
    """
    # Only remove horizontal lines that span across at least 45% of the page
    min_w = max(70, int(binary.shape[1] * 0.45))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_w, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

    # Only remove vertical lines that span across at least 45% of the page
    min_h = max(70, int(binary.shape[0] * 0.45))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_h))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    all_lines = cv2.bitwise_or(h_lines, v_lines)
    clean_ink = cv2.bitwise_and(binary, cv2.bitwise_not(all_lines))
    return clean_ink


def preprocess_and_normalize(
    image: np.ndarray, target_cc_height: float = 32.0
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    1. Crops paper from background.
    2. Corrects uneven illumination.
    3. Removes notebook lines.
    4. Rescales handwriting to canonical letter height (~32px).
    """
    paper = extract_paper_region(image)
    gray = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY) if paper.ndim == 3 else paper.copy()

    # Illumination correction via morphological background division
    k_size = max(21, (min(gray.shape) // 20) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    norm = cv2.divide(gray, bg, scale=255)

    # Adaptive binarization
    binary = cv2.adaptiveThreshold(
        norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15
    )

    # Remove ruled lines
    clean_binary = remove_ruled_and_margin_lines(binary)

    # Detect median letter height from clean handwriting components
    _, _, stats, _ = cv2.connectedComponentsWithStats(clean_binary, connectivity=8)
    valid = stats[1:]
    # Exclude tiny noise and residual horizontal lines
    valid = valid[(valid[:, cv2.CC_STAT_AREA] >= 16) & (valid[:, cv2.CC_STAT_HEIGHT] >= 8) & (valid[:, cv2.CC_STAT_WIDTH] < 200)]

    scale = 1.0
    if len(valid) >= 5:
        median_h = float(np.median(valid[:, cv2.CC_STAT_HEIGHT]))
        scale = float(np.clip(target_cc_height / max(6.0, median_h), 0.15, 4.0))

    if abs(scale - 1.0) > 0.08:
        new_w = max(32, int(round(gray.shape[1] * scale)))
        new_h = max(32, int(round(gray.shape[0] * scale)))
        gray_s = cv2.resize(
            gray, (new_w, new_h),
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        )
        k_s = max(15, (min(gray_s.shape) // 20) | 1)
        bg_s = cv2.morphologyEx(
            gray_s, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (k_s, k_s))
        )
        norm_s = cv2.divide(gray_s, bg_s, scale=255)
        binary_s = cv2.adaptiveThreshold(
            norm_s, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15
        )
        clean_binary_s = remove_ruled_and_margin_lines(binary_s)
        return gray_s, clean_binary_s, scale

    return gray, clean_binary, scale


def extract_visual_features(image: np.ndarray) -> dict[str, float]:
    """
    Extract 8 robust physical handwriting features from an image.
    """
    gray, binary, scale = preprocess_and_normalize(image)
    ink_mask = binary > 0

    # Ink packing density inside ink bounding box
    y_idx, x_idx = np.where(ink_mask)
    if len(y_idx) > 10:
        bbox_area = max(100, (y_idx.max() - y_idx.min() + 1) * (x_idx.max() - x_idx.min() + 1))
        ink_density = float(len(y_idx) / bbox_area)
    else:
        ink_density = float(np.mean(ink_mask))

    # Stroke width via distance transform at canonical scale
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    fg_dist = dist[ink_mask]
    stroke_width_mean = float(np.mean(fg_dist) * 2.0) if fg_dist.size else 0.0
    stroke_width_std = float(np.std(fg_dist) * 2.0) if fg_dist.size else 0.0

    # Slant angle from stroke gradients
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    strong = (mag > np.percentile(mag, 85)) & ink_mask if np.any(ink_mask) else np.zeros_like(ink_mask)
    if np.any(strong):
        angles = np.degrees(np.arctan2(gy[strong], gx[strong]))
        slant_vals = ((angles + 90.0) % 180.0) - 90.0
        slant_mean = float(np.mean(slant_vals))
        slant_std = float(np.std(slant_vals))
    else:
        slant_mean = slant_std = 0.0

    # Letter proportions & scale-invariant ratios
    _, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    valid = stats[1:]
    valid = valid[(valid[:, cv2.CC_STAT_AREA] >= 16) & (valid[:, cv2.CC_STAT_HEIGHT] >= 8) & (valid[:, cv2.CC_STAT_WIDTH] < 200)]
    if len(valid) >= 5:
        heights = valid[:, cv2.CC_STAT_HEIGHT]
        widths = valid[:, cv2.CC_STAT_WIDTH]
        aspects = widths / np.maximum(heights, 1)
        cc_h_mean = float(np.mean(heights))
        cc_h_std = float(np.std(heights))
        cc_aspect = float(np.mean(aspects))
        rel_std = float(cc_h_std / max(1.0, cc_h_mean))
        stroke_ratio = float(stroke_width_mean / max(1.0, cc_h_mean))
    else:
        cc_aspect = 0.75
        rel_std = 0.35
        stroke_ratio = 0.08

    return {
        "vis_ink_density":            ink_density,
        "vis_stroke_width_mean":      stroke_width_mean,
        "vis_stroke_width_std":       stroke_width_std,
        "vis_slant_angle_mean":       slant_mean,
        "vis_slant_angle_std":        slant_std,
        "vis_cc_aspect_mean":         cc_aspect,
        "vis_cc_height_rel_std":      rel_std,
        "vis_stroke_to_height_ratio": stroke_ratio,
    }


def compute_per_feature_similarity(
    ref_feats: dict[str, float],
    cmp_feats: dict[str, float],
) -> dict[str, float]:
    """
    Compute a normalized [0, 1] similarity score per visual feature.
    Handles slant angle smoothly around zero.
    """
    sims: dict[str, float] = {}
    for key in FEATURE_NAMES:
        a = float(ref_feats.get(key, 0.0))
        b = float(cmp_feats.get(key, 0.0))

        if key == "vis_slant_angle_mean":
            # Slant angle difference normalized over 45 degrees
            diff = abs(a - b)
            sim = float(np.clip(1.0 - diff / 45.0, 0.0, 1.0))
        else:
            denom = abs(a) + abs(b) + 1e-6
            sim = float(np.clip(1.0 - abs(a - b) / denom, 0.0, 1.0))

        sims[key] = round(sim, 4)
    return sims


def build_feature_comparison_rows(
    ref_feats: dict[str, float],
    cmp_feats: dict[str, float],
) -> list[dict]:
    """Return a list of FeatureDifference dicts for the API response."""
    rows = []
    for key in FEATURE_NAMES:
        a = ref_feats.get(key, 0.0)
        b = cmp_feats.get(key, 0.0)
        rows.append({
            "feature":             FEATURE_LABELS.get(key, key),
            "reference":           round(float(a), 4),
            "comparison":          round(float(b), 4),
            "absolute_difference": round(float(abs(a - b)), 4),
        })
    rows.sort(key=lambda r: r["absolute_difference"], reverse=True)
    return rows
