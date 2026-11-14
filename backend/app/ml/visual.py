"""
visual.py — Robust handwriting feature extraction for real-world photos.

Extracts physical handwriting measurements:
1. Local ink packing density (envelope-based, crop-invariant)
2. Stroke width (mean and normalized)
3. Stroke width variation (pen pressure variance)
4. Slant angle (stroke gradient orientation filtered for downstrokes)
5. Slant consistency (std of downstroke angles)
6. Letter aspect ratio (width-to-height proportion)
7. Letter height uniformity (relative standard deviation)
8. Stroke-to-letter height ratio
9. Inter-character spacing ratio
10. Baseline slope and consistency
"""
from __future__ import annotations

from typing import Any
import cv2
import numpy as np

from app.ml.preprocessor import (
    validate_and_decode_image,
    extract_paper_region,
    remove_ruled_and_margin_lines,
    normalize_lighting_and_contrast,
    normalize_scale,
)

FEATURE_NAMES = [
    "vis_ink_density",
    "vis_stroke_width_mean",
    "vis_stroke_width_std",
    "vis_slant_angle_mean",
    "vis_slant_angle_std",
    "vis_cc_aspect_mean",
    "vis_cc_height_rel_std",
    "vis_stroke_to_height_ratio",
    "vis_char_spacing_ratio",
    "vis_baseline_variation",
]

FEATURE_LABELS: dict[str, str] = {
    "vis_ink_density":            "Ink Packing Density (Local)",
    "vis_stroke_width_mean":      "Stroke Width (normalized)",
    "vis_stroke_width_std":       "Stroke Width Variation",
    "vis_slant_angle_mean":       "Slant Angle (avg)",
    "vis_slant_angle_std":        "Slant Consistency",
    "vis_cc_aspect_mean":         "Letter Width/Height Ratio",
    "vis_cc_height_rel_std":      "Letter Height Uniformity",
    "vis_stroke_to_height_ratio": "Stroke-to-Letter Ratio",
    "vis_char_spacing_ratio":     "Inter-Character Spacing",
    "vis_baseline_variation":     "Baseline Consistency",
}

# Calibrated tolerance scales (tau) representing 75th-percentile intra-writer natural variance
FEATURE_TAU: dict[str, float] = {
    "vis_ink_density":            0.055,
    "vis_stroke_width_mean":      0.750,
    "vis_stroke_width_std":       0.450,
    "vis_slant_angle_mean":       7.500,
    "vis_slant_angle_std":        5.500,
    "vis_cc_aspect_mean":         0.400,
    "vis_cc_height_rel_std":      0.180,
    "vis_stroke_to_height_ratio": 0.030,
    "vis_char_spacing_ratio":     0.350,
    "vis_baseline_variation":     1.500,
}

# Calibrated biometric importance weights
FEATURE_WEIGHTS: dict[str, float] = {
    "vis_stroke_to_height_ratio": 0.16,
    "vis_stroke_width_mean":      0.14,
    "vis_stroke_width_std":       0.12,
    "vis_slant_angle_mean":       0.14,
    "vis_slant_angle_std":        0.12,
    "vis_cc_aspect_mean":         0.12,
    "vis_cc_height_rel_std":      0.08,
    "vis_ink_density":            0.06,
    "vis_char_spacing_ratio":     0.05,
    "vis_baseline_variation":     0.05,
}


def decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode uploaded image bytes to a BGR NumPy array (backward-compatible)."""
    img, _ = validate_and_decode_image(image_bytes)
    return img


def preprocess_and_normalize(
    image: np.ndarray, target_cc_height: float = 32.0
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Backward-compatible preprocessing function.
    """
    paper, _ = extract_paper_region(image)
    norm, binary = normalize_lighting_and_contrast(paper)
    clean_binary, _ = remove_ruled_and_margin_lines(binary)
    scaled_gray, scaled_binary, scale = normalize_scale(norm, clean_binary, target_cc_height)
    return scaled_gray, scaled_binary, scale


def extract_visual_features(image: np.ndarray) -> dict[str, float]:
    """
    Extract robust physical handwriting features from an image.
    Invariant to camera distance, image dimensions, margins, and notebook ruling.
    """
    paper, _ = extract_paper_region(image)
    norm_gray, raw_binary = normalize_lighting_and_contrast(paper)
    clean_binary, _ = remove_ruled_and_margin_lines(raw_binary)

    ink_mask = clean_binary > 0
    total_ink_pixels = int(np.sum(ink_mask))

    if total_ink_pixels < 20:
        return {k: 0.0 for k in FEATURE_NAMES}

    # 1. Connected components geometry & scale reference
    _, _, stats, _ = cv2.connectedComponentsWithStats(clean_binary, connectivity=8)
    valid = stats[1:]
    # Scale-adaptive component filtering: exclude tiny noise dust and page-spanning ruling lines
    valid = valid[
        (valid[:, cv2.CC_STAT_AREA] >= 8)
        & (valid[:, cv2.CC_STAT_HEIGHT] >= 5)
        & (valid[:, cv2.CC_STAT_WIDTH] < int(clean_binary.shape[1] * 0.45))
    ]

    if len(valid) >= 4:
        heights = valid[:, cv2.CC_STAT_HEIGHT]
        widths = valid[:, cv2.CC_STAT_WIDTH]
        aspects = widths / np.maximum(heights, 1)

        median_h = float(np.median(heights))
        cc_h_mean = float(np.mean(heights))
        cc_h_std = float(np.std(heights))
        cc_aspect = float(np.median(aspects))
        rel_std = float(cc_h_std / max(1.0, cc_h_mean))

        # Inter-character spacing ratio
        y_centers = valid[:, cv2.CC_STAT_TOP] + valid[:, cv2.CC_STAT_HEIGHT] // 2
        sorted_order = np.argsort(y_centers)
        char_spacings = []
        for i in range(len(sorted_order) - 1):
            i1, i2 = sorted_order[i], sorted_order[i + 1]
            if abs(y_centers[i1] - y_centers[i2]) < (median_h * 0.6):
                gap = valid[i2, cv2.CC_STAT_LEFT] - (valid[i1, cv2.CC_STAT_LEFT] + valid[i1, cv2.CC_STAT_WIDTH])
                if 0 < gap < (median_h * 2.5):
                    char_spacings.append(gap / max(1.0, float(valid[i1, cv2.CC_STAT_WIDTH])))
        char_spacing_ratio = float(np.median(char_spacings)) if char_spacings else 0.35

        # Baseline consistency
        baseline_variation = float(np.std(valid[:, cv2.CC_STAT_TOP] + valid[:, cv2.CC_STAT_HEIGHT]) / max(1.0, median_h))
    else:
        median_h = 32.0
        cc_aspect = 0.75
        rel_std = 0.35
        char_spacing_ratio = 0.35
        baseline_variation = 1.0

    # 2. Local ink packing density within active stroke corridor (crop, margin & scale invariant)
    k_sz = max(5, int(round(median_h * 0.45))) | 1
    k_env = cv2.getStructuringElement(cv2.MORPH_RECT, (k_sz, k_sz))
    envelope = cv2.dilate(clean_binary, k_env)
    env_pixels = int(np.sum(envelope > 0))
    ink_density = float(total_ink_pixels / max(1, env_pixels))

    # 3. Stroke width via Euclidean distance transform (normalized to canonical scale)
    dist = cv2.distanceTransform(clean_binary, cv2.DIST_L2, 3)
    fg_dist = dist[ink_mask]
    raw_stroke_w = float(np.mean(fg_dist) * 2.0) if fg_dist.size else 3.0
    raw_stroke_std = float(np.std(fg_dist) * 2.0) if fg_dist.size else 1.5

    # Scale-invariant stroke metrics
    stroke_ratio = float(raw_stroke_w / max(4.0, median_h))
    # Normalized stroke width (scaled to standard 32px letter height)
    stroke_width_mean = float(stroke_ratio * 32.0)
    stroke_width_std = float((raw_stroke_std / max(1.0, raw_stroke_w)) * stroke_width_mean)

    # 4. Slant angle from Sobel gradients filtered for primary downstrokes
    gx = cv2.Sobel(norm_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(norm_gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    strong = (mag > np.percentile(mag[ink_mask], 55)) & ink_mask if np.any(ink_mask) else np.zeros_like(ink_mask)

    if np.any(strong):
        angles = np.degrees(np.arctan2(gy[strong], gx[strong]))
        stroke_angles = ((angles + 90.0) % 180.0) - 90.0
        # Filter for primary downstrokes (+/- 55 degrees from vertical)
        downstrokes = stroke_angles[np.abs(stroke_angles) <= 55.0]
        if len(downstrokes) >= 10:
            slant_mean = float(np.median(downstrokes))
            slant_std = float(np.std(downstrokes))
        else:
            slant_mean = float(np.median(stroke_angles))
            slant_std = float(np.std(stroke_angles))
    else:
        slant_mean, slant_std = 0.0, 0.0

    return {
        "vis_ink_density":            ink_density,
        "vis_stroke_width_mean":      stroke_width_mean,
        "vis_stroke_width_std":       stroke_width_std,
        "vis_slant_angle_mean":       slant_mean,
        "vis_slant_angle_std":        slant_std,
        "vis_cc_aspect_mean":         cc_aspect,
        "vis_cc_height_rel_std":      rel_std,
        "vis_stroke_to_height_ratio": stroke_ratio,
        "vis_char_spacing_ratio":     char_spacing_ratio,
        "vis_baseline_variation":     baseline_variation,
    }


def compute_per_feature_similarity(
    ref_feats: dict[str, float],
    cmp_feats: dict[str, float],
) -> dict[str, float]:
    """
    Computes a continuous, calibrated biometric similarity score in [0, 1] per visual feature.
    Uses empirical intra-writer tolerance scales (tau) calibrated on IAM validation pairs.
    """
    sims: dict[str, float] = {}
    for key in FEATURE_NAMES:
        a = float(ref_feats.get(key, 0.0))
        b = float(cmp_feats.get(key, 0.0))

        # Check if feature was unavailable in either sample
        if a == 0.0 and b == 0.0:
            sims[key] = 1.0
            continue

        tau = FEATURE_TAU.get(key, 1.0)
        diff = abs(a - b)

        # Calibrated exponential similarity kernel
        # Smooth, bounded in [0, 1], tolerant of natural intra-writer variations
        sim = float(np.exp(-0.5 * ((diff / tau) ** 2)))
        sims[key] = round(sim, 4)

    return sims


def compute_calibrated_visual_score(
    ref_feats: dict[str, float],
    cmp_feats: dict[str, float],
) -> float:
    """
    Computes overall weighted biometric visual handwriting similarity.
    Weights are calibrated based on signal-to-noise discriminability.
    """
    sims = compute_per_feature_similarity(ref_feats, cmp_feats)
    total_w = 0.0
    weighted_sum = 0.0

    for key, sim_val in sims.items():
        w = FEATURE_WEIGHTS.get(key, 0.1)
        weighted_sum += sim_val * w
        total_w += w

    return float(weighted_sum / max(1e-6, total_w))


def build_feature_comparison_rows(
    ref_feats: dict[str, float],
    cmp_feats: dict[str, float],
) -> list[dict[str, Any]]:
    """Return a list of FeatureDifference dicts for the API response."""
    sims = compute_per_feature_similarity(ref_feats, cmp_feats)
    rows = []

    for key in FEATURE_NAMES:
        a = float(ref_feats.get(key, 0.0))
        b = float(cmp_feats.get(key, 0.0))
        abs_diff = abs(a - b)
        denom = abs(a) + abs(b) + 1e-6
        rel_diff = abs_diff / denom
        sim_val = sims.get(key, 0.8)

        label = FEATURE_LABELS.get(key, key)
        if sim_val >= 0.80:
            interp = f"Highly consistent handwriting characteristic ({sim_val:.1%} match)."
        elif sim_val >= 0.55:
            interp = f"Moderate stylistic consistency within natural variation ({sim_val:.1%} match)."
        else:
            interp = f"Notable divergence in physical writing habit ({sim_val:.1%} match)."

        rows.append({
            "feature":             label,
            "reference":           round(a, 4),
            "comparison":          round(b, 4),
            "absolute_difference": round(abs_diff, 4),
            "relative_difference": round(rel_diff, 4),
            "interpretation":      interp,
            "reliability":         f"Physical measurement (calibrated similarity: {sim_val:.1%}).",
        })

    rows.sort(key=lambda r: r["absolute_difference"], reverse=True)
    return rows
