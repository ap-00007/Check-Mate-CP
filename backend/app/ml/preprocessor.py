"""
preprocessor.py — Robust document preprocessing and text-line segmentation for handwriting.

Pipeline:
1. Image validation: format, dimensions, corruption check.
2. Document/paper detection with adaptive framing (preserves border strokes).
3. Illumination correction via morphological background division.
4. Adaptive binarization and contrast normalization.
5. Multi-angle ruled and margin line removal (removes notebook lines without severing downstrokes).
6. Scale-invariant representation (preserves original stroke geometry without bitmap distortion).
7. Top-to-bottom text line segmentation for line-based TrOCR and spacing analysis.
8. Preservation of the original image for visual feature extraction.
"""
from __future__ import annotations

from typing import Any
import cv2
import numpy as np


class ImageValidationError(ValueError):
    """Raised when an uploaded image fails validation."""
    pass


def validate_and_decode_image(image_bytes: bytes) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Validates uploaded bytes and decodes into a BGR NumPy array.
    Checks minimal resolution, memory limits, and channel consistency.
    """
    if not image_bytes or len(image_bytes) == 0:
        raise ImageValidationError("Uploaded image file is empty.")

    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image is None:
        raise ImageValidationError(
            "Could not decode image. Supported formats: PNG, JPEG, WEBP."
        )

    h, w = image.shape[:2]
    if h < 32 or w < 32:
        raise ImageValidationError(
            f"Image dimensions ({w}x{h}) are too small for handwriting verification. "
            f"Minimum size is 32x32 pixels."
        )

    if h > 8192 or w > 8192:
        raise ImageValidationError(
            f"Image dimensions ({w}x{h}) exceed maximum allowed dimension (8192px)."
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))

    meta = {
        "original_width": w,
        "original_height": h,
        "channels": image.shape[2] if image.ndim == 3 else 1,
        "mean_intensity": round(mean_val, 2),
        "std_intensity": round(std_val, 2),
        "is_low_contrast": bool(std_val < 10.0),
    }

    return image, meta


def extract_paper_region(image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Detects paper boundary when non-paper backgrounds (desks, bedsheets, dark surfaces)
    are present. If the photograph is already an in-frame paper document, avoids destructive
    cropping so text near page margins is preserved.
    """
    h, w = image.shape[:2]
    total_area = h * w
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()

    # Check if image is already a framed sheet (high mean brightness, low dark border area)
    border_pixels = np.concatenate([
        gray[0:max(1, h//20), :].flatten(),
        gray[-max(1, h//20):, :].flatten(),
        gray[:, 0:max(1, w//20)].flatten(),
        gray[:, -max(1, w//20):].flatten(),
    ])
    border_mean = float(np.mean(border_pixels))
    border_dark_ratio = float(np.mean(border_pixels < 100))

    crop_info = {
        "paper_detected": False,
        "crop_box": [0, 0, w, h],
        "crop_area_ratio": 1.0,
    }

    # If border is already light paper and dark background is minimal, do not crop
    if border_mean > 170 and border_dark_ratio < 0.12:
        return image, crop_info

    # Otherwise, detect paper contour on contrasting background
    blurred = cv2.GaussianBlur(gray, (25, 25), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if cnts:
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area > (total_area * 0.25):
            x, y, cw, ch = cv2.boundingRect(c)
            # Only crop if it removes genuine non-paper border
            if (cw * ch) < (total_area * 0.94):
                px = max(2, int(cw * 0.015))
                py = max(2, int(ch * 0.015))
                x1 = max(0, x + px)
                y1 = max(0, y + py)
                x2 = min(w, x + cw - px)
                y2 = min(h, y + ch - py)
                if (x2 - x1) > 64 and (y2 - y1) > 64:
                    crop_info = {
                        "paper_detected": True,
                        "crop_box": [x1, y1, x2 - x1, y2 - y1],
                        "crop_area_ratio": round(float((x2 - x1) * (y2 - y1)) / total_area, 4),
                    }
                    return image[y1:y2, x1:x2], crop_info

    return image, crop_info


def remove_ruled_and_margin_lines(binary: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Robust multi-angle notebook line removal:
    1. Removes perfectly horizontal lines via morphological open.
    2. Detects tilted notebook lines (within +/- 8.5 degrees) via Hough transform.
    3. Protects vertical handwriting strokes with spatial dilation so downstrokes
       and letter bodies are NOT cut or broken.
    """
    h, w = binary.shape[:2]
    line_mask = np.zeros_like(binary)

    # 1. Morphological horizontal line detection (isolated thin lines >= 15% width)
    min_w = max(40, int(w * 0.15))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_w, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    line_mask = cv2.bitwise_or(line_mask, h_lines)

    # 2. Vertical margin lines (>= 35% height)
    min_h = max(60, int(h * 0.35))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_h))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    line_mask = cv2.bitwise_or(line_mask, v_lines)

    # 3. Probabilistic Hough transform for tilted notebook lines
    try:
        min_len = max(60, int(w * 0.18))
        hough_lines = cv2.HoughLinesP(
            binary, 1, np.pi / 180,
            threshold=70,
            minLineLength=min_len,
            maxLineGap=25,
        )
        if hough_lines is not None:
            for line_arr in hough_lines:
                x1, y1, x2, y2 = line_arr.flatten()
                angle = np.degrees(np.arctan2(abs(y2 - y1), max(1, abs(x2 - x1))))
                # Only target ruled lines within +/- 8.5 degrees
                if angle <= 8.5:
                    cv2.line(line_mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, 3)
    except Exception:
        pass

    # 4. Protect vertical handwriting downstrokes and cursive letter bodies
    vert_protect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    vertical_strokes = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vert_protect_kernel)
    vert_protect = cv2.dilate(vertical_strokes, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

    # Erase lines only where they do not intersect protected handwriting strokes
    line_to_erase = cv2.bitwise_and(line_mask, cv2.bitwise_not(vert_protect))
    clean_ink = cv2.bitwise_and(binary, cv2.bitwise_not(line_to_erase))

    removed_pixels = int(np.sum(line_to_erase > 0))
    return clean_ink, {
        "ruled_lines_detected": bool(removed_pixels > 80),
        "removed_line_pixels": removed_pixels,
    }


def normalize_lighting_and_contrast(
    paper_image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalizes uneven illumination and shadow gradients via multi-scale morphological
    background division, followed by contrast-adaptive binarization.
    """
    gray = cv2.cvtColor(paper_image, cv2.COLOR_BGR2GRAY) if paper_image.ndim == 3 else paper_image.copy()

    # Multi-scale background estimation for lighting gradients
    k_size = max(25, (min(gray.shape) // 16) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    norm = cv2.divide(gray, bg, scale=255)

    # Adaptive binarization with conservative constant to preserve delicate strokes
    binary = cv2.adaptiveThreshold(
        norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 11
    )

    # Small noise suppression (isolated 1-2 pixel salt)
    clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary_denoised = cv2.morphologyEx(binary, cv2.MORPH_OPEN, clean_kernel)

    # If denoised keeps >= 90% ink, use it
    if np.sum(binary_denoised > 0) >= 0.85 * np.sum(binary > 0):
        binary = binary_denoised

    return norm, binary


def normalize_scale(
    gray: np.ndarray,
    binary: np.ndarray,
    target_cc_height: float = 32.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Computes scale factor from median letter height without distorting original strokes.
    """
    _, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    valid = stats[1:]
    # Filter noise and residual ruled lines
    valid = valid[
        (valid[:, cv2.CC_STAT_AREA] >= 16)
        & (valid[:, cv2.CC_STAT_HEIGHT] >= 8)
        & (valid[:, cv2.CC_STAT_WIDTH] < 250)
    ]

    scale = 1.0
    if len(valid) >= 5:
        median_h = float(np.median(valid[:, cv2.CC_STAT_HEIGHT]))
        scale = float(np.clip(target_cc_height / max(6.0, median_h), 0.25, 3.0))

    return gray, binary, scale


def segment_text_lines(
    paper_image: np.ndarray,
    binary_mask: np.ndarray,
    min_line_height: int = 14,
    padding: int = 6,
) -> list[dict[str, Any]]:
    """
    Segments document into text lines using horizontal ink projection profile.
    Preserves top-to-bottom reading order.
    Returns individual line image crops and coordinate bounds.
    """
    h, w = binary_mask.shape[:2]
    row_ink = np.sum(binary_mask > 0, axis=1)
    threshold = max(3, int(w * 0.003))
    active = row_ink > threshold

    ranges: list[tuple[int, int]] = []
    start: int | None = None
    for idx, is_act in enumerate(active):
        if is_act and start is None:
            start = idx
        elif not is_act and start is not None:
            if idx - start >= min_line_height:
                ranges.append((start, idx))
            start = None
    if start is not None and len(active) - start >= min_line_height:
        ranges.append((start, len(active)))

    # Merge ranges with small vertical gaps (< 6px)
    merged_ranges: list[tuple[int, int]] = []
    for r in ranges:
        if not merged_ranges:
            merged_ranges.append(r)
        else:
            prev_start, prev_end = merged_ranges[-1]
            if r[0] - prev_end <= 6:
                merged_ranges[-1] = (prev_start, r[1])
            else:
                merged_ranges.append(r)

    crops: list[dict[str, Any]] = []
    for line_idx, (y1, y2) in enumerate(merged_ranges):
        pad_y1 = max(0, y1 - padding)
        pad_y2 = min(h, y2 + padding)
        line_crop = paper_image[pad_y1:pad_y2, :]
        if line_crop.size > 0:
            crops.append({
                "line_index": line_idx,
                "y1": pad_y1,
                "y2": pad_y2,
                "height": pad_y2 - pad_y1,
                "width": w,
                "image": line_crop,
            })

    if not crops and paper_image.size > 0:
        crops.append({
            "line_index": 0,
            "y1": 0,
            "y2": h,
            "height": h,
            "width": w,
            "image": paper_image,
        })

    return crops


def preprocess_handwriting_document(image_bytes: bytes) -> dict[str, Any]:
    """
    Complete document preprocessing pipeline returning all intermediate
    artifacts, original unmanipulated image, and segmented lines.
    """
    original_img, validation_meta = validate_and_decode_image(image_bytes)
    paper_img, paper_crop_meta = extract_paper_region(original_img)
    norm_gray, raw_binary = normalize_lighting_and_contrast(paper_img)
    clean_binary, line_removal_meta = remove_ruled_and_margin_lines(raw_binary)
    scaled_gray, scaled_binary, scale_factor = normalize_scale(norm_gray, clean_binary)
    line_crops = segment_text_lines(paper_img, clean_binary)

    # Local handwriting envelope density (crop-invariant)
    k_env = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    envelope = cv2.dilate(clean_binary, k_env)
    env_pixels = int(np.sum(envelope > 0))
    ink_pixels = int(np.sum(clean_binary > 0))

    local_ink_density = float(ink_pixels / max(1, env_pixels)) if env_pixels > 0 else 0.0

    metadata = {
        **validation_meta,
        **paper_crop_meta,
        **line_removal_meta,
        "scale_factor": round(scale_factor, 4),
        "ink_density": round(local_ink_density, 5),
        "detected_line_count": len(line_crops),
        "is_sufficient_ink": bool(ink_pixels >= 40),
    }

    return {
        "original_image": original_img,
        "paper_image": paper_img,
        "normalized_gray": scaled_gray,
        "clean_binary": scaled_binary,
        "scale_factor": scale_factor,
        "line_crops": line_crops,
        "line_count": len(line_crops),
        "metadata": metadata,
    }
