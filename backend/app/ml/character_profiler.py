"""
character_profiler.py — Safe character-level profiling and component geometry.

Rules:
1. Only profiles character samples when reliable, isolated glyphs can be extracted.
2. Never assumes characters A-Z or 0-9 are present in every document.
3. Explicitly marks unavailable characters as unavailable / null.
4. Deep visual encoders (e.g. ResNet) are treated strictly as optional experiments
   and not claimed as authoritative author identification.
"""
from __future__ import annotations

from typing import Any
import cv2
import numpy as np


def extract_character_candidates(
    binary_image: np.ndarray,
    min_area: int = 25,
    max_area: int = 1500,
    min_height: int = 10,
    max_height: int = 80,
) -> list[dict[str, Any]]:
    """
    Extracts connected components that resemble isolated character glyphs.
    Filters out full words, multiline blobs, and tiny ink noise.
    """
    _, _, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    valid_components = []

    for idx in range(1, len(stats)):
        x, y, w, h, area = stats[idx]
        if min_area <= area <= max_area and min_height <= h <= max_height and w < 120:
            aspect = float(w) / max(1.0, float(h))
            valid_components.append({
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
                "area": int(area),
                "aspect_ratio": round(aspect, 3),
                "centroid": (float(centroids[idx][0]), float(centroids[idx][1])),
            })

    return valid_components


def profile_character_geometry(
    binary_image: np.ndarray,
    min_samples: int = 8,
) -> dict[str, Any]:
    """
    Profiles character connected-component geometry.
    If fewer than `min_samples` can be isolated, reports status as 'unavailable'
    rather than fabricating character distributions.
    """
    candidates = extract_character_candidates(binary_image)

    if len(candidates) < min_samples:
        return {
            "status": "unavailable",
            "sample_count": len(candidates),
            "message": (
                f"Insufficient isolated character glyphs detected ({len(candidates)} found, "
                f"minimum {min_samples} required for reliable character-level profiling)."
            ),
            "glyph_aspect_mean": None,
            "glyph_height_mean": None,
            "glyph_area_mean": None,
            "character_bank_coverage": "0.0%",
            "available_character_classes": [],
        }

    aspects = [c["aspect_ratio"] for c in candidates]
    heights = [c["height"] for c in candidates]
    areas = [c["area"] for c in candidates]

    return {
        "status": "available",
        "sample_count": len(candidates),
        "glyph_aspect_mean": round(float(np.mean(aspects)), 3),
        "glyph_height_mean": round(float(np.mean(heights)), 2),
        "glyph_area_mean": round(float(np.mean(areas)), 2),
        "character_bank_coverage": "partial_unsupervised",
        "available_character_classes": [],
        "note": "Unsupervised character component statistics. OCR line-based transcription provides lexical verification.",
    }
