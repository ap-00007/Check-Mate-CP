"""
verifier.py — Pairwise handwriting verification.

Computes a composite physical verification score using:
1. Scale- and illumination-normalized OpenCV handwriting features.
2. Direct geometric similarity across stroke width, pressure variance, slant angle,
   slant consistency, aspect ratio, and ink density.
3. Decision threshold calibrated for real-world paper photos and notebook scans.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from joblib import load

from app.ml.visual import (
    FEATURE_NAMES,
    FEATURE_LABELS,
    decode_image,
    extract_visual_features,
    compute_per_feature_similarity,
    build_feature_comparison_rows,
)
from app.core.config import settings

logger = logging.getLogger(__name__)

# Default threshold: 82% physical stylistic similarity for real-world photos
DEFAULT_THRESHOLD = 0.82


def verify_visual_pair(
    reference_bytes: bytes,
    comparison_bytes: bytes,
) -> dict:
    """
    Run visual Check Mate verification on two uploaded images.
    Returns a dict matching the VerificationResponse schema.
    """
    ref_img = decode_image(reference_bytes)
    cmp_img = decode_image(comparison_bytes)

    ref_feats = extract_visual_features(ref_img)
    cmp_feats = extract_visual_features(cmp_img)

    # Insufficient-evidence guard: check for near-empty ink
    MIN_INK = 0.005
    if ref_feats["vis_ink_density"] < MIN_INK or cmp_feats["vis_ink_density"] < MIN_INK:
        return {
            "status":          "insufficient_evidence",
            "message":         "One or both images appear to contain very little ink. "
                               "Please upload clear handwriting samples.",
            "nlp_score":       None,
            "visual_score":    None,
            "combined_score":  None,
            "reference_text":  None,
            "comparison_text": None,
            "feature_comparison": [],
            "visual_feature_scores": {},
            "warnings": [
                "Image too blank or low-contrast for reliable handwriting analysis."
            ],
        }

    # Compute per-feature physical similarities
    per_feature_sims = compute_per_feature_similarity(ref_feats, cmp_feats)
    feature_rows = build_feature_comparison_rows(ref_feats, cmp_feats)

    # Overall visual score: mean of physical handwriting feature similarities
    visual_score = float(np.mean(list(per_feature_sims.values())))
    threshold = DEFAULT_THRESHOLD

    if visual_score >= threshold:
        status = "compatible"
        message = (
            f"Handwriting styles appear compatible "
            f"(visual similarity {visual_score:.1%}, threshold {threshold:.1%})."
        )
    else:
        status = "not_compatible"
        message = (
            f"Handwriting styles appear different "
            f"(visual similarity {visual_score:.1%}, threshold {threshold:.1%})."
        )

    # Group similarities into dashboard-friendly module scores
    visual_module_scores = {
        "stroke_similarity": round(
            float(np.mean([
                per_feature_sims.get("vis_stroke_width_mean", 0.8),
                per_feature_sims.get("vis_stroke_width_std", 0.8),
                per_feature_sims.get("vis_stroke_to_height_ratio", 0.8),
                per_feature_sims.get("vis_ink_density", 0.8),
            ])), 4
        ),
        "slant_similarity": round(
            float(np.mean([
                per_feature_sims.get("vis_slant_angle_mean", 0.8),
                per_feature_sims.get("vis_slant_angle_std", 0.8),
            ])), 4
        ),
        "letter_proportion_similarity": round(
            float(np.mean([
                per_feature_sims.get("vis_cc_aspect_mean", 0.8),
                per_feature_sims.get("vis_cc_height_rel_std", 0.8),
            ])), 4
        ),
    }

    return {
        "status":          status,
        "message":         message,
        "nlp_score":       None,
        "visual_score":    round(visual_score, 4),
        "combined_score":  round(visual_score, 4),
        "reference_text":  None,
        "comparison_text": None,
        "feature_comparison":    feature_rows,
        "visual_feature_scores": visual_module_scores,
        "warnings": [
            "Visual handwriting analysis active (stroke width, slant consistency, and letter geometry).",
            "This analysis is designed for stylistic verification and is not forensic proof of identity.",
        ],
    }
