"""
test_visual_investigation.py — Validation of visual handwriting verification invariance.

Validates the 7 required test cases:
1. Two samples from the same writer with different words.
2. Two samples from the same writer with different image crops.
3. Two samples from the same writer with different writing sizes (scale).
4. Two samples from different writers.
5. Two samples with different lighting conditions (shadow gradients).
6. Very short handwriting samples (triggers insufficient evidence guard).
7. Samples with notebook lines and paper backgrounds.
"""
from pathlib import Path
import cv2
import numpy as np
import pytest

from app.ml.visual import extract_visual_features, compute_per_feature_similarity, compute_calibrated_visual_score
from app.ml.verifier import verify_handwriting_pair_core


IAM_DIR = Path(__file__).resolve().parent.parent.parent / "notebooks" / "data" / "iam_docs"


def _get_iam_pair(writer_prefix: str = "a01-000u") -> tuple[np.ndarray, np.ndarray]:
    """Loads a real ground-truth same-writer pair from IAM."""
    p_a = IAM_DIR / f"{writer_prefix}_partA.png"
    p_b = IAM_DIR / f"{writer_prefix}_partB.png"
    if p_a.exists() and p_b.exists():
        img_a = cv2.imread(str(p_a))
        img_b = cv2.imread(str(p_b))
        if img_a is not None and img_b is not None:
            return img_a, img_b

    # Fallback to synthetic if files not found
    img_a = np.full((300, 600, 3), 250, dtype=np.uint8)
    img_b = np.full((300, 600, 3), 250, dtype=np.uint8)
    for y in [60, 110, 160, 210]:
        cv2.putText(img_a, "The handwriting sample text verification", (30, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.7, (30, 30, 30), 2)
        cv2.putText(img_b, "Different words written in similar style", (30, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.7, (30, 30, 30), 2)
    return img_a, img_b


def test_scenario_1_same_writer_different_words():
    """1. Two samples from the same writer with different words."""
    img_a, img_b = _get_iam_pair("a01-000u")
    fa = extract_visual_features(img_a)
    fb = extract_visual_features(img_b)

    score = compute_calibrated_visual_score(fa, fb)
    res = verify_handwriting_pair_core(fa, fb, ref_text=None, cmp_text=None, ref_nlp_feats=None, cmp_nlp_feats=None)

    assert score >= 0.65, f"Expected visual similarity >= 65%, got {score:.1%}"
    assert res["status"] == "compatible"


def test_scenario_2_same_writer_different_crops():
    """2. Two samples from the same writer with different image crops."""
    img_a, _ = _get_iam_pair("a01-000u")

    # Add large asymmetric white padding to simulate completely different margins/crop
    h, w = img_a.shape[:2]
    padded = np.full((h + 250, w + 350, 3), 255, dtype=np.uint8)
    padded[150:150+h, 180:180+w] = img_a

    fa = extract_visual_features(img_a)
    fb = extract_visual_features(padded)

    score = compute_calibrated_visual_score(fa, fb)
    res = verify_handwriting_pair_core(fa, fb, ref_text=None, cmp_text=None, ref_nlp_feats=None, cmp_nlp_feats=None)

    assert score >= 0.80, f"Expected crop-invariant similarity >= 80%, got {score:.1%}"
    assert res["status"] == "compatible"


def test_scenario_3_same_writer_different_writing_sizes():
    """3. Two samples from the same writer with different writing sizes (0.65x scale)."""
    img_a, _ = _get_iam_pair("a01-000u")
    h, w = img_a.shape[:2]

    # Resize by 0.65x to simulate different camera distance
    scaled = cv2.resize(img_a, (int(w * 0.65), int(h * 0.65)), interpolation=cv2.INTER_AREA)

    fa = extract_visual_features(img_a)
    fb = extract_visual_features(scaled)

    score = compute_calibrated_visual_score(fa, fb)
    res = verify_handwriting_pair_core(fa, fb, ref_text=None, cmp_text=None, ref_nlp_feats=None, cmp_nlp_feats=None)

    assert score >= 0.70, f"Expected scale-invariant similarity >= 70%, got {score:.1%}"
    assert res["status"] == "compatible"


def test_scenario_4_different_writers():
    """4. Two samples from clearly different writers."""
    img_a, _ = _get_iam_pair("a01-000u")
    img_b, _ = _get_iam_pair("a01-003")

    fa = extract_visual_features(img_a)
    fb = extract_visual_features(img_b)

    score = compute_calibrated_visual_score(fa, fb)
    res = verify_handwriting_pair_core(fa, fb, ref_text=None, cmp_text=None, ref_nlp_feats=None, cmp_nlp_feats=None)

    # Different writers should score lower than threshold or not_compatible
    assert score < 0.60 or res["status"] == "not_compatible"


def test_scenario_5_different_lighting_shadows():
    """5. Two samples with different lighting conditions (shadow gradient)."""
    img_a, _ = _get_iam_pair("a01-000u")
    h, w = img_a.shape[:2]

    # Apply severe diagonal lighting shadow gradient
    shadow = img_a.copy().astype(float)
    x_grad = np.linspace(0.40, 1.0, w).reshape(1, w, 1)
    shadow = np.clip(shadow * x_grad, 0, 255).astype(np.uint8)

    fa = extract_visual_features(img_a)
    fb = extract_visual_features(shadow)

    score = compute_calibrated_visual_score(fa, fb)
    res = verify_handwriting_pair_core(fa, fb, ref_text=None, cmp_text=None, ref_nlp_feats=None, cmp_nlp_feats=None)

    assert score >= 0.70, f"Expected illumination-invariant similarity >= 70%, got {score:.1%}"
    assert res["status"] == "compatible"


def test_scenario_6_very_short_handwriting_sample():
    """6. Very short handwriting samples (insufficient evidence)."""
    # Create empty / tiny ink patch
    blank = np.full((200, 300, 3), 255, dtype=np.uint8)
    fa = extract_visual_features(blank)
    img_b, _ = _get_iam_pair("a01-000u")
    fb = extract_visual_features(img_b)

    res = verify_handwriting_pair_core(fa, fb, ref_text=None, cmp_text=None, ref_nlp_feats=None, cmp_nlp_feats=None)

    assert res["status"] == "insufficient_evidence"
    assert "little or no detectable handwriting ink" in res["message"]


def test_scenario_7_notebook_lines_and_paper_background():
    """7. Samples with notebook lines and paper background."""
    img_a, _ = _get_iam_pair("a01-000u")
    h, w = img_a.shape[:2]

    # Add notebook ruled lines (horizontal + slight 1 degree tilt)
    ruled = img_a.copy()
    for y in range(40, h, 45):
        cv2.line(ruled, (0, y), (w, y + int(w * 0.015)), (160, 160, 160), 2)

    fa = extract_visual_features(img_a)
    fb = extract_visual_features(ruled)

    score = compute_calibrated_visual_score(fa, fb)
    res = verify_handwriting_pair_core(fa, fb, ref_text=None, cmp_text=None, ref_nlp_feats=None, cmp_nlp_feats=None)

    assert score >= 0.65, f"Expected ruled-line resilient similarity >= 65%, got {score:.1%}"
    assert res["status"] == "compatible"
