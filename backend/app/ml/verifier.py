"""
verifier.py — Reusable Model Loading Service and Pairwise Verification.

Integrates:
1. Calibrated biometric visual graphology verification (continuous, scale-invariant, shadow- & line-resilient).
2. NLP stylometric pair classifier (116 linguistic differences + subword character n-gram TF-IDF cosine).
3. Validated multi-modal fusion combining visual physical graphology and linguistic stylometry.
4. Robust evidence validation and insufficient-evidence guards.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
import joblib
import numpy as np
from sklearn.metrics.pairwise import paired_cosine_distances

from app.core.config import settings
from app.ml.stylometry import (
    ALL_NLP_FEATURE_COLS,
    compute_nlp_module_similarities,
    build_nlp_feature_comparison_rows,
)
from app.ml.visual import (
    FEATURE_NAMES,
    compute_per_feature_similarity,
    compute_calibrated_visual_score,
    build_feature_comparison_rows,
)

logger = logging.getLogger(__name__)


class VerificationModelService:
    """
    Singleton service managing artifact loading and model inference.
    Loads models once, reuses them, and provides graceful fallbacks.
    """
    def __init__(self, artifact_dir: Path | str = settings.artifact_dir):
        self.artifact_dir = Path(artifact_dir)
        self._visual_model = None
        self._visual_cols = None
        self._nlp_model = None
        self._nlp_cols = None
        self._char_vectorizer = None
        self._fusion_meta = None
        self._config: dict[str, Any] = {}
        self._loaded = False
        self._load_error: str | None = None

    def load_artifacts(self) -> None:
        if self._loaded:
            return

        try:
            cfg_path = self.artifact_dir / "fusion_config.json"
            if cfg_path.exists():
                self._config = json.loads(cfg_path.read_text(encoding="utf-8"))

            vis_cols_path = self.artifact_dir / "visual_feature_columns_opencv.joblib"
            if vis_cols_path.exists():
                self._visual_cols = joblib.load(vis_cols_path)
            else:
                self._visual_cols = FEATURE_NAMES

            nlp_path = self.artifact_dir / "nlp_pair_classifier.joblib"
            if nlp_path.exists():
                self._nlp_model = joblib.load(nlp_path)

            nlp_cols_path = self.artifact_dir / "nlp_feature_columns.joblib"
            if nlp_cols_path.exists():
                self._nlp_cols = joblib.load(nlp_cols_path)
            else:
                self._nlp_cols = ALL_NLP_FEATURE_COLS

            char_vec_path = self.artifact_dir / "char_tfidf_vectorizer.joblib"
            if char_vec_path.exists():
                self._char_vectorizer = joblib.load(char_vec_path)

            meta_path = self.artifact_dir / "fusion_meta_model.joblib"
            if meta_path.exists():
                self._fusion_meta = joblib.load(meta_path)

            self._loaded = True
            logger.info("Successfully loaded Check Mate verification models and artifacts.")

        except Exception as exc:
            msg = f"Failed to load verification artifacts ({type(exc).__name__}: {exc})"
            logger.warning(msg)
            self._load_error = msg
            self._loaded = True

    @property
    def config(self) -> dict[str, Any]:
        self.load_artifacts()
        return self._config

    def predict_visual_pair(
        self,
        ref_feats: dict[str, float],
        cmp_feats: dict[str, float],
    ) -> float:
        """
        Predicts visual handwriting compatibility score.
        Uses calibrated biometric distance mapping invariant to camera distance,
        crop differences, margins, and notebook ruling.
        """
        score = compute_calibrated_visual_score(ref_feats, cmp_feats)
        return float(np.clip(score, 0.0, 1.0))

    def predict_nlp_pair(
        self,
        ref_nlp: dict[str, float],
        cmp_nlp: dict[str, float],
        ref_text: str,
        cmp_text: str,
    ) -> tuple[float | None, float | None]:
        """
        Predicts NLP stylometric compatibility score and character n-gram cosine.
        Returns (nlp_score, char_ngram_cosine).
        """
        self.load_artifacts()
        if not ref_text.strip() or not cmp_text.strip():
            return None, None

        # Subword character n-gram TF-IDF cosine distance
        char_cos: float | None = None
        if self._char_vectorizer is not None:
            try:
                tfidf_mat = self._char_vectorizer.transform([ref_text, cmp_text])
                char_cos = float(1.0 - paired_cosine_distances(tfidf_mat[0], tfidf_mat[1])[0])
                char_cos = float(np.clip(char_cos, 0.0, 1.0))
            except Exception as exc:
                logger.warning(f"Character vectorizer transform failed: {exc}")
                char_cos = None

        if self._nlp_model is None:
            # Fallback to module similarity mean if trained model offline
            mod_sims = compute_nlp_module_similarities(ref_nlp, cmp_nlp, char_ngram_cosine=char_cos)
            return float(np.mean(list(mod_sims.values()))), char_cos

        try:
            cols = self._nlp_cols or ALL_NLP_FEATURE_COLS
            ref_vec = np.array([ref_nlp.get(c, 0.0) for c in cols], dtype=float)
            cmp_vec = np.array([cmp_nlp.get(c, 0.0) for c in cols], dtype=float)
            diff = np.abs(ref_vec - cmp_vec)

            expected_dim = getattr(self._nlp_model, "n_features_in_", len(cols))
            if expected_dim == len(diff) + 1:
                cos_val = char_cos if char_cos is not None else 0.5
                x_pair = np.append(diff, cos_val).reshape(1, -1)
            else:
                x_pair = diff[:expected_dim].reshape(1, -1)

            prob = float(self._nlp_model.predict_proba(x_pair)[0, 1])
            return float(np.clip(prob, 0.0, 1.0)), char_cos
        except Exception as exc:
            logger.warning(f"NLP model inference failed: {exc}")
            mod_sims = compute_nlp_module_similarities(ref_nlp, cmp_nlp, char_ngram_cosine=char_cos)
            return float(np.mean(list(mod_sims.values()))), char_cos

    def predict_fusion(
        self,
        nlp_score: float | None,
        visual_score: float | None,
    ) -> float:
        """
        Computes validated multi-modal fusion score.
        Uses calibrated alpha and beta from fusion_config.json.
        """
        self.load_artifacts()
        alpha = float(self._config.get("alpha", settings.alpha))
        beta = float(self._config.get("beta", settings.beta))

        if nlp_score is not None and visual_score is not None:
            combined = alpha * nlp_score + beta * visual_score
            return float(np.clip(combined, 0.0, 1.0))
        elif visual_score is not None:
            return visual_score
        elif nlp_score is not None:
            return nlp_score
        return 0.0


# Global singleton instance
verification_service = VerificationModelService()


def verify_handwriting_pair_core(
    ref_visual_feats: dict[str, float],
    cmp_visual_feats: dict[str, float],
    ref_text: str | None,
    cmp_text: str | None,
    ref_nlp_feats: dict[str, float] | None,
    cmp_nlp_feats: dict[str, float] | None,
    min_words: int = settings.min_words_for_verification,
) -> dict[str, Any]:
    """
    Core pairwise verification combining visual and stylometric evidence.
    Returns scores, feature differences, status, message, and warnings.
    """
    warnings: list[str] = [
        "This analysis estimates stylistic compatibility and is not forensic proof of identity.",
    ]

    # Guard 1: Physical ink density check
    MIN_INK = 0.003
    ref_ink = ref_visual_feats.get("vis_ink_density", 0.0)
    cmp_ink = cmp_visual_feats.get("vis_ink_density", 0.0)

    if ref_ink < MIN_INK or cmp_ink < MIN_INK:
        return {
            "status": "insufficient_evidence",
            "message": (
                "One or both images appear to contain very little or no detectable handwriting ink. "
                "Please upload clear handwriting samples."
            ),
            "nlp_score": None,
            "visual_score": None,
            "combined_score": None,
            "calibrated_threshold": None,
            "decision_margin": None,
            "reference_text": ref_text,
            "comparison_text": cmp_text,
            "feature_comparison": [],
            "visual_feature_scores": {},
            "nlp_module_scores": {},
            "nlp_feature_comparison": [],
            "warnings": [
                "Image too blank or low-contrast for reliable handwriting analysis."
            ],
        }

    # 1. Visual Verification (Calibrated Biometric Formulation)
    visual_score = verification_service.predict_visual_pair(ref_visual_feats, cmp_visual_feats)
    per_feat_sims = compute_per_feature_similarity(ref_visual_feats, cmp_visual_feats)
    visual_feature_rows = build_feature_comparison_rows(ref_visual_feats, cmp_visual_feats)

    visual_module_scores = {
        "stroke_similarity": round(
            float(np.mean([
                per_feat_sims.get("vis_stroke_width_mean", 0.8),
                per_feat_sims.get("vis_stroke_width_std", 0.8),
                per_feat_sims.get("vis_stroke_to_height_ratio", 0.8),
                per_feat_sims.get("vis_ink_density", 0.8),
            ])), 4
        ),
        "slant_similarity": round(
            float(np.mean([
                per_feat_sims.get("vis_slant_angle_mean", 0.8),
                per_feat_sims.get("vis_slant_angle_std", 0.8),
            ])), 4
        ),
        "letter_proportion_similarity": round(
            float(np.mean([
                per_feat_sims.get("vis_cc_aspect_mean", 0.8),
                per_feat_sims.get("vis_cc_height_rel_std", 0.8),
                per_feat_sims.get("vis_char_spacing_ratio", 0.8),
                per_feat_sims.get("vis_baseline_variation", 0.8),
            ])), 4
        ),
    }

    # 2. NLP Stylometric Verification
    ref_clean_text = (ref_text or "").strip()
    cmp_clean_text = (cmp_text or "").strip()
    ref_wc = len(ref_clean_text.split())
    cmp_wc = len(cmp_clean_text.split())

    nlp_score: float | None = None
    char_cos: float | None = None
    nlp_module_scores: dict[str, float] = {}
    nlp_feature_rows: list[dict[str, Any]] = []

    if ref_nlp_feats is not None and cmp_nlp_feats is not None:
        if ref_wc < min_words or cmp_wc < min_words:
            warnings.append(
                f"Sample contains fewer than {min_words} words: NLP stylometric score omitted due to insufficient text sample."
            )
        else:
            nlp_score, char_cos = verification_service.predict_nlp_pair(
                ref_nlp_feats, cmp_nlp_feats, ref_clean_text, cmp_clean_text
            )
            nlp_module_scores = compute_nlp_module_similarities(
                ref_nlp_feats, cmp_nlp_feats, char_ngram_cosine=char_cos
            )
            nlp_feature_rows = build_nlp_feature_comparison_rows(
                ref_nlp_feats, cmp_nlp_feats, ref_wc, cmp_wc
            )
            warnings.append("OCR transcription errors can influence NLP stylometric measurements.")
    else:
        warnings.append("OCR/Text unavailable: running in visual-only verification mode.")

    # 3. Validated Multi-Modal Fusion & Threshold Decision
    cfg = verification_service.config
    if nlp_score is not None:
        combined_score = verification_service.predict_fusion(nlp_score, visual_score)
        threshold = float(cfg.get("calibrated_threshold", settings.default_threshold))
    else:
        # Visual-only calibrated threshold (calibrated at 0.58 on writer-disjoint IAM pairs)
        combined_score = visual_score
        threshold = float(cfg.get("opencv_only_threshold", 0.58))

    status = "compatible" if combined_score >= threshold else "not_compatible"
    decision_margin = round(combined_score - threshold, 4)

    if status == "compatible":
        message = (
            f"Handwriting styles appear compatible "
            f"(visual similarity {combined_score:.1%}, threshold {threshold:.1%}, margin {decision_margin:+.1%})."
        )
    else:
        message = (
            f"Handwriting styles appear different "
            f"(visual similarity {combined_score:.1%}, threshold {threshold:.1%}, margin {decision_margin:+.1%})."
        )

    return {
        "status": status,
        "message": message,
        "visual_score": round(visual_score, 4),
        "nlp_score": round(nlp_score, 4) if nlp_score is not None else None,
        "combined_score": round(combined_score, 4),
        "calibrated_threshold": round(threshold, 4),
        "decision_margin": decision_margin,
        "reference_text": ref_clean_text if ref_clean_text else None,
        "comparison_text": cmp_clean_text if cmp_clean_text else None,
        "feature_comparison": visual_feature_rows,
        "visual_feature_scores": visual_module_scores,
        "nlp_module_scores": nlp_module_scores,
        "nlp_feature_comparison": nlp_feature_rows,
        "warnings": warnings,
    }
