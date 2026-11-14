"""
pipeline.py — Complete Functional Check Mate Verification Pipeline.

Architecture:
Image Upload
→ Image Validation
→ Image Preprocessing
→ Text Line Segmentation
→ Handwritten OCR (TrOCR)
→ NLP Stylometric Feature Extraction
→ Visual Handwriting Feature Extraction
→ Pairwise Verification
→ Evidence Comparison
→ Explainability
→ Structured API Response

Each stage records execution time, truthful status, diagnostic messages,
and intermediate outputs stored in AnalysisStore.
"""
from __future__ import annotations

import time
import uuid
import logging
from typing import Any

from app.ml.preprocessor import preprocess_handwriting_document, ImageValidationError
from app.ml.ocr import transcribe_document_lines
from app.ml.stylometry import extract_stylometric_features
from app.ml.visual import extract_visual_features
from app.ml.character_profiler import profile_character_geometry
from app.ml.verifier import verify_handwriting_pair_core
from app.ml.explainability import build_feature_explanations
from app.services.analysis_store import analysis_store
from app.schemas.verification import StageInfo, DocumentOCRResult

logger = logging.getLogger(__name__)


def execute_pipeline(
    reference_bytes: bytes,
    comparison_bytes: bytes,
    reference_text_override: str | None = None,
    comparison_text_override: str | None = None,
) -> dict[str, Any]:
    """
    Executes the complete Check Mate handwriting verification and analysis pipeline.
    """
    analysis_id = str(uuid.uuid4())
    pipeline_stages: dict[str, StageInfo] = {}
    stage_start = time.perf_counter()

    # ---------------------------------------------------------
    # Stage 1: Image Validation & Preprocessing
    # ---------------------------------------------------------
    t0 = time.perf_counter()
    try:
        ref_prep = preprocess_handwriting_document(reference_bytes)
        cmp_prep = preprocess_handwriting_document(comparison_bytes)
        dt_prep = round((time.perf_counter() - t0) * 1000, 2)
        pipeline_stages["preprocessing"] = StageInfo(
            status="completed",
            execution_time_ms=dt_prep,
            message="Images validated, paper cropped, illumination normalized, notebook lines removed.",
            details={
                "reference": ref_prep["metadata"],
                "comparison": cmp_prep["metadata"],
            },
        )
    except ImageValidationError as val_err:
        dt_val = round((time.perf_counter() - t0) * 1000, 2)
        pipeline_stages["preprocessing"] = StageInfo(
            status="failed",
            execution_time_ms=dt_val,
            message=str(val_err),
            details={"error": str(val_err)},
        )
        raise

    # ---------------------------------------------------------
    # Stage 2: Text Line Segmentation
    # ---------------------------------------------------------
    t1 = time.perf_counter()
    ref_lines = ref_prep["line_crops"]
    cmp_lines = cmp_prep["line_crops"]
    dt_seg = round((time.perf_counter() - t1) * 1000, 2)
    pipeline_stages["line_segmentation"] = StageInfo(
        status="completed" if (ref_lines and cmp_lines) else "partial",
        execution_time_ms=dt_seg,
        message=f"Segmented {len(ref_lines)} reference line(s) and {len(cmp_lines)} comparison line(s).",
        details={
            "reference_lines_detected": len(ref_lines),
            "comparison_lines_detected": len(cmp_lines),
        },
    )

    # ---------------------------------------------------------
    # Stage 3: Handwritten OCR (TrOCR)
    # ---------------------------------------------------------
    t2 = time.perf_counter()
    ref_ocr = transcribe_document_lines(ref_lines, text_override=reference_text_override)
    cmp_ocr = transcribe_document_lines(cmp_lines, text_override=comparison_text_override)
    dt_ocr = round((time.perf_counter() - t2) * 1000, 2)

    if ref_ocr.status == "completed" and cmp_ocr.status == "completed":
        ocr_status = "completed"
    elif ref_ocr.status == "not_configured" or cmp_ocr.status == "not_configured":
        ocr_status = "not_configured"
    else:
        ocr_status = "partial"

    pipeline_stages["ocr"] = StageInfo(
        status=ocr_status,
        execution_time_ms=dt_ocr,
        message=(
            f"OCR {ocr_status}: Transcribed {ref_ocr.word_count} reference words "
            f"and {cmp_ocr.word_count} comparison words."
        ),
        details={
            "reference_ocr_status": ref_ocr.status,
            "comparison_ocr_status": cmp_ocr.status,
            "reference_word_count": ref_ocr.word_count,
            "comparison_word_count": cmp_ocr.word_count,
            "reference_confidence": ref_ocr.confidence,
            "comparison_confidence": cmp_ocr.confidence,
        },
    )

    # ---------------------------------------------------------
    # Stage 4: Visual Handwriting Feature Extraction
    # ---------------------------------------------------------
    t3 = time.perf_counter()
    ref_vis = extract_visual_features(ref_prep["paper_image"])
    cmp_vis = extract_visual_features(cmp_prep["paper_image"])
    dt_vis = round((time.perf_counter() - t3) * 1000, 2)
    pipeline_stages["visual_feature_extraction"] = StageInfo(
        status="completed",
        execution_time_ms=dt_vis,
        message="Extracted 8 normalized physical handwriting features.",
        details={
            "reference_visual": ref_vis,
            "comparison_visual": cmp_vis,
        },
    )

    # ---------------------------------------------------------
    # Stage 5: Character Profiling (safe, guarded)
    # ---------------------------------------------------------
    t4 = time.perf_counter()
    ref_char_prof = profile_character_geometry(ref_prep["clean_binary"])
    cmp_char_prof = profile_character_geometry(cmp_prep["clean_binary"])
    dt_char = round((time.perf_counter() - t4) * 1000, 2)
    char_status = "completed" if ref_char_prof["status"] == "available" else "partial"
    pipeline_stages["character_profiling"] = StageInfo(
        status=char_status,
        execution_time_ms=dt_char,
        message=f"Character profiling status: {ref_char_prof['status']}.",
        details={
            "reference_character_profile": ref_char_prof,
            "comparison_character_profile": cmp_char_prof,
        },
    )

    # ---------------------------------------------------------
    # Stage 6: NLP Stylometric Feature Extraction
    # ---------------------------------------------------------
    t5 = time.perf_counter()
    ref_has_text = bool(ref_ocr.full_text.strip())
    cmp_has_text = bool(cmp_ocr.full_text.strip())

    if ref_has_text and cmp_has_text:
        ref_nlp = extract_stylometric_features(ref_ocr.full_text, line_count=ref_ocr.line_count)
        cmp_nlp = extract_stylometric_features(cmp_ocr.full_text, line_count=cmp_ocr.line_count)
        nlp_stage_status = "completed"
    else:
        ref_nlp = None
        cmp_nlp = None
        nlp_stage_status = "not_configured" if ocr_status == "not_configured" else "partial"

    dt_nlp = round((time.perf_counter() - t5) * 1000, 2)
    pipeline_stages["stylometry"] = StageInfo(
        status=nlp_stage_status,
        execution_time_ms=dt_nlp,
        message=(
            "Extracted 116 stylometric features across Lexical, Syntactic, Structural, "
            "Punctuation, and Subword modules." if ref_nlp else "Stylometry skipped due to empty transcription."
        ),
        details={
            "features_extracted": 116 if ref_nlp else 0,
        },
    )

    # ---------------------------------------------------------
    # Stage 7 & 8: Pairwise Verification & Evidence Comparison
    # ---------------------------------------------------------
    t6 = time.perf_counter()
    verif_res = verify_handwriting_pair_core(
        ref_visual_feats=ref_vis,
        cmp_visual_feats=cmp_vis,
        ref_text=ref_ocr.full_text,
        cmp_text=cmp_ocr.full_text,
        ref_nlp_feats=ref_nlp,
        cmp_nlp_feats=cmp_nlp,
    )
    dt_verif = round((time.perf_counter() - t6) * 1000, 2)
    pipeline_stages["pairwise_verification"] = StageInfo(
        status="completed" if verif_res["status"] != "insufficient_evidence" else "partial",
        execution_time_ms=dt_verif,
        message=verif_res["message"],
        details={
            "visual_score": verif_res.get("visual_score"),
            "nlp_score": verif_res.get("nlp_score"),
            "combined_score": verif_res.get("combined_score"),
            "calibrated_threshold": verif_res.get("calibrated_threshold"),
            "decision_margin": verif_res.get("decision_margin"),
        },
    )

    # ---------------------------------------------------------
    # Stage 9: Explainability
    # ---------------------------------------------------------
    t7 = time.perf_counter()
    explanations = build_feature_explanations(
        visual_rows=verif_res.get("feature_comparison", []),
        nlp_rows=verif_res.get("nlp_feature_comparison", []),
    )
    dt_exp = round((time.perf_counter() - t7) * 1000, 2)
    pipeline_stages["explainability"] = StageInfo(
        status="completed",
        execution_time_ms=dt_exp,
        message=f"Compiled educational explanations for {len(explanations)} features.",
        details={"explanations_count": len(explanations)},
    )

    total_pipeline_time = round((time.perf_counter() - stage_start) * 1000, 2)

    full_response = {
        # Core backward-compatible fields
        "status": verif_res["status"],
        "message": verif_res["message"],
        "nlp_score": verif_res["nlp_score"],
        "visual_score": verif_res["visual_score"],
        "combined_score": verif_res["combined_score"],
        "reference_text": verif_res["reference_text"],
        "comparison_text": verif_res["comparison_text"],
        "feature_comparison": verif_res["feature_comparison"],
        "visual_feature_scores": verif_res["visual_feature_scores"],
        "warnings": verif_res["warnings"],

        # Extended fields
        "analysis_id": analysis_id,
        "calibrated_threshold": verif_res.get("calibrated_threshold"),
        "decision_margin": verif_res.get("decision_margin"),
        "pipeline_stages": pipeline_stages,
        "reference_ocr": ref_ocr,
        "comparison_ocr": cmp_ocr,
        "nlp_module_scores": verif_res["nlp_module_scores"],
        "nlp_feature_comparison": verif_res["nlp_feature_comparison"],
        "explanations": explanations,
        "metadata": {
            "total_execution_time_ms": total_pipeline_time,
            "reference_lines": len(ref_lines),
            "comparison_lines": len(cmp_lines),
            "reference_words": ref_ocr.word_count,
            "comparison_words": cmp_ocr.word_count,
        },
    }

    # Store analysis session in-memory
    analysis_store.save(analysis_id, full_response)

    return full_response


def analyze_image_pair(
    reference_bytes: bytes,
    comparison_bytes: bytes,
    reference_text_override: str | None = None,
    comparison_text_override: str | None = None,
) -> dict[str, Any]:
    """
    Standard entry point for verification service.
    """
    return execute_pipeline(
        reference_bytes=reference_bytes,
        comparison_bytes=comparison_bytes,
        reference_text_override=reference_text_override,
        comparison_text_override=comparison_text_override,
    )
