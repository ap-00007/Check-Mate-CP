"""
verification_service.py — Service layer orchestrating verification requests and analysis retrieval.
"""
from __future__ import annotations

from typing import Any
from app.ml.pipeline import analyze_image_pair
from app.services.analysis_store import analysis_store


async def verify_handwriting(
    reference_bytes: bytes,
    comparison_bytes: bytes,
    reference_text_override: str | None = None,
    comparison_text_override: str | None = None,
) -> dict[str, Any]:
    """
    Executes handwriting verification pipeline asynchronously.
    """
    result = analyze_image_pair(
        reference_bytes=reference_bytes,
        comparison_bytes=comparison_bytes,
        reference_text_override=reference_text_override,
        comparison_text_override=comparison_text_override,
    )
    return result


def get_analysis_by_id(analysis_id: str) -> dict[str, Any] | None:
    return analysis_store.get(analysis_id)


def get_stage_by_id(analysis_id: str, stage_name: str) -> dict[str, Any] | None:
    return analysis_store.get_stage(analysis_id, stage_name)


def get_feature_explanation_by_id(analysis_id: str, feature_name: str) -> dict[str, Any] | None:
    return analysis_store.get_feature_explanation(analysis_id, feature_name)
