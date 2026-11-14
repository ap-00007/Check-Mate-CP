from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from app.schemas.verification import (
    VerificationResponse,
    StageDetailResponse,
    FeatureExplanationResponse,
    LearningModuleItem,
    LearningModuleListResponse,
)
from app.services.verification_service import (
    verify_handwriting,
    get_analysis_by_id,
    get_stage_by_id,
    get_feature_explanation_by_id,
)
from app.ml.explainability import get_all_learning_modules, get_learning_module_by_id
from app.core.config import settings

router = APIRouter()

MAX_BYTES = settings.max_upload_mb * 1024 * 1024
ALLOWED_MIME_TYPES = {"image/png", "image/jpeg", "image/webp"}


@router.post("/verify", response_model=VerificationResponse)
async def verify(
    reference_image: UploadFile = File(...),
    comparison_image: UploadFile = File(...),
    reference_text: Optional[str] = Form(None),
    comparison_text: Optional[str] = Form(None),
):
    """
    Submits two handwriting images for multi-modal authorship verification.
    Supports optional text overrides for benchmarking and test isolation.
    """
    if (
        reference_image.content_type not in ALLOWED_MIME_TYPES
        or comparison_image.content_type not in ALLOWED_MIME_TYPES
    ):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Upload PNG, JPEG, or WEBP images.",
        )

    ref_bytes = await reference_image.read()
    cmp_bytes = await comparison_image.read()

    if not ref_bytes or not cmp_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Both reference and comparison images are required.",
        )

    if len(ref_bytes) > MAX_BYTES or len(cmp_bytes) > MAX_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Each image must be under {settings.max_upload_mb} MB.",
        )

    try:
        result = await verify_handwriting(
            reference_bytes=ref_bytes,
            comparison_bytes=cmp_bytes,
            reference_text_override=reference_text,
            comparison_text_override=comparison_text,
        )
        return result
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(val_err),
        )


@router.get("/analysis/{analysis_id}", response_model=VerificationResponse)
def get_analysis(analysis_id: str):
    """
    Retrieves full verification results, stage metadata, and evidence for a past analysis.
    """
    record = get_analysis_by_id(analysis_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis session '{analysis_id}' not found or expired.",
        )
    return record


@router.get("/analysis/{analysis_id}/stage/{stage}", response_model=StageDetailResponse)
def get_analysis_stage(analysis_id: str, stage: str):
    """
    Retrieves execution details and diagnostics for a specific pipeline stage.
    """
    stage_data = get_stage_by_id(analysis_id, stage)
    if not stage_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stage '{stage}' not found for analysis '{analysis_id}'.",
        )
    return stage_data


@router.get("/analysis/{analysis_id}/explain/{feature}", response_model=FeatureExplanationResponse)
def get_feature_explanation(analysis_id: str, feature: str):
    """
    Retrieves mathematical definition, formula, evidence difference, and interpretation
    for a specific feature within an analysis.
    """
    exp_data = get_feature_explanation_by_id(analysis_id, feature)
    if not exp_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Feature explanation for '{feature}' not found in analysis '{analysis_id}'.",
        )
    return exp_data


@router.get("/learning/modules", response_model=LearningModuleListResponse)
def list_learning_modules():
    """
    Lists all curated educational learning modules.
    """
    modules = get_all_learning_modules()
    return LearningModuleListResponse(modules=modules, count=len(modules))


@router.get("/learning/modules/{module_id}", response_model=LearningModuleItem)
def get_learning_module(module_id: str):
    """
    Retrieves a specific educational concept module by ID.
    """
    mod = get_learning_module_by_id(module_id)
    if not mod:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Learning module '{module_id}' not found.",
        )
    return mod
