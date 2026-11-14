from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


StageStatus = Literal[
    "pending",
    "processing",
    "completed",
    "partial",
    "failed",
    "not_configured",
]


class FeatureDifference(BaseModel):
    feature: str
    reference: float
    comparison: float
    absolute_difference: float
    relative_difference: float | None = None
    interpretation: str | None = None
    reliability: str | None = None


class StageInfo(BaseModel):
    status: StageStatus
    execution_time_ms: float | None = None
    message: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class OCRLineResult(BaseModel):
    line_number: int
    text: str
    confidence: float | None = None


class DocumentOCRResult(BaseModel):
    status: StageStatus
    full_text: str = ""
    lines: list[OCRLineResult] = Field(default_factory=list)
    line_count: int = 0
    word_count: int = 0
    confidence: float | None = None
    warnings: list[str] = Field(default_factory=list)


class FeatureExplanation(BaseModel):
    feature: str
    name: str
    module: str
    definition: str
    formula: str | None = None
    reference_value: float | None = None
    comparison_value: float | None = None
    difference: float | None = None
    interpretation: str
    reliability_limitations: str


class VerificationResponse(BaseModel):
    # Core legacy fields (strictly preserved for frontend compatibility)
    status: Literal["analysis_only", "compatible", "not_compatible", "insufficient_evidence"]
    message: str
    nlp_score: float | None = None
    visual_score: float | None = None
    combined_score: float | None = None
    reference_text: str | None = None
    comparison_text: str | None = None
    feature_comparison: list[FeatureDifference] = Field(default_factory=list)
    visual_feature_scores: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    # Extended functional fields
    analysis_id: str | None = None
    calibrated_threshold: float | None = None
    decision_margin: float | None = None
    pipeline_stages: dict[str, StageInfo] = Field(default_factory=dict)
    reference_ocr: DocumentOCRResult | None = None
    comparison_ocr: DocumentOCRResult | None = None
    nlp_module_scores: dict[str, float] = Field(default_factory=dict)
    nlp_feature_comparison: list[FeatureDifference] = Field(default_factory=list)
    explanations: list[FeatureExplanation] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class StageDetailResponse(BaseModel):
    analysis_id: str
    stage: str
    status: StageStatus
    execution_time_ms: float | None = None
    message: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class FeatureExplanationResponse(BaseModel):
    analysis_id: str | None = None
    feature: str
    explanation: FeatureExplanation


class LearningModuleItem(BaseModel):
    id: str
    title: str
    category: str
    summary: str
    detailed_explanation: str
    mathematical_formulation: str | None = None
    relevance_to_authorship: str
    limitations: str


class LearningModuleListResponse(BaseModel):
    modules: list[LearningModuleItem]
    count: int
