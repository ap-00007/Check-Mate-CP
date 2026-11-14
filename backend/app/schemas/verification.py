from pydantic import BaseModel, Field
from typing import Literal

class FeatureDifference(BaseModel):
    feature: str
    reference: float
    comparison: float
    absolute_difference: float

class VerificationResponse(BaseModel):
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
