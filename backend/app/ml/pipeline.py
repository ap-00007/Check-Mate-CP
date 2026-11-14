"""
pipeline.py — Phase 1: Visual-only Check Mate verification.

Delegates to app.ml.verifier which loads the trained OpenCV-only classifier
from backend/artifacts/. No TrOCR or spaCy dependency in Phase 1.
"""
from typing import Any

from app.ml.verifier import verify_visual_pair


def analyze_image_pair(reference_bytes: bytes, comparison_bytes: bytes) -> dict[str, Any]:
    """
    Entry point called by the API route.

    Phase 1: runs the OpenCV visual fingerprint comparison.
    Returns a dict that is unpacked into VerificationResponse.
    """
    return verify_visual_pair(reference_bytes, comparison_bytes)
