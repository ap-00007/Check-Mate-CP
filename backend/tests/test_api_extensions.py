"""
Integration tests for extended API endpoints:
- Multi-modal verification with text overrides
- Analysis session retrieval: GET /api/v1/analysis/{analysis_id}
- Stage inspection: GET /api/v1/analysis/{analysis_id}/stage/{stage}
- Feature explainability: GET /api/v1/analysis/{analysis_id}/explain/{feature}
- Educational learning modules: GET /api/v1/learning/modules and /{module_id}
"""
import io
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _make_sample_png(seed: int = 1) -> bytes:
    img = np.full((300, 500, 3), 250, dtype=np.uint8)
    for y in range(60, 260, 45):
        cv2.putText(
            img,
            f"Handwritten sample {seed} line verification test",
            (30, y),
            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
            0.7,
            (30, 30, 30),
            2,
        )
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


TEXT_A = (
    "The historical development of handwritten documents reveals deeply ingrained "
    "patterns of personal expression. Although modern communication has largely transitioned "
    "to digital keystrokes, handwriting remains an intimate cognitive signature. Each writer "
    "unconsciously modulates letter proportions, stroke velocities, and grammatical structures. "
    "When analyzing authorship, stylometric markers provide robust evidence across varying topics."
)

TEXT_B = (
    "Although digital systems dominate modern correspondence, handwritten manuscripts preserve "
    "invaluable linguistic fingerprints. When researchers study personal expression, they observe "
    "consistent lexical richness and syntactic pacing. Each author exhibits distinct subconscious "
    "habits in function word placement and clause organization, providing durable verification signals."
)


def test_verify_with_text_overrides():
    png_a = _make_sample_png(seed=1)
    png_b = _make_sample_png(seed=2)

    res = client.post(
        "/api/v1/verify",
        files={
            "reference_image": ("ref.png", io.BytesIO(png_a), "image/png"),
            "comparison_image": ("cmp.png", io.BytesIO(png_b), "image/png"),
        },
        data={
            "reference_text": TEXT_A,
            "comparison_text": TEXT_B,
        },
    )

    assert res.status_code == 200
    body = res.json()
    assert body["status"] in {"compatible", "not_compatible"}
    assert body["visual_score"] is not None
    assert body["nlp_score"] is not None
    assert body["combined_score"] is not None
    assert body["analysis_id"] is not None

    # Verify pipeline stages
    assert "pipeline_stages" in body
    assert "preprocessing" in body["pipeline_stages"]
    assert "line_segmentation" in body["pipeline_stages"]
    assert "ocr" in body["pipeline_stages"]
    assert "stylometry" in body["pipeline_stages"]
    assert "pairwise_verification" in body["pipeline_stages"]
    assert "explainability" in body["pipeline_stages"]

    # Verify NLP breakdown
    assert "nlp_module_scores" in body
    assert "lexical" in body["nlp_module_scores"]
    assert "syntactic" in body["nlp_module_scores"]

    # Verify explanations
    assert len(body["explanations"]) > 0

    analysis_id = body["analysis_id"]

    # Test GET /api/v1/analysis/{analysis_id}
    get_res = client.get(f"/api/v1/analysis/{analysis_id}")
    assert get_res.status_code == 200
    assert get_res.json()["analysis_id"] == analysis_id

    # Test GET /api/v1/analysis/{analysis_id}/stage/preprocessing
    stage_res = client.get(f"/api/v1/analysis/{analysis_id}/stage/preprocessing")
    assert stage_res.status_code == 200
    assert stage_res.json()["stage"] == "preprocessing"
    assert stage_res.json()["status"] == "completed"

    # Test GET /api/v1/analysis/{analysis_id}/explain/{feature}
    first_feat = body["explanations"][0]["feature"]
    exp_res = client.get(f"/api/v1/analysis/{analysis_id}/explain/{first_feat}")
    assert exp_res.status_code == 200
    assert exp_res.json()["feature"] == first_feat


def test_insufficient_text_omits_nlp_score():
    png_a = _make_sample_png(seed=1)
    png_b = _make_sample_png(seed=2)

    res = client.post(
        "/api/v1/verify",
        files={
            "reference_image": ("ref.png", io.BytesIO(png_a), "image/png"),
            "comparison_image": ("cmp.png", io.BytesIO(png_b), "image/png"),
        },
        data={
            "reference_text": "Too short.",
            "comparison_text": "Also short note.",
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["nlp_score"] is None
    assert body["visual_score"] is not None
    assert any("fewer than 20 words" in w for w in body["warnings"])


def test_analysis_not_found():
    res = client.get("/api/v1/analysis/nonexistent-id-12345")
    assert res.status_code == 404


def test_stage_not_found():
    png_a = _make_sample_png(seed=1)
    post_res = client.post(
        "/api/v1/verify",
        files={
            "reference_image": ("ref.png", io.BytesIO(png_a), "image/png"),
            "comparison_image": ("cmp.png", io.BytesIO(png_a), "image/png"),
        },
    )
    aid = post_res.json()["analysis_id"]
    res = client.get(f"/api/v1/analysis/{aid}/stage/fake_stage")
    assert res.status_code == 404


def test_learning_modules_endpoints():
    res = client.get("/api/v1/learning/modules")
    assert res.status_code == 200
    data = res.json()
    assert data["count"] >= 10
    assert any(m["id"] == "ocr" for m in data["modules"])
    assert any(m["id"] == "stylometry" for m in data["modules"])
    assert any(m["id"] == "evaluation_metrics" for m in data["modules"])

    mod_res = client.get("/api/v1/learning/modules/ocr")
    assert mod_res.status_code == 200
    assert mod_res.json()["title"] == "Optical Character Recognition (TrOCR)"

    # Test unknown module returns 404
    missing_res = client.get("/api/v1/learning/modules/unknown_module_xyz")
    assert missing_res.status_code == 404
