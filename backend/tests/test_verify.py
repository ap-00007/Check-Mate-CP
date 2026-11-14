"""
Integration tests for /api/v1/verify (Phase 1 — visual only).
Runs against the TestClient so no live server is needed.
"""
import io
import numpy as np
import cv2
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _make_png(text_ink: bool = True, seed: int = 42) -> bytes:
    """Generate a synthetic handwriting-like sample as PNG bytes."""
    img = np.full((300, 500, 3), 255, dtype=np.uint8)
    if text_ink:
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


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_verify_same_style():
    png = _make_png(seed=1)
    r = client.post(
        "/api/v1/verify",
        files={
            "reference_image":  ("ref.png", io.BytesIO(png),  "image/png"),
            "comparison_image": ("cmp.png", io.BytesIO(png),  "image/png"),
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in {"compatible", "not_compatible", "insufficient_evidence"}
    assert body["visual_score"] is not None
    assert isinstance(body["feature_comparison"], list)
    assert len(body["feature_comparison"]) >= 8


def test_verify_different_styles():
    png_a = _make_png(seed=1)
    png_b = _make_png(seed=99)
    r = client.post(
        "/api/v1/verify",
        files={
            "reference_image":  ("ref.png", io.BytesIO(png_a), "image/png"),
            "comparison_image": ("cmp.png", io.BytesIO(png_b), "image/png"),
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert "visual_score" in body
    assert "combined_score" in body


def test_verify_blank_image_returns_insufficient():
    blank = _make_png(text_ink=False)
    r = client.post(
        "/api/v1/verify",
        files={
            "reference_image":  ("ref.png", io.BytesIO(blank), "image/png"),
            "comparison_image": ("cmp.png", io.BytesIO(blank), "image/png"),
        },
    )
    assert r.status_code == 200
    assert r.json()["status"] == "insufficient_evidence"


def test_verify_wrong_mime_type():
    r = client.post(
        "/api/v1/verify",
        files={
            "reference_image":  ("ref.txt", io.BytesIO(b"hello"), "text/plain"),
            "comparison_image": ("cmp.png", io.BytesIO(b"data"),  "image/png"),
        },
    )
    assert r.status_code == 415


def test_verify_missing_one_image():
    png = _make_png()
    r = client.post(
        "/api/v1/verify",
        files={
            "reference_image": ("ref.png", io.BytesIO(png), "image/png"),
        },
    )
    assert r.status_code == 422

