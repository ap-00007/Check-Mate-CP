"""
Unit tests for document preprocessing and text-line segmentation.
"""
import numpy as np
import cv2
import pytest

from app.ml.preprocessor import (
    validate_and_decode_image,
    extract_paper_region,
    remove_ruled_and_margin_lines,
    normalize_lighting_and_contrast,
    normalize_scale,
    segment_text_lines,
    preprocess_handwriting_document,
    ImageValidationError,
)


def _synthetic_doc(h=400, w=600, n_lines=4) -> bytes:
    img = np.full((h, w, 3), 245, dtype=np.uint8)
    for i in range(n_lines):
        y = 60 + i * 70
        cv2.putText(img, f"Document line {i+1} testing text segmentation", (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def test_validate_and_decode_valid():
    raw = _synthetic_doc()
    img, meta = validate_and_decode_image(raw)
    assert img is not None
    assert meta["original_width"] == 600
    assert meta["original_height"] == 400
    assert meta["channels"] == 3


def test_validate_empty_bytes_raises():
    with pytest.raises(ImageValidationError, match="empty"):
        validate_and_decode_image(b"")


def test_validate_corrupted_bytes_raises():
    with pytest.raises(ImageValidationError, match="decode"):
        validate_and_decode_image(b"not an image byte stream")


def test_validate_too_small_raises():
    tiny = np.zeros((16, 16, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".png", tiny)
    with pytest.raises(ImageValidationError, match="too small"):
        validate_and_decode_image(buf.tobytes())


def test_paper_crop_and_ruled_lines():
    raw = _synthetic_doc()
    img, _ = validate_and_decode_image(raw)
    paper, crop_meta = extract_paper_region(img)
    assert paper.shape[0] > 50 and paper.shape[1] > 50
    assert "paper_detected" in crop_meta

    norm, binary = normalize_lighting_and_contrast(paper)
    clean_bin, line_meta = remove_ruled_and_margin_lines(binary)
    assert clean_bin.shape == binary.shape
    assert "ruled_lines_detected" in line_meta


def test_text_line_segmentation():
    raw = _synthetic_doc(n_lines=3)
    prep = preprocess_handwriting_document(raw)
    assert prep["line_count"] >= 1
    assert len(prep["line_crops"]) == prep["line_count"]
    for crop in prep["line_crops"]:
        assert "image" in crop
        assert crop["height"] > 0
        assert crop["width"] > 0
        assert crop["y2"] > crop["y1"]
