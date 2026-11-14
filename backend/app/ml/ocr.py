"""
ocr.py — Line-based handwritten OCR using Microsoft TrOCR.

Model: microsoft/trocr-small-handwritten
Architecture:
- Text-line image crops passed in top-to-bottom reading order.
- Lazy-loaded singleton model and processor (loaded once, reused).
- Hardware acceleration: Apple Silicon (MPS), CUDA, with CPU fallback.
- True confidence calculation when beam/token scores available (no fabricated scores).
- Graceful degradation when torch/transformers or weights are unavailable.
"""
from __future__ import annotations

import logging
from typing import Any
import cv2
import numpy as np

from app.core.config import settings
from app.schemas.verification import DocumentOCRResult, OCRLineResult

logger = logging.getLogger(__name__)

# Module-level singletons
_trocr_processor = None
_trocr_model = None
_trocr_device = None
_trocr_load_error: str | None = None
_trocr_initialized = False


def detect_device() -> str:
    """Select best available hardware device."""
    user_choice = (settings.device or "auto").lower()
    if user_choice in {"mps", "cuda", "cpu"}:
        return user_choice

    try:
        import importlib
        torch_mod = importlib.import_module("torch")
        if torch_mod.backends.mps.is_available():
            return "mps"
        if torch_mod.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def get_trocr_engine() -> tuple[Any, Any, str | None, str | None]:
    """
    Lazy-loads TrOCR processor and model singleton.
    Returns (processor, model, device, error_message).
    """
    global _trocr_processor, _trocr_model, _trocr_device, _trocr_load_error, _trocr_initialized

    if _trocr_initialized:
        return _trocr_processor, _trocr_model, _trocr_device, _trocr_load_error

    _trocr_initialized = True
    try:
        import importlib
        torch_mod = importlib.import_module("torch")
        transformers_mod = importlib.import_module("transformers")
        TrOCRProcessor = getattr(transformers_mod, "TrOCRProcessor")
        VisionEncoderDecoderModel = getattr(transformers_mod, "VisionEncoderDecoderModel")

        device_str = detect_device()
        logger.info(f"Loading TrOCR model '{settings.trocr_model_name}' on device '{device_str}'...")

        processor = TrOCRProcessor.from_pretrained(settings.trocr_model_name)
        model = VisionEncoderDecoderModel.from_pretrained(settings.trocr_model_name)
        model.to(device_str)
        model.eval()

        _trocr_processor = processor
        _trocr_model = model
        _trocr_device = device_str
        _trocr_load_error = None
        logger.info(f"TrOCR successfully loaded on {device_str}.")
        return _trocr_processor, _trocr_model, _trocr_device, None

    except Exception as exc:
        msg = f"TrOCR model initialization unavailable ({type(exc).__name__}: {exc})."
        logger.warning(msg)
        _trocr_processor = None
        _trocr_model = None
        _trocr_device = None
        _trocr_load_error = msg
        return None, None, None, _trocr_load_error


def transcribe_single_line(
    line_image: np.ndarray,
    processor: Any,
    model: Any,
    device: str,
) -> tuple[str, float | None]:
    """
    Transcribes a single segmented line crop using TrOCR.
    Extracts true mean token confidence if available; returns None otherwise.
    """
    import importlib
    torch = importlib.import_module("torch")
    pil_image_mod = importlib.import_module("PIL.Image")
    Image = getattr(pil_image_mod, "Image", pil_image_mod)

    if line_image.ndim == 2:
        pil_img = Image.fromarray(line_image).convert("RGB")
    else:
        pil_img = Image.fromarray(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB))

    pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            max_new_tokens=128,
            return_dict_in_generate=True,
            output_scores=True,
        )

    text = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0].strip()

    # Calculate true confidence from output transition scores if available
    confidence: float | None = None
    if hasattr(outputs, "scores") and outputs.scores:
        try:
            # outputs.scores is a tuple of length (generated_tokens)
            # each tensor of shape (batch_size, vocab_size)
            probs = []
            for step_logits in outputs.scores:
                step_prob = torch.softmax(step_logits, dim=-1)
                max_prob = torch.max(step_prob, dim=-1).values.item()
                probs.append(max_prob)
            if probs:
                confidence = round(float(np.mean(probs)), 4)
        except Exception:
            confidence = None

    return text, confidence


def transcribe_document_lines(
    line_crops: list[dict[str, Any]],
    text_override: str | None = None,
) -> DocumentOCRResult:
    """
    Line-based OCR transcription for a preprocessed document.
    Preserves line ordering and aggregates lines into document transcription.
    Supports user text override for benchmarking and test isolation.
    """
    warnings: list[str] = []

    # Handle text override (e.g. ground-truth or user transcript)
    if text_override is not None:
        lines = [ln.strip() for ln in text_override.splitlines() if ln.strip()]
        line_results = [
            OCRLineResult(line_number=idx + 1, text=ln, confidence=None)
            for idx, ln in enumerate(lines)
        ]
        words = [w for w in text_override.split() if w.strip()]
        return DocumentOCRResult(
            status="completed",
            full_text=text_override.strip(),
            lines=line_results,
            line_count=len(line_results),
            word_count=len(words),
            confidence=None,
            warnings=["Using user-provided transcript override."],
        )

    processor, model, device, err = get_trocr_engine()
    if err or processor is None or model is None or device is None:
        return DocumentOCRResult(
            status="not_configured",
            full_text="",
            lines=[],
            line_count=len(line_crops),
            word_count=0,
            confidence=None,
            warnings=[
                f"OCR model offline: {err or 'Missing TrOCR runtime'}.",
                "Handwriting verification running in visual-only mode.",
            ],
        )

    if not line_crops:
        return DocumentOCRResult(
            status="partial",
            full_text="",
            lines=[],
            line_count=0,
            word_count=0,
            confidence=None,
            warnings=["No text lines detected in document."],
        )

    line_results: list[OCRLineResult] = []
    confidences: list[float] = []

    for item in line_crops:
        line_idx = item.get("line_index", len(line_results))
        img = item.get("image")
        if img is None or img.size == 0:
            continue

        try:
            txt, conf = transcribe_single_line(img, processor, model, device)
            line_results.append(
                OCRLineResult(line_number=line_idx + 1, text=txt, confidence=conf)
            )
            if conf is not None:
                confidences.append(conf)
        except Exception as line_exc:
            logger.warning(f"Failed to transcribe line {line_idx + 1}: {line_exc}")
            warnings.append(f"Line {line_idx + 1} transcription failed: {line_exc}")

    full_text = "\n".join(r.text for r in line_results if r.text).strip()
    words = [w for w in full_text.split() if w.strip()]
    doc_conf = round(float(np.mean(confidences)), 4) if confidences else None

    status = "completed" if full_text else "failed"

    return DocumentOCRResult(
        status=status,
        full_text=full_text,
        lines=line_results,
        line_count=len(line_results),
        word_count=len(words),
        confidence=doc_conf,
        warnings=warnings,
    )
