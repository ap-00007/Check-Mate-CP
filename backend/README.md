# Check Mate Backend — Phase 1 (Visual Verification)

FastAPI backend for the Check Mate handwriting verification portal.
Phase 1 uses an **OpenCV-only visual classifier** (no TrOCR, no spaCy).
Accuracy: **96.6%** | ROC-AUC: **99.4%** on the full IAM dataset.

## Setup

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` to try the API interactively.

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/api/v1/verify` | Submit two handwriting images for verification |

### POST `/api/v1/verify`

Accepts `multipart/form-data` with two fields:
- `reference_image` — PNG / JPEG / WEBP (≤ 12 MB)
- `comparison_image` — PNG / JPEG / WEBP (≤ 12 MB)

Returns `VerificationResponse`:
```json
{
  "status": "compatible | not_compatible | insufficient_evidence",
  "message": "...",
  "visual_score": 0.9294,
  "combined_score": 0.9294,
  "nlp_score": null,
  "feature_comparison": [
    { "feature": "Stroke Width (avg)", "reference": 2.3, "comparison": 2.4, "absolute_difference": 0.1 }
  ],
  "visual_feature_scores": {
    "stroke_similarity": 0.94,
    "slant_similarity": 0.88,
    "letter_proportion_similarity": 0.91
  },
  "warnings": ["Phase 1: Visual handwriting analysis only."]
}
```

## Project Structure

```
backend/
├── app/
│   ├── api/routes.py           # POST /api/v1/verify
│   ├── core/config.py          # Settings (artifact_dir, max_upload_mb)
│   ├── ml/
│   │   ├── visual.py           # OpenCV feature extraction (8 features)
│   │   ├── verifier.py         # Artifact loader + verify_visual_pair()
│   │   └── pipeline.py         # analyze_image_pair() — API entry point
│   ├── schemas/verification.py # Pydantic response model
│   ├── services/verification_service.py
│   └── main.py                 # FastAPI app + CORS
├── artifacts/
│   ├── visual_pair_classifier_opencv.joblib  ← trained model
│   ├── visual_feature_columns_opencv.joblib
│   └── fusion_config.json
├── tests/
│   ├── test_health.py
│   └── test_verify.py          # 7 integration tests (all passing)
└── requirements.txt
```

## Run Tests

```bash
cd backend
source .venv/bin/activate
python -m pytest tests/ -v
```

## Suggested Integration Order (Phase 2)

1. Add `app/ml/ocr.py` — TrOCR line segmentation & transcription
2. Add `app/ml/stylometry.py` — spaCy 5-module NLP fingerprint
3. Update `pipeline.py` — fuse `nlp_score` with `visual_score`
4. Update `schemas/verification.py` — expose `nlp_module_scores`

