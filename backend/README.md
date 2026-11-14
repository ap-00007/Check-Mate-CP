# Check Mate Backend — Handwritten Authorship Verification & Stylometric Analysis

FastAPI backend for **Check Mate**, a handwriting verification and educational analysis system combining computer vision physical graphology and NLP stylometry.

Accuracy on Full IAM Evaluation:
- **OpenCV Visual-Only Model**: **96.6%** Accuracy | **99.4%** ROC-AUC
- **NLP Stylometric-Only Model**: **70.0%** Accuracy | **75.6%** ROC-AUC
- **Multi-Modal Calibrated Fusion**: **85.0%** Accuracy | **94.4%** ROC-AUC

---

## 🏗️ Architecture & Logical Pipeline

```text
Image Upload
    │
    ▼
Image Validation (Resolution, Format, Integrity)
    │
    ▼
Image Preprocessing (Paper Boundary Crop, Illumination Correction, Ruled-Line Removal, Scale Normalization)
    │
    ▼
Text Line Segmentation (Horizontal Projection Profile, Top-to-Bottom Line Images)
    │
    ▼
Handwritten OCR (Microsoft TrOCR line-based transcription or transcript override)
    │
    ▼
NLP Stylometric Analysis (5 Modules: Lexical, Syntactic, Structural, Punctuation, Character Patterns)
    │
    ▼
Visual Handwriting Analysis (8 OpenCV Features: Ink Density, Stroke Width, Slant Angle, Aspect, Spacing)
    │
    ▼
Pairwise Verification (Trained RandomForest Pipelines + Subword TF-IDF Cosine + Calibrated Fusion)
    │
    ▼
Evidence Comparison & Explainability (Feature-Level Breakdown + Curated Educational Registry)
    │
    ▼
Structured API Response & Ephemeral Analysis Caching
```

---

## 🚀 Setup & Running

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

- Interactive API Docs (Swagger UI): `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

---

## 📡 API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check, active hardware device (MPS/CUDA/CPU), loaded model config |
| `POST` | `/api/v1/verify` | Submit two handwriting samples for multi-modal verification |
| `GET` | `/api/v1/analysis/{analysis_id}` | Retrieve past analysis results, stage timings, and metadata |
| `GET` | `/api/v1/analysis/{analysis_id}/stage/{stage}` | Inspect execution details for a specific pipeline stage |
| `GET` | `/api/v1/analysis/{analysis_id}/explain/{feature}` | Get mathematical definition, formula, and interpretation for a feature |
| `GET` | `/api/v1/learning/modules` | List all curated educational learning modules |
| `GET` | `/api/v1/learning/modules/{module_id}` | Retrieve deep explanation for an educational concept |

### `POST /api/v1/verify` Parameters
Accepts `multipart/form-data`:
- `reference_image` *(required)*: PNG, JPEG, or WEBP (≤ 12 MB)
- `comparison_image` *(required)*: PNG, JPEG, or WEBP (≤ 12 MB)
- `reference_text` *(optional)*: Transcript override for testing or ground-truth verification
- `comparison_text` *(optional)*: Transcript override for testing or ground-truth verification

### Sample Verification Response
```json
{
  "status": "compatible",
  "message": "Handwriting styles appear compatible (confidence 87.4%, threshold 61.1%, margin +26.3%).",
  "visual_score": 0.8542,
  "nlp_score": 0.8803,
  "combined_score": 0.8738,
  "calibrated_threshold": 0.6111,
  "decision_margin": 0.2627,
  "reference_text": "...",
  "comparison_text": "...",
  "feature_comparison": [
    {
      "feature": "Slant Angle (avg)",
      "reference": -12.4,
      "comparison": -11.8,
      "absolute_difference": 0.6,
      "relative_difference": 0.0248,
      "interpretation": "Highly consistent handwriting stroke geometry.",
      "reliability": "Physical OpenCV optical measurement."
    }
  ],
  "visual_feature_scores": {
    "stroke_similarity": 0.9412,
    "slant_similarity": 0.8920,
    "letter_proportion_similarity": 0.9145
  },
  "nlp_module_scores": {
    "lexical": 0.8421,
    "syntactic": 0.8910,
    "structural": 0.8654,
    "punctuation": 0.9120,
    "character_nlp": 0.8735
  },
  "analysis_id": "c1f7b822-4a9f-4311-bf32-61783de63481",
  "pipeline_stages": {
    "preprocessing": { "status": "completed", "execution_time_ms": 42.1 },
    "line_segmentation": { "status": "completed", "execution_time_ms": 14.5 },
    "ocr": { "status": "completed", "execution_time_ms": 110.2 },
    "stylometry": { "status": "completed", "execution_time_ms": 8.3 },
    "pairwise_verification": { "status": "completed", "execution_time_ms": 5.1 },
    "explainability": { "status": "completed", "execution_time_ms": 2.2 }
  },
  "warnings": [
    "This analysis estimates stylistic compatibility and is not forensic proof of identity."
  ]
}
```

---

## 🔬 Scientific Evaluation & Training

Check Mate strictly adheres to writer-disjoint splitting to prevent data leakage:
- Writers appearing in the test or validation sets **never** appear in training.
- Positive pairs represent documents from the same author ($y=1$).
- Negative pairs represent documents from different authors ($y=0$).

### Reproduce Model Training & Benchmarking

```bash
cd backend
python scripts/train_eval.py --manifest ../notebooks/data/manifest.csv --pairs-per-class 1000
```

### Biometric Evaluation Metrics
- **Accuracy**: Overall fraction of correct pair classifications.
- **Precision, Recall, F1-Score**: Harmonic balance for same-writer detection.
- **ROC-AUC**: Threshold-independent discriminatory power.
- **False Acceptance Rate (FAR)**: $\text{FP} / (\text{FP} + \text{TN})$ — rate of impostors wrongly accepted.
- **False Rejection Rate (FRR)**: $\text{FN} / (\text{FN} + \text{TP})$ — rate of genuine authors rejected.
- **Equal Error Rate (EER)**: The operating threshold where $\text{FAR} = \text{FRR}$.

---

## 🧪 Running Tests

```bash
cd backend
source .venv/bin/activate
python -m pytest tests/ -v
```

The test suite covers:
- Image decoding and corruption handling
- Document paper cropping & ruled line removal
- Text-line segmentation
- 5-module NLP stylometric feature extraction (116 features)
- Insufficient text sample handling (< 20 words)
- Low-contrast / blank sample handling
- Model loading, pairwise prediction, and threshold calibration
- Full API integration & stage inspection

---

## 📋 Implementation Status

| Feature | Classification | Description |
|---|---|---|
| **OpenCV Preprocessing & Segmentation** | **Implemented & Tested** | Paper crop, background division, notebook line removal, horizontal projection line segmentation. |
| **Visual Feature Extraction** | **Implemented & Tested** | 8 physical features (stroke width, slant, aspect, height uniformity, ink density). |
| **TrOCR Handwritten Line OCR** | **Implemented & Tested** | Lazy-loaded singleton, MPS/CUDA/CPU device support, fallback mode, line-based processing. |
| **NLP Stylometric Fingerprinting** | **Implemented & Tested** | 5 modules: Lexical, Syntactic, Structural, Punctuation, Character subwords (116 features). |
| **Pairwise Verification & Fusion** | **Implemented & Tested** | Writer-disjoint trained models, calibrated decision threshold, decision margin reporting. |
| **Educational Explainability Registry** | **Implemented & Tested** | Curated modules for 11 core NLP/CV concepts and per-feature evidence formulas/definitions. |
| **API Endpoints & Analysis Store** | **Implemented & Tested** | `/verify`, `/analysis/{id}`, `/stage/{stage}`, `/explain/{feature}`, `/learning/modules`. |
| **Deep ResNet18 Embeddings** | **Experimental** | Tested in research notebook; evaluated as optional because OpenCV features delivered higher accuracy (96.6% vs 85.0%). |
| **Unsupervised Character Bank** | **Partially Implemented** | Component glyph isolation active; flagged as unavailable when isolated character samples are insufficient. |
