from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router as api_router
from app.ml.ocr import detect_device
from app.ml.verifier import verification_service

app = FastAPI(
    title="Check Mate API",
    description="Handwriting author verification and educational analysis system",
    version="2.0.0",
)

# Allow all localhost origins and dev ports
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")


@app.get("/health")
def health():
    cfg = verification_service.config
    return {
        "status": "ok",
        "phase": "2 — multi-modal visual + stylometry",
        "device": detect_device(),
        "calibrated_threshold": cfg.get("calibrated_threshold", 0.6111),
        "alpha": cfg.get("alpha", 0.75),
        "beta": cfg.get("beta", 0.25),
    }
