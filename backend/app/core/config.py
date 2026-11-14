from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # OCR & Stylometry models
    trocr_model_name: str = "microsoft/trocr-small-handwritten"
    spacy_model_name: str = "en_core_web_sm"
    device: str = "auto"  # "auto", "mps", "cuda", or "cpu"

    # Artifact paths
    artifact_dir: str = str(Path(__file__).parent.parent.parent / "artifacts")
    max_upload_mb: int = 12

    # Verification & Fusion defaults
    alpha: float = 0.75  # NLP weight
    beta: float = 0.25   # Visual weight
    default_threshold: float = 0.6111
    min_words_for_verification: int = 20

    # Storage TTL for analysis retrieval (in minutes)
    analysis_cache_ttl_minutes: int = 60
    max_cached_analyses: int = 500

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
