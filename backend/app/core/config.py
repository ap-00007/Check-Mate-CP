from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    trocr_model_name: str = "microsoft/trocr-small-handwritten"
    spacy_model_name: str = "en_core_web_sm"
    # Resolve artifact_dir relative to this config file so it works from any cwd
    artifact_dir: str = str(Path(__file__).parent.parent.parent / "artifacts")
    max_upload_mb: int = 12

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
