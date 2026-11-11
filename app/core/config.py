"""
Application configuration loaded from environment variables.
"""
import os
from pathlib import Path
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- Application ---
    APP_NAME: str = "Audio Transcription Pipeline"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # --- File upload ---
    UPLOAD_DIR: str = "uploads"
    MAX_FILE_SIZE_MB: int = 500  # 500 MB hard cap
    ALLOWED_EXTENSIONS: List[str] = ["wav", "mp3", "m4a", "flac", "ogg"]

    # --- Whisper model ---
    WHISPER_MODEL_SIZE: str = "small"
    WHISPER_DEVICE: str = "cpu"          # "cpu" or "cuda"
    WHISPER_COMPUTE_TYPE: str = "int8"   # "int8", "float16", "float32"
    WHISPER_VAD_FILTER: bool = True

    # --- Chunk-based processing ---
    CHUNK_DURATION_SECONDS: int = 360    # 6 minutes per chunk
    CHUNK_OVERLAP_SECONDS: int = 5       # 5-second overlap at boundaries

    # --- Database ---
    DATABASE_URL: str = "sqlite:///./transcription.db"

    # --- Celery / Redis ---
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"
    WORKER_CONCURRENCY: int = 2

    # --- Retry policy ---
    MAX_RETRY_ATTEMPTS: int = 3
    RETRY_BACKOFF_BASE_SECONDS: int = 60  # exponential: 60, 120, 240 s

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }

    @property
    def upload_path(self) -> Path:
        path = Path(self.UPLOAD_DIR)
        path.mkdir(parents=True, exist_ok=True)
        return path


settings = Settings()
