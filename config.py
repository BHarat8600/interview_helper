from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "AI Interview Backend"
    app_env: str = "production"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_whisper_model: str = "whisper-large-v3-turbo"
    groq_llm_model: str = "llama-3.1-8b-instant"
    groq_timeout_seconds: float = 8.0

    llm_system_prompt: str = (
        "You are a professional technical consultant in a live client meeting. "
        "Provide short, confident, business-ready answers. "
        "No long explanations. "
        "No filler words. "
        "Keep answers under 4 sentences."
    )

    cors_allow_origins: str = "*"
    cors_allow_methods: str = "GET,POST,OPTIONS"
    cors_allow_headers: str = "*"
    max_audio_size_mb: int = 15


settings = Settings()


def get_groq_api_key() -> str:
    """
    Returns a normalized Groq API key from settings:
    - strips surrounding whitespace
    - strips surrounding single/double quotes
    """
    raw = (settings.groq_api_key or "").strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {"'", '"'}:
        raw = raw[1:-1].strip()
    return raw


def is_groq_key_configured() -> bool:
    key = get_groq_api_key()
    if not key:
        return False
    if key.lower() in {"replace_with_real_key", "your_groq_api_key_here"}:
        return False
    return True
