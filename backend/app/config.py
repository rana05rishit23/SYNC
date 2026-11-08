"""Runtime configuration from environment variables (see .env.example). No secrets in code."""
from __future__ import annotations
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    groq_api_key: str | None
    advisor_model: str      # tool-calling advisor
    fast_model: str         # cheap classification / language detection
    stt_model: str          # speech-to-text for the voice stretch goal
    groq_base_url: str = "https://api.groq.com/openai/v1"


def get_settings() -> Settings:
    return Settings(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        advisor_model=os.getenv("ADVISOR_MODEL", "openai/gpt-oss-120b"),
        fast_model=os.getenv("FAST_MODEL", "openai/gpt-oss-20b"),
        stt_model=os.getenv("STT_MODEL", "whisper-large-v3-turbo"),
    )
