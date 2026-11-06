"""Settings from the environment (.env supported). Empty values fall back to defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _get(name: str, default: str) -> str:
    v = os.getenv(name, "").strip()
    return v if v else default


@dataclass(frozen=True)
class Settings:
    ollama_url: str = _get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
    ollama_model: str = _get("OLLAMA_MODEL", "qwen2.5:3b")
    ollama_timeout: float = float(_get("OLLAMA_TIMEOUT", "120"))
    disable_llm: bool = _get("DISABLE_LLM", "0") in {"1", "true", "yes"}
    spacy_model: str = _get("SPACY_MODEL", "")
    bill_limit: float = float(_get("BILL_LIMIT", "50000"))
    min_ocr_chars: int = int(_get("MIN_OCR_CHARS", "40"))
    min_ocr_confidence: float = float(_get("MIN_OCR_CONFIDENCE", "60"))
    max_upload_mb: float = float(_get("MAX_UPLOAD_MB", "10"))
    max_pages: int = int(_get("MAX_PAGES", "5"))
    otel_console: bool = _get("OTEL_CONSOLE", "0") in {"1", "true", "yes"}


def get_settings() -> Settings:
    # Re-read on each call so tests can monkeypatch the environment.
    load_dotenv()
    return Settings(
        ollama_url=_get("OLLAMA_URL", "http://localhost:11434").rstrip("/"),
        ollama_model=_get("OLLAMA_MODEL", "qwen2.5:3b"),
        ollama_timeout=float(_get("OLLAMA_TIMEOUT", "120")),
        disable_llm=_get("DISABLE_LLM", "0") in {"1", "true", "yes"},
        spacy_model=_get("SPACY_MODEL", ""),
        bill_limit=float(_get("BILL_LIMIT", "50000")),
        min_ocr_chars=int(_get("MIN_OCR_CHARS", "40")),
        min_ocr_confidence=float(_get("MIN_OCR_CONFIDENCE", "60")),
        max_upload_mb=float(_get("MAX_UPLOAD_MB", "10")),
        max_pages=int(_get("MAX_PAGES", "5")),
        otel_console=_get("OTEL_CONSOLE", "0") in {"1", "true", "yes"},
    )
