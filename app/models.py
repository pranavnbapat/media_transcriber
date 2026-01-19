# app/models.py

from typing import Optional, Dict, Any, Literal

from pydantic import BaseModel, HttpUrl, Field


class TranscribeRequest(BaseModel):
    url: HttpUrl

    # Whisper model size for CPU: "tiny", "base", "small", "medium"
    whisper_model: str = Field(default="medium")
    mode: Optional[Literal["auto", "audio", "image"]] = "auto"

class EngineResult(BaseModel):
    text: str
    meta: Dict[str, Any] = Field(default_factory=dict)

class TranscribeResponse(BaseModel):
    url: str
    language: Optional[str]
    whisper: Optional[EngineResult] = None
    ocr: Optional[EngineResult] = None
    timings_ms: Dict[str, float] = Field(default_factory=dict)

