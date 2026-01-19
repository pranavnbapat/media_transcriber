# app/transcribe.py

from __future__ import annotations

import os

from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from faster_whisper import WhisperModel


# ---- Whisper model cache (process-wide) ----
_WHISPER_CACHE: Dict[str, WhisperModel] = {}


def get_whisper(model_size: str) -> WhisperModel:
    """
    Cache Whisper models in memory so you don't reload for every request.
    """
    if model_size not in _WHISPER_CACHE:
        device = os.getenv("WHISPER_DEVICE", "cuda").strip().lower()  # "cuda" or "cpu"
        compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "float16").strip().lower()

        _WHISPER_CACHE[model_size] = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )

        # Make it obvious in logs what we actually initialised.
        # (Print is fine, or use your logger if you prefer.)
        print(f"[Whisper] Loaded model={model_size} device={device} compute_type={compute_type}")

    return _WHISPER_CACHE[model_size]


def transcribe_whisper(wav_path: Path, model_size: str, language: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    model = get_whisper(model_size)

    # Running with VAD filter often improves quality on messy audio.
    segments, info = model.transcribe(
        str(wav_path),
        language=None,
        vad_filter=True,
        beam_size=1,
        best_of=1,
    )

    texts = []
    seg_meta = []
    for seg in segments:
        texts.append(seg.text.strip())
        seg_meta.append({"start": seg.start, "end": seg.end, "text": seg.text})

    full_text = " ".join([t for t in texts if t])
    meta = {
        "detected_language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
        "segments": seg_meta,
    }
    return full_text, meta
