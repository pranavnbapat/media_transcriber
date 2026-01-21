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
        device = os.getenv("WHISPER_DEVICE", "auto").strip().lower()  # auto|cuda|cpu
        compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "").strip().lower()

        # Reasonable defaults:
        # - CPU: int8 is typically the best speed/size trade-off
        # - CUDA: float16 is typical
        if device == "auto":
            preferred = [("cuda", compute_type or "float16"), ("cpu", compute_type or "int8")]
        elif device == "cuda":
            preferred = [("cuda", compute_type or "float16")]
        else:
            preferred = [("cpu", compute_type or "int8")]

        last_err: Exception | None = None
        for dev, ctype in preferred:
            try:
                _WHISPER_CACHE[model_size] = WhisperModel(
                    model_size,
                    device=dev,
                    compute_type=ctype,
                )
                print(f"[Whisper] Loaded model={model_size} device={dev} compute_type={ctype}")
                last_err = None
                break
            except Exception as e:
                last_err = e
                print(f"[Whisper] Failed init model={model_size} device={dev} compute_type={ctype} err={e}")

        if last_err is not None:
            raise last_err

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
