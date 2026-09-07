# app/transcribe.py

from __future__ import annotations

import os

from pathlib import Path
from typing import Optional, Dict, Any, Tuple, TYPE_CHECKING

import requests

from .audio import to_mp3_for_upload

if TYPE_CHECKING:  # pragma: no cover - typing only
    from faster_whisper import WhisperModel


# ---- Hosted transcription (OpenAI-compatible /v1/audio/transcriptions) ------
#
# Set WHISPER_API_URL and WHISPER_API_KEY to transcribe through a hosted
# endpoint instead of loading a model into this process. Scaleway:
#
#   WHISPER_API_URL=https://api.scaleway.ai/<project-id>
#   WHISPER_API_MODEL=whisper-large-v3
#
# Both unset is the previous behaviour exactly: faster-whisper in-process. That
# fallback is deliberate, so an existing deployment keeps working untouched.
#
# Why hosted at all: whisper-large on CPU is impractically slow, and the
# Kubernetes nodes have no GPU. Running the model here would mean either a GPU
# node or transcription that never finishes.
WHISPER_API_URL = (os.getenv("WHISPER_API_URL") or "").strip().rstrip("/")
WHISPER_API_KEY = (os.getenv("WHISPER_API_KEY") or "").strip()
WHISPER_API_MODEL = (os.getenv("WHISPER_API_MODEL") or "whisper-large-v3").strip()

# A hosted endpoint caps upload size where a local model did not. 25 MB is the
# usual limit. A 16 kHz mono 16-bit WAV is ~1.9 MB per minute, so this is
# reached around 13 minutes of audio — well inside MAX_VIDEO_DURATION_SEC, which
# defaults to an hour. Checked before upload so the failure names the cause
# rather than surfacing as an opaque API error.
WHISPER_API_MAX_BYTES = int(os.getenv("WHISPER_API_MAX_BYTES", str(25 * 1024 * 1024)))
WHISPER_API_TIMEOUT = float(os.getenv("WHISPER_API_TIMEOUT", "600"))
# Mono MP3 bitrate for upload. 32 kbps is ample for speech recognition.
WHISPER_API_BITRATE = (os.getenv("WHISPER_API_BITRATE") or "32k").strip()


def remote_transcription_enabled() -> bool:
    """True when transcription should go to the hosted endpoint."""
    return bool(WHISPER_API_URL and WHISPER_API_KEY)


def _transcribe_remote(wav_path: Path, language: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    """Transcribe via an OpenAI-compatible /v1/audio/transcriptions endpoint.

    Returns the same (text, meta) shape as the local path, so callers and the
    HTTP response model are unchanged.
    """
    # Upload compressed, not raw PCM. The WAV is 32 KB/s, which would exceed the
    # endpoint's ~25 MB limit after about 13.6 minutes — shorter than this
    # service already accepts. Mono MP3 at 32 kbps is ~14 MB/hour instead.
    upload_path = to_mp3_for_upload(wav_path, bitrate=WHISPER_API_BITRATE)
    try:
        return _post_audio(upload_path, language)
    finally:
        try:
            upload_path.unlink()
        except OSError:
            pass


def _post_audio(upload_path: Path, language: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    size = upload_path.stat().st_size
    if size > WHISPER_API_MAX_BYTES:
        # A backstop, not the primary defence: at 32 kbps this needs roughly
        # three hours of audio, well past MAX_VIDEO_DURATION_SEC. If it ever
        # fires, the answer is chunk-and-stitch, not a lower bitrate.
        raise RuntimeError(
            f"compressed audio is {size / 1048576:.1f} MB, over the "
            f"{WHISPER_API_MAX_BYTES / 1048576:.0f} MB limit of {WHISPER_API_URL}. "
            "The media is too long to transcribe in one request."
        )

    url = f"{WHISPER_API_URL}/v1/audio/transcriptions"
    data = {
        "model": WHISPER_API_MODEL,
        # verbose_json is what carries segments and the detected language; plain
        # json returns only the text and would drop both from meta.
        "response_format": "verbose_json",
    }
    if language:
        data["language"] = language

    with upload_path.open("rb") as fh:
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {WHISPER_API_KEY}"},
            data=data,
            files={"file": (upload_path.name, fh, "audio/mpeg")},
            timeout=WHISPER_API_TIMEOUT,
        )

    if resp.status_code >= 400:
        raise RuntimeError(
            f"transcription endpoint returned HTTP {resp.status_code}: {(resp.text or '')[:300]}"
        )

    body = resp.json()
    segments = []
    for seg in body.get("segments") or []:
        segments.append({
            "start": seg.get("start"),
            "end": seg.get("end"),
            "text": seg.get("text"),
        })

    meta = {
        "detected_language": body.get("language"),
        # The OpenAI-compatible response carries no confidence for the detected
        # language. Kept as a key so the shape does not change between paths.
        "language_probability": None,
        "segments": segments,
        "engine": f"remote:{WHISPER_API_MODEL}",
    }
    return (body.get("text") or "").strip(), meta


# ---- Whisper model cache (process-wide) ----
_WHISPER_CACHE: Dict[str, "WhisperModel"] = {}


def get_whisper(model_size: str) -> "WhisperModel":
    """
    Cache Whisper models in memory so you don't reload for every request.

    faster-whisper is imported here rather than at module scope so this module
    can be imported without it. That matters for a remote-only deployment: the
    library pulls in ctranslate2, onnxruntime and the HuggingFace stack, which
    is a large dependency to ship for a model that is never loaded.
    """
    from faster_whisper import WhisperModel

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
    if remote_transcription_enabled():
        # model_size is the caller's requested local size ("medium", "large-v1").
        # The hosted endpoint serves one model, named by WHISPER_API_MODEL, so the
        # request is honoured in spirit rather than literally. Logged when they
        # differ, because silently ignoring a caller's choice is confusing later.
        if model_size and model_size not in (WHISPER_API_MODEL, ""):
            print(f"[Whisper] requested={model_size} served={WHISPER_API_MODEL} (hosted endpoint)")
        return _transcribe_remote(wav_path, language)

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
