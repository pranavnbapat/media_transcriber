# app/main.py

from __future__ import annotations

import logging
import os
import time

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.concurrency import run_in_threadpool
from fastapi.security import APIKeyHeader

import secrets

from .audio import to_wav_16k_mono, AudioError
from .download import download_to_tempfile, DownloadError
from .models import TranscribeRequest, TranscribeResponse, EngineResult
from .transcribe import transcribe_whisper, get_whisper


try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

logger = logging.getLogger("media_transcriber")

WHISPER_MIN_CHARS = 20  # treat shorter output as failure

# ---------------- API key auth ----------------
API_KEY = os.getenv("API_KEY", "").strip()
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def require_api_key(api_key: str | None = Security(api_key_header)) -> None:
    """
    Simple header-based auth:
      - Client must send: X-API-Key: <secret>
      - Secret is read from env var API_KEY
    """
    # If API_KEY isn't set, fail closed (safer than accidentally running open)
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Server misconfigured: API_KEY is not set.")

    if not api_key or not secrets.compare_digest(api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Unauthorised")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Warm up Whisper model on startup (optional).
    Controlled by env vars:
      - WHISPER_WARM=1|0
      - WHISPER_WARM_MODEL=medium|large-v1|...
    """
    if os.getenv("WHISPER_WARM", "1") == "1":
        model_name = os.getenv("WHISPER_WARM_MODEL", "medium")
        logger.info("Warming Whisper model: %s", model_name)

        def _load():
            # Downloads (if needed) + initialises model
            get_whisper(model_name)

        try:
            await run_in_threadpool(_load)
            logger.info("Whisper warm-up complete: %s", model_name)
        except Exception as e:
            logger.exception("Whisper warm-up failed for %s: %s", model_name, e)

    yield

docs_url = None if os.getenv("DISABLE_DOCS", "0") == "1" else "/docs"
openapi_url = None if os.getenv("DISABLE_DOCS", "0") == "1" else "/openapi.json"
app = FastAPI(
    title="Media Transcriber (Whisper)",
    version="1.0",
    docs_url=docs_url,
    openapi_url=openapi_url,
    lifespan=lifespan,
    dependencies=[Depends(require_api_key)],
)


def _safe_unlink(p: Path) -> None:
    try:
        if p and p.exists():
            p.unlink()
    except Exception:
        # Intentionally ignore cleanup errors (disk full / perms etc.)
        pass

@app.post("/transcribe", response_model=TranscribeResponse, dependencies=[Depends(require_api_key)])
async def transcribe(req: TranscribeRequest) -> TranscribeResponse:
    media_path = None
    wav_path = None

    t0 = time.perf_counter()
    timings_ms: dict[str, float] = {}

    allowed = {
        s.strip()
        for s in os.getenv("WHISPER_ALLOWED_MODELS", "large-v1,medium,small").split(",")
        if s.strip()
    }
    if req.whisper_model not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"whisper_model must be one of: {sorted(allowed)}"
        )

    try:
        # 1) Download (timed, once)
        t_dl = time.perf_counter()
        try:
            media_path = await download_to_tempfile(str(req.url))
        except DownloadError as e:
            raise HTTPException(status_code=400, detail=str(e))
        timings_ms["download_ms"] = (time.perf_counter() - t_dl) * 1000

        # 2) Convert
        t_cv = time.perf_counter()
        try:
            wav_path = await run_in_threadpool(to_wav_16k_mono, media_path)
        except AudioError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # wav_path = await run_in_threadpool(to_wav_16k_mono, media_path)
        timings_ms["convert_ms"] = (time.perf_counter() - t_cv) * 1000

        # 3) Whisper (timed, once)
        t_wh = time.perf_counter()
        try:
            w_text, w_meta = await run_in_threadpool(
                transcribe_whisper, wav_path, req.whisper_model, None
            )
        except Exception as e:
            logger.exception(
                "Whisper transcription failed",
                extra={"url": str(req.url), "whisper_model": req.whisper_model, "error": str(e)},
            )
            raise HTTPException(status_code=422, detail="Transcription failed (Whisper error).")

        timings_ms["whisper_ms"] = (time.perf_counter() - t_wh) * 1000
        detected_lang = w_meta.get("detected_language")
        w_meta = dict(w_meta)  # defensive copy
        w_meta.pop("detected_language", None)

        whisper_res = EngineResult(text=w_text or "", meta=w_meta)

        if len(whisper_res.text.strip()) < WHISPER_MIN_CHARS:
            logger.warning(
                "Whisper produced too little text; treating as failure",
                extra={"url": str(req.url), "detected_lang": detected_lang},
            )
            raise HTTPException(status_code=422, detail="Transcription failed: too little text produced.")

        timings_ms["total_ms"] = (time.perf_counter() - t0) * 1000

        timings_ms["unaccounted_ms"] = max(
            0.0,
            timings_ms["total_ms"] - (
                    timings_ms.get("download_ms", 0.0)
                    + timings_ms.get("convert_ms", 0.0)
                    + timings_ms.get("whisper_ms", 0.0)
            )
        )

        total_s = time.perf_counter() - t0
        logger.info("Transcription total time", extra={"url": str(req.url), "total_s": round(total_s, 3)})

        return TranscribeResponse(
            url=str(req.url),
            language=detected_lang,
            whisper=whisper_res,
            timings_ms=timings_ms,
        )

    finally:
        # 4) Always discard files
        if wav_path:
            _safe_unlink(wav_path)
        if media_path:
            _safe_unlink(media_path)

@app.get("/health")
def health():
    return {"ok": True}
