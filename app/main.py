# app/main.py

from __future__ import annotations

import logging
import os
import time
import uuid

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.concurrency import run_in_threadpool
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

import secrets

from .audio import to_wav_16k_mono, AudioError
from .download import download_to_tempfile, DownloadError
from .models import TranscribeRequest, TranscribeResponse, EngineResult
from .ocr import ocr_image
from .transcribe import transcribe_whisper, get_whisper


try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

logger = logging.getLogger("media_transcriber")
logger.setLevel(LOG_LEVEL)

WHISPER_MIN_CHARS = 20  # treat shorter output as failure
IMAGE_MIN_CHARS = int(os.getenv("OCR_MIN_CHARS", "3"))

# ---------------- API key auth ----------------
BASIC_USER = os.getenv("BASIC_USER", "").strip()
BASIC_PASS = os.getenv("BASIC_PASS", "").strip()

basic_security = HTTPBasic(auto_error=False)

def require_basic_auth(
    basic: HTTPBasicCredentials | None = Security(basic_security),
) -> None:
    """
    Gate everything (including /docs) behind Basic Auth.
    """
    basic_user = os.getenv("BASIC_USER", "").strip()
    basic_pass = os.getenv("BASIC_PASS", "").strip()
    if not (basic_user and basic_pass):
        raise HTTPException(status_code=500, detail="Server misconfigured: BASIC_USER/BASIC_PASS not set.")

    if not basic:
        raise HTTPException(
            status_code=401,
            detail="Unauthorised",
            headers={"WWW-Authenticate": "Basic"},
        )

    user_ok = secrets.compare_digest(basic.username, basic_user)
    pass_ok = secrets.compare_digest(basic.password, basic_pass)
    if not (user_ok and pass_ok):
        raise HTTPException(
            status_code=401,
            detail="Unauthorised",
            headers={"WWW-Authenticate": "Basic"},
        )


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

DISABLE_DOCS = os.getenv("DISABLE_DOCS", "0") == "1"
docs_url = None
openapi_url = None

app = FastAPI(
    title="Media Transcriber (Whisper)",
    version="1.0",
    docs_url=docs_url,
    openapi_url=openapi_url,
    lifespan=lifespan,
    dependencies=[Depends(require_basic_auth)],
)

if not DISABLE_DOCS:

    @app.get("/openapi.json", dependencies=[Depends(require_basic_auth)], include_in_schema=False)
    def openapi_json():
        schema = get_openapi(
            title=app.title,
            version=app.version,
            routes=app.routes,
        )
        return JSONResponse(schema)

    @app.get("/docs", dependencies=[Depends(require_basic_auth)], include_in_schema=False)
    def swagger_docs():
        return get_swagger_ui_html(
            openapi_url="/openapi.json",
            title=f"{app.title} - Swagger UI",
        )


def _safe_unlink(p: Path) -> None:
    try:
        if p and p.exists():
            p.unlink()
    except Exception:
        # Intentionally ignore cleanup errors (disk full / perms etc.)
        pass

@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(req: TranscribeRequest) -> TranscribeResponse:
    media_path = None
    wav_path = None
    content_type = ""
    request_id = uuid.uuid4().hex[:12]  # short, grep-friendly

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
            media_path, content_type = await download_to_tempfile(str(req.url))
        except DownloadError as e:
            logger.warning(
                "DownloadError",
                extra={"request_id": request_id, "url": str(req.url), "error": str(e)},
            )
            raise HTTPException(status_code=400, detail=f"[stage=download request_id={request_id}] {str(e)}")
        except Exception:
            # This catches unexpected aiohttp/tempfile/disk issues and prints a full traceback
            logger.exception(
                "Unexpected download failure",
                extra={"request_id": request_id, "url": str(req.url)},
            )
            raise HTTPException(status_code=502, detail=f"[stage=download request_id={request_id}] Download failed.")

        timings_ms["download_ms"] = (time.perf_counter() - t_dl) * 1000

        try:
            media_bytes = media_path.stat().st_size if media_path else None
        except Exception:
            media_bytes = None

        logger.info(
            "Download complete",
            extra={
                "request_id": request_id,
                "url": str(req.url),
                "content_type": content_type,
                "media_bytes": media_bytes,
            },
        )

        mode = (req.mode or "auto").lower()
        is_image = content_type.startswith("image/")

        if mode == "image" or (mode == "auto" and is_image):
            # ---- OCR path ----
            t_ocr = time.perf_counter()
            try:
                o_text, o_meta = await run_in_threadpool(ocr_image, media_path)
            except Exception as e:
                logger.exception(
                    "OCR failed",
                    extra={"request_id": request_id, "url": str(req.url), "content_type": content_type,
                           "error": str(e),},
                )
                raise HTTPException(
                    status_code=422,
                    detail=f"[stage=ocr request_id={request_id}] Image-to-text failed (content_type={content_type or 'unknown'})."
                )

            timings_ms["ocr_ms"] = (time.perf_counter() - t_ocr) * 1000

            if len((o_text or "").strip()) < IMAGE_MIN_CHARS:
                raise HTTPException(status_code=422, detail="Image-to-text failed: too little text produced.")

            timings_ms["total_ms"] = (time.perf_counter() - t0) * 1000
            timings_ms["unaccounted_ms"] = max(
                0.0,
                timings_ms["total_ms"] - timings_ms.get("download_ms", 0.0) - timings_ms.get("ocr_ms", 0.0)
            )

            return TranscribeResponse(
                url=str(req.url),
                language=None,
                ocr=EngineResult(text=o_text or "", meta=o_meta),
                whisper=None,
                timings_ms=timings_ms,
            )

        # 2) Convert
        t_cv = time.perf_counter()
        try:
            wav_path = await run_in_threadpool(to_wav_16k_mono, media_path)
        except AudioError as e:
            logger.warning(
                "Audio conversion failed",
                extra={"request_id": request_id, "url": str(req.url), "error": str(e)},
            )
            raise HTTPException(status_code=400, detail=f"[stage=convert request_id={request_id}] {str(e)}")
        except Exception:
            logger.exception(
                "Unexpected audio conversion failure",
                extra={"request_id": request_id, "url": str(req.url)},
            )
            raise HTTPException(status_code=422,
                                detail=f"[stage=convert request_id={request_id}] Audio conversion failed.")

        # wav_path = await run_in_threadpool(to_wav_16k_mono, media_path)
        timings_ms["convert_ms"] = (time.perf_counter() - t_cv) * 1000
        try:
            wav_bytes = wav_path.stat().st_size if wav_path else None
        except Exception:
            wav_bytes = None
        logger.info("Convert complete", extra={"request_id": request_id, "url": str(req.url),
                                               "wav_bytes": wav_bytes},)

        # 3) Whisper (timed, once)
        t_wh = time.perf_counter()
        try:
            w_text, w_meta = await run_in_threadpool(
                transcribe_whisper, wav_path, req.whisper_model, None
            )
        except Exception as e:
            logger.exception(
                "Whisper transcription failed",
                extra={"request_id": request_id, "url": str(req.url), "whisper_model": req.whisper_model,
                       "error": str(e),},
            )
            raise HTTPException(
                status_code=422,
                detail=f"[stage=whisper request_id={request_id}] Transcription failed (Whisper error)."
            )

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
        logger.info("Transcription total time", extra={"request_id": request_id, "url": str(req.url),
                                                       "total_s": round(total_s, 3)},)


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
