# app/main.py

from __future__ import annotations

import logging
import os
import time
import uuid

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Security, Depends, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

import secrets

from .audio import to_wav_16k_mono, probe_media, AudioError, MediaProbeError
from .download import download_to_tempfile, upload_to_tempfile, DownloadError, UploadError
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
MAX_VIDEO_DURATION_SEC = float(os.getenv("MAX_VIDEO_DURATION_SEC", "3600"))

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

def _allowed_whisper_models() -> set[str]:
    return {
        s.strip()
        for s in os.getenv("WHISPER_ALLOWED_MODELS", "large-v1,medium,small").split(",")
        if s.strip()
    }


def _validate_whisper_model(whisper_model: str) -> None:
    allowed = _allowed_whisper_models()
    if whisper_model not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"whisper_model must be one of: {sorted(allowed)}",
        )


def _validate_video_duration(
    *,
    duration_seconds: float | None,
    is_video: bool,
    request_id: str,
) -> None:
    if is_video and duration_seconds and duration_seconds > MAX_VIDEO_DURATION_SEC:
        raise HTTPException(
            status_code=400,
            detail=(
                f"[stage=probe request_id={request_id}] "
                f"Video too long: maximum allowed duration is {int(MAX_VIDEO_DURATION_SEC // 60)} minutes."
            ),
        )


async def _preflight_remote_video(url: str) -> None:
    request_id = uuid.uuid4().hex[:12]
    try:
        media_info = await run_in_threadpool(probe_media, url)
    except MediaProbeError as e:
        logger.info(
            "Remote media preflight skipped",
            extra={"request_id": request_id, "url": url, "error": str(e)},
        )
        return
    except Exception:
        logger.exception(
            "Unexpected remote media preflight failure",
            extra={"request_id": request_id, "url": url},
        )
        return

    _validate_video_duration(
        duration_seconds=media_info.get("duration_seconds"),
        is_video=bool(media_info.get("has_video")),
        request_id=request_id,
    )


async def _transcribe_media(
    *,
    media_path: Path,
    content_type: str,
    source: str,
    whisper_model: str,
    mode: Literal["auto", "audio", "image"] | str = "auto",
    url: str | None = None,
    filename: str | None = None,
) -> TranscribeResponse:
    wav_path = None
    request_id = uuid.uuid4().hex[:12]  # short, grep-friendly

    t0 = time.perf_counter()
    timings_ms: dict[str, float] = {}

    try:
        try:
            media_bytes = media_path.stat().st_size if media_path else None
        except Exception:
            media_bytes = None

        logger.info(
            "Source ready",
            extra={
                "request_id": request_id,
                "source": source,
                "content_type": content_type,
                "media_bytes": media_bytes,
            },
        )

        mode = (mode or "auto").lower()
        is_image = content_type.startswith("image/")

        if mode == "image" or (mode == "auto" and is_image):
            # ---- OCR path ----
            t_ocr = time.perf_counter()
            try:
                o_text, o_meta = await run_in_threadpool(ocr_image, media_path)
            except Exception as e:
                logger.exception(
                    "OCR failed",
                    extra={"request_id": request_id, "source": source, "content_type": content_type,
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
                source=source,
                url=url,
                filename=filename,
                language=None,
                ocr=EngineResult(text=o_text or "", meta=o_meta),
                whisper=None,
                timings_ms=timings_ms,
            )

        try:
            media_info = await run_in_threadpool(probe_media, media_path)
        except MediaProbeError as e:
            logger.warning(
                "Media probe failed",
                extra={"request_id": request_id, "source": source, "error": str(e)},
            )
            raise HTTPException(
                status_code=400,
                detail=f"[stage=probe request_id={request_id}] {str(e)}",
            )
        except Exception:
            logger.exception(
                "Unexpected media probe failure",
                extra={"request_id": request_id, "source": source},
            )
            raise HTTPException(
                status_code=422,
                detail=f"[stage=probe request_id={request_id}] Media inspection failed.",
            )

        _validate_video_duration(
            duration_seconds=media_info.get("duration_seconds"),
            is_video=bool(content_type.startswith("video/") or media_info.get("has_video")),
            request_id=request_id,
        )

        # 2) Convert
        t_cv = time.perf_counter()
        try:
            wav_path = await run_in_threadpool(to_wav_16k_mono, media_path)
        except AudioError as e:
            logger.warning(
                "Audio conversion failed",
                extra={"request_id": request_id, "source": source, "error": str(e)},
            )
            raise HTTPException(status_code=400, detail=f"[stage=convert request_id={request_id}] {str(e)}")
        except Exception:
            logger.exception(
                "Unexpected audio conversion failure",
                extra={"request_id": request_id, "source": source},
            )
            raise HTTPException(status_code=422,
                                detail=f"[stage=convert request_id={request_id}] Audio conversion failed.")

        # wav_path = await run_in_threadpool(to_wav_16k_mono, media_path)
        timings_ms["convert_ms"] = (time.perf_counter() - t_cv) * 1000
        try:
            wav_bytes = wav_path.stat().st_size if wav_path else None
        except Exception:
            wav_bytes = None
        logger.info("Convert complete", extra={"request_id": request_id, "source": source,
                                               "wav_bytes": wav_bytes},)

        # 3) Whisper (timed, once)
        t_wh = time.perf_counter()
        try:
            w_text, w_meta = await run_in_threadpool(
                transcribe_whisper, wav_path, whisper_model, None
            )
        except Exception as e:
            logger.exception(
                "Whisper transcription failed",
                extra={"request_id": request_id, "source": source, "whisper_model": whisper_model,
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
                extra={"source": source, "detected_lang": detected_lang},
            )
            raise HTTPException(status_code=422, detail="Transcription failed: too little text produced.")

        timings_ms["total_ms"] = (time.perf_counter() - t0) * 1000

        timings_ms["unaccounted_ms"] = max(
            0.0,
            timings_ms["total_ms"] - (
                timings_ms.get("convert_ms", 0.0)
                + timings_ms.get("whisper_ms", 0.0)
            )
        )

        total_s = time.perf_counter() - t0
        logger.info("Transcription total time", extra={"request_id": request_id, "source": source,
                                                       "total_s": round(total_s, 3)},)


        return TranscribeResponse(
            source=source,
            url=url,
            filename=filename,
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


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(req: TranscribeRequest) -> TranscribeResponse:
    _validate_whisper_model(req.whisper_model)
    await _preflight_remote_video(str(req.url))

    t_dl = time.perf_counter()
    try:
        media_path, content_type = await download_to_tempfile(str(req.url))
    except DownloadError as e:
        logger.warning("DownloadError", extra={"url": str(req.url), "error": str(e)})
        raise HTTPException(status_code=400, detail=f"[stage=download] {str(e)}")
    except Exception:
        logger.exception("Unexpected download failure", extra={"url": str(req.url)})
        raise HTTPException(status_code=502, detail="[stage=download] Download failed.")
    download_ms = (time.perf_counter() - t_dl) * 1000

    resp = await _transcribe_media(
        media_path=media_path,
        content_type=content_type,
        source=str(req.url),
        url=str(req.url),
        whisper_model=req.whisper_model,
        mode=req.mode or "auto",
    )
    resp.timings_ms["download_ms"] = download_ms
    resp.timings_ms["total_ms"] = resp.timings_ms.get("total_ms", 0.0) + download_ms
    resp.timings_ms["unaccounted_ms"] = max(
        0.0,
        resp.timings_ms.get("total_ms", 0.0)
        - resp.timings_ms.get("download_ms", 0.0)
        - resp.timings_ms.get("convert_ms", 0.0)
        - resp.timings_ms.get("whisper_ms", 0.0)
        - resp.timings_ms.get("ocr_ms", 0.0),
    )
    return resp


@app.post("/transcribe/upload", response_model=TranscribeResponse)
async def transcribe_upload(
    file: UploadFile = File(...),
    whisper_model: str = Form(default="medium"),
    mode: Literal["auto", "audio", "image"] = Form(default="auto"),
) -> TranscribeResponse:
    _validate_whisper_model(whisper_model)

    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")

    t_up = time.perf_counter()
    try:
        media_path, content_type = await upload_to_tempfile(file)
    except UploadError as e:
        logger.warning("UploadError", extra={"uploaded_filename": file.filename, "error": str(e)})
        raise HTTPException(status_code=400, detail=f"[stage=upload] {str(e)}")
    except Exception:
        logger.exception("Unexpected upload failure", extra={"uploaded_filename": file.filename})
        raise HTTPException(status_code=502, detail="[stage=upload] Upload failed.")
    upload_ms = (time.perf_counter() - t_up) * 1000

    resp = await _transcribe_media(
        media_path=media_path,
        content_type=content_type,
        source=file.filename,
        filename=file.filename,
        whisper_model=whisper_model,
        mode=mode,
    )
    resp.timings_ms["upload_ms"] = upload_ms
    resp.timings_ms["total_ms"] = resp.timings_ms.get("total_ms", 0.0) + upload_ms
    resp.timings_ms["unaccounted_ms"] = max(
        0.0,
        resp.timings_ms.get("total_ms", 0.0)
        - resp.timings_ms.get("upload_ms", 0.0)
        - resp.timings_ms.get("convert_ms", 0.0)
        - resp.timings_ms.get("whisper_ms", 0.0)
        - resp.timings_ms.get("ocr_ms", 0.0),
    )
    return resp

@app.get("/health")
def health():
    return {"ok": True}
