# app/download.py

import aiohttp
import asyncio
import logging
import os
import tempfile

from pathlib import Path
from typing import Tuple

logger = logging.getLogger("media_transcriber")

class DownloadError(RuntimeError):
    pass

async def download_to_tempfile(url: str, max_bytes: int = 250 * 1024 * 1024) -> Tuple[Path, str]:
    max_bytes = int(os.getenv("DOWNLOAD_MAX_BYTES", str(max_bytes)))

    """
    Streams a URL to disk and enforces a hard size cap.
    Returns a Path to a temp file.
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".bin")
    tmp_path = Path(tmp.name)
    tmp.close()

    timeout_s = int(os.getenv("DOWNLOAD_TIMEOUT_SEC", "1800"))
    timeout = aiohttp.ClientTimeout(total=timeout_s)

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, allow_redirects=True, headers={"User-Agent": "media-transcriber/1.0"}) as resp:
                if resp.status != 200:
                    raise DownloadError(f"Failed to download (HTTP {resp.status})")

                content_type = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
                content_length = (resp.headers.get("Content-Length") or "").strip()

                logger.info(
                    "Download response",
                    extra={"url": url, "status": resp.status, "content_type": content_type,
                           "content_length": content_length},
                )

                total = 0
                with tmp_path.open("wb") as f:
                    async for chunk in resp.content.iter_chunked(1024 * 1024):
                        if not chunk:
                            continue
                        total += len(chunk)
                        if total > max_bytes:
                            raise DownloadError(f"File too large (>{max_bytes} bytes)")
                        f.write(chunk)

                # Log what we actually wrote (source of truth vs Content-Length)
                logger.info(
                    "Download complete",
                    extra={
                        "url": url,
                        "content_type": content_type,
                        "content_length": content_length,
                        "bytes_written": total,
                        "tmp_path": str(tmp_path),
                    },
                )

            return tmp_path, content_type

    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        raise


