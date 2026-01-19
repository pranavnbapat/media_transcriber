# app/download.py

import aiohttp
import tempfile

from pathlib import Path


class DownloadError(RuntimeError):
    pass

async def download_to_tempfile(url: str, max_bytes: int = 250 * 1024 * 1024) -> Path:
    """
    Streams a URL to disk and enforces a hard size cap.
    Returns a Path to a temp file.
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".bin")
    tmp_path = Path(tmp.name)
    tmp.close()

    timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes total download time
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url, allow_redirects=True) as resp:
            if resp.status != 200:
                raise DownloadError(f"Failed to download (HTTP {resp.status})")

            total = 0
            with tmp_path.open("wb") as f:
                async for chunk in resp.content.iter_chunked(1024 * 1024):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > max_bytes:
                        raise DownloadError(f"File too large (>{max_bytes} bytes)")
                    f.write(chunk)

    return tmp_path
