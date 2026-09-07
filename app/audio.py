# app/audio.py

import json
import subprocess
import tempfile

from pathlib import Path
from typing import Any


class AudioError(RuntimeError):
    pass


class MediaProbeError(RuntimeError):
    pass


def probe_media(input_path: str | Path) -> dict[str, Any]:
    """
    Returns basic ffprobe metadata needed for policy checks.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_entries", "format=duration:stream=codec_type",
        str(input_path),
    ]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise MediaProbeError(f"ffprobe failed: {proc.stderr[-2000:]}")

    try:
        payload = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise MediaProbeError("ffprobe returned invalid JSON") from exc

    streams = payload.get("streams") or []
    has_video = any((stream or {}).get("codec_type") == "video" for stream in streams)

    raw_duration = ((payload.get("format") or {}).get("duration"))
    try:
        duration_seconds = float(raw_duration) if raw_duration not in (None, "") else None
    except (TypeError, ValueError):
        duration_seconds = None

    return {
        "has_video": has_video,
        "duration_seconds": duration_seconds,
    }


def to_mp3_for_upload(input_path: Path, bitrate: str = "32k") -> Path:
    """Encode audio to mono MP3 for upload to a hosted transcription endpoint.

    A hosted endpoint caps the upload at around 25 MB. The 16 kHz mono PCM WAV
    used for local transcription is 32 KB/s, which reaches that in about 13.6
    minutes — shorter than MAX_VIDEO_DURATION_SEC allows, so long media would be
    rejected before transcription began.

    There is no reason to send uncompressed PCM over the wire. Mono MP3 at 32
    kbps is roughly 14 MB per hour, so an hour of audio fits comfortably, and
    speech recognition is unaffected at this bitrate. MP3 is chosen over Opus
    because every OpenAI-compatible audio endpoint accepts it.

    Returns a new temporary file; the caller is responsible for removing it.
    """
    out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    out_path = Path(out.name)
    out.close()

    cmd = [
        "ffmpeg",
        "-y",                    # overwrite output
        "-i", str(input_path),   # input file
        "-ac", "1",              # mono
        "-ar", "16000",          # 16k sample rate, as whisper expects
        "-b:a", bitrate,         # speech-suitable constant bitrate
        "-vn",                   # no video
        "-f", "mp3",
        str(out_path),
    ]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise AudioError(f"ffmpeg failed: {proc.stderr[-2000:]}")

    return out_path


def to_wav_16k_mono(input_path: Path) -> Path:
    """
    Converts any media file to PCM WAV 16kHz mono using ffmpeg.
    Good baseline format for transcription.
    """
    out = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    out_path = Path(out.name)
    out.close()

    cmd = [
        "ffmpeg",
        "-y",                    # overwrite output
        "-i", str(input_path),   # input file
        "-ac", "1",              # mono
        "-ar", "16000",          # 16k sample rate
        "-vn",                   # no video
        "-f", "wav",
        str(out_path),
    ]

    # Capture stderr for debugging.
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise AudioError(f"ffmpeg failed: {proc.stderr[-2000:]}")

    return out_path
