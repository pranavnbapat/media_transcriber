# app/audio.py

import subprocess
import tempfile

from pathlib import Path


class AudioError(RuntimeError):
    pass

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
