# Media Transcriber

FastAPI service for transcribing media with `faster-whisper` and extracting text from images with Tesseract OCR.

It supports two input modes:

- URL-based media ingestion via JSON
- Direct file uploads via multipart form data

The service accepts audio, video, and image inputs:

- Audio and video are converted to 16 kHz mono WAV and transcribed with Whisper
- Images are processed with OCR

## Features

- URL-based transcription: `POST /transcribe`
- File upload transcription: `POST /transcribe/upload`
- OCR for image inputs
- Basic Auth protection for all endpoints, including `/docs`
- Configurable Whisper warm-up on startup
- Configurable video duration cap
- Hard file-size cap for both downloads and uploads

## API Overview

### `POST /transcribe`

Accepts JSON:

```json
{
  "url": "https://example.com/sample.mp3",
  "whisper_model": "medium",
  "mode": "auto"
}
```

Fields:

- `url`: required, publicly reachable media URL
- `whisper_model`: optional, default `medium`
- `mode`: optional, one of `auto`, `audio`, `image`

Behavior:

- `mode=auto`: images go to OCR, everything else goes to Whisper
- `mode=image`: forces OCR
- `mode=audio`: forces audio/video conversion plus Whisper

### `POST /transcribe/upload`

Accepts `multipart/form-data`:

- `file`: required upload
- `whisper_model`: optional, default `medium`
- `mode`: optional, one of `auto`, `audio`, `image`

### `GET /health`

Returns:

```json
{"ok": true}
```

### `GET /docs`

Swagger UI for the API. Protected by Basic Auth.

## Response Shape

Typical response:

```json
{
  "source": "sample.mp3",
  "url": null,
  "filename": "sample.mp3",
  "language": "en",
  "whisper": {
    "text": "Transcribed text...",
    "meta": {
      "language_probability": 0.98,
      "segments": []
    }
  },
  "ocr": null,
  "timings_ms": {
    "upload_ms": 123.4,
    "convert_ms": 456.7,
    "whisper_ms": 890.1,
    "total_ms": 1470.2,
    "unaccounted_ms": 0.0
  }
}
```

Notes:

- URL requests return `url`
- Upload requests return `filename`
- `source` is a generic identifier for either form of input
- OCR responses populate `ocr` and leave `whisper` as `null`

## Limits

### File size

The hard size limit is controlled by `DOWNLOAD_MAX_BYTES`.

- Default: `262144000` bytes
- Roughly `250 MiB`
- Applies to both URL downloads and file uploads

### Video duration

The hard video duration limit is controlled by `MAX_VIDEO_DURATION_SEC`.

- Default: `3600`
- Equivalent to 60 minutes
- Enforced for video inputs only
- Audio-only files are not duration-limited by code

### Download timeout

The remote download timeout is controlled by `DOWNLOAD_TIMEOUT_SEC`.

- Default: `1800`
- Equivalent to 30 minutes
- This is transfer time, not media duration

## Authentication

All routes are protected with HTTP Basic Auth.

Set:

- `BASIC_USER`
- `BASIC_PASS`

If either is missing, the service returns a server configuration error.

## Supported Whisper Models

Allowed models are controlled by `WHISPER_ALLOWED_MODELS`.

Current default:

```env
WHISPER_ALLOWED_MODELS=large-v1,medium,small
```

The request will be rejected if `whisper_model` is not in that allowlist.

## Environment Variables

Common settings:

```env
WHISPER_DEVICE=
WHISPER_COMPUTE_TYPE=
MAX_VIDEO_DURATION_SEC=3600

BASIC_USER=
BASIC_PASS=
```

Other supported runtime variables:

- `LOG_LEVEL`: logging level, default `INFO`
- `OCR_MIN_CHARS`: minimum OCR output length, default `3`
- `WHISPER_WARM`: warm a Whisper model on startup, default `1`
- `WHISPER_WARM_MODEL`: startup warm-up model, default `medium`
- `WHISPER_ALLOWED_MODELS`: allowed request-time models, default `large-v1,medium,small`
- `DOWNLOAD_MAX_BYTES`: max upload/download size, default `262144000`
- `DOWNLOAD_TIMEOUT_SEC`: remote fetch timeout, default `1800`
- `DISABLE_DOCS`: set to `1` to disable `/docs`

Note:

- `.env.sample` currently includes `API_KEY`, but the application code does not use API key authentication. Authentication is Basic Auth only.

## Run Locally

### Requirements

- Python 3.12
- `ffmpeg` and `ffprobe`
- Tesseract OCR installed on the host

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.sample .env
```

Edit `.env` and set at least:

```env
BASIC_USER=admin
BASIC_PASS=change-me
```

### Start the server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Run With Docker Compose

```bash
docker compose up --build
```

The compose file:

- builds the local image
- mounts the Hugging Face cache directory
- sets CPU and memory limits

Note:

- The current `docker-compose.yml` does not publish a host port. Add a `ports:` mapping if you want to call the service directly from your machine, for example `8000:8000`.

If you use Docker, make sure your environment includes `BASIC_USER` and `BASIC_PASS`.

OCR note for Docker:

- The Python dependency `pytesseract` is installed, but the current Dockerfile does not install the Tesseract system binary. Whisper-based transcription works in the container as built; OCR support in Docker requires adding Tesseract to the image.

## Example Requests

### URL-based audio or video

```bash
curl -u "$BASIC_USER:$BASIC_PASS" \
  -X POST "http://localhost:8000/transcribe" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/media.mp4",
    "whisper_model": "medium",
    "mode": "auto"
  }'
```

### Upload a local file

```bash
curl -u "$BASIC_USER:$BASIC_PASS" \
  -X POST "http://localhost:8000/transcribe/upload" \
  -F "file=@/path/to/media.mp3" \
  -F "whisper_model=medium" \
  -F "mode=auto"
```

### OCR for an image upload

```bash
curl -u "$BASIC_USER:$BASIC_PASS" \
  -X POST "http://localhost:8000/transcribe/upload" \
  -F "file=@/path/to/image.png" \
  -F "mode=image"
```

## Processing Flow

### URL requests

1. Download the remote file to a temporary location
2. Detect whether the input should go to OCR or Whisper
3. Reject video files longer than the configured maximum
4. Convert audio/video to WAV
5. Run Whisper or OCR
6. Delete temporary files

### Upload requests

1. Stream the upload to a temporary file
2. Detect whether the input should go to OCR or Whisper
3. Reject video files longer than the configured maximum
4. Convert audio/video to WAV
5. Run Whisper or OCR
6. Delete temporary files

## Operational Notes

- Whisper models are cached in-process after first load
- Startup warm-up can reduce first-request latency
- Temporary input and WAV files are deleted after each request
- Short OCR or Whisper outputs are treated as failed extraction/transcription

## Health Check

The container health check uses:

```bash
curl -fsS http://localhost:8000/health
```

## Current Stack

- FastAPI
- faster-whisper
- ffmpeg / ffprobe
- Tesseract OCR
- Pillow
- aiohttp
