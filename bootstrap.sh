cat > /workspace/media_transcriber/bootstrap.sh << 'SH'
#!/usr/bin/env bash
set -euo pipefail

apt-get update
apt-get install -y ffmpeg

cd /workspace/media_transcriber
source .venv/bin/activate

nohup uvicorn app.main:app --host 0.0.0.0 --port 8005 > uvicorn_8005.log 2>&1 &
disown
SH

chmod +x /workspace/media_transcriber/bootstrap.sh
