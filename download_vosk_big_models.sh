#!/usr/bin/env bash
set -euo pipefail

BASE_URL="https://alphacephei.com/vosk/models"
DEST="data/vosk_models"

mkdir -p "$DEST"
cd "$DEST"

download_and_link () {
  ZIP="$1"
  LANG="$2"

  echo "Downloading $ZIP"
  curl -L -O "$BASE_URL/$ZIP"

  echo "Unzipping $ZIP"
  unzip -q "$ZIP"
  rm "$ZIP"

  DIR="${ZIP%.zip}"

  echo "Linking $LANG -> $DIR"
  ln -sfn "$DIR" "$LANG"
}

# ---- Big EU models ----
download_and_link vosk-model-en-us-0.22.zip en
download_and_link vosk-model-fr-0.22.zip fr
download_and_link vosk-model-de-0.21.zip de
download_and_link vosk-model-es-0.42.zip es
download_and_link vosk-model-it-0.22.zip it
download_and_link vosk-model-nl-spraakherkenning-0.6.zip nl
download_and_link vosk-model-el-gr-0.7.zip el

echo "All models downloaded and linked."
