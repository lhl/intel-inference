#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=03-openvino/common.sh
source "${SCRIPT_DIR}/common.sh"

ensure_assets_dir

OUTPUT_PATH="${ASSETS_DIR}/how_are_you_doing_today.wav"
SAMPLE_URL="https://storage.openvinotoolkit.org/models_contrib/speech/2021.2/librispeech_s5/how_are_you_doing_today.wav"

if [[ -f "$OUTPUT_PATH" ]]; then
    log "sample already present at ${OUTPUT_PATH}"
    exit 0
fi

log "downloading ${SAMPLE_URL} -> ${OUTPUT_PATH}"
python - "$SAMPLE_URL" "$OUTPUT_PATH" <<'PY'
import sys
import urllib.request

url = sys.argv[1]
path = sys.argv[2]
urllib.request.urlretrieve(url, path)
PY

log "wrote ${OUTPUT_PATH}"
