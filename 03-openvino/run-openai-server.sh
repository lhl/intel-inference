#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=03-openvino/common.sh
source "${SCRIPT_DIR}/common.sh"
# shellcheck source=03-openvino/model-presets.sh
source "${SCRIPT_DIR}/model-presets.sh"

ENV_NAME="intel-inf-openvino-genai"
MODEL_ALIAS=""
MODEL_DIR=""
DEVICE="GPU"
HOST="127.0.0.1"
PORT=8010
SERVED_MODEL_NAME=""

usage() {
    cat <<'EOF'
Usage:
  ./03-openvino/run-openai-server.sh [options]

Options:
  --env-name NAME        OpenVINO GenAI env name (default: intel-inf-openvino-genai)
  --model-alias NAME     Exported model alias under 03-openvino/models/
  --model-dir PATH       Explicit exported model directory
  --device NAME          Device to use (default: GPU)
  --host HOST            Bind host (default: 127.0.0.1)
  --port PORT            Bind port (default: 8010)
  --served-model-name N  Name to report from /v1/models and request payloads
  -h, --help             Show this help text
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-name)
            ENV_NAME="$2"
            shift 2
            ;;
        --model-alias)
            MODEL_ALIAS="$2"
            shift 2
            ;;
        --model-dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --served-model-name)
            SERVED_MODEL_NAME="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
done

if [[ -z "$MODEL_DIR" ]]; then
    [[ -n "$MODEL_ALIAS" ]] || die "either --model-alias or --model-dir is required"
    MODEL_DIR="${MODELS_DIR}/${MODEL_ALIAS}"
fi
[[ -d "$MODEL_DIR" ]] || die "model directory not found: $MODEL_DIR"

if [[ -z "$SERVED_MODEL_NAME" ]]; then
    if [[ -n "$MODEL_ALIAS" ]]; then
        SERVED_MODEL_NAME="$(resolve_served_model_name "$MODEL_ALIAS")"
    else
        SERVED_MODEL_NAME="$(basename "$MODEL_DIR")"
    fi
fi

if [[ "$DEVICE" == "NPU" || "$DEVICE" == NPU* ]]; then
    maybe_source_npu_workaround
fi

run_in_env "$ENV_NAME" python "${SCRIPT_DIR}/openvino-openai-server.py" \
    --model-dir "$MODEL_DIR" \
    --device "$DEVICE" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name "$SERVED_MODEL_NAME"
