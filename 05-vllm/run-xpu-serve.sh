#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=05-vllm/common.sh
source "${SCRIPT_DIR}/common.sh"
# shellcheck source=05-vllm/model-presets.sh
source "${SCRIPT_DIR}/model-presets.sh"

ENV_NAME="intel-inf-vllm-xpu"
MODEL_ALIAS=""
MODEL_DIR=""
HOST="127.0.0.1"
PORT=8020
SERVED_MODEL_NAME=""
DTYPE="bfloat16"
MAX_MODEL_LEN=2048
GPU_MEMORY_UTILIZATION=0.15
ENABLE_TRUST_REMOTE_CODE=""

usage() {
    cat <<'EOF'
Usage:
  ./05-vllm/run-xpu-serve.sh [options]

Options:
  --env-name NAME            vLLM XPU env name (default: intel-inf-vllm-xpu)
  --model-alias NAME         Model alias from 03-openvino/model-presets.sh
  --model-dir PATH           Explicit local HF snapshot path
  --host HOST                Bind host (default: 127.0.0.1)
  --port PORT                Bind port (default: 8020)
  --served-model-name NAME   Name to report from /v1/models
  --dtype NAME               vLLM dtype (default: bfloat16)
  --max-model-len N          Max model length (default: 2048)
  --gpu-memory-utilization F GPU memory utilization target (default: 0.60)
                           Use a low default on shared-memory iGPU systems.
  --trust-remote-code        Force enable trust_remote_code
  --no-trust-remote-code     Force disable trust_remote_code
  -h, --help                 Show this help text
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
        --dtype)
            DTYPE="$2"
            shift 2
            ;;
        --max-model-len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        --gpu-memory-utilization)
            GPU_MEMORY_UTILIZATION="$2"
            shift 2
            ;;
        --trust-remote-code)
            ENABLE_TRUST_REMOTE_CODE="yes"
            shift
            ;;
        --no-trust-remote-code)
            ENABLE_TRUST_REMOTE_CODE="no"
            shift
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
    MODEL_DIR="$(resolve_local_snapshot_path "$MODEL_ALIAS")"
fi
[[ -d "$MODEL_DIR" ]] || die "model directory not found: $MODEL_DIR"

if [[ -z "$SERVED_MODEL_NAME" ]]; then
    if [[ -n "$MODEL_ALIAS" ]]; then
        SERVED_MODEL_NAME="$(resolve_served_model_name "$MODEL_ALIAS")"
    else
        SERVED_MODEL_NAME="$(basename "$MODEL_DIR")"
    fi
fi

TRUST_REMOTE_CODE=0
if [[ -n "$MODEL_ALIAS" ]]; then
    if [[ "$(resolve_trust_remote_code "$MODEL_ALIAS")" == "true" ]]; then
        TRUST_REMOTE_CODE=1
    fi
fi
case "$ENABLE_TRUST_REMOTE_CODE" in
    yes)
        TRUST_REMOTE_CODE=1
        ;;
    no)
        TRUST_REMOTE_CODE=0
        ;;
    "")
        ;;
    *)
        die "unexpected trust_remote_code override: $ENABLE_TRUST_REMOTE_CODE"
        ;;
esac

extra_args=()
if [[ "$TRUST_REMOTE_CODE" -eq 1 ]]; then
    extra_args+=(--trust-remote-code)
fi

run_in_env "$ENV_NAME" env \
    VLLM_TARGET_DEVICE=xpu \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    vllm serve "$MODEL_DIR" \
        --host "$HOST" \
        --port "$PORT" \
        --served-model-name "$SERVED_MODEL_NAME" \
        --dtype "$DTYPE" \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        "${extra_args[@]}"
