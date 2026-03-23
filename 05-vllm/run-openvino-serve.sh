#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=05-vllm/common.sh
source "${SCRIPT_DIR}/common.sh"
# shellcheck source=05-vllm/model-presets.sh
source "${SCRIPT_DIR}/model-presets.sh"

ENV_NAME="intel-inf-vllm-openvino"
MODEL_ALIAS=""
MODEL_DIR=""
DEVICE="GPU"
HOST="127.0.0.1"
PORT=8030
SERVED_MODEL_NAME=""
DTYPE="bfloat16"
MAX_MODEL_LEN=2048
GPU_MEMORY_UTILIZATION=0.15
ENABLE_TRUST_REMOTE_CODE=""
KV_CACHE_PRECISION="i8"
KV_CACHE_SPACE="4"
USE_V1=1

usage() {
    cat <<'EOF'
Usage:
  ./05-vllm/run-openvino-serve.sh [options]

Options:
  --env-name NAME            vLLM OpenVINO env name (default: intel-inf-vllm-openvino)
  --model-alias NAME         Model alias from 03-openvino/model-presets.sh
  --model-dir PATH           Explicit local HF snapshot path
  --device CPU|GPU           OpenVINO target device (default: GPU)
  --host HOST                Bind host (default: 127.0.0.1)
  --port PORT                Bind port (default: 8030)
  --served-model-name NAME   Name to report from /v1/models
  --dtype NAME               vLLM dtype (default: bfloat16)
  --max-model-len N          Max model length (default: 2048)
  --gpu-memory-utilization F GPU memory utilization target (default: 0.60)
                           Use a low default on shared-memory iGPU systems.
  --kv-cache-precision NAME  OpenVINO KV cache precision (default: i8)
  --kv-cache-space GB        OpenVINO KV cache reservation in GB (default: 4)
  --use-v1 0|1               Set VLLM_USE_V1 (default: 1)
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
        --kv-cache-precision)
            KV_CACHE_PRECISION="$2"
            shift 2
            ;;
        --kv-cache-space)
            KV_CACHE_SPACE="$2"
            shift 2
            ;;
        --use-v1)
            USE_V1="$2"
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
    VLLM_TARGET_DEVICE=empty \
    VLLM_OPENVINO_DEVICE="$DEVICE" \
    VLLM_OPENVINO_KV_CACHE_PRECISION="$KV_CACHE_PRECISION" \
    VLLM_OPENVINO_KVCACHE_SPACE="$KV_CACHE_SPACE" \
    VLLM_USE_V1="$USE_V1" \
    vllm serve "$MODEL_DIR" \
        --host "$HOST" \
        --port "$PORT" \
        --served-model-name "$SERVED_MODEL_NAME" \
        --dtype "$DTYPE" \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        "${extra_args[@]}"
