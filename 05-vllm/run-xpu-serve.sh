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
KV_CACHE_MEMORY_BYTES=""
BLOCK_SIZE=""
ATTENTION_BACKEND=""
ENFORCE_EAGER=0
ZE_AFFINITY_MASK=""
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
  --gpu-memory-utilization F GPU memory utilization target (default: 0.15)
                           Use a low default on shared-memory iGPU systems.
  --kv-cache-memory-bytes N  Explicit KV cache budget in bytes
  --block-size N             Explicit vLLM block size
  --attention-backend NAME   Explicit attention backend (e.g. TRITON_ATTN)
  --enforce-eager            Force eager mode
  --ze-affinity-mask MASK    Set ZE_AFFINITY_MASK for device selection
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
        --kv-cache-memory-bytes)
            KV_CACHE_MEMORY_BYTES="$2"
            shift 2
            ;;
        --block-size)
            BLOCK_SIZE="$2"
            shift 2
            ;;
        --attention-backend)
            ATTENTION_BACKEND="$2"
            shift 2
            ;;
        --enforce-eager)
            ENFORCE_EAGER=1
            shift
            ;;
        --ze-affinity-mask)
            ZE_AFFINITY_MASK="$2"
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
if [[ -n "$KV_CACHE_MEMORY_BYTES" ]]; then
    extra_args+=(--kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES")
fi
if [[ -n "$BLOCK_SIZE" ]]; then
    extra_args+=(--block-size "$BLOCK_SIZE")
fi
if [[ -n "$ATTENTION_BACKEND" ]]; then
    extra_args+=(--attention-backend "$ATTENTION_BACKEND")
fi
if [[ "$ENFORCE_EAGER" -eq 1 ]]; then
    extra_args+=(--enforce-eager)
fi

env_args=(
    VLLM_TARGET_DEVICE=xpu
    VLLM_WORKER_MULTIPROC_METHOD=spawn
)
if [[ -n "$ZE_AFFINITY_MASK" ]]; then
    env_args+=(ZE_AFFINITY_MASK="$ZE_AFFINITY_MASK")
fi

run_in_env "$ENV_NAME" env \
    "${env_args[@]}" \
    vllm serve "$MODEL_DIR" \
        --host "$HOST" \
        --port "$PORT" \
        --served-model-name "$SERVED_MODEL_NAME" \
        --dtype "$DTYPE" \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        "${extra_args[@]}"
