#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=05-vllm/common.sh
source "${SCRIPT_DIR}/common.sh"
# shellcheck source=05-vllm/model-presets.sh
source "${SCRIPT_DIR}/model-presets.sh"

BACKEND="xpu"
DEVICE="GPU"
HOST="127.0.0.1"
PORT=8020
REPEATS=1
MAX_TOKENS=96
MAX_MODEL_LEN=2048
PROMPT_FILE="${SCRIPT_DIR}/../benchmarks/prompts/small-llm-chat.jsonl"
MODELS=()
GPU_MEMORY_UTILIZATION=0.15
KV_CACHE_MEMORY_BYTES=""
BLOCK_SIZE=""
ATTENTION_BACKEND=""
ENFORCE_EAGER=0
ZE_AFFINITY_MASK=""
KV_CACHE_PRECISION="i8"
KV_CACHE_SPACE="4"
PREFIX="$(timestamp_utc)-openai-bench"

usage() {
    cat <<'EOF'
Usage:
  ./05-vllm/run-openai-bench.sh [options]

Options:
  --backend xpu|openvino   Runtime to benchmark (default: xpu)
  --device CPU|GPU         OpenVINO target device (default: GPU)
  --host HOST              Bind host for the temporary server (default: 127.0.0.1)
  --port PORT              Starting port for the first model (default: 8020)
  --repeats N              Measured repeats per prompt (default: 1)
  --max-tokens N           max_tokens for chat/completions (default: 96)
  --max-model-len N        Max model length for the runtime (default: 2048)
  --gpu-memory-utilization F
                           Runtime GPU memory utilization target (default: 0.15)
  --kv-cache-memory-bytes N  XPU-only explicit KV cache budget in bytes
  --block-size N             XPU-only explicit vLLM block size
  --attention-backend NAME   XPU-only attention backend (e.g. TRITON_ATTN)
  --enforce-eager            XPU-only eager mode
  --ze-affinity-mask MASK    XPU-only ZE_AFFINITY_MASK override
  --kv-cache-precision N   OpenVINO KV cache precision (default: i8)
  --kv-cache-space GB      OpenVINO KV cache reservation in GB (default: 4)
  --prompt-file PATH       JSONL prompt file (default: benchmarks/prompts/small-llm-chat.jsonl)
  --models NAME...         Explicit model aliases; defaults to the small vLLM set
  --prefix NAME            Output filename prefix inside 05-vllm/results/
  -h, --help               Show this help text
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)
            BACKEND="$2"
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
        --repeats)
            REPEATS="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
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
        --kv-cache-precision)
            KV_CACHE_PRECISION="$2"
            shift 2
            ;;
        --kv-cache-space)
            KV_CACHE_SPACE="$2"
            shift 2
            ;;
        --prompt-file)
            PROMPT_FILE="$2"
            shift 2
            ;;
        --models)
            shift
            MODELS=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                MODELS+=("$1")
                shift
            done
            ;;
        --prefix)
            PREFIX="$2"
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

ensure_results_dir
[[ -f "$PROMPT_FILE" ]] || die "prompt file not found: $PROMPT_FILE"

case "$BACKEND" in
    xpu|openvino)
        ;;
    *)
        die "unsupported backend: $BACKEND"
        ;;
esac

if [[ "${#MODELS[@]}" -eq 0 ]]; then
    mapfile -t MODELS < <(list_default_vllm_models)
fi

current_port="$PORT"
for alias in "${MODELS[@]}"; do
    [[ "$(resolve_model_family "$alias")" == "llm" ]] || die "model alias is not an llm: $alias"

    xpu_extra_args=()
    if [[ -n "$KV_CACHE_MEMORY_BYTES" ]]; then
        xpu_extra_args+=(--kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES")
    fi
    if [[ -n "$BLOCK_SIZE" ]]; then
        xpu_extra_args+=(--block-size "$BLOCK_SIZE")
    fi
    if [[ -n "$ATTENTION_BACKEND" ]]; then
        xpu_extra_args+=(--attention-backend "$ATTENTION_BACKEND")
    fi
    if [[ "$ENFORCE_EAGER" -eq 1 ]]; then
        xpu_extra_args+=(--enforce-eager)
    fi
    if [[ -n "$ZE_AFFINITY_MASK" ]]; then
        xpu_extra_args+=(--ze-affinity-mask "$ZE_AFFINITY_MASK")
    fi

    server_log="${RESULTS_DIR}/${PREFIX}-${BACKEND}-${alias}-server.log"
    output_jsonl="${RESULTS_DIR}/${PREFIX}-${BACKEND}-${alias}.jsonl"
    summary_json="${RESULTS_DIR}/${PREFIX}-${BACKEND}-${alias}-summary.json"

    if [[ "$BACKEND" == "xpu" ]]; then
        log "starting temporary vLLM XPU server for ${alias} on ${HOST}:${current_port}"
        (
            "${SCRIPT_DIR}/run-xpu-serve.sh" \
                --model-alias "$alias" \
                --host "$HOST" \
                --port "$current_port" \
                --max-model-len "$MAX_MODEL_LEN" \
                --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
                "${xpu_extra_args[@]}"
        ) >"$server_log" 2>&1 &
    else
        log "starting temporary vLLM OpenVINO server for ${alias} on ${HOST}:${current_port}"
        (
            "${SCRIPT_DIR}/run-openvino-serve.sh" \
                --model-alias "$alias" \
                --device "$DEVICE" \
                --host "$HOST" \
                --port "$current_port" \
                --max-model-len "$MAX_MODEL_LEN" \
                --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
                --kv-cache-precision "$KV_CACHE_PRECISION" \
                --kv-cache-space "$KV_CACHE_SPACE"
        ) >"$server_log" 2>&1 &
    fi
    server_pid=$!

    cleanup() {
        if kill -0 "$server_pid" >/dev/null 2>&1; then
            kill "$server_pid" >/dev/null 2>&1 || true
            wait "$server_pid" 2>/dev/null || true
        fi
    }
    trap cleanup EXIT

    ready=0
    for _ in $(seq 1 480); do
        if python - "http://${HOST}:${current_port}/health" <<'PY'
import sys
import urllib.request

url = sys.argv[1]
try:
    with urllib.request.urlopen(url, timeout=1.0) as response:
        raise SystemExit(0 if 200 <= response.status < 500 else 1)
except Exception:
    raise SystemExit(1)
PY
        then
            ready=1
            break
        fi
        if ! kill -0 "$server_pid" >/dev/null 2>&1; then
            tail -n 80 "$server_log" >&2 || true
            die "server exited before becoming ready for ${alias}"
        fi
        sleep 0.5
    done

    if [[ "$ready" -ne 1 ]]; then
        tail -n 80 "$server_log" >&2 || true
        die "timed out waiting for ${alias} server readiness"
    fi

    python "${SCRIPT_DIR}/../benchmarks/openai_api_bench.py" \
        --base-url "http://${HOST}:${current_port}" \
        --model "$(resolve_served_model_name "$alias")" \
        --prompt-file "$PROMPT_FILE" \
        --output-jsonl "$output_jsonl" \
        --summary-json "$summary_json" \
        --repeats "$REPEATS" \
        --max-tokens "$MAX_TOKENS" \
        --temperature 0.0 \
        --stream

    cleanup
    trap - EXIT
    printf 'Wrote %s\n' "$server_log"
    printf 'Wrote %s\n' "$output_jsonl"
    printf 'Wrote %s\n' "$summary_json"
    current_port=$((current_port + 1))
done
