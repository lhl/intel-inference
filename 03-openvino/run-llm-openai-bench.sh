#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=03-openvino/common.sh
source "${SCRIPT_DIR}/common.sh"
# shellcheck source=03-openvino/model-presets.sh
source "${SCRIPT_DIR}/model-presets.sh"

DEVICE="GPU"
HOST="127.0.0.1"
PORT=8010
REPEATS=1
MAX_TOKENS=96
PROMPT_FILE="${SCRIPT_DIR}/../benchmarks/prompts/small-llm-chat.jsonl"
MODELS=()
PREFIX="$(timestamp_utc)-llm-openai-bench"

usage() {
    cat <<'EOF'
Usage:
  ./03-openvino/run-llm-openai-bench.sh [options]

Options:
  --device NAME        Device to use (default: GPU)
  --host HOST          Bind host for the temporary server (default: 127.0.0.1)
  --port PORT          Starting port for the first model (default: 8010)
  --repeats N          Measured repeats per prompt (default: 1)
  --max-tokens N       max_tokens for chat/completions (default: 96)
  --prompt-file PATH   JSONL prompt file (default: benchmarks/prompts/small-llm-chat.jsonl)
  --models NAME...     Explicit model aliases; defaults to the standard small LLM set
  --prefix NAME        Output filename prefix inside 03-openvino/results/
  -h, --help           Show this help text
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
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

if [[ "${#MODELS[@]}" -eq 0 ]]; then
    mapfile -t MODELS < <(list_default_llm_models)
fi

current_port="$PORT"
for alias in "${MODELS[@]}"; do
    [[ "$(resolve_model_family "$alias")" == "llm" ]] || die "model alias is not an llm: $alias"
    model_dir="${MODELS_DIR}/${alias}"
    [[ -d "$model_dir" ]] || die "model directory not found: $model_dir; run export-models.sh first"

    server_log="${RESULTS_DIR}/${PREFIX}-${alias}-server.log"
    output_jsonl="${RESULTS_DIR}/${PREFIX}-${alias}.jsonl"
    summary_json="${RESULTS_DIR}/${PREFIX}-${alias}-summary.json"

    log "starting temporary OpenAI-compatible server for ${alias} on ${HOST}:${current_port}"
    (
        "${SCRIPT_DIR}/run-openai-server.sh" \
            --model-alias "$alias" \
            --device "$DEVICE" \
            --host "$HOST" \
            --port "$current_port"
    ) >"$server_log" 2>&1 &
    server_pid=$!

    cleanup() {
        if kill -0 "$server_pid" >/dev/null 2>&1; then
            kill "$server_pid" >/dev/null 2>&1 || true
            wait "$server_pid" 2>/dev/null || true
        fi
    }
    trap cleanup EXIT

    wait_for_http "http://${HOST}:${current_port}/health" 180

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
