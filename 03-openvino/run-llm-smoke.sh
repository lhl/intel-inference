#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=03-openvino/common.sh
source "${SCRIPT_DIR}/common.sh"
# shellcheck source=03-openvino/model-presets.sh
source "${SCRIPT_DIR}/model-presets.sh"

ENV_NAME="intel-inf-openvino-genai"
DEVICE="GPU"
PROMPT_FILE="${SCRIPT_DIR}/../benchmarks/prompts/small-llm-chat.jsonl"
MAX_TOKENS=96
MODELS=()
PREFIX="$(timestamp_utc)-llm-smoke"

usage() {
    cat <<'EOF'
Usage:
  ./03-openvino/run-llm-smoke.sh [options]

Options:
  --env-name NAME      OpenVINO GenAI env name (default: intel-inf-openvino-genai)
  --device NAME        Device to use (default: GPU)
  --prompt-file PATH   JSONL prompt file (default: benchmarks/prompts/small-llm-chat.jsonl)
  --max-tokens N       max_new_tokens for generation (default: 96)
  --models NAME...     Explicit model aliases; defaults to the standard small LLM set
  --prefix NAME        Output filename prefix inside 03-openvino/results/
  -h, --help           Show this help text
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-name)
            ENV_NAME="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --prompt-file)
            PROMPT_FILE="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
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

if [[ "$DEVICE" == "NPU" || "$DEVICE" == NPU* ]]; then
    maybe_source_npu_workaround
fi

for alias in "${MODELS[@]}"; do
    [[ "$(resolve_model_family "$alias")" == "llm" ]] || die "model alias is not an llm: $alias"
    model_dir="${MODELS_DIR}/${alias}"
    [[ -d "$model_dir" ]] || die "model directory not found: $model_dir; run export-models.sh first"
    output_json="${RESULTS_DIR}/${PREFIX}-${alias}.json"
    log "llm smoke ${alias} on ${DEVICE}"
    run_in_env "$ENV_NAME" python "${SCRIPT_DIR}/openvino-llm-smoke.py" \
        --model-dir "$model_dir" \
        --device "$DEVICE" \
        --prompt-file "$PROMPT_FILE" \
        --max-tokens "$MAX_TOKENS" \
        --output-json "$output_json"
    printf 'Wrote %s\n' "$output_json"
done
