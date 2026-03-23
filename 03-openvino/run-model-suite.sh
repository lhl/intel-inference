#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=03-openvino/model-presets.sh
source "${SCRIPT_DIR}/model-presets.sh"

DEVICE="GPU"
REPEATS=1
MAX_TOKENS=96
KEEP_HF_TOKEN=0
FORCE_EXPORT=0
LLM_MODELS=()
ASR_MODELS=()

usage() {
    cat <<'EOF'
Usage:
  ./03-openvino/run-model-suite.sh [options]

Options:
  --device NAME       Device to use for runtime tests (default: GPU)
  --repeats N         OpenAI benchmark repeats per prompt (default: 1)
  --max-tokens N      max_tokens for chat/completions (default: 96)
  --llm-models NAME...  Explicit LLM model aliases; defaults to the supported OpenVINO baseline set
  --asr-models NAME...  Explicit ASR model aliases; defaults to both Whisper models
  --keep-hf-token     Keep HF_TOKEN during export instead of unsetting it
  --force-export      Re-export models even if they already exist
  -h, --help          Show this help text
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)
            DEVICE="$2"
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
        --llm-models)
            shift
            LLM_MODELS=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                LLM_MODELS+=("$1")
                shift
            done
            ;;
        --asr-models)
            shift
            ASR_MODELS=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                ASR_MODELS+=("$1")
                shift
            done
            ;;
        --keep-hf-token)
            KEEP_HF_TOKEN=1
            shift
            ;;
        --force-export)
            FORCE_EXPORT=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'unknown argument: %s\n' "$1" >&2
            exit 2
            ;;
    esac
done

if [[ "${#LLM_MODELS[@]}" -eq 0 ]]; then
    mapfile -t LLM_MODELS < <(list_default_llm_models)
fi

if [[ "${#ASR_MODELS[@]}" -eq 0 ]]; then
    mapfile -t ASR_MODELS < <(list_default_asr_models)
fi

export_args=()
[[ "$KEEP_HF_TOKEN" -eq 1 ]] && export_args+=(--keep-hf-token)
[[ "$FORCE_EXPORT" -eq 1 ]] && export_args+=(--force)
all_models=("${LLM_MODELS[@]}" "${ASR_MODELS[@]}")

"${SCRIPT_DIR}/prepare-samples.sh"
"${SCRIPT_DIR}/export-models.sh" --models "${all_models[@]}" "${export_args[@]}"
"${SCRIPT_DIR}/run-llm-smoke.sh" --device "$DEVICE" --max-tokens "$MAX_TOKENS" --models "${LLM_MODELS[@]}"
"${SCRIPT_DIR}/run-llm-openai-bench.sh" --device "$DEVICE" --repeats "$REPEATS" --max-tokens "$MAX_TOKENS" --models "${LLM_MODELS[@]}"
"${SCRIPT_DIR}/run-whisper-smoke.sh" --device "$DEVICE" --models "${ASR_MODELS[@]}"
