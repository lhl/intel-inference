#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=03-openvino/common.sh
source "${SCRIPT_DIR}/common.sh"
# shellcheck source=03-openvino/model-presets.sh
source "${SCRIPT_DIR}/model-presets.sh"

ENV_NAME="intel-inf-openvino-genai"
DEVICE="GPU"
AUDIO_PATH="${ASSETS_DIR}/how_are_you_doing_today.wav"
MODELS=()
PREFIX="$(timestamp_utc)-whisper-smoke"

usage() {
    cat <<'EOF'
Usage:
  ./03-openvino/run-whisper-smoke.sh [options]

Options:
  --env-name NAME      OpenVINO GenAI env name (default: intel-inf-openvino-genai)
  --device NAME        Device to use (default: GPU)
  --audio-path PATH    WAV sample to transcribe (default: 03-openvino/assets/how_are_you_doing_today.wav)
  --models NAME...     Explicit ASR model aliases; defaults to whisper-large-v3-turbo and whisper-large-v3
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
        --audio-path)
            AUDIO_PATH="$2"
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
[[ -f "$AUDIO_PATH" ]] || die "audio sample not found: $AUDIO_PATH; run prepare-samples.sh first"

if [[ "${#MODELS[@]}" -eq 0 ]]; then
    mapfile -t MODELS < <(list_default_asr_models)
fi

if [[ "$DEVICE" == "NPU" || "$DEVICE" == NPU* ]]; then
    maybe_source_npu_workaround
fi

for alias in "${MODELS[@]}"; do
    [[ "$(resolve_model_family "$alias")" == "asr" ]] || die "model alias is not asr: $alias"
    model_dir="${MODELS_DIR}/${alias}"
    [[ -d "$model_dir" ]] || die "model directory not found: $model_dir; run export-models.sh first"
    output_json="${RESULTS_DIR}/${PREFIX}-${alias}.json"
    log "whisper smoke ${alias} on ${DEVICE}"
    run_in_env "$ENV_NAME" python "${SCRIPT_DIR}/openvino-whisper-smoke.py" \
        --model-dir "$model_dir" \
        --audio-path "$AUDIO_PATH" \
        --device "$DEVICE" \
        --output-json "$output_json"
    printf 'Wrote %s\n' "$output_json"
done
