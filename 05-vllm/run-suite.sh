#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=05-vllm/common.sh
source "${SCRIPT_DIR}/common.sh"

REPEATS=1
MAX_TOKENS=96
MAX_MODEL_LEN=2048
RUN_XPU=1
RUN_OPENVINO=1
OPENVINO_DEVICE="GPU"
PREFIX="$(timestamp_utc)"

usage() {
    cat <<'EOF'
Usage:
  ./05-vllm/run-suite.sh [options]

Options:
  --repeats N              Measured repeats per prompt (default: 1)
  --max-tokens N           max_tokens for chat/completions (default: 96)
  --max-model-len N        Max model length passed to the runtimes (default: 2048)
  --xpu-only               Run only the upstream vLLM XPU track
  --openvino-only          Run only the vllm-openvino track
  --openvino-device NAME   OpenVINO target device (default: GPU)
  --prefix NAME            Filename prefix in 05-vllm/results/ (default: UTC timestamp)
  -h, --help               Show this help text
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
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
        --xpu-only)
            RUN_XPU=1
            RUN_OPENVINO=0
            shift
            ;;
        --openvino-only)
            RUN_XPU=0
            RUN_OPENVINO=1
            shift
            ;;
        --openvino-device)
            OPENVINO_DEVICE="$2"
            shift 2
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

"${SCRIPT_DIR}/run-env-checks.sh" --prefix "${PREFIX}-env-checks"

if [[ "$RUN_XPU" -eq 1 ]]; then
    "${SCRIPT_DIR}/run-openai-bench.sh" \
        --backend xpu \
        --repeats "$REPEATS" \
        --max-tokens "$MAX_TOKENS" \
        --max-model-len "$MAX_MODEL_LEN" \
        --port 8020 \
        --prefix "${PREFIX}-xpu"
fi

if [[ "$RUN_OPENVINO" -eq 1 ]]; then
    "${SCRIPT_DIR}/run-openai-bench.sh" \
        --backend openvino \
        --device "$OPENVINO_DEVICE" \
        --repeats "$REPEATS" \
        --max-tokens "$MAX_TOKENS" \
        --max-model-len "$MAX_MODEL_LEN" \
        --port 8030 \
        --prefix "${PREFIX}-openvino"
fi
