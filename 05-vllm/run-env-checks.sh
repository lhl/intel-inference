#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=05-vllm/common.sh
source "${SCRIPT_DIR}/common.sh"

BACKEND="all"
PREFIX="$(timestamp_utc)-env-checks"
XPU_ENV_NAME="intel-inf-vllm-xpu"
OPENVINO_ENV_NAME="intel-inf-vllm-openvino"

usage() {
    cat <<'EOF'
Usage:
  ./05-vllm/run-env-checks.sh [options]

Options:
  --backend xpu|openvino|all
                        Which runtime to validate (default: all)
  --xpu-env-name NAME   XPU env name (default: intel-inf-vllm-xpu)
  --openvino-env-name NAME
                        OpenVINO env name (default: intel-inf-vllm-openvino)
  --prefix NAME         Output filename prefix inside 05-vllm/results/
  -h, --help            Show this help text
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        --xpu-env-name)
            XPU_ENV_NAME="$2"
            shift 2
            ;;
        --openvino-env-name)
            OPENVINO_ENV_NAME="$2"
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

case "$BACKEND" in
    xpu|openvino|all)
        ;;
    *)
        die "unsupported backend selector: $BACKEND"
        ;;
esac

if [[ "$BACKEND" == "xpu" || "$BACKEND" == "all" ]]; then
    log "collecting vLLM XPU env details"
    run_in_env "$XPU_ENV_NAME" env VLLM_TARGET_DEVICE=xpu \
        python "${SCRIPT_DIR}/env-check.py" \
            --backend xpu \
            --output-json "${RESULTS_DIR}/${PREFIX}-xpu.json" \
        | tee "${RESULTS_DIR}/${PREFIX}-xpu.log"
fi

if [[ "$BACKEND" == "openvino" || "$BACKEND" == "all" ]]; then
    log "collecting vLLM OpenVINO env details"
    run_in_env "$OPENVINO_ENV_NAME" env VLLM_TARGET_DEVICE=empty VLLM_OPENVINO_DEVICE=GPU \
        python "${SCRIPT_DIR}/env-check.py" \
            --backend openvino \
            --output-json "${RESULTS_DIR}/${PREFIX}-openvino.json" \
        | tee "${RESULTS_DIR}/${PREFIX}-openvino.log"
fi
