#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=01-hardware/common.sh
source "${SCRIPT_DIR}/common.sh"

ENV_NAME="intel-inf-openvino"
PREFIX="$(timestamp_utc)-npu-microbench"
DEVICE="NPU"
MODEL_KINDS=(matmul)
SHAPES=(256 512 1024)
REPEATS=20
WARMUPS=5
QUICK=0

usage() {
    cat <<'EOF'
Usage:
  ./01-hardware/run-npu-microbench.sh [options]

Options:
  --env-name NAME        Conda/mamba env name (default: intel-inf-openvino)
  --device NAME          OpenVINO device (default: NPU)
  --model-kinds NAME...  matmul | mlp (default: matmul)
  --shapes N...          Square tensor shapes (default: 256 512 1024)
  --repeats N            Timed iterations per shape (default: 20)
  --warmups N            Warmup iterations per shape (default: 5)
  --prefix NAME          Output file prefix inside 01-hardware/results/
  --quick                Use a lighter sweep: shapes 256 512, repeats 10, warmups 3
  -h, --help             Show this help text
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
        --model-kinds)
            shift
            MODEL_KINDS=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                MODEL_KINDS+=("$1")
                shift
            done
            ;;
        --shapes)
            shift
            SHAPES=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                SHAPES+=("$1")
                shift
            done
            ;;
        --repeats)
            REPEATS="$2"
            shift 2
            ;;
        --warmups)
            WARMUPS="$2"
            shift 2
            ;;
        --prefix)
            PREFIX="$2"
            shift 2
            ;;
        --quick)
            QUICK=1
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

if [[ "$QUICK" -eq 1 ]]; then
    SHAPES=(256 512)
    REPEATS=10
    WARMUPS=3
fi

# On this Arch setup, OpenVINO NPU enumeration requires the Intel NPU userspace
# library directory to be exposed through LD_LIBRARY_PATH.
NPU_ENV_HELPER="${SCRIPT_DIR}/../00-setup/npu-env.sh"
if [[ "$DEVICE" == "NPU" && -f "$NPU_ENV_HELPER" && -f /usr/lib/x86_64-linux-gnu/libze_intel_npu.so ]]; then
    # shellcheck source=00-setup/npu-env.sh
    source "$NPU_ENV_HELPER"
fi

ensure_results_dir
LOGFILE="${RESULTS_DIR}/${PREFIX}.log"
JSON_OUT="${RESULTS_DIR}/${PREFIX}.json"

run_in_env "$ENV_NAME" python "${SCRIPT_DIR}/openvino-npu-microbench.py" \
    --device "$DEVICE" \
    --model-kinds "${MODEL_KINDS[@]}" \
    --shapes "${SHAPES[@]}" \
    --repeats "$REPEATS" \
    --warmups "$WARMUPS" \
    --json-out "$JSON_OUT" | tee "$LOGFILE"

printf '\nWrote %s\n' "$LOGFILE"
printf 'Wrote %s\n' "$JSON_OUT"
