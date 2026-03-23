#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=03-openvino/common.sh
source "${SCRIPT_DIR}/common.sh"

ENV_NAME="intel-inf-openvino"
PREFIX="$(timestamp_utc)-device-bench"
DEVICES=(CPU GPU NPU)
MODEL_KINDS=(matmul mlp)
SHAPES=(256 512 1024)
REPEATS=20
WARMUPS=5
QUICK=0

usage() {
    cat <<'EOF'
Usage:
  ./03-openvino/run-device-bench.sh [options]

Options:
  --env-name NAME       OpenVINO env name (default: intel-inf-openvino)
  --prefix NAME         Output prefix inside 03-openvino/results/
  --devices NAME...     Devices to probe (default: CPU GPU NPU)
  --model-kinds NAME... matmul | mlp
  --shapes N...         Square model shapes (default: 256 512 1024)
  --repeats N           Timed iterations per case (default: 20)
  --warmups N           Warmup iterations per case (default: 5)
  --quick               Use a lighter sweep
  -h, --help            Show this help text
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-name)
            ENV_NAME="$2"
            shift 2
            ;;
        --prefix)
            PREFIX="$2"
            shift 2
            ;;
        --devices)
            shift
            DEVICES=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                DEVICES+=("$1")
                shift
            done
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

maybe_source_npu_workaround
ensure_results_dir

LOGFILE="${RESULTS_DIR}/${PREFIX}.log"
JSON_OUT="${RESULTS_DIR}/${PREFIX}.json"

run_in_env "$ENV_NAME" python "${SCRIPT_DIR}/openvino-device-bench.py" \
    --devices "${DEVICES[@]}" \
    --model-kinds "${MODEL_KINDS[@]}" \
    --shapes "${SHAPES[@]}" \
    --repeats "$REPEATS" \
    --warmups "$WARMUPS" \
    --json-out "$JSON_OUT" | tee "$LOGFILE"

printf '\nWrote %s\n' "$LOGFILE"
printf 'Wrote %s\n' "$JSON_OUT"
