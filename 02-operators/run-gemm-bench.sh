#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=02-operators/common.sh
source "${SCRIPT_DIR}/common.sh"

ENV_NAME="intel-inf-torch-xpu"
PREFIX="$(timestamp_utc)-gemm-bench"
CASES=("1x4096x4096" "128x4096x4096" "512x4096x4096" "128x4096x11008" "128x11008x4096")
DTYPES=(float32 bfloat16 float16 int8)
VARIANTS=(eager compile)
REPEATS=20
WARMUPS=5
QUICK=0

usage() {
    cat <<'EOF'
Usage:
  ./02-operators/run-gemm-bench.sh [options]

Options:
  --env-name NAME       Conda/mamba env name (default: intel-inf-torch-xpu)
  --prefix NAME         Output file prefix inside 02-operators/results/
  --cases CASE...       GEMM cases in MxNxK format
  --dtypes NAME...      float32 | bfloat16 | float16 | int8
  --variants NAME...    eager | compile
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
        --cases)
            shift
            CASES=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                CASES+=("$1")
                shift
            done
            ;;
        --dtypes)
            shift
            DTYPES=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                DTYPES+=("$1")
                shift
            done
            ;;
        --variants)
            shift
            VARIANTS=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                VARIANTS+=("$1")
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
    CASES=("1x4096x4096" "128x4096x4096" "128x4096x11008" "128x11008x4096")
    REPEATS=10
    WARMUPS=3
fi

ensure_results_dir
LOGFILE="${RESULTS_DIR}/${PREFIX}.log"
JSON_OUT="${RESULTS_DIR}/${PREFIX}.json"

run_in_env "$ENV_NAME" python "${SCRIPT_DIR}/gemm-bench.py" \
    --cases "${CASES[@]}" \
    --dtypes "${DTYPES[@]}" \
    --variants "${VARIANTS[@]}" \
    --repeats "$REPEATS" \
    --warmups "$WARMUPS" \
    --json-out "$JSON_OUT" | tee "$LOGFILE"

printf '\nWrote %s\n' "$LOGFILE"
printf 'Wrote %s\n' "$JSON_OUT"
