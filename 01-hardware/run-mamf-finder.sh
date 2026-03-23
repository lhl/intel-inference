#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=01-hardware/common.sh
source "${SCRIPT_DIR}/common.sh"

ENV_NAME="intel-inf-torch-xpu"
PREFIX="$(timestamp_utc)-mamf-finder-xpu"
SHAPES=(1024 2048 4096)
DTYPES=(float32 bfloat16 float16 int8)
REPEATS=20
WARMUPS=5
QUICK=0

usage() {
    cat <<'EOF'
Usage:
  ./01-hardware/run-mamf-finder.sh [options]

Options:
  --env-name NAME       Conda env name (default: intel-inf-torch-xpu)
  --prefix NAME         Output file prefix inside 01-hardware/results/
  --shapes N...         Square GEMM sizes (default: 1024 2048 4096)
  --dtypes NAME...      Dtypes (default: float32 bfloat16 float16 int8)
  --repeats N           Timed iterations per shape (default: 20)
  --warmups N           Warmup iterations per shape (default: 5)
  --quick               Use a lighter sweep: shapes 1024 2048, repeats 10, warmups 3
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
        --shapes)
            shift
            SHAPES=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                SHAPES+=("$1")
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
    SHAPES=(1024 2048)
    REPEATS=10
    WARMUPS=3
fi

ensure_results_dir
LOGFILE="${RESULTS_DIR}/${PREFIX}.log"
JSON_OUT="${RESULTS_DIR}/${PREFIX}.json"

conda run -n "$ENV_NAME" python "${SCRIPT_DIR}/mamf-finder-xpu.py" \
    --shapes "${SHAPES[@]}" \
    --dtypes "${DTYPES[@]}" \
    --repeats "$REPEATS" \
    --warmups "$WARMUPS" \
    --json-out "$JSON_OUT" | tee "$LOGFILE"

printf '\nWrote %s\n' "$LOGFILE"
printf 'Wrote %s\n' "$JSON_OUT"
