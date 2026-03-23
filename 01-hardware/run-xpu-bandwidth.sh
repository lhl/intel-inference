#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=01-hardware/common.sh
source "${SCRIPT_DIR}/common.sh"

ENV_NAME="intel-inf-torch-xpu"
PREFIX="$(timestamp_utc)-xpu-bandwidth"
SIZES_MIB=(64 128 256 512)
REPEATS=20
WARMUPS=5
VECTOR_DTYPE="float16"

usage() {
    cat <<'EOF'
Usage:
  ./01-hardware/run-xpu-bandwidth.sh [options]

Options:
  --env-name NAME       Conda env name (default: intel-inf-torch-xpu)
  --prefix NAME         Output file prefix inside 01-hardware/results/
  --sizes-mib N...      Tensor sizes in MiB (default: 64 128 256 512)
  --repeats N           Timed iterations per case (default: 20)
  --warmups N           Warmup iterations per case (default: 5)
  --vector-dtype TYPE   float16 | bfloat16 | float32 (default: float16)
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
        --sizes-mib)
            shift
            SIZES_MIB=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                SIZES_MIB+=("$1")
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
        --vector-dtype)
            VECTOR_DTYPE="$2"
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
LOGFILE="${RESULTS_DIR}/${PREFIX}.log"
JSON_OUT="${RESULTS_DIR}/${PREFIX}.json"

conda run -n "$ENV_NAME" python "${SCRIPT_DIR}/xpu-bandwidth.py" \
    --sizes-mib "${SIZES_MIB[@]}" \
    --repeats "$REPEATS" \
    --warmups "$WARMUPS" \
    --vector-dtype "$VECTOR_DTYPE" \
    --json-out "$JSON_OUT" | tee "$LOGFILE"

printf '\nWrote %s\n' "$LOGFILE"
printf 'Wrote %s\n' "$JSON_OUT"
