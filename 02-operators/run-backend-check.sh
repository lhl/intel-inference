#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=02-operators/common.sh
source "${SCRIPT_DIR}/common.sh"

ENV_NAME="intel-inf-torch-xpu"
PREFIX="$(timestamp_utc)-backend-check"

usage() {
    cat <<'EOF'
Usage:
  ./02-operators/run-backend-check.sh [options]

Options:
  --env-name NAME       Conda/mamba env name (default: intel-inf-torch-xpu)
  --prefix NAME         Output file prefix inside 02-operators/results/
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

run_in_env "$ENV_NAME" python "${SCRIPT_DIR}/backend-check.py" \
    --json-out "$JSON_OUT" | tee "$LOGFILE"

printf '\nWrote %s\n' "$LOGFILE"
printf 'Wrote %s\n' "$JSON_OUT"
