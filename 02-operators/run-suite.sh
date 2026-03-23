#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
QUICK=0

usage() {
    cat <<'EOF'
Usage:
  ./02-operators/run-suite.sh [options]

Options:
  --quick     Run a lighter operator sweep suitable for first validation
  -h, --help  Show this help text
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)
            QUICK=1
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

"${SCRIPT_DIR}/run-backend-check.sh" --prefix "${TIMESTAMP}-backend-check"

if [[ "$QUICK" -eq 1 ]]; then
    "${SCRIPT_DIR}/run-gemm-bench.sh" --quick --prefix "${TIMESTAMP}-gemm-quick"
    "${SCRIPT_DIR}/run-attention-bench.sh" --quick --prefix "${TIMESTAMP}-attention-quick"
else
    "${SCRIPT_DIR}/run-gemm-bench.sh" --prefix "${TIMESTAMP}-gemm"
    "${SCRIPT_DIR}/run-attention-bench.sh" --prefix "${TIMESTAMP}-attention"
fi
