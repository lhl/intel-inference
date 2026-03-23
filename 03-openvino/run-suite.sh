#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
QUICK=0

usage() {
    cat <<'EOF'
Usage:
  ./03-openvino/run-suite.sh [options]

Options:
  --quick     Run a lighter OpenVINO sweep suitable for first validation
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

"${SCRIPT_DIR}/run-env-checks.sh" --prefix "${TIMESTAMP}-env-checks"

if [[ "$QUICK" -eq 1 ]]; then
    "${SCRIPT_DIR}/run-device-bench.sh" --quick --prefix "${TIMESTAMP}-device-bench-quick"
else
    "${SCRIPT_DIR}/run-device-bench.sh" --prefix "${TIMESTAMP}-device-bench"
fi
