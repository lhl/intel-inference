#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
QUICK=0
WITH_MODELS=0
DEVICE="GPU"

usage() {
    cat <<'EOF'
Usage:
  ./03-openvino/run-suite.sh [options]

Options:
  --quick          Run a lighter OpenVINO synthetic sweep suitable for first validation
  --with-models    Also run the real model suite after env and device checks
  --device NAME    Runtime device for model tests when --with-models is set (default: GPU)
  -h, --help       Show this help text
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)
            QUICK=1
            shift
            ;;
        --with-models)
            WITH_MODELS=1
            shift
            ;;
        --device)
            DEVICE="$2"
            shift 2
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

if [[ "$WITH_MODELS" -eq 1 ]]; then
    "${SCRIPT_DIR}/run-model-suite.sh" --device "$DEVICE"
fi
