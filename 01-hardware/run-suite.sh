#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
QUICK=0

usage() {
    cat <<'EOF'
Usage:
  ./01-hardware/run-suite.sh [options]

Options:
  --quick     Run a lighter sweep suitable for first validation
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

if [[ "$QUICK" -eq 1 ]]; then
    "${SCRIPT_DIR}/run-host-mbw.sh" --mbw-size-mib 1024 --mbw-runs 3 --sysbench-total 16G --prefix "${TIMESTAMP}-host-mbw-quick"
    "${SCRIPT_DIR}/run-xpu-bandwidth.sh" --sizes-mib 64 128 256 --repeats 10 --warmups 3 --prefix "${TIMESTAMP}-xpu-bandwidth-quick"
    "${SCRIPT_DIR}/collect-intel-gpu-top.sh" --seconds 20 --sample-ms 500 --prefix "${TIMESTAMP}-intel-gpu-top-quick" -- "${SCRIPT_DIR}/run-mamf-finder.sh" --quick --prefix "${TIMESTAMP}-mamf-finder-quick"
else
    "${SCRIPT_DIR}/run-host-mbw.sh" --prefix "${TIMESTAMP}-host-mbw"
    "${SCRIPT_DIR}/run-xpu-bandwidth.sh" --prefix "${TIMESTAMP}-xpu-bandwidth"
    "${SCRIPT_DIR}/collect-intel-gpu-top.sh" --seconds 45 --sample-ms 500 --prefix "${TIMESTAMP}-intel-gpu-top" -- "${SCRIPT_DIR}/run-mamf-finder.sh" --prefix "${TIMESTAMP}-mamf-finder"
fi
