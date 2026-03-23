#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=04-llama.cpp/common.sh
source "${SCRIPT_DIR}/common.sh"

CLEAN=0
PREFIX="$(timestamp_utc)-vulkan-build"

usage() {
    cat <<'EOF'
Usage:
  ./04-llama.cpp/build-vulkan.sh [options]

Options:
  --clean          Remove the existing build directory before configuring
  --prefix NAME    Prefix for the raw results log
  -h, --help       Show this help text
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean)
            CLEAN=1
            shift
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

need_cmd cmake
need_cmd ninja
need_cmd vulkaninfo
ensure_results_dir
ensure_checkout "$VULKAN_CHECKOUT"

LOG_PATH="${RESULTS_DIR}/${PREFIX}.log"
exec > >(tee "$LOG_PATH") 2>&1

log "checkout=${VULKAN_CHECKOUT}"
log "build_dir=${VULKAN_BUILD_DIR}"
log "log_path=${LOG_PATH}"

vulkaninfo --summary >/dev/null

if [[ "$CLEAN" -eq 1 ]]; then
    rm -rf "$VULKAN_BUILD_DIR"
fi

cmake -S "$VULKAN_CHECKOUT" \
    -B "$VULKAN_BUILD_DIR" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_VULKAN=ON

cmake --build "$VULKAN_BUILD_DIR" --parallel

"${VULKAN_BUILD_DIR}/bin/llama-bench" --list-devices
"${VULKAN_BUILD_DIR}/bin/llama-cli" --help | sed -n '1,40p'

log "result=PASS"
