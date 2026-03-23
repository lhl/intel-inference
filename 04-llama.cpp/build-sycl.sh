#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=04-llama.cpp/common.sh
source "${SCRIPT_DIR}/common.sh"

CLEAN=0
PREFIX="$(timestamp_utc)-sycl-build"
ONEAPI_ENV="${REPO_ROOT}/00-setup/oneapi-env.sh"

usage() {
    cat <<'EOF'
Usage:
  ./04-llama.cpp/build-sycl.sh [options]

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
[[ -f "$ONEAPI_ENV" ]] || die "missing oneAPI helper: $ONEAPI_ENV"
ensure_results_dir
ensure_checkout "$SYCL_CHECKOUT"

LOG_PATH="${RESULTS_DIR}/${PREFIX}.log"
exec > >(tee "$LOG_PATH") 2>&1

log "checkout=${SYCL_CHECKOUT}"
log "build_dir=${SYCL_BUILD_DIR}"
log "log_path=${LOG_PATH}"

# shellcheck source=00-setup/oneapi-env.sh
set +u
source "$ONEAPI_ENV"
set -u

need_cmd icx
need_cmd icpx

if [[ "$CLEAN" -eq 1 ]]; then
    rm -rf "$SYCL_BUILD_DIR"
fi

cmake -S "$SYCL_CHECKOUT" \
    -B "$SYCL_BUILD_DIR" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_SYCL=ON \
    -DCMAKE_C_COMPILER=icx \
    -DCMAKE_CXX_COMPILER=icpx

cmake --build "$SYCL_BUILD_DIR" --parallel

"${SYCL_BUILD_DIR}/bin/llama-cli" --help | sed -n '1,40p'

log "result=PASS"
