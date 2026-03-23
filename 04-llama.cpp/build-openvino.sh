#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=04-llama.cpp/common.sh
source "${SCRIPT_DIR}/common.sh"

CLEAN=0
PREFIX="$(timestamp_utc)-openvino-build"

usage() {
    cat <<'EOF'
Usage:
  ./04-llama.cpp/build-openvino.sh [options]

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

ensure_results_dir
ensure_checkout "$OPENVINO_CHECKOUT"

LOG_PATH="${RESULTS_DIR}/${PREFIX}.log"
exec > >(tee "$LOG_PATH") 2>&1

log "checkout=${OPENVINO_CHECKOUT}"
log "build_dir=${OPENVINO_BUILD_DIR}"
log "log_path=${LOG_PATH}"
log "env=intel-inf-openvino"

TMP_SCRIPT="$(mktemp "${RESULTS_DIR}/openvino-build.XXXXXX.sh")"
trap 'rm -f "$TMP_SCRIPT"' EXIT

cat >"$TMP_SCRIPT" <<'EOF'
set -euo pipefail

ov_base="$(python -c 'import openvino, pathlib; print(pathlib.Path(openvino.__file__).resolve().parent)')"
shim_root="${ov_base}/3rdparty/tbb"
shim_cmake="${shim_root}/lib/cmake/TBB"
shim_lib="${shim_root}/lib"

mkdir -p "$shim_cmake" "$shim_lib"

# llama.cpp currently expects an archive-style OpenVINO/TBB layout. The pip wheel
# has the CMake files, but not in the directory tree that ggml-openvino includes.
for f in TBBConfig.cmake TBBConfigVersion.cmake TBBTargets.cmake TBBTargets-none.cmake; do
    ln -sfn "/usr/lib/cmake/TBB/${f}" "${shim_cmake}/${f}"
done
rm -f "${shim_cmake}/TBBTargets-release.cmake"

ln -sfn /usr/include "${shim_root}/include"

for lib in \
    libtbb.so libtbb.so.12 libtbb.so.12.17 \
    libtbbmalloc.so libtbbmalloc.so.2 libtbbmalloc.so.2.17 \
    libtbbmalloc_proxy.so libtbbmalloc_proxy.so.2 libtbbmalloc_proxy.so.2.17 \
    libtbbbind_2_5.so libtbbbind_2_5.so.3 libtbbbind_2_5.so.3.17 \
    libirml.so libirml.so.1
do
    if [[ -e "/usr/lib/${lib}" ]]; then
        ln -sfn "/usr/lib/${lib}" "${shim_lib}/${lib}"
    fi
done

export OpenVINO_DIR="${ov_base}/cmake"

if [[ "${CLEAN}" == "1" ]]; then
    rm -rf "${OPENVINO_BUILD_DIR}"
fi

cmake -S "${OPENVINO_CHECKOUT}" \
    -B "${OPENVINO_BUILD_DIR}" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_OPENVINO=ON

cmake --build "${OPENVINO_BUILD_DIR}" --parallel

"${OPENVINO_BUILD_DIR}/bin/llama-cli" --help | sed -n '1,40p'

GGML_OPENVINO_DEVICE=GPU \
    "${OPENVINO_BUILD_DIR}/bin/llama-bench" --list-devices

if [[ -f "${REPO_ROOT}/00-setup/npu-env.sh" ]]; then
    # shellcheck source=00-setup/npu-env.sh
    source "${REPO_ROOT}/00-setup/npu-env.sh" >/dev/null 2>&1 || true
    GGML_OPENVINO_DEVICE=NPU \
        "${OPENVINO_BUILD_DIR}/bin/llama-bench" --list-devices
fi
EOF

if ! REPO_ROOT="$REPO_ROOT" \
    OPENVINO_CHECKOUT="$OPENVINO_CHECKOUT" \
    OPENVINO_BUILD_DIR="$OPENVINO_BUILD_DIR" \
    CLEAN="$CLEAN" \
    run_in_env intel-inf-openvino bash "$TMP_SCRIPT"
then
    die "OpenVINO build or sanity checks failed"
fi

log "result=PASS"
