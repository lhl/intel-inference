#!/usr/bin/env bash

# Common helpers for 04-llama.cpp scripts.

LLAMA4_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${LLAMA4_DIR}/.." && pwd)"
RESULTS_DIR="${LLAMA4_DIR}/results"

VULKAN_CHECKOUT="${LLAMA4_DIR}/llama.cpp-vulkan"
VULKAN_BUILD_DIR="${VULKAN_CHECKOUT}/build-intel"

SYCL_CHECKOUT="${LLAMA4_DIR}/llama.cpp-sycl"
SYCL_BUILD_DIR="${SYCL_CHECKOUT}/build-intel"

OPENVINO_CHECKOUT="${LLAMA4_DIR}/llama.cpp-openvino"
OPENVINO_BUILD_DIR="${OPENVINO_CHECKOUT}/build-intel"

log() {
    printf '[%s] %s\n' "$(basename "$0")" "$*"
}

die() {
    printf '[%s] Error: %s\n' "$(basename "$0")" "$*" >&2
    exit 1
}

have_cmd() {
    command -v "$1" >/dev/null 2>&1
}

need_cmd() {
    have_cmd "$1" || die "missing required command: $1"
}

env_run_bin() {
    if have_cmd mamba; then
        printf 'mamba'
    elif have_cmd conda; then
        printf 'conda'
    else
        die "neither mamba nor conda is available"
    fi
}

run_in_env() {
    local env_name="$1"
    shift
    local run_bin
    run_bin="$(env_run_bin)"
    "$run_bin" run -n "$env_name" "$@"
}

timestamp_utc() {
    date -u +%Y%m%dT%H%M%SZ
}

ensure_results_dir() {
    mkdir -p "$RESULTS_DIR"
}

ensure_checkout() {
    local path="$1"
    [[ -d "$path/.git" ]] || die "expected a git checkout at $path"
}
