#!/usr/bin/env bash

# Common helpers for 03-openvino scripts.

OV_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${OV_DIR}/results"

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

ensure_results_dir() {
    mkdir -p "$RESULTS_DIR"
}

timestamp_utc() {
    date -u +%Y%m%dT%H%M%SZ
}

run_in_env() {
    local env_name="$1"
    shift
    local run_bin
    run_bin="$(env_run_bin)"
    "$run_bin" run -n "$env_name" "$@"
}

maybe_source_npu_workaround() {
    local helper="${OV_DIR}/../00-setup/npu-env.sh"
    if [[ -f "$helper" && -f /usr/lib/x86_64-linux-gnu/libze_intel_npu.so ]]; then
        # shellcheck source=00-setup/npu-env.sh
        source "$helper"
    fi
}
