#!/usr/bin/env bash

# Common helpers for 02-operators scripts.

OPS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${OPS_DIR}/results"

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

print_cmd() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
}

run_in_env() {
    local env_name="$1"
    shift
    local run_bin
    run_bin="$(env_run_bin)"
    "$run_bin" run -n "$env_name" "$@"
}
