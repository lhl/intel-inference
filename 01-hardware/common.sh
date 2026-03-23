#!/usr/bin/env bash

# Common helpers for 01-hardware scripts.

HW_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${HW_DIR}/results"

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
