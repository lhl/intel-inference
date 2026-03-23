#!/usr/bin/env bash

# Common helpers for 00-setup scripts.

COMMON_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${COMMON_DIR}/results"

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

conda_create_bin() {
    if have_cmd mamba; then
        printf 'mamba'
    elif have_cmd conda; then
        printf 'conda'
    else
        die "neither mamba nor conda is available"
    fi
}

conda_run_bin() {
    if have_cmd conda; then
        printf 'conda'
    elif have_cmd mamba; then
        printf 'mamba'
    else
        die "neither conda nor mamba is available"
    fi
}

print_cmd() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
}

maybe_run() {
    if [[ "${DRY_RUN:-0}" -eq 1 ]]; then
        print_cmd "$@"
    else
        "$@"
    fi
}

env_exists() {
    local env_name="$1"
    local run_bin
    run_bin="$(conda_run_bin)"
    "$run_bin" env list --json | python3 -c '
import json
import sys

env_name = sys.argv[1]
data = json.load(sys.stdin)
needle = f"/{env_name}"
for prefix in data.get("envs", []):
    if prefix == env_name or prefix.endswith(needle):
        raise SystemExit(0)
raise SystemExit(1)
' "$env_name"
}

ensure_env() {
    local env_name="$1"
    local python_version="$2"
    local create_bin
    create_bin="$(conda_create_bin)"
    if env_exists "$env_name"; then
        log "env already exists: $env_name"
        return 0
    fi
    maybe_run "$create_bin" create -n "$env_name" "python=${python_version}" pip -y
}

conda_run_env() {
    local env_name="$1"
    shift
    local run_bin
    run_bin="$(conda_run_bin)"
    maybe_run "$run_bin" run -n "$env_name" "$@"
}

require_arch_linux() {
    [[ -r /etc/os-release ]] || die "/etc/os-release is missing"
    grep -q '^ID=arch$' /etc/os-release || die "this script currently targets Arch Linux"
}

write_section() {
    local title="$1"
    printf '\n## %s\n' "$title"
}
