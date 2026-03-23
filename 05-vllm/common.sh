#!/usr/bin/env bash

# Common helpers for 05-vllm scripts.

VLLM_PHASE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${VLLM_PHASE_DIR}/results"

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

sanitize_name() {
    printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9._-' '_'
}

wait_for_http() {
    local url="$1"
    local timeout_s="${2:-180}"
    python - "$url" "$timeout_s" <<'PY'
import sys
import time
import urllib.request

url = sys.argv[1]
timeout_s = float(sys.argv[2])
deadline = time.time() + timeout_s

while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=2.0) as response:
            if 200 <= response.status < 500:
                raise SystemExit(0)
    except Exception:
        time.sleep(0.5)

raise SystemExit(f"timed out waiting for {url}")
PY
}
