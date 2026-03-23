#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=01-hardware/common.sh
source "${SCRIPT_DIR}/common.sh"

SECONDS_TO_RUN=15
SAMPLE_MS=500
PREFIX="$(timestamp_utc)-intel-gpu-top"
DEVICE=""

usage() {
    cat <<'EOF'
Usage:
  ./01-hardware/collect-intel-gpu-top.sh [options] [-- command ...]

Options:
  --seconds N           Capture duration in seconds (default: 15)
  --sample-ms N         Sample period in milliseconds (default: 500)
  --prefix NAME         Output file prefix inside 01-hardware/results/
  --device FILTER       intel_gpu_top device filter, e.g. drm:/dev/dri/card0
  -h, --help            Show this help text

If a command is passed after `--`, intel_gpu_top will run in the background
while that command executes.
EOF
}

COMMAND=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seconds)
            SECONDS_TO_RUN="$2"
            shift 2
            ;;
        --sample-ms)
            SAMPLE_MS="$2"
            shift 2
            ;;
        --prefix)
            PREFIX="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --)
            shift
            COMMAND=("$@")
            break
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

need_cmd intel_gpu_top
need_cmd python3
ensure_results_dir

OUTFILE="${RESULTS_DIR}/${PREFIX}.json"
ERRFILE="${RESULTS_DIR}/${PREFIX}.stderr.log"
SAMPLES=$(( (SECONDS_TO_RUN * 1000) / SAMPLE_MS ))
if [[ "$SAMPLES" -lt 1 ]]; then
    SAMPLES=1
fi

if [[ -z "$DEVICE" ]]; then
    DEVICE="$(intel_gpu_top -L 2>/dev/null | awk 'NF {print $NF; exit}')"
fi

[[ -n "$DEVICE" ]] || die "could not auto-detect an intel_gpu_top device filter; try --device"

CMD=(intel_gpu_top -J -s "$SAMPLE_MS" -n "$SAMPLES" -o "$OUTFILE")
CMD+=(-d "$DEVICE")

write_failure_payload() {
    local message_file="$1"
    python3 - "$OUTFILE" "$DEVICE" "$message_file" <<'PY'
import json
import pathlib
import sys

outfile = pathlib.Path(sys.argv[1])
device = sys.argv[2]
message_path = pathlib.Path(sys.argv[3])
message = message_path.read_text(encoding="utf-8") if message_path.exists() else ""
payload = {
    "status": "unsupported",
    "tool": "intel_gpu_top",
    "device": device,
    "error": message.strip(),
}
outfile.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

if [[ "${#COMMAND[@]}" -eq 0 ]]; then
    if ! "${CMD[@]}" 2>"$ERRFILE"; then
        write_failure_payload "$ERRFILE"
        printf 'intel_gpu_top unavailable; wrote %s and %s\n' "$OUTFILE" "$ERRFILE" >&2
    fi
else
    "${CMD[@]}" 2>"$ERRFILE" >/dev/null &
    GPU_TOP_PID=$!
    trap 'kill "$GPU_TOP_PID" >/dev/null 2>&1 || true' EXIT
    "${COMMAND[@]}"
    if ! wait "$GPU_TOP_PID"; then
        write_failure_payload "$ERRFILE"
        printf 'intel_gpu_top unavailable; wrote %s and %s\n' "$OUTFILE" "$ERRFILE" >&2
    fi
    trap - EXIT
fi

printf 'Wrote %s\n' "$OUTFILE"
