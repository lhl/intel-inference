#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
SYSTEMS_DIR="$SCRIPT_DIR/systems"
SYSTEM_ID="machine"
WRITE_TRACKED_SUMMARY=0
TRACKED_SUMMARY=""

usage() {
    cat <<'EOF'
Usage:
  ./00-setup/collect-system-info.sh [options]

Options:
  --results-dir DIR         Directory for ignored raw captures
  --system-id ID            Sanitized stable system label for filenames and summaries
  --write-tracked-summary   Also write a sanitized tracked summary under 00-setup/systems/
  --tracked-summary PATH    Explicit path for the sanitized tracked summary
  -h, --help                Show this help text
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --system-id)
            SYSTEM_ID="$2"
            shift 2
            ;;
        --write-tracked-summary)
            WRITE_TRACKED_SUMMARY=1
            shift
            ;;
        --tracked-summary)
            TRACKED_SUMMARY="$2"
            WRITE_TRACKED_SUMMARY=1
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'Unknown argument: %s\n\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

mkdir -p "$RESULTS_DIR"
if [[ "$WRITE_TRACKED_SUMMARY" -eq 1 ]]; then
    mkdir -p "$SYSTEMS_DIR"
fi

TIMESTAMP_UTC="$(date -u +%Y%m%dT%H%M%SZ)"
OUTFILE="$RESULTS_DIR/system-info-${SYSTEM_ID}-${TIMESTAMP_UTC}.txt"
SUMMARY_FILE="${TRACKED_SUMMARY:-$SYSTEMS_DIR/${SYSTEM_ID}.md}"

append_text() {
    local title="$1"
    shift
    {
        printf '\n## %s\n' "$title"
        printf '%s\n' "$*"
    } >>"$OUTFILE"
}

run_cmd() {
    local title="$1"
    shift
    {
        printf '\n## %s\n' "$title"
        printf '$'
        printf ' %q' "$@"
        printf '\n'
        "$@"
    } >>"$OUTFILE" 2>&1 || true
}

run_shell() {
    local title="$1"
    local command="$2"
    {
        printf '\n## %s\n' "$title"
        printf '$ bash -lc %q\n' "$command"
        bash -lc "$command"
    } >>"$OUTFILE" 2>&1 || true
}

capture_shell() {
    local command="$1"
    bash -lc "$command" 2>/dev/null || true
}

tool_presence_report() {
    local tool
    for tool in xpu-smi sycl-ls clinfo vulkaninfo cmake gcc clang icx icpx python3; do
        if command -v "$tool" >/dev/null 2>&1; then
            printf '%s=present\n' "$tool"
        else
            printf '%s=not-found\n' "$tool"
        fi
    done
}

{
    printf '# Intel inference setup inventory\n'
    printf 'generated_utc=%s\n' "$TIMESTAMP_UTC"
    printf 'system_id=%s\n' "$SYSTEM_ID"
} >"$OUTFILE"

run_cmd "uname" uname -srm
run_cmd "os-release" cat /etc/os-release
run_cmd "date" date -Is

run_cmd "cpu" lscpu
run_cmd "memory" free -h
run_cmd "block-devices" lsblk
run_cmd "pci-display-devices" lspci -nnk
run_shell "intel-display-filter" "lspci -nnk | grep -A4 -E 'VGA compatible controller|3D controller|Display controller'"

run_cmd "dri-nodes" ls -l /dev/dri
run_shell "kernel-modules" "lsmod | grep -E '^(xe|i915|drm|ivpu|intel_vpu)\\b' || true"

{
    printf '\n## tool-presence\n'
    tool_presence_report
} >>"$OUTFILE" 2>&1
run_cmd "xpu-smi-version" xpu-smi -v
run_cmd "xpu-smi-discovery" xpu-smi discovery
run_shell "clinfo-summary" "clinfo | sed -n '1,160p'"
run_shell "vulkaninfo-summary" "vulkaninfo --summary | sed -n '1,200p'"
run_cmd "python3-version" python3 --version
run_shell "compiler-versions" "gcc --version | sed -n '1,3p'; clang --version | sed -n '1,3p'; icx --version | sed -n '1,3p'; icpx --version | sed -n '1,3p'"
run_shell "intel-env-var-names" "env | grep -E '^(ONEAPI|SYCL|ZE_|OCL_|VULKAN_|LIBVA_)' | cut -d= -f1 | sort || true"

if [[ "$WRITE_TRACKED_SUMMARY" -eq 1 ]]; then
    OS_PRETTY_NAME="$(awk -F= '/^PRETTY_NAME=/{gsub(/"/, "", $2); print $2; exit}' /etc/os-release 2>/dev/null || true)"
    KERNEL_SUMMARY="$(uname -srm 2>/dev/null || true)"
    CPU_MODEL="$(lscpu | awk -F: '/Model name:/ {sub(/^[ \t]+/, "", $2); print $2; exit}' 2>/dev/null || true)"
    CPU_COUNT="$(lscpu | sed -n 's/^CPU(s):[[:space:]]*//p' | head -n 1)"
    MEM_TOTAL="$(free -h | awk '/^Mem:/ {print $2}' 2>/dev/null || true)"
    SWAP_TOTAL="$(free -h | awk '/^Swap:/ {print $2}' 2>/dev/null || true)"
    DISPLAY_FILTER="$(capture_shell "lspci -nnk | grep -A4 -E 'VGA compatible controller|3D controller|Display controller'")"
    NPU_FILTER="$(capture_shell "lspci -nnk | grep -A4 -E 'Processing accelerators|NPU' | sed -n '1,20p'")"
    DRI_NODES="$(capture_shell "ls -l /dev/dri")"
    TOOL_PATHS="$(tool_presence_report)"
    XPU_DISCOVERY="$(capture_shell "xpu-smi discovery | sed -n '1,120p'")"
    CLINFO_SUMMARY="$(capture_shell "clinfo | awk '/Platform Name/ && !seen_p++ {print} /Platform Vendor/ && !seen_pv++ {print} /Platform Version/ && !seen_pver++ {print} /Device Name/ && !seen_d++ {print} /Device Vendor/ && !seen_dv++ {print} /Device Version/ && !seen_dver++ {print} /Driver Version/ && !seen_drv++ {print}'")"
    VULKAN_SUMMARY="$(capture_shell "vulkaninfo --summary | awk '/^GPU[0-9]+:/ {print} /deviceType/ {print} /deviceName/ {print} /driverName/ {print} /driverInfo/ {print} /vendorID/ {print} /deviceID/ {print}' | sed -n '1,40p'")"
    INTEL_ENV_NAMES="$(capture_shell "env | grep -E '^(ONEAPI|SYCL|ZE_|OCL_|VULKAN_|LIBVA_)' | cut -d= -f1 | sort || true")"

    cat >"$SUMMARY_FILE" <<EOF
# System Profile: $SYSTEM_ID

This is a sanitized system profile intended to be safe to check into git for benchmark context and reproducibility.

- captured_utc: $TIMESTAMP_UTC
- os: ${OS_PRETTY_NAME:-unknown}
- kernel: ${KERNEL_SUMMARY:-unknown}

## CPU and memory

~~~text
cpu_model: ${CPU_MODEL:-unknown}
cpu_count: ${CPU_COUNT:-unknown}
memory_total: ${MEM_TOTAL:-unknown}
swap_total: ${SWAP_TOTAL:-unknown}
~~~

## Intel accelerator inventory

~~~text
${DISPLAY_FILTER:-unavailable}
~~~

## NPU inventory

~~~text
${NPU_FILTER:-not-detected}
~~~

## DRI nodes

~~~text
${DRI_NODES:-unavailable}
~~~

## Tool availability

~~~text
${TOOL_PATHS:-unavailable}
~~~

## xpu-smi discovery

~~~text
${XPU_DISCOVERY:-unavailable}
~~~

## OpenCL summary

~~~text
${CLINFO_SUMMARY:-unavailable}
~~~

## Vulkan summary

~~~text
${VULKAN_SUMMARY:-unavailable}
~~~

## Intel env var names

~~~text
${INTEL_ENV_NAMES:-none}
~~~

## Raw capture

The full raw capture for this run was written to:

~~~text
00-setup/results/$(basename "$OUTFILE")
~~~
EOF
fi

append_text "output" "wrote $OUTFILE"
if [[ "$WRITE_TRACKED_SUMMARY" -eq 1 ]]; then
    append_text "tracked-summary" "wrote $SUMMARY_FILE"
fi
printf 'Wrote %s\n' "$OUTFILE"
if [[ "$WRITE_TRACKED_SUMMARY" -eq 1 ]]; then
    printf 'Wrote %s\n' "$SUMMARY_FILE"
fi
