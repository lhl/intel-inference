#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=01-hardware/common.sh
source "${SCRIPT_DIR}/common.sh"

MBW_SIZE_MIB=4096
MBW_RUNS=5
SYSBENCH_TOTAL="64G"
SYSBENCH_BLOCK="1M"
SYSBENCH_THREADS="$(nproc)"
TIMESTAMP="$(timestamp_utc)"
PREFIX="${TIMESTAMP}-host-mbw"

usage() {
    cat <<'EOF'
Usage:
  ./01-hardware/run-host-mbw.sh [options]

Options:
  --mbw-size-mib N       mbw array size in MiB (default: 4096)
  --mbw-runs N           mbw runs per test (default: 5)
  --sysbench-total SIZE  sysbench memory-total-size (default: 64G)
  --sysbench-block SIZE  sysbench memory-block-size (default: 1M)
  --threads N            sysbench memory threads (default: nproc)
  --prefix NAME          output file prefix inside 01-hardware/results/
  -h, --help             Show this help text
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mbw-size-mib)
            MBW_SIZE_MIB="$2"
            shift 2
            ;;
        --mbw-runs)
            MBW_RUNS="$2"
            shift 2
            ;;
        --sysbench-total)
            SYSBENCH_TOTAL="$2"
            shift 2
            ;;
        --sysbench-block)
            SYSBENCH_BLOCK="$2"
            shift 2
            ;;
        --threads)
            SYSBENCH_THREADS="$2"
            shift 2
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

need_cmd mbw
need_cmd sysbench
ensure_results_dir

OUTFILE="${RESULTS_DIR}/${PREFIX}.log"

{
    printf '# host-memory-bandwidth\n'
    printf 'generated_utc=%s\n' "$(timestamp_utc)"
    printf 'mbw_size_mib=%s\n' "$MBW_SIZE_MIB"
    printf 'mbw_runs=%s\n' "$MBW_RUNS"
    printf 'sysbench_total=%s\n' "$SYSBENCH_TOTAL"
    printf 'sysbench_block=%s\n' "$SYSBENCH_BLOCK"
    printf 'sysbench_threads=%s\n' "$SYSBENCH_THREADS"
    printf '\n## notes\n'
    printf 'The /usr/bin/stream command on this Arch machine is ImageMagick, not the STREAM memory benchmark.\n'
    printf 'This script intentionally uses mbw and sysbench instead.\n'

    printf '\n## mbw memcpy\n'
    mbw -q -n "$MBW_RUNS" -t0 "$MBW_SIZE_MIB"
    printf '\n## mbw dumb\n'
    mbw -q -n "$MBW_RUNS" -t1 "$MBW_SIZE_MIB"
    printf '\n## mbw fixed-block memcpy\n'
    mbw -q -n "$MBW_RUNS" -t2 -b 262144 "$MBW_SIZE_MIB"

    printf '\n## sysbench write\n'
    sysbench memory \
        --threads="$SYSBENCH_THREADS" \
        --memory-total-size="$SYSBENCH_TOTAL" \
        --memory-block-size="$SYSBENCH_BLOCK" \
        --memory-oper=write \
        --memory-access-mode=seq \
        run

    printf '\n## sysbench read\n'
    sysbench memory \
        --threads="$SYSBENCH_THREADS" \
        --memory-total-size="$SYSBENCH_TOTAL" \
        --memory-block-size="$SYSBENCH_BLOCK" \
        --memory-oper=read \
        --memory-access-mode=seq \
        run
} | tee "$OUTFILE"

printf '\nWrote %s\n' "$OUTFILE"
