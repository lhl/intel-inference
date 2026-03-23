#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=00-setup/common.sh
source "${SCRIPT_DIR}/common.sh"

DRY_RUN=0
NO_CONFIRM=0
STRATEGY="granular"

usage() {
    cat <<'EOF'
Usage:
  ./00-setup/install-oneapi-arch.sh [options]

Options:
  --strategy granular|basekit   Install the smaller compiler+SYCL set or the full basekit
  --dry-run                     Print the install command without executing it
  --noconfirm                   Pass --noconfirm to pacman
  -h, --help                    Show this help text
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --strategy)
            STRATEGY="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --noconfirm)
            NO_CONFIRM=1
            shift
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

require_arch_linux
need_cmd pacman
need_cmd sudo

case "$STRATEGY" in
    granular)
        PACKAGES=(
            intel-oneapi-dpcpp-cpp
            intel-oneapi-mkl-sycl
        )
        ;;
    basekit)
        PACKAGES=(
            intel-oneapi-basekit
        )
        ;;
    *)
        die "unsupported strategy: $STRATEGY"
        ;;
esac

log "using Arch official packages for oneAPI"
log "strategy: $STRATEGY"
for pkg in "${PACKAGES[@]}"; do
    pacman -Si "$pkg" >/dev/null 2>&1 || die "package not found in configured repos: $pkg"
done

CMD=(sudo pacman -S --needed)
if [[ "$NO_CONFIRM" -eq 1 ]]; then
    CMD+=(--noconfirm)
fi
CMD+=(-- "${PACKAGES[@]}")

if [[ "$DRY_RUN" -eq 1 ]]; then
    print_cmd "${CMD[@]}"
    exit 0
fi

log "this may prompt for your sudo password"
"${CMD[@]}"
log "oneAPI package install completed"
log "next step: source 00-setup/oneapi-env.sh and run 00-setup/verify-oneapi.sh"
