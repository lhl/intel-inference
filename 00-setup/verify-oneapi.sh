#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=00-setup/common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
    cat <<'EOF'
Usage:
  ./00-setup/verify-oneapi.sh

Checks whether the Arch oneAPI compiler toolchain is installed and whether the
expected commands become available after sourcing /opt/intel/oneapi/setvars.sh.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

require_arch_linux
need_cmd pacman
need_cmd bash

STATUS=0
SETVARS_SH="/opt/intel/oneapi/setvars.sh"

write_section "packages"
if pacman -Q intel-oneapi-basekit >/dev/null 2>&1; then
    pacman -Q intel-oneapi-basekit
else
    pacman -Q intel-oneapi-dpcpp-cpp intel-oneapi-mkl-sycl 2>/dev/null || STATUS=1
fi

write_section "setvars"
if [[ -f "$SETVARS_SH" ]]; then
    printf 'setvars=%s\n' "$SETVARS_SH"
else
    printf 'missing=%s\n' "$SETVARS_SH"
    STATUS=1
fi

write_section "toolchain"
if [[ -f "$SETVARS_SH" ]]; then
    bash -lc "
        source \"$SETVARS_SH\" >/dev/null &&
        command -v icx &&
        command -v icpx &&
        command -v sycl-ls &&
        icx --version | sed -n '1p' &&
        icpx --version | sed -n '1p' &&
        sycl-ls | sed -n '1,20p'
    " || STATUS=1
fi

if [[ "$STATUS" -ne 0 ]]; then
    printf '\nresult=FAIL\n'
    exit 1
fi

printf '\nresult=PASS\n'
