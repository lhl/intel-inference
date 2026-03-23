#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=00-setup/common.sh
source "${SCRIPT_DIR}/common.sh"

OPENVINO_ENV="intel-inf-openvino"
NPU_LIBDIR="/usr/lib/x86_64-linux-gnu"
NPU_LIB="${NPU_LIBDIR}/libze_intel_npu.so"

usage() {
    cat <<'EOF'
Usage:
  ./00-setup/verify-npu-stack.sh [options]

Options:
  --openvino-env NAME    OpenVINO env to validate (default: intel-inf-openvino)
  -h, --help             Show this help text
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --openvino-env)
            OPENVINO_ENV="$2"
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

STATUS=0

write_section "pci"
lspci -nnk | grep -A4 -E 'Processing accelerators|NPU' || STATUS=1

write_section "device-nodes"
ls -l /dev/accel || STATUS=1

write_section "driver-package"
pacman -Q intel-npu-driver-bin 2>/dev/null || printf 'intel-npu-driver-bin not found via pacman\n'

write_section "level-zero-npu"
if [[ -f "$NPU_LIB" ]]; then
    printf 'libze_intel_npu=%s\n' "$NPU_LIB"
    if python -c "import ctypes; ctypes.CDLL('libze_intel_npu.so'); print('loader_visible=yes')" 2>/dev/null; then
        :
    else
        printf 'loader_visible=no\n'
        printf 'recommended_env=source ./00-setup/npu-env.sh\n'
    fi
else
    printf 'missing=%s\n' "$NPU_LIB"
fi

write_section "openvino-plugin"
if env_exists "$OPENVINO_ENV"; then
    conda_run_env "$OPENVINO_ENV" python -c "from pathlib import Path; import openvino as ov; p=Path(ov.__file__).resolve().parent; plugin=p/'libs'/'libopenvino_intel_npu_plugin.so'; print('plugin_path', plugin); print('plugin_exists', plugin.exists())"
else
    printf 'missing env: %s\n' "$OPENVINO_ENV"
    STATUS=1
fi

write_section "openvino-devices"
if env_exists "$OPENVINO_ENV"; then
    if conda_run_env "$OPENVINO_ENV" python -c "import openvino as ov; core=ov.Core(); print('available_devices', core.available_devices); print('npu_full_device_name', core.get_property('NPU','FULL_DEVICE_NAME'))"; then
        RESULT_LABEL="PASS"
    elif [[ -f "$NPU_LIB" ]] && conda_run_env "$OPENVINO_ENV" env LD_LIBRARY_PATH="${NPU_LIBDIR}:${LD_LIBRARY_PATH:-}" python -c "import openvino as ov; core=ov.Core(); print('available_devices', core.available_devices); print('npu_full_device_name', core.get_property('NPU','FULL_DEVICE_NAME'))"; then
        RESULT_LABEL="PASS_WITH_WORKAROUND"
        cat <<'EOF'
workaround=Set LD_LIBRARY_PATH to include /usr/lib/x86_64-linux-gnu before using OpenVINO NPU on Arch with intel-npu-driver-bin.
recommended_env=source ./00-setup/npu-env.sh
EOF
    else
        STATUS=1
        cat <<'EOF'
diagnosis=OpenVINO NPU plugin is present but the runtime did not enumerate a usable NPU device.
likely_causes=plugin/driver version mismatch, unsupported distro/userspace combination, or incomplete NPU userspace setup
EOF
    fi
fi

if [[ "$STATUS" -ne 0 ]]; then
    printf '\nresult=FAIL\n'
    exit 1
fi

printf '\nresult=%s\n' "${RESULT_LABEL:-PASS}"
