#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=00-setup/common.sh
source "${SCRIPT_DIR}/common.sh"

TORCH_ENV="intel-inf-torch-xpu"
OPENVINO_ENV="intel-inf-openvino"

usage() {
    cat <<'EOF'
Usage:
  ./00-setup/verify-gpu-stack.sh [options]

Options:
  --torch-env NAME       Torch XPU env to validate (default: intel-inf-torch-xpu)
  --openvino-env NAME    OpenVINO env to validate (default: intel-inf-openvino)
  -h, --help             Show this help text
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --torch-env)
            TORCH_ENV="$2"
            shift 2
            ;;
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
lspci -nnk | grep -A4 -E 'VGA compatible controller|3D controller|Display controller' || STATUS=1

write_section "dri"
ls -l /dev/dri || STATUS=1

write_section "opencl"
clinfo | awk '/Platform Name/ && !seen_p++ {print} /Device Name/ && !seen_d++ {print} /Driver Version/ && !seen_drv++ {print}' || STATUS=1

write_section "vulkan"
vulkaninfo --summary 2>/dev/null | awk '/^GPU[0-9]+:/ {print} /deviceName/ {print} /driverInfo/ {print}' | sed -n '1,20p' || STATUS=1

write_section "torch-xpu"
if env_exists "$TORCH_ENV"; then
    conda_run_env "$TORCH_ENV" python -c "import torch; print('torch', torch.__version__); print('xpu_available', torch.xpu.is_available()); print('xpu_count', torch.xpu.device_count() if torch.xpu.is_available() else 0); print('xpu_name', torch.xpu.get_device_name(0) if torch.xpu.is_available() else 'n/a')"
else
    printf 'missing env: %s\n' "$TORCH_ENV"
    STATUS=1
fi

write_section "openvino"
if env_exists "$OPENVINO_ENV"; then
    conda_run_env "$OPENVINO_ENV" python -c "import openvino as ov; core=ov.Core(); print('openvino', ov.__version__); print('available_devices', core.available_devices)"
else
    printf 'missing env: %s\n' "$OPENVINO_ENV"
    STATUS=1
fi

if [[ "$STATUS" -ne 0 ]]; then
    printf '\nresult=FAIL\n'
    exit 1
fi

printf '\nresult=PASS\n'
