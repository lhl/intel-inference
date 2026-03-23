#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=00-setup/common.sh
source "${SCRIPT_DIR}/common.sh"

ENV_NAME="intel-inf-torch-xpu"
PYTHON_VERSION="3.11"
DRY_RUN=0
CHANNEL="release"

usage() {
    cat <<'EOF'
Usage:
  ./00-setup/setup-torch-xpu-env.sh [options]

Options:
  --env-name NAME        Conda/mamba env name (default: intel-inf-torch-xpu)
  --python VERSION       Python version to create with (default: 3.11)
  --channel release|nightly
                          PyTorch XPU wheel channel (default: release)
  --dry-run              Print actions without executing them
  -h, --help             Show this help text
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-name)
            ENV_NAME="$2"
            shift 2
            ;;
        --python)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --channel)
            CHANNEL="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
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

case "$CHANNEL" in
    release)
        INDEX_URL="https://download.pytorch.org/whl/xpu"
        PRE_ARGS=()
        ;;
    nightly)
        INDEX_URL="https://download.pytorch.org/whl/nightly/xpu"
        PRE_ARGS=(--pre)
        ;;
    *)
        die "unsupported channel: $CHANNEL"
        ;;
esac

ensure_env "$ENV_NAME" "$PYTHON_VERSION"
conda_run_env "$ENV_NAME" python -m pip install --upgrade pip
conda_run_env "$ENV_NAME" python -m pip install --upgrade "${PRE_ARGS[@]}" torch torchvision torchaudio --index-url "$INDEX_URL"
conda_run_env "$ENV_NAME" python -c "import torch; print('torch', torch.__version__); print('xpu_available', torch.xpu.is_available()); print('xpu_count', torch.xpu.device_count() if torch.xpu.is_available() else 0); print('xpu_name', torch.xpu.get_device_name(0) if torch.xpu.is_available() else 'n/a')"
