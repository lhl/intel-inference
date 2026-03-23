#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VLLM_OPENVINO_DIR="${REPO_ROOT}/05-vllm/vllm-openvino"
# shellcheck source=00-setup/common.sh
source "${SCRIPT_DIR}/common.sh"

ENV_NAME="intel-inf-vllm-openvino"
PYTHON_VERSION="3.12"
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage:
  ./00-setup/setup-vllm-openvino-env.sh [options]

Options:
  --env-name NAME        Conda/mamba env name (default: intel-inf-vllm-openvino)
  --python VERSION       Python version to create with (default: 3.12)
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

[[ -d "$VLLM_OPENVINO_DIR" ]] || die "vllm-openvino checkout not found: $VLLM_OPENVINO_DIR"

ensure_env "$ENV_NAME" "$PYTHON_VERSION"
conda_run_env "$ENV_NAME" python -m pip install --upgrade pip
if [[ "$DRY_RUN" -eq 1 ]]; then
    ENV_PYTHON="/path/to/env/bin/python"
else
    ENV_PYTHON="$(conda_run_env "$ENV_NAME" python -c "import sys; print(sys.executable)")"
fi

env \
    VLLM_TARGET_DEVICE=empty \
    PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu" \
    "$ENV_PYTHON" -m pip install -v "$VLLM_OPENVINO_DIR"
conda_run_env "$ENV_NAME" python -m pip uninstall -y triton triton-xpu || true

conda_run_env "$ENV_NAME" python - <<'PY'
import importlib.metadata as md
import openvino as ov
import torch
import vllm
import vllm_openvino

core = ov.Core()
print("python_ok", True)
print("openvino", ov.__version__)
print("torch", torch.__version__)
print("vllm", md.version("vllm"))
print("vllm_openvino", md.version("vllm-openvino"))
print("available_devices", core.available_devices)
print("plugin_module", vllm_openvino.__file__)
print("vllm_module", vllm.__file__)
PY
