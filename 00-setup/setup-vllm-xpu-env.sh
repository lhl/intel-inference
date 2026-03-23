#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VLLM_DIR="${REPO_ROOT}/05-vllm/vllm"
ONEAPI_COMPILER_BIN="${ONEAPI_ROOT:-/opt/intel/oneapi}/compiler/latest/bin"
# shellcheck source=00-setup/common.sh
source "${SCRIPT_DIR}/common.sh"

ENV_NAME="intel-inf-vllm-xpu"
PYTHON_VERSION="3.12"
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage:
  ./00-setup/setup-vllm-xpu-env.sh [options]

Options:
  --env-name NAME        Conda/mamba env name (default: intel-inf-vllm-xpu)
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

[[ -d "$VLLM_DIR" ]] || die "vllm checkout not found: $VLLM_DIR"
[[ -x "${ONEAPI_COMPILER_BIN}/icx" ]] || die "missing oneAPI compiler binary: ${ONEAPI_COMPILER_BIN}/icx"

ensure_env "$ENV_NAME" "$PYTHON_VERSION"
conda_run_env "$ENV_NAME" python -m pip install --upgrade pip
conda_run_env "$ENV_NAME" python -m pip install -v -r "${VLLM_DIR}/requirements/xpu.txt"
conda_run_env "$ENV_NAME" python -m pip install --upgrade grpcio-tools protobuf nanobind
conda_run_env "$ENV_NAME" python -m pip uninstall -y triton triton-xpu || true
conda_run_env "$ENV_NAME" python -m pip install triton-xpu==3.6.0 --extra-index-url https://download.pytorch.org/whl/xpu

if [[ "$DRY_RUN" -eq 1 ]]; then
    ENV_PYTHON="/path/to/env/bin/python"
    SITE_PACKAGES="/path/to/site-packages"
else
    ENV_PYTHON="$(conda_run_env "$ENV_NAME" python -c "import sys; print(sys.executable)")"
    SITE_PACKAGES="$(conda_run_env "$ENV_NAME" python -c "import site; paths = site.getsitepackages(); print(paths[0] if paths else '')")"
fi

bash -c \
    "export PATH='${ONEAPI_COMPILER_BIN}:$(dirname "$ENV_PYTHON")':\$PATH && export CC=icx && export CXX=icpx && export CMAKE_PREFIX_PATH='${SITE_PACKAGES}':\${CMAKE_PREFIX_PATH:-} && export VLLM_TARGET_DEVICE=xpu && '${ENV_PYTHON}' -m pip install --no-build-isolation -e '${VLLM_DIR}' -v"
conda_run_env "$ENV_NAME" python -m pip uninstall -y triton triton-xpu || true
conda_run_env "$ENV_NAME" python -m pip install --force-reinstall triton-xpu==3.6.0 --extra-index-url https://download.pytorch.org/whl/xpu

conda_run_env "$ENV_NAME" python - <<'PY'
import importlib.metadata as md
import torch
import vllm

print("python_ok", True)
print("torch", torch.__version__)
print("vllm", md.version("vllm"))
print("xpu_available", torch.xpu.is_available())
print("xpu_count", torch.xpu.device_count() if torch.xpu.is_available() else 0)
print("xpu_name", torch.xpu.get_device_name(0) if torch.xpu.is_available() else "n/a")
print("vllm_module", vllm.__file__)
PY
