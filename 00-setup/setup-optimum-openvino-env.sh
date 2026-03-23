#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=00-setup/common.sh
source "${SCRIPT_DIR}/common.sh"

ENV_NAME="intel-inf-optimum-openvino"
PYTHON_VERSION="3.11"
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage:
  ./00-setup/setup-optimum-openvino-env.sh [options]

Options:
  --env-name NAME        Conda/mamba env name (default: intel-inf-optimum-openvino)
  --python VERSION       Python version to create with (default: 3.11)
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

ensure_env "$ENV_NAME" "$PYTHON_VERSION"
conda_run_env "$ENV_NAME" python -m pip install --upgrade pip
conda_run_env "$ENV_NAME" python -m pip install --upgrade openvino "optimum-intel[openvino]"

conda_run_env "$ENV_NAME" python - <<'PY'
import importlib.metadata as md
import openvino as ov
from optimum.intel import OVModelForCausalLM

core = ov.Core()
print("openvino", ov.__version__)
print("optimum_intel", md.version("optimum-intel"))
print("torch", md.version("torch"))
print("available_devices", core.available_devices)
print("ov_model_for_causal_lm", OVModelForCausalLM.__name__)
PY
