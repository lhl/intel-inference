#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=00-setup/common.sh
source "${SCRIPT_DIR}/common.sh"

ENV_NAME="intel-inf-openvino-genai"
PYTHON_VERSION="3.11"
DRY_RUN=0
WITH_OPTIMUM=0

usage() {
    cat <<'EOF'
Usage:
  ./00-setup/setup-openvino-genai-env.sh [options]

Options:
  --env-name NAME        Conda/mamba env name (default: intel-inf-openvino-genai)
  --python VERSION       Python version to create with (default: 3.11)
  --with-optimum         Also install optimum-intel[openvino]
  --no-optimum           Skip optimum-intel[openvino] even if requested earlier
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
        --with-optimum)
            WITH_OPTIMUM=1
            shift
            ;;
        --no-optimum)
            WITH_OPTIMUM=0
            shift
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

PIP_ARGS=(
    python -m pip install --upgrade
    openvino
    openvino-genai
)
if [[ "$WITH_OPTIMUM" -eq 1 ]]; then
    PIP_ARGS+=("optimum-intel[openvino]")
fi
conda_run_env "$ENV_NAME" "${PIP_ARGS[@]}"

conda_run_env "$ENV_NAME" python - <<'PY'
import re
import openvino as ov
import openvino_genai as ovg

def family(version: str) -> str:
    nums = re.findall(r'\d+', version)
    return '.'.join(nums[:3]) if nums else version

print("openvino", ov.__version__)
print("openvino_genai", ovg.__version__)
print("openvino_family", family(ov.__version__))
print("openvino_genai_family", family(ovg.__version__))
PY
