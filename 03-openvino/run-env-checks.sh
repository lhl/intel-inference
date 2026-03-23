#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=03-openvino/common.sh
source "${SCRIPT_DIR}/common.sh"

OPENVINO_ENV="intel-inf-openvino"
GENAI_ENV="intel-inf-openvino-genai"
OPTIMUM_ENV="intel-inf-optimum-openvino"
PREFIX="$(timestamp_utc)-env-checks"

usage() {
    cat <<'EOF'
Usage:
  ./03-openvino/run-env-checks.sh [options]

Options:
  --openvino-env NAME   OpenVINO env name (default: intel-inf-openvino)
  --genai-env NAME      OpenVINO GenAI env name (default: intel-inf-openvino-genai)
  --optimum-env NAME    Optimum Intel env name (default: intel-inf-optimum-openvino)
  --prefix NAME         Output prefix inside 03-openvino/results/
  -h, --help            Show this help text
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --openvino-env)
            OPENVINO_ENV="$2"
            shift 2
            ;;
        --genai-env)
            GENAI_ENV="$2"
            shift 2
            ;;
        --optimum-env)
            OPTIMUM_ENV="$2"
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

maybe_source_npu_workaround
ensure_results_dir

LOGFILE="${RESULTS_DIR}/${PREFIX}.log"
JSON_OPENVINO="${RESULTS_DIR}/${PREFIX}-openvino.json"
JSON_GENAI="${RESULTS_DIR}/${PREFIX}-openvino-genai.json"
JSON_OPTIMUM="${RESULTS_DIR}/${PREFIX}-optimum-openvino.json"

{
    printf '## openvino\n'
    run_in_env "$OPENVINO_ENV" python "${SCRIPT_DIR}/env-check.py" \
        --label openvino \
        --json-out "$JSON_OPENVINO"

    printf '\n## openvino-genai\n'
    run_in_env "$GENAI_ENV" python "${SCRIPT_DIR}/env-check.py" \
        --label openvino-genai \
        --check-genai \
        --json-out "$JSON_GENAI"

    printf '\n## optimum-openvino\n'
    run_in_env "$OPTIMUM_ENV" python "${SCRIPT_DIR}/env-check.py" \
        --label optimum-openvino \
        --check-optimum \
        --json-out "$JSON_OPTIMUM"
} | tee "$LOGFILE"

printf '\nWrote %s\n' "$LOGFILE"
printf 'Wrote %s\n' "$JSON_OPENVINO"
printf 'Wrote %s\n' "$JSON_GENAI"
printf 'Wrote %s\n' "$JSON_OPTIMUM"
