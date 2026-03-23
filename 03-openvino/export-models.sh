#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=03-openvino/common.sh
source "${SCRIPT_DIR}/common.sh"
# shellcheck source=03-openvino/model-presets.sh
source "${SCRIPT_DIR}/model-presets.sh"

ENV_NAME="intel-inf-optimum-openvino"
FAMILY="all"
FORCE=0
KEEP_HF_TOKEN=0
MODELS=()

usage() {
    cat <<'EOF'
Usage:
  ./03-openvino/export-models.sh [options]

Options:
  --env-name NAME      Optimum/OpenVINO env name (default: intel-inf-optimum-openvino)
  --family NAME        llm | asr | all (default: all)
  --models NAME...     Explicit model aliases to export
  --force              Re-export even if the output directory already looks populated
  --keep-hf-token      Keep HF_TOKEN in the environment instead of unsetting it for public-model exports
  -h, --help           Show this help text
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-name)
            ENV_NAME="$2"
            shift 2
            ;;
        --family)
            FAMILY="$2"
            shift 2
            ;;
        --models)
            shift
            MODELS=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                MODELS+=("$1")
                shift
            done
            ;;
        --force)
            FORCE=1
            shift
            ;;
        --keep-hf-token)
            KEEP_HF_TOKEN=1
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

ensure_models_dir

if [[ "${#MODELS[@]}" -eq 0 ]]; then
    case "$FAMILY" in
        llm)
            mapfile -t MODELS < <(list_default_llm_models)
            ;;
        asr)
            mapfile -t MODELS < <(list_default_asr_models)
            ;;
        all)
            mapfile -t MODELS < <(list_default_llm_models)
            mapfile -t _asr_models < <(list_default_asr_models)
            MODELS+=("${_asr_models[@]}")
            ;;
        *)
            die "unsupported family: $FAMILY"
            ;;
    esac
fi

for alias in "${MODELS[@]}"; do
    is_known_model_alias "$alias" || die "unknown model alias: $alias"
    model_id="$(resolve_hf_model_id "$alias")"
    model_source="$model_id"
    source_kind="hf"
    export_task="$(resolve_export_task "$alias")"
    export_marker="$(resolve_export_marker "$alias")"
    weight_format="$(resolve_weight_format "$alias")"
    trust_remote_code="$(resolve_trust_remote_code "$alias")"
    output_dir="${MODELS_DIR}/${alias}"

    if local_snapshot="$(resolve_local_snapshot_path "$alias" 2>/dev/null)"; then
        model_source="$local_snapshot"
        source_kind="local"
    fi

    if [[ -f "${output_dir}/${export_marker}" && "$FORCE" -eq 0 ]]; then
        log "skipping ${alias}; ${output_dir} already contains ${export_marker}"
        continue
    fi

    mkdir -p "$output_dir"
    cmd=(optimum-cli export openvino --model "$model_source" --task "$export_task" --weight-format "$weight_format")
    if [[ "$trust_remote_code" == "true" ]]; then
        cmd+=(--trust-remote-code)
    fi
    cmd+=("$output_dir")

    log "exporting ${alias} from ${model_id} (${source_kind}) into ${output_dir}"
    if [[ "$KEEP_HF_TOKEN" -eq 1 ]]; then
        run_in_env "$ENV_NAME" "${cmd[@]}"
    else
        run_in_env "$ENV_NAME" env -u HF_TOKEN "${cmd[@]}"
    fi
done
