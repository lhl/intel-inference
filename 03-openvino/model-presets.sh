#!/usr/bin/env bash

MODEL_PRESETS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

list_default_llm_models() {
    printf '%s\n' \
        llama32_1b_instruct \
        lfm2_1p2b
}

list_experimental_llm_models() {
    printf '%s\n' \
        qwen35_0p8b \
        lfm2_8b_a1b
}

list_default_asr_models() {
    printf '%s\n' \
        whisper_large_v3_turbo \
        whisper_large_v3
}

is_known_model_alias() {
    case "$1" in
        llama32_1b_instruct|qwen35_0p8b|lfm2_1p2b|lfm25_1p2b_instruct|lfm2_8b_a1b|whisper_large_v3_turbo|whisper_large_v3)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

resolve_model_family() {
    case "$1" in
        llama32_1b_instruct|qwen35_0p8b|lfm2_1p2b|lfm25_1p2b_instruct|lfm2_8b_a1b)
            printf 'llm\n'
            ;;
        whisper_large_v3_turbo|whisper_large_v3)
            printf 'asr\n'
            ;;
        *)
            return 1
            ;;
    esac
}

resolve_hf_model_id() {
    case "$1" in
        llama32_1b_instruct)
            printf 'meta-llama/Llama-3.2-1B-Instruct\n'
            ;;
        qwen35_0p8b)
            printf 'Qwen/Qwen3.5-0.8B\n'
            ;;
        lfm2_1p2b)
            printf 'LiquidAI/LFM2-1.2B\n'
            ;;
        lfm25_1p2b_instruct)
            printf 'LiquidAI/LFM2.5-1.2B-Instruct\n'
            ;;
        lfm2_8b_a1b)
            printf 'LiquidAI/LFM2-8B-A1B\n'
            ;;
        whisper_large_v3_turbo)
            printf 'openai/whisper-large-v3-turbo\n'
            ;;
        whisper_large_v3)
            printf 'openai/whisper-large-v3\n'
            ;;
        *)
            return 1
            ;;
    esac
}

resolve_export_task() {
    case "$(resolve_model_family "$1")" in
        llm)
            printf 'text-generation-with-past\n'
            ;;
        asr)
            printf 'automatic-speech-recognition-with-past\n'
            ;;
        *)
            return 1
            ;;
    esac
}

resolve_weight_format() {
    printf 'fp16\n'
}

resolve_export_marker() {
    case "$(resolve_model_family "$1")" in
        llm)
            printf 'openvino_model.xml\n'
            ;;
        asr)
            printf 'openvino_encoder_model.xml\n'
            ;;
        *)
            return 1
            ;;
    esac
}

resolve_trust_remote_code() {
    case "$1" in
        qwen35_0p8b|lfm2_1p2b|lfm25_1p2b_instruct|lfm2_8b_a1b)
            printf 'true\n'
            ;;
        *)
            printf 'false\n'
            ;;
    esac
}

resolve_served_model_name() {
    printf '%s\n' "$1"
}

resolve_local_snapshot_path() {
    local model_id
    model_id="$(resolve_hf_model_id "$1")" || return 1
    python "${MODEL_PRESETS_DIR}/resolve-hf-snapshot.py" "$model_id"
}
