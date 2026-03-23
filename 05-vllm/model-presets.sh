#!/usr/bin/env bash

MODEL_PRESETS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=03-openvino/model-presets.sh
source "${MODEL_PRESETS_DIR}/../03-openvino/model-presets.sh"

list_default_vllm_models() {
    printf '%s\n' \
        llama32_1b_instruct \
        qwen35_0p8b \
        lfm2_1p2b
}

list_experimental_vllm_models() {
    printf '%s\n' \
        lfm2_8b_a1b
}
