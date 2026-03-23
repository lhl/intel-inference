#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

SCRIPTS=(
    build-vulkan.sh
    build-sycl.sh
    build-openvino.sh
)

summary_status=0

for script in "${SCRIPTS[@]}"; do
    printf '\n== %s ==\n' "$script"
    if "${SCRIPT_DIR}/${script}" "$@"; then
        printf 'status=PASS script=%s\n' "$script"
    else
        printf 'status=FAIL script=%s\n' "$script" >&2
        summary_status=1
    fi
done

exit "$summary_status"
