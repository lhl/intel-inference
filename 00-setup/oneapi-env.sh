#!/usr/bin/env bash

# This script is meant to be sourced, not executed.
# shellcheck shell=bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    printf 'source this script instead of executing it:\n' >&2
    printf '  source ./00-setup/oneapi-env.sh\n' >&2
    exit 2
fi

ONEAPI_ROOT="${ONEAPI_ROOT:-/opt/intel/oneapi}"
SETVARS_SH="${ONEAPI_ROOT}/setvars.sh"

if [[ ! -f "$SETVARS_SH" ]]; then
    printf 'missing oneAPI setvars: %s\n' "$SETVARS_SH" >&2
    return 1
fi

if [[ "${INTEL_INF_ONEAPI_ACTIVE:-0}" == "1" ]]; then
    return 0
fi

# shellcheck disable=SC1090
source "$SETVARS_SH" >/dev/null
export INTEL_INF_ONEAPI_ACTIVE=1
export CC="${CC:-icx}"
export CXX="${CXX:-icpx}"
