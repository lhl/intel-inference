#!/usr/bin/env bash

# Source this on Arch when using intel-npu-driver-bin so the dynamic loader can
# find libze_intel_npu.so under its Debian-style install path.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    printf 'source this file instead of executing it: source ./00-setup/npu-env.sh\n' >&2
    exit 1
fi

NPU_LIBDIR="/usr/lib/x86_64-linux-gnu"
NPU_LIB="${NPU_LIBDIR}/libze_intel_npu.so"

if [[ ! -f "$NPU_LIB" ]]; then
    printf '[npu-env.sh] no Intel NPU userspace library found at %s\n' "$NPU_LIB" >&2
    return 1
fi

case ":${LD_LIBRARY_PATH:-}:" in
    *":${NPU_LIBDIR}:"*)
        ;;
    *)
        export LD_LIBRARY_PATH="${NPU_LIBDIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        ;;
esac

printf '[npu-env.sh] LD_LIBRARY_PATH includes %s\n' "$NPU_LIBDIR"
