#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


@dataclass
class SmokeResult:
    category: str
    name: str
    status: str
    detail: str


def safe_detail(exc: Exception) -> str:
    text = str(exc).strip()
    if not text:
        return exc.__class__.__name__
    return text.splitlines()[0][:240]


def synchronize() -> None:
    torch.xpu.synchronize()


def dtype_smoke_results() -> list[SmokeResult]:
    results: list[SmokeResult] = []

    for dtype_name in ("float32", "bfloat16", "float16"):
        dtype = getattr(torch, dtype_name)
        try:
            a = torch.randn((256, 256), device="xpu", dtype=dtype)
            b = torch.randn((256, 256), device="xpu", dtype=dtype)
            out = torch.matmul(a, b)
            synchronize()
            results.append(
                SmokeResult(
                    category="matmul_dtype",
                    name=dtype_name,
                    status="OK",
                    detail=str(tuple(out.shape)),
                )
            )
        except Exception as exc:
            results.append(
                SmokeResult(
                    category="matmul_dtype",
                    name=dtype_name,
                    status="ERR",
                    detail=safe_detail(exc),
                )
            )

    if hasattr(torch, "_int_mm"):
        try:
            a = torch.randint(-128, 127, (256, 256), device="xpu", dtype=torch.int8)
            b = torch.randint(-128, 127, (256, 256), device="xpu", dtype=torch.int8)
            out = torch._int_mm(a, b)
            synchronize()
            results.append(
                SmokeResult(
                    category="matmul_dtype",
                    name="int8",
                    status="OK",
                    detail=f"shape={tuple(out.shape)} dtype={out.dtype}",
                )
            )
        except Exception as exc:
            results.append(
                SmokeResult(
                    category="matmul_dtype",
                    name="int8",
                    status="ERR",
                    detail=safe_detail(exc),
                )
            )
    else:
        results.append(SmokeResult("matmul_dtype", "int8", "MISSING", "torch._int_mm not present"))

    for dtype_name in ("float32", "bfloat16", "float16"):
        dtype = getattr(torch, dtype_name)
        try:
            q = torch.randn((1, 4, 256, 64), device="xpu", dtype=dtype)
            k = torch.randn((1, 4, 256, 64), device="xpu", dtype=dtype)
            v = torch.randn((1, 4, 256, 64), device="xpu", dtype=dtype)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            synchronize()
            results.append(
                SmokeResult(
                    category="sdpa_dtype",
                    name=dtype_name,
                    status="OK",
                    detail=str(tuple(out.shape)),
                )
            )
        except Exception as exc:
            results.append(
                SmokeResult(
                    category="sdpa_dtype",
                    name=dtype_name,
                    status="ERR",
                    detail=safe_detail(exc),
                )
            )

    return results


def backend_results() -> list[SmokeResult]:
    q = torch.randn((1, 4, 256, 64), device="xpu", dtype=torch.float16)
    k = torch.randn((1, 4, 256, 64), device="xpu", dtype=torch.float16)
    v = torch.randn((1, 4, 256, 64), device="xpu", dtype=torch.float16)

    checks = [
        "FLASH_ATTENTION",
        "EFFICIENT_ATTENTION",
        "MATH",
        "OVERRIDEABLE",
        "CUDNN_ATTENTION",
    ]
    results: list[SmokeResult] = []
    for name in checks:
        backend = getattr(SDPBackend, name)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with sdpa_kernel(backend):
                    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                synchronize()
            results.append(SmokeResult("forced_sdpa_backend", name, "OK", str(tuple(out.shape))))
        except Exception as exc:
            results.append(
                SmokeResult(
                    category="forced_sdpa_backend",
                    name=name,
                    status="ERR",
                    detail=safe_detail(exc),
                )
            )
    return results


def viability_results() -> list[SmokeResult]:
    results: list[SmokeResult] = []
    try:
        from torch.nn.attention import SDPAParams, can_use_efficient_attention, can_use_flash_attention

        q = torch.randn((1, 4, 256, 64), device="xpu", dtype=torch.float16)
        k = torch.randn((1, 4, 256, 64), device="xpu", dtype=torch.float16)
        v = torch.randn((1, 4, 256, 64), device="xpu", dtype=torch.float16)
        params = SDPAParams(q, k, v, None, 0.0, True, False)
        results.append(
            SmokeResult(
                "sdpa_viability",
                "can_use_flash_attention",
                "OK",
                str(bool(can_use_flash_attention(params, debug=True))),
            )
        )
        results.append(
            SmokeResult(
                "sdpa_viability",
                "can_use_efficient_attention",
                "OK",
                str(bool(can_use_efficient_attention(params, debug=True))),
            )
        )
    except Exception as exc:
        results.append(SmokeResult("sdpa_viability", "api_check", "ERR", safe_detail(exc)))
    return results


def device_summary() -> dict[str, object]:
    props = torch.xpu.get_device_properties(0)
    return {
        "torch": torch.__version__,
        "xpu_available": torch.xpu.is_available(),
        "device_name": torch.xpu.get_device_name(0),
        "driver_version": getattr(props, "driver_version", None),
        "platform_name": getattr(props, "platform_name", None),
        "type": getattr(props, "type", None),
        "total_memory_mb": getattr(props, "total_memory", None),
        "max_compute_units": getattr(props, "max_compute_units", None),
        "gpu_eu_count": getattr(props, "gpu_eu_count", None),
        "gpu_subslice_count": getattr(props, "gpu_subslice_count", None),
        "max_work_group_size": getattr(props, "max_work_group_size", None),
        "sub_group_sizes": list(getattr(props, "sub_group_sizes", [])),
        "has_fp16": getattr(props, "has_fp16", None),
        "has_fp64": getattr(props, "has_fp64", None),
        "has_atomic64": getattr(props, "has_atomic64", None),
        "has_scaled_mm": hasattr(torch, "_scaled_mm"),
        "has_int_mm": hasattr(torch, "_int_mm"),
        "has_compile": hasattr(torch, "compile"),
        "has_profiler": hasattr(torch, "profiler"),
        "flash_impls": list(torch.nn.attention.list_flash_attention_impls()),
        "current_flash_impl": torch.nn.attention.current_flash_attention_impl(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check PyTorch XPU operator and SDPA backend support")
    parser.add_argument("--json-out", type=str)
    args = parser.parse_args()

    if not torch.xpu.is_available():
        raise SystemExit("torch.xpu is not available")

    summary = device_summary()
    results = dtype_smoke_results() + viability_results() + backend_results()

    print("torch", summary["torch"])
    print("device", summary["device_name"])
    print("driver_version", summary["driver_version"])
    print("flash_impls", summary["flash_impls"])
    print("current_flash_impl", summary["current_flash_impl"])
    print()
    for result in results:
        print(f"{result.category:<20} {result.name:<28} {result.status:<8} {result.detail}")

    if args.json_out:
        payload = {
            "summary": summary,
            "results": [asdict(result) for result in results],
        }
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
