#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import time
import warnings
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


@dataclass
class AttentionResult:
    backend: str
    dtype: str
    mode: str
    b: int
    h: int
    s: int
    d: int
    warmups: int
    repeats: int
    status: str
    detail: str
    mean_ms: float | None
    median_ms: float | None
    min_ms: float | None
    max_ms: float | None
    mean_tflops: float | None
    median_tflops: float | None
    max_tflops: float | None


def synchronize() -> None:
    torch.xpu.synchronize()


def parse_case(text: str) -> tuple[int, int, int, int]:
    for sep in ("x", ":", ","):
        if sep in text:
            parts = text.split(sep)
            if len(parts) == 4:
                return tuple(int(part) for part in parts)  # type: ignore[return-value]
    raise ValueError(f"invalid case format: {text}")


def parse_mode(text: str) -> bool:
    if text == "causal":
        return True
    if text == "noncausal":
        return False
    raise ValueError(f"unsupported mode: {text}")


def sdpa_flops(b: int, h: int, s: int, d: int, causal: bool) -> int:
    pair_count = s * (s + 1) // 2 if causal else s * s
    return 4 * b * h * d * pair_count


def tflops(op_count: int, elapsed_ms: float) -> float:
    return op_count / (elapsed_ms / 1e3) / 1e12


def time_attention(fn, repeats: int, warmups: int) -> list[float]:
    for _ in range(warmups):
        fn()
    synchronize()
    timings_ms: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        fn()
        synchronize()
        timings_ms.append((time.perf_counter_ns() - start) / 1e6)
    return timings_ms


def backend_context(name: str):
    if name == "DEFAULT":
        return contextlib.nullcontext()
    return sdpa_kernel(getattr(SDPBackend, name))


def run_case(
    *,
    backend_name: str,
    dtype: torch.dtype,
    mode_name: str,
    shape: tuple[int, int, int, int],
    repeats: int,
    warmups: int,
) -> AttentionResult:
    b, h, s, d = shape
    causal = parse_mode(mode_name)
    q = torch.randn((b, h, s, d), device="xpu", dtype=dtype)
    k = torch.randn((b, h, s, d), device="xpu", dtype=dtype)
    v = torch.randn((b, h, s, d), device="xpu", dtype=dtype)
    op_count = sdpa_flops(b, h, s, d, causal)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            context = backend_context(backend_name)
            with context:
                timings = time_attention(
                    lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal),
                    repeats,
                    warmups,
                )
        scores = [tflops(op_count, elapsed_ms) for elapsed_ms in timings]
        return AttentionResult(
            backend=backend_name,
            dtype=str(dtype).replace("torch.", ""),
            mode=mode_name,
            b=b,
            h=h,
            s=s,
            d=d,
            warmups=warmups,
            repeats=repeats,
            status="OK",
            detail="",
            mean_ms=statistics.mean(timings),
            median_ms=statistics.median(timings),
            min_ms=min(timings),
            max_ms=max(timings),
            mean_tflops=statistics.mean(scores),
            median_tflops=statistics.median(scores),
            max_tflops=max(scores),
        )
    except Exception as exc:
        return AttentionResult(
            backend=backend_name,
            dtype=str(dtype).replace("torch.", ""),
            mode=mode_name,
            b=b,
            h=h,
            s=s,
            d=d,
            warmups=warmups,
            repeats=repeats,
            status="ERR",
            detail=str(exc).strip().splitlines()[0][:240],
            mean_ms=None,
            median_ms=None,
            min_ms=None,
            max_ms=None,
            mean_tflops=None,
            median_tflops=None,
            max_tflops=None,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SDPA operator paths on PyTorch XPU")
    parser.add_argument("--cases", nargs="+", default=["1x8x128x128", "1x8x512x128", "1x8x2048x128"])
    parser.add_argument("--dtypes", nargs="+", default=["float32", "bfloat16", "float16"])
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["DEFAULT", "MATH", "OVERRIDEABLE", "FLASH_ATTENTION", "EFFICIENT_ATTENTION"],
    )
    parser.add_argument("--modes", nargs="+", default=["causal"])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--json-out", type=str)
    args = parser.parse_args()

    if not torch.xpu.is_available():
        raise SystemExit("torch.xpu is not available")

    parsed_cases = [parse_case(text) for text in args.cases]
    results: list[AttentionResult] = []
    for dtype_name in args.dtypes:
        dtype = getattr(torch, dtype_name)
        for mode_name in args.modes:
            for backend_name in args.backends:
                for shape in parsed_cases:
                    results.append(
                        run_case(
                            backend_name=backend_name,
                            dtype=dtype,
                            mode_name=mode_name,
                            shape=shape,
                            repeats=args.repeats,
                            warmups=args.warmups,
                        )
                    )

    header = (
        f"{'backend':<18} {'dtype':<10} {'mode':<10} {'b':>3} {'h':>3} {'s':>6} {'d':>4} "
        f"{'status':<6} {'median_ms':>12} {'median_tf':>12} detail"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        median_ms = "-" if result.median_ms is None else f"{result.median_ms:.3f}"
        median_tf = "-" if result.median_tflops is None else f"{result.median_tflops:.2f}"
        detail = result.detail
        print(
            f"{result.backend:<18} {result.dtype:<10} {result.mode:<10} {result.b:>3} {result.h:>3} "
            f"{result.s:>6} {result.d:>4} {result.status:<6} {median_ms:>12} {median_tf:>12} {detail}"
        )

    if args.json_out:
        payload = {
            "torch": torch.__version__,
            "device": torch.xpu.get_device_name(0),
            "repeats": args.repeats,
            "warmups": args.warmups,
            "cases": args.cases,
            "backends": args.backends,
            "modes": args.modes,
            "results": [asdict(result) for result in results],
        }
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
