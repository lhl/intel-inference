#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass

import torch


@dataclass
class BatchedGemmResult:
    dtype: str
    variant: str
    batch: int
    m: int
    n: int
    k: int
    warmups: int
    repeats: int
    status: str
    detail: str
    first_ms: float | None
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
    raise ValueError(f"invalid batched case format: {text}")


def tflops(op_count: int, elapsed_ms: float) -> float:
    return op_count / (elapsed_ms / 1e3) / 1e12


def time_callable(fn, repeats: int, warmups: int) -> tuple[float, list[float]]:
    start = time.perf_counter_ns()
    fn()
    synchronize()
    first_ms = (time.perf_counter_ns() - start) / 1e6

    for _ in range(warmups):
        fn()
    synchronize()

    timings_ms: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        fn()
        synchronize()
        timings_ms.append((time.perf_counter_ns() - start) / 1e6)
    return first_ms, timings_ms


def error_result(
    *,
    dtype: str,
    variant: str,
    batch: int,
    m: int,
    n: int,
    k: int,
    warmups: int,
    repeats: int,
    detail: str,
) -> BatchedGemmResult:
    return BatchedGemmResult(
        dtype=dtype,
        variant=variant,
        batch=batch,
        m=m,
        n=n,
        k=k,
        warmups=warmups,
        repeats=repeats,
        status="ERR",
        detail=detail,
        first_ms=None,
        mean_ms=None,
        median_ms=None,
        min_ms=None,
        max_ms=None,
        mean_tflops=None,
        median_tflops=None,
        max_tflops=None,
    )


def run_case(
    batch: int,
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    variant: str,
    repeats: int,
    warmups: int,
) -> BatchedGemmResult:
    try:
        a = torch.randn((batch, m, k), device="xpu", dtype=dtype)
        b = torch.randn((batch, k, n), device="xpu", dtype=dtype)
        if variant == "compile":
            compiled = torch.compile(lambda x, y: torch.matmul(x, y), backend="inductor")
            fn = lambda: compiled(a, b)
        else:
            fn = lambda: torch.matmul(a, b)
        synchronize()
        first_ms, timings = time_callable(fn, repeats, warmups)
        op_count = 2 * batch * m * n * k
        scores = [tflops(op_count, elapsed_ms) for elapsed_ms in timings]
        return BatchedGemmResult(
            dtype=str(dtype).replace("torch.", ""),
            variant=variant,
            batch=batch,
            m=m,
            n=n,
            k=k,
            warmups=warmups,
            repeats=repeats,
            status="OK",
            detail="",
            first_ms=first_ms,
            mean_ms=statistics.mean(timings),
            median_ms=statistics.median(timings),
            min_ms=min(timings),
            max_ms=max(timings),
            mean_tflops=statistics.mean(scores),
            median_tflops=statistics.median(scores),
            max_tflops=max(scores),
        )
    except Exception as exc:
        return error_result(
            dtype=str(dtype).replace("torch.", ""),
            variant=variant,
            batch=batch,
            m=m,
            n=n,
            k=k,
            warmups=warmups,
            repeats=repeats,
            detail=str(exc).strip().splitlines()[0][:240],
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark batched GEMM on PyTorch XPU")
    parser.add_argument(
        "--cases",
        nargs="+",
        default=[
            "4x128x4096x4096",
            "4x128x4096x11008",
            "4x128x11008x4096",
        ],
    )
    parser.add_argument("--dtypes", nargs="+", default=["float32", "bfloat16", "float16"])
    parser.add_argument("--variants", nargs="+", default=["eager", "compile"])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--json-out", type=str)
    args = parser.parse_args()

    if not torch.xpu.is_available():
        raise SystemExit("torch.xpu is not available")

    results: list[BatchedGemmResult] = []
    parsed_cases = [parse_case(text) for text in args.cases]
    for variant in args.variants:
        for dtype_name in args.dtypes:
            dtype = getattr(torch, dtype_name)
            for batch, m, n, k in parsed_cases:
                results.append(run_case(batch, m, n, k, dtype, variant, args.repeats, args.warmups))

    header = (
        f"{'dtype':<10} {'variant':<8} {'batch':>6} {'m':>6} {'n':>6} {'k':>6} "
        f"{'status':<6} {'first_ms':>10} {'median_ms':>12} {'median_tf':>12} detail"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        first_ms = "-" if result.first_ms is None else f"{result.first_ms:.3f}"
        median_ms = "-" if result.median_ms is None else f"{result.median_ms:.3f}"
        median_tf = "-" if result.median_tflops is None else f"{result.median_tflops:.2f}"
        print(
            f"{result.dtype:<10} {result.variant:<8} {result.batch:>6} {result.m:>6} {result.n:>6} {result.k:>6} "
            f"{result.status:<6} {first_ms:>10} {median_ms:>12} {median_tf:>12} {result.detail}"
        )

    if args.json_out:
        payload = {
            "torch": torch.__version__,
            "device": torch.xpu.get_device_name(0),
            "repeats": args.repeats,
            "warmups": args.warmups,
            "cases": args.cases,
            "variants": args.variants,
            "results": [asdict(result) for result in results],
        }
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
