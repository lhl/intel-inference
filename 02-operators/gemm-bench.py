#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass

import torch


@dataclass
class GemmResult:
    op: str
    dtype: str
    variant: str
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
    unit: str | None
    mean_throughput: float | None
    median_throughput: float | None
    max_throughput: float | None


def synchronize() -> None:
    torch.xpu.synchronize()


def throughput(op_count: int, elapsed_ms: float) -> float:
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


def parse_case(text: str) -> tuple[int, int, int]:
    for sep in ("x", ":", ","):
        if sep in text:
            parts = text.split(sep)
            if len(parts) == 3:
                return tuple(int(part) for part in parts)  # type: ignore[return-value]
    raise ValueError(f"invalid case format: {text}")


def summarise(
    *,
    op: str,
    dtype: str,
    variant: str,
    m: int,
    n: int,
    k: int,
    warmups: int,
    repeats: int,
    first_ms: float,
    timings_ms: list[float],
) -> GemmResult:
    op_count = 2 * m * n * k
    scores = [throughput(op_count, elapsed_ms) for elapsed_ms in timings_ms]
    unit = "TOPS" if dtype == "int8" else "TFLOPS"
    return GemmResult(
        op=op,
        dtype=dtype,
        variant=variant,
        m=m,
        n=n,
        k=k,
        warmups=warmups,
        repeats=repeats,
        status="OK",
        detail="",
        first_ms=first_ms,
        mean_ms=statistics.mean(timings_ms),
        median_ms=statistics.median(timings_ms),
        min_ms=min(timings_ms),
        max_ms=max(timings_ms),
        unit=unit,
        mean_throughput=statistics.mean(scores),
        median_throughput=statistics.median(scores),
        max_throughput=max(scores),
    )


def error_result(
    *,
    op: str,
    dtype: str,
    variant: str,
    m: int,
    n: int,
    k: int,
    warmups: int,
    repeats: int,
    detail: str,
) -> GemmResult:
    return GemmResult(
        op=op,
        dtype=dtype,
        variant=variant,
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
        unit=None,
        mean_throughput=None,
        median_throughput=None,
        max_throughput=None,
    )


def make_float_callable(m: int, n: int, k: int, dtype: torch.dtype, variant: str):
    a = torch.randn((m, k), device="xpu", dtype=dtype)
    b = torch.randn((k, n), device="xpu", dtype=dtype)
    if variant == "compile":
        compiled = torch.compile(lambda x, y: torch.matmul(x, y), backend="inductor")
        return lambda: compiled(a, b)
    return lambda: torch.matmul(a, b)


def make_int8_callable(m: int, n: int, k: int, variant: str):
    if not hasattr(torch, "_int_mm"):
        raise RuntimeError("torch._int_mm not present")
    a = torch.randint(-128, 127, (m, k), device="xpu", dtype=torch.int8)
    b = torch.randint(-128, 127, (k, n), device="xpu", dtype=torch.int8)
    if variant == "compile":
        compiled = torch.compile(lambda x, y: torch._int_mm(x, y), backend="inductor")
        return lambda: compiled(a, b)
    return lambda: torch._int_mm(a, b)


def run_float_case(
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    variant: str,
    repeats: int,
    warmups: int,
) -> GemmResult:
    try:
        fn = make_float_callable(m, n, k, dtype, variant)
        synchronize()
        first_ms, timings = time_callable(fn, repeats, warmups)
        return summarise(
            op="matmul",
            dtype=str(dtype).replace("torch.", ""),
            variant=variant,
            m=m,
            n=n,
            k=k,
            warmups=warmups,
            repeats=repeats,
            first_ms=first_ms,
            timings_ms=timings,
        )
    except Exception as exc:
        return error_result(
            op="matmul",
            dtype=str(dtype).replace("torch.", ""),
            variant=variant,
            m=m,
            n=n,
            k=k,
            warmups=warmups,
            repeats=repeats,
            detail=str(exc).strip().splitlines()[0][:240],
        )


def run_int8_case(m: int, n: int, k: int, variant: str, repeats: int, warmups: int) -> GemmResult:
    try:
        fn = make_int8_callable(m, n, k, variant)
        synchronize()
        first_ms, timings = time_callable(fn, repeats, warmups)
        return summarise(
            op="int_mm",
            dtype="int8",
            variant=variant,
            m=m,
            n=n,
            k=k,
            warmups=warmups,
            repeats=repeats,
            first_ms=first_ms,
            timings_ms=timings,
        )
    except Exception as exc:
        return error_result(
            op="int_mm",
            dtype="int8",
            variant=variant,
            m=m,
            n=n,
            k=k,
            warmups=warmups,
            repeats=repeats,
            detail=str(exc).strip().splitlines()[0][:240],
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark transformer-shaped GEMM operators on PyTorch XPU")
    parser.add_argument(
        "--cases",
        nargs="+",
        default=[
            "1x4096x4096",
            "128x4096x4096",
            "512x4096x4096",
            "128x4096x11008",
            "128x11008x4096",
        ],
    )
    parser.add_argument("--dtypes", nargs="+", default=["float32", "bfloat16", "float16", "int8"])
    parser.add_argument("--variants", nargs="+", default=["eager", "compile"])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--json-out", type=str)
    args = parser.parse_args()

    if not torch.xpu.is_available():
        raise SystemExit("torch.xpu is not available")

    results: list[GemmResult] = []
    parsed_cases = [parse_case(text) for text in args.cases]
    for variant in args.variants:
        for dtype_name in args.dtypes:
            for m, n, k in parsed_cases:
                if dtype_name == "int8":
                    results.append(run_int8_case(m, n, k, variant, args.repeats, args.warmups))
                    continue
                dtype = getattr(torch, dtype_name)
                results.append(run_float_case(m, n, k, dtype, variant, args.repeats, args.warmups))

    header = (
        f"{'op':<10} {'dtype':<10} {'variant':<8} {'m':>6} {'n':>6} {'k':>6} "
        f"{'status':<6} {'first_ms':>10} {'median_ms':>12} {'median':>10} {'unit':>8} detail"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        first_ms = "-" if result.first_ms is None else f"{result.first_ms:.3f}"
        median_ms = "-" if result.median_ms is None else f"{result.median_ms:.3f}"
        median = "-" if result.median_throughput is None else f"{result.median_throughput:.2f}"
        unit = "-" if result.unit is None else result.unit
        print(
            f"{result.op:<10} {result.dtype:<10} {result.variant:<8} {result.m:>6} {result.n:>6} {result.k:>6} "
            f"{result.status:<6} {first_ms:>10} {median_ms:>12} {median:>10} {unit:>8} {result.detail}"
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
