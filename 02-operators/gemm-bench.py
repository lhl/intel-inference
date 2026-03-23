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
    m: int
    n: int
    k: int
    warmups: int
    repeats: int
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    unit: str
    mean_throughput: float
    median_throughput: float
    max_throughput: float


def synchronize() -> None:
    torch.xpu.synchronize()


def throughput(op_count: int, elapsed_ms: float) -> float:
    return op_count / (elapsed_ms / 1e3) / 1e12


def time_callable(fn, repeats: int, warmups: int) -> list[float]:
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
    m: int,
    n: int,
    k: int,
    warmups: int,
    repeats: int,
    timings_ms: list[float],
) -> GemmResult:
    op_count = 2 * m * n * k
    scores = [throughput(op_count, elapsed_ms) for elapsed_ms in timings_ms]
    unit = "TOPS" if dtype == "int8" else "TFLOPS"
    return GemmResult(
        op=op,
        dtype=dtype,
        m=m,
        n=n,
        k=k,
        warmups=warmups,
        repeats=repeats,
        mean_ms=statistics.mean(timings_ms),
        median_ms=statistics.median(timings_ms),
        min_ms=min(timings_ms),
        max_ms=max(timings_ms),
        unit=unit,
        mean_throughput=statistics.mean(scores),
        median_throughput=statistics.median(scores),
        max_throughput=max(scores),
    )


def run_float_case(m: int, n: int, k: int, dtype: torch.dtype, repeats: int, warmups: int) -> GemmResult:
    a = torch.randn((m, k), device="xpu", dtype=dtype)
    b = torch.randn((k, n), device="xpu", dtype=dtype)
    out = torch.empty((m, n), device="xpu", dtype=dtype)
    synchronize()
    timings = time_callable(lambda: torch.matmul(a, b, out=out), repeats, warmups)
    return summarise(
        op="matmul",
        dtype=str(dtype).replace("torch.", ""),
        m=m,
        n=n,
        k=k,
        warmups=warmups,
        repeats=repeats,
        timings_ms=timings,
    )


def run_int8_case(m: int, n: int, k: int, repeats: int, warmups: int) -> GemmResult | None:
    if not hasattr(torch, "_int_mm"):
        return None
    a = torch.randint(-128, 127, (m, k), device="xpu", dtype=torch.int8)
    b = torch.randint(-128, 127, (k, n), device="xpu", dtype=torch.int8)
    synchronize()
    timings = time_callable(lambda: torch._int_mm(a, b), repeats, warmups)
    return summarise(
        op="int_mm",
        dtype="int8",
        m=m,
        n=n,
        k=k,
        warmups=warmups,
        repeats=repeats,
        timings_ms=timings,
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
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--json-out", type=str)
    args = parser.parse_args()

    if not torch.xpu.is_available():
        raise SystemExit("torch.xpu is not available")

    results: list[GemmResult] = []
    parsed_cases = [parse_case(text) for text in args.cases]
    for dtype_name in args.dtypes:
        for m, n, k in parsed_cases:
            if dtype_name == "int8":
                result = run_int8_case(m, n, k, args.repeats, args.warmups)
                if result is not None:
                    results.append(result)
                continue
            dtype = getattr(torch, dtype_name)
            results.append(run_float_case(m, n, k, dtype, args.repeats, args.warmups))

    header = (
        f"{'op':<10} {'dtype':<10} {'m':>6} {'n':>6} {'k':>6} "
        f"{'median_ms':>12} {'mean_ms':>12} {'median':>10} {'max':>10} {'unit':>8}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result.op:<10} {result.dtype:<10} {result.m:>6} {result.n:>6} {result.k:>6} "
            f"{result.median_ms:>12.3f} {result.mean_ms:>12.3f} "
            f"{result.median_throughput:>10.2f} {result.max_throughput:>10.2f} {result.unit:>8}"
        )

    if args.json_out:
        payload = {
            "torch": torch.__version__,
            "device": torch.xpu.get_device_name(0),
            "repeats": args.repeats,
            "warmups": args.warmups,
            "cases": args.cases,
            "results": [asdict(result) for result in results],
        }
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
