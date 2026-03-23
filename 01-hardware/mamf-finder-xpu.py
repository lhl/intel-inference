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
    dtype: str
    mode: str
    m: int
    n: int
    k: int
    warmups: int
    repeats: int
    mean_ms: float
    median_ms: float
    min_ms: float
    mean_tops: float
    median_tops: float
    max_tops: float


def synchronize() -> None:
    torch.xpu.synchronize()


def tops_for_time(op_count: int, elapsed_ms: float) -> float:
    return op_count / (elapsed_ms / 1e3) / 1e12


def bench(fn, repeats: int, warmups: int) -> list[float]:
    for _ in range(warmups):
        fn()
    synchronize()
    timings_ms: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        fn()
        synchronize()
        end = time.perf_counter_ns()
        timings_ms.append((end - start) / 1e6)
    return timings_ms


def make_result(
    *,
    dtype: str,
    mode: str,
    m: int,
    n: int,
    k: int,
    warmups: int,
    repeats: int,
    timings_ms: list[float],
) -> GemmResult:
    op_count = 2 * m * n * k
    tops = [tops_for_time(op_count, t) for t in timings_ms]
    return GemmResult(
        dtype=dtype,
        mode=mode,
        m=m,
        n=n,
        k=k,
        warmups=warmups,
        repeats=repeats,
        mean_ms=statistics.mean(timings_ms),
        median_ms=statistics.median(timings_ms),
        min_ms=min(timings_ms),
        mean_tops=statistics.mean(tops),
        median_tops=statistics.median(tops),
        max_tops=max(tops),
    )


def run_float_gemm(shape: int, dtype: torch.dtype, repeats: int, warmups: int) -> GemmResult:
    a = torch.randn((shape, shape), device="xpu", dtype=dtype)
    b = torch.randn((shape, shape), device="xpu", dtype=dtype)
    out = torch.empty((shape, shape), device="xpu", dtype=dtype)
    synchronize()

    timings = bench(lambda: torch.matmul(a, b, out=out), repeats, warmups)
    return make_result(
        dtype=str(dtype).replace("torch.", ""),
        mode="float_mm",
        m=shape,
        n=shape,
        k=shape,
        warmups=warmups,
        repeats=repeats,
        timings_ms=timings,
    )


def run_int8_gemm(shape: int, repeats: int, warmups: int) -> GemmResult | None:
    if not hasattr(torch, "_int_mm"):
        return None

    a = torch.randint(-128, 127, (shape, shape), device="xpu", dtype=torch.int8)
    b = torch.randint(-128, 127, (shape, shape), device="xpu", dtype=torch.int8)
    synchronize()

    timings = bench(lambda: torch._int_mm(a, b), repeats, warmups)
    return make_result(
        dtype="int8",
        mode="int_mm",
        m=shape,
        n=shape,
        k=shape,
        warmups=warmups,
        repeats=repeats,
        timings_ms=timings,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="mamf-finder-style GEMM sweep for PyTorch XPU")
    parser.add_argument("--shapes", nargs="+", type=int, default=[1024, 2048, 4096])
    parser.add_argument("--dtypes", nargs="+", default=["float32", "bfloat16", "float16", "int8"])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--json-out", type=str)
    args = parser.parse_args()

    if not torch.xpu.is_available():
        raise SystemExit("torch.xpu is not available")

    results: list[GemmResult] = []
    for dtype_name in args.dtypes:
        for shape in args.shapes:
            if dtype_name == "int8":
                result = run_int8_gemm(shape, args.repeats, args.warmups)
                if result is not None:
                    results.append(result)
                continue

            dtype = getattr(torch, dtype_name)
            results.append(run_float_gemm(shape, dtype, args.repeats, args.warmups))

    header = (
        f"{'dtype':<10} {'mode':<9} {'shape':>8} {'median_ms':>12} "
        f"{'mean_ms':>12} {'median':>10} {'max':>10}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        unit = "TOPS" if result.dtype == "int8" else "TFLOPS"
        print(
            f"{result.dtype:<10} {result.mode:<9} {result.m:>8} "
            f"{result.median_ms:>12.3f} {result.mean_ms:>12.3f} "
            f"{result.median_tops:>10.2f} {result.max_tops:>10.2f} {unit}"
        )

    if args.json_out:
        payload = {
            "torch": torch.__version__,
            "device": torch.xpu.get_device_name(0),
            "repeats": args.repeats,
            "warmups": args.warmups,
            "results": [asdict(r) for r in results],
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")


if __name__ == "__main__":
    main()
