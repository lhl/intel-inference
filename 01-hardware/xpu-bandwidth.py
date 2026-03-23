#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass

import torch


@dataclass
class Result:
    test: str
    size_mib: int
    dtype: str
    bytes_moved: int
    repeats: int
    warmups: int
    median_ms: float
    mean_ms: float
    max_gbps: float
    median_gbps: float
    mean_gbps: float


def synchronize() -> None:
    torch.xpu.synchronize()


def time_callable(fn, repeats: int, warmups: int) -> list[float]:
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


def throughput_gbps(bytes_moved: int, elapsed_ms: float) -> float:
    return bytes_moved / (elapsed_ms / 1e3) / 1e9


def summarise(
    *,
    test: str,
    size_mib: int,
    dtype: torch.dtype,
    bytes_moved: int,
    repeats: int,
    warmups: int,
    timings_ms: list[float],
) -> Result:
    gbps = [throughput_gbps(bytes_moved, t) for t in timings_ms]
    return Result(
        test=test,
        size_mib=size_mib,
        dtype=str(dtype).replace("torch.", ""),
        bytes_moved=bytes_moved,
        repeats=repeats,
        warmups=warmups,
        median_ms=statistics.median(timings_ms),
        mean_ms=statistics.mean(timings_ms),
        max_gbps=max(gbps),
        median_gbps=statistics.median(gbps),
        mean_gbps=statistics.mean(gbps),
    )


def benchmark_copy(size_mib: int, repeats: int, warmups: int) -> list[Result]:
    num_bytes = size_mib * 1024 * 1024
    src_host = torch.empty(num_bytes, dtype=torch.uint8, pin_memory=True)
    dst_host = torch.empty(num_bytes, dtype=torch.uint8, pin_memory=True)
    src_dev = torch.empty(num_bytes, dtype=torch.uint8, device="xpu")
    dst_dev = torch.empty(num_bytes, dtype=torch.uint8, device="xpu")

    src_host.random_(0, 256)
    src_dev.random_(0, 256)
    synchronize()

    results = []

    timings = time_callable(lambda: dst_dev.copy_(src_dev), repeats, warmups)
    results.append(
        summarise(
            test="d2d_copy",
            size_mib=size_mib,
            dtype=torch.uint8,
            bytes_moved=num_bytes,
            repeats=repeats,
            warmups=warmups,
            timings_ms=timings,
        )
    )

    timings = time_callable(lambda: src_dev.copy_(src_host, non_blocking=True), repeats, warmups)
    results.append(
        summarise(
            test="h2d_copy",
            size_mib=size_mib,
            dtype=torch.uint8,
            bytes_moved=num_bytes,
            repeats=repeats,
            warmups=warmups,
            timings_ms=timings,
        )
    )

    timings = time_callable(lambda: dst_host.copy_(src_dev, non_blocking=True), repeats, warmups)
    results.append(
        summarise(
            test="d2h_copy",
            size_mib=size_mib,
            dtype=torch.uint8,
            bytes_moved=num_bytes,
            repeats=repeats,
            warmups=warmups,
            timings_ms=timings,
        )
    )

    return results


def benchmark_vector_add(size_mib: int, repeats: int, warmups: int, dtype: torch.dtype) -> Result:
    elem_size = torch.empty((), dtype=dtype).element_size()
    numel = (size_mib * 1024 * 1024) // elem_size
    a = torch.randn(numel, device="xpu", dtype=dtype)
    b = torch.randn(numel, device="xpu", dtype=dtype)
    out = torch.empty(numel, device="xpu", dtype=dtype)
    synchronize()

    timings = time_callable(lambda: torch.add(a, b, out=out), repeats, warmups)
    bytes_moved = 3 * numel * elem_size
    return summarise(
        test="vector_add_effective",
        size_mib=size_mib,
        dtype=dtype,
        bytes_moved=bytes_moved,
        repeats=repeats,
        warmups=warmups,
        timings_ms=timings,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure basic XPU memory bandwidth with PyTorch XPU.")
    parser.add_argument("--sizes-mib", nargs="+", type=int, default=[64, 128, 256, 512])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--vector-dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--json-out", type=str)
    args = parser.parse_args()

    if not torch.xpu.is_available():
        raise SystemExit("torch.xpu is not available")

    dtype = getattr(torch, args.vector_dtype)
    results: list[Result] = []
    for size_mib in args.sizes_mib:
        results.extend(benchmark_copy(size_mib, args.repeats, args.warmups))
        results.append(benchmark_vector_add(size_mib, args.repeats, args.warmups, dtype))

    header = (
        f"{'test':<22} {'size_mib':>8} {'dtype':>8} {'median_ms':>12} "
        f"{'mean_ms':>12} {'median_gbps':>14} {'max_gbps':>12}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result.test:<22} {result.size_mib:>8} {result.dtype:>8} "
            f"{result.median_ms:>12.3f} {result.mean_ms:>12.3f} "
            f"{result.median_gbps:>14.2f} {result.max_gbps:>12.2f}"
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
