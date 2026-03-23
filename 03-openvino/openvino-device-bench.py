#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import openvino as ov
from openvino import opset13 as ops


@dataclass
class BenchResult:
    model_kind: str
    device: str
    shape: int
    dtype: str
    status: str
    detail: str
    op_count: int | None
    cache_files: int | None
    warmups: int
    repeats: int
    compile_cold_ms: float | None
    compile_warm_ms: float | None
    first_infer_ms: float | None
    median_infer_ms: float | None
    mean_infer_ms: float | None
    min_infer_ms: float | None
    max_infer_ms: float | None
    median_tops: float | None
    max_tops: float | None


def tops_for_time(op_count: int, elapsed_ms: float) -> float:
    return op_count / (elapsed_ms / 1e3) / 1e12


def build_model(model_kind: str, shape: int, dtype: ov.Type) -> tuple[ov.Model, int]:
    rng = np.random.default_rng(1234)
    x = ops.parameter([shape, shape], dtype, name="x")

    if model_kind == "matmul":
        weight = ops.constant(rng.standard_normal((shape, shape), dtype=np.float32).astype(np.float16))
        y = ops.matmul(x, weight, False, False)
        op_count = 2 * shape * shape * shape
        return ov.Model([y], [x], f"{model_kind}_{shape}"), op_count

    if model_kind == "mlp":
        hidden = shape * 4
        w1 = ops.constant(rng.standard_normal((shape, hidden), dtype=np.float32).astype(np.float16))
        b1 = ops.constant(rng.standard_normal((hidden,), dtype=np.float32).astype(np.float16))
        w2 = ops.constant(rng.standard_normal((hidden, shape), dtype=np.float32).astype(np.float16))
        h = ops.matmul(x, w1, False, False)
        h = ops.add(h, b1)
        h = ops.gelu(h, "erf")
        y = ops.matmul(h, w2, False, False)
        op_count = 16 * shape * shape * shape
        return ov.Model([y], [x], f"{model_kind}_{shape}"), op_count

    raise ValueError(f"unsupported model kind: {model_kind}")


def compile_model(model: ov.Model, device: str, cache_dir: Path) -> tuple[ov.CompiledModel, float]:
    core = ov.Core()
    core.set_property({"CACHE_DIR": str(cache_dir)})
    start = time.perf_counter_ns()
    compiled = core.compile_model(model, device)
    elapsed_ms = (time.perf_counter_ns() - start) / 1e6
    return compiled, elapsed_ms


def time_infer(
    compiled: ov.CompiledModel,
    input_name: str,
    input_value: np.ndarray,
    repeats: int,
    warmups: int,
) -> tuple[float, list[float]]:
    request = compiled.create_infer_request()

    start = time.perf_counter_ns()
    request.infer({input_name: input_value})
    first_infer_ms = (time.perf_counter_ns() - start) / 1e6

    for _ in range(warmups):
        request.infer({input_name: input_value})

    timings_ms: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        request.infer({input_name: input_value})
        timings_ms.append((time.perf_counter_ns() - start) / 1e6)

    return first_infer_ms, timings_ms


def skip_result(model_kind: str, device: str, shape: int, warmups: int, repeats: int, detail: str) -> BenchResult:
    return BenchResult(
        model_kind=model_kind,
        device=device,
        shape=shape,
        dtype="float16",
        status="SKIP",
        detail=detail,
        op_count=None,
        cache_files=None,
        warmups=warmups,
        repeats=repeats,
        compile_cold_ms=None,
        compile_warm_ms=None,
        first_infer_ms=None,
        median_infer_ms=None,
        mean_infer_ms=None,
        min_infer_ms=None,
        max_infer_ms=None,
        median_tops=None,
        max_tops=None,
    )


def benchmark_case(
    *,
    available_devices: list[str],
    model_kind: str,
    shape: int,
    device: str,
    repeats: int,
    warmups: int,
) -> BenchResult:
    if device not in available_devices:
        return skip_result(model_kind, device, shape, warmups, repeats, f"device not available: {available_devices}")

    model, op_count = build_model(model_kind, shape, ov.Type.f16)
    input_name = model.inputs[0].get_any_name()
    input_value = np.random.default_rng(4321).standard_normal((shape, shape), dtype=np.float32).astype(np.float16)

    cache_dir = Path(tempfile.mkdtemp(prefix=f"ov-cache-{device}-{model_kind}-{shape}-"))
    try:
        compiled_cold, compile_cold_ms = compile_model(model, device, cache_dir)
        cache_files = sum(1 for path in cache_dir.rglob("*") if path.is_file())
        _, compile_warm_ms = compile_model(model, device, cache_dir)
        first_infer_ms, timings_ms = time_infer(compiled_cold, input_name, input_value, repeats, warmups)
    except Exception as exc:
        return skip_result(model_kind, device, shape, warmups, repeats, str(exc).strip().splitlines()[0][:240])
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)

    scores = [tops_for_time(op_count, elapsed_ms) for elapsed_ms in timings_ms]
    return BenchResult(
        model_kind=model_kind,
        device=device,
        shape=shape,
        dtype="float16",
        status="OK",
        detail="",
        op_count=op_count,
        cache_files=cache_files,
        warmups=warmups,
        repeats=repeats,
        compile_cold_ms=compile_cold_ms,
        compile_warm_ms=compile_warm_ms,
        first_infer_ms=first_infer_ms,
        median_infer_ms=statistics.median(timings_ms),
        mean_infer_ms=statistics.mean(timings_ms),
        min_infer_ms=min(timings_ms),
        max_infer_ms=max(timings_ms),
        median_tops=statistics.median(scores),
        max_tops=max(scores),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark simple OpenVINO graphs across devices")
    parser.add_argument("--devices", nargs="+", default=["CPU", "GPU", "NPU"])
    parser.add_argument("--model-kinds", nargs="+", choices=["matmul", "mlp"], default=["matmul"])
    parser.add_argument("--shapes", nargs="+", type=int, default=[256, 512, 1024])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--json-out", type=str)
    args = parser.parse_args()

    core = ov.Core()
    available_devices = list(core.available_devices)

    results: list[BenchResult] = []
    for device in args.devices:
        for model_kind in args.model_kinds:
            for shape in args.shapes:
                results.append(
                    benchmark_case(
                        available_devices=available_devices,
                        model_kind=model_kind,
                        shape=shape,
                        device=device,
                        repeats=args.repeats,
                        warmups=args.warmups,
                    )
                )

    print("openvino", ov.__version__)
    print("available_devices", available_devices)
    header = (
        f"{'device':<6} {'model':<8} {'shape':>8} {'status':<6} "
        f"{'compile_cold_ms':>16} {'compile_warm_ms':>16} {'first_ms':>12} "
        f"{'median_ms':>12} {'median':>10} detail"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        compile_cold = "-" if result.compile_cold_ms is None else f"{result.compile_cold_ms:.3f}"
        compile_warm = "-" if result.compile_warm_ms is None else f"{result.compile_warm_ms:.3f}"
        first_ms = "-" if result.first_infer_ms is None else f"{result.first_infer_ms:.3f}"
        median_ms = "-" if result.median_infer_ms is None else f"{result.median_infer_ms:.3f}"
        median_tops = "-" if result.median_tops is None else f"{result.median_tops:.3f}"
        print(
            f"{result.device:<6} {result.model_kind:<8} {result.shape:>8} {result.status:<6} "
            f"{compile_cold:>16} {compile_warm:>16} {first_ms:>12} {median_ms:>12} {median_tops:>10} {result.detail}"
        )

    if args.json_out:
        payload = {
            "openvino": ov.__version__,
            "available_devices": available_devices,
            "devices": args.devices,
            "model_kinds": args.model_kinds,
            "shapes": args.shapes,
            "repeats": args.repeats,
            "warmups": args.warmups,
            "results": [asdict(result) for result in results],
        }
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
