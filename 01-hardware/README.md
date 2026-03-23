# 01-hardware

This directory is for low-level hardware characterization before framework or model-level testing.

The point of this stage is to establish ceilings and bottlenecks:

- host memory bandwidth
- GPU-accessible memory bandwidth
- NPU-relevant bandwidth or transfer limits if measurable
- GEMM or FLOPs-style compute throughput
- thermal, clock, and power behavior during sustained load

## What belongs here

- raw memory-bandwidth microbenchmarks
- raw compute microbenchmarks
- device-transfer tests
- telemetry capture
- notes on which tools are credible and which are misleading
- saved raw outputs and summarized results

## Current script set

The first practical pass for this repo is:

- `run-host-mbw.sh`
  - host memory bandwidth using `mbw` and `sysbench`
  - intentionally does not use `/usr/bin/stream`, because on this Arch machine that binary is ImageMagick, not the STREAM benchmark
- `run-xpu-bandwidth.sh`
  - PyTorch XPU bandwidth checks for:
    - host-to-device copy
    - device-to-host copy
    - device-to-device copy
    - vector-add effective bandwidth
- `run-mamf-finder.sh`
  - a `mamf-finder`-style PyTorch XPU GEMM sweep
  - float dtypes report `TFLOPS`
  - int8 reports `TOPS` via `torch._int_mm` when that kernel path exists
- `run-npu-microbench.sh`
  - synthetic OpenVINO NPU compile and infer microbench for static-shape graphs
  - current focus is compile latency, first-token-equivalent cold-start behavior, warm infer latency, and rough effective throughput on simple graph shapes
- `collect-intel-gpu-top.sh`
  - captures `intel_gpu_top` JSON, optionally while another benchmark command runs
  - falls back to an explicit `unsupported` JSON payload if `intel_gpu_top` cannot read usable engine data on the current driver stack
- `run-suite.sh`
  - ties the host and GPU first-wave scripts together
  - NPU remains separate because it is device-specific and currently needs the Arch loader-path workaround on this machine

## Current measurement stance

For now:

- host MBW is measured with host tools
- GPU MBW is measured through the clean `torch+xpu` stack, not a custom SYCL microbench yet
- GPU compute is measured through square GEMM sweeps in the clean `torch+xpu` env
- NPU now has a first synthetic OpenVINO microbench path, but only for static-shape compile and infer testing
- NPU bandwidth still does not have a credible first-class microbench here; the current NPU numbers are compile and inference measurements with only rough effective throughput estimates
- GPU telemetry is currently best-effort only; on this xe/Lunar Lake machine `intel_gpu_top` may fail to expose usable PMU engine counters

That means this directory is a real benchmark layer now, but still not the final word on pure hardware peak. The current numbers should be treated as:

- host-tool MBW
- XPU-runtime-visible copy bandwidth
- XPU-runtime-visible GEMM throughput
- NPU compile and warm-infer behavior for simple static-shape OpenVINO graphs

not as exhaustive architectural limits

## Benchmark environments

This phase intentionally mixes system tools and pinned Python envs. The benchmark README should always state which is which.

- `run-host-mbw.sh`
  - no `mamba` env
  - uses system tools such as `mbw` and `sysbench`
- `run-xpu-bandwidth.sh`
  - `intel-inf-torch-xpu`
- `run-mamf-finder.sh`
  - `intel-inf-torch-xpu`
- `run-npu-microbench.sh`
  - `intel-inf-openvino`
  - requires `source ./00-setup/npu-env.sh` on this Arch machine so OpenVINO can enumerate `NPU`
- `collect-intel-gpu-top.sh`
  - no `mamba` env
  - uses the system `intel_gpu_top` tool
- `run-suite.sh`
  - mixes the above and should be read as a multi-env wrapper, not one single-env benchmark

## Current measured results

The current fuller host and GPU pass is from `./01-hardware/run-suite.sh` on the tracked Lunar Lake machine:

- system profile:
  - [lunarlake-ultra7-258v-32gb.md](/home/lhl/github/lhl/intel-inference/00-setup/systems/lunarlake-ultra7-258v-32gb.md)
- run date:
  - March 23, 2026

Headline host and GPU results:

- host memory bandwidth:
  - `mbw memcpy`: `17771.1 MiB/s` average over the current full pass
  - `mbw dumb`: `9760.7 MiB/s`
  - `mbw fixed-block memcpy`: `8561.5 MiB/s`
  - `sysbench write`: `57315.6 MiB/s`
  - `sysbench read`: `197848.7 MiB/s`
- XPU copy bandwidth:
  - H2D median: about `45.5` to `53.7 GB/s`
  - D2H median: about `51.2` to `54.1 GB/s`
  - D2D median: about `51.6` to `54.1 GB/s`
  - vector-add effective bandwidth: about `92.8` to `106.6 GB/s`
- XPU GEMM throughput:
  - `float32`: about `2.4 TFLOPS` at `1024`, `3.4 TFLOPS` at `2048`, `3.9 TFLOPS` at `4096`
  - `bfloat16`: about `10.5 TFLOPS` at `1024`, `24.4 TFLOPS` at `2048`, `22.7 TFLOPS` at `4096`
  - `float16`: about `10.5 TFLOPS` at `1024`, `24.6 TFLOPS` at `2048`, `21.9 TFLOPS` at `4096`
  - `int8`: about `12.8 TOPS` at `1024`, `26.8 TOPS` at `2048`, `41.1 TOPS` at `4096`, with `46.0 TOPS` best observed

The current initial NPU pass is from `./01-hardware/run-npu-microbench.sh`:

- NPU compile and infer:
  - static-shape `float16` OpenVINO matmul graphs compiled successfully on `NPU`
  - cold compile: about `94` to `117 ms`
  - warm compile with cache reuse: about `4` to `7 ms`
  - first infer after compile: about `191` to `664 ms`
  - warm infer median:
    - `16.0 ms` at `256`
    - `25.0 ms` at `512`
    - `51.7 ms` at `1024`
  - rough effective throughput from this synthetic matmul test:
    - about `0.002 TOPS` at `256`
    - about `0.011 TOPS` at `512`
    - about `0.042 TOPS` at `1024`

Current caveats:

- these are still first-pass benchmark numbers, not full thermal-soak or sustained-power characterizations
- `sysbench` and `mbw` are not directly interchangeable; they use different access patterns
- the current GPU bandwidth path is through `torch+xpu`, so these are runtime-visible transfer rates rather than a pure driver-level or SYCL-native peak
- the current NPU figures come from synthetic static-shape OpenVINO graphs and should be treated as path validation plus a rough lower-bound throughput estimate, not as a statement of peak NPU capability or LLM performance
- the NPU microbench is currently matmul-centric; it does not yet capture attention-heavy or decode-heavy inference behavior
- `intel_gpu_top` is still not usable on this `xe` stack here; the current telemetry result is `unsupported` because engine counters could not be detected

## Exit criteria

We should not treat framework-level numbers as meaningful until we have:

1. at least one repeatable system-memory bandwidth measurement
2. at least one repeatable GPU-focused bandwidth measurement
3. a basic compute microbench path for the GPU
4. a record of clocks, power mode, and thermal conditions during those runs
5. clear notes on what is measured and what is only inferred

## Expected work products

Current or expected files here:

- `common.sh`
- `run-host-mbw.sh`
- `xpu-bandwidth.py`
- `run-xpu-bandwidth.sh`
- `mamf-finder-xpu.py`
- `run-mamf-finder.sh`
- `openvino-npu-microbench.py`
- `run-npu-microbench.sh`
- `collect-intel-gpu-top.sh`
- `run-suite.sh`
- `results/`

## Related docs

- [RESEARCH-hardware.md](/home/lhl/github/lhl/intel-inference/RESEARCH-hardware.md)
- [TESTING.md](/home/lhl/github/lhl/intel-inference/TESTING.md)
