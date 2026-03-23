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
- `collect-intel-gpu-top.sh`
  - captures `intel_gpu_top` JSON, optionally while another benchmark command runs
  - falls back to an explicit `unsupported` JSON payload if `intel_gpu_top` cannot read usable engine data on the current driver stack
- `run-suite.sh`
  - ties the first-wave scripts together

## Current measurement stance

For now:

- host MBW is measured with host tools
- GPU MBW is measured through the clean `torch+xpu` stack, not a custom SYCL microbench yet
- GPU compute is measured through square GEMM sweeps in the clean `torch+xpu` env
- NPU bandwidth and compute are still not first-class here; those remain a later pass
- GPU telemetry is currently best-effort only; on this xe/Lunar Lake machine `intel_gpu_top` may fail to expose usable PMU engine counters

That means this directory is a real benchmark layer now, but still not the final word on pure hardware peak. The current numbers should be treated as:

- host-tool MBW
- XPU-runtime-visible copy bandwidth
- XPU-runtime-visible GEMM throughput

not as exhaustive architectural limits

## Current quick-pass results

The current quick validation run is from `./01-hardware/run-suite.sh --quick` on the tracked Lunar Lake machine:

- system profile:
  - [lunarlake-ultra7-258v-32gb.md](/home/lhl/github/lhl/intel-inference/00-setup/systems/lunarlake-ultra7-258v-32gb.md)
- run date:
  - March 23, 2026

Headline results:

- host memory bandwidth:
  - `mbw memcpy`: `16245.6 MiB/s`
  - `mbw dumb`: `9241.0 MiB/s`
  - `mbw fixed-block memcpy`: `8658.7 MiB/s`
  - `sysbench write`: `50187.7 MiB/s`
  - `sysbench read`: `118441.4 MiB/s`
- XPU copy bandwidth:
  - H2D median: about `52.9` to `54.4 GB/s`
  - D2H median: about `47.4` to `53.5 GB/s`
  - D2D median: about `52.2` to `54.2 GB/s`
  - vector-add effective bandwidth: about `92.5` to `107.1 GB/s`
- XPU GEMM throughput:
  - `float32`: about `2.1` to `3.4 TFLOPS` median
  - `bfloat16`: about `10.6 TFLOPS` at `1024`, `24.6 TFLOPS` at `2048`
  - `float16`: quick-pass median ranged from about `10.4` to `13.8 TFLOPS`, with higher peak outliers
  - `int8`: about `9.8 TOPS` at `1024`, `25.0 TOPS` median at `2048`, `34.9 TOPS` best observed

Current caveats:

- these are quick-pass validation numbers, not long-duration sustained peaks
- `sysbench` and `mbw` are not directly interchangeable; they use different access patterns
- the current GPU bandwidth path is through `torch+xpu`, so these are runtime-visible transfer rates rather than a pure driver-level or SYCL-native peak
- `float16` GEMM showed more variance than `bfloat16` in the quick pass and needs deeper repeat testing
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
- `collect-intel-gpu-top.sh`
- `run-suite.sh`
- `results/`

## Related docs

- [RESEARCH-hardware.md](/home/lhl/github/lhl/intel-inference/RESEARCH-hardware.md)
- [TESTING.md](/home/lhl/github/lhl/intel-inference/TESTING.md)
