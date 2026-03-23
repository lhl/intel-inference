# 02-operators

This directory is for operator-level testing on top of the validated `torch+xpu` and OpenVINO setup from earlier phases.

The point of this stage is to answer:

- which PyTorch XPU operator paths actually exist
- which dtypes are usable
- which SDPA backends are real versus nominal
- how transformer-shaped GEMMs behave before we involve full models

## What belongs here

- backend capability checks
- GEMM and linear benchmark sweeps
- attention and SDPA benchmark sweeps
- quantized operator checks where the kernel path is real
- saved raw outputs and summarized results

## Current script set

The first practical pass for this repo is:

- `run-backend-check.sh`
  - checks PyTorch XPU operator and SDPA backend availability
  - records dtype smoke results for `matmul`, `int8_mm`, and default SDPA
- `run-gemm-bench.sh`
  - benchmarks transformer-shaped GEMM cases on `torch+xpu`
  - focuses on realistic `MxNxK` linear-layer shapes, not just square matrices
- `run-batched-gemm-bench.sh`
  - benchmarks batched transformer-shaped GEMM cases on `torch+xpu`
  - covers more realistic inference batching than plain 2D matmul alone
- `run-attention-bench.sh`
  - benchmarks SDPA forward on `torch+xpu`
  - compares `DEFAULT`, `MATH`, `OVERRIDEABLE`, and nominal optimized backends where available
  - can compare `eager` and `torch.compile` variants across causal and non-causal SDPA
- `run-suite.sh`
  - ties the initial backend check, 2D GEMM sweep, batched GEMM sweep, and attention sweep together

## Current measurement stance

For now:

- this phase is PyTorch XPU first
- GEMM and attention are measured as inference-side forward operators
- both `eager` and `torch.compile` paths are now in scope
- backend forcing results are treated as ground truth for what PyTorch XPU exposes on this machine
- the current attention results are for SDPA forward only, with both causal and non-causal coverage, not backward or training
- OpenVINO operator work remains a later pass

That means this directory is currently the operator bridge between `01-hardware` and later full-runtime/model benchmarking, not the final framework-comparison layer.

## Current quick-pass results

The current initial validation run is from `./02-operators/run-suite.sh --quick` on the tracked Lunar Lake machine:

- system profile:
  - [lunarlake-ultra7-258v-32gb.md](/home/lhl/github/lhl/intel-inference/00-setup/systems/lunarlake-ultra7-258v-32gb.md)
- run date:
  - March 23, 2026

Headline results:

- backend and dtype support:
  - `torch 2.10.0+xpu` sees the XPU normally
  - `matmul` worked in `float32`, `bfloat16`, and `float16`
  - `torch._int_mm` worked in `int8`
  - default SDPA worked in `float32`, `bfloat16`, and `float16`
  - `torch.compile` worked for:
    - `float16` matmul
    - `int8` `torch._int_mm`
    - `float16` SDPA
- forced SDPA backend results:
  - `OVERRIDEABLE`: works
  - `MATH`: works
  - `FLASH_ATTENTION`: does not work
  - `EFFICIENT_ATTENTION`: does not work
  - `CUDNN_ATTENTION`: does not work
  - `can_use_flash_attention` and `can_use_efficient_attention` both reported `False`
- operator-level GEMM throughput:
  - eager 2D `bfloat16` and `float16` transformer-style GEMMs at `128x4096x11008` and `128x11008x4096` landed around `11.6` to `11.8 TFLOPS`
  - eager 2D `float16` `128x4096x4096` landed around `10.9 TFLOPS`
  - eager 2D `int8` landed around `15.7 TOPS` for `128x4096x4096` and about `21.1 TOPS` for `128x4096x11008`
  - compile 2D GEMM worked, but in this quick pass it was not a clear win:
    - `float16` `128x4096x11008`: about `11.4 TFLOPS`
    - `bfloat16` `128x11008x4096`: about `11.2 TFLOPS`
    - `int8` compile variants were notably worse than eager in the current quick pass
- batched GEMM throughput:
  - eager batched `bfloat16` and `float16` at `4x128x4096x4096` landed around `12.1 TFLOPS`
  - eager batched `float16` `4x128x4096x11008` landed around `12.2 TFLOPS`
  - compile batched `bfloat16` and `float16` landed around `11.8` to `12.0 TFLOPS`, so again not a clear quick-pass win
- operator-level attention throughput:
  - at `1x8x512x128`, `OVERRIDEABLE` was clearly faster than `MATH`
  - eager `float16` causal SDPA:
    - `OVERRIDEABLE`: about `0.143 ms`, `3.77 TFLOPS`
    - `MATH`: about `1.165 ms`, `0.46 TFLOPS`
  - eager `float16` non-causal SDPA:
    - `OVERRIDEABLE`: about `0.154 ms`, `6.95 TFLOPS`
    - `MATH`: about `1.133 ms`, `0.95 TFLOPS`
  - eager `bfloat16` non-causal SDPA:
    - `OVERRIDEABLE`: about `0.182 ms`, `5.91 TFLOPS`
    - `MATH`: about `1.098 ms`, `0.98 TFLOPS`
  - compile SDPA worked for `DEFAULT`, `MATH`, and `OVERRIDEABLE`, but the current quick pass did not show a stable steady-state advantage over eager `OVERRIDEABLE`

Current interpretation:

- on this Lunar Lake `64A0` XPU stack, PyTorch XPU SDPA is real, but the useful paths are currently `OVERRIDEABLE` and `MATH`
- the backend-check warning explicitly says current XPU flash attention support is for `intel_gpu_pvc`, `intel_gpu_pvc_vg`, and `intel_gpu_bmg_g21`, not this machine
- `OVERRIDEABLE` looks like the operator path that matters for current Intel client-GPU SDPA behavior here
- `torch.compile` is usable on this stack, but in the current quick operator pass it is best treated as something to test, not something to assume is faster
- the current GEMM numbers are below the raw square-GEMM peaks from `01-hardware`, which is expected because these are more model-shaped operator cases rather than peak-friendly square sweeps

Current caveats:

- these are quick-pass numbers, not the final full operator characterization
- attention results are forward-only; backward and training-side behavior are still untested
- non-causal attention is now included, but deeper mode coverage and larger shape coverage are still to come
- first-call timings for compile variants are not yet a clean isolated compile-cost metric because compiler cache reuse can affect later cases in the same run
- backend availability is machine-specific and stack-specific; do not generalize these results to Battlemage or datacenter Intel GPUs yet

## Expected work products

Current or expected files here:

- `common.sh`
- `backend-check.py`
- `run-backend-check.sh`
- `gemm-bench.py`
- `run-gemm-bench.sh`
- `batched-gemm-bench.py`
- `run-batched-gemm-bench.sh`
- `attention-bench.py`
- `run-attention-bench.sh`
- `run-suite.sh`
- `results/`

## Related docs

- [TESTING.md](/home/lhl/github/lhl/intel-inference/TESTING.md)
- [01-hardware/README.md](/home/lhl/github/lhl/intel-inference/01-hardware/README.md)
