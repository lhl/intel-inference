# Intel AI/ML Toolchain Research

Research and documentation for the AI/ML inference toolchain on Intel hardware (Linux), with a focus on Intel Arc GPUs — specifically Lunar Lake (Ultra 7 258V, 32 GB shared memory).

## Test hardware

| Component | Detail |
|-----------|--------|
| CPU | Intel Core Ultra 7 258V (Lunar Lake) |
| GPU | Intel Arc integrated (Xe2 LNL), shared memory |
| RAM | 32 GB |
| OS | Arch Linux, kernel 7.0.0-rc3 |
| NPU | Intel AI Boost NPU |

## Performance results

All results below are from this single Lunar Lake machine. Models that could not be tested due to toolchain version constraints are noted in [Model coverage gaps](#model-coverage-gaps).

### llama.cpp — Vulkan (AC power)

`llama-bench -fa 1`, 5 runs, pp512/tg128. TTY = plain console, WM = desktop session.

| Model | Quant | Prompt (TTY) | Gen (TTY) | Prompt (WM) | Gen (WM) |
|-------|-------|-------------:|----------:|------------:|----------:|
| Llama-3.2-1B-Instruct | BF16 | 1112 tok/s | 36.3 tok/s | 924 tok/s | 31.9 tok/s |
| Llama-3.2-1B-Instruct | Q4_K_XL | 2063 tok/s | 48.5 tok/s | 1705 tok/s | 42.0 tok/s |
| LFM2.5-1.2B-Instruct | BF16 | 1148 tok/s | 38.3 tok/s | 930 tok/s | 34.1 tok/s |
| LFM2.5-1.2B-Instruct | Q4_K_XL | 2222 tok/s | 59.9 tok/s | 1790 tok/s | 50.4 tok/s |

TTY is consistently ~15–20% faster than a desktop session, likely due to reduced GPU contention from the compositor.

### OpenVINO GenAI — GPU serving (OpenAI-compatible API)

| Model | Median TTFT | Median total latency | Median gen throughput |
|-------|------------:|---------------------:|----------------------:|
| Llama-3.2-1B-Instruct | 157.8 ms | 1955 ms | 32.2 tok/s |
| LFM2-1.2B | 96.6 ms | 3048 ms | 25.8 tok/s |

### OpenVINO GenAI — Whisper

| Model | Device | Latency |
|-------|--------|--------:|
| whisper-large-v3-turbo | GPU | 954 ms |
| whisper-large-v3-turbo | NPU | 614 ms |
| whisper-large-v3 | GPU | 1890 ms |

### OpenVINO GenAI — NPU LLM

| Model | Device | Latency |
|-------|--------|--------:|
| Llama-3.2-1B-Instruct | NPU | 867 ms |

### vLLM — OpenVINO backend (GPU)

| Model | Median TTFT | Median total latency | Median gen throughput | Status |
|-------|------------:|---------------------:|----------------------:|--------|
| Llama-3.2-1B-Instruct | 53 ms | 1990 ms | 28.1 tok/s | Single successful run; rerun failed during OpenVINO export |

### vLLM — upstream XPU

| Model | Status | Detail |
|-------|--------|--------|
| Llama-3.2-1B-Instruct | Works with tuning | `TRITON_ATTN` plus low-memory settings produced a clean 3/3 prompt benchmark run; median TTFT about 90 ms, median total latency about 2019 ms, median gen throughput about 29.9 tok/s |
| LFM2-1.2B | Fails | Internal KV-cache page-size assertion after engine init |
| Qwen3.5-0.8B | Fails | Gets through recognition and weight loading, then crashes in XPU multimodal attention path with `Only XE2 cutlass kernel is supported currently` |

### Intel downstream — llm-scaler

| Stack | Status | Detail |
|-------|--------|--------|
| llm-scaler host-side reconstruction | Partial bring-up | Patched downstream `vllm` and `arctic_inference` install locally, but host-side `vllm-xpu-kernels` build fails under local `icpx 2025.0.4`; the kernel repo expects newer oneAPI and Intel's current downstream Dockerfile installs `intel-oneapi-dpcpp-ct=2025.2.0-517` |

### llama.cpp — other backends

| Backend | Build | Runtime | Notes |
|---------|-------|---------|-------|
| SYCL | Fails | N/A | Arch oneAPI packages missing `oneapi/mkl.hpp` |
| OpenVINO GPU | Builds (with TBB shim) | Segfaults | `llama-bench` segfaults on both BF16 and Q4_K_XL |
| OpenVINO CPU | Builds | Fails | Prompt warmup fails on Q4_K_XL (`res = -3`) |

## Model coverage gaps

The following models could not be tested on any runtime due to version constraints in the current toolchain:

| Model | Blocked by | Reason |
|-------|------------|--------|
| **Qwen/Qwen3.5-0.8B** | `optimum-intel` / `transformers` | Architecture `qwen3_5` not recognized by transformers 4.57.6 (OpenVINO) or 4.51.3 (vllm-openvino) |
| **LiquidAI/LFM2-8B-A1B** (MoE) | `optimum-intel` / `transformers` | Architecture `lfm2_moe` not recognized by current export toolchain |

These are toolchain version gaps, not hardware limitations. They should resolve as `optimum-intel` and `transformers` add support for these newer architectures.

## Hardware characterization

### Memory bandwidth

| Test | Bandwidth |
|------|----------:|
| Host memcpy (mbw) | 17,771 MiB/s |
| XPU Host-to-Device | 45–54 GB/s |
| XPU Device-to-Host | 51–54 GB/s |
| XPU Device-to-Device | 52–54 GB/s |
| XPU vector-add effective | 93–107 GB/s |

### XPU compute (GEMM)

| Dtype | 1024 | 2048 | 4096 |
|-------|-----:|-----:|-----:|
| float32 | 2.4 TFLOPS | 3.4 TFLOPS | 3.9 TFLOPS |
| bfloat16 | 10.5 TFLOPS | 24.4 TFLOPS | 22.7 TFLOPS |
| float16 | 10.5 TFLOPS | 24.6 TFLOPS | 21.9 TFLOPS |
| int8 | 12.8 TOPS | 26.8 TOPS | 41.1 TOPS |

### PyTorch XPU operator support

| SDPA backend | Status |
|-------------|--------|
| OVERRIDEABLE | Works (primary useful path) |
| MATH | Works (slow fallback) |
| FLASH_ATTENTION | Not available on Lunar Lake |
| EFFICIENT_ATTENTION | Not available |
| CUDNN_ATTENTION | Not available |

OVERRIDEABLE is the backend that matters for Intel client-GPU attention. Flash Attention is only available on PVC, BMG_G21, and similar discrete parts — not Lunar Lake.

## Current read

- **Best-supported inference paths today**: OpenVINO / OpenVINO GenAI / Optimum Intel, and `llama.cpp` via Vulkan
- **OpenVINO GenAI** is a genuine first-class runtime, not just a thin wrapper — it ships its own performance tooling and supports GPU + NPU
- **llama.cpp Vulkan** is the cleanest GPU inference path, with the highest token generation rates in our tests (up to ~60 tok/s Q4)
- **vllm-openvino** produced a promising initial Llama result but is not yet stable across reruns
- **Upstream vLLM XPU** is still exploratory, but a small Llama text-serving path now works on this machine with `TRITON_ATTN` and aggressive low-memory tuning
- **Upstream Xe2 multimodal coverage** is still incomplete; both local testing and an open upstream Arc `140V` issue hit `Only XE2 cutlass kernel is supported currently`
- **PyTorch XPU** has a real upstream path but still needs model-family and quantization validation
- **Intel llm-scaler** is now tracked as a downstream reference stack; its public docs remain B60-centric, and the local host-side bring-up is currently blocked by an older host oneAPI compiler line (`icpx 2025.0.4` versus newer downstream expectations)
- **IPEX-LLM** is archived and should not be treated as a default path
- **NPU** works for Whisper (actually faster than GPU for whisper-large-v3-turbo) and basic LLM inference, but throughput is much lower than GPU

## Setup

The recommended starting point:

1. Install the Intel Linux GPU driver stack
2. Install the Intel NPU driver if your machine has one and you intend to test it
3. Use separate `mamba`/`conda` environments per stack
4. Start with one of these paths:
   - **OpenVINO / OpenVINO GenAI** for the smallest maintained runtime baseline
   - **llama.cpp Vulkan** for the lightest GPU bring-up
   - **PyTorch XPU** for the upstream framework path
5. Treat vLLM and SGLang as later-wave stacks

Arch-specific note: with `intel-npu-driver-bin`, OpenVINO NPU enumeration only works after exposing `/usr/lib/x86_64-linux-gnu` through `LD_LIBRARY_PATH` (codified in [`00-setup/npu-env.sh`](00-setup/npu-env.sh)).

For full install steps, see [`IMPLEMENTATION.md`](IMPLEMENTATION.md). For machine bring-up status, see [`00-setup/STATUS.md`](00-setup/STATUS.md).

## Repository layout

| Directory | Purpose |
|-----------|---------|
| [`00-setup/`](00-setup/) | System bring-up, driver/toolchain verification, per-stack env validation |
| [`01-hardware/`](01-hardware/) | Raw memory-bandwidth, compute, and telemetry characterization |
| [`02-operators/`](02-operators/) | Operator-level PyTorch XPU benchmarks (GEMM, SDPA) |
| [`03-openvino/`](03-openvino/) | OpenVINO-family runtime validation, model export, device benchmarks |
| [`04-llama.cpp/`](04-llama.cpp/) | Backend-specific llama.cpp build validation and GGUF runtime benchmarks |
| [`05-vllm/`](05-vllm/) | vLLM XPU and vllm-openvino setup, support matrix, and serving benchmarks |
| [`benchmarks/`](benchmarks/) | Shared OpenAI-compatible benchmark tooling reused across phases |
| [`llama.cpp/`](llama.cpp/) | Pinned upstream submodule |
| [`reference/`](reference/) | Tracked source material and pinned upstream reference submodules |

Supporting docs:

- [`ANALYSIS.md`](ANALYSIS.md) — framework and model-family support synthesis
- [`RESEARCH-hardware.md`](RESEARCH-hardware.md) — sourced hardware tables and throughput estimates for Intel client/Arc GPUs
- [`IMPLEMENTATION.md`](IMPLEMENTATION.md) — Linux install guide, environment strategy, backend limitations
- [`TESTING.md`](TESTING.md) — layered benchmark plan
- [`TODO.md`](TODO.md) — research backlog

To clone with submodules:

```bash
git clone --recurse-submodules <repo-url>
# or in an existing clone:
git submodule update --init --recursive
```

## Benchmark run contract

Default behavior for "run benchmarks" or "rerun phase X" requests in this repo. Unless explicitly overridden:

- Use the current machine/session state as-is (AC vs battery, TTY vs WM)
- Say exactly what was run and what was not
- Treat failures, unsupported models, and build blockers as results, not reasons to stop
- Leave raw logs under ignored `results/` directories
- Only summarize tracked results from runs that actually completed

For phase-specific defaults (what scripts to run, which models to check), see the individual phase READMEs.
