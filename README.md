# Intel AI/ML Toolchain Research

This repository is for researching and documenting the state of the AI/ML toolchain on Intel hardware on Linux, with a focus on modern Intel Arc GPUs.

Initial scope:

- Modern Intel hardware, with a focus on Intel Arc discrete GPUs and Intel integrated graphics based on Xe, Xe2, and Xe3
- Intel NPU support where relevant
- Linux-focused; Linux is the only platform in scope for now
- Inference first
- Training and general AI/ML development are secondary interests
- PyTorch, OpenVINO, OpenVINO GenAI, Optimum Intel, vLLM, llama.cpp, and adjacent serving/runtime paths such as SGLang where relevant
- Model coverage should include standard LLMs, newer hybrid architectures, multimodal models, ASR, and TTS where Intel support meaningfully differs
- Reproducible setup notes, validation steps, and known limitations

This is still an early docs-first repo. The goal right now is to turn scattered Intel ecosystem information into tested, Linux-focused guidance. Anything not explicitly validated here should still be treated as provisional.

## Current read

Based on the maintained docs and pinned reference repos in this checkout:

- Best-supported maintained Intel inference paths today look like:
  - OpenVINO, OpenVINO GenAI, and Optimum Intel
  - `llama.cpp` via SYCL, Vulkan, or OpenVINO
- `openvino.genai` looks like a genuine first-class runtime layer, not just a thin wrapper, and it ships its own performance and accuracy-oriented tooling.
- For speech-specific work, the two most concrete maintained paths worth tracking are `openvino.genai` and `whisper.cpp`.
- PyTorch XPU now has a real upstream path and is important, but still needs model-family and quantization validation on Intel hardware.
- Upstream `vLLM` XPU and SGLang XPU both exist, but they currently look narrower and less mature on Intel than the CUDA- and HIP-first paths.
- `vllm-openvino` is a distinct OpenVINO-backed serving path, not a synonym for generic Intel `vLLM` support.
- `IPEX-LLM` is useful as historical reference material and gap-mapping, but it is archived and should not be treated as the default path.
- The first `05-vllm` pass now makes that split concrete:
  - `vllm-openvino` currently has the best initial Llama result on this machine, but it is not yet robust across reruns
  - upstream `vLLM` XPU is newer and broader, but currently less stable

## Current validated OpenVINO status

The `03-openvino` phase is no longer only env-level:

- OpenVINO GenAI LLM serving now works locally behind an OpenAI-compatible `/v1/chat/completions` adapter, using the shared [benchmarks/openai_api_bench.py](/home/lhl/github/lhl/intel-inference/benchmarks/openai_api_bench.py) client that we plan to reuse for `llama.cpp` and `vLLM`.
- Current maintained export/runtime baseline that works here:
  - `meta-llama/Llama-3.2-1B-Instruct`
  - `LiquidAI/LFM2-1.2B`
  - `openai/whisper-large-v3-turbo`
  - `openai/whisper-large-v3`
- Current maintained Optimum/OpenVINO export blockers:
  - `Qwen/Qwen3.5-0.8B`
  - `LiquidAI/LFM2-8B-A1B`
- Model-level NPU validation is real now, not just synthetic:
  - `Llama-3.2-1B-Instruct` ran on `NPU`
  - `whisper-large-v3-turbo` ran on `NPU`
- The current maintained OpenVINO export path is also version-constrained:
  - on this machine it is `optimum-intel 1.27.0` with `transformers 4.57.6`
  - that matters because newer upstream architectures can fail at export time even when OpenVINO Runtime itself is fine

For the exact export status, exact failure modes, and current GPU/NPU timings, use [03-openvino/README.md](/home/lhl/github/lhl/intel-inference/03-openvino/README.md).

## Current validated llama.cpp status

The `04-llama.cpp` phase is now at backend build-validation level:

- `Vulkan`
  - builds cleanly on this Arch Lunar Lake machine with the system Mesa/Vulkan stack
  - `llama-bench --list-devices` sees `Vulkan0: Intel(R) Graphics (LNL)`
- `SYCL`
  - configure succeeds after sourcing [00-setup/oneapi-env.sh](/home/lhl/github/lhl/intel-inference/00-setup/oneapi-env.sh)
  - build currently fails on Arch because the shipped oneAPI package set does not provide `oneapi/mkl.hpp`, which upstream `ggml-sycl` includes
- `OpenVINO`
  - builds successfully against the `intel-inf-openvino` env
  - on this Arch machine, the current `openvino` pip wheel is not sufficient by itself for `llama.cpp`'s OpenVINO CMake expectations; a small env-local TBB shim is required
  - backend enumeration works on `CPU`, `GPU`, and `NPU`

That means the current practical `llama.cpp` read here is:

- Vulkan is the cleanest GPU baseline
- OpenVINO builds with an Arch-specific workaround, but the current GGUF runtime path is not yet usable enough here for a real comparison set
- SYCL is still blocked on the current Arch oneAPI packaging layout

The first real GGUF runtime pass tightened that further:

- Vulkan completed a full `llama-bench -fa 1` sweep on `Llama-3.2-1B` and `LFM2.5-1.2B` in both `BF16` and `Q4_K_XL`
- the current OpenVINO GGUF path did not produce comparable model-level numbers:
  - GPU `llama-bench` segfaulted on both `BF16` and `Q4_K_XL`
  - GPU `llama-cli` also segfaulted on `Q4_K_XL`
  - default-device OpenVINO `CPU` `llama-bench` failed prompt warmup on `Q4_K_XL`

For the exact commands, current workaround details, and backend-specific build scripts, use [04-llama.cpp/README.md](/home/lhl/github/lhl/intel-inference/04-llama.cpp/README.md).

## Current validated vLLM status

The `05-vllm` phase is now active rather than planned.

Current machine read:

- `vllm-openvino`
  - `Llama-3.2-1B-Instruct` completed one successful `GPU` benchmark run through the shared OpenAI-compatible benchmark client
  - the current measured pass is about `1990 ms` median total latency, `53 ms` median TTFT, and `28.1 tok/s` median generation speed
  - an immediate rerun then failed in the on-the-fly OpenVINO export/load path, so this should not be treated as a stable baseline yet
  - `Qwen3.5-0.8B` and `LFM2-1.2B` both fail before serve because the pinned `vllm-openvino` stack is on `transformers 4.51.3`
- upstream `vLLM` XPU
  - uses a newer local source build and can recognize more model families than `vllm-openvino`
  - `Llama-3.2-1B-Instruct` starts and serves, but the first shared benchmark run was unstable and returned `500` errors on `2/3` prompts
  - `LFM2-1.2B` reaches engine init, then fails on an internal KV-cache assertion
  - `Qwen3.5-0.8B` reaches engine init on the newer stack, then fails in the XPU multimodal attention path

That means the current recommendation for this repo is:

- treat `vllm-openvino` as the current best initial Intel `vLLM` result for simple Llama-class GPU serving, but not yet as a stable baseline
- treat upstream `vLLM` XPU as promising but still exploratory on this hardware

For the exact env split, scripts, and failure details, use [05-vllm/README.md](/home/lhl/github/lhl/intel-inference/05-vllm/README.md).

## Basic Linux setup

The current recommended starting point is:

1. Install the Intel Linux GPU driver stack first.
2. Install the Intel NPU driver only if your machine has an NPU and you intend to test it.
3. Use separate `mamba` or `conda` environments per stack instead of one shared environment.
4. Start with one of these paths:
   - OpenVINO for the smallest maintained runtime baseline
   - OpenVINO GenAI for a more productized local pipeline/runtime layer on top of OpenVINO
   - a separate Optimum Intel env when you actually need Hugging Face export or Optimum integration
   - `llama.cpp` Vulkan for the lightest initial GPU bring-up
   - `llama.cpp` SYCL if you want the Intel-specific GPU backend
   - PyTorch XPU if you want the upstream framework path
5. Treat `vLLM`, `vllm-openvino`, and SGLang as later-wave stacks after the baseline paths work.
6. For this repo's current `vLLM` work, prefer the dedicated setup scripts under `00-setup/`:
   - `setup-vllm-xpu-env.sh`
   - `setup-vllm-openvino-env.sh`

For the actual package, driver, oneAPI, env-var, and build steps, use [IMPLEMENTATION.md](/home/lhl/github/lhl/intel-inference/IMPLEMENTATION.md).
For the current validated machine bring-up state, use [00-setup/STATUS.md](/home/lhl/github/lhl/intel-inference/00-setup/STATUS.md).

Current Arch-specific note from live validation:

- with `intel-npu-driver-bin`, OpenVINO NPU enumeration worked only after exposing `/usr/lib/x86_64-linux-gnu` through `LD_LIBRARY_PATH`
- the repo codifies that via [00-setup/npu-env.sh](/home/lhl/github/lhl/intel-inference/00-setup/npu-env.sh)

## What currently looks alive, weak, or dead

- Strongest maintained direction:
  - OpenVINO, OpenVINO GenAI, Optimum Intel, and Intel-backed OpenVINO/NPU flows
  - `openvino.genai` for Whisper and SpeechT5-class speech pipelines
  - upstream PyTorch XPU
  - `llama.cpp` Intel-relevant backends
  - `whisper.cpp`, especially where a lightweight local ASR path or OpenVINO-backed encoder acceleration is useful
- Likely weaker or more constrained today:
  - upstream `vLLM` XPU
  - `vllm-openvino`
  - SGLang XPU
- Dead or deprecated as a default path:
  - `IPEX-LLM`, because it is archived and security-flagged

## Where current effort seems to be going

The current development story appears to center on:

- upstream PyTorch XPU rather than a separate Intel-only PyTorch fork
- OpenVINO, OpenVINO GenAI, and Optimum Intel for optimized Intel inference, especially across GPU and NPU
- `llama.cpp` backend work for practical local inference on Intel hardware
- OpenVINO GenAI and OVMS-style serving layers rather than only Hugging Face wrapper flows
- selective Intel support in serving stacks rather than clear first-class parity with CUDA-first ecosystems

That is enough to start writing useful setup and support guidance now, even before the performance testing phase is complete.

Planned contents:

- Hardware and platform support matrix
- Linux driver and runtime setup guidance
- Framework-specific setup for PyTorch, vLLM, llama.cpp, and OpenVINO-related flows where relevant
- Model-family coverage notes for text, multimodal, speech, and TTS workloads
- Backend and kernel support notes, especially where Intel support differs from CUDA- and HIP-first ecosystems
- Minimal smoke tests to confirm the stack is working
- Troubleshooting notes for common failure modes

Current docs:

- [`ANALYSIS.md`](ANALYSIS.md): first-pass synthesis of current Intel/Linux support across frameworks and model families
- [`RESEARCH-hardware.md`](RESEARCH-hardware.md): sourced hardware tables, derived throughput estimates, and bandwidth notes for current Intel client and Arc GPUs
- [`IMPLEMENTATION.md`](IMPLEMENTATION.md): current Linux install guide, environment strategy, and backend limitations
- [`TESTING.md`](TESTING.md): layered benchmark plan from raw hardware characterization up through full runtime and model testing
- [`TODO.md`](TODO.md): research backlog and documentation checklist
- [`00-setup/`](00-setup/): numbered bring-up area for drivers, oneAPI, envs, and smoke tests
- [`01-hardware/`](01-hardware/): numbered low-level benchmark area for bandwidth, compute, and telemetry work
- [`02-operators/`](02-operators/): PyTorch XPU operator bring-up, GEMM, and SDPA benchmarking
- [`03-openvino/`](03-openvino/): OpenVINO, OpenVINO GenAI, and Optimum env/device validation plus real model export, runtime, and OpenAI-compatible benchmark checks
- [`04-llama.cpp/`](04-llama.cpp/): backend-specific `llama.cpp` build validation for `Vulkan`, `SYCL`, and `OpenVINO`
- [`05-vllm/`](05-vllm/): upstream `vLLM` XPU and `vllm-openvino` env validation plus initial OpenAI-compatible serving benchmarks
- [`benchmarks/`](benchmarks/): shared benchmark clients, prompt sets, and comparison harnesses reused across runtime phases

Repository layout:

- [`00-setup/`](00-setup/): system bring-up, driver/toolchain verification, and per-stack env validation
- [`01-hardware/`](01-hardware/): raw memory-bandwidth, compute, and telemetry characterization
- [`02-operators/`](02-operators/): operator-level PyTorch XPU benchmark layer
- [`03-openvino/`](03-openvino/): OpenVINO-family runtime validation and device microbenchmarks
- [`04-llama.cpp/`](04-llama.cpp/): backend-specific `llama.cpp` build validation and upcoming GGUF/runtime sweep layer
- [`05-vllm/`](05-vllm/): `vLLM` XPU and `vllm-openvino` setup, support, and initial benchmark layer
- [`benchmarks/`](benchmarks/): shared OpenAI-compatible benchmark tooling reused across runtime phases
- [`llama.cpp/`](llama.cpp/): pinned upstream submodule used for llama.cpp backend experiments
- [`reference/`](reference/): tracked source material plus pinned upstream reference submodules

If cloning this repository fresh, initialize the pinned upstream checkouts with:

```bash
git clone --recurse-submodules <repo-url>
```

or, in an existing clone:

```bash
git submodule update --init --recursive
```

Current recommendation order for new testing work:

1. Finish `00-setup/` first so the driver, oneAPI, env, and smoke-test story is recorded.
2. Finish `01-hardware/` next so we have raw bandwidth and compute baselines before runtime conclusions.
3. Use `02-operators/` to establish what PyTorch XPU kernels and SDPA paths actually exist on the current machine.
4. Then move into `03-openvino/` and `04-llama.cpp/` for runtime and backend-level comparisons.
5. Use `05-vllm/` for upstream `vLLM` XPU and `vllm-openvino` rather than mixing those into the OpenVINO baseline phase.
6. Only after that start broader model-family work.
7. Every benchmark phase should state the exact env or system-tool context used by each script.
8. Only after that spend time on SGLang and wider serving-stack comparisons.

The next major step is to turn the docs-derived guidance into validated setup notes, model coverage findings, and real benchmark results.
