# Intel AI/ML Toolchain Research

This repository is for researching and documenting the state of the AI/ML toolchain on Intel hardware on Linux, with a focus on modern Intel Arc GPUs.

Initial scope:

- Modern Intel hardware, with a focus on Intel Arc discrete GPUs and Intel integrated graphics based on Xe, Xe2, and Xe3
- Intel NPU support where relevant
- Linux-focused; Linux is the only platform in scope for now
- Inference first
- Training and general AI/ML development are secondary interests
- PyTorch, vLLM, llama.cpp, and adjacent serving/runtime paths such as OpenVINO-backed flows and SGLang where relevant
- Model coverage should include standard LLMs, newer hybrid architectures, multimodal models, ASR, and TTS where Intel support meaningfully differs
- Reproducible setup notes, validation steps, and known limitations

This is still an early docs-first repo. The goal right now is to turn scattered Intel ecosystem information into tested, Linux-focused guidance. Anything not explicitly validated here should still be treated as provisional.

## Current read

Based on the maintained docs and pinned reference repos in this checkout:

- Best-supported maintained Intel inference paths today look like:
  - OpenVINO and Optimum Intel
  - `llama.cpp` via SYCL, Vulkan, or OpenVINO
- PyTorch XPU now has a real upstream path and is important, but still needs model-family and quantization validation on Intel hardware.
- Upstream `vLLM` XPU and SGLang XPU both exist, but they currently look narrower and less mature on Intel than the CUDA- and HIP-first paths.
- `vllm-openvino` is a distinct OpenVINO-backed serving path, not a synonym for generic Intel `vLLM` support.
- `IPEX-LLM` is useful as historical reference material and gap-mapping, but it is archived and should not be treated as the default path.

## Basic Linux setup

The current recommended starting point is:

1. Install the Intel Linux GPU driver stack first.
2. Install the Intel NPU driver only if your machine has an NPU and you intend to test it.
3. Use separate `mamba` or `conda` environments per stack instead of one shared environment.
4. Start with one of these paths:
   - OpenVINO and Optimum Intel for the most maintained Intel-focused path
   - `llama.cpp` Vulkan for the lightest initial GPU bring-up
   - `llama.cpp` SYCL if you want the Intel-specific GPU backend
   - PyTorch XPU if you want the upstream framework path
5. Treat `vLLM`, `vllm-openvino`, and SGLang as later-wave stacks after the baseline paths work.

For the actual package, driver, oneAPI, env-var, and build steps, use [IMPLEMENTATION.md](/home/lhl/github/lhl/intel-inference/IMPLEMENTATION.md).

## What currently looks alive, weak, or dead

- Strongest maintained direction:
  - OpenVINO, Optimum Intel, and Intel-backed OpenVINO/NPU flows
  - upstream PyTorch XPU
  - `llama.cpp` Intel-relevant backends
- Likely weaker or more constrained today:
  - upstream `vLLM` XPU
  - `vllm-openvino`
  - SGLang XPU
- Dead or deprecated as a default path:
  - `IPEX-LLM`, because it is archived and security-flagged

## Where current effort seems to be going

The current development story appears to center on:

- upstream PyTorch XPU rather than a separate Intel-only PyTorch fork
- OpenVINO and Optimum Intel for optimized Intel inference, especially across GPU and NPU
- `llama.cpp` backend work for practical local inference on Intel hardware
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

Repository layout:

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

1. Get OpenVINO or `llama.cpp` running first.
2. Bring up PyTorch XPU and basic operator checks.
3. Only then spend time on `vLLM`, `vllm-openvino`, and SGLang.

The next major step is to turn the docs-derived guidance into validated setup notes, model coverage findings, and real benchmark results.
