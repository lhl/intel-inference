# Testing Plan

This document defines how this repo should move from docs-derived conclusions to repeatable measurements on Linux Intel hardware.

The main lesson from the updated [../strix-halo-testing](/home/lhl/github/lhl/strix-halo-testing/README.md) repo is that the work should be split cleanly:

- `hardware-test`: raw platform and bandwidth characterization
- `torch-*`: operator and kernel bring-up
- `llm-bench`: end-to-end `llama.cpp` sweep automation
- targeted backend investigations when a specific path underperforms

We should follow the same structure here rather than mixing everything into one benchmark bucket.

## Goals

We want to answer five different questions, not one:

1. What can the hardware do in theory?
2. What does the hardware actually deliver at the low level?
3. Which kernels and operator paths exist on Intel, and how fast are they?
4. How much performance do full runtimes like PyTorch, OpenVINO, `llama.cpp`, `vLLM`, and SGLang extract from the same hardware?
5. Which model families work well enough to document, and which ones only work on paper or in narrow configurations?

## Principles

- Keep low-level and end-to-end testing separate.
- Record enough system state to explain regressions later.
- Use one or two stable comparison models across stacks before expanding model coverage.
- Save machine-readable artifacts, not just prose summaries.
- Label results as `smoke`, `bench`, or `validated`.

## Repository organization we should grow into

The Strix Halo repo suggests the right shape:

```text
hardware-test/
torch-xpu/
llm-bench/
backend-investigations/
results/
```

Suggested Intel equivalents:

- `hardware-test/`
  - memory bandwidth
  - compute microbenches
  - telemetry capture
- `torch-xpu/`
  - GEMM
  - attention / SDPA
  - dtype and quant-path checks
- `llm-bench/`
  - `llama.cpp` sweep automation
  - result summarization
- `backend-investigations/`
  - one-off deep dives when a backend behaves oddly

## Layer 0: System baseline

Every serious run should capture:

- machine name
- CPU model
- GPU model
- NPU presence
- RAM capacity and speed
- distro
- kernel
- driver versions
- oneAPI version if used
- OpenVINO version
- PyTorch version
- `llama.cpp` commit and backend flags
- power profile / governor

Minimum commands:

```bash
uname -a
cat /etc/os-release
lscpu
lsmem
lspci -nn | grep -Ei 'vga|display|3d'
clinfo
vulkaninfo --summary
sycl-ls
xpu-smi discovery
python - <<'PY'
import torch
print(torch.__version__)
print("xpu available:", torch.xpu.is_available())
PY
```

Additional telemetry worth capturing:

- `intel_gpu_top`
- `xpu-smi stats`
- `sensors`

## Layer 1: Low-level hardware characterization

This layer answers "what does the machine itself deliver before full-model overhead?"

### 1. System memory bandwidth

Questions:

- What is the practical CPU-visible DRAM bandwidth?
- On UMA systems, how much of the theoretical bus can the CPU actually reach?
- How much does power tuning or firmware change the answer?

Outputs to record:

- theoretical bandwidth
- measured read / write / copy bandwidth
- measured / theoretical ratio

This is directly analogous to the `hardware-test` work in the Strix Halo repo, where CPU and GPU bandwidth were treated as separate ceilings.

### 2. GPU memory bandwidth

Questions:

- What is the effective bandwidth available to the Intel GPU for inference-like access patterns?
- How different are Vulkan, OpenCL, and SYCL-visible paths on the same system?

Plan:

- measure large-buffer read / write / copy bandwidth
- collect clocks, power, and thermal state during the run
- compare:
  - Intel iGPU
  - Intel Arc dGPU
  - Vulkan-visible path
  - SYCL-visible path

This is the Intel-side version of what the Strix Halo repo did with `memtest_vulkan` plus ROCm bandwidth tests.

### 3. NPU bandwidth

This is currently the weakest-measured layer.

There is no obvious public NPU bandwidth microbench in our current source set, so the initial plan should be:

1. compile synthetic OpenVINO graphs for NPU
2. use static-shape, linear-heavy workloads
3. use OpenVINO profiling output to estimate where time is going
4. treat the first NPU bandwidth numbers as derived estimates, not hard ground truth

Until we find or write a better tool, NPU "bandwidth" should be treated as an approximation.

### 4. Compute microbenchmarks

This is where `mamf-finder`-style work belongs.

The updated Strix Halo repo now has [torch-therock/mamf-finder.py](/home/lhl/github/lhl/strix-halo-testing/torch-therock/mamf-finder.py) and [torch-therock/run-mamf-finder.sh](/home/lhl/github/lhl/strix-halo-testing/torch-therock/run-mamf-finder.sh), which is exactly the pattern we should mirror.

Questions:

- What are the best GEMM shapes for Intel GPU by dtype?
- How much XMX does PyTorch actually unlock?
- What shape ranges are safe defaults for Intel microbench sweeps?

Initial Intel compute microbench plan:

- GEMM / linear sweep
- batched GEMM sweep
- SDPA / attention sweep
- quantized matmul sweep where relevant

Outputs to record:

- best-achieved TFLOPS by dtype
- best shape
- mean / median / max results across a sweep
- elapsed sweep time

## Layer 2: Kernel and operator testing

This layer asks "which kernels exist and how good are they?"

### PyTorch XPU

Borrow the spirit of the Strix Halo `torch-therock` and `flash-attention` trees:

- basic XPU sanity
- GEMM benchmark
- SDPA benchmark
- BF16 vs FP16 checks
- attention backend checks
- operator-level performance before full models

We should have a dedicated Intel tree for:

- `gemm-bench.py`
- `attention-bench.py`
- `backend-check.py`
- `run-mamf-finder.sh`

### OpenVINO

Test:

- compile time
- first-run cost
- steady-state throughput
- profile output for large linear / attention-heavy models
- NPU-specific export and static-shape behavior

### llama.cpp

This is where backend-specific operator behavior becomes visible indirectly.

We should use `llama.cpp`'s own tools rather than building a fake wrapper around them.

## Layer 3: Framework and runtime benchmarks

This is the layer most users care about.

### PyTorch XPU

Test categories:

- eager inference
- `torch.compile`
- Hugging Face `transformers`
- operator microbenches

Metrics:

- throughput
- first-run latency
- compile overhead
- peak memory

### OpenVINO and Optimum Intel

Test categories:

- export time
- compile time
- first token latency
- steady-state throughput
- GPU vs NPU behavior

Model types to start with:

- one dense LLM
- Whisper
- one multimodal model

### llama.cpp

This needs its own sub-layer because it already ships the right measurement tools.

#### `llama-bench`

Primary uses:

- prompt processing (`pp`)
- text generation (`tg`)
- prompt+generation (`pg`)
- context-depth sweeps (`-d`)
- batch and ubatch sweeps
- `-ngl` sweeps
- Flash Attention on/off
- backend comparisons

Standard sweep dimensions:

- `pp512`, `pp1024`, `pp2048`
- `tg32`, `tg128`, `tg256`
- `d=0`, `d=4096`, `d=8192`, `d=16384`, `d=32768`
- `b=128,256,512,1024,2048`
- `ub=128,256,512,1024`
- `-fa 0/1`
- backend:
  - Vulkan
  - SYCL
  - OpenVINO

This is also where we should mirror the deeper context-scaling work visible in the Strix Halo repo, not just stop at `pp512` / `tg128`.

#### `llama-perplexity`

Use this for quality and quantization regression testing.

Primary uses:

- compare `Q4_0`, `Q4_K_M`, `Q6_K`, `Q8_0`, `FP16`
- check whether the fastest Intel path also preserves acceptable model quality

#### `llama-cli`

Use this for:

- smoke tests
- multimodal checks
- interactive latency
- `--perf` timing output

#### `llama-server`

Use this for:

- serving stability
- concurrency
- continuous batching
- request-level latency

There is already a server benchmark path in [llama.cpp/tools/server/bench/README.md](/home/lhl/github/lhl/intel-inference/llama.cpp/tools/server/bench/README.md), so we should use that rather than inventing our own first.

### vLLM

Test separately for:

- upstream XPU
- `vllm-openvino`

Do not mix those results into one Intel "vLLM" table.

Questions:

- How much of the gap vs CUDA is kernel coverage?
- How much is quantization availability?
- What is the TTFT / TPOT profile?

### SGLang

Questions:

- Is `intel_xpu` actually competitive for supported models?
- Which page sizes work best?
- How limited is the multimodal and quantized story on Intel?

## Layer 4: End-to-end model coverage

After the baseline is stable, expand to model families.

### Initial shared baselines

Use one or two models that can be compared across many stacks:

- `Llama-2-7B Q4_0` for historical `llama.cpp` comparability
- `Llama-3.2-3B` or `Qwen2.5-1.5B` for modern dense baseline

### Follow-on model families

- dense modern LLM
- hybrid / recurrent / Mamba-like
- MoE
- multimodal
- ASR
- TTS

## Metrics to record

Each benchmark folder should try to capture:

- throughput
- TTFT
- TPOT
- compile time
- model load time
- peak RAM
- peak GPU memory / shared memory / GTT where applicable
- GPU utilization
- NPU utilization if available
- clocks
- temperatures
- power
- exact command and env vars

Derived metrics we care about:

- bandwidth efficiency
- compute efficiency
- quality loss from quantization
- performance per watt

## Result artifacts

The Strix Halo repo already shows a good pattern: keep machine-readable files, per-model folders, and generated summaries.

Per benchmark folder, keep:

- `system_info.json`
- `run_info.json`
- `raw_runs.jsonl`
- `results.jsonl`
- plots
- short `README.md`

Suggested folder shape:

```text
results/
  <machine>/
    <stack>/
      <model>/
        system_info.json
        run_info.json
        raw_runs.jsonl
        results.jsonl
        README.md
```

## First practical wave

The first real Intel pass should be:

1. Baseline system capture.
2. CPU and GPU memory bandwidth tests.
3. A `mamf-finder`-style GEMM sweep for PyTorch XPU.
4. A basic attention benchmark for PyTorch XPU.
5. OpenVINO dense-LLM smoke plus one NPU compile test if hardware exists.
6. `llama.cpp` backend comparison across:
   - Vulkan
   - SYCL
   - OpenVINO
7. `llama-bench` context-depth sweeps.
8. `llama-perplexity` on a small quant subset.
9. One upstream `vLLM` XPU pass.
10. One SGLang XPU pass.

## Open questions

- What is the best public low-level GPU bandwidth microbench for Intel on Linux?
- Is there any credible low-level NPU throughput microbench, or do we need to build one?
- Which model families can we keep truly apples-to-apples across PyTorch, OpenVINO, `llama.cpp`, `vLLM`, and SGLang?

Until those are answered, this document should be treated as the testing blueprint, not the results.
