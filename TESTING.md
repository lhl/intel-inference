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
4. How much performance do full runtimes like PyTorch, OpenVINO, OpenVINO GenAI, `llama.cpp`, `vLLM`, and SGLang extract from the same hardware?
5. Which model families work well enough to document, and which ones only work on paper or in narrow configurations?

## Principles

- Keep low-level and end-to-end testing separate.
- Record enough system state to explain regressions later.
- Use one canonical source artifact per family, then derive stack-specific formats from it.
- Treat model architecture, model size, and quantization as first-class benchmark axes.
- Never compare a GGUF quant against a source-format checkpoint as if they were the same artifact.
- Save machine-readable artifacts, not just prose summaries.
- Label results as `smoke`, `bench`, or `validated`.

## Stack tiers and sequencing

We should not treat every stack as equally mature on Intel Linux.

- Tier 1 baseline runtimes:
  - OpenVINO and Optimum Intel
  - OpenVINO GenAI
  - `llama.cpp` Vulkan
  - `llama.cpp` SYCL
  - `llama.cpp` OpenVINO
- Tier 1 support work:
  - PyTorch XPU operator bring-up and small-model smoke tests
- Tier 2 exploratory runtimes:
  - PyTorch XPU end-to-end `transformers` comparisons
  - upstream `vLLM` XPU
  - `vllm-openvino`
  - SGLang XPU

Rules:

- Tier 1 baselines must be working and measured before Tier 2 results are used for repo-level conclusions.
- PyTorch XPU microbench work can start early because it helps explain kernel ceilings, but it should not replace Tier 1 runtime baselines.
- `vLLM`, `vllm-openvino`, and SGLang should be treated as later-wave investigation targets, not the first source of Intel conclusions.

## Repository organization

The Strix Halo repo suggests the right idea, but this repo should be even more explicit about phase ordering.

Planned numbered layout:

```text
00-setup/
01-hardware/
02-operators/
03-runtime/
04-llama.cpp/
05-models/
99-results/
```

Phase meanings:

- `00-setup/`
  - driver, toolchain, and env bring-up
  - system inventory and smoke tests
- `01-hardware/`
  - memory bandwidth
  - compute microbenches
  - telemetry capture
- `02-operators/`
  - GEMM
  - attention / SDPA
  - dtype and quant-path checks
- `03-runtime/`
  - PyTorch, OpenVINO, OpenVINO GenAI, and serving-runtime comparisons
- `04-llama.cpp/`
  - backend-specific sweep automation
  - context-depth and quant testing
- `05-models/`
  - architecture-family and multimodal validation
- `99-results/`
  - summarized machine-readable outputs and final tables

For now, the first two directories to populate are `00-setup/` and `01-hardware/`.

## Canonical comparison artifacts

Cross-stack comparisons need explicit artifact rules.

### Dense text baselines

- Canonical source baseline:
  - `Qwen2.5-1.5B-Instruct`
- Canonical size-up baseline:
  - `Qwen2.5-7B-Instruct`
- Canonical GGUF derivatives for `llama.cpp`-only comparison:
  - `Q8_0`
  - `Q4_K_M`

Rules:

- PyTorch, OpenVINO, OpenVINO GenAI, Optimum Intel, `vLLM`, and SGLang should be compared using the same source-format checkpoint where possible.
- `llama.cpp` can participate in source-format-adjacent comparisons only through explicitly documented conversion steps from that same source checkpoint.
- GGUF results belong in a separate comparison table from source-format results.

### Architecture coverage

We should intentionally test beyond one dense decoder model.

- Dense decoder LLM:
  - baseline dense checkpoints above
- Hybrid or recurrent:
  - `Qwen3.5` family where supported
  - `Nemotron 3` or another Mamba2-style target where support is real rather than nominal
- Multimodal:
  - one LLaVA-class or Qwen2.5-VL or Qwen3 omni-capable model that multiple stacks can actually run
- ASR:
  - Whisper first
  - prioritize `openvino.genai`, Optimum Intel/OpenVINO, and `whisper.cpp` as the first real runtime paths
  - Fast Conformer only after we confirm a maintained path
- TTS:
  - one practical baseline such as SpeechT5
  - prioritize `openvino.genai` and Optimum Intel if the model is supported there

### Size classes

- Small:
  - 1B to 3B class, for broad cross-stack bring-up
- Medium:
  - 7B to 8B class, for realistic local inference comparisons
- Large:
  - only after the smaller baselines are stable, and only where the hardware fits the test cleanly

### Quantization coverage

We should test quantization explicitly rather than treating it as an implementation detail.

- Source-format baseline:
  - BF16 when supported
  - FP16 where BF16 is not available or not representative
- Source-format quant baselines:
  - one 8-bit path where the stack has a real maintained implementation
  - one 4-bit path where the stack has a real maintained implementation
- `llama.cpp` GGUF baselines:
  - `Q8_0`
  - `Q6_K`
  - `Q4_K_M`

Rules:

- Quant comparisons must state the exact method, tooling, and calibration or conversion path.
- Cross-stack quant comparisons should be labeled as approximate unless the quant recipe is genuinely equivalent.
- Architecture coverage and quant coverage should expand independently. A backend that supports a model family in BF16 does not automatically count as supporting it in low-bit form.

## Measurement protocol

Results will be noisy unless we standardize the run procedure.

### Environment and thermal state

- Test on AC power only.
- Record the active governor or power mode.
- Let the system idle for 5 minutes before each benchmark block.
- Disable unrelated heavy background jobs during a measured block.

### Warmup and repetition

- Do 1 cold setup run when compile or load time is part of the metric.
- Do 3 warmup runs and discard them.
- Do at least 5 measured repeats for each point.
- Report mean, median, min, max, and standard deviation when practical.

### Cache policy

- Measure cold and warm behavior separately.
- For cold runs, clear or bypass relevant caches where practical and document the method.
- For warm runs, reuse the same compiled artifacts and report that the caches were hot.
- OpenVINO FEIL and FIL should be recorded separately when the backend exposes them cleanly.

### Prompt and token protocol

- Use a fixed prompt set per model family.
- For decoder LLMs, keep at least these standard lengths:
  - prefill 128, 512, 2048
  - generation 32, 128, 256
- Record the exact chat template, tokenizer revision, and stop conditions.
- TTFT and TPOT numbers are invalid if prompt formatting differs across stacks.

### Affinity and execution settings

- Record CPU affinity or cgroup constraints if used.
- Record batch size, ubatch size, context length, and concurrency.
- Record backend flags such as `-ngl`, `-fa`, attention backend selection, and OpenVINO cache or device settings.
- Synchronize the device before stopping timers when the stack requires explicit sync.

### Failure and invalidation rules

- Mark a run invalid if it silently falls back to CPU.
- Mark a run invalid if the requested quant or kernel path is not actually selected.
- Mark a run invalid if thermal throttling starts mid-block and materially changes clocks.
- Mark a run invalid if OOM, repeated kernel errors, or unstable latency outliers dominate the measured repeats.

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

The first practical implementation in this repo is now:

- [run-host-mbw.sh](/home/lhl/github/lhl/intel-inference/01-hardware/run-host-mbw.sh)
- [run-xpu-bandwidth.sh](/home/lhl/github/lhl/intel-inference/01-hardware/run-xpu-bandwidth.sh)
- [run-mamf-finder.sh](/home/lhl/github/lhl/intel-inference/01-hardware/run-mamf-finder.sh)
- [run-npu-microbench.sh](/home/lhl/github/lhl/intel-inference/01-hardware/run-npu-microbench.sh)
- [collect-intel-gpu-top.sh](/home/lhl/github/lhl/intel-inference/01-hardware/collect-intel-gpu-top.sh)

The first practical `02-operators` implementation should be:

- [run-backend-check.sh](/home/lhl/github/lhl/intel-inference/02-operators/run-backend-check.sh)
- [run-gemm-bench.sh](/home/lhl/github/lhl/intel-inference/02-operators/run-gemm-bench.sh)
- [run-batched-gemm-bench.sh](/home/lhl/github/lhl/intel-inference/02-operators/run-batched-gemm-bench.sh)
- [run-attention-bench.sh](/home/lhl/github/lhl/intel-inference/02-operators/run-attention-bench.sh)

### OpenVINO

Test:

- compile time
- first-run cost
- steady-state throughput
- profile output for large linear / attention-heavy models
- NPU-specific export and static-shape behavior

### OpenVINO GenAI

Test:

- `LLMPipeline` and `VLMPipeline` bring-up
- `WhisperPipeline` and `Text2SpeechPipeline` bring-up
- continuous batching and prefix caching behavior
- speculative decoding and sparse-attention options where they apply
- `tools/llm_bench` as the default GenAI-side performance harness for LLM/VLM comparisons
- `tools/who_what_benchmark` as the default OpenVINO-side similarity and regression harness
- do not use `tools/who_what_benchmark` to make performance claims; use it only to detect output drift after export, quantization, or backend changes
- embeddings and rerank later, after the core generation paths are stable

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

### OpenVINO, OpenVINO GenAI, and Optimum Intel

Test categories:

- export time
- compile time
- first token latency
- steady-state throughput
- GPU vs NPU behavior
- pipeline API behavior for:
  - `LLMPipeline`
  - `VLMPipeline`
  - `WhisperPipeline`
  - `Text2SpeechPipeline`
- continuous batching behavior where GenAI exposes it
- OpenVINO GenAI `llm_bench` performance runs before inventing custom wrappers
- OpenVINO GenAI `wwb` similarity checks when quantization or export changes may affect output quality
- no throughput or latency conclusion should cite `wwb`; those claims must come from `llm_bench` or our own controlled measurements

Model types to start with:

- one dense LLM
- Whisper
- one multimodal model
- one TTS model
- one embedding or rerank model later, after the generation paths are stable

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
- Which model families are genuinely supported beyond dense decoder LLMs?
- Which quants are Intel-supported in practice, not just listed in a general feature matrix?

### SGLang

Questions:

- Is `intel_xpu` actually competitive for supported models?
- Which page sizes work best?
- How limited is the multimodal and quantized story on Intel?
- Does SGLang XPU support enough of our target architectures to justify ongoing effort?

## Layer 4: End-to-end model coverage

After the baseline is stable, expand to model families.

### Initial cross-stack baselines

- Shared source-format baseline:
  - `Qwen2.5-1.5B-Instruct`
- Shared size-up source-format baseline:
  - `Qwen2.5-7B-Instruct`
- Shared `llama.cpp`-only GGUF baseline:
  - derived from the same dense source baseline, not from a different upstream artifact

This gives us one canonical source model family and one canonical GGUF derivative path, instead of mixing unrelated artifacts.

### Follow-on model families

- dense modern LLM
- hybrid / recurrent / Mamba-like
- MoE
- multimodal
- ASR
- TTS

Recommended evaluation order:

1. Dense baseline
2. Dense size-up baseline
3. Hybrid or recurrent model
4. Multimodal
5. Whisper
6. TTS

For each family, test at least:

- source-format baseline precision
- one higher-precision or native baseline
- one lower-bit path if the backend actually supports it

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
5. Tier 1 dense baseline smoke on OpenVINO and Optimum Intel.
6. One OpenVINO GenAI dense-model smoke run.
7. One OpenVINO GenAI Whisper smoke run.
8. One OpenVINO NPU compile and cache test if hardware exists.
9. `llama.cpp` backend comparison across:
   - Vulkan
   - SYCL
   - OpenVINO
10. `llama-bench` context-depth sweeps.
11. `llama-perplexity` on `Q8_0`, `Q6_K`, and `Q4_K_M`.
12. One PyTorch XPU end-to-end dense-model smoke run with the same canonical source checkpoint.
13. Only after the above is stable:
    - one upstream `vLLM` XPU pass
    - one `vllm-openvino` pass
    - one SGLang XPU pass
14. After dense baselines are stable, expand to:
    - hybrid or recurrent architecture
    - multimodal
    - Whisper
    - TTS

## Open questions

- What is the best public low-level GPU bandwidth microbench for Intel on Linux?
- Is there any credible low-level NPU throughput microbench, or do we need to build one?
- Which model families can we keep truly apples-to-apples across PyTorch, OpenVINO, OpenVINO GenAI, `llama.cpp`, `vLLM`, and SGLang?
- Which hybrid and multimodal model families have enough maintained Intel support to deserve first-wave benchmarking?
- Which quantization methods are actually equivalent enough across stacks to justify direct comparison?

Until those are answered, this document should be treated as the testing blueprint, not the results.
