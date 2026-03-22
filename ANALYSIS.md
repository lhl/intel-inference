# Analysis

This document is a first-pass, docs-backed analysis of the current Intel AI/ML inference stack on Linux, with a focus on modern Intel Arc GPUs and adjacent Intel hardware. It started from the pinned local source set in this repository and has since been cross-checked against current maintained upstream docs used by [IMPLEMENTATION.md](/home/lhl/github/lhl/intel-inference/IMPLEMENTATION.md). It should still be read as a working analysis, not as a final support statement.

## Scope

- Platform focus: Linux only
- Hardware focus: Intel Arc dGPU, Xe-family iGPU, Intel NPU where relevant
- Workload focus: inference first
- Framework/runtime focus: PyTorch/XPU, OpenVINO, Optimum Intel, llama.cpp, vLLM, and SGLang where evidence exists
- Model-family focus: standard decoder LLMs, hybrid architectures, multimodal, ASR, and TTS
- Benchmark focus: apples-to-apples comparisons across architecture, size class, and quantization where possible

## Corrections and updates

- This is a docs-derived pass. No local runtime validation has been done yet.
- This analysis now combines the pinned local repos with a later maintained-doc verification pass. When this file and [IMPLEMENTATION.md](/home/lhl/github/lhl/intel-inference/IMPLEMENTATION.md) differ on setup details, prefer the maintained-source guidance in [IMPLEMENTATION.md](/home/lhl/github/lhl/intel-inference/IMPLEMENTATION.md).
- `IPEX-LLM` is valuable as background and gap-mapping, but its own README says the project is archived and has known security issues, so it is not treated as a default recommendation.
- At the time of the local-source pass, `SGLang` had no local repo footprint here. We now have maintained upstream XPU docs for it, but it should still be treated as an exploratory later-wave stack rather than a baseline path.
- At the time of the local-source pass, `vLLM` looked mostly indirect from Intel-specific evidence. We now have maintained upstream Intel XPU docs, but the current story is still narrower and more constrained than CUDA-first paths.
- `FastConformer` does not currently have a direct first-party signal in this local source set. The closest speech-related evidence is Whisper, SpeechT5, `Speech_Paraformer-Large`, and some Conformer-related conversion or export code.

## Evidence base

| Source | Commit or snapshot | Role in this analysis |
|---|---|---|
| `llama.cpp` | `3306dbaef` | Primary maintained source for Intel-relevant local inference backends: SYCL, Vulkan, OpenVINO |
| `reference/openvino` | `ea61688641` | Primary maintained source for Intel inference runtime positioning across GPU and NPU |
| `reference/openvino.genai` | `bb6dc5c114` | Primary maintained source for OpenVINO speech, GenAI pipeline APIs, and device-level generative runtime usage |
| `reference/optimum-intel` | `abe58d751` | Primary maintained source for Hugging Face plus OpenVINO flows |
| `reference/ipex-llm` | `de6bce2713` | Broad historical Intel coverage across model families and runtimes, but archived |
| `reference/pytorch-2-10torchao.html` | local HTML snapshot, captured 2026-03-22 | Point-in-time signal for current PyTorch XPU messaging on Intel Linux |
| PyTorch XPU docs | current upstream docs | Maintained install and support statement for Intel GPU on Linux |
| Intel GPU driver docs | current upstream docs | Maintained Linux driver baseline for Arc and Intel GPU userspace |
| OpenVINO install and NPU docs | current upstream docs | Maintained install and device guidance for Linux GPU and NPU |
| vLLM GPU and quantization docs | current upstream docs | Maintained upstream statement of Intel XPU install constraints and quant coverage |
| `vllm-openvino` | current upstream repo | Maintained limitations and device-specific guidance for OpenVINO-backed serving |
| SGLang XPU and backend docs | current upstream docs | Maintained but still thin evidence for Intel XPU serving |

## Support rubric

| Status | Meaning |
|---|---|
| Maintained and documented | Current evidence set shows active maintained docs for the path |
| Implemented, docs thin | Code or tests suggest support, but user-facing documentation is sparse |
| Integration-specific | Support exists through a fork, backend, or vendor-specific path rather than a clean upstream path |
| Archived | Support exists in an archived project and must be treated cautiously |
| No local evidence | Current evidence set does not provide enough evidence |

## Stage 1: descriptive analysis

### 1. Ecosystem view

| Area | Current state from current evidence | Status |
|---|---|---|
| OpenVINO | OpenVINO presents itself as a maintained Intel inference stack spanning CPU, GPU, and NPU, with explicit links to `torch.compile`, Optimum Intel, and `vllm-openvino`, plus examples for LLaVa and Whisper. | Maintained and documented |
| OpenVINO GenAI | OpenVINO GenAI exposes maintained pipeline APIs for LLMs, VLMs, Whisper speech recognition, and SpeechT5 speech generation on top of OpenVINO Runtime. | Maintained and documented |
| Optimum Intel | Optimum Intel positions itself as the Hugging Face to OpenVINO bridge for export, inference, and quantization. The README includes `OVModelForCausalLM` and Whisper quantization examples. | Maintained and documented |
| llama.cpp SYCL | llama.cpp has an Intel-first SYCL backend with explicit Linux support, verified Intel GPU families, oneAPI dependencies, and backend-specific tuning notes. | Maintained and documented |
| llama.cpp Vulkan | llama.cpp offers a Linux Vulkan build path that is generic rather than Intel-specific, but the docs show Intel GPU detection in the runtime example. | Maintained and documented |
| llama.cpp OpenVINO | llama.cpp also has an OpenVINO backend for Intel CPU, GPU, and NPU, but the backend docs still describe it as validated mainly on recent Intel AI PC platforms. | Maintained and documented |
| whisper.cpp | whisper.cpp is a maintained local Whisper runtime with quantization, Vulkan, and explicit OpenVINO encoder acceleration for Intel CPU and GPU. | Maintained and documented |
| PyTorch XPU | Official PyTorch docs provide Intel GPU install instructions, `torch.xpu` availability checks, binary wheels, and support statements for inference, training, eager mode, and `torch.compile`. | Maintained and documented |
| vLLM on Intel | Upstream vLLM now documents an Intel XPU install path, but labels it as initial support with explicit Python and kernel-package constraints, and the quant and kernel story remains narrower than CUDA or HIP. | Maintained, but narrower than CUDA/HIP |
| SGLang on Intel | SGLang has upstream XPU docs plus an `intel_xpu` attention backend path, but the current evidence is still thinner than for its CUDA-first paths. | Implemented, docs thin |
| IPEX-LLM | Broadest coverage in the local repo set across runtimes and model types, but explicitly archived and security-flagged. | Archived |

### 2. Linux requirements snapshot

The strongest Linux-specific requirements now come from maintained sources:

- Intel client GPU driver documentation
- PyTorch XPU install documentation
- OpenVINO install and NPU device documentation
- `llama.cpp` SYCL, Vulkan, and OpenVINO backend docs

`IPEX-LLM` remains useful as a historical reference for env vars, package combinations, and coverage breadth, but it should not anchor the default Linux setup story.

Observed recurring requirements:

- Intel GPU driver plus Level Zero or OpenCL userspace pieces
- oneAPI components for SYCL-centered paths
- `render` and sometimes `video` group membership
- verification tools like `clinfo`, `vulkaninfo`, and `xpu-smi`
- environment variables such as `SYCL_CACHE_PERSISTENT=1` and sometimes `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1`

The maintained stack is not uniform:

- `llama.cpp` Vulkan is the lightest-weight path in docs terms: install Vulkan packages, verify `vulkaninfo`, build with `-DGGML_VULKAN=1`.
- `llama.cpp` SYCL and `IPEX-LLM` both pull in a heavier oneAPI-based path.
- OpenVINO-based flows shift the stack toward OpenVINO runtime and device plugins rather than raw SYCL programming.
- PyTorch XPU is now documented as a direct upstream path rather than an Intel-only sidecar stack.
- `vLLM` XPU inherits meaningful constraints from the PyTorch XPU layer and adds its own Python 3.12 and `vllm-xpu-kernels` requirements.

### 3. Model-family view

| Model family | Strongest current Intel path from current evidence | Notes |
|---|---|---|
| Standard decoder LLMs | OpenVINO, Optimum Intel, PyTorch XPU, llama.cpp SYCL/OpenVINO, and IPEX-LLM all show meaningful evidence | This is the best-covered category by far |
| Hybrid SSM or recurrent-attention architectures | Mixed but promising: `llama.cpp` has Mamba-family model handling, and `optimum-intel` contains explicit Mamba2 and GatedDeltaNet code paths | User-facing docs are weaker than the implementation signals |
| MoE and large sparse LLMs | Evidence exists mainly in IPEX-LLM, including DeepSeek and Qwen3MoE claims on Arc | Strongest breadth comes from an archived project |
| Multimodal and VLM | OpenVINO tutorials mention LLaVa; IPEX-LLM lists Qwen-VL, Qwen2-VL, Phi-3-Vision, MiniCPM-V, MiniCPM-o; llama.cpp has multimodal docs and models | Broad evidence exists, but with uneven maturity across stacks |
| ASR and speech understanding | OpenVINO and OpenVINO GenAI both expose Whisper paths; Optimum Intel documents Whisper quantization; whisper.cpp provides a practical local ASR implementation with OpenVINO acceleration; IPEX-LLM adds archived breadth beyond that | Speech now looks materially more concrete than this repo first assumed |
| TTS | OpenVINO GenAI explicitly supports SpeechT5; Optimum Intel code and tests show SpeechT5 support; IPEX-LLM lists Bark and SpeechT5 | TTS still looks thinner than text LLMs, but no longer purely implementation-level |

Specific architectures called out for later validation:

- `Qwen 3.5`: local evidence exists in `llama.cpp` conversion mappings and in Optimum Intel's `qwen3_next` GatedDeltaNet patching path, but not yet as a polished Intel Linux setup story.
- `Nemotron 3`: local evidence exists in `llama.cpp` benchmark material for Nemotron-3 GGUF models, but that material is CUDA-oriented and does not by itself establish an Intel backend path.
- `FastConformer`: no direct local documentation signal yet; this remains an explicit research gap.

### 4. Backend and kernel view

| Stack | Kernel or backend signal | Interpretation |
|---|---|---|
| PyTorch XPU | Official docs plus the PyTorch snapshot claim support for inference, training, `torch.compile`, and common precision modes on Intel GPU | Intel is pushing upstream XPU parity, but exact model-family and quant coverage still need direct verification |
| OpenVINO | OpenVINO emphasizes graph compilation, kernel fusion, device plugins, `torch.compile`, and GenAI pipelines | This looks like the main maintained Intel optimization layer, especially outside pure PyTorch eager execution |
| llama.cpp SYCL | Intel-first backend with oneDNN, oneMKL, Level Zero, and Flash Attention notes | This is likely the strongest Intel-specific local inference path when GGUF is acceptable |
| llama.cpp Vulkan | Generic GPU path with relatively simple Linux build requirements | Attractive as a fallback path, but the docs are not Intel-specialized |
| vLLM | Upstream docs show explicit Intel XPU support, but it is still constrained and custom-kernel-dependent | Likely still a major gap area versus NVIDIA and AMD |
| SGLang | Upstream docs show `intel_xpu` support, but Intel coverage appears much thinner than CUDA-centric paths | Must be treated as exploratory until validated |

### 5. Source-backed observations by project

#### PyTorch XPU

- The local PyTorch 2.10 snapshot frames the current story around XPU, TorchAO, and SYCL rather than a separate Intel-only extension.
- Official PyTorch docs now make the baseline clearer:
  - install the Intel GPU driver first
  - install `torch`, `torchvision`, and `torchaudio` from the XPU wheel index
  - verify with `torch.xpu.is_available()`
  - both inference and training are documented, along with eager mode and `torch.compile`
- This is enough to treat PyTorch XPU as a maintained upstream path, though not enough to assume architecture-level parity without testing.

#### OpenVINO and Optimum Intel

- OpenVINO explicitly positions itself as an inference toolkit for Intel CPU, GPU, and NPU, and explicitly points users to Optimum Intel, `torch.compile`, and `vllm-openvino`.
- OpenVINO also explicitly highlights LLaVa and Whisper tutorial material and maintains NPU-specific device and GenAI documentation.
- OpenVINO GenAI adds maintained pipeline-level APIs for:
  - Whisper speech recognition
  - SpeechT5 speech generation
  - LLM, VLM, and embedding pipelines on the same runtime base
- Optimum Intel clearly documents causal LM export and inference, and also documents Whisper quantization.
- Beyond the README, Optimum Intel contains implementation-level signals for newer hybrid architectures:
  - `modeling_decoder.py` contains explicit `Mamba2` state handling.
  - `model_patcher.py` contains a patched recurrent `GatedDeltaNet` implementation adapted from `qwen3_next`.
- This suggests the maintained Intel/OpenVINO path may be ahead of its top-level docs for some newer architectures.
- It also suggests that newer Qwen hybrid or recurrent-attention architectures may land in OpenVINO-based Intel paths before they are clearly surfaced in README-level support matrices.

#### llama.cpp

- `llama.cpp` currently gives Intel three distinct Linux inference paths:
  - SYCL for Intel GPUs with explicit verified device coverage
  - Vulkan as a generic GPU backend
  - OpenVINO for Intel CPU, GPU, and NPU
- The SYCL docs are notably concrete about Arc A, Arc B, built-in Arc, and older Intel iGPU support boundaries.
- The OpenVINO backend docs are concrete about device classes and quantization constraints, but the validation claims are narrower than OpenVINO's broad top-level platform claims.
- `llama.cpp` also has multimodal support and explicit Mamba-family model coverage, which matters for hybrid architecture evaluation.
- `llama.cpp` additionally contains direct signals for `Qwen 3.5` conversion support and Nemotron-3 GGUF benchmark artifacts, but those signals are not automatically Intel-specific support claims.

#### whisper.cpp

- `whisper.cpp` is the clearest local, speech-specific sibling to `llama.cpp` in this repo set.
- Its README documents:
  - local quantized Whisper inference
  - Vulkan support
  - OpenVINO encoder acceleration for Intel CPU and GPU
- That makes it a meaningful maintained ASR path in its own right, not just a side note under generic OpenVINO or Hugging Face flows.

#### IPEX-LLM

- `IPEX-LLM` remains the broadest source in the current local set for Intel-specific model-family breadth.
- It documents:
  - Linux GPU installation with oneAPI 2024.0 and runtime environment variables
  - Intel-only Linux vLLM integration via a dedicated `analytics-zoo/vllm` branch
  - verified model tables spanning Qwen2.5, Qwen2-Audio, Whisper, Distil-Whisper, Mamba, Bark, SpeechT5, Phi-3-Vision, MiniCPM multimodal variants, and NPU speech workloads like `Speech_Paraformer-Large`
- This makes it extremely useful as a map of what Intel had working or working enough to document.
- The problem is governance, not breadth: because the repo is archived and security-flagged, it is evidence of prior work, not a clean default path or baseline setup guide.

#### vLLM and SGLang

- Upstream `vLLM` now has an explicit Intel XPU installation section for Linux.
- The upstream docs still frame Intel support as early:
  - initial support wording
  - Intel ARC and Data Center GPU called out
  - Python 3.12 requirement for the provided `vllm-xpu-kernels` wheel
- The upstream quantization matrix shows Intel GPU support for AWQ and GPTQ, but not Marlin, FP8, or `bitsandbytes`; it also shows `GGUF` support in the generic matrix, which should be treated carefully until the Intel XPU runtime path is directly validated.
- `vllm-openvino` is a distinct path, not just "vLLM on Intel" generically. Its own README says:
  - only LLM models are currently supported
  - LLaVA and encoder-decoder models are not enabled
  - LoRA serving, tensor parallelism, and pipeline parallelism are not enabled
- Upstream SGLang now documents an XPU platform page and an `intel_xpu` attention backend, but the Intel path still looks materially thinner than the CUDA-centric surface area in the rest of its docs.

## Stage 2: evaluative analysis

### 1. Where the evidence is strong

- Strongest maintained inference story:
  - OpenVINO plus Optimum Intel for general Intel inference, especially GPU and NPU
  - llama.cpp SYCL and OpenVINO for local LLM inference on Intel hardware
- Strongest evidence that Intel can cover non-text workloads:
  - OpenVINO tutorial positioning for LLaVa and Whisper
  - OpenVINO GenAI Whisper and SpeechT5 pipeline support
  - whisper.cpp as a practical local Whisper runtime with Intel-relevant acceleration
  - Optimum Intel Whisper quantization
  - IPEX-LLM model tables covering multimodal, ASR, and TTS-adjacent workloads

### 2. Where the evidence is weaker than it first appears

- PyTorch XPU:
  - The maintained-source signal is now stronger than when this file started.
  - The remaining gap is not "is there an upstream path" but "how much real model-family and quant coverage exists on Intel XPU in practice".
  - We should not assume model-family parity from generic claims about operators and libraries.

- vLLM:
  - Upstream Intel XPU support is now explicit, so the question is no longer whether it exists.
  - The real issue is maturity:
    - narrower hardware story
    - extra kernel-package constraints
    - narrower quant and kernel coverage than CUDA
  - `vllm-openvino` is useful, but it is a separate OpenVINO integration with notable feature limits.

- SGLang:
  - Upstream Intel XPU docs exist now.
  - The remaining concern is still maturity and breadth, not total absence.

- Hybrid architectures:
  - The strongest signals are implementation-level:
    - `Mamba2` handling in Optimum Intel
    - `GatedDeltaNet` patching for `qwen3_next`
    - Mamba-family support in llama.cpp
  - That is meaningful engineering evidence, but it is not the same thing as a polished, supported end-user path.

### 3. Main tensions

1. Breadth vs maintainability

The broadest Intel-specific evidence is in `IPEX-LLM`, but it is archived. The most maintainable-looking paths are OpenVINO, Optimum Intel, and llama.cpp, but those sources currently communicate less breadth at the top level than IPEX-LLM did.

2. Backend maturity vs ecosystem ergonomics

Intel appears to have real backends:

- XPU in PyTorch
- OpenVINO for graph-lowered inference
- SYCL and OpenVINO in llama.cpp

But the serving-engine ergonomics appear weaker than in CUDA-first ecosystems, especially for vLLM and probably SGLang.

3. Model-family breadth vs explicit validation

OpenVINO and Optimum Intel likely support more than their READMEs explicitly advertise, but many of the strongest signals for newer families currently live in implementation code and tests, not in polished Linux setup guides.

### 4. Evidence-quality ratings

| Claim | Type | Evidence level | Credence | Why |
|---|---|---:|---:|---|
| OpenVINO is the main maintained Intel inference path across GPU and NPU in this repo set | `[T]` | E3 | 0.77 | Multiple maintained sources and integrations point in the same direction |
| llama.cpp is currently one of the most practical Intel Linux inference paths | `[T]` | E3 | 0.75 | Detailed backend docs and explicit Intel device coverage |
| vLLM Intel support now has an upstream-documented path, but it still looks narrower than CUDA or HIP in practice | `[H]` | E3 | 0.77 | Upstream docs exist, but the constraints and feature gaps are explicit |
| SGLang Intel support exists upstream, but still looks exploratory and thinly evidenced | `[H]` | E3 | 0.71 | Upstream docs exist, but Intel coverage is much thinner than CUDA-oriented paths |
| Intel support breadth is much stronger for standard LLMs than for hybrid, multimodal, ASR, or TTS workloads | `[T]` | E3 | 0.74 | Broad but uneven signals across sources |
| IPEX-LLM is useful for background but unsafe as a default recommendation | `[F]` | E2 | 0.9 | The repo itself says it is archived and has known security issues |

## Stage 3: dialectical analysis

### Steelman

Intel's Linux inference story is more real than many people assume:

- upstream-facing PyTorch XPU work exists
- OpenVINO and Optimum Intel provide a maintained path across GPU and NPU
- llama.cpp gives Intel multiple viable local inference backends
- historical Intel work in IPEX-LLM shows that multimodal, speech, TTS, and even large sparse models were not ignored

On this reading, Intel is not missing an inference stack. It is missing a single clean narrative that ties together maintained paths, model-family coverage, and serving-engine competitiveness.

### Counterargument

The best-known CUDA and HIP ecosystems still look stronger in first-class serving support and custom-kernel maturity:

- vLLM support for Intel exists, but still looks constrained relative to CUDA and HIP
- SGLang Intel support exists, but it still is not evidenced as deeply as the maintained Tier 1 paths
- some of the most impressive Intel breadth sits in an archived project
- hybrid architecture support is visible in code before it is visible in polished end-user docs

On this reading, Intel has a credible inference substrate, but it still trails in the high-velocity serving-engine layer where NVIDIA and AMD attract first-class optimization work.

### Synthesis

The most defensible current synthesis is:

1. Standard LLM inference on Intel Linux is credible today.
2. The strongest maintained documentation path is OpenVINO plus Optimum Intel, with llama.cpp as a practical local inference path and escape hatch.
3. PyTorch XPU matters and looks promising, but it needs direct model-family verification.
4. Breadth beyond standard text LLMs exists, but a lot of that breadth is either archived, implementation-level, or insufficiently documented.
5. vLLM and SGLang are still where Intel is most likely to look behind CUDA and HIP ecosystems.

## Priority research passes after this document

1. Add first-party reference repos for upstream `vllm`, `vllm-openvino`, and `sglang`.
2. Trace exact model-family coverage in `optimum-intel` and OpenVINO tests for:
   - hybrid models
   - multimodal models
   - ASR
   - TTS
3. Map Intel Linux requirements into one matrix:
   - Arc dGPU
   - Xe-family iGPU
   - NPU
   - backend-specific packages and env vars
4. Validate one representative example each for:
   - standard LLM
   - hybrid LLM
   - multimodal model
   - ASR
   - TTS
5. Map quantization support by stack and by architecture family instead of treating "quant support" as one column.
6. Define canonical benchmark artifacts:
   - one source-format dense baseline
   - one source-format size-up baseline
   - one GGUF derivative path for `llama.cpp`

## Source references

- `reference/ipex-llm/README.md` at `de6bce2713`
- `reference/ipex-llm/docs/mddocs/Quickstart/install_linux_gpu.md` at `de6bce2713`
- `reference/ipex-llm/docs/mddocs/Quickstart/vLLM_quickstart.md` at `de6bce2713`
- `reference/ipex-llm/docs/mddocs/Quickstart/npu_quickstart.md` at `de6bce2713`
- `reference/openvino/README.md` at `ea61688641`
- `reference/openvino/tests/model_hub_tests/pytorch/test_hf_transformers.py` at `ea61688641`
- `reference/openvino.genai/README.md` at `bb6dc5c114`
- `reference/openvino.genai/tests/python_tests/test_whisper_pipeline.py` at `bb6dc5c114`
- `reference/optimum-intel/README.md` at `abe58d751`
- `reference/optimum-intel/optimum/intel/openvino/modeling_decoder.py` at `abe58d751`
- `reference/optimum-intel/optimum/exporters/openvino/model_patcher.py` at `abe58d751`
- `reference/optimum-intel/optimum/exporters/openvino/convert.py` at `abe58d751`
- `reference/optimum-intel/tests/openvino/test_modeling.py` at `abe58d751`
- `llama.cpp/docs/backend/SYCL.md` at `3306dbaef`
- `llama.cpp/docs/backend/OPENVINO.md` at `3306dbaef`
- `llama.cpp/docs/build.md` at `3306dbaef`
- `llama.cpp/convert_hf_to_gguf.py` at `3306dbaef`
- `llama.cpp/convert_hf_to_gguf_update.py` at `3306dbaef`
- `llama.cpp/benches/nemotron/nemotron-dgx-spark.md` at `3306dbaef`
- `reference/whisper.cpp/README.md` at `76684141a5`
- `reference/pytorch-2-10torchao.html` local snapshot captured on 2026-03-22
- PyTorch XPU docs: `https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html`
- Intel client GPU driver docs: `https://dgpu-docs.intel.com/driver/client/overview.html`
- OpenVINO install docs: `https://docs.openvino.ai/2026/get-started/install-openvino.html`
- OpenVINO NPU device docs: `https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html`
- OpenVINO GenAI on NPU: `https://docs.openvino.ai/2026/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html`
- vLLM GPU install docs: `https://docs.vllm.ai/en/latest/getting_started/installation/gpu/`
- vLLM quantization docs: `https://docs.vllm.ai/en/latest/features/quantization/`
- `vllm-openvino`: `https://github.com/vllm-project/vllm-openvino`
- SGLang XPU docs: `https://docs.sglang.io/platforms/xpu.html`
- SGLang attention backend docs: `https://docs.sglang.io/advanced_features/attention_backend.html`
- SGLang quantization docs: `https://docs.sglang.io/advanced_features/quantization.html`
