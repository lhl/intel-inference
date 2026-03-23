# TODO

This file tracks the docs-first research and validation work needed before the README becomes a real Linux setup guide.

## 1. Scope and research method

- Keep the project Linux-only for now.
- Treat inference as the primary target.
- Treat training and general AI/ML development as secondary unless they become necessary to explain the inference stack.
- Use the pinned upstream submodules and tracked local snapshots as the initial evidence base.
- Label conclusions as either docs-derived or locally validated.
- Record source repo, commit, file path, and any important caveats for each claim we keep.

## 2. Current working assumptions from the docs pass

- `llama.cpp` appears to offer three serious Linux backend paths for Intel hardware: SYCL, Vulkan, and OpenVINO.
- `OpenVINO`, `OpenVINO GenAI`, and `Optimum Intel` appear to be the main maintained Intel inference path across CPU, GPU, and NPU.
- `IPEX-LLM` contains useful Linux GPU, vLLM, and NPU setup material, but the repo is archived and explicitly warns about known security issues, so it should be treated as a reference source rather than a default recommendation.
- vLLM support in the current local references looks integration-specific rather than something we should assume is natively upstream and straightforward on Intel hardware.
- Intel support may vary sharply by model family and kernel coverage, especially relative to CUDA-first and HIP-enabled ecosystems.

## 3. Research passes by source repo

- Review [`llama.cpp/`](llama.cpp/) docs for SYCL, Vulkan, and OpenVINO backend support on Linux.
- Review [`reference/openvino/`](reference/openvino/) docs for install paths, supported devices, GenAI positioning, and Intel GPU/NPU runtime requirements.
- Review [`reference/openvino.genai/`](reference/openvino.genai/) docs, samples, and tests for runtime APIs, supported model families, and NPU-specific evidence.
- Review [`reference/optimum-intel/`](reference/optimum-intel/) docs for Hugging Face export, inference, and quantization flows through OpenVINO.
- Review [`reference/ipex-llm/`](reference/ipex-llm/) docs for Arc, Xe, NPU, and vLLM setup claims on Linux, while clearly separating useful guidance from archived-project risk.
- Review tracked local snapshots in [`reference/`](reference/) for supporting claims and point-in-time references that may not match current upstream docs.

## 4. Questions we want to answer

- What is the actual Linux support story for Arc dGPU, Xe-family iGPU, and Intel NPU across our target frameworks?
- Which paths are upstream-native, which are Intel-specific integrations, and which depend on archived or forked projects?
- What distro, kernel, driver, and userspace runtime combinations are explicitly documented?
- When are `i915`, `xe`, Level Zero, oneAPI, OpenCL, Vulkan, OpenVINO, or other components actually required?
- Which environment variables, user-group changes, or device-selection flags are required versus merely recommended?
- What are the documented model, precision, or backend limitations for each path?
- Which model families work well enough to document versus merely compile or run experimentally?
- Where does Intel have real kernel/backend coverage, and where is support materially behind CUDA or HIP ecosystems?

## 5. Linux requirements matrix

- Define the Linux distributions we care about first, starting with what upstream docs actually mention.
- Extract documented kernel requirements and special cases, including `force_probe` and any Arc/iGPU generation-specific caveats.
- Map driver and package requirements for Arc dGPU, Xe-family iGPU, and NPU.
- Record required group memberships such as `render` and `video`.
- Record verification tools such as `clinfo`, `vulkaninfo`, and `xpu-smi`.
- Record runtime environment variables such as `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS` and backend device selectors.

## 6. Framework and backend research tracks

### PyTorch

- Determine the current Intel story for PyTorch inference on Linux for Arc and Xe graphics.
- Separate plain upstream PyTorch, Intel-specific PyTorch paths, `torch.compile` with OpenVINO, and any other Intel runtime layers.
- Define the minimal device-discovery and inference examples we should test later.
- Record any documented precision limits, unsupported ops, or performance caveats.
- Evaluate model-family coverage through PyTorch/Hugging Face for standard decoder LLMs.
- Evaluate model-family coverage through PyTorch/Hugging Face for newer hybrid architectures such as Nemotron-style Mamba2 hybrids and Qwen 3.5 gated-delta/hybrid variants.
- Evaluate model-family coverage through PyTorch/Hugging Face for multimodal models.
- Evaluate model-family coverage through PyTorch/Hugging Face for ASR models such as Whisper and Fast Conformer variants.
- Evaluate model-family coverage through PyTorch/Hugging Face for TTS models.
- Record where support depends on generic ATen coverage versus custom kernels or vendor-specific fused paths.

### vLLM

- Determine whether Linux Intel support is via upstream vLLM, an Intel-maintained fork or branch, the OpenVINO backend, or some combination of those.
- Extract the exact installation path that the docs currently describe.
- Record any documented model restrictions, backend limits, warm-up behavior, or tuning knobs.
- Decide whether we need to add `vllm` or `vllm-openvino` as additional reference repos later.
- Track how Intel support compares to NVIDIA and AMD paths, especially around custom kernels, attention backends, quantization paths, and first-class maintenance.

### SGLang

- Determine whether SGLang has any meaningful Intel GPU or NPU support story on Linux or whether it is effectively CUDA-first today.
- Record whether Intel usage, if any, depends on generic PyTorch execution, OpenVINO, custom forks, or unsupported paths.
- Compare the practical support story against vLLM so we can explain where Intel is clearly behind.

### llama.cpp

- Compare SYCL, Vulkan, and OpenVINO as Linux backend options for Intel hardware.
- Extract build prerequisites, build flags, runtime device-selection flags, and verification commands for each backend.
- Record which backends are explicitly documented as working on Arc dGPU, Xe-family iGPU, and NPU.
- Record model-format, precision, and first-run warm-up caveats where documented.
- Determine how far llama.cpp can serve as an escape hatch when PyTorch-native or serving-engine-native Intel support is weak.

### OpenVINO, OpenVINO GenAI, and Optimum Intel

- Determine when plain `openvino` is enough and when `optimum-intel` adds value for the workflows we care about.
- Record the OpenVINO-supported device story for GPU and NPU on Linux.
- Determine whether OpenVINO should be treated as a primary path for PyTorch-adjacent inference and/or as the Intel path for vLLM.
- Determine where OpenVINO GenAI should be treated as a first-class runtime path rather than just an OpenVINO adjunct.
- Identify which OpenVINO GenAI or Optimum Intel flows are worth documenting in the first public version of this repo.
- Determine which model types are actually covered well by OpenVINO-based paths beyond standard text LLMs, especially multimodal, ASR, and TTS workloads.
- Determine whether OpenVINO GenAI's continuous batching, speculative decoding, embeddings, and rerank features deserve their own section in the public docs.
- Decide how much of `reference/openvino.genai/tools/llm_bench/` and `reference/openvino.genai/tools/who_what_benchmark/` should become part of the repo's default validation story.

### IPEX-LLM

- Extract the Linux Arc, Xe, B580, NPU, and vLLM setup paths documented in the repo.
- Record exact kernel, oneAPI, driver, package, and environment variable requirements that appear in its Linux docs.
- Identify which parts are still useful as evidence and which parts are likely stale because the project is archived.
- Keep any eventual recommendation conservative because of the archive status and security warning.
- Use it to identify missing coverage in current maintained Intel stacks, especially around model types or serving paths that Intel previously invested in.

## 7. Evidence capture and conflict tracking

- Create source notes that point to exact files in each repo for important claims.
- Record where different repos describe conflicting support status, requirements, or recommended stacks.
- Separate hard requirements from tuning advice and optional optimizations.
- Track unknowns that can only be resolved by local validation later.
- Track whether a claimed path is upstream-maintained, vendor-maintained, fork-specific, archived, or effectively unsupported.
- Track whether support is model-generic or limited to specific architectures, kernels, or fused operators.

## 8. Validation plan for later

- Populate [`00-setup/`](00-setup/) with system inventory capture, driver verification, env bring-up, and smoke-test scripts.
- Populate [`01-hardware/`](01-hardware/) with baseline memory-bandwidth, compute, and telemetry workflows.
- Pick a small model set for smoke tests once the docs review stabilizes.
- Define how we will verify that inference is actually using the intended Intel device.
- Define one minimal validation path per framework or backend.
- Record warm-up expectations, likely first-run delays, and common failure modes.
- Leave benchmarking as a later phase until setup reliability is clear.
- Include at least one representative standard LLM in the validation set.
- Include at least one representative hybrid LLM architecture in the validation set.
- Include at least one representative multimodal model in the validation set.
- Include at least one representative ASR model in the validation set.
- Include at least one representative TTS model in the validation set.

## 9. Documentation outputs we expect to write

- Expand [`README.md`](README.md) into a concise Linux-first landing page once the research solidifies.
- Add a support matrix that separates Arc dGPU, Xe-family iGPU, and NPU.
- Add a model-family matrix that separates standard LLMs, hybrid architectures, multimodal, ASR, and TTS.
- Add an ecosystem comparison section for PyTorch/HF, OpenVINO, vLLM, SGLang, and llama.cpp.
- Add framework or backend-specific setup documents if the root README becomes too dense.
- Add a troubleshooting section based on both docs conflicts and real local failures.
- Keep a clear distinction between confirmed guidance, tentative guidance, and unresolved questions.
