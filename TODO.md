# TODO

This file tracks the research and validation work needed before the README becomes a real setup guide.

## 1. Define the support matrix

- Identify which Intel GPUs we want to explicitly cover.
- Split coverage between Arc discrete GPUs and Xe-family integrated GPUs.
- Confirm the product families and naming we will use for Xe, Xe2, and Xe3.
- Decide whether the first supported target is Linux only or Linux plus Windows.
- Define which distros, kernels, and package managers we want to test first.

## 2. Map the software stack

- Document which driver stack each target needs.
- Research the role of `i915` vs `xe` on Linux and when each applies.
- Confirm required user-space components such as Level Zero, oneAPI, SYCL, OpenCL, Vulkan, and OpenVINO where relevant.
- Identify what must be installed from distro packages versus vendor packages versus source builds.
- Determine which environment variables or runtime flags are actually required.

## 3. PyTorch research

- Confirm the current Intel-supported path for PyTorch inference on Arc and Xe graphics.
- Determine whether plain upstream PyTorch is sufficient or whether Intel extensions are required.
- Validate device discovery, tensor placement, and a minimal inference example.
- Record known unsupported ops, precision constraints, and performance caveats.
- Define a minimal smoke test script for the README.

## 4. vLLM research

- Confirm whether vLLM currently supports Intel Arc or Xe graphics directly, experimentally, or only through specific branches or integrations.
- Identify the exact installation path: pip packages, source build, patches, or unsupported.
- Validate one small-model inference flow if support exists.
- Record any backend limitations, model restrictions, or missing features.
- Decide whether vLLM belongs in the first public setup guide or should remain marked as research.

## 5. llama.cpp research

- Compare the viable backends for Intel hardware: SYCL, Vulkan, OpenVINO, and CPU fallback.
- Determine which backends work on Arc dGPU and which work on Xe-family iGPU.
- Validate a reproducible build path for each backend we decide to cover.
- Record runtime flags, offload options, and model-format expectations.
- Define one minimal inference command and one simple validation checklist.

## 6. Validation and testing

- Pick a small common model set for smoke tests.
- Define how we verify the GPU is actually being used.
- Record expected outputs, logs, and device-query commands.
- Capture failure cases and recovery steps during setup.
- Decide what level of benchmarking belongs in this repo.

## 7. Documentation structure

- Expand `README.md` into a concise landing page once the research is validated.
- Create per-framework setup documents if the root README becomes too dense.
- Add a support matrix table once the target combinations are confirmed.
- Add a troubleshooting section based on real test failures.
- Keep a clear distinction between confirmed guidance and open questions.

## 8. Reference review

- Review the existing material in [`reference/`](reference/).
- Extract the relevant upstream installation paths from these directories.
- `reference/ipex-llm`
- `reference/llama.cpp-openvino`
- `reference/llama.cpp-sycl`
- `reference/llama.cpp-vulkan`
- `reference/openvino`
- `reference/optimum-intel`
- Note where upstream docs conflict, overlap, or leave gaps for Arc users.
