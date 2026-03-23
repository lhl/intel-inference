# 04-llama.cpp

This directory is for backend-specific `llama.cpp` testing after the OpenVINO-family runtime phase is working.

The point of this stage is to answer:

- how `Vulkan`, `SYCL`, and `OpenVINO` compare on the same Intel machine
- which quantizations and model families actually run on each backend
- how context depth, `-ngl`, `-fa`, batch, and ubatch change performance
- where `llama.cpp` works as the practical Intel fallback when other stacks are narrower

## Planned benchmark environments

This phase will not be one single env. The build and run context depends on backend:

- Vulkan backend
  - no `mamba` env by default
  - uses the system Vulkan stack and a dedicated `llama.cpp-*` build directory
- SYCL backend
  - no `mamba` env by default
  - requires `source ./00-setup/oneapi-env.sh`
  - uses a dedicated `llama.cpp-*` build directory
- OpenVINO backend
  - no `mamba` env by default for the compiled `llama.cpp` binary itself
  - may additionally use `intel-inf-openvino` for helper scripts or OpenVINO-side cross-checks
  - uses a dedicated `llama.cpp-*` build directory

That build split is intentional: the ignored `llama.cpp-*` directories are separate backend build trees, while [llama.cpp](/home/lhl/github/lhl/intel-inference/llama.cpp/) is the single pinned source checkout.

## Planned work

The initial `04-llama.cpp` pass should include:

- backend-specific build scripts
- `llama-bench` sweeps
- `llama-perplexity` quant regression checks
- `llama-cli` and `llama-server` smoke and latency tests
- explicit backend limitation notes for unsupported model types or quantizations

## Related docs

- [IMPLEMENTATION.md](/home/lhl/github/lhl/intel-inference/IMPLEMENTATION.md)
- [TESTING.md](/home/lhl/github/lhl/intel-inference/TESTING.md)
- [03-openvino/README.md](/home/lhl/github/lhl/intel-inference/03-openvino/README.md)
