# 00-setup

This directory is for machine bring-up and environment validation.

The goal here is simple: before we benchmark anything, we should be able to prove that the Intel Linux stack is installed, visible, and usable.

## What belongs here

- system inventory capture
- sanitized tracked machine profiles for benchmark context
- driver verification for Intel GPU and optional Intel NPU
- oneAPI and SYCL verification when needed
- OpenVINO and OpenVINO GenAI env bring-up
- PyTorch XPU env bring-up
- `llama.cpp` backend-specific build and smoke-test notes
- saved setup artifacts such as version dumps and command outputs

## Exit criteria

We should not move on to hardware benchmarking until we have:

1. the exact machine, distro, kernel, and package versions recorded
2. GPU visibility confirmed with the relevant tools
3. NPU visibility confirmed if the machine has an NPU
4. oneAPI verified for SYCL work when that path is in scope
5. separate `mamba` or `conda` envs created for the stacks we actually plan to test
6. at least one minimal smoke test passing for each stack we want to benchmark later

## Expected work products

Current or likely files or scripts here:

- `collect-system-info.sh`
- `CHECKLIST.md`
- `systems/`
- `verify-gpu-stack.sh`
- `verify-npu-stack.sh`
- `verify-oneapi.sh`
- `setup-openvino-env.sh`
- `setup-openvino-genai-env.sh`
- `setup-torch-xpu-env.sh`
- `smoke-*.sh`

## Related docs

- [IMPLEMENTATION.md](/home/lhl/github/lhl/intel-inference/IMPLEMENTATION.md)
- [TESTING.md](/home/lhl/github/lhl/intel-inference/TESTING.md)
