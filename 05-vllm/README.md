# 05-vllm

This directory is for `vLLM`-family testing after the OpenVINO baseline and `llama.cpp` backend phases are working.

The point of this stage is to answer:

- what upstream `vLLM` XPU can actually do on Intel Linux
- how `vllm-openvino` compares to upstream `vLLM` XPU on the same machine
- which models, quantizations, and serving features really work versus only looking nominally supported
- where the gap versus CUDA- and HIP-first stacks comes from: kernels, quantization, scheduling, or model-family coverage

## Benchmark environments

This phase should keep the two runtime tracks separate:

- upstream `vLLM` XPU
  - dedicated `mamba` env, likely `intel-inf-vllm-xpu`
- `vllm-openvino`
  - dedicated `mamba` env, likely `intel-inf-vllm-openvino`
  - may also need `source ./00-setup/npu-env.sh` if we test OpenVINO `NPU`

Do not treat the full phase as one single env or one single runtime.

## Planned work

The initial `05-vllm` pass should include:

- environment and import checks for upstream `vLLM` XPU
- environment and import checks for `vllm-openvino`
- small-model smoke tests on a shared source-format checkpoint
- explicit recording of unsupported models, unsupported quants, and missing kernels
- side-by-side TTFT, TPOT, throughput, and memory notes once the paths are stable

## Comparison rules

- upstream `vLLM` XPU and `vllm-openvino` must be reported separately
- `vllm-openvino` results should be compared against the `03-openvino` baseline, not used to redefine what raw OpenVINO can do
- any quantization result must state the exact method and runtime path
- any unsupported architecture or kernel should be logged as a first-class finding, not hidden as a generic failure

## Related docs

- [TESTING.md](/home/lhl/github/lhl/intel-inference/TESTING.md)
- [03-openvino/README.md](/home/lhl/github/lhl/intel-inference/03-openvino/README.md)
- [IMPLEMENTATION.md](/home/lhl/github/lhl/intel-inference/IMPLEMENTATION.md)
