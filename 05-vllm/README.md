# 05-vllm

This directory is for `vLLM`-family testing after the OpenVINO baseline and `llama.cpp` backend phases are working.

The point of this stage is to answer:

- what upstream `vLLM` XPU can actually do on Intel Linux
- how `vllm-openvino` compares to upstream `vLLM` XPU on the same machine
- which models, quantizations, and serving features really work versus only looking nominally supported
- where the gap versus CUDA- and HIP-first stacks comes from: kernels, quantization, scheduling, or model-family coverage

## Benchmark environments

This phase keeps the two runtime tracks separate:

- upstream `vLLM` XPU
  - env: `intel-inf-vllm-xpu`
  - Python: `3.12`
  - local source checkout: `05-vllm/vllm` at `ffb5b32b5`
  - current validated package line:
    - `vllm 0.18.1rc1.dev53+gffb5b32b5.xpu`
    - `torch 2.10.0+xpu`
    - `transformers 4.57.6`
    - `triton-xpu 3.6.0`
- `vllm-openvino`
  - env: `intel-inf-vllm-openvino`
  - Python: `3.12`
  - local source checkout: `05-vllm/vllm-openvino` at `dbabf37`
  - current validated package line:
    - `vllm-openvino 0.8.4`
    - `vllm 0.8.4`
    - `openvino 2026.0.0`
    - `torch 2.6.0+cu124`
    - `transformers 4.51.3`

Do not treat the full phase as one single env or one single runtime.

One important packaging detail from live validation:

- `vllm-openvino` currently installs a large CUDA-oriented dependency set even on this Intel path
- upstream `vLLM` XPU needs a source build and currently requires repairing `triton-xpu` after install because the editable install can reintroduce generic `triton`

## Current validated state

### `vllm-openvino` on OpenVINO `GPU`

- validated serving path:
  - `meta-llama/Llama-3.2-1B-Instruct`
- current measured OpenAI-compatible benchmark result on this Lunar Lake machine:
  - median total latency: about `1990 ms`
  - median TTFT: about `53 ms`
  - median generation speed: about `28.1 tok/s`
- reproducibility caveat:
  - that initial Llama run succeeded, but an immediate rerun failed during the on-the-fly Optimum/OpenVINO export and IR load path
  - an additional rerun under AC power and in a plain TTY session still did not produce a stable repeat result
  - treat this as a promising first result, not yet as a stable benchmark baseline
- current model-family blockers on the maintained `vllm-openvino` stack:
  - `Qwen/Qwen3.5-0.8B`
    - fails before serve because `transformers 4.51.3` does not recognize `qwen3_5`
  - `LiquidAI/LFM2-1.2B`
    - fails before serve because `transformers 4.51.3` does not recognize `lfm2`

Current read:

- `vllm-openvino` is the cleaner Intel `vLLM`-family path for plain Llama-class text generation today
- it is still not robust enough to call production-ready on this machine
- it is also on a materially older stack than the upstream XPU path, and that older stack shows up directly as model-architecture lag

### Upstream `vLLM` XPU

- validated bring-up:
  - env imports cleanly
  - `torch.xpu` is available
  - `vllm` reports `XPUPlatform`
- current Llama result:
  - `meta-llama/Llama-3.2-1B-Instruct` starts and serves requests on XPU, but the current run was not stable enough to treat as a clean benchmark result
  - with the shared three-prompt benchmark, `1/3` prompts completed and `2/3` returned HTTP `500`
  - reruns under AC power and in a plain TTY session did not cleanly resolve that instability
- current model-family findings:
  - `Qwen/Qwen3.5-0.8B`
    - gets past model recognition on the newer upstream stack
    - on this machine, engine init then fails in the multimodal attention path with `RuntimeError: Only XE2 cutlass kernel is supported currently.`
  - `LiquidAI/LFM2-1.2B`
    - gets past model recognition on the newer upstream stack
    - engine init then fails with an internal KV-cache page-size assertion

Current read:

- upstream XPU is newer and clearly broader than `vllm-openvino` at the model-registration layer
- that broader coverage does not yet translate into a stable small-model baseline on this machine
- on this Lunar Lake iGPU, `gpu_memory_utilization` needed to be lowered well below typical defaults; `0.10` to `0.15` is a more realistic starting range here than `0.6`

## Comparison rules

- upstream `vLLM` XPU and `vllm-openvino` must be reported separately
- `vllm-openvino` results should be compared against the `03-openvino` baseline, not used to redefine what raw OpenVINO can do
- any quantization result must state the exact method and runtime path
- any unsupported architecture or kernel should be logged as a first-class finding, not hidden as a generic failure

## Scripts

Setup and validation:

- [../00-setup/setup-vllm-xpu-env.sh](/home/lhl/github/lhl/intel-inference/00-setup/setup-vllm-xpu-env.sh)
- [../00-setup/setup-vllm-openvino-env.sh](/home/lhl/github/lhl/intel-inference/00-setup/setup-vllm-openvino-env.sh)
- [run-env-checks.sh](/home/lhl/github/lhl/intel-inference/05-vllm/run-env-checks.sh)

Serving and benchmarking:

- [run-xpu-serve.sh](/home/lhl/github/lhl/intel-inference/05-vllm/run-xpu-serve.sh)
- [run-openvino-serve.sh](/home/lhl/github/lhl/intel-inference/05-vllm/run-openvino-serve.sh)
- [run-openai-bench.sh](/home/lhl/github/lhl/intel-inference/05-vllm/run-openai-bench.sh)
- shared client: [../benchmarks/openai_api_bench.py](/home/lhl/github/lhl/intel-inference/benchmarks/openai_api_bench.py)

## Recommended next work

- rerun the upstream XPU Llama path with a tighter, explicit low-memory profile and a manual request sanity pass before trusting benchmark numbers
- test whether any text-only Qwen 3.5 checkpoint avoids the current multimodal XPU kernel failure
- test `vllm-openvino` on exported or older checkpoint families that fit its current `transformers 4.51` line
- only after the basic serve path is stable, start quantization and feature sweeps

## Related docs

- [TESTING.md](/home/lhl/github/lhl/intel-inference/TESTING.md)
- [03-openvino/README.md](/home/lhl/github/lhl/intel-inference/03-openvino/README.md)
- [00-setup/README.md](/home/lhl/github/lhl/intel-inference/00-setup/README.md)
- [IMPLEMENTATION.md](/home/lhl/github/lhl/intel-inference/IMPLEMENTATION.md)
