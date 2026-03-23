# 03-openvino

This directory is for OpenVINO-family runtime testing after the lower-level hardware and PyTorch XPU operator phases.

The point of this stage is to answer:

- which OpenVINO-family envs are actually healthy on this machine
- which OpenVINO devices are visible and usable
- how CPU, GPU, and NPU compare on simple OpenVINO graphs before we involve full models
- when OpenVINO GenAI and Optimum Intel are ready for later model-level testing

## What belongs here

- OpenVINO env and device checks
- OpenVINO GenAI env checks
- Optimum Intel env checks
- synthetic compile / first-run / warm-run OpenVINO benchmarks
- later model export, compile-cache, and runtime-level benchmark scripts

## Current script set

The first practical pass for this repo is:

- `run-env-checks.sh`
  - checks the `openvino`, `openvino-genai`, and `optimum-openvino` envs
  - records device visibility and key import health
- `run-device-bench.sh`
  - benchmarks simple OpenVINO graphs across `CPU`, `GPU`, and `NPU`
  - reports cold compile, warm compile, first infer, and warm infer timings
- `run-suite.sh`
  - ties the env checks and device benchmark together

## Current measurement stance

For now:

- this phase is still synthetic and env-first, not yet model-first
- device benchmarking uses static-shape `float16` graphs
- NPU results depend on the Arch loader-path workaround from [00-setup/npu-env.sh](/home/lhl/github/lhl/intel-inference/00-setup/npu-env.sh)
- OpenVINO GenAI and Optimum Intel are currently checked for env health, not yet benchmarked on real models here

That means this directory is the bridge from setup/operator validation into real OpenVINO runtime and model tests, not the final OpenVINO performance story.

## Benchmark environments

This phase deliberately splits env health checks from runtime benchmarking:

- `run-env-checks.sh`
  - `intel-inf-openvino`
  - `intel-inf-openvino-genai`
  - `intel-inf-optimum-openvino`
- `run-device-bench.sh`
  - `intel-inf-openvino`
  - requires `source ./00-setup/npu-env.sh` on this Arch machine if `NPU` should be visible
- `run-suite.sh`
  - combines the above, so the full phase is not one single env

That split matters because `optimum-intel` pulls in a much heavier Hugging Face and `torch` stack than the minimal OpenVINO runtime envs.

## Current quick-pass results

The current initial validation run is from `./03-openvino/run-suite.sh --quick` on the tracked Lunar Lake machine:

- system profile:
  - [lunarlake-ultra7-258v-32gb.md](/home/lhl/github/lhl/intel-inference/00-setup/systems/lunarlake-ultra7-258v-32gb.md)
- run date:
  - March 23, 2026

Headline env and device results:

- env health:
  - `intel-inf-openvino`: `openvino 2026.0.0`, devices `CPU`, `GPU`, `NPU`
  - `intel-inf-openvino-genai`: `openvino 2026.0.0`, `openvino-genai 2026.0.0.0`, devices `CPU`, `GPU`, `NPU`
  - `intel-inf-optimum-openvino`: `openvino 2026.0.0`, `optimum-intel 1.27.0`, devices `CPU`, `GPU`, `NPU`
- pipeline surface in `intel-inf-openvino-genai`:
  - `LLMPipeline=True`
  - `WhisperPipeline=True`
  - `Text2SpeechPipeline=True`

Headline device-benchmark results from the current quick pass:

- CPU:
  - `matmul 256`: warm infer median about `0.313 ms`, rough median throughput about `0.107 TOPS`
  - `matmul 512`: warm infer median about `2.367 ms`, rough median throughput about `0.113 TOPS`
  - `mlp 512`: warm infer median about `7.536 ms`, rough median throughput about `0.285 TOPS`
- GPU:
  - `matmul 256`: warm infer median about `0.196 ms`, rough median throughput about `0.171 TOPS`
  - `matmul 512`: warm infer median about `0.380 ms`, rough median throughput about `0.707 TOPS`
  - `mlp 512`: warm infer median about `0.465 ms`, rough median throughput about `4.617 TOPS`
- NPU:
  - `matmul 256`: warm infer median about `16.888 ms`, rough median throughput about `0.002 TOPS`
  - `matmul 512`: warm infer median about `24.612 ms`, rough median throughput about `0.011 TOPS`
  - `mlp 512`: warm infer median about `38.660 ms`, rough median throughput about `0.056 TOPS`

Current interpretation:

- the OpenVINO family envs are healthy on this machine once the Arch NPU loader-path workaround is in place
- `GPU` is already the obvious fast path for these small synthetic `float16` graphs
- `NPU` is now validated for compile and infer, but these quick synthetic results are still far below GPU throughput and should be treated as path validation, not a claim about real-model competitiveness
- OpenVINO GenAI is no longer hypothetical here; the env has the LLM, Whisper, and TTS pipelines we care about for later model-level testing

## Expected work products

Current or expected files here:

- `common.sh`
- `env-check.py`
- `run-env-checks.sh`
- `openvino-device-bench.py`
- `run-device-bench.sh`
- `run-suite.sh`
- `results/`

## Related docs

- [IMPLEMENTATION.md](/home/lhl/github/lhl/intel-inference/IMPLEMENTATION.md)
- [TESTING.md](/home/lhl/github/lhl/intel-inference/TESTING.md)
- [00-setup/STATUS.md](/home/lhl/github/lhl/intel-inference/00-setup/STATUS.md)
