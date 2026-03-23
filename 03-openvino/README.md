# 03-openvino

This directory is for OpenVINO-family runtime testing after the lower-level hardware and PyTorch XPU operator phases.

The point of this stage is to answer:

- which OpenVINO-family envs are actually healthy on this machine
- which OpenVINO devices are visible and usable
- which real models export cleanly on the maintained Optimum/OpenVINO path
- how OpenVINO GenAI behaves on `GPU` and `NPU` for LLM and Whisper workloads
- whether the shared OpenAI-compatible benchmark path is ready for later `llama.cpp` and `vLLM` comparison

## Current script set

- `run-env-checks.sh`
  - checks the `openvino`, `openvino-genai`, and `optimum-openvino` envs
  - records device visibility and key import health
- `run-device-bench.sh`
  - benchmarks simple OpenVINO graphs across `CPU`, `GPU`, and `NPU`
  - reports cold compile, warm compile, first infer, and warm infer timings
- `prepare-samples.sh`
  - downloads the tracked Whisper sample audio used by this phase
- `export-models.sh`
  - exports supported LLM and ASR checkpoints into `03-openvino/models/`
  - prefers complete local Hugging Face snapshots and falls back to HF repo IDs when needed
- `run-llm-smoke.sh`
  - runs direct OpenVINO GenAI LLM generation on exported models
- `run-openai-server.sh`
  - serves an exported OpenVINO GenAI LLM behind a minimal OpenAI-compatible `/v1/chat/completions` API
- `run-llm-openai-bench.sh`
  - benchmarks the temporary OpenAI-compatible server with the shared client in [benchmarks/openai_api_bench.py](/home/lhl/github/lhl/intel-inference/benchmarks/openai_api_bench.py)
- `run-whisper-smoke.sh`
  - runs direct OpenVINO GenAI Whisper transcription on exported models
- `run-model-suite.sh`
  - ties sample prep, export, LLM smoke, OpenAI-compatible LLM benchmark, and Whisper smoke together
- `run-suite.sh`
  - ties env checks and synthetic device benchmarking together
  - optionally runs `run-model-suite.sh` via `--with-models`

## Benchmark environments

This phase deliberately splits export, runtime, and benchmark-client contexts:

- `run-env-checks.sh`
  - `intel-inf-openvino`
  - `intel-inf-openvino-genai`
  - `intel-inf-optimum-openvino`
- `run-device-bench.sh`
  - `intel-inf-openvino`
- `export-models.sh`
  - `intel-inf-optimum-openvino`
- `run-llm-smoke.sh`
  - `intel-inf-openvino-genai`
- `run-openai-server.sh`
  - `intel-inf-openvino-genai`
- `run-llm-openai-bench.sh`
  - server in `intel-inf-openvino-genai`
  - benchmark client uses the repo-local stdlib-only [openai_api_bench.py](/home/lhl/github/lhl/intel-inference/benchmarks/openai_api_bench.py)
- `run-whisper-smoke.sh`
  - `intel-inf-openvino-genai`

On this Arch machine:

- `NPU` tests require `source ./00-setup/npu-env.sh`
- `CPU`, `GPU`, and `NPU` enumeration is already validated in [00-setup/STATUS.md](/home/lhl/github/lhl/intel-inference/00-setup/STATUS.md)

## Current export status

Current validated export targets on the maintained `optimum-intel 1.27.0` + `transformers 4.57.6` path:

- works:
  - `meta-llama/Llama-3.2-1B-Instruct`
  - `LiquidAI/LFM2-1.2B`
  - `openai/whisper-large-v3-turbo`
  - `openai/whisper-large-v3`
- tracked but currently blocked on this maintained export stack:
  - `Qwen/Qwen3.5-0.8B`
    - export fails because `transformers 4.57.6` does not recognize `model_type=qwen3_5`
  - `LiquidAI/LFM2-8B-A1B`
    - export fails because `transformers 4.57.6` does not recognize `model_type=lfm2_moe`

Those two blocked models remain important research targets for later phases, but they are not part of the default runnable `03-openvino` baseline because the current maintained Optimum stack does not export them cleanly.

## Important version-skew caveat

This needs to be called out explicitly:

- the current Intel/OpenVINO export story is not just "install the latest Transformers and go"
- on this machine, the maintained export path is `optimum-intel 1.27.0` with `transformers 4.57.6`
- that is not an arbitrary local pin; it is the top end of the currently supported `optimum-intel` dependency window
- the practical consequence is that OpenVINO runtime support and OpenVINO export support are different questions

In other words:

- once a model is already exported to OpenVINO IR, OpenVINO Runtime and OpenVINO GenAI do not need to track the newest Hugging Face `transformers` release
- but exporting a new Hugging Face checkpoint into a working OpenVINO IR artifact does depend on the `optimum-intel` plus `transformers` compatibility window
- when that window lags upstream model releases, Intel support effectively lags too, even if the runtime itself is healthy

That is exactly what happened in this phase:

- `qwen3_5` failed before runtime because the maintained `transformers 4.57.6` stack does not recognize it
- `lfm2_moe` failed for the same reason

So this repo should keep treating "requires a specific older Transformers range on the maintained Intel path" as a first-class limitation, not as a minor packaging detail.

## Current validated results

The current first real model pass is from March 23, 2026 on the tracked Lunar Lake machine:

- system profile:
  - [lunarlake-ultra7-258v-32gb.md](/home/lhl/github/lhl/intel-inference/00-setup/systems/lunarlake-ultra7-258v-32gb.md)

### Synthetic device microbench

From the earlier `./03-openvino/run-suite.sh --quick` synthetic pass:

- `GPU` is already the obvious fast path for small `float16` graphs
- `NPU` compiles and runs synthetic graphs, but the synthetic throughput is far below `GPU`
- those numbers should be treated as device-path validation, not as a real-model ranking

### LLM results on GPU

From the validated `./03-openvino/run-model-suite.sh --device GPU` pass:

- `meta-llama/Llama-3.2-1B-Instruct`
  - OpenAI-compatible benchmark median total latency: about `1955.3 ms`
  - median TTFT: about `157.8 ms`
  - median completion throughput: about `32.2 tok/s`
  - direct smoke prompts produced sensible outputs on all three prompts
- `LiquidAI/LFM2-1.2B`
  - OpenAI-compatible benchmark median total latency: about `3047.8 ms`
  - median TTFT: about `96.6 ms`
  - median completion throughput: about `25.8 tok/s`
  - runtime works, but output quality on the current prompt set is weak:
    - the short math prompt returned an empty completion
    - the longer prompts produced repetitive formatting-heavy outputs

That means the OpenAI-compatible serving and measurement path works for both models, but LFM2 currently looks much less usable than Llama on this prompt set even though it exports and runs.

### Whisper results on GPU and NPU

Validated Whisper smoke results:

- `openai/whisper-large-v3-turbo`
  - `GPU`: about `954.1 ms`
  - `NPU`: about `614.2 ms`
  - transcription: `" How are you doing today?"`
- `openai/whisper-large-v3`
  - `GPU`: about `1890.4 ms`
  - transcription: `" How are you doing today?"`

Both Whisper models now work through `openvino_genai.WhisperPipeline` after exporting them with `automatic-speech-recognition-with-past`.

### Model-level NPU validation

Model-level NPU status is now stronger than just synthetic graph enumeration:

- `meta-llama/Llama-3.2-1B-Instruct`
  - direct one-prompt smoke on `NPU` completed in about `866.8 ms`
  - returned the correct math answer
- `openai/whisper-large-v3-turbo`
  - direct `NPU` smoke completed in about `614.2 ms`
  - returned the expected transcription

So on this machine, OpenVINO GenAI on `NPU` is not just nominally visible. It runs at least one real small LLM and one real Whisper model successfully.

## Current interpretation

- OpenVINO, OpenVINO GenAI, and Optimum Intel are now validated here at the env, synthetic-device, and real-model levels.
- The shared OpenAI-compatible benchmark path is live for OpenVINO LLM testing and is ready to be reused in `04-llama.cpp` and `05-vllm`.
- The maintained OpenVINO export layer is still narrower than the desired model set:
  - `qwen3_5` and `lfm2_moe` are blocked before runtime because the pinned maintained `transformers` range does not recognize those architectures
- OpenVINO GenAI runtime quality is model-dependent:
  - Llama 3.2 1B looks healthy
  - LFM2 1.2B exports and runs but does not yet look production-healthy on the current prompt set
- Whisper on OpenVINO GenAI is now a real working path on both `GPU` and `NPU`

## Related docs

- [IMPLEMENTATION.md](/home/lhl/github/lhl/intel-inference/IMPLEMENTATION.md)
- [TESTING.md](/home/lhl/github/lhl/intel-inference/TESTING.md)
- [00-setup/STATUS.md](/home/lhl/github/lhl/intel-inference/00-setup/STATUS.md)
- [benchmarks/README.md](/home/lhl/github/lhl/intel-inference/benchmarks/README.md)
