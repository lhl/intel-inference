# Reference Index

This directory is used for two different kinds of material:

- Tracked reference documents and local snapshots that we want to keep in git
- Pinned upstream submodules used for source review and reproducible reference points
- Ignored local working directories used for backend-specific llama.cpp experiments

Tracked reference files:

- [`OpenVINO_Quick_Start_Guide.pdf`](OpenVINO_Quick_Start_Guide.pdf)
- [`pytorch-2-10torchao.html`](pytorch-2-10torchao.html)
- [`reddit-localllama-llamacpp-compute-memory-bandwidth-efficiency.md`](reddit-localllama-llamacpp-compute-memory-bandwidth-efficiency.md)
- [`reddit-localllama-testing-llamacpp-intel-xe2-igpu-core-ultra.md`](reddit-localllama-testing-llamacpp-intel-xe2-igpu-core-ultra.md)

Pinned upstream submodules in this directory:

- `ipex-llm`
- `openvino`
- `openvino.genai`
- `optimum-intel`
- `whisper.cpp`

Ignored local working directories:

- `llama.cpp-openvino`
- `llama.cpp-sycl`
- `llama.cpp-vulkan`

Notes:

- The main `llama.cpp` upstream checkout is tracked separately as a root-level submodule.
- `whisper.cpp` is tracked here as the speech-side sibling to `llama.cpp`, and it includes its own OpenVINO-backed Intel path.
- `openvino.genai` is tracked separately from the main OpenVINO repo because it gives us the maintained Whisper and SpeechT5 pipeline layer we care about for ASR and TTS.
- `openvino.genai` also includes `tools/llm_bench` and `tools/who_what_benchmark`, which are likely to become part of this repo's default Intel benchmarking and regression workflow.
- The `llama.cpp-*` directories are intentionally ignored so we can keep backend-specific working trees and build artifacts without polluting repository history.
