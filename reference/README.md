# Reference Index

This directory is used for two different kinds of material:

- Tracked reference documents and local snapshots that we want to keep in git
- Pinned upstream submodules used for source review and reproducible reference points
- Ignored local working directories used for backend-specific llama.cpp experiments

Tracked reference files:

- [`OpenVINO_Quick_Start_Guide.pdf`](OpenVINO_Quick_Start_Guide.pdf)
- [`pytorch-2-10torchao.html`](pytorch-2-10torchao.html)

Pinned upstream submodules in this directory:

- `ipex-llm`
- `openvino`
- `optimum-intel`

Ignored local working directories:

- `llama.cpp-openvino`
- `llama.cpp-sycl`
- `llama.cpp-vulkan`

Notes:

- The main `llama.cpp` upstream checkout is tracked separately as a root-level submodule.
- The `llama.cpp-*` directories are intentionally ignored so we can keep backend-specific working trees and build artifacts without polluting repository history.
