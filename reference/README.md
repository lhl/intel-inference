# Reference Index

This directory is used for two different kinds of material:

- Tracked reference documents and local snapshots that we want to keep in git
- Untracked local upstream checkouts used for source review, experiments, and backend-specific builds

Tracked reference files:

- [`OpenVINO_Quick_Start_Guide.pdf`](OpenVINO_Quick_Start_Guide.pdf)
- [`pytorch-2-10torchao.html`](pytorch-2-10torchao.html)

Ignored local checkouts:

- `ipex-llm`
- `llama.cpp-openvino`
- `llama.cpp-sycl`
- `llama.cpp-vulkan`
- `openvino`
- `optimum-intel`

Notes:

- The `llama.cpp-*` directories are separate local checkouts used for different backend builds and comparisons.
- These checkouts are intentionally ignored by git so we can keep local source trees and build artifacts without polluting the repository history.
