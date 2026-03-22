# Intel AI/ML Toolchain Research

This repository is for researching and documenting the state of the AI/ML toolchain on Intel hardware on Linux, with a focus on modern Intel Arc GPUs.

Initial scope:

- Modern Intel hardware, with a focus on Intel Arc discrete GPUs and Intel integrated graphics based on Xe, Xe2, and Xe3
- Intel NPU support where relevant
- Linux-focused; Linux is the only platform in scope for now
- Inference first
- Training and general AI/ML development are secondary interests
- PyTorch, vLLM, llama.cpp, and adjacent serving/runtime paths such as OpenVINO-backed flows and SGLang where relevant
- Model coverage should include standard LLMs, newer hybrid architectures, multimodal models, ASR, and TTS where Intel support meaningfully differs
- Reproducible setup notes, validation steps, and known limitations

This is an early scaffold. The goal right now is to organize the work, collect references, and turn research into tested guidance. We should treat anything not explicitly validated in this repo as unconfirmed.

Planned contents:

- Hardware and platform support matrix
- Linux driver and runtime setup guidance
- Framework-specific setup for PyTorch, vLLM, llama.cpp, and OpenVINO-related flows where relevant
- Model-family coverage notes for text, multimodal, speech, and TTS workloads
- Backend and kernel support notes, especially where Intel support differs from CUDA- and HIP-first ecosystems
- Minimal smoke tests to confirm the stack is working
- Troubleshooting notes for common failure modes

Repository layout:

- [`llama.cpp/`](llama.cpp/): pinned upstream submodule used for llama.cpp backend experiments
- [`TODO.md`](TODO.md): research backlog and documentation checklist
- [`reference/`](reference/): tracked source material plus pinned upstream reference submodules

If cloning this repository fresh, initialize the pinned upstream checkouts with:

```bash
git clone --recurse-submodules <repo-url>
```

or, in an existing clone:

```bash
git submodule update --init --recursive
```

Next step is to validate the support matrix and software stack for each framework before writing prescriptive setup instructions.
