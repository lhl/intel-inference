# Intel AI/ML Toolchain Research

This repository is for researching and documenting the state of the AI/ML toolchain on Intel hardware, with a focus on modern Intel Arc GPUs.

Initial scope:

- Modern Intel hardware, with a focus on Intel Arc discrete GPUs and Intel integrated graphics based on Xe, Xe2, and Xe3
- Intel NPU support where relevant
- Inference first
- Training and general AI/ML development are secondary interests
- PyTorch, vLLM, and llama.cpp, plus OpenVINO-backed paths where relevant
- Reproducible setup notes, validation steps, and known limitations

This is an early scaffold. The goal right now is to organize the work, collect references, and turn research into tested guidance. We should treat anything not explicitly validated in this repo as unconfirmed.

Planned contents:

- Hardware and platform support matrix
- Driver and runtime setup guidance
- Framework-specific setup for PyTorch, vLLM, llama.cpp, and OpenVINO-related flows where relevant
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
