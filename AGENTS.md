# intel-inference Agent Guide

This repository is a Linux-first research and validation repo for the Intel AI/ML toolchain, with a focus on modern Intel Arc GPUs and adjacent Intel NPU paths.

Instruction precedence: platform, system, and developer instructions override this file.

## Repo Intent

The point of this repo is not just to collect links. It is to turn scattered Intel ecosystem claims into:

- documented setup steps
- validated machine bring-up
- reproducible benchmark procedures
- explicit support and limitation notes

Keep docs honest. Separate:

- docs-derived
- locally validated
- benchmarked

Do not blur those states.

## Primary Docs

Read the relevant docs before changing direction:

- [README.md](README.md): repo scope and current high-level guidance
- [IMPLEMENTATION.md](IMPLEMENTATION.md): install and environment strategy
- [ANALYSIS.md](ANALYSIS.md): support synthesis and evidence framing
- [TESTING.md](TESTING.md): benchmark methodology and staged test plan
- [TODO.md](TODO.md): remaining research backlog
- [00-setup/STATUS.md](00-setup/STATUS.md): current machine bring-up status

## Repo Layout

- `00-setup/`: driver, toolchain, env, and smoke-test bring-up
- `01-hardware/`: low-level bandwidth, compute, and telemetry work
- `reference/`: pinned upstream repos and captured external reference material
- `llama.cpp/`: upstream submodule for local backend work

Captured logs go under ignored `results/` directories. Commit-safe machine summaries go under tracked `systems/`.

## Working Rules

- Run `git status -sb` before editing and before committing.
- Keep work incremental and testable.
- Stage only files for the active task.
- Leave unrelated dirty files alone.
- Do not rewrite or delete ignored result logs unless the task requires it.
- Do not claim support for a model, backend, quant, or device unless the claim is backed by either upstream source evidence or local validation.
- If something works only with a workaround, document it as `works with workaround`, not `works`.

## Validation Standards

Minimum expectations by change type:

- Docs-only updates: verify the docs still match the current repo state and referenced scripts.
- `00-setup` shell changes: `bash -n 00-setup/*.sh` and rerun the relevant setup or verify scripts.
- Machine-profile changes: refresh with `./00-setup/collect-system-info.sh --system-id <id> --write-tracked-summary`.
- oneAPI changes: run `./00-setup/verify-oneapi.sh` and a sourced `oneapi-env.sh` smoke check.
- GPU validation changes: run `./00-setup/verify-gpu-stack.sh`.
- NPU validation changes: run `./00-setup/verify-npu-stack.sh`, and if applicable rerun after `source ./00-setup/npu-env.sh`.
- Benchmark-plan changes: keep [TESTING.md](TESTING.md) aligned with the actual script and results layout.

If a step requires `sudo`, hardware access, or manual intervention, say so explicitly.

## Environment And Benchmark Rules

- Keep separate envs per stack. Do not collapse everything into one Python env.
- Treat `intel-inf-openvino` as the minimal OpenVINO runtime baseline.
- Treat `intel-inf-openvino-genai` as the GenAI runtime baseline.
- Treat `intel-inf-optimum-openvino` as the heavier Hugging Face / export / Optimum env.
- Treat `intel-inf-torch-xpu` as the upstream PyTorch XPU baseline.
- Keep benchmark conclusions tied to the exact machine, distro, kernel, driver/runtime versions, and env versions.
- Prefer measured baselines before exploratory stacks. Validate maintained paths first.

## Reference And Submodule Rules

- `reference/` upstream repos are for research and comparison; treat them as read-only unless the task is explicitly to update submodule pins.
- Do not patch vendored upstream code casually.
- Keep captured external docs in `reference/` and index them in `reference/README.md` when relevant.
- The `llama.cpp/` submodule has its own [AGENTS.md](llama.cpp/AGENTS.md). Respect it when working inside that submodule.

## Current Known Repo-Specific Caveats

- On the current Arch test machine, OpenVINO NPU requires `source ./00-setup/npu-env.sh` because the Intel NPU userspace library is not on the default loader path.
- On the current Arch test machine, oneAPI tooling is installed but not on the default shell `PATH`; source `./00-setup/oneapi-env.sh` before SYCL-native builds.
- `xpu-smi` is currently non-authoritative on this machine. Do not use `xpu-smi discovery` failure alone to claim the GPU stack is broken when OpenCL, Vulkan, PyTorch XPU, and OpenVINO are working.

## Git Rules

- Never use `git add .`, `git add -A`, or `git commit -a`.
- Review staged files with `git diff --staged --name-only` and `git diff --staged` before committing.
- Commit completed logical units promptly.
- Use concise conventional prefixes when they fit: `docs:`, `feat:`, `fix:`, `test:`, `refactor:`, `chore:`.

## Communication

- Lead with verified results.
- Distinguish verified behavior from inference.
- When reporting failures, include the actual failure mode, not just “it doesn’t work”.
- When behavior is distro-specific or machine-specific, say that directly.
