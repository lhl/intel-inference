# vLLM Issue #35638 XPU Support Comment

Source URL:

- <https://github.com/vllm-project/vllm/issues/35638#issuecomment-3991459508>

Capture metadata:

- issue title: `[Question][XPU]: Best practices and optimized arguments for running 30B+ models on Intel Arc B580 (Dual GPU) via vLLM-XPU`
- issue opened: March 1, 2026
- captured comment author: `jikunshang`
- captured comment date: March 3, 2026
- local capture date for this note: March 26, 2026

Status and evidence class:

- this is a maintainer/collaborator support comment on an upstream GitHub issue
- treat it as support-discussion evidence, not as a merged-upstream guarantee and not as local validation in this repo

Quoted support direction from the comment:

- recommends Intel downstream `intel/vllm`
- recommends `intel/llm-scaler`

Main claims from the comment:

- in `v0.16.0`, XPU switched from `ipex` kernels to `vllm-xpu-kernels`
- FP8 KV cache was not fully supported yet in `vllm-xpu-kernels`
- if FP8 KV cache is required, the comment suggests trying `v0.15.0` or earlier
- `VLLM_XPU_FLASH_ATTN` was described as likely not used for XPU
- `Qwen3-next` and `Qwen3.5` still had a `flash_attention` backend gap at the time of the comment
- the suggested direction for that gap was `TRITON_ATTN` once `vllm-project/vllm#33657` merged
- recommended XPU block sizes were `64/128`
- `ONEAPI_DEVICE_SELECTOR` was described as mainly card selection, with `ZE_AFFINITY_MASK` preferred for control
- `mp` was described as usually performing better than Ray from test experience
- no major mandatory Xe2-specific env vars were called out beyond known issues in `dockerfile.xpu`

Cross-checks against current upstream material as of March 26, 2026:

- upstream XPU install docs already require `vllm-xpu-kernels` and replacing generic `triton` with `triton-xpu`
- upstream XPU install docs already document single-node pipeline parallel with `--distributed-executor-backend=mp`
- local upstream source in `05-vllm/vllm/vllm/platforms/xpu.py` already sets `ZE_AFFINITY_MASK` as the XPU device-control env var and defaults XPU cache block size to `64`
- `vllm-project/vllm#33657` exists as `[XPU] Support Qwen3-next/Qwen3.5`; on March 26, 2026 it is still a draft PR rather than merged mainline support
- `intel/llm-scaler` currently describes itself as an Intel downstream GenAI solution that leverages standard frameworks such as `vLLM`, rather than as upstream `vLLM` itself

Repo reading:

- this reference is strong evidence that Intel downstream remains the recommended path for newer Xe2 and larger-model B580 deployments
- it is not evidence that upstream XPU has already closed the Qwen3.5 / flash-attention gap
- it should be read alongside this repo's own local validation, which currently comes from a different machine class and should not be generalized to dual B580 without direct testing
