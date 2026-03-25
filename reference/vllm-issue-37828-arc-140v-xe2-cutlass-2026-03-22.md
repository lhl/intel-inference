# vLLM Issue #37828 Arc 140V XE2 Cutlass Error

Source URL:

- <https://github.com/vllm-project/vllm/issues/37828>

Capture metadata:

- issue title: `[Bug]: Intel ARC 140v not supported as XE2 cutlass kernel`
- issue opened: March 22, 2026
- captured issue author: `PterosDiacos`
- local capture date for this note: March 26, 2026

Status and evidence class:

- this is an upstream GitHub bug report from a user, not a maintainer support statement
- treat it as issue-report evidence, not as a support guarantee and not as local validation in this repo
- as captured on March 26, 2026, the issue is still open, unassigned, and has no milestone or linked PR

Main details from the issue:

- reported environment:
  - Ubuntu `24.04.4 LTS`
  - WSL2 kernel `6.6.87.2-microsoft-standard-WSL2`
  - Intel Core Ultra 7 `268V`
  - Intel Arc `140V`
  - `torch 2.10.0+xpu`
  - `triton-xpu 3.6.0`
  - `transformers 4.57.6`
  - upstream `vllm 0.18.1rc1.dev27+g63f49b8bd.d20260322`
- the user says they followed the official XPU install flow and then hit:
  - `RuntimeError: Only XE2 cutlass kernel is supported currently.`
- the stack trace in the report goes through the multimodal encoder or ViT flash-attention path:
  - `mm_encoder_attention.py`
  - `vit_attn_wrappers.py`
  - `vllm_xpu_kernels/flash_attn_interface.py`
- the user also says that before the February 2026 switch away from `ipex` and toward `vllm-xpu-kernels`, they had previously been able to run vLLM on this class of hardware

Repo reading:

- this is direct upstream evidence that Arc `140V` should not be treated as a settled upstream XPU path as of March 22, 2026
- it is still not a clean native-Linux support statement, because the reported environment is WSL2 rather than a plain Linux host
- it does matter for this repo because the error signature matches this repo's own local upstream `Qwen3.5` multimodal failure:
  - `RuntimeError: Only XE2 cutlass kernel is supported currently.`
- that matching error does not prove the exact same root cause across both environments, but it does strengthen the current repo reading that the Xe2 integrated-GPU multimodal flash-attention path is still incomplete upstream
