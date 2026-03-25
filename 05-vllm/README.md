# 05-vllm

This directory is for `vLLM`-family testing after the OpenVINO baseline and `llama.cpp` backend phases are working.

The point of this stage is to answer:

- what upstream `vLLM` XPU can actually do on Intel Linux
- how `vllm-openvino` compares to upstream `vLLM` XPU on the same machine
- which models, quantizations, and serving features really work versus only looking nominally supported
- where the gap versus CUDA- and HIP-first stacks comes from: kernels, quantization, scheduling, or model-family coverage

## Benchmark environments

This phase keeps the two runtime tracks separate:

- upstream `vLLM` XPU
  - env: `intel-inf-vllm-xpu`
  - Python: `3.12`
  - local source checkout: `05-vllm/vllm` at `ffb5b32b5`
  - current validated package line:
    - `vllm 0.18.1rc1.dev53+gffb5b32b5.xpu`
    - `torch 2.10.0+xpu`
    - `transformers 4.57.6`
    - `triton-xpu 3.6.0`
- `vllm-openvino`
  - env: `intel-inf-vllm-openvino`
  - Python: `3.12`
  - local source checkout: `05-vllm/vllm-openvino` at `dbabf37`
  - current validated package line:
    - `vllm-openvino 0.8.4`
    - `vllm 0.8.4`
    - `openvino 2026.0.0`
    - `torch 2.6.0+cu124`
    - `transformers 4.51.3`

Do not treat the full phase as one single env or one single runtime.

One important packaging detail from live validation:

- `vllm-openvino` currently installs a large CUDA-oriented dependency set even on this Intel path
- upstream `vLLM` XPU needs a source build and currently requires repairing `triton-xpu` after install because the editable install can reintroduce generic `triton`
- Intel downstream `llm-scaler` is now tracked locally under `reference/llm-scaler` at `e874953`, but it should be treated as a separate downstream stack rather than as a proof that upstream `vLLM` XPU has feature parity

## Current validated state

### `vllm-openvino` on OpenVINO `GPU`

- validated serving path:
  - `meta-llama/Llama-3.2-1B-Instruct`
- current measured OpenAI-compatible benchmark result on this Lunar Lake machine:
  - median total latency: about `1990 ms`
  - median TTFT: about `53 ms`
  - median generation speed: about `28.1 tok/s`
- reproducibility caveat:
  - that initial Llama run succeeded, but an immediate rerun failed during the on-the-fly Optimum/OpenVINO export and IR load path
  - treat this as a promising first result, not yet as a stable benchmark baseline
- current model-family blockers on the maintained `vllm-openvino` stack:
  - `Qwen/Qwen3.5-0.8B`
    - fails before serve because `transformers 4.51.3` does not recognize `qwen3_5`
  - `LiquidAI/LFM2-1.2B`
    - fails before serve because `transformers 4.51.3` does not recognize `lfm2`

Current read:

- `vllm-openvino` is the cleaner Intel `vLLM`-family path for plain Llama-class text generation today
- it is still not robust enough to call production-ready on this machine
- it is also on a materially older stack than the upstream XPU path, and that older stack shows up directly as model-architecture lag

### Upstream `vLLM` XPU

- validated bring-up:
  - env imports cleanly
  - `torch.xpu` is available
  - `vllm` reports `XPUPlatform`
- memory-accounting baseline:
  - upstream XPU is not simply failing to see unified memory on this Lunar Lake machine
  - `torch.xpu.mem_get_info()` and the `vLLM` startup logs both report about `28.49 GiB` total XPU-visible memory
  - the practical issue is the runtime's startup and KV-cache budgeting behavior on a shared-memory iGPU, not a trivial "GTT is invisible" bug
- current Llama result:
  - the older default-path result was not stable enough to treat as a clean benchmark result
  - a later diagnostic rerun with `TRITON_ATTN` and tighter memory limits did reach `/health`, `/v1/models`, and returned a simple OpenAI-compatible chat completion on this Lunar Lake machine
  - a later short benchmark rerun on the active desktop session also completed `3/3` prompts successfully with the same general XPU path, but only after lowering the utilization target further to `0.05`
  - that short benchmark result was:
    - median total latency: about `2019 ms`
    - median TTFT: about `90 ms`
    - median generation speed: about `29.9 tok/s`
  - the first clean bootable/servable profile on this machine was:
    - `--max-model-len 1024`
    - `--gpu-memory-utilization 0.08`
    - `--kv-cache-memory-bytes 1073741824`
    - `--block-size 64`
    - `--attention-backend TRITON_ATTN`
    - `--enforce-eager`
- current XPU tuning result:
  - the first broad tuning pass did not find a stable "just lower memory" profile
  - explicit `kv_cache_memory_bytes` alone was not enough to produce a reliable boot path
  - `TRITON_ATTN` at `--gpu-memory-utilization 0.20` still failed the startup allocator check because free XPU memory was only about `3.7-4.1 GiB`, below the requested `5.7 GiB`
  - lowering the utilization target to `0.08` was the first configuration that got cleanly past allocator gating and into a real serve path
  - on the active desktop session, the same path still showed startup-memory variability
  - a retry at `0.08` later failed with only `0.38/28.49 GiB` free at the allocator check
  - lowering to `0.05` was enough to get a clean `3/3` short benchmark run on that noisier session
  - practical read:
    - startup memory pressure is one problem
    - backend selection and block-size control also matter on this iGPU
- current model-family findings:
  - `Qwen/Qwen3.5-0.8B`
    - gets past model recognition on the newer upstream stack
    - with the same low-memory Triton profile, it also gets through weight loading
    - engine init still fails during multimodal encoder profiling with `RuntimeError: Only XE2 cutlass kernel is supported currently.`
    - the log shows the main model path selecting Triton/GDN kernels, but the ViT or multimodal encoder path still using XPU `FLASH_ATTN`
  - `LiquidAI/LFM2-1.2B`
    - gets past model recognition on the newer upstream stack
    - engine init then fails with an internal KV-cache page-size assertion

Current read:

- upstream XPU is newer and clearly broader than `vllm-openvino` at the model-registration layer
- that broader coverage still does not translate into broad stable model-family coverage on this machine
- on this Lunar Lake iGPU, the default `gpu_memory_utilization` assumptions are too aggressive
- however, lowering `gpu_memory_utilization` alone is not enough
- the current XPU story is:
  - shared-memory startup budgeting is fragile
  - a small text-only Llama baseline is now possible with `TRITON_ATTN` plus aggressive low-memory settings
  - that same change does not fix `Qwen3.5` because the multimodal encoder still falls back into the XPU flash-attention path
- treat the XPU path here as exploratory rather than benchmark-ready

### Recent upstream support-discussion reference

This repo now tracks one useful upstream support-discussion reference for newer Xe2 and dual-Arc usage:

- docs/support-discussion-derived, not locally validated:
  - on March 3, 2026, `jikunshang` replied on upstream issue `vllm-project/vllm#35638` for a dual Arc B580 `v0.16.0` setup
  - the comment recommends Intel downstream `intel/vllm` and `intel/llm-scaler` for that workload tier
  - it says XPU `v0.16.0` had switched from `ipex` kernels to `vllm-xpu-kernels`
  - it says FP8 KV cache was not fully supported yet in `vllm-xpu-kernels`
  - it says `VLLM_XPU_FLASH_ATTN` is likely not the relevant XPU knob and points users toward `TRITON_ATTN` work for `Qwen3-next` and `Qwen3.5`
  - it recommends XPU block sizes `64/128`, prefers `ZE_AFFINITY_MASK` over `ONEAPI_DEVICE_SELECTOR` for card selection, and says `mp` usually performs better from test experience

How to read that reference:

- it is meaningful maintainer-side support guidance, but it is still a GitHub issue comment, not a stable feature contract
- it lines up with current upstream docs and code that already show `vllm-xpu-kernels`, `triton-xpu`, `ZE_AFFINITY_MASK`, and single-node pipeline parallel with `--distributed-executor-backend=mp`
- it also reinforces an important repo distinction:
  - `intel/llm-scaler` is not "upstream vLLM updated"
  - it is an Intel downstream stack built around `vLLM` and other frameworks with its own release cadence and model matrix

Reference note:

- [../reference/vllm-issue-35638-xpu-support-comment-2026-03-03.md](/home/lhl/github/lhl/intel-inference/reference/vllm-issue-35638-xpu-support-comment-2026-03-03.md)

### Recent upstream Arc 140V issue report

This repo now also tracks a newer upstream bug report that is directly relevant to Arc `140V` expectations:

- docs/issue-report-derived, not locally validated:
  - on March 22, 2026, upstream issue `vllm-project/vllm#37828` was opened for an Intel Core Ultra 7 `268V` with integrated Arc `140V`
  - the reported environment is Ubuntu `24.04.4` under WSL2, with `torch 2.10.0+xpu`, `triton-xpu 3.6.0`, and upstream `vllm 0.18.1rc1.dev27+g63f49b8bd.d20260322`
  - the user says they followed the official XPU install flow and then hit:
    - `RuntimeError: Only XE2 cutlass kernel is supported currently.`
  - the stack trace runs through the multimodal encoder or ViT XPU flash-attention path
  - as captured on March 26, 2026, the issue is still open, unassigned, and has no linked fix or milestone

How to read that reference:

- it is a user bug report, not maintainer guidance and not a general support statement for native Linux
- because it is WSL2-based, it should not be read as a one-to-one statement about this repo's Arch host
- it still matters because the failure signature matches this repo's own local upstream `Qwen3.5` multimodal error on a Lunar Lake integrated GPU
- current repo read:
  - do not treat upstream Arc `140V` multimodal support as solved
  - do not treat the existence of some working small text-only XPU paths as proof that Xe2 integrated-GPU model-family coverage is broadly complete

Reference note:

- [../reference/vllm-issue-37828-arc-140v-xe2-cutlass-2026-03-22.md](/home/lhl/github/lhl/intel-inference/reference/vllm-issue-37828-arc-140v-xe2-cutlass-2026-03-22.md)

### Intel Downstream `llm-scaler`

This repo now also tracks Intel's downstream `llm-scaler` stack locally:

- local reference checkout:
  - `reference/llm-scaler` at `e874953`
- docs-backed scope:
  - primary messaging is `Arc Pro B60`
  - the current `vllm` docs also include `B70` reference-perf artifacts and an `A770` supplement
  - there is no documented `Arc 140V` or Lunar Lake path in the public repo docs
- current repo read:
  - treat `llm-scaler` as a useful downstream Battlemage or workstation reference
  - do not blur it with upstream `vLLM` XPU status
  - if it works on this machine, that is a separate experiment result, not evidence that it is a documented target platform
- current host-side experiment state on this Arch Lunar Lake machine:
  - Docker-based bring-up is blocked from this shell because the local Docker daemon is not accessible without elevated privileges
  - a manual host-side reconstruction of the published stack did get as far as:
    - `torch 2.10.0+xpu`
    - `intel_extension_for_pytorch 2.10.10.post1+xpu`
    - `arctic_inference 0.1.1`
    - patched downstream `vllm 0.14.1.dev0+gb17039bcc.d20260325.xpu`
  - the first real serve attempt initially failed immediately with:
    - `ModuleNotFoundError: No module named 'vllm_xpu_kernels'`
  - that failure matches Intel's Dockerfile, which installs `vllm-xpu-kernels` separately in a later stage
  - a manual `vllm-xpu-kernels` build at commit `4c83144` first failed under local oneAPI `2025.0.4` during later XPU kernel targets
  - the concrete build errors on this host were:
    - missing `sycl::ext::oneapi::experimental::info::kernel_queue_specific::max_num_work_groups`
    - missing `sycl::ext::oneapi::experimental::work_group_scratch_size`
    - missing `sycl::ext::oneapi::experimental::architecture::intel_gpu_bmg_g31`
  - local compiler state:
    - active host compiler: `Intel(R) oneAPI DPC++/C++ Compiler 2025.0.4`
    - `vllm-xpu-kernels` README says: `Currently we use PyTorch 2.10, oneapi 2025.3.`
    - current Intel downstream `reference/llm-scaler/vllm/docker/Dockerfile` installs `intel-oneapi-dpcpp-ct=2025.2.0-517`
  - a side-by-side user install of oneAPI `2025.3.2` under `/home/lhl/opt/intel/oneapi-2025.3` removed that earlier compiler mismatch
  - the retried native build then got much further:
    - it cleared the old XE2 grouped-GEMM and `chunk_prefill` compile failures
    - it progressed through generated XE2 attention targets for head sizes `128`, `192`, and into `256`
    - but it remained a long native build and was no longer the fastest path to a runnable host stack
  - the practical host-side path turned out to be the published wheel:
    - `vllm-xpu-kernels 0.1.3.1`
    - wheel tag: `cp38-abi3-manylinux_2_28_x86_64`
  - with that wheel installed, downstream `vllm` on this machine reached three distinct states:
    - low-memory configs such as `--gpu-memory-utilization 0.05` or `0.10` still failed during KV-cache planning with `ValueError: No available memory for the cache blocks`
    - an overly aggressive config (`--gpu-memory-utilization 1.0`, `--max-model-len 256`) got past cache planning but hit `UR_RESULT_ERROR_DEVICE_LOST` during KV-cache tensor allocation
    - a middle config worked:
      - `--max-model-len 256`
      - `--gpu-memory-utilization 0.20`
      - `--block-size 64`
      - `--enforce-eager`
    - that successful config served `meta-llama/Llama-3.2-1B-Instruct` and completed a 3/3 short benchmark run with summary at:
      - [20260326T0648Z-llmscaler-wheel-xpu-llama32-shortbench-summary.json](/home/lhl/github/lhl/intel-inference/05-vllm/results/20260326T0648Z-llmscaler-wheel-xpu-llama32-shortbench-summary.json)
      - median total latency about `1215.6 ms`
      - median TTFT about `55.7 ms`
      - median generation speed about `27.6 tok/s`
  - one important failure mode from the same downstream stack:
    - `--block-size 16` is not viable here once requests actually run
    - the request-time failure was: `XeTLA ChunkPrefill FP8KV: only support block_size >= 64`

Current read:

- `llm-scaler` is not a documented `Arc 140V` target, but there is now one verified local text-serving path on this machine
- the fastest practical host-side path is not the full native kernel build; it is the downstream Python stack plus the published `vllm-xpu-kernels` wheel
- the working envelope is narrower than upstream `vLLM` XPU:
  - smaller context (`--max-model-len 256`)
  - `--gpu-memory-utilization 0.20`
  - `--block-size 64`
  - `--enforce-eager`
- treat this as `works with workaround`, not as a clean or documented downstream support story for Lunar Lake or Arc `140V`

## Comparison rules

- upstream `vLLM` XPU and `vllm-openvino` must be reported separately
- `vllm-openvino` results should be compared against the `03-openvino` baseline, not used to redefine what raw OpenVINO can do
- any quantization result must state the exact method and runtime path
- any unsupported architecture or kernel should be logged as a first-class finding, not hidden as a generic failure

## Scripts

Setup and validation:

- [../00-setup/setup-vllm-xpu-env.sh](/home/lhl/github/lhl/intel-inference/00-setup/setup-vllm-xpu-env.sh)
- [../00-setup/setup-vllm-openvino-env.sh](/home/lhl/github/lhl/intel-inference/00-setup/setup-vllm-openvino-env.sh)
- [run-env-checks.sh](/home/lhl/github/lhl/intel-inference/05-vllm/run-env-checks.sh)

Serving and benchmarking:

- [run-xpu-serve.sh](/home/lhl/github/lhl/intel-inference/05-vllm/run-xpu-serve.sh)
- [run-openvino-serve.sh](/home/lhl/github/lhl/intel-inference/05-vllm/run-openvino-serve.sh)
- [run-openai-bench.sh](/home/lhl/github/lhl/intel-inference/05-vllm/run-openai-bench.sh)
- shared client: [../benchmarks/openai_api_bench.py](/home/lhl/github/lhl/intel-inference/benchmarks/openai_api_bench.py)

## Recommended next work

- benchmark the now-working Llama `TRITON_ATTN` low-memory profile instead of treating it only as a boot diagnostic:
  - `--max-model-len 1024`
  - `--gpu-memory-utilization 0.05` on the current active desktop session
  - `--kv-cache-memory-bytes 1073741824`
  - `--block-size 64`
  - `--attention-backend TRITON_ATTN`
  - `--enforce-eager`
- rerun the downstream `llm-scaler` wheel-backed Llama path with a cleaner desktop session and higher context targets to see whether the working window extends beyond `--max-model-len 256`
- test whether downstream `llm-scaler` can avoid the current request-time `XeTLA ChunkPrefill FP8KV: only support block_size >= 64` limitation in any smaller-block profile, or whether `64` is the practical minimum on this path
- rerun the same profile in a cleaner session with fewer desktop GPU clients to see whether `0.08` is still a stable target when the device is less busy
- test whether any text-only Qwen 3.5 checkpoint avoids the current multimodal XPU kernel failure
- inspect whether the `Qwen3.5` multimodal encoder can be forced away from the current XPU `FLASH_ATTN` path, because `TRITON_ATTN` on the main model path alone is not enough
- if a source-built downstream stack is still desired, continue the now-unblocked oneAPI `2025.3` native `vllm-xpu-kernels` build rather than assuming the old `2025.0.4` compiler failure is still the active blocker
- test `vllm-openvino` on exported or older checkpoint families that fit its current `transformers 4.51` line
- only after the basic serve path is stable, start quantization and feature sweeps

## Related docs

- [TESTING.md](/home/lhl/github/lhl/intel-inference/TESTING.md)
- [03-openvino/README.md](/home/lhl/github/lhl/intel-inference/03-openvino/README.md)
- [00-setup/README.md](/home/lhl/github/lhl/intel-inference/00-setup/README.md)
- [IMPLEMENTATION.md](/home/lhl/github/lhl/intel-inference/IMPLEMENTATION.md)
