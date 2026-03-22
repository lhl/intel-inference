# Reddit Snapshot: Testing llama.cpp with Intel's Xe2 iGPU

Source URL: https://www.reddit.com/r/LocalLLaMA/comments/1gheslj/testing_llamacpp_with_intels_xe2_igpu_core_ultra/

Captured: 2026-03-22

Capture note:

- Direct command-line fetches from this environment hit Reddit's block page.
- The text below is a local markdown snapshot assembled from browser-accessible page content so we still have a useful tracked reference.

## Original post

Author: `randomfoo2`

> I have a Lunar Lake laptop and recently sat down and did some testing on how llama.cpp works with it.
>
> The 258V has 32GB of LPDDR5-8533, which has a theoretical maximum memory bandwidth of 136.5 GB/s.
>
> The 140V Xe2 GPU on the 258V has XMX units that Intel specs at 64 INT8 TOPS, implying a ballpark 32 FP16 TOPS.
>
> For my testing, I use Llama 2 7B Q4_0 as my standard benchmark. All testing was done with very up-to-date HEAD compiles of llama.cpp (`build: ba6f62eb (4008)`).

Primary result table:

```text
Backend        pp512 t/s   tg128 t/s   t/TFLOP   MBW %
CPU             25.05       11.59       52.74    30.23
Vulkan          44.65        5.54        1.40    14.45
SYCL FP32      180.77       14.39        5.65    37.53
SYCL FP16      526.38       13.51       16.45    35.23
IPEX-LLM       708.15       24.35       22.13    63.51
```

Explanation from the post:

> `pp` is prompt processing and is compute bound.
>
> `tg` is token generation and is generally memory bandwidth bound.
>
> I included a `t/TFLOP` compute efficiency metric for each backend and a `MBW %` metric for memory efficiency.

Setup notes from the post:

> The system itself is running CachyOS and a very new 6.12 kernel with `linux-firmware-git` and `mesa-git` for maximum Lunar Lake / Xe2 support.
>
> For CPU, I use `-t 4`, which uses all 4 non-hyperthreaded P-cores.
>
> For SYCL and IPEX-LLM you will need the Intel oneAPI Base Toolkit.
>
> I used version `2025.0.0` for SYCL, but IPEX-LLM's llama.cpp required `2024.2.1`.

Interpretation in the post:

> The IPEX-LLM results are much better than all the other backends.

## Update: k-quant fix

Later update from the same post:

> I reported the llama.cpp k-quant issue and can confirm that it is now fixed. It was broken with `ipex-llm[cpp] 2.2.0b20241031` and fixed in `2.2.0b20241105`.
>
> Even with `ZES_ENABLE_SYSMAN=1`, llama.cpp still complains about `ext_intel_free_memory` not being supported, but it doesn't seem to affect the run.

Updated sanity-check run:

```text
ZES_ENABLE_SYSMAN=1 ./llama-bench -m ~/ai/models/gguf/llama-2-7b.Q4_0.gguf
| 0 | [level_zero:gpu:0] | Intel Graphics [0x64a0] | 1.6 | 64 | 1024 | 32 | 15064M | 1.3.31294 |
| llama 7B Q4_0 | 3.56 GiB | 6.74 B | SYCL | 99 | pp512 | 705.09 ± 7.19 |
| llama 7B Q4_0 | 3.56 GiB | 6.74 B | SYCL | 99 | tg128 | 24.27 ± 0.19 |
build: 1d5f8dd (1)
```

Updated Q4_K_M run:

```text
ZES_ENABLE_SYSMAN=1 ./llama-bench -m ~/ai/models/gguf/llama-2-7b.Q4_K_M.gguf
| 0 | [level_zero:gpu:0] | Intel Graphics [0x64a0] | 1.6 | 64 | 1024 | 32 | 15064M | 1.3.31294 |
| llama 7B Q4_K - Medium | 3.80 GiB | 6.74 B | SYCL | 99 | pp512 | 595.64 ± 0.52 |
| llama 7B Q4_K - Medium | 3.80 GiB | 6.74 B | SYCL | 99 | tg128 | 20.41 ± 0.19 |
build: 1d5f8dd (1)
```

Updated Mistral 7B Q4_K_M run:

```text
ZES_ENABLE_SYSMAN=1 ./llama-bench -m ~/ai/models/gguf/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf
| 0 | [level_zero:gpu:0] | Intel Graphics [0x64a0] | 1.6 | 64 | 1024 | 32 | 15064M | 1.3.31294 |
| llama 7B Q4_K - Medium | 4.07 GiB | 7.25 B | SYCL | 99 | pp512 | 549.94 ± 4.09 |
| llama 7B Q4_K - Medium | 4.07 GiB | 7.25 B | SYCL | 99 | tg128 | 19.25 ± 0.06 |
build: 1d5f8dd (1)
```

## Extra comparison notes from the post

The same thread also includes ballpark comparisons against:

- Apple M3 Pro / projected M4 Pro
- AMD Radeon 780M and projected 890M

Projected comparison excerpt:

```text
Type              pp512 t/s   tg128 t/s   t/TFLOP   MBW %
140V IPEX-LLM      705.09      24.27       22.03    63.30
780M ROCm          240.79      18.61       14.51    79.55
projected 890M     344.76      24.92       14.51    79.55
```
