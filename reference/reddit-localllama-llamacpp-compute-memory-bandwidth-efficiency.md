# Reddit Snapshot: llama.cpp Compute and Memory Bandwidth Efficiency

Source URL: https://www.reddit.com/r/LocalLLaMA/comments/1ghvwsj/llamacpp_compute_and_memory_bandwidth_efficiency/

Captured: 2026-03-22

Capture note:

- Direct command-line fetches from this environment hit Reddit's block page.
- The text below is a local markdown snapshot assembled from browser-accessible page content so we still have a useful tracked reference.

## Original post

Author: `randomfoo2`

> One of the things that I noticed from my recent Intel Xe2 iGPU testing with llama.cpp was that theoretical max FP16 TFLOPS and MBW only told a part of the story.
>
> I thought I'd share these numbers since it's pretty interesting to see how TFLOPS and MBW are actually only one part of the equation, and there's a huge variance in t/TFLOP efficiency and MBW efficiency between backends and devices.

Posted comparison table:

```text
Build Hardware Backend FP16 TFLOPS MBW GB/s pp512 t/s tg128 t/s t/TFLOP MBW %
b4008 EPYC 9274F CPU 3.2 460.8 184.61 39.41 58.61 30.45
b4008 Arc 140V IPEX-LLM 32.0 136.5 656.5 22.98 20.52 59.93
b4008 Radeon 780M ROCm 16.6 89.6 240.79 18.61 14.51 73.94
b4008 W7900 ROCm 122.6 864 2872.74 95.56 23.43 39.37
b4008 7900 XTX ROCm 122.8 960 3206.94 102.92 26.12 38.17
b4008 RTX 3050 6GB CUDA (FA)13.6 168 1250.59 37.77 92.29 80.04
b4011 RTX 3090 CUDA (FA)71.0 936.2 6073.39 167.28 85.54 63.61
b4011 RTX 4090 CUDA (FA)165.2 1008 13944.43 187.7 84.41 66.29
b4011 M2 (10CU)Metal 7.1 100 185.34 21.67 26.10 77.15
???M2 (10CU) ^Metal 7.1 100 179.57 21.91 25.29 78.00
???M3 Pro (18CU) ^Metal 12.8 150 341.67 30.74 26.73 72.96
???M3 Max (40CU) ^Metal 28.4 400 759.7 66.31 26.75 59.02
```

Other notes from the post:

> The rest of the numbers are from tests I ran with very recent builds of `llama.cpp` (b4008-4011) on various Linux systems.
>
> All tests were done with the Q4_0 quant of `TheBloke/Llama-2-7B-GGUF`.
>
> The pp/tg numbers are generated from `llama-bench`, typically with no additional options. CUDA runs are with `-fa 1`.
>
> `t/TFLOPS` is just `pp512 / TFLOPS`.
>
> `MBW %` is `100 * tg128 / (MBW / 3.56)`.

## Selected comments

### Arc A770 data

Commenter: `easyfab`

> Some data with an ARC 770.

```text
Vulkan backend :
Vulkan0: Intel(R) Arc(TM) A770 Graphics (Intel Corporation) | uma: 0 | fp16: 1 | warp size: 32
| llama 7B Q4_0 | 3.56 GiB | 6.74 B | Vulkan,RPC | 99 | pp512 | 158.49 ± 0.60 |
| llama 7B Q4_0 | 3.56 GiB | 6.74 B | Vulkan,RPC | 99 | tg128 | 34.67 ± 0.07 |
build: ab3d71f9 (3999)

SYCL backend :
| 0 | [level_zero:gpu:0] | Intel Arc A770 Graphics | 1.5 | 512 | 1024 | 32 | 16704M | 1.3.31093 |
| llama 7B Q4_0 | 3.56 GiB | 6.74 B | SYCL | 99 | pp512 | 917.08 ± 9.85 |
| llama 7B Q4_0 | 3.56 GiB | 6.74 B | SYCL | 99 | tg128 | 42.10 ± 0.18 |
build: ab3d71f9 (3999)

IPEX LLM Backend :
| 0 | [level_zero:gpu:0] | Intel Arc A770 Graphics | 1.5 | 512 | 1024 | 32 | 16704M | 1.3.31093 |
| llama 7B Q4_0 | 3.56 GiB | 6.74 B | SYCL | 99 | pp512 | 2206.05 ± 7.15 |
| llama 7B Q4_0 | 3.56 GiB | 6.74 B | SYCL | 99 | tg128 | 72.66 ± 0.15 |
build: 1d5f8dd (1)
```

### CPU / model-size bandwidth observation

Commenter: `fairydreaming`

> Dear OP, I crushed some numbers for ya (llama.cpp b4011, 1 x Epyc 9374F):

```text
./llama-bench --numa distribute -t 32 -m models/llama-2-7b.Q4_0.gguf -r 20
| llama 7B Q4_0 | 3.56 GiB | 6.74 B | CPU | 32 | pp512 | 223.88 ± 0.21 |
| llama 7B Q4_0 | 3.56 GiB | 6.74 B | CPU | 32 | tg128 | 54.68 ± 0.05 |
```

> Very small models tend to have poor MBW utilization on Epyc.

Follow-up:

> For 1B model 210.57 GB/s, for 3B model 251.98 GB/s, for 8B model 283.89 GB/s, for 70B model 328.85 GB/s. (all Q8_0)

### IPEX vs SYCL

Commenter: `FullstackSensei`

> I only became aware of IPEX recently, and I'm amazed at how much more efficient it is compared to Vulkan, and even SYCL.

Reply from `fallingdowndizzyvr`:

> IPEX is SYCL. The difference is how it's being used. The SYCL backend for llama.cpp was semi-auto generated. The Intel one is better tuned and thus more performant.
