# 04-llama.cpp

This directory is for backend-specific `llama.cpp` testing after the OpenVINO-family runtime phase is working.

The point of this stage is to answer:

- how `Vulkan`, `SYCL`, and `OpenVINO` compare on the same Intel machine
- which quantizations and model families actually run on each backend
- how context depth, `-ngl`, `-fa`, batch, and ubatch change performance
- where `llama.cpp` works as the practical Intel fallback when other stacks are narrower

All three local backend checkouts here currently point at the same upstream `llama.cpp` commit:

- `3306dbaef`

Those directories are intentionally separate local clones so each backend can keep its own build tree without fighting over CMake state.

## Current script set

- `common.sh`
  - shared helpers for backend build scripts
- `build-vulkan.sh`
  - configures, builds, and sanity-checks the Vulkan backend
- `build-sycl.sh`
  - configures and attempts to build the SYCL backend after sourcing oneAPI
- `build-openvino.sh`
  - configures, builds, and sanity-checks the OpenVINO backend from the `intel-inf-openvino` env
  - also applies the current Arch-specific OpenVINO pip-wheel TBB shim
- `run-suite.sh`
  - runs all three backend build scripts and prints a simple pass/fail summary

## Benchmark and build environments

This phase will not be one single env. The build and run context depends on backend:

- Vulkan backend
  - no `mamba` env by default
  - uses the system Vulkan stack and a dedicated `llama.cpp-*` build directory
- SYCL backend
  - no `mamba` env by default
  - requires `source ./00-setup/oneapi-env.sh`
  - uses a dedicated `llama.cpp-*` build directory
- OpenVINO backend
  - no `mamba` env by default for the compiled `llama.cpp` binary itself
  - may additionally use `intel-inf-openvino` for helper scripts or OpenVINO-side cross-checks
  - uses a dedicated `llama.cpp-*` build directory

That build split is intentional: the ignored `llama.cpp-*` directories are separate backend build trees, while [llama.cpp](/home/lhl/github/lhl/intel-inference/llama.cpp/) is the single pinned source checkout.

## Current validated backend status

Validated on March 24, 2026 on the tracked Lunar Lake machine:

- system profile:
  - [lunarlake-ultra7-258v-32gb.md](/home/lhl/github/lhl/intel-inference/00-setup/systems/lunarlake-ultra7-258v-32gb.md)

Current backend read:

- `Vulkan`
  - configure: pass
  - build: pass
  - sanity: pass
  - `llama-bench --list-devices` sees `Vulkan0: Intel(R) Graphics (LNL)`
- `SYCL`
  - configure: pass
  - build: fail
  - failure: Arch's current oneAPI package set does not provide `oneapi/mkl.hpp`, which `ggml-sycl` includes through `dpct/helper.hpp`
- `OpenVINO`
  - configure: pass
  - build: pass
  - sanity: pass
  - caveat: on this Arch machine, the pip-installed OpenVINO runtime is not enough by itself for `llama.cpp`'s current CMake expectations; an env-local TBB shim is required

What is validated here right now:

- Vulkan backend build and device enumeration
- OpenVINO backend build and backend enumeration on `CPU`, `GPU`, and `NPU`
- the exact SYCL failure mode on the current Arch oneAPI toolchain

What is not validated yet:

- real GGUF model runs on `OpenVINO` or `SYCL`
- broader quant-by-quant compatibility beyond the current `BF16` and `Q4_K_XL` spot checks
- OpenAI-compatible server benchmarking through `llama-server`

## Initial /models/gguf bench pass

First runtime comparison pass from March 24, 2026:

- models:
  - `/models/gguf/Llama-3.2-1B-Instruct-BF16.gguf`
  - `/models/gguf/Llama-3.2-1B-Instruct-UD-Q4_K_XL.gguf`
  - `/models/gguf/LFM2.5-1.2B-Instruct-BF16.gguf`
  - `/models/gguf/LFM2.5-1.2B-Instruct-UD-Q4_K_XL.gguf`
- benchmark shape:
  - `llama-bench`
  - default `r=5`
  - default `pp512/tg128`
  - `-fa 1`

### Vulkan results

Backend and command shape:

```bash
04-llama.cpp/llama.cpp-vulkan/build-intel/bin/llama-bench \
  -m /models/gguf/<model>.gguf \
  -fa 1 \
  -dev Vulkan0 \
  -o jsonl
```

Summary:

| Model | Prompt tok/s | Gen tok/s |
| --- | ---: | ---: |
| `Llama-3.2-1B-Instruct-BF16` | `924.05` | `31.93` |
| `Llama-3.2-1B-Instruct-Q4_K_XL` | `1705.28` | `42.04` |
| `LFM2.5-1.2B-Instruct-BF16` | `930.25` | `34.06` |
| `LFM2.5-1.2B-Instruct-Q4_K_XL` | `1790.30` | `50.37` |

Immediate read:

- Vulkan is the only backend that produced a full comparable result set on this machine
- both Q4 models are clearly faster than BF16 on prompt processing and generation
- `LFM2.5` is close to `Llama-3.2` in BF16 prompt throughput and a bit faster in generation
- `LFM2.5 Q4_K_XL` was the fastest generation case in this first pass

### Plugged-in / TTY-only Vulkan rerun

Rerun on March 24, 2026 under the intended higher-stability condition:

- machine on AC power
- plain TTY session instead of the normal desktop session
- same command shape: `llama-bench`, default `r=5`, default `pp512/tg128`, `-fa 1`

Updated Vulkan results:

| Model | Prompt tok/s | Gen tok/s |
| --- | ---: | ---: |
| `Llama-3.2-1B-Instruct-BF16` | `1112.37` | `36.32` |
| `Llama-3.2-1B-Instruct-Q4_K_XL` | `2062.64` | `48.47` |
| `LFM2.5-1.2B-Instruct-BF16` | `1147.68` | `38.34` |
| `LFM2.5-1.2B-Instruct-Q4_K_XL` | `2221.51` | `59.86` |

Current read after the rerun:

- the AC-power / TTY-only condition is materially better for the Vulkan backend on this Lunar Lake machine
- the new Vulkan numbers are the better reference set for "best current local result" on this repo
- the overall ordering did not change:
  - Q4 still beats BF16 on both models
  - `LFM2.5 Q4_K_XL` is still the fastest generation case

Raw files:

- `04-llama.cpp/results/bench-vulkan-fa1-r5-20260323T190254Z.jsonl`
- `04-llama.cpp/results/bench-vulkan-fa1-r5-20260323T190254Z.stderr.log`

### OpenVINO results

OpenVINO did not produce a comparable model-level result set.

What failed:

- `llama-bench` with `GGML_OPENVINO_DEVICE=GPU` segfaulted immediately on:
  - `Llama-3.2-1B-Instruct-BF16`
  - `Llama-3.2-1B-Instruct-Q4_K_XL`
- removing `-dev OPENVINO0` did not help
- `llama-cli` on `Llama-3.2-1B-Instruct-Q4_K_XL` with `GGML_OPENVINO_DEVICE=GPU` also segfaulted:
  - with `-fa on`
  - with `-fa off`
- `llama-bench` on the default OpenVINO `CPU` device for `Llama-3.2-1B-Instruct-Q4_K_XL` did not segfault, but still failed benchmark warmup:
  - `test_prompt: failed to decode prompt batch, res = -3`

So the current finding is not just "OpenVINO GPU is slower." The current OpenVINO GGUF runtime path is not usable enough on this machine to produce a valid comparison set yet.

Raw files:

- `04-llama.cpp/results/bench-openvino-gpu-fa1-r5-20260323T190422Z.jsonl`
- `04-llama.cpp/results/bench-openvino-gpu-fa1-r5-20260323T190422Z.stderr.log`

### SYCL status for this bench pass

There is no SYCL result set because the SYCL backend still does not build on this Arch machine:

- compile fails on `fatal error: 'oneapi/mkl.hpp' file not found`

So for this first small-model pass, the practical comparison is:

- Vulkan: working
- OpenVINO: not benchmarkable yet on the requested path
- SYCL: build-blocked

## Backend-specific results

### Vulkan

Validated build path on this machine:

```bash
./04-llama.cpp/build-vulkan.sh
```

The script configures:

```bash
cmake -S 04-llama.cpp/llama.cpp-vulkan \
  -B 04-llama.cpp/llama.cpp-vulkan/build-intel \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_VULKAN=ON
cmake --build 04-llama.cpp/llama.cpp-vulkan/build-intel --parallel
```

Validated sanity result:

```text
Available devices:
  Vulkan0: Intel(R) Graphics (LNL)
```

So Vulkan is currently the cleanest `llama.cpp` GPU path on this Arch Lunar Lake machine.

### SYCL

Validated build path on this machine:

```bash
./04-llama.cpp/build-sycl.sh
```

The script configures with oneAPI:

```bash
source ./00-setup/oneapi-env.sh
cmake -S 04-llama.cpp/llama.cpp-sycl \
  -B 04-llama.cpp/llama.cpp-sycl/build-intel \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_SYCL=ON \
  -DCMAKE_C_COMPILER=icx \
  -DCMAKE_CXX_COMPILER=icpx
cmake --build 04-llama.cpp/llama.cpp-sycl/build-intel --parallel
```

Current failure:

```text
fatal error: 'oneapi/mkl.hpp' file not found
```

Important interpretation:

- this is not a generic "SYCL unsupported on Intel" result
- configure succeeds and `icx`/`icpx` are found
- the concrete blocker is the current Arch oneAPI packaging layout versus what upstream `ggml-sycl` expects

So on this machine today, `llama.cpp` SYCL is not yet a usable baseline backend.

### OpenVINO

Validated build path on this machine:

```bash
./04-llama.cpp/build-openvino.sh
```

The script builds from the `intel-inf-openvino` env, then configures:

```bash
cmake -S 04-llama.cpp/llama.cpp-openvino \
  -B 04-llama.cpp/llama.cpp-openvino/build-intel \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_OPENVINO=ON
cmake --build 04-llama.cpp/llama.cpp-openvino/build-intel --parallel
```

Current Arch-specific caveat:

- the `openvino` pip wheel ships `OpenVINOConfig.cmake`, but `llama.cpp`'s OpenVINO backend also expects a `3rdparty/tbb/.../TBBConfig.cmake` layout that the wheel does not provide directly
- the build now works by creating an env-local shim inside `intel-inf-openvino` that points the expected TBB paths at Arch's system `TBB` CMake files, headers, and libraries

Validated sanity results:

- default backend load:
  - `llama-cli --help` prints `OpenVINO: using device CPU`
- with `GGML_OPENVINO_DEVICE=GPU`:
  - `llama-bench --list-devices` prints `OpenVINO: using device GPU`
  - on this machine it also prints `Failed to get OpenCL device: -1`, but still enumerates the backend and does not abort
- with `GGML_OPENVINO_DEVICE=NPU` plus [npu-env.sh](/home/lhl/github/lhl/intel-inference/00-setup/npu-env.sh):
  - `llama-bench --list-devices` prints `OpenVINO: using device NPU`

So the OpenVINO backend is buildable and backend-visible here, but it still needs real GGUF model validation before we draw any performance or support conclusions.

## Next work in this phase

The initial `04-llama.cpp` pass should include:

- GGUF smoke runs once local model files are in place
- `llama-bench` sweeps
- `llama-perplexity` quant regression checks
- `llama-cli` and `llama-server` smoke and latency tests
- OpenAI-compatible server benchmarks via [benchmarks/openai_api_bench.py](/home/lhl/github/lhl/intel-inference/benchmarks/openai_api_bench.py) so `llama.cpp` can be compared directly against the `03-openvino` and `05-vllm` serving layers
- explicit backend limitation notes for unsupported model types or quantizations

## Related docs

- [README.md](/home/lhl/github/lhl/intel-inference/README.md)
- [IMPLEMENTATION.md](/home/lhl/github/lhl/intel-inference/IMPLEMENTATION.md)
- [TESTING.md](/home/lhl/github/lhl/intel-inference/TESTING.md)
- [03-openvino/README.md](/home/lhl/github/lhl/intel-inference/03-openvino/README.md)
