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

- real GGUF model runs on any `04-llama.cpp` backend
- quant-by-quant compatibility on `Vulkan`, `SYCL`, or `OpenVINO`
- OpenAI-compatible server benchmarking through `llama-server`

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
