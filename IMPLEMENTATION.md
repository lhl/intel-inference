# Linux Implementation Guide

This document is the current best attempt at a real install and setup guide for Intel AI/ML inference on Linux. It is intentionally practical, but parts of it are still docs-derived rather than fully validated in this repo.

## Status

- Scope: Linux only
- Bias: inference first
- Validation level: source-backed and current-link-checked, but not yet fully reproduced end-to-end in this repo

## Companion docs

- [`README.md`](README.md): repo scope and document map
- [`ANALYSIS.md`](ANALYSIS.md): broader support analysis and evidence framing
- [`TODO.md`](TODO.md): remaining research and validation backlog

## Scope

- Linux only
- Inference first
- Intel Arc dGPU and Xe-family iGPU first
- Intel NPU included, but still secondary
- PyTorch, OpenVINO, Optimum Intel, vLLM, SGLang, and `llama.cpp`

## Current recommendation in one page

If you want the least confusing setup today:

1. Use Ubuntu 24.04 with the HWE kernel for most Arc/Xe work, or Ubuntu 25.10 if you are targeting the newest client platforms and want the cleanest upstream story.
2. Install Intel GPU and NPU drivers at the system level, not inside Conda.
3. Use `mamba` or `conda` only for Python environment isolation.
4. Use separate envs per stack.
5. Use official PyTorch XPU wheels for native PyTorch.
6. Use OpenVINO plus Optimum Intel for the most maintained Intel inference path across GPU and NPU.
7. Use `llama.cpp` with separate build directories for `SYCL`, `Vulkan`, and `OpenVINO`.
8. Do not treat archived `IPEX-LLM` instructions as the default baseline, especially for old oneAPI pinning.

## Environment strategy

Repo recommendation:

- Use `mamba` or `conda` to create one env per stack.
- Install low-level drivers and runtimes globally:
  - Intel GPU driver packages
  - Level Zero
  - OpenCL
  - Vulkan
  - oneAPI, when required
  - NPU driver, when required
- Install Python packages inside the env with `pip`, especially for PyTorch XPU wheels and source builds like `vLLM` and `SGLang`.

This is the main difference from the older Intel GPU experience. For maintained binary installs, manual oneAPI installation is no longer the default. You still need oneAPI for `llama.cpp` SYCL and similar SYCL-native build flows. The old "pin a specific oneAPI release because newer ones break and older ones disappear" pattern shows up most clearly in archived `IPEX-LLM` instructions, not in the maintained PyTorch XPU or OpenVINO paths we should prefer today.

## Linux baseline

The official docs are Ubuntu-heavy, so this guide is Ubuntu-first.

Current docs point in this direction:

- Intel client GPU docs recommend Ubuntu 25.10 for Lunar Lake, Battlemage, and Panther Lake for full out-of-box support.
- Intel client GPU docs also say Ubuntu 24.04 works for hardware that needs kernel 6.8 or newer, but 24.04 must be on the HWE kernel for Lunar Lake, Battlemage, and Panther Lake.
- PyTorch XPU docs validate Intel client GPU support on Ubuntu 24.04 or 25.04 for Arc A-series, Arc B-series, Meteor Lake-H, Arrow Lake-H, and Lunar Lake, and on Ubuntu 25.10 for Panther Lake.

Practical repo recommendation:

- Arc A-series, Arc B-series, Meteor Lake, Arrow Lake, Lunar Lake:
  - start with Ubuntu 24.04 LTS on HWE
- Panther Lake and newer client platforms:
  - prefer Ubuntu 25.10 or newer if possible

## 1. Base system setup

Install the generic tools we need for the rest of the stacks:

```bash
sudo apt-get update
sudo apt-get install -y \
  software-properties-common \
  build-essential \
  git \
  cmake \
  ninja-build \
  pkg-config \
  python3-pip \
  curl \
  wget \
  tar
```

For most Intel compute and inference flows, your user should have access to the `render` group:

```bash
sudo gpasswd -a "${USER}" render
newgrp render
```

If you just changed group membership, a full logout and login is often cleaner than trying to continue in the same shell.

## 2. Intel GPU driver setup

For Ubuntu 24.04 and 25.10 client GPU installs, Intel currently documents the `intel-graphics` PPA path.

```bash
sudo apt-get update
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:kobuk-team/intel-graphics

sudo apt-get install -y \
  libze-intel-gpu1 \
  libze1 \
  intel-metrics-discovery \
  intel-opencl-icd \
  clinfo \
  intel-gsc

sudo apt-get install -y \
  intel-media-va-driver-non-free \
  libmfx-gen1 \
  libvpl2 \
  libvpl-tools \
  libva-glx2 \
  va-driver-all \
  vainfo

# Intel explicitly calls these out as additional packages for PyTorch.
sudo apt-get install -y libze-dev intel-ocloc
```

Verify the GPU stack:

```bash
clinfo | grep "Device Name"
```

If `clinfo` does not show the Intel GPU, check:

- that you are on the right kernel for your hardware
- that your user can access `/dev/dri/renderD*`
- that you really are on the HWE kernel if you are using Ubuntu 24.04 with newer client GPUs

## 3. Intel NPU driver setup

Use this only if you have an Intel Core Ultra platform with an NPU and you actually want NPU inference. The maintained user-facing path today is OpenVINO.

Two important realities from the current docs:

- OpenVINO's NPU device docs still describe Ubuntu 22.04 with kernel 6.6+ as the documented platform.
- The current `intel/linux-npu-driver` release artifacts also ship Ubuntu 24.04 packages, so the real install story is partly driven by the latest release notes rather than a single stable distro page.

Practical path for Ubuntu 24.04:

1. Open the latest Linux NPU driver release notes:
   - https://github.com/intel/linux-npu-driver/releases
2. Download the `ubuntu2404` tarball for the current release.
3. Install the release packages.
4. Ensure `level-zero` is installed if the release notes say it is missing.
5. Ensure your user can access `/dev/accel/accel0`.

The current release notes follow this pattern:

```bash
sudo dpkg --purge --force-remove-reinstreq \
  intel-driver-compiler-npu \
  intel-fw-npu \
  intel-level-zero-npu \
  intel-level-zero-npu-dbgsym

sudo apt update
sudo apt install -y libtbb12

# Replace <release-tarball> with the current ubuntu2404 archive from:
# https://github.com/intel/linux-npu-driver/releases
tar -xf <release-tarball>
sudo dpkg -i *.deb

# If Level Zero is missing, install the matching package called out by the release notes.
dpkg -l level-zero
```

Verify the device:

```bash
ls /dev/accel/accel0
sudo dmesg | tail -n 50
```

If access is wrong, the current release notes show this pattern:

```bash
sudo chown root:render /dev/accel/accel0
sudo chmod g+rw /dev/accel/accel0
sudo usermod -a -G render "${USER}"
```

To persist that permission model across reloads and reboots, the same release notes also show this udev rule:

```bash
sudo bash -c "echo 'SUBSYSTEM==\"accel\", KERNEL==\"accel*\", GROUP=\"render\", MODE=\"0660\"' > /etc/udev/rules.d/10-intel-vpu.rules"
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=accel
```

## 4. PyTorch XPU

This is the cleanest native PyTorch path on Intel GPU right now.

Important current behavior:

- if you install PyTorch from official XPU wheels, you do not need to separately install Intel Deep Learning Essentials
- if you build PyTorch from source, then Intel points you back to the driver plus Deep Learning Essentials path

### Recommended env layout

```bash
mamba create -n torch-xpu python=3.11 pip -y
mamba activate torch-xpu
python -m pip install --upgrade pip
```

### Install official XPU wheels

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
```

Nightly:

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu
```

### Verify

```bash
python - <<'PY'
import torch
print("xpu available:", torch.xpu.is_available())
if torch.xpu.is_available():
    print("device count:", torch.xpu.device_count())
    print("device 0:", torch.xpu.get_device_name(0))
PY
```

### Minimal inference smoke test

```bash
python - <<'PY'
import torch
import torchvision.models as models

model = models.resnet50(weights="ResNet50_Weights.DEFAULT").eval().to("xpu")
data = torch.rand(1, 3, 224, 224, device="xpu")
with torch.no_grad():
    model(data)
torch.xpu.synchronize()
print("PyTorch XPU inference ok")
PY
```

### Current limits worth remembering

- PyTorch still describes Intel client GPU support as "prototype ready" from PyTorch 2.5 onward.
- PyTorch documents FP32, BF16, FP16, AMP, eager mode, and `torch.compile` support on XPU.
- The binary install path is much easier than the historical Intel GPU path.
- This does not automatically mean every model family has first-class fused-kernel coverage comparable to CUDA.

## 5. OpenVINO plus Optimum Intel

This is the strongest maintained Intel inference path in the current source set, especially if you care about GPU and NPU together.

### Recommended env layout

```bash
mamba create -n openvino python=3.11 pip -y
mamba activate openvino
python -m pip install --upgrade pip
```

### Official Python install

```bash
python -m pip install -U openvino
python -m pip install -U "optimum-intel[openvino]"
```

Note:

- Optimum Intel's README still documents the `optimum-intel[openvino]` extra.
- The same README also says extras are deprecated and will be removed in a future release.
- When that changes, the practical replacement will likely be a separate `openvino` install plus plain `optimum-intel`.

### Optional Conda-Forge path

If you want a more Conda-native OpenVINO env, OpenVINO documents a Conda-Forge path:

```bash
mamba create -n openvino-cf -c conda-forge python=3.10 openvino ocl-icd-system -y
```

But OpenVINO's own docs warn that the Conda-Forge channel is community-maintained and not their preferred production path. For this repo, I would use `pip install openvino` unless there is a specific Conda reason not to.

### Verify

```bash
python - <<'PY'
import openvino as ov
core = ov.Core()
print("OpenVINO devices:", core.available_devices)
PY
```

### Minimal Optimum Intel text-generation flow

Export a Hugging Face model:

```bash
optimum-cli export openvino --model TinyLlama/TinyLlama_v1.1 ov_TinyLlama_v1_1
```

Run it:

```bash
python - <<'PY'
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer, pipeline

model_id = "ov_TinyLlama_v1_1"
model = OVModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
print(pipe("Hey, how are you doing today?", max_new_tokens=32)[0]["generated_text"])
PY
```

### Why this path matters

- OpenVINO explicitly positions itself across CPU, GPU, and NPU.
- Optimum Intel is the Hugging Face-facing bridge.
- Optimum Intel already documents Whisper quantization, so speech is not an afterthought here.

### NPU-specific reality

If you want LLM inference on NPU through OpenVINO, the current docs are much stricter than GPU:

- only static-shape models are currently supported on NPU
- Optimum Intel is the primary export path for NPU LLMs
- current OpenVINO GenAI docs say NPU LLM export should use:
  - symmetric weights
  - INT4 or NF4
  - group size `128` or channel-wise quantization
- NF4 is only supported on Intel Core Ultra Series 2 NPUs and newer

That means the NPU path is real, but much narrower than the generic GPU path.

## 6. vLLM on Intel

There are two Intel-relevant stories now:

- upstream `vLLM` on Intel XPU
- `vllm-openvino`

They are not the same thing.

### 6.1 Upstream vLLM XPU

The important change from older Intel guidance is that upstream `vLLM` now documents Intel XPU directly.

The less convenient part is that the current XPU path is still source-build oriented.

#### Env

```bash
mamba create -n vllm-xpu python=3.12 pip -y
mamba activate vllm-xpu
python -m pip install --upgrade pip
```

#### Install

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -v -r requirements/xpu.txt

# Replace any NVIDIA triton package with the Intel XPU build.
pip uninstall -y triton triton-xpu
pip install triton-xpu==3.6.0 --extra-index-url https://download.pytorch.org/whl/xpu

VLLM_TARGET_DEVICE=xpu pip install --no-build-isolation -e . -v
```

#### Verify the base stack

```bash
python - <<'PY'
import torch
print("xpu available:", torch.xpu.is_available())
PY
```

#### Practical notes

- Use `--dtype half` as the safe starting point on Arc A770.
- `vLLM`'s XPU platform code documents a BF16 accuracy issue on Arc A770 and tells users to switch to FP16 explicitly.

Example:

```bash
vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct --device xpu --dtype half
```

#### Documented feature limits

Current upstream `vLLM` docs show these Intel GPU caveats:

- the feature matrix still marks `LoRA`, `enc-dec`, and multimodal input as supported on Intel GPU, so the gap is more about kernels and quantization breadth than about basic model-class plumbing
- `CUDA graph` support is not available on Intel GPU
- Intel GPU quantization support is narrower than CUDA:
  - supported: `AWQ`, `GPTQ`
  - not supported on Intel GPU: `Marlin`, `INT8 (W8A8)`, `FP8 (W8A8)`, `bitsandbytes`, `GGUF`

This is one of the cleanest examples of Intel still lagging the CUDA-first kernel ecosystem.

### 6.2 vLLM OpenVINO

This is a separate backend and should not be confused with upstream XPU.

#### Env

```bash
mamba create -n vllm-openvino python=3.11 pip -y
mamba activate vllm-openvino
python -m pip install --upgrade pip
```

#### Install from source

```bash
git clone https://github.com/vllm-project/vllm-openvino.git
cd vllm-openvino
VLLM_TARGET_DEVICE="empty" PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu" python -m pip install -v .

# The repo README explicitly says to remove triton afterwards.
python -m pip uninstall -y triton
```

#### Device selection

```bash
export VLLM_OPENVINO_DEVICE=GPU
```

Use `CPU`, `GPU`, or an indexed GPU like `GPU.1` as needed.

#### Important limits

The `vllm-openvino` README is unusually explicit here:

- CPU and Intel GPU only
- no prebuilt wheels
- no prebuilt images
- LoRA serving is not supported
- only LLMs are supported
- LLaVA is not enabled
- encoder-decoder models are not enabled
- tensor parallelism is not enabled
- pipeline parallelism is not enabled
- prefix caching and chunked prefill both exist, but cannot be used together
- chunked prefill is documented as broken on `openvino==2025.2`; the README points users to `openvino==2025.1` or a nightly `2025.3` build if they need it
- the GPU performance section is written around quantized weights, with 8-bit and 4-bit integer weight paths called out explicitly

So this backend can still be useful, but it is not a drop-in substitute for the full CUDA-side vLLM feature story.

## 7. SGLang on Intel

Yes, SGLang now has Intel support, but it is clearly younger than its CUDA path.

Current official docs show:

- a dedicated `XPU` platform page
- source installation only
- Docker for XPU still under active development
- explicit use of the `intel_xpu` attention backend
- a short verified model list on Intel Arc B580

### Env

```bash
mamba create -n sgl-xpu python=3.12 pip -y
mamba activate sgl-xpu
python -m pip install --upgrade pip setuptools
```

### Install from source

```bash
pip install torch==2.10.0+xpu torchao torchvision torchaudio triton-xpu==3.6.0 \
  --index-url https://download.pytorch.org/whl/xpu

# SGLang's XPU docs explicitly avoid a conflicting CUDA-enabled triton path here.
pip install xgrammar --no-deps

git clone https://github.com/sgl-project/sglang.git
cd sglang/python
cp pyproject_xpu.toml pyproject.toml
pip install -v . --extra-index-url https://download.pytorch.org/whl/xpu
```

### Run

```bash
python -m sglang.launch_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --device xpu \
  --attention-backend intel_xpu \
  --page-size 64 \
  --trust-remote-code
```

The XPU docs say the `intel_xpu` backend supports page sizes `32`, `64`, and `128`.

### Current support signals and limits

The current SGLang XPU page only lists a small optimized model set:

- `Llama-3.2-3B`
- `Llama-3.1-8B`
- `Qwen2.5-1.5B`

The quantization docs also show meaningful limits:

- mixed-bit quantization is not fully supported
- quantized MoE models can fail because of kernel gaps
- quantized VLM support is limited
- `Qwen2.5-VL-7B` is specifically documented as failing in some GPTQ paths, while AWQ works

The attention-backend matrix adds more Intel-specific limits for `intel_xpu`:

- no FP8 KV cache
- no FP4 KV cache
- no speculative decoding top-k path in the current matrix
- no multimodal attention backend in the current matrix
- sliding-window support is marked available

So SGLang on Intel is no longer "no support", but it is not yet a broad first-class Intel story in the way its CUDA path is.

## 8. llama.cpp

In this repo, the right way to think about `llama.cpp` on Intel is:

- `SYCL` for the most Intel-native GPU backend
- `Vulkan` for the lightest generic GPU backend
- `OpenVINO` for Intel CPU, GPU, and NPU, including a real NPU path

This repo already ignores separate build directories like `llama.cpp-sycl`, `llama.cpp-vulkan`, and `llama.cpp-openvino`, so use those.

### 8.1 SYCL build

This is the backend that still clearly needs oneAPI.

`llama.cpp` recommends either:

- Intel oneAPI Base Toolkit
- Intel Deep Learning Essentials

and lists verified oneAPI releases `2025.2.1`, `2025.1`, and `2024.1`.

#### oneAPI install

Use Intel's official oneAPI page or the Linux package-manager docs:

- Base Toolkit page:
  - https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html
- Linux APT install guide:
  - https://www.intel.com/content/www/us/en/docs/oneapi/installation-guide-linux/2025-0/apt-005.html

The current Intel APT flow is:

```bash
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
  | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null

echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
  | sudo tee /etc/apt/sources.list.d/oneAPI.list

sudo apt update
sudo apt install -y intel-oneapi-base-toolkit
```

#### Verify oneAPI

```bash
source /opt/intel/oneapi/setvars.sh
sycl-ls
```

You should see at least one `level_zero:gpu` device.

#### Build

From the repo root:

```bash
source /opt/intel/oneapi/setvars.sh

# FP32 path
cmake -S llama.cpp -B llama.cpp-sycl \
  -DGGML_SYCL=ON \
  -DCMAKE_C_COMPILER=icx \
  -DCMAKE_CXX_COMPILER=icpx

cmake --build llama.cpp-sycl --config Release -j
```

Optional FP16 build:

```bash
source /opt/intel/oneapi/setvars.sh
cmake -S llama.cpp -B llama.cpp-sycl \
  -DGGML_SYCL=ON \
  -DGGML_SYCL_F16=ON \
  -DCMAKE_C_COMPILER=icx \
  -DCMAKE_CXX_COMPILER=icpx

cmake --build llama.cpp-sycl --config Release -j
```

#### Verify and run

```bash
source /opt/intel/oneapi/setvars.sh
./llama.cpp-sycl/bin/llama-ls-sycl-device
```

Single GPU:

```bash
source /opt/intel/oneapi/setvars.sh
export ONEAPI_DEVICE_SELECTOR="level_zero:0"
ZES_ENABLE_SYSMAN=1 ./llama.cpp-sycl/bin/llama-cli \
  -m /path/to/model.gguf \
  -ngl 99 \
  -sm none \
  -mg 0
```

Multi-GPU:

```bash
source /opt/intel/oneapi/setvars.sh
ZES_ENABLE_SYSMAN=1 ./llama.cpp-sycl/bin/llama-cli \
  -m /path/to/model.gguf \
  -ngl 99 \
  -sm layer
```

#### Important limits

- Intel GPU only
- no NPU path
- `--split-mode row` is not supported
- memory pressure is a practical limit very quickly
- older Intel GPUs are not the intended target; the docs call out Arc, Max, Flex, and 11th-gen-or-newer iGPU support

### 8.2 Vulkan build

Use this when you want the lightest setup and do not want oneAPI.

#### Install

```bash
sudo apt-get install -y libvulkan-dev glslc

# If vulkaninfo is missing on your distro, install vulkan-tools as well.
```

#### Verify

```bash
vulkaninfo
```

#### Build

```bash
cmake -S llama.cpp -B llama.cpp-vulkan -DGGML_VULKAN=1
cmake --build llama.cpp-vulkan --config Release -j
```

#### Run

```bash
./llama.cpp-vulkan/bin/llama-cli -m /path/to/model.gguf -ngl 99
```

`llama.cpp` says you should see the Vulkan backend detect the Intel GPU in the logs.

#### Important limits

- GPU only
- no NPU
- generic backend, not Intel-specific
- current docs do not publish an Intel-specific model-family validation table

This makes Vulkan a strong fallback backend, not necessarily the most feature-complete Intel backend.

### 8.3 OpenVINO build

This is the `llama.cpp` backend that matters if you want Intel NPU support.

#### Install OpenVINO runtime

For Python workflows, `pip install openvino` is the simplest install.

For the `llama.cpp` OpenVINO backend build, upstream docs still assume an archive or system-style install that provides:

```bash
/opt/intel/openvino/setupvars.sh
```

So for this specific backend, follow the official OpenVINO Linux install docs if you do not already have a system install:

- https://docs.openvino.ai/2026/get-started/install-openvino.html

#### Build

```bash
source /opt/intel/openvino/setupvars.sh
cmake -S llama.cpp -B llama.cpp-openvino -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_OPENVINO=ON
cmake --build llama.cpp-openvino --parallel
```

#### Run on GPU

```bash
export GGML_OPENVINO_DEVICE=GPU
./llama.cpp-openvino/bin/llama-cli -m /path/to/model.gguf
```

#### Run on NPU

```bash
export GGML_OPENVINO_DEVICE=NPU
./llama.cpp-openvino/bin/llama-cli -m /path/to/model.gguf
```

If you want caching on CPU or GPU:

```bash
export GGML_OPENVINO_CACHE_DIR=/tmp/ov_cache
```

#### Important limits

This backend is much more explicit about its constraints than SYCL or Vulkan:

- validated mainly on Intel Core Ultra Series 1 and Series 2 AI PCs
- supported model precisions are limited to:
  - `FP16`
  - `BF16` on Intel Xeon
  - `Q8_0`
  - `Q4_0`
  - `Q4_1`
  - `Q4_K`
  - `Q4_K_M`
  - `Q5_K` and `Q6_K` via runtime conversion
- on NPU, the primary supported quantization is `Q4_0`
- quantized-model accuracy and performance are still called a work in progress
- `GGML_OPENVINO_STATEFUL_EXECUTION=1` is experimental
- stateful execution is not effective on NPU
- stateful execution is not supported in `llama-server` and `llama-perplexity`
- the validated model list in the backend docs is text-LLM-centric; multimodal and audio paths are not explicitly validated there yet

## 9. Backend and model-type limits that matter right now

This is the short version of "what will probably disappoint you first".

| Stack | Good fit | Documented limits that matter |
|---|---|---|
| PyTorch XPU | Native PyTorch and Hugging Face inference on Intel GPU | Easier install than before, but client GPU support is still described as prototype-ready; do not assume CUDA-level fused-kernel coverage everywhere |
| OpenVINO + Optimum Intel | Best maintained Intel path across CPU, GPU, NPU | NPU LLM path is much stricter: static shapes plus INT4 or NF4-style export constraints |
| vLLM XPU | First-party upstream serving path on Intel GPU | Source build, no CUDA graph on Intel GPU, narrow Intel GPU quantization coverage, no GGUF |
| vLLM OpenVINO | LLM serving through OpenVINO | LLM-only, no LLaVA, no encoder-decoder, no LoRA serving, no tensor or pipeline parallelism |
| SGLang XPU | Emerging Intel GPU serving path | Source build only, Docker not ready, short verified model list, quantized MoE and quantized VLM limitations |
| llama.cpp SYCL | Strong Intel GPU GGUF path | No NPU, oneAPI required, no `split-mode row`, model-family validation is thin beyond standard GGUF usage |
| llama.cpp Vulkan | Simple fallback GPU path | No NPU, generic backend, Intel-specific validation is thin |
| llama.cpp OpenVINO | GGUF inference on Intel CPU/GPU/NPU | Quantization support is narrower, NPU prefers `Q4_0`, stateful mode is experimental and not universal |

Two extra `llama.cpp` notes:

- `llama.cpp` overall supports a wide range of architectures, including multimodal and Mamba-family models, but the Intel backend docs do not yet give a clean backend-by-backend model-family matrix.
- `llama.cpp` multimodal audio support is currently described as highly experimental even before you add Intel-backend-specific uncertainty.

## 10. What I would actually use first

If I were setting up this repo on a fresh Linux machine today, I would do it in this order:

1. System GPU driver install and `clinfo` verification.
2. `torch-xpu` env for native PyTorch smoke tests.
3. `openvino` env for the best maintained Intel path and NPU-adjacent work.
4. `llama.cpp-vulkan` as the lightest GGUF baseline.
5. `llama.cpp-sycl` once the oneAPI toolchain is in place.
6. `llama.cpp-openvino` if NPU or OpenVINO-backed GGUF is the goal.
7. `vLLM` and `SGLang` only after the basic GPU stack is known-good.

That order reduces ambiguity. If `vLLM` or `SGLang` breaks first, you then know whether you have a real driver/runtime problem or just a serving-engine-specific limitation.

## Sources

Official web sources:

- Intel client GPU driver docs: https://dgpu-docs.intel.com/driver/client/overview.html
- PyTorch XPU getting started: https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html
- Intel oneAPI Base Toolkit page: https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html
- Intel oneAPI Linux APT install guide: https://www.intel.com/content/www/us/en/docs/oneapi/installation-guide-linux/2025-0/apt-005.html
- OpenVINO install docs: https://docs.openvino.ai/2026/get-started/install-openvino.html
- OpenVINO Conda-Forge install docs: https://docs.openvino.ai/2026/get-started/install-openvino/install-openvino-conda.html
- OpenVINO NPU device docs: https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html
- OpenVINO GenAI on NPU: https://docs.openvino.ai/2025/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html
- Intel Linux NPU driver releases: https://github.com/intel/linux-npu-driver/releases
- vLLM GPU installation docs: https://docs.vllm.ai/en/latest/getting_started/installation/gpu/
- vLLM features matrix: https://docs.vllm.ai/en/stable/features/index.html
- vLLM quantization docs: https://docs.vllm.ai/en/latest/features/quantization/
- vLLM OpenVINO repo: https://github.com/vllm-project/vllm-openvino
- SGLang XPU docs: https://docs.sglang.io/platforms/xpu.html
- SGLang attention backend docs: https://docs.sglang.io/advanced_features/attention_backend.html
- SGLang quantization docs: https://docs.sglang.io/advanced_features/quantization.html

Pinned local sources in this repo:

- [llama.cpp/docs/backend/SYCL.md](/home/lhl/github/lhl/intel-inference/llama.cpp/docs/backend/SYCL.md)
- [llama.cpp/docs/backend/OPENVINO.md](/home/lhl/github/lhl/intel-inference/llama.cpp/docs/backend/OPENVINO.md)
- [llama.cpp/docs/build.md](/home/lhl/github/lhl/intel-inference/llama.cpp/docs/build.md)
- [reference/openvino/README.md](/home/lhl/github/lhl/intel-inference/reference/openvino/README.md)
- [reference/optimum-intel/README.md](/home/lhl/github/lhl/intel-inference/reference/optimum-intel/README.md)
- [reference/ipex-llm/README.md](/home/lhl/github/lhl/intel-inference/reference/ipex-llm/README.md)
