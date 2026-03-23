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
- PyTorch, OpenVINO, OpenVINO GenAI, Optimum Intel, vLLM, SGLang, and `llama.cpp`

## Current recommendation in one page

If you want the least confusing setup today:

1. Use Ubuntu 24.04 with the HWE kernel for most Arc/Xe work, or Ubuntu 25.10 if you are targeting the newest client platforms and want the cleanest upstream story.
2. Install Intel GPU and NPU drivers at the system level, not inside Conda.
3. Use `mamba` or `conda` only for Python environment isolation.
4. Use separate envs per stack.
5. Use official PyTorch XPU wheels for native PyTorch.
6. Use OpenVINO for the smallest maintained Intel runtime baseline.
7. Use a separate Optimum Intel env for Hugging Face export and Optimum-specific runtime testing.
8. Use OpenVINO GenAI when you want a lighter-weight pipeline/runtime API layer for local LLM, VLM, Whisper, TTS, embedding, or rerank work.
9. Use `llama.cpp` with separate build directories for `SYCL`, `Vulkan`, and `OpenVINO`.
10. Do not treat archived `IPEX-LLM` instructions as the default baseline, especially for old oneAPI pinning.

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
- Keep OpenVINO and OpenVINO GenAI in the same release family if you mix them in one env.
- Keep the default OpenVINO envs minimal. Add `optimum-intel[openvino]` only when you need Optimum export or Hugging Face integration.

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

### Arch-specific note from live validation

On the current Arch Lunar Lake machine in this repo:

- the kernel driver and `/dev/accel/accel0` were present
- the OpenVINO NPU plugin was present
- OpenVINO still failed to enumerate NPU until `/usr/lib/x86_64-linux-gnu` was added to `LD_LIBRARY_PATH`

That indicates a userspace loader-path issue on this distro/package layout, not a missing kernel driver.

For this repo's Arch setup scripts, the practical fix is:

```bash
source ./00-setup/npu-env.sh
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

## 5. OpenVINO and Optimum Intel

This is the strongest maintained Intel inference path in the current source set, especially if you care about GPU and NPU together.

### Recommended env layout

Keep the runtime and Optimum layers separate by default:

```bash
mamba create -n openvino python=3.11 pip -y
mamba create -n optimum-openvino python=3.11 pip -y
```

Use:

```bash
mamba activate openvino
python -m pip install --upgrade pip
python -m pip install -U openvino
```

And only create the heavier Optimum env when needed:

```bash
mamba activate optimum-openvino
python -m pip install --upgrade pip
python -m pip install -U openvino "optimum-intel[openvino]"
```

### Official Python install

Minimal runtime env:

```bash
python -m pip install -U openvino
```

Add Optimum only when needed:

```bash
python -m pip install -U "optimum-intel[openvino]"
```

Note:

- Optimum Intel's README still documents the `optimum-intel[openvino]` extra.
- The same README also says extras are deprecated and will be removed in a future release.
- When that changes, the practical replacement will likely be a separate `openvino` install plus plain `optimum-intel`.
- In live repo setup on March 23, 2026, adding `optimum-intel[openvino]` also pulled a large generic `torch` stack from PyPI. That is another reason not to make it the default baseline env.

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
- OpenVINO GenAI adds a maintained `WhisperPipeline` and SpeechT5 pipeline layer on top of OpenVINO Runtime.

### Speech-specific maintained paths

If your immediate goal is ASR or TTS rather than decoder-only LLMs:

- start with `openvino.genai` if you want the clearest maintained Intel speech pipeline story for:
  - Whisper
  - SpeechT5
  - CPU, GPU, and potentially NPU-backed OpenVINO execution
- keep `whisper.cpp` as a second practical path for lightweight local Whisper testing, especially if you want:
  - a small self-contained C/C++ stack
  - local quantized Whisper models
  - optional OpenVINO encoder acceleration on Intel CPU or GPU

For this repo, that means ASR and TTS testing should not be framed only as "generic Hugging Face on Intel". We now have concrete speech-oriented reference implementations to compare.

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

## 6. OpenVINO GenAI

This is not just "OpenVINO but nicer." It is a separate maintained runtime and pipeline layer on top of OpenVINO Runtime.

Why it matters:

- it has first-class pipeline APIs rather than only Hugging Face wrapper flows
- it covers more than text LLMs:
  - LLM
  - VLM
  - Whisper ASR
  - SpeechT5 TTS
  - text embeddings
  - text rerank
- it exposes serving-oriented features such as:
  - continuous batching
  - prefix caching
  - speculative decoding
  - sparse-attention prefill
- the local repo and tests show stronger evidence than the top-level OpenVINO README alone, including NPU-oriented Whisper and VLM test coverage

### Recommended env layout

If you already have an OpenVINO env, the simplest path is to keep `openvino-genai` there.

```bash
mamba create -n openvino-genai python=3.11 pip -y
mamba activate openvino-genai
python -m pip install --upgrade pip
```

### Install

Quickest maintained binary path:

```bash
python -m pip install -U openvino openvino-genai
```

If you want model export in the same env:

```bash
python -m pip install -U openvino
python -m pip install -U "optimum-intel[openvino]"
```

### Verify

```bash
python - <<'PY'
import openvino_genai as ov_genai
print("LLMPipeline:", ov_genai.LLMPipeline)
print("WhisperPipeline:", ov_genai.WhisperPipeline)
print("Text2SpeechPipeline:", ov_genai.Text2SpeechPipeline)
PY
```

### Minimal LLM flow

Export a model:

```bash
optimum-cli export openvino --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 TinyLlama_1_1b_v1_ov
```

Run it:

```bash
python - <<'PY'
import openvino_genai as ov_genai
pipe = ov_genai.LLMPipeline("TinyLlama_1_1b_v1_ov", "GPU")
print(pipe.generate("What is OpenVINO?", max_new_tokens=64))
PY
```

### Minimal Whisper flow

Export a Whisper model:

```bash
optimum-cli export openvino --trust-remote-code --model openai/whisper-base whisper-base
python -m pip install librosa
```

Run it:

```bash
python - <<'PY'
import librosa
import openvino_genai as ov_genai

audio, _ = librosa.load("how_are_you_doing_today.wav", sr=16000)
pipe = ov_genai.WhisperPipeline("whisper-base", "GPU")
result = pipe.generate(audio.tolist(), return_timestamps=True)
print(result.texts[0])
PY
```

### Minimal SpeechT5 flow

OpenVINO GenAI's current text-to-speech path is explicitly SpeechT5-based.

```bash
optimum-cli export openvino --model microsoft/speecht5_tts --model-kwargs "{\"vocoder\": \"microsoft/speecht5_hifigan\"}" speecht5_tts
```

Then use `openvino_genai.Text2SpeechPipeline`.

### Built-in benchmark and validation tooling

OpenVINO GenAI is also worth treating as a testing surface, not only an inference library.

The repo already includes:

- `tools/llm_bench` for performance-focused benchmarking across OpenVINO GenAI, Optimum Intel, and selected PyTorch/OpenVINO flows
- `tools/who_what_benchmark` for similarity-oriented regression checks between baseline Hugging Face outputs and optimized OpenVINO or OpenVINO GenAI outputs

Use these after basic bring-up succeeds; they are a better fit than inventing a custom benchmark harness too early.

Practical rule:

- use `llm_bench` for performance work
- use `who_what_benchmark` for quality/regression work
- do not use `who_what_benchmark` to argue that one backend is faster than another

### Build-from-source note

If you build OpenVINO GenAI from source, the repo's own build docs warn about ABI and Python binding issues when OpenVINO GenAI and OpenVINO are not built in a compatible way.

Practical repo rule:

- prefer the binary distribution channels unless you specifically need source builds
- if you do build from source, keep OpenVINO and OpenVINO GenAI aligned in the same build environment

## 7. vLLM on Intel

There are two Intel-relevant stories now:

- upstream `vLLM` on Intel XPU
- `vllm-openvino`

They are not the same thing.

### 7.1 Upstream vLLM XPU

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

### 7.2 vLLM OpenVINO

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

## 8. SGLang on Intel

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

## 9. llama.cpp

In this repo, the right way to think about `llama.cpp` on Intel is:

- `SYCL` for the most Intel-native GPU backend
- `Vulkan` for the lightest generic GPU backend
- `OpenVINO` for Intel CPU, GPU, and NPU, including a real NPU path

This repo already ignores separate build directories like `llama.cpp-sycl`, `llama.cpp-vulkan`, and `llama.cpp-openvino`, so use those.

### 9.1 SYCL build

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

### 9.2 Vulkan build

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

### 9.3 OpenVINO build

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

## 10. Backend and model-type limits that matter right now

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

## 11. What I would actually use first

If I were setting up this repo on a fresh Linux machine today, I would do it in this order:

1. System GPU driver install and `clinfo` verification.
2. `openvino` env for the best maintained Intel path and NPU-adjacent work.
3. `openvino-genai` env or package layer for LLM, VLM, Whisper, TTS, embedding, and rerank bring-up.
4. `torch-xpu` env for native PyTorch smoke tests.
5. `llama.cpp-vulkan` as the lightest GGUF baseline.
6. `llama.cpp-sycl` once the oneAPI toolchain is in place.
7. `llama.cpp-openvino` if NPU or OpenVINO-backed GGUF is the goal.
8. `vLLM` and `SGLang` only after the basic GPU stack is known-good.

That order reduces ambiguity. If `vLLM` or `SGLang` breaks first, you then know whether you have a real driver/runtime problem or just a serving-engine-specific limitation.

## 12. Sources

Official web sources:

- Intel client GPU driver docs: https://dgpu-docs.intel.com/driver/client/overview.html
- PyTorch XPU getting started: https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html
- Intel oneAPI Base Toolkit page: https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html
- Intel oneAPI Linux APT install guide: https://www.intel.com/content/www/us/en/docs/oneapi/installation-guide-linux/2025-0/apt-005.html
- OpenVINO install docs: https://docs.openvino.ai/2026/get-started/install-openvino.html
- OpenVINO Conda-Forge install docs: https://docs.openvino.ai/2026/get-started/install-openvino/install-openvino-conda.html
- OpenVINO NPU device docs: https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html
- OpenVINO GenAI install docs: https://openvinotoolkit.github.io/openvino.genai/docs/getting-started/installation/
- OpenVINO GenAI on NPU: https://docs.openvino.ai/2026/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html
- OpenVINO GenAI docs: https://openvinotoolkit.github.io/openvino.genai/
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
- [reference/openvino.genai/README.md](/home/lhl/github/lhl/intel-inference/reference/openvino.genai/README.md)
- [reference/openvino.genai/samples/python/whisper_speech_recognition/README.md](/home/lhl/github/lhl/intel-inference/reference/openvino.genai/samples/python/whisper_speech_recognition/README.md)
- [reference/openvino.genai/samples/python/speech_generation/README.md](/home/lhl/github/lhl/intel-inference/reference/openvino.genai/samples/python/speech_generation/README.md)
- [reference/openvino.genai/src/docs/BUILD.md](/home/lhl/github/lhl/intel-inference/reference/openvino.genai/src/docs/BUILD.md)
- [reference/openvino.genai/tools/llm_bench/README.md](/home/lhl/github/lhl/intel-inference/reference/openvino.genai/tools/llm_bench/README.md)
- [reference/openvino.genai/tools/who_what_benchmark/README.md](/home/lhl/github/lhl/intel-inference/reference/openvino.genai/tools/who_what_benchmark/README.md)
- [reference/optimum-intel/README.md](/home/lhl/github/lhl/intel-inference/reference/optimum-intel/README.md)
- [reference/ipex-llm/README.md](/home/lhl/github/lhl/intel-inference/reference/ipex-llm/README.md)
