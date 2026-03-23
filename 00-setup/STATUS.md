# 00-setup Status

Current setup status for the tracked Lunar Lake test machine.

## Validated

- GPU stack passes `./00-setup/verify-gpu-stack.sh`
- `intel-inf-openvino` is created and sees `['CPU', 'GPU']`
- `intel-inf-openvino-genai` is created and imports `LLMPipeline` and `WhisperPipeline`
- `intel-inf-optimum-openvino` is created and imports `OVModelForCausalLM`
- `intel-inf-torch-xpu` is created and reports `torch 2.10.0+xpu` with `torch.xpu.is_available() == True`

## NPU

- kernel device is present: `intel_vpu` with `/dev/accel/accel0`
- OpenVINO NPU plugin is present in the env
- direct OpenVINO enumeration fails on Arch unless `LD_LIBRARY_PATH` includes `/usr/lib/x86_64-linux-gnu`
- with that loader-path workaround, OpenVINO sees `['CPU', 'GPU', 'NPU']` and reports `Intel(R) AI Boost`

Use:

```bash
source ./00-setup/npu-env.sh
./00-setup/verify-npu-stack.sh
```

## oneAPI

- `verify-oneapi.sh` currently fails because `/opt/intel/oneapi/setvars.sh` is missing
- preferred Arch install path is:

```bash
./00-setup/install-oneapi-arch.sh
```

That currently resolves to:

```bash
sudo pacman -S --needed -- intel-oneapi-dpcpp-cpp intel-oneapi-mkl-sycl
```

## Current env versions

- `intel-inf-openvino`: `openvino 2026.0.0`
- `intel-inf-openvino-genai`: `openvino 2026.0.0`, `openvino-genai 2026.0.0.0`
- `intel-inf-optimum-openvino`: `openvino 2026.0.0`, `optimum-intel 1.27.0`, `torch 2.10.0`
- `intel-inf-torch-xpu`: `torch 2.10.0+xpu`, `torchvision 0.25.0+xpu`, `torchaudio 2.10.0+xpu`

## Notes

- The minimal OpenVINO env should stay minimal by default. `optimum-intel[openvino]` pulls a large generic `torch` stack and should be opt-in.
- The current NPU failure mode is userspace loader-path related, not a missing kernel driver.
