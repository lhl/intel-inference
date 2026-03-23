# 00-setup Checklist

Use this as the first practical bring-up checklist for a new machine.

## 1. Capture baseline system state

Run:

```bash
./00-setup/collect-system-info.sh --system-id <sanitized-label> --write-tracked-summary
```

That should give us:

- an ignored timestamped raw inventory under `00-setup/results/`
- a sanitized tracked summary under `00-setup/systems/`

## 2. Verify Linux graphics and device access

Check:

- Intel GPU is visible in `lspci`
- `/dev/dri` exists
- your user is in the `render` and `video` groups when required
- `vulkaninfo` works without silently preferring only software devices
- `clinfo` works if OpenCL paths matter for the stack under test

## 3. Verify optional Intel-specific tooling

If SYCL or oneAPI is in scope:

- verify oneAPI is installed
- verify `sycl-ls` works if available
- verify Level Zero related tooling is present where needed

If NPU is in scope:

- verify the NPU driver is installed
- verify the device is visible through the relevant OpenVINO or system tooling

## 4. Create per-stack envs

The default env split should be:

- `openvino`
- `openvino-genai`
- `torch-xpu`
- optional source-build envs for `vLLM` and SGLang

Do not try to force everything into one shared Python env.

## 5. Run smoke tests before benchmarking

At minimum, confirm:

- OpenVINO can load on the target machine
- OpenVINO GenAI imports and exposes `LLMPipeline`
- PyTorch can see `torch.xpu` if that path is in scope
- `llama.cpp` backend builds complete for the backends we intend to benchmark

## 6. Record exact versions

Before moving into `01-hardware/`, record:

- kernel
- Mesa or Intel driver versions
- oneAPI version if used
- OpenVINO version
- OpenVINO GenAI version
- PyTorch version
- relevant submodule commits for `llama.cpp` and other test targets

## Related docs

- [README.md](/home/lhl/github/lhl/intel-inference/README.md)
- [IMPLEMENTATION.md](/home/lhl/github/lhl/intel-inference/IMPLEMENTATION.md)
- [TESTING.md](/home/lhl/github/lhl/intel-inference/TESTING.md)
