#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
from typing import Any

import openvino as ov


def maybe_version(dist_name: str) -> str | None:
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def device_info(core: ov.Core, device: str) -> dict[str, Any]:
    info: dict[str, Any] = {"device": device, "status": "OK"}
    try:
        info["full_device_name"] = core.get_property(device, "FULL_DEVICE_NAME")
    except Exception as exc:
        info["status"] = "ERR"
        info["detail"] = str(exc).strip().splitlines()[0][:240]
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description="Check OpenVINO-family env state")
    parser.add_argument("--label", required=True)
    parser.add_argument("--probe-devices", nargs="+", default=["CPU", "GPU", "NPU"])
    parser.add_argument("--check-genai", action="store_true")
    parser.add_argument("--check-optimum", action="store_true")
    parser.add_argument("--json-out", type=str)
    args = parser.parse_args()

    core = ov.Core()
    payload: dict[str, Any] = {
        "label": args.label,
        "openvino": ov.__version__,
        "available_devices": list(core.available_devices),
        "probe_devices": [device_info(core, device) for device in args.probe_devices],
    }

    if args.check_genai:
        import openvino_genai as ovg

        payload["openvino_genai"] = {
            "version": maybe_version("openvino-genai"),
            "module_version": getattr(ovg, "__version__", None),
            "has_llm_pipeline": hasattr(ovg, "LLMPipeline"),
            "has_whisper_pipeline": hasattr(ovg, "WhisperPipeline"),
            "has_tts_pipeline": hasattr(ovg, "Text2SpeechPipeline"),
        }

    if args.check_optimum:
        from optimum.intel.openvino import OVModelForCausalLM

        payload["optimum_intel"] = {
            "version": maybe_version("optimum-intel"),
            "has_ovmodel_for_causallm": OVModelForCausalLM is not None,
            "optimum_cli": shutil.which("optimum-cli"),
        }

    print("label", payload["label"])
    print("openvino", payload["openvino"])
    print("available_devices", payload["available_devices"])
    for device in payload["probe_devices"]:
        line = f"{device['device']}: {device['status']}"
        if "full_device_name" in device:
            line += f" | {device['full_device_name']}"
        if "detail" in device:
            line += f" | {device['detail']}"
        print(line)

    if "openvino_genai" in payload:
        genai = payload["openvino_genai"]
        print(
            "openvino_genai",
            genai["version"],
            f"LLMPipeline={genai['has_llm_pipeline']}",
            f"WhisperPipeline={genai['has_whisper_pipeline']}",
            f"Text2SpeechPipeline={genai['has_tts_pipeline']}",
        )

    if "optimum_intel" in payload:
        optimum = payload["optimum_intel"]
        print(
            "optimum_intel",
            optimum["version"],
            f"OVModelForCausalLM={optimum['has_ovmodel_for_causallm']}",
            f"optimum_cli={optimum['optimum_cli']}",
        )

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
