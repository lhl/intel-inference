#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.metadata as md
import importlib.util
import json
import os
from pathlib import Path


def collect_xpu() -> dict[str, object]:
    import torch
    import vllm
    from vllm.platforms import current_platform

    return {
        "backend": "xpu",
        "python": os.sys.version.split()[0],
        "torch": torch.__version__,
        "vllm": md.version("vllm"),
        "xpu_available": torch.xpu.is_available(),
        "xpu_count": torch.xpu.device_count() if torch.xpu.is_available() else 0,
        "xpu_name": torch.xpu.get_device_name(0) if torch.xpu.is_available() else None,
        "current_platform": current_platform.__class__.__name__,
        "vllm_origin": getattr(vllm, "__file__", None),
        "vllm_target_device": os.environ.get("VLLM_TARGET_DEVICE"),
    }


def collect_openvino() -> dict[str, object]:
    import openvino as ov
    import torch
    import vllm
    import vllm_openvino

    core = ov.Core()
    return {
        "backend": "openvino",
        "python": os.sys.version.split()[0],
        "openvino": ov.__version__,
        "torch": torch.__version__,
        "vllm": md.version("vllm"),
        "vllm_openvino": md.version("vllm-openvino"),
        "available_devices": core.available_devices,
        "vllm_openvino_device": os.environ.get("VLLM_OPENVINO_DEVICE"),
        "vllm_origin": getattr(vllm, "__file__", None),
        "plugin_origin": getattr(vllm_openvino, "__file__", None),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect environment details for 05-vllm")
    parser.add_argument("--backend", required=True, choices=("xpu", "openvino"))
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    payload = collect_xpu() if args.backend == "xpu" else collect_openvino()
    text = json.dumps(payload, indent=2) + "\n"
    print(text, end="")
    if args.output_json is not None:
        args.output_json.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
