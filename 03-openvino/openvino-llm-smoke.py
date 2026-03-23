#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import openvino_genai as ov_genai

from openvino_llm_utils import (
    build_generation_config,
    count_prompt_tokens,
    count_tokens_from_text,
    extract_text,
    load_prompt_records,
)


def cache_dir_for(model_dir: Path, device: str) -> Path | None:
    if device == "CPU":
        return None
    safe_device = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in device)
    return model_dir / ".ov_cache" / safe_device


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a direct OpenVINO GenAI LLM smoke test")
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--prompt-file", required=True, type=Path)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    ov_config: dict[str, Any] = {}
    cache_dir = cache_dir_for(args.model_dir, args.device)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        ov_config["CACHE_DIR"] = str(cache_dir)

    pipe = ov_genai.LLMPipeline(str(args.model_dir), args.device, **ov_config)
    tokenizer = pipe.get_tokenizer()
    prompt_records = load_prompt_records(args.prompt_file)

    rows: list[dict[str, Any]] = []
    for prompt_index, record in enumerate(prompt_records):
        prompt_id = record.get("id", f"prompt-{prompt_index}")
        messages = record["messages"]
        generation_config = build_generation_config(
            ov_genai,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            stop_strings=record.get("stop"),
        )
        started = time.perf_counter()
        result = pipe.generate(ov_genai.ChatHistory(messages), generation_config=generation_config)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        text = extract_text(result)
        prompt_tokens = count_prompt_tokens(tokenizer, messages)
        completion_tokens = count_tokens_from_text(tokenizer, text)
        row = {
            "prompt_id": prompt_id,
            "device": args.device,
            "model_dir": str(args.model_dir),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "elapsed_ms": elapsed_ms,
            "tokens_per_s": completion_tokens / (elapsed_ms / 1000.0) if completion_tokens > 0 and elapsed_ms > 0 else None,
            "response_text": text,
        }
        rows.append(row)
        print(
            f"{prompt_id}: elapsed_ms={elapsed_ms:.1f} prompt_tokens={prompt_tokens} "
            f"completion_tokens={completion_tokens}"
        )

    payload = {
        "device": args.device,
        "model_dir": str(args.model_dir),
        "rows": rows,
    }
    if args.output_json:
        args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
