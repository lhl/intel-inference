#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_prompt_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if "messages" not in payload:
                if "prompt" not in payload:
                    raise ValueError(f"{path}:{lineno}: expected 'messages' or 'prompt'")
                payload["messages"] = [{"role": "user", "content": payload.pop("prompt")}]
            records.append(payload)
    if not records:
        raise ValueError(f"no prompt records found in {path}")
    return records


def _normalize_ids(data: Any) -> list[int]:
    if hasattr(data, "tolist"):
        data = data.tolist()
    if isinstance(data, list) and data and isinstance(data[0], list):
        return list(data[0])
    if isinstance(data, list):
        return list(data)
    return []


def count_tokens_from_text(tokenizer: Any, text: str) -> int:
    encoded = tokenizer.encode(text, add_special_tokens=False).input_ids.data
    return len(_normalize_ids(encoded))


def count_prompt_tokens(tokenizer: Any, messages: list[dict[str, Any]]) -> int:
    templated = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    return count_tokens_from_text(tokenizer, templated)


def extract_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    texts = getattr(result, "texts", None)
    if texts:
        return str(texts[0])
    text_attr = getattr(result, "text", None)
    if text_attr is not None:
        return str(text_attr)
    return str(result)


def build_generation_config(
    ov_genai: Any,
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    stop_strings: list[str] | None,
) -> Any:
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = int(max_tokens)
    config.apply_chat_template = True
    if temperature > 0:
        config.do_sample = True
        config.temperature = float(temperature)
        config.top_p = float(top_p)
    else:
        config.do_sample = False
    if stop_strings:
        config.stop_strings = set(stop_strings)
    config.validate()
    return config
