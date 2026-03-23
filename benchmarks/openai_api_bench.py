#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
import uuid
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
                    raise ValueError(f"{path}:{lineno}: each JSONL entry needs either 'messages' or 'prompt'")
                payload["messages"] = [{"role": "user", "content": payload.pop("prompt")}]
            records.append(payload)
    if not records:
        raise ValueError(f"no prompt records found in {path}")
    return records


def content_from_nonstream_response(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    choice = choices[0]
    if "message" in choice and isinstance(choice["message"], dict):
        return str(choice["message"].get("content") or "")
    if "text" in choice:
        return str(choice["text"] or "")
    return ""


def make_request(
    *,
    base_url: str,
    api_key: str,
    body: dict[str, Any],
    timeout_s: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    request_id = f"bench-{uuid.uuid4().hex}"
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        method="POST",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "X-Benchmark-Request-Id": request_id,
        },
    )

    t0 = time.perf_counter()
    if body.get("stream"):
        text_parts: list[str] = []
        usage: dict[str, Any] | None = None
        response_id: str | None = None
        finish_reason: str | None = None
        first_token_ms: float | None = None
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                event = json.loads(payload)
                response_id = response_id or event.get("id")
                if event.get("usage") is not None:
                    usage = event["usage"]
                choices = event.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                finish_reason = choice.get("finish_reason") or finish_reason
                delta = choice.get("delta") or {}
                piece = delta.get("content")
                if piece:
                    if first_token_ms is None:
                        first_token_ms = (time.perf_counter() - t0) * 1000.0
                    text_parts.append(piece)
        total_ms = (time.perf_counter() - t0) * 1000.0
        response_payload = {
            "id": response_id,
            "choices": [{"finish_reason": finish_reason, "message": {"role": "assistant", "content": "".join(text_parts)}}],
            "usage": usage,
        }
        metrics = {
            "request_id": request_id,
            "ttft_ms": first_token_ms,
            "total_ms": total_ms,
        }
        return response_payload, metrics

    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        payload = json.loads(response.read().decode("utf-8"))
    total_ms = (time.perf_counter() - t0) * 1000.0
    metrics = {
        "request_id": request_id,
        "ttft_ms": None,
        "total_ms": total_ms,
    }
    return payload, metrics


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    overall = {
        "num_runs": len(rows),
        "median_total_ms": statistics.median([row["total_ms"] for row in rows]) if rows else None,
        "median_ttft_ms": statistics.median([row["ttft_ms"] for row in rows if row["ttft_ms"] is not None]) if any(
            row["ttft_ms"] is not None for row in rows
        ) else None,
        "median_tokens_per_s": statistics.median(
            [row["tokens_per_s"] for row in rows if row["tokens_per_s"] is not None]
        ) if any(row["tokens_per_s"] is not None for row in rows) else None,
    }

    by_prompt: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_prompt.setdefault(str(row["prompt_id"]), []).append(row)

    prompt_summary: dict[str, Any] = {}
    for prompt_id, prompt_rows in by_prompt.items():
        prompt_summary[prompt_id] = {
            "num_runs": len(prompt_rows),
            "median_total_ms": statistics.median([row["total_ms"] for row in prompt_rows]),
            "median_ttft_ms": statistics.median([row["ttft_ms"] for row in prompt_rows if row["ttft_ms"] is not None])
            if any(row["ttft_ms"] is not None for row in prompt_rows)
            else None,
            "median_tokens_per_s": statistics.median([row["tokens_per_s"] for row in prompt_rows if row["tokens_per_s"] is not None])
            if any(row["tokens_per_s"] is not None for row in prompt_rows)
            else None,
        }

    return {
        "overall": overall,
        "by_prompt": prompt_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark an OpenAI-compatible chat/completions endpoint")
    parser.add_argument("--base-url", required=True, help="Server base URL, e.g. http://127.0.0.1:8010")
    parser.add_argument("--api-key", default="local-benchmark", help="Bearer token for the OpenAI-compatible endpoint")
    parser.add_argument("--model", required=True, help="Model name to send in the request payload")
    parser.add_argument("--prompt-file", required=True, type=Path, help="JSONL file containing prompts or messages")
    parser.add_argument("--output-jsonl", required=True, type=Path, help="Path to write per-run JSONL rows")
    parser.add_argument("--summary-json", type=Path, help="Optional path to write aggregate summary JSON")
    parser.add_argument("--repeats", type=int, default=1, help="Measured repeats per prompt")
    parser.add_argument("--max-tokens", type=int, default=128, help="max_tokens for chat/completions")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature; 0 means deterministic")
    parser.add_argument("--top-p", type=float, default=1.0, help="top_p value if sampling is enabled")
    parser.add_argument("--timeout", type=float, default=300.0, help="HTTP timeout in seconds")
    parser.add_argument("--stream", action="store_true", help="Use streaming chat/completions and measure TTFT")
    args = parser.parse_args()

    prompt_records = load_prompt_records(args.prompt_file)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for prompt_index, prompt_record in enumerate(prompt_records):
            prompt_id = prompt_record.get("id", f"prompt-{prompt_index}")
            messages = prompt_record["messages"]
            for repeat in range(args.repeats):
                body = {
                    "model": args.model,
                    "messages": messages,
                    "max_tokens": args.max_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "stream": args.stream,
                }
                if args.stream:
                    body["stream_options"] = {"include_usage": True}

                try:
                    response_payload, metrics = make_request(
                        base_url=args.base_url,
                        api_key=args.api_key,
                        body=body,
                        timeout_s=args.timeout,
                    )
                except urllib.error.HTTPError as exc:
                    detail = exc.read().decode("utf-8", errors="replace")
                    row = {
                        "prompt_id": prompt_id,
                        "repeat": repeat,
                        "ok": False,
                        "error": f"HTTP {exc.code}",
                        "detail": detail[:1000],
                    }
                    rows.append(row)
                    handle.write(json.dumps(row, ensure_ascii=True) + "\n")
                    handle.flush()
                    print(f"{prompt_id} repeat={repeat}: HTTP {exc.code}", file=sys.stderr)
                    continue
                except Exception as exc:  # pragma: no cover - operational fallback
                    row = {
                        "prompt_id": prompt_id,
                        "repeat": repeat,
                        "ok": False,
                        "error": type(exc).__name__,
                        "detail": str(exc)[:1000],
                    }
                    rows.append(row)
                    handle.write(json.dumps(row, ensure_ascii=True) + "\n")
                    handle.flush()
                    print(f"{prompt_id} repeat={repeat}: {type(exc).__name__}", file=sys.stderr)
                    continue

                usage = response_payload.get("usage") or {}
                completion_tokens = usage.get("completion_tokens")
                total_ms = metrics["total_ms"]
                ttft_ms = metrics["ttft_ms"]
                gen_ms = None if ttft_ms is None else max(total_ms - ttft_ms, 0.0)
                tokens_per_s = None
                if completion_tokens and gen_ms and gen_ms > 0:
                    tokens_per_s = completion_tokens / (gen_ms / 1000.0)

                row = {
                    "prompt_id": prompt_id,
                    "repeat": repeat,
                    "ok": True,
                    "request_id": metrics["request_id"],
                    "response_id": response_payload.get("id"),
                    "prompt_messages": messages,
                    "response_text": content_from_nonstream_response(response_payload),
                    "finish_reason": ((response_payload.get("choices") or [{}])[0]).get("finish_reason"),
                    "usage": usage,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": completion_tokens,
                    "total_tokens": usage.get("total_tokens"),
                    "total_ms": total_ms,
                    "ttft_ms": ttft_ms,
                    "generation_ms": gen_ms,
                    "tokens_per_s": tokens_per_s,
                    "stream": args.stream,
                    "model": args.model,
                    "base_url": args.base_url,
                    "max_tokens": args.max_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                }
                rows.append(row)
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
                handle.flush()
                summary_bits = [
                    f"prompt={prompt_id}",
                    f"repeat={repeat}",
                    f"total_ms={total_ms:.1f}",
                ]
                if ttft_ms is not None:
                    summary_bits.append(f"ttft_ms={ttft_ms:.1f}")
                if tokens_per_s is not None:
                    summary_bits.append(f"tok_s={tokens_per_s:.2f}")
                print(" ".join(summary_bits))

    summary = summarize([row for row in rows if row.get("ok")])
    if args.summary_json is not None:
        args.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
