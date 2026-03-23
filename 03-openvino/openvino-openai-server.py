#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import openvino_genai as ov_genai

from openvino_llm_utils import (
    build_generation_config,
    count_prompt_tokens,
    count_tokens_from_text,
    extract_text,
)


def safe_device_label(device: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in device)


class LoadedModel:
    def __init__(self, model_dir: Path, device: str, served_model_name: str) -> None:
        ov_config: dict[str, Any] = {}
        if device != "CPU":
            cache_dir = model_dir / ".ov_cache" / safe_device_label(device)
            cache_dir.mkdir(parents=True, exist_ok=True)
            ov_config["CACHE_DIR"] = str(cache_dir)
        self.model_dir = model_dir
        self.device = device
        self.served_model_name = served_model_name
        self.pipe = ov_genai.LLMPipeline(str(model_dir), device, **ov_config)
        self.tokenizer = self.pipe.get_tokenizer()
        self.lock = threading.Lock()

    def run_chat(
        self,
        payload: dict[str, Any],
        stream_callback: Any | None = None,
    ) -> dict[str, Any]:
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("chat/completions requires a non-empty messages array")

        generation_config = build_generation_config(
            ov_genai,
            max_tokens=int(payload.get("max_tokens") or payload.get("max_completion_tokens") or 128),
            temperature=float(payload.get("temperature", 0.0)),
            top_p=float(payload.get("top_p", 1.0)),
            stop_strings=list(payload.get("stop") or []),
        )

        completion_parts: list[str] = []

        def streamer(piece: str) -> None:
            if not piece:
                return
            completion_parts.append(piece)
            if stream_callback is not None:
                stream_callback(piece)

        with self.lock:
            result = self.pipe.generate(
                ov_genai.ChatHistory(messages),
                generation_config=generation_config,
                streamer=streamer if stream_callback is not None else None,
            )

        text = extract_text(result)
        if not text and completion_parts:
            text = "".join(completion_parts)
        prompt_tokens = count_prompt_tokens(self.tokenizer, messages)
        completion_tokens = count_tokens_from_text(self.tokenizer, text)
        return {
            "text": text,
            "finish_reason": "stop",
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }


MODEL: LoadedModel | None = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[openvino-openai-server] {self.address_string()} - {fmt % args}")

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def sse_event(self, payload: dict[str, Any]) -> None:
        data = ("data: " + json.dumps(payload) + "\n\n").encode("utf-8")
        self.wfile.write(data)
        self.wfile.flush()

    def do_GET(self) -> None:  # noqa: N802
        assert MODEL is not None
        if self.path == "/health":
            self.send_json(
                200,
                {
                    "status": "ok",
                    "model": MODEL.served_model_name,
                    "device": MODEL.device,
                },
            )
            return
        if self.path == "/v1/models":
            self.send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": MODEL.served_model_name,
                            "object": "model",
                            "owned_by": "local-openvino",
                        }
                    ],
                },
            )
            return
        self.send_json(404, {"error": {"message": f"unknown path: {self.path}"}})

    def do_POST(self) -> None:  # noqa: N802
        assert MODEL is not None
        if self.path != "/v1/chat/completions":
            self.send_json(404, {"error": {"message": f"unknown path: {self.path}"}})
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(content_length) or b"{}")
        stream = bool(payload.get("stream"))
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        try:
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                self.sse_event(
                    {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": MODEL.served_model_name,
                        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                    }
                )

                include_usage = bool((payload.get("stream_options") or {}).get("include_usage"))

                def stream_piece(piece: str) -> None:
                    self.sse_event(
                        {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": MODEL.served_model_name,
                            "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
                        }
                    )

                result = MODEL.run_chat(payload, stream_piece)
                final_payload = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": MODEL.served_model_name,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": result["finish_reason"]}],
                }
                if include_usage:
                    final_payload["usage"] = result["usage"]
                self.sse_event(final_payload)
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return

            result = MODEL.run_chat(payload)
            self.send_json(
                200,
                {
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": MODEL.served_model_name,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": result["text"]},
                            "finish_reason": result["finish_reason"],
                        }
                    ],
                    "usage": result["usage"],
                },
            )
        except Exception as exc:
            self.send_json(500, {"error": {"message": f"{type(exc).__name__}: {exc}"}})


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve an OpenVINO GenAI LLM behind a minimal OpenAI-compatible API")
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--served-model-name")
    args = parser.parse_args()

    global MODEL
    MODEL = LoadedModel(args.model_dir, args.device, args.served_model_name or args.model_dir.name)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        json.dumps(
            {
                "status": "listening",
                "host": args.host,
                "port": args.port,
                "model": MODEL.served_model_name,
                "device": MODEL.device,
            }
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
