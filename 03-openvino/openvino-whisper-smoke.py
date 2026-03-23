#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
import wave
from array import array
from pathlib import Path
from typing import Any

import openvino_genai as ov_genai


def read_wav_mono_16k(path: Path) -> list[float]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())

    if sample_width != 2:
        raise RuntimeError(f"expected 16-bit PCM WAV, got sample width {sample_width} in {path}")
    if sample_rate != 16000:
        raise RuntimeError(f"expected 16 kHz WAV, got {sample_rate} Hz in {path}")

    samples = array("h")
    samples.frombytes(frames)
    if channels > 1:
        mono: list[float] = []
        for idx in range(0, len(samples), channels):
            frame = samples[idx : idx + channels]
            mono.append(sum(frame) / (len(frame) * 32768.0))
        return mono
    return [value / 32768.0 for value in samples]


def cache_dir_for(model_dir: Path, device: str) -> Path | None:
    if device == "CPU":
        return None
    safe_device = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in device)
    return model_dir / ".ov_cache" / safe_device


def result_text(result: Any) -> str:
    text = getattr(result, "text", None)
    if text is not None:
        return str(text)
    texts = getattr(result, "texts", None)
    if texts:
        return str(texts[0])
    return str(result)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a direct OpenVINO GenAI Whisper smoke test")
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--audio-path", required=True, type=Path)
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--language", default="<|en|>")
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    ov_config: dict[str, Any] = {}
    cache_dir = cache_dir_for(args.model_dir, args.device)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        ov_config["CACHE_DIR"] = str(cache_dir)

    pipe = ov_genai.WhisperPipeline(str(args.model_dir), args.device, **ov_config)
    config = pipe.get_generation_config()
    config.language = args.language
    config.task = args.task
    config.max_new_tokens = args.max_tokens

    audio = read_wav_mono_16k(args.audio_path)
    started = time.perf_counter()
    result = pipe.generate(audio, config)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    payload = {
        "device": args.device,
        "model_dir": str(args.model_dir),
        "audio_path": str(args.audio_path),
        "elapsed_ms": elapsed_ms,
        "text": result_text(result),
        "chunks": [
            {"start_ts": chunk.start_ts, "end_ts": chunk.end_ts, "text": chunk.text}
            for chunk in (getattr(result, "chunks", None) or [])
        ],
    }
    if args.output_json:
        args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"elapsed_ms={elapsed_ms:.1f}")
    print(payload["text"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
