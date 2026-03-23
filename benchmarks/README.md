# Benchmarks

This directory holds shared benchmark tooling that multiple runtime phases can reuse.

Current shared tools:

- `openai_api_bench.py`
  - a dependency-light benchmark client for OpenAI-compatible `chat/completions` endpoints
  - intended to benchmark `03-openvino` adapter servers, `04-llama.cpp` server runs, and `05-vllm` server runs through the same request path
- `prompts/small-llm-chat.jsonl`
  - a small default chat prompt set for first-pass LLM smoke and latency checks

Results should normally live in the phase-specific `results/` directories rather than here.
