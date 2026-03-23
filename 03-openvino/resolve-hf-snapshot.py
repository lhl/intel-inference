#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path


def repo_cache_dir(repo_id: str, repo_type: str, hf_home: Path) -> Path:
    safe_repo_id = repo_id.replace("/", "--")
    return hf_home / "hub" / f"{repo_type}s--{safe_repo_id}"


def snapshot_looks_complete(snapshot_dir: Path) -> bool:
    marker_names = (
        "model.safetensors",
        "pytorch_model.bin",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
        "openvino_model.xml",
    )
    return any((snapshot_dir / marker_name).exists() for marker_name in marker_names)


def resolve_snapshot_dir(repo_id: str, repo_type: str, hf_home: Path) -> Path | None:
    cache_dir = repo_cache_dir(repo_id, repo_type, hf_home)
    snapshots_dir = cache_dir / "snapshots"
    refs_dir = cache_dir / "refs"

    main_ref = refs_dir / "main"
    if main_ref.is_file():
        ref_name = main_ref.read_text(encoding="utf-8").strip()
        candidate = snapshots_dir / ref_name
        if candidate.is_dir() and snapshot_looks_complete(candidate):
            return candidate

    if not snapshots_dir.is_dir():
        return None

    candidates = sorted(
        (path for path in snapshots_dir.iterdir() if path.is_dir() and snapshot_looks_complete(path)),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve a local Hugging Face snapshot path if present")
    parser.add_argument("repo_id", help="Model repo ID, e.g. meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--repo-type", default="model", choices=("model", "dataset", "space"))
    args = parser.parse_args()

    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    snapshot_dir = resolve_snapshot_dir(args.repo_id, args.repo_type, hf_home)
    if snapshot_dir is None:
        return 1

    print(snapshot_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
