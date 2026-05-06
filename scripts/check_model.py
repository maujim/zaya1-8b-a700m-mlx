#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_ID = "Zyphra/ZAYA1-8B"


def bytes_to_gib(size: int) -> str:
    return f"{size / (1024 ** 3):.2f} GiB"


def main() -> None:
    parser = argparse.ArgumentParser(description="Check/download the local ZAYA1-8B snapshot.")
    parser.add_argument("--download", action="store_true", help="Resume the Hugging Face download first.")
    args = parser.parse_args()

    if args.download:
        model_path = Path(snapshot_download(MODEL_ID))
    else:
        model_path = Path(snapshot_download(MODEL_ID, local_files_only=True))

    index_path = model_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    expected = sorted(set(index["weight_map"].values()))

    missing = []
    total = 0
    for filename in expected:
        path = model_path / filename
        if path.exists():
            total += path.stat().st_size
        else:
            missing.append(filename)

    print(f"snapshot: {model_path}")
    print(f"expected shards: {len(expected)}")
    print(f"present shards: {len(expected) - len(missing)}")
    print(f"present weight bytes: {bytes_to_gib(total)}")
    print(f"index total size: {bytes_to_gib(int(index['metadata']['total_size']))}")

    if missing:
        print("missing shards:")
        for filename in missing:
            print(f"  - {filename}")
        raise SystemExit(1)

    print("all weight shards are present")


if __name__ == "__main__":
    main()
