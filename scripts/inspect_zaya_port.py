#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors import safe_open

from run_zaya_mlx import MODEL_ID, ZayaArgs, ZayaForCausalLM


def iter_index_keys(model_path: Path) -> list[str]:
    index_path = model_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    return sorted(index["weight_map"])


def iter_mlx_parameter_keys(config_path: Path) -> list[str]:
    args = ZayaArgs.from_json(config_path)
    model = ZayaForCausalLM(args)
    keys: list[str] = []

    def walk(prefix: str, value):
        if isinstance(value, dict):
            for key, child in value.items():
                walk(f"{prefix}.{key}" if prefix else str(key), child)
        elif isinstance(value, (list, tuple)):
            for i, child in enumerate(value):
                walk(f"{prefix}.{i}" if prefix else str(i), child)
        else:
            keys.append(prefix)

    walk("", model.parameters())
    return sorted(keys)


def inspect_shapes(model_path: Path, limit: int) -> None:
    shards = sorted(model_path.glob("model-*.safetensors"))
    printed = 0
    for shard in shards:
        with safe_open(shard, framework="np") as f:
            for key in f.keys():
                tensor = f.get_slice(key)
                print(f"{key}: {tensor.get_shape()} {tensor.get_dtype()}")
                printed += 1
                if printed >= limit:
                    return


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect ZAYA MLX port metadata without loading model tensors.")
    parser.add_argument("--model-path", type=Path, help="Use an existing Hugging Face snapshot directory.")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--show-shapes", type=int, default=0, metavar="N", help="Print the first N safetensor shapes.")
    args = parser.parse_args()

    model_path = args.model_path or Path(snapshot_download(MODEL_ID, local_files_only=args.local_files_only))
    hf_keys = iter_index_keys(model_path)
    mlx_keys = iter_mlx_parameter_keys(model_path / "config.json")

    missing_in_mlx = sorted(set(hf_keys) - set(mlx_keys) - {"lm_head.weight"})
    extra_in_mlx = sorted(set(mlx_keys) - set(hf_keys))

    print(f"snapshot: {model_path}")
    print(f"hf indexed tensors: {len(hf_keys)}")
    print(f"mlx parameters: {len(mlx_keys)}")
    print(f"missing in mlx: {len(missing_in_mlx)}")
    for key in missing_in_mlx[:50]:
        print(f"  - {key}")
    if len(missing_in_mlx) > 50:
        print(f"  ... {len(missing_in_mlx) - 50} more")
    print(f"extra in mlx: {len(extra_in_mlx)}")
    for key in extra_in_mlx[:50]:
        print(f"  - {key}")
    if len(extra_in_mlx) > 50:
        print(f"  ... {len(extra_in_mlx) - 50} more")

    if args.show_shapes:
        inspect_shapes(model_path, args.show_shapes)

    if missing_in_mlx or extra_in_mlx:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
