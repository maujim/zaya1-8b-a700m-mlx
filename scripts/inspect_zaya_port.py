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


def iter_mlx_parameters(config_path: Path) -> dict[str, tuple[int, ...]]:
    args = ZayaArgs.from_json(config_path)
    model = ZayaForCausalLM(args)
    params: dict[str, tuple[int, ...]] = {}

    def walk(prefix: str, value):
        if isinstance(value, dict):
            for key, child in value.items():
                walk(f"{prefix}.{key}" if prefix else str(key), child)
        elif isinstance(value, (list, tuple)):
            for i, child in enumerate(value):
                walk(f"{prefix}.{i}" if prefix else str(i), child)
        else:
            params[prefix] = tuple(value.shape)

    walk("", model.parameters())
    return dict(sorted(params.items()))


def iter_hf_indexed_shapes(model_path: Path) -> dict[str, tuple[int, ...]]:
    index = json.loads((model_path / "model.safetensors.index.json").read_text())
    by_shard: dict[str, list[str]] = {}
    for key, shard in index["weight_map"].items():
        by_shard.setdefault(shard, []).append(key)

    shapes: dict[str, tuple[int, ...]] = {}
    for shard, keys in by_shard.items():
        with safe_open(model_path / shard, framework="np") as f:
            for key in keys:
                shape = tuple(f.get_slice(key).get_shape())
                if ".conv_qk.0.weight" in key or ".conv_qk.1.weight" in key:
                    shape = (shape[0], shape[2], shape[1])
                shapes[key] = shape
    return dict(sorted(shapes.items()))


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
    hf_shapes = iter_hf_indexed_shapes(model_path)
    mlx_shapes = iter_mlx_parameters(model_path / "config.json")
    hf_keys = sorted(hf_shapes)
    mlx_keys = sorted(mlx_shapes)

    missing_in_mlx = sorted(set(hf_keys) - set(mlx_keys) - {"lm_head.weight"})
    extra_in_mlx = sorted(set(mlx_keys) - set(hf_keys))
    common = sorted((set(hf_keys) & set(mlx_keys)) - {"lm_head.weight"})
    shape_mismatches = [(key, hf_shapes[key], mlx_shapes[key]) for key in common if hf_shapes[key] != mlx_shapes[key]]

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
    print(f"shape mismatches after sanitize rules: {len(shape_mismatches)}")
    for key, hf_shape, mlx_shape in shape_mismatches[:50]:
        print(f"  - {key}: hf/sanitized={hf_shape}, mlx={mlx_shape}")
    if len(shape_mismatches) > 50:
        print(f"  ... {len(shape_mismatches) - 50} more")

    if args.show_shapes:
        inspect_shapes(model_path, args.show_shapes)

    if missing_in_mlx or extra_in_mlx or shape_mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
