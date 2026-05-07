#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download

from run_zaya_mlx import MODEL_ID, Profiler, load_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile loading ZAYA1-8B into MLX without generating tokens.")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--model-path", type=Path, help="Use an existing Hugging Face snapshot directory.")
    parser.add_argument("--profile-json", type=Path, help="Write full profiling events and summary as JSON.")
    parser.add_argument(
        "--no-clear-cache",
        action="store_true",
        help="Do not clear the MLX cache before loading. By default this script clears it for cleaner memory numbers.",
    )
    args = parser.parse_args()

    profiler = Profiler(enabled=True, profile_layers=False)

    if not args.no_clear_cache:
        clear_cache = getattr(mx, "clear_cache", None)
        if clear_cache is not None:
            with profiler.span("clear_mlx_cache"):
                clear_cache()

    with profiler.span("resolve_model_path"):
        model_path = args.model_path or Path(snapshot_download(MODEL_ID, local_files_only=args.local_files_only))
    print(f"loading MLX model from: {model_path}", flush=True)

    model = load_model(model_path, profiler)

    # Force one final synchronization over all parameters so the load profile means
    # "resident/evaluated in MLX" rather than merely Python objects constructed.
    with profiler.span("final_parameter_sync", force_eval=model.parameters()):
        pass

    print("model loaded and MLX parameters synchronized", flush=True)
    profiler.print_report()

    if args.profile_json:
        args.profile_json.write_text(json.dumps(profiler.report(), indent=2))
        print(f"profile JSON written to {args.profile_json}")


if __name__ == "__main__":
    main()
