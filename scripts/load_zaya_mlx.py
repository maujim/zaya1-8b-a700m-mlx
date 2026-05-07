#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

from run_zaya_mlx import MODEL_ID, Profiler, load_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Load ZAYA1-8B with the local MLX port and print a profile.")
    parser.add_argument("--profile-json", type=Path, help="Write the full profile as JSON.")
    args = parser.parse_args()

    profiler = Profiler(enabled=True)

    with profiler.span("resolve_model_path"):
        model_path = Path(snapshot_download(MODEL_ID))
    print(f"MLX model path: {model_path}", flush=True)

    model = load_model(model_path, profiler)
    with profiler.span("final_parameter_sync", force_eval=model.parameters()):
        pass

    print("MLX model loaded and synchronized", flush=True)
    profiler.print_report()

    if args.profile_json:
        args.profile_json.write_text(json.dumps(profiler.report(), indent=2))
        print(f"profile JSON written to {args.profile_json}")


if __name__ == "__main__":
    main()
