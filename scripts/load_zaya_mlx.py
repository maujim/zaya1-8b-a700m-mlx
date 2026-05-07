#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

from run_zaya_mlx import MODEL_ID, QUANT_CHOICES, Profiler, load_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Load ZAYA1-8B with the local MLX port and print a profile.")
    parser.add_argument("--profile-json", type=Path, help="Write the full profile as JSON.")
    parser.add_argument("--quant", choices=QUANT_CHOICES, default="q4", help="Weight mode: full BF16 weights or dynamic in-memory q8/q6/q4 quantization after load (default: q4 for local speed).")
    parser.add_argument(
        "--q8-min-weight-size",
        type=int,
        default=1_000_000,
        help="Only quantize Linear weights with at least this many parameters for q8/q6/q4. Use 0 for exhaustive behavior.",
    )
    args = parser.parse_args()

    profiler = Profiler(enabled=True)

    with profiler.span("resolve_model_path"):
        model_path = Path(snapshot_download(MODEL_ID))
    print(f"MLX model path: {model_path}", flush=True)

    model = load_model(model_path, profiler, quant=args.quant, q8_min_weight_size=args.q8_min_weight_size)
    with profiler.span("final_parameter_sync", force_eval=model.parameters()):
        pass

    print("MLX model loaded and synchronized", flush=True)
    profiler.print_report()

    if args.profile_json:
        args.profile_json.write_text(json.dumps(profiler.report(), indent=2))
        print(f"profile JSON written to {args.profile_json}")


if __name__ == "__main__":
    main()
