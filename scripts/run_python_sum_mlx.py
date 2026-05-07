#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from run_zaya_mlx import MODEL_ID, Profiler, generate, load_model

PROMPT = """Write a tiny Python function named sum_numbers that takes two numbers and returns their sum. Include only the code block."""


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask ZAYA1-8B via MLX to write a tiny Python sum function.")
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--profile-json", type=Path, help="Write the full profile as JSON.")
    parser.add_argument("--profile-layers", action="store_true", help="Slow: time every layer during generation.")
    args = parser.parse_args()

    profiler = Profiler(enabled=True, profile_layers=args.profile_layers)

    with profiler.span("resolve_model_path"):
        model_path = Path(snapshot_download(MODEL_ID))
    print(f"MLX model path: {model_path}", flush=True)

    model = load_model(model_path, profiler)
    with profiler.span("final_parameter_sync", force_eval=model.parameters()):
        pass
    print("MLX model loaded and synchronized", flush=True)

    with profiler.span("load_tokenizer"):
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print("\n=== prompt ===")
    print(PROMPT)
    print("\n=== answer ===", flush=True)

    pieces: list[str] = []
    for token in generate(model, tokenizer, PROMPT, args.max_new_tokens, args.temperature, profiler):
        text = tokenizer.decode([token], skip_special_tokens=True)
        pieces.append(text)
        print(text, end="", flush=True)
    if pieces:
        print()

    profiler.print_report()
    if args.profile_json:
        args.profile_json.write_text(json.dumps(profiler.report(), indent=2))
        print(f"profile JSON written to {args.profile_json}")


if __name__ == "__main__":
    main()
