#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def fmt_ms(value: Any) -> str:
    return "-" if value is None else f"{float(value):.1f} ms"


def fmt_tps(value: Any) -> str:
    return "-" if value is None else f"{float(value):.2f} tok/s"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize ZAYA MLX profile JSON speed stats.")
    parser.add_argument("profile", type=Path)
    args = parser.parse_args()

    data = json.loads(args.profile.read_text())
    gen = data.get("generation_stats", {})
    memory = data.get("memory", {})

    print(f"profile: {args.profile}")
    if "prompt_tokens" in gen:
        print(f"prompt_tokens: {gen['prompt_tokens']}")
    if prefill := gen.get("prefill"):
        print(f"prefill: total={fmt_ms(prefill.get('total_ms'))} count={prefill.get('count')}")
    if decode := gen.get("decode"):
        print(
            "decode: "
            f"count={decode.get('count')} "
            f"avg={fmt_ms(decode.get('avg_ms'))} "
            f"p50={fmt_ms(decode.get('p50_ms'))} "
            f"p90={fmt_ms(decode.get('p90_ms'))} "
            f"p99={fmt_ms(decode.get('p99_ms'))} "
            f"speed={fmt_tps(decode.get('tokens_per_s'))}"
        )
    if uncached := gen.get("uncached_generate"):
        print(
            "uncached_generate: "
            f"count={uncached.get('count')} "
            f"avg={fmt_ms(uncached.get('avg_ms'))} "
            f"p50={fmt_ms(uncached.get('p50_ms'))} "
            f"p90={fmt_ms(uncached.get('p90_ms'))} "
            f"p99={fmt_ms(uncached.get('p99_ms'))} "
            f"speed={fmt_tps(uncached.get('tokens_per_s'))}"
        )
    if memory:
        mem = ", ".join(f"{key}={float(value):.1f} MB" for key, value in memory.items())
        print(f"memory: {mem}")
    print(f"total_wall: {fmt_ms(data.get('total_wall_ms'))}")


if __name__ == "__main__":
    main()
