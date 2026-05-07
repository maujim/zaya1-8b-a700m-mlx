#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Zyphra/ZAYA1-8B"


def tick(label: str, start: float) -> float:
    now = time.perf_counter()
    print(f"{label:<24} {(now - start) * 1000:10.1f} ms", flush=True)
    return time.perf_counter()


def main() -> None:
    parser = argparse.ArgumentParser(description="Load ZAYA1-8B through Hugging Face Transformers/PyTorch.")
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    args = parser.parse_args()

    total = time.perf_counter()
    t = total
    model_path = Path(snapshot_download(MODEL_ID))
    t = tick("resolve_model_path", t)
    print(f"HF model path: {model_path}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    t = tick("load_tokenizer", t)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    t = tick("load_model", t)

    model.to(args.device)
    t = tick(f"move_to_{args.device}", t)

    # Keep refs alive and make linters happy.
    _ = tokenizer, model
    print(f"total wall               {(time.perf_counter() - total) * 1000:10.1f} ms")


if __name__ == "__main__":
    main()
