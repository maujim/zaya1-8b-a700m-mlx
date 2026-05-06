#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from run_zaya_mlx import MODEL_ID, generate_from_messages, load_model


def decode_reply(tokenizer, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Start a minimal terminal chat with ZAYA1-8B on MLX.")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--model-path", type=Path, help="Use an existing Hugging Face snapshot directory.")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    model_path = args.model_path or Path(snapshot_download(MODEL_ID, local_files_only=args.local_files_only))
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = load_model(model_path)

    validation = list(
        generate_from_messages(
            model,
            tokenizer,
            [{"role": "user", "content": "Say hello."}],
            max_new_tokens=1,
            temperature=0.0,
        )
    )
    if not validation:
        raise SystemExit("MLX port validation failed: no token generated")
    print(f"MLX port validation passed: token_id={validation[0]}")

    messages: list[dict[str, str]] = []
    while True:
        try:
            user_text = input("you> ").strip()
        except EOFError:
            print()
            break
        if not user_text:
            break

        messages.append({"role": "user", "content": user_text})
        token_ids = list(generate_from_messages(model, tokenizer, messages, args.max_new_tokens, args.temperature))
        reply = decode_reply(tokenizer, token_ids)
        print(f"zaya> {reply}")
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
