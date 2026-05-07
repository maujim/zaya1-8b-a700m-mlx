#!/usr/bin/env python3
"""Small, memory-conscious token parity harness for ZAYA MLX generation modes.

Default behavior is intentionally conservative for 24GB machines: one full-precision
model load, one short prompt, and a small generation length. Use --include-q8 or
--prompts-file when you want broader coverage.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from run_zaya_mlx import MODEL_ID, generate, load_model, set_moe_decode_fast_path


DEFAULT_PROMPT = "Write a Python function that returns the sum of a list."
DEFAULT_PROMPTS_FILE = Path(__file__).with_name("parity_prompts.txt")


@dataclass(frozen=True)
class Mode:
    quant: str
    use_cache: bool
    moe_fast: bool

    @property
    def name(self) -> str:
        cache = "cache_on" if self.use_cache else "cache_off"
        moe = "moe_fast_on" if self.moe_fast else "moe_fast_off"
        return f"{self.quant},{cache},{moe}"


def first_mismatch(expected: list[int], actual: list[int]) -> int | None:
    for i, (left, right) in enumerate(zip(expected, actual)):
        if left != right:
            return i
    if len(expected) != len(actual):
        return min(len(expected), len(actual))
    return None


def mismatch_values(expected: list[int], actual: list[int], index: int | None) -> tuple[int | None, int | None]:
    if index is None:
        return None, None
    expected_id = expected[index] if index < len(expected) else None
    actual_id = actual[index] if index < len(actual) else None
    return expected_id, actual_id


def decode_id(tokenizer, token_id: int | None) -> str | None:
    if token_id is None:
        return None
    return tokenizer.decode([token_id], skip_special_tokens=False)


def self_test() -> None:
    assert first_mismatch([1, 2, 3], [1, 2, 3]) is None
    assert first_mismatch([1, 2, 3], [1, 9, 3]) == 1
    assert first_mismatch([1, 2, 3], [1, 2]) == 2
    assert first_mismatch([1, 2], [1, 2, 3]) == 2
    assert mismatch_values([1, 2, 3], [1, 9, 3], 1) == (2, 9)
    assert mismatch_values([1, 2], [1, 2, 3], 2) == (None, 3)
    assert mismatch_values([1, 2, 3], [1, 2], 2) == (3, None)


def prompt_token_count(tokenizer, prompt: str) -> int:
    # Match generate()'s chat wrapping path.
    from run_zaya_mlx import render_messages

    text = render_messages(tokenizer, [{"role": "user", "content": prompt}])
    return int(tokenizer(text, return_tensors="np")["input_ids"].shape[1])


def run_mode(model, tokenizer, prompt: str, max_new_tokens: int, mode: Mode, debug_cache: bool = False) -> list[int]:
    set_moe_decode_fast_path(model, mode.moe_fast)
    return list(generate(model, tokenizer, prompt, max_new_tokens, temperature=0.0, use_cache=mode.use_cache, debug_cache=debug_cache))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare generated token IDs across ZAYA MLX generation modes.")
    parser.add_argument("--model-path", type=Path, help="Use an existing Hugging Face snapshot directory.")
    parser.add_argument("--max-new-tokens", type=int, default=8, help="Keep this small on 24GB machines (default: 8).")
    parser.add_argument("--prompt", action="append", help="Prompt to test. Can be passed multiple times.")
    parser.add_argument("--prompts-file", type=Path, help="JSON list or newline-delimited prompts.")
    parser.add_argument("--default-prompts-file", action="store_true", help=f"Use bundled prompts from {DEFAULT_PROMPTS_FILE}.")
    parser.add_argument("--include-q8", action="store_true", help="Also load and compare Q8 modes after full modes. Increases runtime/startup work.")
    parser.add_argument("--q8-min-weight-size", type=int, default=1_000_000)
    parser.add_argument("--json", type=Path, help="Write machine-readable results.")
    parser.add_argument("--debug-cache", action="store_true", help="Assert cache shape invariants for cached modes.")
    parser.add_argument(
        "--mode",
        action="append",
        choices=["cache-fast", "nocache-fast", "cache-slow", "nocache-slow"],
        help="Limit compared modes. Can be repeated. Default compares all four full/q8 modes.",
    )
    parser.add_argument("--self-test", action="store_true", help="Run lightweight helper tests and exit without loading the model.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned prompts/modes and exit before loading the model.")
    parser.add_argument("--stop-on-first-fail", action="store_true", help="Exit after the first mismatch instead of completing the planned matrix.")
    args = parser.parse_args()
    if args.prompts_file and args.default_prompts_file:
        parser.error("--prompts-file and --default-prompts-file are mutually exclusive")
    return args


def load_prompts(args: argparse.Namespace) -> list[str]:
    prompts = list(args.prompt or [])
    prompts_file = DEFAULT_PROMPTS_FILE if args.default_prompts_file else args.prompts_file
    if prompts_file:
        text = prompts_file.read_text().strip()
        if text.startswith("["):
            prompts.extend(json.loads(text))
        else:
            prompts.extend(line for line in text.splitlines() if line.strip())
    return prompts or [DEFAULT_PROMPT]


def write_results_json(
    path: Path,
    *,
    model_path: Path,
    max_new_tokens: int,
    include_q8: bool,
    requested_modes: list[str] | None,
    planned_modes: dict[str, list[str]],
    stopped_early: bool,
    results: list[dict],
) -> None:
    payload = {
        "model_path": str(model_path),
        "max_new_tokens": max_new_tokens,
        "include_q8": include_q8,
        "requested_modes": requested_modes,
        "planned_modes": planned_modes,
        "stopped_early": stopped_early,
        "results": results,
    }
    path.write_text(json.dumps(payload, indent=2))


def selected_modes(quant: str, requested: list[str] | None) -> list[Mode]:
    modes_by_name = {
        "cache-fast": Mode(quant, use_cache=True, moe_fast=True),
        "nocache-fast": Mode(quant, use_cache=False, moe_fast=True),
        "cache-slow": Mode(quant, use_cache=True, moe_fast=False),
        "nocache-slow": Mode(quant, use_cache=False, moe_fast=False),
    }
    names = requested or list(modes_by_name)
    modes = [modes_by_name[name] for name in names]
    baseline = modes_by_name["nocache-slow"]
    if baseline not in modes:
        modes.append(baseline)
    return modes


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        print("self-test passed")
        return
    prompts = load_prompts(args)
    quant_modes = ["full"] + (["q8"] if args.include_q8 else [])
    planned_modes = {quant: [mode.name for mode in selected_modes(quant, args.mode)] for quant in quant_modes}
    if args.dry_run:
        print(json.dumps({"max_new_tokens": args.max_new_tokens, "prompts": prompts, "planned_modes": planned_modes, "resource_note": "Q8 loads a second quantized model group; keep --include-q8 off for safest 24GB runs."}, indent=2))
        return

    model_path = args.model_path or Path(snapshot_download(MODEL_ID))
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    all_results: list[dict] = []
    failed = False

    for quant in quant_modes:
        print(f"\nLoading quant={quant} (model is loaded once for this quant group)...")
        model = load_model(model_path, quant=quant, q8_min_weight_size=args.q8_min_weight_size)
        modes = selected_modes(quant, args.mode)

        for prompt_index, prompt in enumerate(prompts):
            baseline_mode = modes[-1]  # no cache, no MoE shortcut: slow but simplest reference.
            baseline_ids = run_mode(model, tokenizer, prompt, args.max_new_tokens, baseline_mode, debug_cache=args.debug_cache)
            prompt_tokens = prompt_token_count(tokenizer, prompt)
            print(f"\nprompt[{prompt_index}] prompt_tokens={prompt_tokens} text={prompt!r}")
            print(f"baseline={baseline_mode.name}")
            print("mode | baseline | prompt_tokens | generated_ids | first_mismatch | expected_id | actual_id | expected_text | actual_text | pass/fail")

            for mode in modes:
                ids = baseline_ids if mode == baseline_mode else run_mode(model, tokenizer, prompt, args.max_new_tokens, mode, debug_cache=args.debug_cache)
                mismatch = first_mismatch(baseline_ids, ids)
                passed = mismatch is None
                expected_id, actual_id = mismatch_values(baseline_ids, ids, mismatch)
                expected_text = decode_id(tokenizer, expected_id)
                actual_text = decode_id(tokenizer, actual_id)
                failed = failed or not passed
                result = {
                    "prompt_index": prompt_index,
                    "prompt": prompt,
                    "mode": mode.name,
                    "baseline_mode": baseline_mode.name,
                    "prompt_tokens": prompt_tokens,
                    "generated_ids": ids,
                    "first_mismatch": mismatch,
                    "expected_id": expected_id,
                    "actual_id": actual_id,
                    "expected_text": expected_text,
                    "actual_text": actual_text,
                    "pass": passed,
                }
                all_results.append(result)
                mismatch_text = "-" if mismatch is None else str(mismatch)
                status = "PASS" if passed else "FAIL"
                expected_id_text = "-" if expected_id is None else str(expected_id)
                actual_id_text = "-" if actual_id is None else str(actual_id)
                expected_piece = "-" if expected_text is None else repr(expected_text)
                actual_piece = "-" if actual_text is None else repr(actual_text)
                print(f"{mode.name} | {baseline_mode.name} | {prompt_tokens} | {ids} | {mismatch_text} | {expected_id_text} | {actual_id_text} | {expected_piece} | {actual_piece} | {status}")
                if not passed and args.stop_on_first_fail:
                    if args.json:
                        write_results_json(
                            args.json,
                            model_path=model_path,
                            max_new_tokens=args.max_new_tokens,
                            include_q8=args.include_q8,
                            requested_modes=args.mode,
                            planned_modes=planned_modes,
                            stopped_early=True,
                            results=all_results,
                        )
                        print(f"\nwrote {args.json}")
                    raise SystemExit(1)

        del model

    if args.json:
        write_results_json(
            args.json,
            model_path=model_path,
            max_new_tokens=args.max_new_tokens,
            include_q8=args.include_q8,
            requested_modes=args.mode,
            planned_modes=planned_modes,
            stopped_early=False,
            results=all_results,
        )
        print(f"\nwrote {args.json}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
