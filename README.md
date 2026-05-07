# ZAYA1-8B MLX port

Experimental Apple Silicon MLX port for `Zyphra/ZAYA1-8B`. It is useful for local testing and profiling, but it is not production-ready.

## Requirements

- macOS on Apple Silicon
- Python 3.12+
- `uv`
- Enough disk/RAM for the HF snapshot (~18GB weights)
- Hugging Face access/network for the first download; after that it uses the normal HF cache

## Setup

```bash
uv sync
```

## Run stuff

Load with the local MLX port, profiled by default:

```bash
uv run python scripts/load_zaya_mlx.py
```

Load with Hugging Face/Transformers/PyTorch, CPU by default:

```bash
uv run python scripts/load_zaya_hf.py
```

Ask the MLX model to write a tiny Python sum function, profiled by default. The wrapper defaults to `--quant q4` for local speed; pass `--quant full` if you need full BF16.

```bash
./scripts/run_python_sum_mlx.sh
```

Try in-memory quantization, no extra model copy on disk. Quantized modes skip the pre-quant full-parameter sync, then quantize only large Linear weights by default for faster startup; use `--q8-min-weight-size 0` to reproduce exhaustive quantization. For speed experiments, try q8 first, then q6/q4 if memory bandwidth looks like the bottleneck.

```bash
./scripts/run_python_sum_mlx.sh --quant q8
./scripts/run_python_sum_mlx.sh --quant q6
./scripts/run_python_sum_mlx.sh --quant q4
./scripts/run_python_sum_mlx.sh --quant q8 --q8-min-weight-size 0  # slower startup, exhaustive
```

Run the OpenAI-compatible MLX server. It defaults to `--quant q4` plus fast generation paths (`--cache` and `--moe-decode-fast-path`) for local speed; pass `--quant full`, `--no-cache`, or `--no-moe-decode-fast-path` to compare old behavior. Streaming responses use SSE when the request includes `"stream": true`. Generation is serialized with an in-process lock; this fiddle server is intentionally one request at a time.

```bash
uv run python scripts/server_zaya_mlx.py --port 8123
uv run python scripts/server_zaya_mlx.py --quant q8 --port 8123
curl http://127.0.0.1:8123/healthz
```

## pi integration

This repo has a local pi extension at:

```text
.pi/extensions/zaya-mlx.ts
```

Start the server, run `/reload` in pi, then select:

```text
zaya-mlx/zaya-mlx
```

## Token parity checks

Before adding more speedups, run the conservative parity harness. By default it loads one full-precision model, uses one short prompt, generates 8 deterministic tokens, and compares token IDs for cache on/off and MoE decode fast path on/off. Q8 is opt-in to avoid extra memory/startup pressure on 24GB machines.

```bash
uv run python scripts/compare_generation_modes.py
uv run python scripts/compare_generation_modes.py --max-new-tokens 4 --json parity.json
uv run python scripts/compare_generation_modes.py --debug-cache
uv run python scripts/compare_generation_modes.py --mode cache-fast --max-new-tokens 2 --stop-on-first-fail  # tiny subset, baseline auto-added
uv run python scripts/compare_generation_modes.py --dry-run --default-prompts-file  # show matrix, no model load
uv run python scripts/compare_generation_modes.py --include-q8  # slower; loads a second quantized model group
```

The table is:

```text
mode | baseline | prompt_tokens | generated_ids | first_mismatch | expected_id | actual_id | expected_text | actual_text | pass/fail
```

The script exits nonzero on any mismatch.

## Speed benchmarking

Use the conservative benchmark wrapper for before/after speed loops. It writes profile JSON with prefill/decode stats and defaults to q4 plus 32 generated tokens so it is less aggressive on 24GB machines than the full Python-sum wrapper.

```bash
scripts/benchmark_speed.sh
scripts/benchmark_speed.sh --quant q8
scripts/benchmark_speed.sh --quant q4
uv run python scripts/summarize_profile.py profiles/speed-YYYYmmdd-HHMMSS.json
```

## Useful profiling and comparison knobs

```bash
./scripts/run_python_sum_mlx.sh --profile-json profile.json
./scripts/run_python_sum_mlx.sh --profile-layers
./scripts/run_python_sum_mlx.sh --no-cache
./scripts/run_python_sum_mlx.sh --no-moe-decode-fast-path
./scripts/run_python_sum_mlx.sh --quant q8 --q8-min-weight-size 0
./scripts/run_python_sum_mlx.sh --quant q4
uv run python scripts/load_zaya_mlx.py --quant q8 --profile-json q8-load-profile.json
uv run python scripts/server_zaya_mlx.py --quant q8
```

## Current perf state

`scripts/run_zaya_mlx.py` is the experimental MLX implementation and CLI. `scripts/run_python_sum_mlx.sh` is a wrapper around it with the Python sum prompt. The repo currently defaults to cached prefill/decode, MoE single-token expert selection, RoPE/mask reuse, and large-linear-only Q8 quantization. Profiling output separates cached prefill from steady-state decode and reports decode p50/p90/p99 plus tokens/sec. This is a dev/test repo, so the fastest paths are default-on and comparison flags remain available.
