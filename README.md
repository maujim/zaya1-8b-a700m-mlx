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

Ask the MLX model to write a tiny Python sum function, profiled by default:

```bash
./scripts/run_python_sum_mlx.sh
```

Try in-memory Q8 quantization, no extra model copy on disk. By default Q8 only quantizes large Linear weights for faster startup; use `--q8-min-weight-size 0` to reproduce exhaustive quantization.

```bash
./scripts/run_python_sum_mlx.sh --quant q8
./scripts/run_python_sum_mlx.sh --quant q8 --q8-min-weight-size 0  # slower startup, exhaustive
```

Run the OpenAI-compatible MLX server. Fast generation paths are on by default (`--cache` and `--moe-decode-fast-path`); pass `--no-cache` or `--no-moe-decode-fast-path` to compare old behavior. Streaming responses use SSE when the request includes `"stream": true`.

```bash
uv run python scripts/server_zaya_mlx.py --port 8123
uv run python scripts/server_zaya_mlx.py --quant q8 --port 8123
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

## Useful profiling and comparison knobs

```bash
./scripts/run_python_sum_mlx.sh --profile-json profile.json
./scripts/run_python_sum_mlx.sh --profile-layers
./scripts/run_python_sum_mlx.sh --no-cache
./scripts/run_python_sum_mlx.sh --no-moe-decode-fast-path
./scripts/run_python_sum_mlx.sh --quant q8 --q8-min-weight-size 0
uv run python scripts/load_zaya_mlx.py --quant q8 --profile-json q8-load-profile.json
uv run python scripts/server_zaya_mlx.py --quant q8
```

## Current perf state

`scripts/run_zaya_mlx.py` is the experimental MLX implementation and CLI. `scripts/run_python_sum_mlx.sh` is a wrapper around it with the Python sum prompt. The repo currently defaults to cached prefill/decode, MoE single-token expert selection, RoPE/mask reuse, and large-linear-only Q8 quantization. This is a dev/test repo, so the fastest paths are default-on and comparison flags remain available.
