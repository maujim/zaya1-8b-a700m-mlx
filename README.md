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

Try in-memory Q8 quantization, no extra model copy on disk:

```bash
./scripts/run_python_sum_mlx.sh --quant q8
```

Run the OpenAI-compatible MLX server:

```bash
uv run python scripts/server_zaya_mlx.py --port 8123
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

## Useful profiling knobs

```bash
./scripts/run_python_sum_mlx.sh --profile-json profile.json
./scripts/run_python_sum_mlx.sh --profile-layers
uv run python scripts/server_zaya_mlx.py --quant q8
```

## What is still bad

`scripts/run_zaya_mlx.py` is the experimental MLX implementation and CLI. `scripts/run_python_sum_mlx.sh` is just a tiny wrapper around it with the Python sum prompt. It is slow because generation has no KV/CCA cache yet, and the MoE path still evaluates every expert over the whole batch before masking routed outputs.
