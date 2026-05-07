# ZAYA1-8B on Apple Silicon

Small local experiments for `Zyphra/ZAYA1-8B`.

## Setup

```bash
uv sync
```

The scripts use the normal Hugging Face cache. If the snapshot is already present, it is reused; otherwise Hugging Face downloads it.

## Scripts

Load the model with the local MLX port and print profiling by default:

```bash
uv run python scripts/load_zaya_mlx.py
```

Load the model through default Hugging Face Transformers/PyTorch, CPU by default:

```bash
uv run python scripts/load_zaya_hf.py
```

Run the Python sum-code prompt through MLX with a usable token budget and profiling by default:

```bash
uv run python scripts/run_python_sum_mlx.py
```

Serve the MLX port through a minimal OpenAI-compatible API:

```bash
uv run python scripts/server_zaya_mlx.py
```

This repo includes a project-local pi extension in `.pi/extensions/zaya-mlx.ts` that registers `zaya-mlx/zaya-mlx` against `http://127.0.0.1:8123/v1`. Start the server, reload pi, then select that model.

Optional detailed artifacts:

```bash
uv run python scripts/run_python_sum_mlx.py --profile-json profile.json
uv run python scripts/run_python_sum_mlx.py --profile-layers
```

## Notes

`scripts/run_zaya_mlx.py` contains the experimental local MLX implementation used by the MLX scripts. It does not implement the upstream KV/CCA generation cache yet, and its MoE path still evaluates each expert over the whole batch before masking routed outputs. Those are the main targets for improving `scripts/run_python_sum_mlx.py` timings.
