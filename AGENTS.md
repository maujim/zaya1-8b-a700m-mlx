# AGENTS.md

## Project focus

This repo is for improving the local MLX port of `Zyphra/ZAYA1-8B`, especially end-to-end latency for the Python sum-code prompt.

## Commands

Use `uv run ...` for Python commands.

```bash
uv run python scripts/load_zaya_mlx.py
uv run python scripts/load_zaya_hf.py
uv run python scripts/run_python_sum_mlx.py
```

For a detailed generation trace:

```bash
uv run python scripts/run_python_sum_mlx.py --profile-json profile.json
uv run python scripts/run_python_sum_mlx.py --profile-layers
```

## Notes

- The scripts should assume the Hugging Face snapshot is usually cached, but allow `snapshot_download()` to fetch it if missing.
- Avoid adding `--local-files-only` flags to the main scripts.
- `scripts/run_zaya_mlx.py` is the MLX implementation/library used by the simpler scripts.
- Current likely performance targets: generation cache support and avoiding full-batch evaluation of every MoE expert.
