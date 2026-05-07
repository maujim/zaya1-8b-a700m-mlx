# AGENTS.md

## Project focus

This repo is for improving the local MLX port of `Zyphra/ZAYA1-8B`, especially end-to-end latency for the Python sum-code prompt.

## Commands

Use `uv run ...` for Python commands.

```bash
uv run python scripts/load_zaya_mlx.py
uv run python scripts/load_zaya_hf.py
./scripts/run_python_sum_mlx.sh
uv run python scripts/server_zaya_mlx.py --port 8123
```

For a detailed generation trace:

```bash
./scripts/run_python_sum_mlx.sh --profile-json profile.json
./scripts/run_python_sum_mlx.sh --profile-layers
./scripts/run_python_sum_mlx.sh --quant q8
```

## Notes

- The scripts should assume the Hugging Face snapshot is usually cached, but allow `snapshot_download()` to fetch it if missing.
- Avoid adding `--local-files-only` flags to the main scripts.
- `scripts/run_zaya_mlx.py` is the MLX implementation/library used by the simpler scripts.
- `.pi/extensions/zaya-mlx.ts` registers the local OpenAI-compatible server as `zaya-mlx/zaya-mlx` for pi.
- MLX scripts support `--quant full` and `--quant q8`; Q8 is dynamic in-memory quantization after loading and must not write a second model copy to disk.
- Current likely performance targets: generation cache support and avoiding full-batch evaluation of every MoE expert.
