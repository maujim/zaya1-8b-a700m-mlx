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
./scripts/run_python_sum_mlx.sh --quant q8 --q8-min-weight-size 0
```

## Notes

- The scripts should assume the Hugging Face snapshot is usually cached, but allow `snapshot_download()` to fetch it if missing.
- `scripts/run_zaya_mlx.py` is the MLX implementation/library used by the simpler scripts.
- `.pi/extensions/zaya-mlx.ts` registers the local OpenAI-compatible server as `zaya-mlx/zaya-mlx` for pi.
- MLX scripts support `--quant full` and `--quant q8`; Q8 is dynamic in-memory quantization after loading and must not write a second model copy to disk.
- Q8 defaults to quantizing only large Linear weights (`--q8-min-weight-size 1000000`) for faster startup; use `--q8-min-weight-size 0` for exhaustive old behavior.
- Fast generation paths are default-on in this dev repo: KV/CCA cache and MoE single-token fast path. Use `--no-cache` and `--no-moe-decode-fast-path` only for comparisons/debugging.
- Current likely performance targets: prove cache correctness/token parity, improve profiling, and reduce any remaining Q8/cached decode overhead.
