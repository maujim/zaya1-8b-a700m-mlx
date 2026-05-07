# AGENTS.md

## Project notes

- This repo contains experimental runners for `Zyphra/ZAYA1-8B` on Apple Silicon.
- Prefer `uv run ...` for Python commands so the project environment is used.
- Large model snapshots live in the Hugging Face cache and should not be committed.

## Useful commands

```bash
uv sync
uv run python scripts/smoke_tiny_zaya_mlx.py
uv run python scripts/inspect_zaya_port.py --local-files-only
uv run python scripts/run_zaya_mlx.py --local-files-only --max-new-tokens 1 --show-token-ids "Say hello."
uv run python scripts/profile_load_zaya_mlx.py --local-files-only
```

## Profiling

`scripts/profile_load_zaya_mlx.py` profiles MLX model loading/synchronization by default and does not generate tokens.

`scripts/run_zaya_mlx.py` supports lightweight generation profiling:

```bash
uv run python scripts/run_zaya_mlx.py --local-files-only --profile --max-new-tokens 1 "Say hello."
```

For slower but more detailed per-layer timings:

```bash
uv run python scripts/run_zaya_mlx.py --local-files-only --profile-layers --profile-json profile.json --max-new-tokens 1 "Say hello."
```

`--profile-layers` forces MLX synchronization around each layer, so use it for diagnosis rather than normal generation benchmarks.
