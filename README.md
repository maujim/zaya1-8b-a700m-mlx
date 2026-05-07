# ZAYA1-8B on Apple Silicon

This repo sets up a local runner for `Zyphra/ZAYA1-8B` on an M2 Mac. It uses the Hugging Face cache; the 18 GB weight files should stay outside the repo.

## Setup

```bash
uv sync
```

## Finish or Check the Download

Resume the model download:

```bash
uv run python scripts/check_model.py --download
```

Check an already downloaded snapshot without network:

```bash
uv run python scripts/check_model.py
```

## Run

Minimal terminal chat:

```bash
uv run python scripts/chat_zaya_mlx.py --local-files-only
```

The chat loads the experimental MLX port, then accepts messages at the `you>` prompt. Press Return on an empty prompt or
Ctrl-D to exit. To force a one-token startup generation check, add `--validate-startup`; this can be slow or unstable on
memory-constrained Macs.

Inspect the local MLX module/key/shape mapping without loading the 8B weights:

```bash
uv run python scripts/inspect_zaya_port.py --local-files-only
```

Run a tiny synthetic MLX model through the same code paths without loading ZAYA weights:

```bash
uv run python scripts/smoke_tiny_zaya_mlx.py
```

MLX runner:

```bash
uv run python scripts/run_zaya_mlx.py \
  --local-files-only \
  --max-new-tokens 1 \
  --show-token-ids \
  "Say hello."
```

Add `--profile` to print load/generation timings and MLX memory stats, or `--profile-layers` for slower per-layer timings. Use `--profile-json profile.json` to save the full event log.

To test only MLX model loading/synchronization with profiling enabled by default:

```bash
uv run python scripts/profile_load_zaya_mlx.py --local-files-only
```

The snapshot is resolved from the normal Hugging Face cache, e.g.
`~/.cache/huggingface/hub/models--Zyphra--ZAYA1-8B/snapshots/...`. You can also pass that directory explicitly with
`--model-path`.

PyTorch/Transformers fallback:

```bash
uv run python scripts/run_zaya.py \
  --local-files-only \
  --max-new-tokens 64 \
  "Write a tiny Python function that adds two numbers."
```

The model card's official serving path is Zyphra's custom `vllm` fork plus Zyphra's `transformers` fork. Native vLLM serving is not a practical path on an M2 Air because vLLM is designed around CUDA/Linux GPUs. This project uses the Zyphra `transformers` fork directly, which is the path most likely to work locally on macOS CPU/MPS.

`mlx-lm` does not currently ship a `zaya` architecture backend, so `scripts/run_zaya_mlx.py` contains a local MLX implementation for this checkpoint. The port now matches the checkpoint's tensor names and shapes, but it is still experimental: it does not implement the upstream KV/CCA generation cache and its MoE path still evaluates each expert over the whole batch before masking routed outputs, so long prompts/generation can be slow on an M2 Air.
