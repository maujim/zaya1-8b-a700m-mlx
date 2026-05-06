# ZAYA1-8B on Apple Silicon

This repo sets up a local runner for `Zyphra/ZAYA1-8B` on an M2 Mac. It uses the Hugging Face cache; the 18 GB weight files should stay outside the repo.

## Setup

```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync
```

## Finish or Check the Download

Resume the model download:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/check_model.py --download
```

Check an already downloaded snapshot without network:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/check_model.py
```

## Run

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_zaya.py \
  --local-files-only \
  --max-new-tokens 64 \
  "Write a tiny Python function that adds two numbers."
```

The model card's official serving path is Zyphra's custom `vllm` fork plus Zyphra's `transformers` fork. Native vLLM serving is not a practical path on an M2 Air because vLLM is designed around CUDA/Linux GPUs. This project uses the Zyphra `transformers` fork directly, which is the path most likely to work locally on macOS CPU/MPS.
