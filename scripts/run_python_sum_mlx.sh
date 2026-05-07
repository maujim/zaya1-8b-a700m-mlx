#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run_python_sum_mlx.sh [OPTIONS]

Run the local MLX ZAYA model on the Python sum-code prompt.

This wrapper enables profiling and forwards OPTIONS to scripts/run_zaya_mlx.py.
Common options:
  --max-new-tokens N       Override generated token count (default: MAX_NEW_TOKENS or 500)
  --temperature FLOAT      Override sampling temperature (default: TEMPERATURE or 0)
  --quant {full,q8,q6,q4}  Weight mode
  --q8-min-weight-size N   Only quantize large Linear weights (default: 1000000; 0 = old exhaustive behavior)
  --profile-json PATH      Write profile events and summary as JSON
  --profile-layers         Profile each transformer layer
  --cache / --no-cache     Toggle KV/CCA cached generation (default: on)
  --moe-decode-fast-path / --no-moe-decode-fast-path
                            Toggle single-token MoE fast path (default: on)
  -h, --help               Show this help and the forwarded runner help

Environment:
  MAX_NEW_TOKENS           Default token count if --max-new-tokens is not passed
  TEMPERATURE              Default temperature if --temperature is not passed
EOF
  echo
  uv run python scripts/run_zaya_mlx.py --help
}

case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
esac

PROMPT='Write a tiny Python function named sum_numbers that takes two numbers and returns their sum. Include only the code block.'

uv run python scripts/run_zaya_mlx.py \
  --profile \
  --max-new-tokens "${MAX_NEW_TOKENS:-500}" \
  --temperature "${TEMPERATURE:-0}" \
  "$PROMPT" \
  "$@"
