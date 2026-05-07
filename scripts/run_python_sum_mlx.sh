#!/usr/bin/env bash
set -euo pipefail

PROMPT='Write a tiny Python function named sum_numbers that takes two numbers and returns their sum. Include only the code block.'

uv run python scripts/run_zaya_mlx.py \
  --profile \
  --max-new-tokens "${MAX_NEW_TOKENS:-500}" \
  --temperature "${TEMPERATURE:-0}" \
  "$PROMPT" \
  "$@"
