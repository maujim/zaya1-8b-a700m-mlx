#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/benchmark_speed.sh [OPTIONS]

Conservative speed benchmark for local ZAYA MLX inference. Writes a profile JSON
with prefill/decode stats suitable for before/after speed loops.

Defaults are intentionally modest for 24GB RAM:
  --max-new-tokens: 32 (override with MAX_NEW_TOKENS or CLI)
  --quant: full unless forwarded with --quant q8
  cache and MoE decode fast path: runner defaults (on)

Examples:
  scripts/benchmark_speed.sh
  scripts/benchmark_speed.sh --quant q8
  scripts/benchmark_speed.sh --max-new-tokens 16 --profile-json profiles/test.json

Environment:
  MAX_NEW_TOKENS      Default token count if --max-new-tokens is not passed
  PROFILE_JSON        Output profile path if --profile-json is not passed
  TEMPERATURE         Default temperature if --temperature is not passed
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

mkdir -p profiles
PROMPT='Write a tiny Python function named sum_numbers that takes two numbers and returns their sum. Include only the code block.'
DEFAULT_PROFILE_JSON="profiles/speed-$(date +%Y%m%d-%H%M%S).json"

has_arg() {
  local needle="$1"
  shift
  for arg in "$@"; do
    if [[ "$arg" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

args=(
  --profile
  --max-new-tokens "${MAX_NEW_TOKENS:-32}"
  --temperature "${TEMPERATURE:-0}"
  --profile-json "${PROFILE_JSON:-$DEFAULT_PROFILE_JSON}"
  "$PROMPT"
)

if has_arg --max-new-tokens "$@"; then
  args=(--profile --temperature "${TEMPERATURE:-0}" --profile-json "${PROFILE_JSON:-$DEFAULT_PROFILE_JSON}" "$PROMPT")
fi
if has_arg --temperature "$@"; then
  filtered=()
  skip_next=0
  for arg in "${args[@]}"; do
    if (( skip_next )); then
      skip_next=0
      continue
    fi
    if [[ "$arg" == "--temperature" ]]; then
      skip_next=1
      continue
    fi
    filtered+=("$arg")
  done
  args=("${filtered[@]}")
fi
if has_arg --profile-json "$@"; then
  filtered=()
  skip_next=0
  for arg in "${args[@]}"; do
    if (( skip_next )); then
      skip_next=0
      continue
    fi
    if [[ "$arg" == "--profile-json" ]]; then
      skip_next=1
      continue
    fi
    filtered+=("$arg")
  done
  args=("${filtered[@]}")
fi

uv run python scripts/run_zaya_mlx.py "${args[@]}" "$@"
