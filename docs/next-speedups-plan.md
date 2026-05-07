# ZAYA MLX next speedups and improvements

This repo is now optimized for fast local iteration. Fast paths are default-on; keep safety toggles for comparisons.

## Current default fast path

- `--cache` is on by default; use `--no-cache` to compare full-context recompute.
- `--moe-decode-fast-path` is on by default; use `--no-moe-decode-fast-path` to compare full expert evaluation.
- Server streaming uses SSE when requests pass `"stream": true`.
- RoPE and causal masks are cached per model instance.

## Immediate priority: prove cache correctness

Before adding more speedups, lock down token parity.

### 1. Deterministic parity harness

Add a small script, e.g. `scripts/compare_generation_modes.py`, that runs the same prompt across:

- cached + MoE fast path
- uncached + MoE fast path
- cached + no MoE fast path
- uncached + no MoE fast path
- optionally `--quant q8`

Compare generated token IDs, not decoded text.

Acceptance:

- Prints a compact PASS/FAIL table.
- Fails nonzero on token mismatch.
- Includes prompt tokens, generated token IDs, and first mismatch index.

### 2. Cache shape/debug assertions

Add optional debug checks behind `--debug-cache` or an env var:

- Attention K/V cache sequence length grows as expected.
- CCA conv state has shape `[B, total_padding, C]`.
- CCA `prev_hs` has shape `[B, H]`.
- `cache.seen_tokens` matches tokens already in cache before each decode step.

Acceptance:

- No overhead unless enabled.
- Helpful assertion messages with layer number and expected/actual shapes.

## Performance work

### 3. Profile prefill vs decode clearly

Current profiling has events, but make it easier to compare runs.

Improve `Profiler.print_report()` to group:

- model load
- tokenization/render
- prefill
- decode tokens
- sampling
- token decode/printing if measured

Acceptance:

- Shows p50/p90/avg for `decode_token`.
- Shows first-token/prefill separately.
- JSON keeps enough metadata to plot later.

### 4. Reduce cache append overhead

Current cached attention likely concatenates K/V every token. That may become expensive.

Options:

- Preallocate max-length K/V arrays if MLX update semantics are reasonable.
- Chunked cache pages: append within fixed-size blocks, concatenate only block views for attention.
- Keep current concat path until measurements prove this matters.

Acceptance:

- Decode latency does not grow significantly over 100+ generated tokens.
- Token parity remains intact.

### 5. Avoid `.item()` sync in MoE fast path if it hurts

The MoE fast path uses `choices.item()` to pick the expert in Python. Measure whether sync overhead is smaller than expert savings.

Alternatives:

- Keep default if it wins.
- Disable by default for specific quant/model modes if it loses.
- Explore a small MLX-side expert selection if possible.

Acceptance:

- Document avg decode token time with and without MoE fast path.

### 6. Server concurrency and streaming polish

The server currently shares one model instance. Improve production-ish behavior while staying simple:

- Add request-level profiling logs with prompt tokens, completion tokens, total ms, tokens/sec.
- Ensure streaming flushes small chunks promptly.
- Consider a simple generation lock if concurrent requests corrupt cache/model state.
- Add `/healthz` endpoint.

Acceptance:

- OpenAI-compatible streaming remains working.
- Concurrent request behavior is explicit: either serialized safely or documented unsupported.

## Quality and cleanup

### 7. Minimal smoke tests

Add lightweight tests that do not require downloading the full model:

- CLI help works.
- `ZayaGenerationCache` initializes with synthetic args.
- `render_messages()` handles simple chat messages.
- SSE chunk formatting matches OpenAI shape.

Acceptance:

- `uv run pytest` or equivalent runs quickly without model weights.

### 8. Documentation update

Update README with the fastest recommended commands:

```bash
./scripts/run_python_sum_mlx.sh --quant q8
uv run python scripts/server_zaya_mlx.py --quant q8 --port 8123
```

Mention comparison toggles:

```bash
--no-cache
--no-moe-decode-fast-path
```

## Suggested order

1. Deterministic parity harness.
2. Cache/debug assertions.
3. Better profiling output.
4. Measure defaults on `full` and `q8`.
5. Fix any cache correctness bugs.
6. Optimize cache append growth if profiles show it.
7. Server request logging/concurrency guard.
8. README refresh.
