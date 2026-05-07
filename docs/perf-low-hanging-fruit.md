# ZAYA MLX performance low-hanging fruit

These are small-ish specs intended to be handed to separate agents/branches. Do the work in isolated branches/worktrees, then merge back one at a time. Prefer tiny, reviewable commits. Keep correctness checks deterministic with `temperature=0`.

## Shared rules

- Main implementation file is `scripts/run_zaya_mlx.py`.
- Keep the public scripts working:
  - `uv run python scripts/load_zaya_mlx.py`
  - `uv run python scripts/run_zaya_mlx.py --max-new-tokens 1 "Say hello."`
  - `./scripts/run_python_sum_mlx.sh`
  - `uv run python scripts/server_zaya_mlx.py`
- Do not write a second model copy to disk.
- Keep `--quant full|q8` working.
- Add flags for risky changes first; defaults can change after validation.
- For full-model validation, use `temperature=0` and compare generated token IDs before/after where practical.

## 1. MoE single-token decode short-circuit

Suggested branch: `perf/moe-single-token`

### Problem

`Experts.__call__` currently evaluates every expert over the entire input and masks outputs afterwards. During generation decode, the common shape is `[B, 1, H]`. For batch size 1 and sequence length 1, only one routed expert can be selected, so evaluating all experts is wasteful.

### Goal

Add a fast path for single-token decode that evaluates only the selected expert when possible.

### Implementation sketch

- In `Experts.__call__(hidden_states, choices, use_mod)`:
  - Detect `hidden_states.shape[0] == 1 and hidden_states.shape[1] == 1`.
  - If `use_mod` and the selected choice is the MOD/self route, return `hidden_states`.
  - Otherwise evaluate only `self.local_experts[selected]` on `hidden_states`.
- MLX scalar extraction may require synchronization. Keep this path optional if needed, e.g. `use_decode_fast_path` on model/config, until measured.
- Preserve the old full-batch path for prompt prefill and multi-token inputs.

### Validation

- Add or run a tiny synthetic check that compares old full-expert output vs fast path for `[1, 1, H]` with deterministic choices.
- Full model: compare first 5-10 token IDs at `temperature=0` before/after.
- Profile `generate_token` timing with and without the fast path.

### Risks

- Converting `choices.item()` may force sync and erase gains. Measure it.
- Dynamic Python branching may make MLX graph caching less effective. Measure it.

## 2. Rotary/mask reuse

Suggested branch: `perf/reuse-mask-rope`

### Problem

`ZayaModel.__call__` rebuilds causal masks and rotary embeddings every forward. During current uncached generation, context length grows by one each token, so this work is repeated constantly.

### Goal

Cache/reuse causal masks and rotary cos/sin arrays keyed by sequence length/device/dtype enough to avoid obvious recomputation overhead.

### Implementation sketch

- Add small caches on `ZayaModel`, e.g.:
  - `_mask_cache: dict[tuple[int, str], mx.array]`
  - `_rope_cache: dict[int, tuple[mx.array, mx.array]]`
- For each `seq_len`, reuse previously computed `mask`, `cos`, and `sin`.
- Keep memory bounded simply. This is a throwaway experiment, so a small dict is fine.
- Be careful that mask dtype currently uses `h.dtype`; include dtype or cast as needed.

### Validation

- `uv run python -m py_compile scripts/run_zaya_mlx.py`
- Full model: compare first 5-10 token IDs at `temperature=0` before/after.
- Profile whether `generate_token` changes meaningfully.

### Risks

- Caching many sequence lengths may hold extra memory, though masks/rope are tiny compared to weights.
- dtype mismatches can cause subtle issues.

## 3. KV/CCA cache skeleton

Suggested branch: `perf/cache-skeleton`

### Problem

Generation currently feeds the entire growing token sequence into the model on every new token. This recomputes all layers for all previous tokens. Upstream ZAYA has a `ZayaDynamicCache` that stores attention KV plus CCA conv state and previous hidden state.

### Goal

Introduce an MLX-side cache object and thread it through the model without changing default generation yet. The first merge should be mostly structural and easy to review.

### Implementation sketch

- Add a `ZayaGenerationCache` dataclass with:
  - `key_states: list[mx.array | None]`
  - `value_states: list[mx.array | None]`
  - `conv_states: list[mx.array | None]`
  - `prev_hs: list[mx.array | None]`
  - `seen_tokens: int`
  - `has_previous_state: bool`
- Thread optional `cache` through:
  - `ZayaForCausalLM.__call__`
  - `ZayaModel.__call__`
  - `AttentionLayer.__call__`
  - `Attention.__call__`
  - `CCA.__call__`
- Initially, the cache can be unused or populated only in obvious places. Keep behavior identical when `cache is None`.
- Add comments mapping fields to upstream `ZayaDynamicCache` in Zyphra Transformers.

### Validation

- Existing scripts compile.
- Existing generation without cache is unchanged.
- Optional tiny synthetic model can instantiate and pass a cache object without crashing.

### Risks

- Signature churn conflicts with other branches.
- Easy to accidentally change default behavior. Keep `cache=None` path identical.

## 4. Prefill once, decode one token at a time using cache

Suggested branch: `perf/prefill-decode-loop`

### Problem

Even after cache exists, `generate_from_messages` needs to use it: one full prompt prefill, then only feed the newest token each step.

### Goal

Add an optional cached generation mode, e.g. `--cache`, that:

1. Runs prompt prefill once with all prompt tokens and an empty `ZayaGenerationCache`.
2. Samples/argmaxes the first next token.
3. For subsequent tokens, feeds only the last generated token with the same cache.

### Implementation sketch

- Add `use_cache: bool = False` to `generate_from_messages` / `generate`.
- Add CLI flag `--cache` to `scripts/run_zaya_mlx.py` and server if useful.
- On prefill:
  - call `model(tokens, cache=cache)`
  - update cache state as layers run
  - set `cache.has_previous_state = True`
- On decode:
  - call `model(next_token, cache=cache)` where `next_token` shape is `[B, 1]`
  - attention should append new K/V to cached K/V and attend over all cached tokens.
  - CCA should use cached conv state and previous hidden state for the single-token path.
- Keep uncached generation as fallback/default until validated.

### Validation

- Compare cached vs uncached token IDs at `temperature=0` for at least first 10 generated tokens.
- Profile first token separately from subsequent decode tokens.
- Test with and without `--quant q8`.

### Risks

- Highest correctness risk.
- CCA cache details must match upstream behavior:
  - conv state shape corresponds to last two packed q/k timesteps.
  - `prev_hs` must feed `val_proj2` during single-token decode.
- RoPE positions must use absolute positions, not always zero, during decode.

## Recommended merge order

1. `perf/moe-single-token`
2. `perf/reuse-mask-rope`
3. `perf/cache-skeleton`
4. `perf/prefill-decode-loop`

The first two are safer and should not depend on the cache API. The last two are coupled and should be merged more carefully.
