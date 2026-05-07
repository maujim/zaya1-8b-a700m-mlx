# ZAYA MLX performance low-hanging fruit

These are handoff specs for separate agents/branches. The goal is to improve generation latency for the local MLX port without making the codebase harder to reason about. Do these in isolated branches/worktrees and merge one at a time.

The current bottlenecks are:

1. ~~During generation we repeatedly run the whole growing context through the model.~~ Experimental `--cache` mode pre-fills once and decodes one token at a time.
2. ~~Attention layers do not cache K/V.~~ Experimental `--cache` mode stores repeated post-RoPE K/V per attention layer.
3. ~~ZAYA CCA layers do not cache their convolution state or previous hidden state.~~ Experimental `--cache` mode stores CCA conv windows and delayed hidden state.
4. ~~MoE MLP layers evaluate every expert and mask afterwards.~~ Implemented behind `--moe-decode-fast-path` for CLI and server.
5. ~~We rebuild small helper tensors like RoPE and causal masks every forward.~~ Implemented with per-model mask/RoPE caches.

## Current status

- Done: Task 1 — MoE single-token decode short-circuit (`--moe-decode-fast-path`).
- Done: Task 2 — Rotary and causal mask reuse.
- Done: Task 3 — KV/CCA cache skeleton (`ZayaGenerationCache` and threaded signatures).
- Done: Task 4 — opt-in prefill/decode cached generation loop using the cache skeleton (`--cache` for CLI and server).
- Next: validate token parity/performance more broadly, then decide whether any flags should become defaults.

## Shared repo context

Important files:

- `scripts/run_zaya_mlx.py` — local MLX model implementation, generation loop, profiling, quantization.
- `scripts/load_zaya_mlx.py` — load-only profiler.
- `scripts/server_zaya_mlx.py` — OpenAI-compatible local server using the MLX implementation.
- `scripts/run_python_sum_mlx.sh` — tiny wrapper around `run_zaya_mlx.py` with the Python sum prompt.
- `.pi/extensions/zaya-mlx.ts` — pi extension that registers `zaya-mlx/zaya-mlx` against the local server.

Upstream reference:

- Installed Zyphra Transformers implementation is usually at:
  - `.venv/lib/python*/site-packages/transformers/models/zaya/modeling_zaya.py`
- Especially useful upstream classes/functions:
  - `ZayaDynamicCache`
  - `CCA.forward`
  - `ZayaAttention.forward` / `ZayaSdpaAttention.forward`
  - `ZayaBlock` / expert routing code

## Shared rules for every task

- Keep the main public commands working:

  ```bash
  uv run python scripts/load_zaya_mlx.py
  uv run python scripts/run_zaya_mlx.py --max-new-tokens 1 "Say hello."
  ./scripts/run_python_sum_mlx.sh
  uv run python scripts/server_zaya_mlx.py --help
  ```

- Always run at least:

  ```bash
  uv run python -m py_compile scripts/load_zaya_hf.py scripts/load_zaya_mlx.py scripts/run_zaya_mlx.py scripts/server_zaya_mlx.py
  bash -n scripts/run_python_sum_mlx.sh
  ```

- Do not write any quantized or transformed model copy to disk.
- Keep `--quant full|q8` working.
- Do not remove profiling; add profiling events if useful.
- Risky changes should be opt-in first. Use flags like `--cache` or `--moe-decode-fast-path` until validated.
- Use deterministic generation for correctness checks:
  - `--temperature 0`
  - optionally `--show-token-ids`
- When comparing behavior, compare generated token IDs, not just decoded strings.
- Do not optimize away correctness checks because this model has non-standard CCA behavior.

## Suggested branch strategy

Implementation can happen in parallel, but merging should be serial.

Recommended branches:

1. `perf/moe-single-token`
2. `perf/reuse-mask-rope`
3. `perf/cache-skeleton`
4. `perf/prefill-decode-loop`

Recommended merge order:

1. Merge `perf/moe-single-token` first.
2. Merge `perf/reuse-mask-rope` second.
3. Merge `perf/cache-skeleton` third.
4. Merge `perf/prefill-decode-loop` last.

Reason: the first two should be relatively localized and low-risk. The cache skeleton and cached generation loop will touch function signatures across the model and are more likely to conflict.

---

# Task 1 — MoE single-token decode short-circuit

Suggested branch: `perf/moe-single-token`

## Problem

`Experts.__call__` currently does this:

1. Create `output = mx.zeros_like(hidden_states)`.
2. Loop through every expert.
3. Run each expert on the full `hidden_states` tensor.
4. Mask each expert result with `choices[..., None] == i`.
5. Add all masked results together.

That is acceptable-ish for prompt prefill because many tokens can route to different experts. But during token-by-token generation the input is usually shape `[B, 1, H]`, and in our practical use case `B == 1`. There is exactly one routed expert for that token. Evaluating all experts is pure waste.

## Goal

Add a fast path that evaluates only the chosen expert for single-token decode.

## Scope

Primary target:

- `scripts/run_zaya_mlx.py`
- `class Experts`
- `Experts.__call__`

Optional if needed:

- Add a CLI flag to control the fast path.
- Add a model-level setting to keep the old path available.

## Implementation details

### Current code shape

Look for:

```python
class Experts(nn.Module):
    def __init__(self, args: ZayaArgs):
        super().__init__()
        self.local_experts = [MLP(args) for _ in range(args.num_experts)]

    def __call__(self, hidden_states: mx.array, choices: mx.array, use_mod: bool):
        output = mx.zeros_like(hidden_states)
        for i, expert in enumerate(self.local_experts):
            expert_output = expert(hidden_states)
            output = output + mx.where(choices[..., None] == i, expert_output, mx.zeros_like(expert_output))
        if use_mod:
            output = mx.where(choices[..., None] == len(self.local_experts), hidden_states, output)
        return output
```

### Desired behavior

For general input shape, preserve the old path exactly.

For single-token decode:

- If `hidden_states.shape == (1, 1, H)`:
  - Extract the one selected choice.
  - If `use_mod` and the selected choice equals `len(self.local_experts)`, return `hidden_states`.
  - Else run only `self.local_experts[selected_choice](hidden_states)`.

Pseudo-code:

```python
if hidden_states.shape[0] == 1 and hidden_states.shape[1] == 1:
    selected = int(choices.item())
    if use_mod and selected == len(self.local_experts):
        return hidden_states
    if 0 <= selected < len(self.local_experts):
        return self.local_experts[selected](hidden_states)
    raise ValueError(...)
```

### Important caveat

`choices.item()` forces synchronization. This might erase some or all of the performance benefit. That is why this task must measure both paths.

If sync overhead is too high, keep the fast path behind a flag and document the result. Do not force it on by default unless it clearly wins.

Potential flag shape:

```python
parser.add_argument("--moe-decode-fast-path", action="store_true")
```

But avoid plumbing a flag through every layer unless needed. It may be simpler to add an attribute on `Experts` or `ZayaArgs` if this becomes permanent.

## Validation plan

### Static checks

```bash
uv run python -m py_compile scripts/run_zaya_mlx.py
```

### Tiny correctness check

If adding a test script is too much, use a small inline snippet or a temporary local check to compare:

- Old full-expert path output.
- New single-expert path output.

Use a tiny synthetic model with:

- `B = 1`
- `S = 1`
- fixed choice `0`, `1`, and MOD route if `use_mod=True`

The outputs should match exactly or within normal floating-point tolerance.

### Full model check

Run before/after with deterministic generation:

```bash
uv run python scripts/run_zaya_mlx.py \
  --profile \
  --show-token-ids \
  --max-new-tokens 10 \
  --temperature 0 \
  "Write a tiny Python function named sum_numbers that takes two numbers and returns their sum."
```

Compare token IDs before/after.

### Performance check

Compare `generate_token` average in the printed profile.

Also compare with Q8:

```bash
uv run python scripts/run_zaya_mlx.py --quant q8 --profile --max-new-tokens 10 "Say hello."
```

## Acceptance criteria

- Existing scripts compile.
- General full-batch path remains unchanged.
- Single-token fast path produces identical output for the selected expert path.
- Full-model deterministic token IDs do not change.
- Profiling result is documented in commit message or PR notes.

## Risks

- `.item()` sync overhead may be large.
- Branching on dynamic routing may reduce MLX graph reuse.
- If batch size or sequence length is not exactly one, falling into this path would be wrong; guard carefully.

---

# Task 2 — Rotary and causal mask reuse

Status: implemented on `master` with per-`ZayaModel` caches for causal masks and RoPE tensors.

Suggested branch: `perf/reuse-mask-rope`

## Problem

`ZayaModel.__call__` currently rebuilds:

- Causal attention mask:

  ```python
  mask = mx.triu(mx.full((seq_len, seq_len), -mx.inf), k=1).astype(h.dtype)
  ```

- Rotary embeddings:

  ```python
  cos, sin = rotary_embeddings(self.args, mx.arange(seq_len))
  ```

This happens for every model forward. In uncached generation, every token calls the model with a context length that grows by one. Rebuilding these tensors is not the biggest bottleneck, but it is easy to avoid and reduces noise in profiles.

## Goal

Cache masks and RoPE tensors on `ZayaModel`, keyed by sequence length and dtype where appropriate.

## Scope

Primary target:

- `scripts/run_zaya_mlx.py`
- `class ZayaModel`
- `ZayaModel.__init__`
- `ZayaModel.__call__`
- possibly `rotary_embeddings(...)`

## Implementation details

### Add caches

Implemented in `ZayaModel.__init__`:

```python
self._mask_cache: dict[tuple[int, str], mx.array] = {}
self._rope_cache: dict[int, tuple[mx.array, mx.array]] = {}
```

Since this is an experiment, do not over-engineer cache eviction yet.

### Mask cache

Mask depends on:

- `seq_len`
- dtype used by attention (`h.dtype` today)

Implemented helper:

```python
def causal_mask(self, seq_len: int, dtype) -> mx.array:
    key = (seq_len, str(dtype))
    if key not in self._mask_cache:
        self._mask_cache[key] = mx.triu(mx.full((seq_len, seq_len), -mx.inf), k=1).astype(dtype)
    return self._mask_cache[key]
```

Note: if MLX dtype objects are hashable, use dtype directly; otherwise use `str(dtype)`.

### RoPE cache

RoPE depends on:

- `seq_len`
- model config

The model config is fixed per model instance, so key by `seq_len`.

Implemented helper:

```python
def rope(self, seq_len: int) -> tuple[mx.array, mx.array]:
    if seq_len not in self._rope_cache:
        self._rope_cache[seq_len] = rotary_embeddings(self.args, mx.arange(seq_len))
    return self._rope_cache[seq_len]
```

### Future cache compatibility

Cached generation will need RoPE for absolute positions, not just `mx.arange(seq_len)` from zero. Do not make this task solve cached decode. But leave the code easy to extend, e.g. helper can later accept `positions`.

## Validation plan

### Static checks

```bash
uv run python -m py_compile scripts/run_zaya_mlx.py
```

### Deterministic full model check

Before/after token IDs should match:

```bash
uv run python scripts/run_zaya_mlx.py \
  --profile \
  --show-token-ids \
  --max-new-tokens 10 \
  --temperature 0 \
  "Say hello."
```

### Profile check

This may be a small win. Look for:

- lower `generate_token` avg
- lower layer timings if `--profile-layers` is used

## Acceptance criteria

- No output/token changes at `temperature=0`.
- No extra model files written.
- Works with `--quant full` and `--quant q8`.
- Caches are simple and local to `ZayaModel`.

## Risks

- Dtype mismatch between cached mask and hidden states.
- Caching every sequence length may grow memory. This is probably tiny versus model weights, but mention if observed.
- Cached decode work later needs absolute positions; do not paint that work into a corner.

---

# Task 3 — KV/CCA cache skeleton

Suggested branch: `perf/cache-skeleton`

## Problem

The biggest generation issue is recomputing the entire prompt and generated context every token. Upstream ZAYA supports a cache that includes both standard attention K/V and extra CCA state.

Upstream cache shape from `ZayaDynamicCache`:

- `conv_states`: one per layer, stores recent packed CCA q/k convolution inputs.
- `prev_hs`: one per layer, stores previous hidden state for `val_proj2` during single-token decode.
- inherited DynamicCache K/V storage for attention.
- `has_previous_state`: distinguishes prefill from decode.

Our MLX port currently has no equivalent.

## Goal

Introduce an MLX `ZayaGenerationCache` object and thread it through the model call stack while keeping default behavior identical when no cache is supplied.

This task should be mostly structural. It does not need to make cached generation fast yet.

## Scope

Primary target:

- `scripts/run_zaya_mlx.py`

Classes/functions to touch:

- `ZayaForCausalLM.__call__`
- `ZayaModel.__call__`
- `AttentionLayer.__call__`
- `Attention.__call__`
- `CCA.__call__`

Potentially later:

- `generate_from_messages`
- `generate`

But for this skeleton task, avoid changing generation behavior unless adding an opt-in smoke path.

## Implementation details

### Add cache dataclass

Add near `ZayaArgs` or before `CCA`:

```python
@dataclass
class ZayaGenerationCache:
    args: ZayaArgs
    key_states: list[mx.array | None] = field(init=False)
    value_states: list[mx.array | None] = field(init=False)
    conv_states: list[mx.array | None] = field(init=False)
    prev_hs: list[mx.array | None] = field(init=False)
    seen_tokens: int = 0
    has_previous_state: bool = False

    def __post_init__(self):
        n = self.args.num_hidden_layers
        self.key_states = [None] * n
        self.value_states = [None] * n
        self.conv_states = [None] * n
        self.prev_hs = [None] * n
```

Import `field` from `dataclasses`.

### Add layer numbers where needed

`Attention` and `CCA` need to know which cache slot to use.

Currently:

```python
class Attention(nn.Module):
    def __init__(self, args: ZayaArgs):
        ...
        self.qkv = CCA(args)
```

Change to something like:

```python
class Attention(nn.Module):
    def __init__(self, args: ZayaArgs, layer_n: int):
        self.layer_n = layer_n
        self.qkv = CCA(args, layer_n)
```

Then update `AttentionLayer.__init__`:

```python
self.self_attn = Attention(args, layer_n)
```

### Thread cache through call signatures

Keep default `cache=None` everywhere.

Example:

```python
def __call__(self, input_ids, attention_mask=None, cache: ZayaGenerationCache | None = None):
```

Thread it down through layers.

Do not change old behavior when `cache is None`.

### Optional non-mutating population

If you choose to populate cache in this task, do it only in obvious places and keep it easy to review:

- In attention after computing K/V, store detached/current arrays in `cache.key_states[layer_n]` and `cache.value_states[layer_n]`.
- In CCA after computing `qk0`, store enough state to support future decode.

But it is acceptable for this skeleton task to only add structure and signatures. The next task can populate/use it.

## Validation plan

### Static checks

```bash
uv run python -m py_compile scripts/run_zaya_mlx.py
```

### Default behavior check

Run the same deterministic command before/after. Token IDs should match:

```bash
uv run python scripts/run_zaya_mlx.py \
  --show-token-ids \
  --max-new-tokens 5 \
  --temperature 0 \
  "Say hello."
```

### Cache construction smoke check

Add or manually run a tiny check that:

1. Loads/constructs model args.
2. Creates `ZayaGenerationCache(model.args)`.
3. Calls `model(tokens, cache=cache)` on a tiny synthetic model if available.

If full model is too expensive, synthetic is enough for skeleton.

## Acceptance criteria

- `cache=None` path is unchanged.
- All public scripts compile.
- Layer indices are available anywhere cache state will be stored.
- `ZayaGenerationCache` fields map clearly to upstream concepts in comments.
- No default generation behavior changes yet.

## Risks

- Signature churn will conflict with other branches.
- Accidentally using cache state when `cache=None` should be impossible.
- In-place mutation semantics in MLX differ from PyTorch; prefer assigning arrays to cache lists rather than trying to mutate slices.

---

# Task 4 — Prefill once, decode one token at a time using cache

Status: implemented experimentally on `master` behind `--cache` for CLI and server.

Suggested branch: `perf/prefill-decode-loop`

## Problem

Adding a cache object is not enough. The generation loop must use it. Today `generate_from_messages` does this every token:

```python
logits = model(tokens)[:, -1, :]
...
tokens = mx.concatenate([tokens, next_token], axis=1)
```

So each token recomputes the whole context. With a working cache, generation should run:

1. Full prompt prefill once.
2. One-token decode for every generated token after that.

## Goal

Add an opt-in cached generation mode, probably `--cache`, that uses `ZayaGenerationCache`.

Default can remain uncached until deterministic comparisons pass.

## Scope

Primary target:

- `scripts/run_zaya_mlx.py`

Functions/classes likely touched:

- `ZayaGenerationCache` from Task 3
- `CCA.__call__`
- `Attention.__call__`
- `ZayaModel.__call__`
- `ZayaForCausalLM.__call__`
- `generate_from_messages`
- `generate`
- CLI parser in `main()`
- server request handling if exposing cached mode in `scripts/server_zaya_mlx.py`

## Implementation details

### Generation API

Add a flag/parameter:

```python
def generate_from_messages(..., use_cache: bool = False):
```

CLI:

```python
parser.add_argument("--cache", action="store_true", help="Use experimental KV/CCA generation cache.")
```

### Prefill flow

Pseudo-code:

```python
cache = ZayaGenerationCache(model.args)
logits = model(tokens, cache=cache)[:, -1, :]
cache.seen_tokens = tokens.shape[1]
cache.has_previous_state = True
next_token = sample(logits)
yield next_token
```

### Decode flow

For later tokens:

```python
input_token = next_token  # shape [1, 1]
logits = model(input_token, cache=cache)[:, -1, :]
cache.seen_tokens += 1
next_token = sample(logits)
```

### Attention K/V cache details

In uncached attention today:

1. CCA returns q, k, v for current sequence.
2. q/k get RoPE.
3. k/v may be repeated for GQA.
4. attention runs over current sequence.

For cached decode:

- During prefill, store K/V after RoPE and after shape conversion, before or after repeat consistently.
- During decode, compute K/V for the new token, append to cached K/V on sequence axis, and attend query against all cached keys/values.
- Choose a representation and document it. Suggested cache representation:
  - Store `k` and `v` after RoPE and after GQA repeat, shape `[B, attn_heads/2, T, head_dim]`.
  - Then decode attention can directly call SDPA with `q` shape `[B, attn_heads/2, 1, head_dim]`, cached `k/v` shape `[B, attn_heads/2, T, head_dim]`.
- If storing pre-repeat K/V is easier, that is okay, but be consistent.

### Mask details

During single-token cached decode, query length is 1 and key length is total cached length. If tokens are appended in order, the decode query can attend to all cached keys, so no causal mask may be needed for decode.

During prefill, keep the existing causal mask.

### RoPE position details

This is critical.

Uncached mode uses:

```python
cos, sin = rotary_embeddings(args, mx.arange(seq_len))
```

For cached decode, the single new token is not position 0. It is absolute position `cache.seen_tokens`.

Needed behavior:

- Prefill positions: `0..prompt_len-1`
- First decode token after prefill: position `prompt_len`
- Next: `prompt_len + 1`

Add a way for `ZayaModel.__call__` to use positions:

```python
if cache is not None and cache.has_previous_state:
    positions = mx.array([cache.seen_tokens])
else:
    positions = mx.arange(seq_len)
cos, sin = rotary_embeddings(self.args, positions)
```

Be careful with when `seen_tokens` increments. It should represent the number of tokens already in cache before adding the current input.

### CCA cache details

Upstream CCA cache does two non-standard things:

1. Caches convolution input state for q/k packed stream.
2. Caches previous hidden state for `val_proj2`.

Relevant upstream behavior in words:

- On prefill:
  - Compute `qk_packed0` for the full sequence.
  - Save enough of the packed stream to support the next single-token convolution.
  - Compute convolution over the full padded sequence.
  - Save last hidden state for `prev_hs`.
- On decode:
  - Compute packed q/k for the one new token.
  - Concatenate cached conv state with the new packed q/k token.
  - Run the two CCA conv layers over only that short window.
  - Use cached `prev_hs` as the delayed hidden state for `val_proj2`.
  - Update cached conv state and `prev_hs`.

Shape guidance from upstream:

- `conv_states[layer]`: `[B, in_out_ch, 2]` in PyTorch channel-first layout.
- MLX Conv1d in this port appears to use `[B, T, C]` based on current code:

  ```python
  qk_conv = qk0.transpose(1, 0, 2)  # [B, S, C]
  qk_conv = mx.pad(qk_conv, [(0, 0), (total_padding, 0), (0, 0)])
  qk_conv = self.conv_qk[0](qk_conv)
  qk_conv = self.conv_qk[1](qk_conv)
  qk3 = qk_conv.transpose(1, 0, 2)  # [S, B, C]
  ```

Suggested MLX cache representation:

- Store `conv_states[layer]` as `[B, 2, C]`, i.e. last two packed q/k timesteps in MLX Conv1d layout.
- Store `prev_hs[layer]` as `[B, H]`.

Decode CCA pseudo-code:

```python
qk_current = qk0.transpose(1, 0, 2)  # [B, 1, C]
qk_window = mx.concatenate([cache.conv_states[layer], qk_current], axis=1)  # [B, 3, C]
qk_conv = self.conv_qk[0](qk_window)
qk_conv = self.conv_qk[1](qk_conv)
qk3 = qk_conv.transpose(1, 0, 2)  # should be [1, B, C]
cache.conv_states[layer] = qk_window[:, -2:, :]
hs_d = cache.prev_hs[layer][None, :, :]
cache.prev_hs[layer] = hs[-1]
```

Validate these shapes against MLX Conv1d output carefully.

### Cache mutation timing

Be careful not to update cache too early if later code still needs old state. For example, CCA decode needs old `prev_hs` for `val_proj2`, then should write the new one.

### Profiling additions

Add separate profiler events if useful:

- `prefill`
- `decode_token`
- existing `generate_token` can remain, but distinguishing first token from decode tokens will help.

## Validation plan

### Static checks

```bash
uv run python -m py_compile scripts/run_zaya_mlx.py scripts/server_zaya_mlx.py
bash -n scripts/run_python_sum_mlx.sh
```

### Deterministic comparison

Run uncached and cached with token IDs:

```bash
uv run python scripts/run_zaya_mlx.py \
  --show-token-ids \
  --max-new-tokens 10 \
  --temperature 0 \
  "Say hello."

uv run python scripts/run_zaya_mlx.py \
  --cache \
  --show-token-ids \
  --max-new-tokens 10 \
  --temperature 0 \
  "Say hello."
```

The token IDs should match. If they do not, dump intermediate shapes and inspect RoPE positions and CCA cache first.

### Performance comparison

Run:

```bash
uv run python scripts/run_zaya_mlx.py --profile --max-new-tokens 20 --temperature 0 "Say hello."
uv run python scripts/run_zaya_mlx.py --cache --profile --max-new-tokens 20 --temperature 0 "Say hello."
```

Expected shape of improvement:

- First token may be similar or slightly slower due to cache setup.
- Later decode tokens should be much faster if K/V and CCA cache are correct.

### Q8 check

```bash
uv run python scripts/run_zaya_mlx.py --cache --quant q8 --profile --max-new-tokens 10 "Say hello."
```

### Server check

If server exposes cached mode, test:

```bash
uv run python scripts/server_zaya_mlx.py --quant q8
curl http://127.0.0.1:8123/v1/models
curl http://127.0.0.1:8123/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"zaya-mlx","messages":[{"role":"user","content":"Say hello"}],"max_tokens":10,"temperature":0}'
```

## Acceptance criteria

- Cached mode is opt-in unless token parity is very strong.
- Cached and uncached token IDs match for deterministic short generations.
- Subsequent decode token profile is meaningfully faster.
- CCA cache shape and update logic are commented.
- Uncached mode remains available and unchanged.

## Risks

- This is the highest-risk task.
- RoPE absolute positions are easy to get wrong.
- CCA conv state can be transposed incorrectly because upstream PyTorch uses `[B, C, T]` while this MLX port uses `[B, T, C]` for Conv1d.
- Updating cache state in the wrong order can create one-token shifts.
- If cached and uncached token IDs diverge, do not merge as default.

---

# Task 5 — Better profiling for prefill vs decode

Suggested branch: `perf/profile-prefill-decode`

This can be done independently or folded into Task 4.

## Problem

Current profile has `generate_token`, but once cached generation exists we need to distinguish:

- prompt render/tokenization
- model load
- prefill forward
- first token sampling
- decode forward per token
- token decoding/printing

## Goal

Make profiling expose where time goes after cache changes.

## Implementation details

- In `generate_from_messages`, add separate events:
  - `prefill_forward`
  - `decode_forward`
  - `sample_token`
  - maybe `decode_text`
- Preserve existing summary format.
- Include metadata:
  - `prompt_tokens`
  - `context_tokens`
  - `token_index`
  - `cached: true|false`

## Acceptance criteria

- `--profile` output is still readable.
- `--profile-json` contains enough events to plot prefill vs decode later.

---

# Final merge checklist

Before merging any perf branch to `master`:

```bash
git status --short
uv run python -m py_compile scripts/load_zaya_hf.py scripts/load_zaya_mlx.py scripts/run_zaya_mlx.py scripts/server_zaya_mlx.py
bash -n scripts/run_python_sum_mlx.sh
uv run python scripts/run_zaya_mlx.py --help
uv run python scripts/server_zaya_mlx.py --help
```

If the branch changes generation behavior, also run a deterministic token comparison and include the result in the commit/PR notes.

If the branch changes server behavior, test `/v1/models` and one `/v1/chat/completions` request.
