#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer


MODEL_ID = "Zyphra/ZAYA1-8B"
QUANT_CHOICES = ("full", "q8")


class Profiler:
    def __init__(self, enabled: bool = False, profile_layers: bool = False):
        self.enabled = enabled
        self.profile_layers = profile_layers
        self.events: list[dict[str, Any]] = []
        self.counters: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)
        self._t0 = time.perf_counter()

    @contextmanager
    def span(self, name: str, *, force_eval: Any = None, **meta: Any):
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            if force_eval is not None:
                mx.eval(force_eval)
            elapsed = time.perf_counter() - start
            self.events.append({"name": name, "ms": elapsed * 1000, **meta})
            self.counters[name] += elapsed
            self.counts[name] += 1

    def add_event(self, name: str, ms: float, **meta: Any) -> None:
        if self.enabled:
            self.events.append({"name": name, "ms": ms, **meta})
            self.counters[name] += ms / 1000
            self.counts[name] += 1

    def memory_info(self) -> dict[str, Any]:
        info = {}
        for key, attr in {
            "active_memory_mb": "get_active_memory",
            "peak_memory_mb": "get_peak_memory",
            "cache_memory_mb": "get_cache_memory",
        }.items():
            fn = getattr(mx, attr, None)
            if fn is not None:
                try:
                    info[key] = fn() / (1024 * 1024)
                except Exception:
                    pass
        return info

    def report(self) -> dict[str, Any]:
        total = time.perf_counter() - self._t0
        summary = []
        for name, seconds in sorted(self.counters.items(), key=lambda item: item[1], reverse=True):
            count = self.counts[name]
            summary.append({"name": name, "count": count, "total_ms": seconds * 1000, "avg_ms": seconds * 1000 / count})
        return {"total_wall_ms": total * 1000, "summary": summary, "events": self.events, "memory": self.memory_info()}

    def print_report(self) -> None:
        if not self.enabled:
            return
        report = self.report()
        print("\n\n=== MLX profile ===")
        for row in report["summary"]:
            print(f"{row['name']:<28} {row['count']:>5}x {row['total_ms']:>10.1f} ms avg {row['avg_ms']:>8.1f} ms")
        if report["memory"]:
            mem = ", ".join(f"{k}={v:.1f}" for k, v in report["memory"].items())
            print(f"memory: {mem}")
        print(f"total wall: {report['total_wall_ms']:.1f} ms")


@dataclass
class ZayaGenerationCache:
    """MLX equivalent of upstream ZayaDynamicCache.

    Stores KV cache for attention layers alongside CCA convolution state
    and previous hidden state for val_proj2 during single-token decode.

    Fields:
        args: Model configuration (fixed per model instance).
        key_states: Per-layer cached attention K (post-RoPE, post-GQA-repeat).
        value_states: Per-layer cached attention V (post-GQA-repeat).
        conv_states: Per-layer CCA convolution window [B, 2, C] in MLX Conv1d layout.
        prev_hs: Per-layer previous hidden state for val_proj2 delay [B, H].
        seen_tokens: Number of tokens already consumed (position offset for decode).
        has_previous_state: Distinguishes prefill from decode.
    """

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


@dataclass
class ZayaArgs:
    vocab_size: int
    hidden_size: int
    ffn_hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    num_query_groups: int
    cca_num_q_heads: int
    num_experts: int
    zaya_mlp_expansion: int
    norm_epsilon: float
    rope_theta: float
    max_position_embeddings: int
    partial_rotary_factor: float
    attention_bias: bool
    lm_head_bias: bool
    add_bias_linear: bool
    gated_linear_unit: bool
    bias_activation_fusion: bool
    scale_residual_merge: bool
    residual_in_fp32: bool
    zaya_use_mod: bool
    zaya_use_eda: bool
    moe_router_topk: int
    pad_token_id: int
    bos_token_id: int
    eos_token_id: int
    tie_word_embeddings: bool = True
    cca_time0: int = 2
    cca_time1: int = 2

    @classmethod
    def from_json(cls, path: Path) -> "ZayaArgs":
        data = json.loads(path.read_text())
        return cls(
            vocab_size=data["vocab_size"],
            hidden_size=data["hidden_size"],
            ffn_hidden_size=data["ffn_hidden_size"],
            num_hidden_layers=data["num_hidden_layers"],
            num_attention_heads=data["num_attention_heads"],
            num_key_value_heads=data["num_key_value_heads"],
            num_query_groups=data["num_query_groups"],
            cca_num_q_heads=data["cca_num_q_heads"],
            num_experts=data["num_experts"],
            zaya_mlp_expansion=data["zaya_mlp_expansion"],
            norm_epsilon=data["norm_epsilon"],
            rope_theta=data["rope_theta"],
            max_position_embeddings=data["max_position_embeddings"],
            partial_rotary_factor=data["partial_rotary_factor"],
            attention_bias=data["attention_bias"],
            lm_head_bias=data["lm_head_bias"],
            add_bias_linear=data["add_bias_linear"],
            gated_linear_unit=data["gated_linear_unit"],
            bias_activation_fusion=data["bias_activation_fusion"],
            scale_residual_merge=data["scale_residual_merge"],
            residual_in_fp32=data["residual_in_fp32"],
            zaya_use_mod=data["zaya_use_mod"],
            zaya_use_eda=data["zaya_use_eda"],
            moe_router_topk=data["moe_router_topk"],
            pad_token_id=data["pad_token_id"],
            bos_token_id=data["bos_token_id"],
            eos_token_id=data["eos_token_id"],
            tie_word_embeddings=data.get("tie_word_embeddings", True),
            cca_time0=data.get("cca_time0", 2),
            cca_time1=data.get("cca_time1", 2),
        )


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


class ResidualScaling(nn.Module):
    def __init__(self, args: ZayaArgs, layer_n: int):
        super().__init__()
        self.not_first_layer = layer_n != 0
        self.hidden_states_scale = mx.ones((args.hidden_size,))
        self.hidden_states_bias = mx.zeros((args.hidden_size,))
        if self.not_first_layer:
            self.residual_scale = mx.ones((args.hidden_size,))
            self.residual_bias = mx.zeros((args.hidden_size,))

    def __call__(self, residual: mx.array | None, hidden_states: mx.array):
        hidden_states = (hidden_states + self.hidden_states_bias) * self.hidden_states_scale
        if self.not_first_layer and residual is not None:
            residual = (residual + self.residual_bias) * self.residual_scale
        return residual, hidden_states


def rotate_half(x: mx.array) -> mx.array:
    x1, x2 = mx.split(x, 2, axis=-1)
    return mx.concatenate([-x2, x1], axis=-1)


def rotary_embeddings(args: ZayaArgs, positions: mx.array):
    head_dim = args.hidden_size // args.num_attention_heads
    rotary_dim = int(head_dim * args.partial_rotary_factor)
    rotary_dim -= rotary_dim % 2
    inv_freq = 1.0 / (args.rope_theta ** (mx.arange(0, rotary_dim, 2, dtype=mx.float32) / rotary_dim))
    freqs = positions.astype(mx.float32)[:, None] * inv_freq[None, :]
    emb = mx.concatenate([freqs, freqs], axis=-1)
    return mx.cos(emb), mx.sin(emb)


def apply_rope(q: mx.array, k: mx.array, cos: mx.array, sin: mx.array):
    rotary_dim = cos.shape[-1]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q = mx.concatenate([(q_rot * cos) + (rotate_half(q_rot) * sin), q_pass], axis=-1)
    k = mx.concatenate([(k_rot * cos) + (rotate_half(k_rot) * sin), k_pass], axis=-1)
    return q, k


class CCA(nn.Module):
    def __init__(self, args: ZayaArgs, layer_n: int):
        super().__init__()
        self.args = args
        self.layer_n = layer_n
        self.hidden_size = args.hidden_size
        self.num_kv_heads = args.num_query_groups
        self.num_q_heads = args.cca_num_q_heads
        self.num_heads = args.num_attention_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.latent_k_dim = self.num_kv_heads * self.head_dim
        self.latent_q_dim = self.num_q_heads * self.head_dim
        self.sqrt_head_dim = self.head_dim**0.5
        self.gqa_groups = self.num_q_heads // self.num_kv_heads
        self.total_padding = args.cca_time0 + args.cca_time1 - 2
        self.linear_q = nn.Linear(args.hidden_size, self.latent_q_dim, bias=args.attention_bias)
        self.linear_k = nn.Linear(args.hidden_size, self.latent_k_dim, bias=args.attention_bias)
        self.val_proj1 = nn.Linear(args.hidden_size, self.latent_k_dim // 2, bias=args.attention_bias)
        self.val_proj2 = nn.Linear(args.hidden_size, self.latent_k_dim // 2, bias=args.attention_bias)
        channels = self.latent_k_dim + self.latent_q_dim
        self.conv_qk = [
            nn.Conv1d(channels, channels, args.cca_time0, groups=channels),
            nn.Conv1d(channels, channels, args.cca_time1, groups=(self.num_kv_heads + self.num_q_heads)),
        ]
        self.temp = mx.zeros((self.num_kv_heads,))

    def __call__(self, hidden_states: mx.array, cca_mask: mx.array | None, cache: ZayaGenerationCache | None = None):
        if cca_mask is not None and hidden_states.shape[1] > 1:
            hidden_states = hidden_states * cca_mask[:, :, None]

        hs = hidden_states.transpose(1, 0, 2)
        hs_d = mx.pad(hs[:-1], [(1, 0), (0, 0), (0, 0)])
        q = self.linear_q(hs)
        k = self.linear_k(hs)
        qk0 = mx.concatenate([q, k], axis=-1)

        query_pre = qk0[..., : self.latent_q_dim].reshape(*qk0.shape[:2], self.num_q_heads, self.head_dim)
        key_pre = qk0[..., self.latent_q_dim :].reshape(*qk0.shape[:2], self.num_kv_heads, self.head_dim)
        key_pre = mx.repeat(key_pre[:, :, :, None, :], self.gqa_groups, axis=3).reshape(
            *qk0.shape[:2], self.num_q_heads, self.head_dim
        )
        qk_mean_q = (query_pre + key_pre) / 2
        qk_mean_k = qk_mean_q.reshape(*qk_mean_q.shape[:2], self.num_kv_heads, self.gqa_groups, -1).mean(axis=-2)

        qk_conv = qk0.transpose(1, 0, 2)
        qk_conv = mx.pad(qk_conv, [(0, 0), (self.total_padding, 0), (0, 0)])
        qk_conv = self.conv_qk[0](qk_conv)
        qk_conv = self.conv_qk[1](qk_conv)
        qk3 = qk_conv.transpose(1, 0, 2)

        query = qk3[..., : self.latent_q_dim].reshape(*qk3.shape[:2], self.num_q_heads, self.head_dim) + qk_mean_q
        key = qk3[..., self.latent_q_dim :].reshape(*qk3.shape[:2], self.num_kv_heads, self.head_dim) + qk_mean_k
        v1 = self.val_proj1(hs)
        v2 = self.val_proj2(hs_d)
        value = mx.concatenate([v1, v2], axis=-1).reshape(*hs.shape[:2], self.num_kv_heads, self.head_dim)

        query = query * (self.sqrt_head_dim / mx.linalg.norm(query, axis=-1, keepdims=True))
        key = (key * (self.sqrt_head_dim / mx.linalg.norm(key, axis=-1, keepdims=True))) * self.temp[None, None, :, None]
        query = query.reshape(*query.shape[:2], self.num_q_heads * self.head_dim).transpose(1, 0, 2)
        key = key.reshape(*key.shape[:2], self.num_kv_heads * self.head_dim).transpose(1, 0, 2)
        value = value.reshape(*value.shape[:2], self.num_kv_heads * self.head_dim).transpose(1, 0, 2)
        return query, key, value


class Attention(nn.Module):
    def __init__(self, args: ZayaArgs, layer_n: int):
        super().__init__()
        self.args = args
        self.layer_n = layer_n
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.num_key_value_groups = args.num_attention_heads // args.num_key_value_heads
        self.qkv = CCA(args, layer_n)
        self.o_proj = nn.Linear((args.num_attention_heads // 2) * self.head_dim, args.hidden_size, bias=args.attention_bias)

    def __call__(self, x: mx.array, mask: mx.array | None, cca_mask: mx.array | None, cos: mx.array, sin: mx.array, cache: ZayaGenerationCache | None = None):
        bsz, seq_len, _ = x.shape
        q, k, v = self.qkv(x, cca_mask, cache=cache)
        q = q.reshape(bsz, seq_len, self.args.num_attention_heads // 2, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(bsz, seq_len, self.args.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(bsz, seq_len, self.args.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)
        q, k = apply_rope(q, k, cos, sin)
        repeats = self.num_key_value_groups // 2
        if repeats > 1:
            k = mx.repeat(k, repeats, axis=1)
            v = mx.repeat(v, repeats, axis=1)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim**-0.5, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(bsz, seq_len, self.args.hidden_size // 2)
        return self.o_proj(out)


class Router(nn.Module):
    def __init__(self, args: ZayaArgs, layer_n: int):
        super().__init__()
        self.args = args
        self.use_mod = args.zaya_use_mod
        self.num_experts = args.num_experts + 1 if self.use_mod else args.num_experts
        self.topk = args.moe_router_topk
        self.use_eda = args.zaya_use_eda and layer_n != 1
        self.down_proj = nn.Linear(args.hidden_size, args.zaya_mlp_expansion, bias=True)
        self.rmsnorm_eda = RMSNorm(args.zaya_mlp_expansion, args.norm_epsilon)
        if self.use_eda:
            self.router_states_scale = mx.ones((args.zaya_mlp_expansion,))
        self.router_mlp = [
            nn.Linear(args.zaya_mlp_expansion, args.zaya_mlp_expansion, bias=True),
            nn.GELU(),
            nn.Linear(args.zaya_mlp_expansion, args.zaya_mlp_expansion, bias=True),
            nn.GELU(),
            nn.Linear(args.zaya_mlp_expansion, self.num_experts, bias=False),
        ]
        if self.use_mod:
            bias = mx.concatenate([mx.zeros((self.num_experts - 1,), dtype=mx.float32), mx.array([-1.0])])
        else:
            bias = mx.zeros((self.num_experts,), dtype=mx.float32)
        self.balancing_biases = bias

    def __call__(self, hidden_states: mx.array, router_states: mx.array | None):
        hs = self.down_proj(hidden_states)
        if self.use_eda and router_states is not None:
            hs = hs + router_states * self.router_states_scale
        next_router_states = hs
        hs = self.rmsnorm_eda(hs)
        hs = self.router_mlp[1](self.router_mlp[0](hs))
        hs = self.router_mlp[3](self.router_mlp[2](hs))
        probs = mx.softmax(self.router_mlp[4](hs), axis=-1)
        choice = mx.argmax(probs.astype(mx.float32) + self.balancing_biases, axis=-1)
        route_prob = mx.take_along_axis(probs, choice[..., None], axis=-1)[..., 0]
        return route_prob, choice, next_router_states


class MLP(nn.Module):
    def __init__(self, args: ZayaArgs):
        super().__init__()
        self.args = args
        out = args.ffn_hidden_size // 2 if args.gated_linear_unit else args.ffn_hidden_size
        self.linear_fc1 = nn.Linear(args.hidden_size, args.ffn_hidden_size, bias=args.add_bias_linear)
        self.linear_fc2 = nn.Linear(out, args.hidden_size, bias=args.add_bias_linear)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.linear_fc1(x)
        if self.args.gated_linear_unit:
            x1, x2 = mx.split(x, 2, axis=-1)
            x = nn.silu(x1) * x2
        else:
            x = nn.gelu(x)
        return self.linear_fc2(x)


class Experts(nn.Module):
    def __init__(self, args: ZayaArgs):
        super().__init__()
        self.local_experts = [MLP(args) for _ in range(args.num_experts)]
        self.moe_decode_fast_path = False

    def __call__(self, hidden_states: mx.array, choices: mx.array, use_mod: bool):
        # Fast path for single-token decode (B=1, S=1): evaluate only the chosen expert.
        # This avoids running every expert on the full batch and masking afterwards.
        # Note: choices.item() forces a CPU synchronization, which may add overhead.
        if self.moe_decode_fast_path and hidden_states.shape[0] == 1 and hidden_states.shape[1] == 1:
            selected = int(choices.item())
            if use_mod and selected == len(self.local_experts):
                return hidden_states
            if 0 <= selected < len(self.local_experts):
                return self.local_experts[selected](hidden_states)
            raise ValueError(
                f"Invalid expert choice {selected} for {len(self.local_experts)} experts "
                f"(use_mod={use_mod})"
            )

        # Original full-expert path for general shapes (prefill, batch decode, etc.).
        # The PyTorch reference sorts tokens by selected expert and evaluates each
        # expert only on its routed tokens. MLX has less ergonomic dynamic indexed
        # batching, so this port still evaluates each expert on the whole batch,
        # but it avoids the previous `[B, S, E, H]` stack that could spike memory.
        output = mx.zeros_like(hidden_states)
        for i, expert in enumerate(self.local_experts):
            expert_output = expert(hidden_states)
            output = output + mx.where(choices[..., None] == i, expert_output, mx.zeros_like(expert_output))
        if use_mod:
            output = mx.where(choices[..., None] == len(self.local_experts), hidden_states, output)
        return output


class ZayaBlock(nn.Module):
    def __init__(self, args: ZayaArgs, layer_n: int):
        super().__init__()
        self.args = args
        self.router = Router(args, layer_n)
        self.experts = Experts(args)

    def __call__(self, hidden_states: mx.array, prev_router_hidden_states: mx.array | None):
        probs, choices, prev_router_hidden_states = self.router(hidden_states, prev_router_hidden_states)
        expert_output = self.experts(hidden_states, choices, self.args.zaya_use_mod)
        return expert_output * probs[..., None], prev_router_hidden_states


class AttentionLayer(nn.Module):
    def __init__(self, args: ZayaArgs, layer_n: int):
        super().__init__()
        self.args = args
        self.self_attn = Attention(args, layer_n)
        self.input_norm = RMSNorm(args.hidden_size, args.norm_epsilon)
        if args.scale_residual_merge:
            self.res_scale = ResidualScaling(args, layer_n)

    def __call__(self, hidden_states, residual, mask, cca_mask, cos, sin, prev_router_hidden_states, cache: ZayaGenerationCache | None = None):
        if self.args.scale_residual_merge:
            residual, hidden_states = self.res_scale(residual, hidden_states)
        residual = hidden_states if residual is None else hidden_states + residual
        hidden_states = self.input_norm(residual)
        hidden_states = self.self_attn(hidden_states, mask, cca_mask, cos, sin, cache=cache)
        return hidden_states, residual, prev_router_hidden_states


class MLPLayer(nn.Module):
    def __init__(self, args: ZayaArgs, layer_n: int):
        super().__init__()
        self.args = args
        self.zaya_block = ZayaBlock(args, layer_n)
        self.input_norm = RMSNorm(args.hidden_size, args.norm_epsilon)
        if args.scale_residual_merge:
            self.res_scale = ResidualScaling(args, layer_n)

    def __call__(self, hidden_states, residual, mask, cca_mask, cos, sin, prev_router_hidden_states, cache: ZayaGenerationCache | None = None):
        if self.args.scale_residual_merge:
            residual, hidden_states = self.res_scale(residual, hidden_states)
        residual = hidden_states if residual is None else hidden_states + residual
        hidden_states = self.input_norm(residual)
        hidden_states, prev_router_hidden_states = self.zaya_block(hidden_states, prev_router_hidden_states)
        return hidden_states, residual, prev_router_hidden_states


class ZayaModel(nn.Module):
    def __init__(self, args: ZayaArgs, profiler: Profiler | None = None):
        super().__init__()
        self.args = args
        self.profiler = profiler
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            MLPLayer(args, i) if i % 2 == 1 else AttentionLayer(args, i)
            for i in range(args.num_hidden_layers)
        ]
        if args.scale_residual_merge:
            self.res_scale = ResidualScaling(args, args.num_hidden_layers)
        self.final_norm = RMSNorm(args.hidden_size, args.norm_epsilon)

    def __call__(self, input_ids: mx.array, attention_mask: mx.array | None = None, cache: ZayaGenerationCache | None = None):
        h = self.embed_tokens(input_ids)
        seq_len = h.shape[1]
        mask = mx.triu(mx.full((seq_len, seq_len), -mx.inf), k=1).astype(h.dtype)
        if attention_mask is not None:
            cca_mask = attention_mask
        else:
            cca_mask = None
        cos, sin = rotary_embeddings(self.args, mx.arange(seq_len))
        residual = None
        prev_router_hidden_states = None
        for i, layer in enumerate(self.layers):
            if self.profiler and self.profiler.profile_layers:
                kind = "mlp" if isinstance(layer, MLPLayer) else "attn"
                with self.profiler.span(f"layer.{kind}", force_eval=h, layer=i, seq_len=seq_len):
                    h, residual, prev_router_hidden_states = layer(h, residual, mask, cca_mask, cos, sin, prev_router_hidden_states, cache=cache)
            else:
                h, residual, prev_router_hidden_states = layer(h, residual, mask, cca_mask, cos, sin, prev_router_hidden_states, cache=cache)
        if self.args.scale_residual_merge:
            residual, h = self.res_scale(residual, h)
        residual = h if residual is None else h + residual
        return self.final_norm(residual)


class ZayaForCausalLM(nn.Module):
    def __init__(self, args: ZayaArgs, profiler: Profiler | None = None):
        super().__init__()
        self.args = args
        self.model = ZayaModel(args, profiler)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=args.lm_head_bias)

    def __call__(self, input_ids: mx.array, attention_mask: mx.array | None = None, cache: ZayaGenerationCache | None = None):
        out = self.model(input_ids, attention_mask, cache=cache)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def sanitize(self, weights):
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        sanitized = {}
        for key, value in weights.items():
            if ".conv_qk.0.weight" in key or ".conv_qk.1.weight" in key:
                value = value.transpose(0, 2, 1)
            sanitized[key] = value
        return sanitized


def enable_moe_decode_fast_path(model: ZayaForCausalLM) -> None:
    for layer in model.model.layers:
        if isinstance(layer, MLPLayer):
            layer.zaya_block.experts.moe_decode_fast_path = True


def load_model(model_path: Path, profiler: Profiler | None = None, quant: str = "full") -> ZayaForCausalLM:
    if quant not in QUANT_CHOICES:
        raise ValueError(f"quant must be one of {QUANT_CHOICES}, got {quant!r}")
    with (profiler.span("read_config") if profiler else nullcontext()):
        args = ZayaArgs.from_json(model_path / "config.json")
    model = ZayaForCausalLM(args, profiler)
    weights = {}
    shards = sorted(model_path.glob("model-*.safetensors"))
    for shard in shards:
        start = time.perf_counter()
        weights.update(mx.load(str(shard)))
        if profiler:
            profiler.add_event("load_shard", (time.perf_counter() - start) * 1000, shard=shard.name)
    with (profiler.span("sanitize_weights") if profiler else nullcontext()):
        weights = model.sanitize(weights)
    with (profiler.span("load_weights") if profiler else nullcontext()):
        model.load_weights(list(weights.items()), strict=True)
    with (profiler.span("eval_parameters", force_eval=model.parameters()) if profiler else nullcontext()):
        pass
    if quant == "q8":
        with (profiler.span("quantize_q8") if profiler else nullcontext()):
            nn.quantize(
                model,
                bits=8,
                group_size=64,
                class_predicate=lambda _path, module: isinstance(module, nn.Linear)
                and module.weight.shape[-1] % 64 == 0,
            )
        with (profiler.span("eval_quantized_parameters", force_eval=model.parameters()) if profiler else nullcontext()):
            pass
    return model


def render_messages(tokenizer, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "\n".join(f"{message['role']}: {message['content']}" for message in messages) + "\nassistant:"


def generate_from_messages(
    model,
    tokenizer,
    messages: list[dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    profiler: Profiler | None = None,
):
    with (profiler.span("render_prompt") if profiler else nullcontext()):
        text = render_messages(tokenizer, messages)
    with (profiler.span("tokenize_prompt") if profiler else nullcontext()):
        token_ids = tokenizer(text, return_tensors="np")["input_ids"]
    tokens = mx.array(token_ids)
    if profiler:
        profiler.events.append({"name": "prompt", "prompt_tokens": int(tokens.shape[1]), "max_new_tokens": max_new_tokens})
    for i in range(max_new_tokens):
        start = time.perf_counter()
        logits = model(tokens)[:, -1, :]
        if temperature > 0:
            next_token = mx.random.categorical(logits / temperature, num_samples=1)
        else:
            next_token = mx.argmax(logits, axis=-1, keepdims=True)
        mx.eval(next_token)
        elapsed_ms = (time.perf_counter() - start) * 1000
        if profiler:
            profiler.add_event("generate_token", elapsed_ms, token_index=i, context_tokens=int(tokens.shape[1]))
        token = int(next_token.item())
        if token == tokenizer.eos_token_id:
            break
        yield token
        tokens = mx.concatenate([tokens, next_token], axis=1)


def generate(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float, profiler: Profiler | None = None):
    messages = [{"role": "user", "content": prompt}]
    yield from generate_from_messages(model, tokenizer, messages, max_new_tokens, temperature, profiler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ZAYA1-8B with a local MLX implementation.")
    parser.add_argument("prompt", nargs="?", default="Write one sentence about Apple Silicon.")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--model-path", type=Path, help="Use an existing Hugging Face snapshot directory.")
    parser.add_argument("--show-token-ids", action="store_true", help="Print generated token IDs for smoke tests.")
    parser.add_argument("--quant", choices=QUANT_CHOICES, default="full", help="Weight mode: full BF16 weights or quick dynamic Q8 quantization after load.")
    parser.add_argument("--profile", action="store_true", help="Print timing and MLX memory profile after the run.")
    parser.add_argument("--profile-layers", action="store_true", help="Also synchronize and time each transformer layer. Slower, but shows where generation time goes.")
    parser.add_argument("--profile-json", type=Path, help="Write full profiling events and summary as JSON.")
    parser.add_argument(
        "--moe-decode-fast-path",
        action="store_true",
        help="Experimental: evaluate only the chosen MoE expert during single-token decode. "
        "Skips the default full-expert evaluation when hidden_states shape is (1, 1, H). "
        "May increase or decrease latency depending on choices.item() sync overhead.",
    )
    args = parser.parse_args()

    profiler = Profiler(enabled=args.profile or args.profile_json is not None or args.profile_layers, profile_layers=args.profile_layers)

    with profiler.span("resolve_model_path"):
        model_path = args.model_path or Path(snapshot_download(MODEL_ID))
    with profiler.span("load_tokenizer"):
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = load_model(model_path, profiler, quant=args.quant)

    if args.moe_decode_fast_path:
        enable_moe_decode_fast_path(model)

    pieces = []
    for token in generate(model, tokenizer, args.prompt, args.max_new_tokens, args.temperature, profiler):
        if args.show_token_ids:
            print(f"[token_id={token}]", flush=True)
        text = tokenizer.decode([token], skip_special_tokens=True)
        pieces.append(text)
        print(text, end="", flush=True)
    if pieces:
        print()
    profiler.print_report()
    if args.profile_json:
        args.profile_json.write_text(json.dumps(profiler.report(), indent=2))
        print(f"profile JSON written to {args.profile_json}")


if __name__ == "__main__":
    main()
