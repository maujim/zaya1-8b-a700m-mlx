#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer


MODEL_ID = "Zyphra/ZAYA1-8B"


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
    def __init__(self, args: ZayaArgs):
        super().__init__()
        self.args = args
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

    def __call__(self, hidden_states: mx.array, cca_mask: mx.array | None):
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
    def __init__(self, args: ZayaArgs):
        super().__init__()
        self.args = args
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.num_key_value_groups = args.num_attention_heads // args.num_key_value_heads
        self.qkv = CCA(args)
        self.o_proj = nn.Linear((args.num_attention_heads // 2) * self.head_dim, args.hidden_size, bias=args.attention_bias)

    def __call__(self, x: mx.array, mask: mx.array | None, cca_mask: mx.array | None, cos: mx.array, sin: mx.array):
        bsz, seq_len, _ = x.shape
        q, k, v = self.qkv(x, cca_mask)
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

    def __call__(self, hidden_states: mx.array, choices: mx.array, use_mod: bool):
        outputs = []
        for i, expert in enumerate(self.local_experts):
            outputs.append(expert(hidden_states))
        if use_mod:
            outputs.append(hidden_states)
        stacked = mx.stack(outputs, axis=-2)
        return mx.take_along_axis(stacked, choices[..., None, None], axis=-2)[..., 0, :]


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
        self.self_attn = Attention(args)
        self.input_norm = RMSNorm(args.hidden_size, args.norm_epsilon)
        if args.scale_residual_merge:
            self.res_scale = ResidualScaling(args, layer_n)

    def __call__(self, hidden_states, residual, mask, cca_mask, cos, sin, prev_router_hidden_states):
        if self.args.scale_residual_merge:
            residual, hidden_states = self.res_scale(residual, hidden_states)
        residual = hidden_states if residual is None else hidden_states + residual
        hidden_states = self.input_norm(residual)
        hidden_states = self.self_attn(hidden_states, mask, cca_mask, cos, sin)
        return hidden_states, residual, prev_router_hidden_states


class MLPLayer(nn.Module):
    def __init__(self, args: ZayaArgs, layer_n: int):
        super().__init__()
        self.args = args
        self.zaya_block = ZayaBlock(args, layer_n)
        self.input_norm = RMSNorm(args.hidden_size, args.norm_epsilon)
        if args.scale_residual_merge:
            self.res_scale = ResidualScaling(args, layer_n)

    def __call__(self, hidden_states, residual, mask, cca_mask, cos, sin, prev_router_hidden_states):
        if self.args.scale_residual_merge:
            residual, hidden_states = self.res_scale(residual, hidden_states)
        residual = hidden_states if residual is None else hidden_states + residual
        hidden_states = self.input_norm(residual)
        hidden_states, prev_router_hidden_states = self.zaya_block(hidden_states, prev_router_hidden_states)
        return hidden_states, residual, prev_router_hidden_states


class ZayaModel(nn.Module):
    def __init__(self, args: ZayaArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            MLPLayer(args, i) if i % 2 == 1 else AttentionLayer(args, i)
            for i in range(args.num_hidden_layers)
        ]
        if args.scale_residual_merge:
            self.res_scale = ResidualScaling(args, args.num_hidden_layers)
        self.final_norm = RMSNorm(args.hidden_size, args.norm_epsilon)

    def __call__(self, input_ids: mx.array, attention_mask: mx.array | None = None):
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
        for layer in self.layers:
            h, residual, prev_router_hidden_states = layer(h, residual, mask, cca_mask, cos, sin, prev_router_hidden_states)
        if self.args.scale_residual_merge:
            residual, h = self.res_scale(residual, h)
        residual = h if residual is None else h + residual
        return self.final_norm(residual)


class ZayaForCausalLM(nn.Module):
    def __init__(self, args: ZayaArgs):
        super().__init__()
        self.args = args
        self.model = ZayaModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=args.lm_head_bias)

    def __call__(self, input_ids: mx.array, attention_mask: mx.array | None = None):
        out = self.model(input_ids, attention_mask)
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


def load_model(model_path: Path) -> ZayaForCausalLM:
    args = ZayaArgs.from_json(model_path / "config.json")
    model = ZayaForCausalLM(args)
    weights = {}
    for shard in sorted(model_path.glob("model-*.safetensors")):
        weights.update(mx.load(str(shard)))
    weights = model.sanitize(weights)
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    return model


def generate(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float):
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = prompt
    token_ids = tokenizer(text, return_tensors="np")["input_ids"]
    tokens = mx.array(token_ids)
    for _ in range(max_new_tokens):
        logits = model(tokens)[:, -1, :]
        if temperature > 0:
            next_token = mx.random.categorical(logits / temperature, num_samples=1)
        else:
            next_token = mx.argmax(logits, axis=-1, keepdims=True)
        mx.eval(next_token)
        token = int(next_token.item())
        if token == tokenizer.eos_token_id:
            break
        yield token
        tokens = mx.concatenate([tokens, next_token], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ZAYA1-8B with a local MLX implementation.")
    parser.add_argument("prompt", nargs="?", default="Write one sentence about Apple Silicon.")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    model_path = Path(snapshot_download(MODEL_ID, local_files_only=args.local_files_only))
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = load_model(model_path)

    pieces = []
    for token in generate(model, tokenizer, args.prompt, args.max_new_tokens, args.temperature):
        text = tokenizer.decode([token], skip_special_tokens=True)
        pieces.append(text)
        print(text, end="", flush=True)
    if pieces:
        print()


if __name__ == "__main__":
    main()
