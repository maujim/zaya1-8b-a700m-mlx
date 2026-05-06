#!/usr/bin/env python3
from __future__ import annotations

import mlx.core as mx

from run_zaya_mlx import ZayaArgs, ZayaForCausalLM, generate_from_messages


class TinyTokenizer:
    eos_token_id = 1

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        return text + ("\nassistant:" if add_generation_prompt else "")

    def __call__(self, text, return_tensors="np"):
        ids = [2] + [3 + (ord(ch) % 20) for ch in text[:8]]
        return {"input_ids": [ids]}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids)


def tiny_args() -> ZayaArgs:
    return ZayaArgs(
        vocab_size=32,
        hidden_size=16,
        ffn_hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        num_query_groups=1,
        cca_num_q_heads=2,
        num_experts=2,
        zaya_mlp_expansion=8,
        norm_epsilon=1e-5,
        rope_theta=10000.0,
        max_position_embeddings=32,
        partial_rotary_factor=0.5,
        attention_bias=False,
        lm_head_bias=False,
        add_bias_linear=False,
        gated_linear_unit=True,
        bias_activation_fusion=True,
        scale_residual_merge=True,
        residual_in_fp32=False,
        zaya_use_mod=True,
        zaya_use_eda=True,
        moe_router_topk=1,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=1,
    )


def main() -> None:
    model = ZayaForCausalLM(tiny_args())
    input_ids = mx.array([[2, 5, 6, 7]])
    logits = model(input_ids)
    mx.eval(logits)
    assert logits.shape == (1, 4, 32), logits.shape

    token = next(generate_from_messages(model, TinyTokenizer(), [{"role": "user", "content": "hi"}], 1, 0.0))
    assert isinstance(token, int)
    print("tiny MLX ZAYA smoke passed")


if __name__ == "__main__":
    main()
