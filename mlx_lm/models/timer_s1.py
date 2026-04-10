# Copyright © 2023-2024 Apple Inc.

from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache
from .rope_utils import initialize_rope


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "Timer-S1"
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    intermediate_size: int = 4096
    num_attention_heads: int = 16
    hidden_act: str = "silu"
    max_position_embeddings: int = 12800
    rope_theta: int = 10000
    dropout_rate: float = 0.1
    initializer_range: float = 0.02
    input_token_len: int = 16
    output_token_lens: list = None
    quantiles: list = None
    num_experts: int = 32
    num_experts_per_token: int = 2
    num_mtp_tokens: int = 16

    def __post_init__(self):
        if self.output_token_lens is None:
            self.output_token_lens = [16]
        if self.quantiles is None:
            self.quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


class TimerS1PatchEmbedding(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.input_token_len = args.input_token_len
        self.hidden_layer = nn.Linear(args.input_token_len * 2, args.intermediate_size)
        self.output_layer = nn.Linear(args.intermediate_size, args.hidden_size)
        self.residual_layer = nn.Linear(args.input_token_len * 2, args.hidden_size)
        self.dropout = nn.Dropout(args.dropout_rate)
        self.act = nn.silu

    def __call__(self, x: mx.array) -> mx.array:
        seq_len = x.shape[-1]
        padding = (
            self.input_token_len - (seq_len % self.input_token_len)
        ) % self.input_token_len
        if padding > 0:
            x = mx.concatenate(
                [mx.zeros((*x.shape[:-1], padding), dtype=x.dtype), x], axis=-1
            )
        x = x[..., : x.shape[-1] - (x.shape[-1] % self.input_token_len)]
        patches = x.reshape(*x.shape[:-1], -1, self.input_token_len)
        mask = mx.ones_like(patches)
        combined = mx.concatenate([patches, mask], axis=-1)
        hid = self.act(self.hidden_layer(combined))
        out = self.dropout(self.output_layer(hid))
        return out + self.residual_layer(combined)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(queries, keys, cos, sin):
    q_embed = queries * cos + rotate_half(queries) * sin
    k_embed = keys * cos + rotate_half(keys) * sin
    return q_embed, k_embed


class TimerS1RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position: int = 12800, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_position = max_position
        inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.int32) / dim))
        self.inv_freq = inv_freq
        self._compute_cos_sin_cache(max_position)

    def _compute_cos_sin_cache(self, seq_len: int):
        t = mx.arange(seq_len)
        freqs = mx.outer(t, self.inv_freq)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        self.cos_cached = mx.cos(emb)
        self.sin_cached = mx.sin(emb)

    def __call__(self, queries, keys, offset: int = 0):
        seq_len = queries.shape[-2]
        if offset + seq_len > self.max_position:
            self.max_position = offset + seq_len
            self._compute_cos_sin_cache(self.max_position)
        cos = mx.expand_dims(self.cos_cached[offset : offset + seq_len], 0)
        sin = mx.expand_dims(self.sin_cached[offset : offset + seq_len], 0)
        return apply_rotary_pos_emb(queries, keys, cos, sin)


class TimerS1Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(args.hidden_size, args.hidden_size, bias=True)
        self.k_proj = nn.Linear(args.hidden_size, args.hidden_size, bias=True)
        self.v_proj = nn.Linear(args.hidden_size, args.hidden_size, bias=True)
        self.o_proj = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.gate_proj = nn.Linear(args.hidden_size, args.hidden_size, bias=True)

        self.q_scale = mx.ones(self.head_dim)
        self.k_scale = mx.ones(self.head_dim)

        self.rotary_emb = TimerS1RotaryEmbedding(
            self.head_dim,
            max_position=args.max_position_embeddings,
            base=args.rope_theta,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries = self.q_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = self.k_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        values = self.v_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)

        offset = 0
        if cache is not None:
            offset = cache.offset
            keys, values = cache.update_and_fetch(keys, values)

        queries, keys = self.rotary_emb(queries, keys, offset=offset)

        eps = 1e-6
        q_var = mx.mean(queries**2, axis=-1, keepdims=True) + eps
        queries = queries * mx.rsqrt(q_var) * self.q_scale.reshape(1, 1, 1, -1)
        k_var = mx.mean(keys**2, axis=-1, keepdims=True) + eps
        keys = keys * mx.rsqrt(k_var) * self.k_scale.reshape(1, 1, 1, -1)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )

        gate = (
            mx.sigmoid(self.gate_proj(x))
            .reshape(B, L, self.n_heads, -1)
            .transpose(0, 2, 1, 3)
        )
        output = output * gate

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class TimerS1MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.silu

    def __call__(self, x) -> mx.array:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class TimerS1ExpertsLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_token
        self.num_experts = args.num_experts
        moe_intermediate_size = args.intermediate_size // self.top_k

        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.moe_intermediate_size = moe_intermediate_size

        gate_proj_init = (
            mx.random.normal(
                shape=(args.num_experts, moe_intermediate_size, args.hidden_size)
            )
            * 0.02
        )
        up_proj_init = (
            mx.random.normal(
                shape=(args.num_experts, moe_intermediate_size, args.hidden_size)
            )
            * 0.02
        )
        down_proj_init = (
            mx.random.normal(
                shape=(args.num_experts, args.hidden_size, moe_intermediate_size)
            )
            * 0.02
        )

        self.gate_proj_weight = gate_proj_init
        self.up_proj_weight = up_proj_init
        self.down_proj_weight = down_proj_init


class TimerS1ExpertsLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_token
        self.num_experts = args.num_experts
        moe_intermediate_size = args.intermediate_size // self.top_k
        self.hidden_size = args.hidden_size
        self.moe_intermediate_size = moe_intermediate_size

        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)

        self.gate_proj_weight = (
            mx.random.normal(
                shape=(args.num_experts, moe_intermediate_size, args.hidden_size)
            )
            * 0.02
        )
        self.up_proj_weight = (
            mx.random.normal(
                shape=(args.num_experts, moe_intermediate_size, args.hidden_size)
            )
            * 0.02
        )
        self.down_proj_weight = (
            mx.random.normal(
                shape=(args.num_experts, args.hidden_size, moe_intermediate_size)
            )
            * 0.02
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        B, L, D = hidden_states.shape
        T = B * L
        hidden_flat = hidden_states.reshape(T, D)

        router_logits = self.gate(hidden_flat)
        routing_weights = mx.softmax(router_logits.astype(mx.float32), axis=-1)

        partitioned = mx.argpartition(-routing_weights, kth=self.top_k - 1, axis=-1)
        top_k_indices = partitioned[..., -self.top_k :]
        top_k_weights = mx.take_along_axis(
            routing_weights, top_k_indices, axis=-1
        ).astype(hidden_states.dtype)

        result = mx.zeros((T, D), dtype=hidden_states.dtype)

        for expert_idx in range(self.num_experts):
            for k in range(self.top_k):
                expert_k_mask = top_k_indices[:, k] == expert_idx
                token_indices = [i for i in range(T) if expert_k_mask[i].item()]

                if not token_indices:
                    continue

                token_states = mx.take(hidden_flat, mx.array(token_indices), axis=0)
                weight = top_k_weights[token_indices, k]

                gate_out = nn.silu(
                    mx.matmul(token_states, self.gate_proj_weight[expert_idx].T)
                )
                up_out = mx.matmul(token_states, self.up_proj_weight[expert_idx].T)
                intermediate = gate_out * up_out
                expert_out = mx.matmul(
                    intermediate, self.down_proj_weight[expert_idx].T
                )

                for i, idx in enumerate(token_indices):
                    result[idx] += expert_out[i] * weight[i].item()

        return result.reshape(B, L, D)


class TimerS1DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = TimerS1Attention(args, layer_idx)
        self.ffn_layer = TimerS1ExpertsLayer(args)
        self.norm1 = RMSNorm(args.hidden_size)
        self.norm2 = RMSNorm(args.hidden_size)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.norm1(x), mask, cache)
        h = x + r
        r = self.ffn_layer(self.norm2(h))
        h = h + r
        return h


class TimerS1Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_layer = TimerS1PatchEmbedding(args)
        self.layers = [
            TimerS1DecoderLayer(args, layer_idx)
            for layer_idx in range(args.num_hidden_layers)
        ]
        self.norm = RMSNorm(args.hidden_size)

    def __call__(
        self,
        x: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = self.embed_layer(x)

        if cache is None:
            cache = [None] * len(self.layers)

        mask = create_attention_mask(h, cache[0] if cache[0] else None)

        for layer, layer_cache in zip(self.layers, cache):
            h = layer(h, mask, layer_cache)

        return self.norm(h)


class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.out_dim = len(args.quantiles) * args.output_token_lens[-1]
        self.dropout = nn.Dropout(args.dropout_rate)
        self.hidden_layer = nn.Linear(args.hidden_size, args.hidden_size)
        self.act = nn.silu
        self.output_layer = nn.Linear(args.hidden_size, self.out_dim)
        self.residual_layer = nn.Linear(args.hidden_size, self.out_dim)

    def __call__(self, x: mx.array) -> mx.array:
        hid = self.act(self.hidden_layer(x))
        out = self.dropout(self.output_layer(hid))
        return out + self.residual_layer(x)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.num_quantiles = len(args.quantiles)
        self.model = TimerS1Model(args)
        self.output_patch_embedding = ResidualBlock(args)

    def __call__(
        self,
        x: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = self.model(x, cache)
        predictions = self.output_patch_embedding(h[:, -1, :])
        return predictions.reshape(
            -1, self.num_quantiles, self.args.output_token_lens[-1]
        )

    def sanitize(self, weights):
        return {k: v for k, v in weights.items() if "rotary_emb.inv_freq" not in k}

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [KVCache() for _ in self.layers]
