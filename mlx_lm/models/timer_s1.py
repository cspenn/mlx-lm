# Copyright © 2023-2024 Apple Inc.

from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "Timer-S1"
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    intermediate_size: int = 4096
    num_attention_heads: int = 16
    max_position_embeddings: int = 12800
    rope_theta: int = 10000
    input_token_len: int = 16
    output_token_lens: list = field(default_factory=lambda: [16])
    quantiles: list = field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    )
    num_experts: int = 32
    num_experts_per_token: int = 2
    num_mtp_tokens: int = 16

    @property
    def moe_intermediate_size(self) -> int:
        # Each expert uses a fraction of intermediate_size so total capacity
        # equals intermediate_size when top_k experts fire.
        return self.intermediate_size // self.num_experts_per_token


class TimerS1PatchEmbedding(nn.Module):
    """Projects raw time-series patches into the model's hidden dimension.

    Each patch is concatenated with a ones mask (indicating observed values),
    projected through a two-layer MLP, and added to a residual connection.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.input_token_len = args.input_token_len
        self.hidden_layer = nn.Linear(args.input_token_len * 2, args.intermediate_size)
        self.output_layer = nn.Linear(args.intermediate_size, args.hidden_size)
        self.residual_layer = nn.Linear(args.input_token_len * 2, args.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        batch_size, num_patches = x.shape[0], x.shape[1]
        mask = mx.ones_like(x)
        combined = mx.concatenate([x, mask], axis=-1).reshape(
            batch_size, num_patches, -1
        )
        hid = nn.silu(self.hidden_layer(combined))
        return self.output_layer(hid) + self.residual_layer(combined)


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
        # Gate applied to attention output before projection.
        self.gate_proj = nn.Linear(args.hidden_size, args.hidden_size, bias=True)

        # Learnable per-head scaling applied after QK normalization.
        self.q_scale = mx.ones(self.head_dim)
        self.k_scale = mx.ones(self.head_dim)

        self.rope = initialize_rope(
            dims=self.head_dim,
            base=args.rope_theta,
            traditional=False,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries = self.q_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = self.k_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        values = self.v_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)

        offset = cache.offset if cache is not None else 0

        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        # Per-head QK normalization with learnable scale vectors.
        eps = 1e-6
        q_var = mx.mean(queries**2, axis=-1, keepdims=True) + eps
        queries = queries * mx.rsqrt(q_var) * self.q_scale.reshape(1, 1, 1, -1)
        k_var = mx.mean(keys**2, axis=-1, keepdims=True) + eps
        keys = keys * mx.rsqrt(k_var) * self.k_scale.reshape(1, 1, 1, -1)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask  # type: ignore[arg-type]
        )

        gate = (
            mx.sigmoid(self.gate_proj(x))
            .reshape(B, L, self.n_heads, -1)
            .transpose(0, 2, 1, 3)
        )
        output = (output * gate).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class TimerS1ExpertsLayer(nn.Module):
    """32-expert MoE with top-2 routing using SwitchGLU."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_token
        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.experts = SwitchGLU(
            input_dims=args.hidden_size,
            hidden_dims=args.moe_intermediate_size,
            num_experts=args.num_experts,
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        B, L, D = hidden_states.shape
        x = hidden_states.reshape(-1, D)

        router_logits = self.gate(x)
        routing_weights = mx.softmax(router_logits.astype(mx.float32), axis=-1)

        indices = mx.argpartition(-routing_weights, kth=self.top_k - 1, axis=-1)
        top_k_indices = indices[..., : self.top_k]
        top_k_weights = mx.take_along_axis(
            routing_weights, top_k_indices, axis=-1
        ).astype(hidden_states.dtype)

        y = self.experts(x, top_k_indices)
        result = (mx.expand_dims(top_k_weights, -1) * y).sum(-2)
        return result.reshape(B, L, D)


class TimerS1DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = TimerS1Attention(args, layer_idx)
        self.ffn_layer = TimerS1ExpertsLayer(args)
        self.norm1 = nn.RMSNorm(args.hidden_size)
        self.norm2 = nn.RMSNorm(args.hidden_size)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        h = x + self.self_attn(self.norm1(x), mask, cache)
        return h + self.ffn_layer(self.norm2(h))


class TimerS1Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_layer = TimerS1PatchEmbedding(args)
        self.layers = [
            TimerS1DecoderLayer(args, i) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size)

    def __call__(
        self,
        x: mx.array,
        cache: Any | None = None,
    ) -> mx.array:
        h = self.embed_layer(x)

        if cache is None:
            cache = [None] * len(self.layers)

        mask = create_attention_mask(h, cache[0])

        for layer, layer_cache in zip(self.layers, cache):
            h = layer(h, mask, layer_cache)

        return self.norm(h)


class ResidualBlock(nn.Module):
    """Output head: maps last hidden state to quantile predictions."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.out_dim = len(args.quantiles) * args.output_token_lens[-1]
        self.hidden_layer = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer = nn.Linear(args.hidden_size, self.out_dim)
        self.residual_layer = nn.Linear(args.hidden_size, self.out_dim)

    def __call__(self, x: mx.array) -> mx.array:
        hid = nn.silu(self.hidden_layer(x))
        return self.output_layer(hid) + self.residual_layer(x)


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
        cache: Any | None = None,
    ) -> mx.array:
        h = self.model(x, cache)
        predictions = self.output_patch_embedding(h[:, -1, :])
        return predictions.reshape(
            -1, self.num_quantiles, self.args.output_token_lens[-1]
        )

    def sanitize(self, weights: dict) -> dict:
        """Remap HuggingFace weight names to MLX model parameter paths.

        Transformations applied:
        - Drop mtp_modules.* (training-only auxiliary heads).
        - Drop rotary_emb.* (RoPE is computed on the fly via nn.RoPE).
        - Stack per-expert weights (one key per expert in HF) into a single
          tensor per projection per layer (SwitchGLU format).
        All other weight names are identical between HF and MLX.
        """
        num_layers = self.args.num_hidden_layers

        # Remove training-only and recomputed keys.
        drop_patterns = ("mtp_modules", "rotary_emb")
        weights = {
            k: v for k, v in weights.items() if not any(p in k for p in drop_patterns)
        }

        # Collect per-expert weights by layer.
        gate_weights: dict[int, list[tuple[int, mx.array]]] = {
            l: [] for l in range(num_layers)
        }
        up_weights: dict[int, list[tuple[int, mx.array]]] = {
            l: [] for l in range(num_layers)
        }
        down_weights: dict[int, list[tuple[int, mx.array]]] = {
            l: [] for l in range(num_layers)
        }
        expert_keys: list[str] = []

        for k, v in weights.items():
            if ".ffn_layer.experts." not in k:
                continue
            parts = k.split(".")
            # key pattern: model.layers.<L>.ffn_layer.experts.<E>.<proj>.weight
            layer_idx = int(parts[2])
            expert_idx = int(parts[5])
            proj_name = parts[6]
            if layer_idx >= num_layers:
                continue
            if proj_name == "gate_proj":
                gate_weights[layer_idx].append((expert_idx, v))
            elif proj_name == "up_proj":
                up_weights[layer_idx].append((expert_idx, v))
            elif proj_name == "down_proj":
                down_weights[layer_idx].append((expert_idx, v))
            expert_keys.append(k)

        for k in expert_keys:
            weights.pop(k)

        for layer_idx in range(num_layers):
            gate_weights[layer_idx].sort(key=lambda t: t[0])
            up_weights[layer_idx].sort(key=lambda t: t[0])
            down_weights[layer_idx].sort(key=lambda t: t[0])

            base = f"model.layers.{layer_idx}.ffn_layer.experts"
            weights[f"{base}.gate_proj.weight"] = mx.stack(
                [v for _, v in gate_weights[layer_idx]]
            )
            weights[f"{base}.up_proj.weight"] = mx.stack(
                [v for _, v in up_weights[layer_idx]]
            )
            weights[f"{base}.down_proj.weight"] = mx.stack(
                [v for _, v in down_weights[layer_idx]]
            )

        return weights

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [KVCache() for _ in self.layers]
