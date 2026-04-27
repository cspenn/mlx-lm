# Copyright © 2023-2024 Apple Inc.
import copy
import unittest

import mlx.core as mx
from mlx.utils import tree_map

from mlx_lm.models.timer_s1 import Model, ModelArgs, TimerS1PatchEmbedding


class TestTimerS1(unittest.TestCase):
    """Tests for the Timer-S1 time-series forecasting model.

    Timer-S1 inputs are pre-patched: (batch, num_patches, input_token_len).
    The model outputs quantile forecasts: (batch, num_quantiles, output_token_len).
    """

    def _make_args(self) -> ModelArgs:
        """Small config for fast tests."""
        return ModelArgs(
            hidden_size=128,
            num_hidden_layers=2,
            intermediate_size=256,
            num_attention_heads=4,
            num_experts=4,
            num_experts_per_token=2,
            input_token_len=16,
            output_token_lens=[16],
            quantiles=[0.1, 0.5, 0.9],
        )

    def test_patch_embedding_shape(self):
        args = self._make_args()
        embedding = TimerS1PatchEmbedding(args)
        batch_size = 2
        num_patches = 4
        # Input shape: (batch, num_patches, input_token_len)
        x = mx.random.normal(shape=(batch_size, num_patches, args.input_token_len))
        out = embedding(x)
        self.assertEqual(out.shape, (batch_size, num_patches, args.hidden_size))

    def test_model_output_shape(self):
        args = self._make_args()
        model = Model(args)
        batch_size = 2
        num_patches = 4
        x = mx.random.normal(shape=(batch_size, num_patches, args.input_token_len))
        out = model(x)
        self.assertEqual(
            out.shape,
            (batch_size, len(args.quantiles), args.output_token_lens[-1]),
        )

    def test_model_with_cache(self):
        args = self._make_args()
        model = Model(args)
        cache = model.make_cache()
        batch_size = 1
        num_patches = 4
        x = mx.random.normal(shape=(batch_size, num_patches, args.input_token_len))
        out = model(x, cache=cache)
        self.assertEqual(
            out.shape,
            (batch_size, len(args.quantiles), args.output_token_lens[-1]),
        )

    def test_dtype_float32(self):
        """Model output dtype must match input dtype (float32)."""
        args = self._make_args()
        model = Model(args)
        model.update(tree_map(lambda p: p.astype(mx.float32), model.parameters()))
        x = mx.random.normal(shape=(1, 4, args.input_token_len))
        out = model(x)
        self.assertEqual(out.dtype, mx.float32)

    def test_dtype_float16(self):
        """Model output dtype must match input dtype (float16)."""
        args = self._make_args()
        model = Model(args)
        model.update(tree_map(lambda p: p.astype(mx.float16), model.parameters()))
        x = mx.random.normal(shape=(1, 4, args.input_token_len)).astype(mx.float16)
        out = model(x)
        self.assertEqual(out.dtype, mx.float16)

    def test_batch_size_greater_than_one(self):
        """Output batch dimension must match input batch dimension."""
        args = self._make_args()
        model = Model(args)
        x = mx.random.normal(shape=(3, 4, args.input_token_len))
        out = model(x)
        self.assertEqual(out.shape[0], 3)

    def test_deepcopy(self):
        """Model must be copyable/picklable."""
        args = self._make_args()
        model = Model(args)
        copy.deepcopy(model)

    def test_layers_property(self):
        """layers property must expose the transformer layer list."""
        args = self._make_args()
        model = Model(args)
        self.assertEqual(len(model.layers), args.num_hidden_layers)


if __name__ == "__main__":
    unittest.main()
