# Copyright © 2023-2024 Apple Inc.
import math
import unittest

import mlx.core as mx
from mlx_lm.models.timer_s1 import Model, ModelArgs, TimerS1PatchEmbedding


class TestTimerS1(unittest.TestCase):
    def test_patch_embedding_shape(self):
        args = ModelArgs()
        embedding = TimerS1PatchEmbedding(args)
        batch_size = 2
        seq_len = 64
        x = mx.random.normal(shape=(batch_size, seq_len))
        out = embedding(x)
        expected_patches = seq_len // args.input_token_len
        self.assertEqual(out.shape, (batch_size, expected_patches, args.hidden_size))

    def test_model_output_shape(self):
        args = ModelArgs()
        model = Model(args)
        batch_size = 2
        seq_len = 64
        x = mx.random.normal(shape=(batch_size, seq_len))
        out = model(x)
        expected_quantiles = len(args.quantiles)
        expected_output_len = args.output_token_lens[-1]
        self.assertEqual(
            out.shape, (batch_size, expected_quantiles, expected_output_len)
        )

    def test_model_with_cache(self):
        args = ModelArgs()
        model = Model(args)
        cache = model.make_cache()
        batch_size = 1
        seq_len = 16
        x = mx.random.normal(shape=(batch_size, seq_len))
        out = model(x, cache=cache)
        expected_quantiles = len(args.quantiles)
        expected_output_len = args.output_token_lens[-1]
        self.assertEqual(
            out.shape, (batch_size, expected_quantiles, expected_output_len)
        )
