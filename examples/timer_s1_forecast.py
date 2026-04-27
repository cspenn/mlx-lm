#!/usr/bin/env python3
"""
Timer-S1 forecasting CLI.

Loads a Timer-S1 model (local MLX path or HuggingFace repo), runs a
forward pass on a CSV time series, and writes quantile forecasts to CSV.

Usage:
    python examples/timer_s1_forecast.py --input data.csv --output forecast.csv
    python examples/timer_s1_forecast.py --input data.csv --model-path ~/models/Timer-S1-mlx
    python examples/timer_s1_forecast.py --input data.csv --quantile 0.5
    python examples/timer_s1_forecast.py --input data.csv --quantile 0.5 --overlap 4 --verbose
"""

import argparse
import time

import mlx.core as mx
import mlx_lm
import numpy as np
import pandas as pd


def load_data(csv_path: str) -> np.ndarray:
    """Load time series from a CSV file with columns: date, value."""
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise ValueError(f"{csv_path}: expected at least 2 columns (date, value)")
    df.columns = list(df.columns[:-1]) + ["value"]
    return df["value"].values.astype(np.float32)


def preprocess(
    data: np.ndarray, patch_len: int = 16, overlap: int = 0
) -> tuple[np.ndarray, float, float]:
    """RevIN-normalize and split into patches.

    Args:
        data:      Raw 1-D time series.
        patch_len: Patch size (must match model's input_token_len, default 16).
        overlap:   Samples shared between consecutive patches. Use 0 (default)
                   for non-overlapping patches as described in the Timer-S1 paper.
                   Values > 0 increase context density but correlate adjacent patches.

    Returns:
        patches: float32 array of shape (num_patches, patch_len).
        mean:    Series mean (for de-normalization).
        std:     Series std  (for de-normalization).
    """
    mean = float(data.mean())
    std = float(data.std()) + 1e-6
    normalized = (data - mean) / std

    # Pad to a multiple of patch_len using reflection (avoids introducing new extremes).
    remainder = len(normalized) % patch_len
    if remainder:
        normalized = np.pad(normalized, (0, patch_len - remainder), mode="reflect")

    step = patch_len - overlap if overlap > 0 else patch_len
    patches = np.array(
        [normalized[i : i + patch_len] for i in range(0, len(normalized) - patch_len + 1, step)],
        dtype=np.float32,
    )
    return patches, mean, std


def run_forecast(model, patches: np.ndarray) -> np.ndarray:
    """Run a single Timer-S1 forward pass.

    Args:
        model:   Loaded Timer-S1 model (already in eval mode).
        patches: Shape (num_patches, patch_len) — batch dimension is added here.

    Returns:
        Numpy array of shape (num_quantiles, output_token_len).
    """
    x = mx.array(patches[None], dtype=mx.float32)  # (1, P, 16)
    output = model(x)                               # (1, 9, 16)
    mx.eval(output)                                 # force lazy evaluation
    return np.array(output[0].tolist())             # (9, 16)


def main():
    parser = argparse.ArgumentParser(description="Timer-S1 Time Series Forecasting")
    parser.add_argument("--input", "-i", required=True, help="CSV file (columns: date, value)")
    parser.add_argument("--output", "-o", default="forecast_output.csv", help="Output CSV path")
    parser.add_argument(
        "--model-path",
        default="bytedance-research/Timer-S1",
        help="Local MLX model directory or HuggingFace repo ID (default: downloads from HF)",
    )
    parser.add_argument("--patch-len", type=int, default=16, help="Patch size (default: 16)")
    parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Samples shared between consecutive patches (default: 0 = non-overlapping)",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.5,
        help="Quantile to write to output CSV: 0.1–0.9 (default: 0.5 median)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    # --- Load data ---
    print(f"Loading data from {args.input}...")
    data = load_data(args.input)
    print(f"  {len(data)} observations")

    # --- Preprocess ---
    patches, mean, std = preprocess(data, args.patch_len, args.overlap)
    print(f"  {len(patches)} patches of length {args.patch_len}")

    # --- Load model ---
    print(f"Loading model from {args.model_path}...")
    t0 = time.perf_counter()
    model, _ = mlx_lm.load(args.model_path)
    model.eval()
    load_time = time.perf_counter() - t0
    print(f"  Loaded in {load_time:.2f}s")

    # --- Infer ---
    print("Running inference...")
    t0 = time.perf_counter()
    raw = run_forecast(model, patches)       # (9, 16)
    infer_time = time.perf_counter() - t0
    print(f"  Done in {infer_time:.3f}s")

    # --- Select quantile ---
    quantile_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    q_idx = min(range(len(quantile_list)), key=lambda i: abs(quantile_list[i] - args.quantile))
    q_val = quantile_list[q_idx]

    forecast_values = raw[q_idx] * std + mean   # de-normalize

    print(f"\nForecast (q={q_val}, next {len(forecast_values)} steps):")
    for i, v in enumerate(forecast_values, 1):
        print(f"  t+{i:>2}: {v:.4f}")

    # --- Write output ---
    pd.DataFrame({"step": range(1, len(forecast_values) + 1), "forecast": forecast_values, "quantile": q_val}).to_csv(
        args.output, index=False
    )
    print(f"\nSaved to {args.output}")

    if args.verbose:
        print("\nAll quantile forecasts (de-normalized):")
        header = "step  |  " + "  ".join(f"q{int(q*10)}" for q in quantile_list)
        print(header)
        for i in range(raw.shape[1]):
            row = f"t+{i+1:<3} |  " + "  ".join(f"{raw[j, i] * std + mean:7.2f}" for j in range(len(quantile_list)))
            print(row)

        print(f"\nModel load: {load_time:.2f}s  |  Inference: {infer_time:.3f}s  |  {len(patches)/infer_time:.1f} patches/s")


if __name__ == "__main__":
    main()
