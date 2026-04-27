# Timer-S1 on Apple Silicon with mlx-lm

Timer-S1 is ByteDance's 8.3B-parameter time series foundation model (750M activated per token via MoE). This fork adds a native MLX implementation so it runs on Apple Silicon without PyTorch or Hugging Face `trust_remote_code`.

- Paper: https://arxiv.org/pdf/2603.04791
- Weights: https://huggingface.co/bytedance-research/Timer-S1

---

## Requirements

- macOS 14+ on Apple Silicon (M1 or later)
- Python 3.10+
- `mlx >= 0.31.2`

---

## Installation

Install this fork's `timer-s1` branch. It is a drop-in replacement for the standard `mlx-lm` package.

```bash
pip install git+https://github.com/cspenn/mlx-lm.git@timer-s1
```

Or clone and install in editable mode for development:

```bash
git clone https://github.com/cspenn/mlx-lm.git
cd mlx-lm
git checkout timer-s1
pip install -e .
```

---

## Convert the weights once (recommended)

Converting saves the weights in MLX format locally, so every subsequent load is fast. Do this once:

```bash
mlx_lm.convert \
  --hf-path bytedance-research/Timer-S1 \
  --mlx-path ~/models/Timer-S1-mlx
```

The converted directory is ~16 GB (bfloat16). For devices with less memory, add `--quantize` to produce a 4-bit version (~5 GB):

```bash
mlx_lm.convert \
  --hf-path bytedance-research/Timer-S1 \
  --mlx-path ~/models/Timer-S1-mlx-4bit \
  --quantize
```

---

## Quick start

```python
import mlx.core as mx
import mlx_lm
import numpy as np

# Load from local converted path, or directly from HuggingFace
model, _ = mlx_lm.load("~/models/Timer-S1-mlx")
# model, _ = mlx_lm.load("bytedance-research/Timer-S1")  # downloads on first run

# --- Prepare your time series ---
# Raw 1-D series of historical observations (any cadence: hourly, daily, etc.)
ts = np.array([120.5, 118.3, 125.1, 130.2, 128.0, 127.5, 131.0, 133.4,
               135.2, 134.0, 136.7, 138.1, 137.5, 139.2, 140.0, 141.3,
               142.5, 143.2, 144.0, 145.1, 146.3, 147.0, 148.2, 149.5,
               150.1, 151.3, 152.0, 153.2, 154.1, 155.0, 156.3, 157.2])

# 1. RevIN normalization (removes level/scale so the model sees stationary data)
mean, std = ts.mean(), ts.std() + 1e-8
ts_norm = (ts - mean) / std

# 2. Split into non-overlapping patches of length 16
PATCH_LEN = 16  # model's input_token_len
num_patches = len(ts_norm) // PATCH_LEN
patches = ts_norm[: num_patches * PATCH_LEN].reshape(num_patches, PATCH_LEN)

# 3. Add batch dimension -> shape (1, num_patches, 16)
x = mx.array(patches[None], dtype=mx.float32)

# --- Run inference ---
predictions = model(x)   # shape: (1, 9, 16)
mx.eval(predictions)     # MLX is lazy; this triggers the actual computation

# --- Interpret outputs ---
# Axis 1 holds 9 quantiles: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# Axis 2 holds the 16-step forecast horizon
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# De-normalize back to original scale
forecast = np.array(predictions[0].tolist())  # (9, 16)
forecast = forecast * std + mean

median    = forecast[4]   # q=0.5  — point forecast
lower_90  = forecast[0]   # q=0.1  — lower bound of 90% interval
upper_90  = forecast[8]   # q=0.9  — upper bound of 90% interval

print("Median forecast (next 16 steps):", median.round(2))
print("90% interval lower:", lower_90.round(2))
print("90% interval upper:", upper_90.round(2))
```

---

## Input / output contract

| | Shape | Description |
|---|---|---|
| **Input** `x` | `(batch, num_patches, 16)` | Pre-patched, RevIN-normalized time series |
| **Output** | `(batch, 9, 16)` | 9 quantile forecasts over a 16-step horizon |

**Patch size is fixed at 16.** Your context window must be a multiple of 16 values. The model accepts any number of patches up to its maximum context of 12,800 positions.

**The model only looks at the last patch position** when producing its forecast (`h[:, -1, :]`). Feed all available history as patches — more context is better.

**Output quantile index map:**

| Index | Quantile | Use |
|---|---|---|
| 0 | 0.1 | Lower tail |
| 1 | 0.2 | |
| 2 | 0.3 | |
| 3 | 0.4 | |
| 4 | 0.5 | **Median / point forecast** |
| 5 | 0.6 | |
| 6 | 0.7 | |
| 7 | 0.8 | |
| 8 | 0.9 | Upper tail |

---

## Batch inference

Pass multiple series in a single call for throughput. All series in a batch must have the same number of patches.

```python
import mlx.core as mx
import numpy as np

# series_list: list of 1-D numpy arrays, each length >= 16 and a multiple of 16
def forecast_batch(model, series_list, patch_len=16):
    normalized = []
    stats = []
    for ts in series_list:
        mean, std = ts.mean(), ts.std() + 1e-8
        stats.append((mean, std))
        ts_norm = (ts - mean) / std
        n = (len(ts_norm) // patch_len) * patch_len
        normalized.append(ts_norm[:n].reshape(-1, patch_len))

    # Pad to same number of patches (truncate to shortest)
    min_patches = min(p.shape[0] for p in normalized)
    batch = np.stack([p[-min_patches:] for p in normalized])  # (B, P, 16)

    x = mx.array(batch, dtype=mx.float32)
    preds = model(x)          # (B, 9, 16)
    mx.eval(preds)
    preds_np = np.array(preds.tolist())

    results = []
    for i, (mean, std) in enumerate(stats):
        results.append(preds_np[i] * std + mean)   # (9, 16) for each series
    return results
```

---

## Production pattern: rolling forecast

For production use, maintain a rolling buffer of the most recent history and re-run inference as new observations arrive.

```python
from collections import deque
import mlx.core as mx
import numpy as np

PATCH_LEN = 16
CONTEXT_PATCHES = 64   # 64 * 16 = 1024 observations of history

class TimerS1Forecaster:
    def __init__(self, model, context_patches=CONTEXT_PATCHES):
        self.model = model
        self.context_len = context_patches * PATCH_LEN
        self._buffer = deque(maxlen=self.context_len)

    def update(self, observations):
        """Add new observations (list or array) to the rolling buffer."""
        self._buffer.extend(float(v) for v in observations)

    def forecast(self):
        """Return (9, 16) quantile forecast, or None if not enough history."""
        if len(self._buffer) < PATCH_LEN:
            return None
        ts = np.array(self._buffer)
        mean, std = ts.mean(), ts.std() + 1e-8
        ts_norm = (ts - mean) / std
        n = (len(ts_norm) // PATCH_LEN) * PATCH_LEN
        patches = ts_norm[:n].reshape(-1, PATCH_LEN)
        x = mx.array(patches[None], dtype=mx.float32)
        preds = self.model(x)
        mx.eval(preds)
        return np.array(preds[0].tolist()) * std + mean  # (9, 16)
```

---

## Memory and performance

| Variant | Disk | Unified Memory (inference) |
|---|---|---|
| bfloat16 (default) | ~16 GB | ~17 GB |
| 4-bit quantized | ~5 GB | ~6 GB |

On M2 Ultra (192 GB) or M3 Max (128 GB) the full bfloat16 model fits comfortably. On M1/M2 Pro (16–32 GB) use the 4-bit quantized variant.

Typical latency for a single forecast (M2 Max, bfloat16): **~80–150 ms**.

MLX caches the compiled compute graph after the first call, so subsequent calls on the same input shape are faster.

---

## CLI script

`examples/timer_s1_forecast.py` is a ready-to-run command-line tool for production batch forecasting.

```bash
# Install extra dependencies for the script
pip install pandas huggingface_hub

# Basic usage — downloads model on first run
python examples/timer_s1_forecast.py --input data.csv

# Use a locally converted model, output a specific quantile
python examples/timer_s1_forecast.py \
  --input data.csv \
  --model-path ~/models/Timer-S1-mlx \
  --quantile 0.5 \
  --output forecast.csv

# Show all 9 quantile columns and timing stats
python examples/timer_s1_forecast.py --input data.csv --verbose
```

**Input CSV format** — two columns, date and value (header names are flexible):

```
date,value
2024-01-01,120.5
2024-01-02,118.3
...
```

**Output CSV** — one row per forecast step with columns `step`, `forecast`, `quantile`.

The script uses non-overlapping patches by default (`--overlap 0`), matching the Timer-S1 paper. Pass `--overlap 4` to use overlapping patches for denser context at the cost of correlated inputs.

---

## Keeping this fork current

This implementation lives on the `timer-s1` branch. To pick up upstream `ml-explore/mlx-lm` bug fixes:

```bash
git clone https://github.com/cspenn/mlx-lm.git
cd mlx-lm
git checkout timer-s1
./scripts/sync-upstream.sh
```

See `scripts/sync-upstream.sh` for the full workflow and `docs/timer_s1.md` for architecture notes.
