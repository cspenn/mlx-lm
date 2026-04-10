# Timer-S1

## Model Description

Timer-S1 is a time series foundation model developed by ByteDance Research with 8.3B total parameters and 750M activated parameters per token.

## Supported Architectures

- Timer-S1 (custom decoder-only Transformer with MoE)

## Key Features

- Mixture-of-Experts (MoE) with 32 experts, top-2 routing
- Custom GQA attention with gate projections
- Quantile prediction head (9 outputs for probabilistic forecasting)
- RevIN normalization for time series

## License

Apache 2.0

## References

- [Timer-S1 Paper](https://arxiv.org/pdf/2603.04791)
- [HuggingFace Model](https://huggingface.co/bytedance-research/Timer-S1)
