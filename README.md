# Hybrid Multi-Agent Trading System Integrating Heterogeneous AI Methods

## Documentation

### [docs/features.md](docs/features.md) — Feature sets and selection pipeline ⚠️ Read before training any model

Complete reference for all feature sets (V1: 196 OHLCV features, V3: 21 external
features, V2-1h: 39 structural 1h features, V2-5m: 39 structural 5m features),
the 4-stage selection pipeline (Variance+Corr → MI Top-60 → Walk-forward stability
→ Permutation pruning), and selected features with backtest results for each
experiment (v5, v6). The v7 notebook is the first to include V2-1h structural
features in the pool (255 total).
