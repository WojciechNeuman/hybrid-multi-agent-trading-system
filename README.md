# Hybrid Multi-Agent Trading System Integrating Heterogeneous AI Methods

## Dokumentacja

### [docs/features.md](docs/features.md) — Cechy i selekcja cech ⚠️ Ważne przed tworzeniem modeli

Zawiera pełny opis wszystkich zestawów cech (V1: 196 cech OHLCV, V3: 21 cech
zewnętrznych, V2: 39 cech 5m), aktualny 4-etapowy pipeline selekcji
(Variance+Corr → MI Top-60 → Walk-forward stability → Permutation pruning)
oraz listę 20 cech wybranych w najnowszym eksperymencie LGBM (v5) wraz z
wynikami backtestu. Przejrzyj przed trenowaniem nowego modelu.
