# Feature Engineering Reference

> **Ważne dla tworzenia modeli.** Ten dokument opisuje wszystkie zestawy cech
> używane w projekcie, sposób ich selekcji oraz cechy wybrane w najnowszym
> eksperymencie. Przed trenowaniem nowego modelu przejrzyj ten plik — zawiera
> historię decyzji, których nie ma w kodzie.

---

## Zestaw V1 — 196 cech z danych OHLCV 1h

**Plik źródłowy:** `data/features/BTCUSDT_1h_features.parquet`  
**Rejestr:** `data/features/feature_registry.json`  
**Źródło danych:** wyłącznie OHLCV (Open, High, Low, Close, Volume) świec 1h BTCUSDT  
**Wygenerowano:** 2026-05-16

Wszystkie cechy są czysto ilościowe, bez danych zewnętrznych. 3 kolumny
(`close`, `sma_200`, `atr_14_pct`) to kolumny backtestowe — nie są używane
jako wejście modelu.

### Grupy cech

| Grupa | Liczba | Przykładowe cechy |
|-------|--------|-------------------|
| `returns` | 18 | `ret_1h`, `log_ret_1h`, `ret_6h` … `ret_168h` (1h–7d) |
| `volatility` | 10 | `vol_6h`–`vol_168h`, `atr_14`, `atr_14_pct`, `atr_24` |
| `volatility_regime` | 11 | `gk_vol_24h` (Garman-Klass), `park_vol_24h` (Parkinson), `vol_of_vol_72h`, `bb_squeeze_20/50`, `atr_14_pct_rank`, `range_vs_atr` |
| `ma_ratios` | 12 | `close_vs_sma/ema_7/14/20/50/100/200` (% odchylenie) |
| `long_cycle_ma` | 11 | `close_vs_sma/ema_336/504/720/2160/4320` (2tyg–6mies), `weekly_mom_accel` |
| `ma_crosses` | 9 | `sma50_vs_sma200`, `golden_cross`, `candles_since_cross`, `ma_bull_score`, `ma_ribbon_width` |
| `bollinger` | 4 | `bb_width_20/50`, `bb_position_20/50` |
| `macd` | 4 | `macd_12_26`, `macd_hist_12_26`, `macd_5_13`, `macd_hist_5_13` |
| `oscillators` | 6 | `rsi_7/14/21`, `stoch_k_14/21`, `williams_r` |
| `volume` | 9 | `vol_z_12/24/72/168h`, `vol_ratio_12/24/72/168h`, `obv_z_72` |
| `volume_profile` | 10 | `close_vs_vwap_24/168h`, `mfi_14/21`, `cmf_20`, `ad_z_48/168h`, `vol_spike_2x/3x`, `vw_rsi_14` |
| `candle_structure` | 4 | `candle_body`, `upper_wick`, `lower_wick`, `is_bullish` |
| `candlestick_patterns` | 8 | `bull_engulf`, `bear_engulf`, `doji`, `hammer`, `shooting_star`, `bull/bear_streak`, `marubozu` |
| `price_position` | 3 | `hl_position_24/48/168h` |
| `support_resistance` | 13 | `close_vs_pivot/r1/r2/s1/s2`, `dist_round_1000/10000`, `breakout_up/down_24/48/168h` |
| `fibonacci` | 10 | `fib_position_48/168h`, `fib_nearest_dist/level/dist_618/below_618` × 2 lookbacki |
| `ichimoku` | 10 | `tk_ratio`, `tk_cross_bull/bear`, `close_vs_cloud_top/bottom`, `inside_cloud`, `cloud_thickness/bullish/flip_recency`, `close_vs_kijun` |
| `supertrend` | 8 | `supertrend_dir/dist_15/20/30`, `supertrend_consensus`, `supertrend_flip_recency` |
| `divergences` | 5 | `rsi_divergence`, `macd_divergence`, `obv_divergence`, `vol_price_div_12/24h` |
| `statistical` | 13 | `hurst_168h`, `skew/kurt_24/72/168h`, `autocorr_ret_1/6/12/24h`, `var_ratio_6/24h` |
| `composite` | 7 | `trend_score`, `rsi_vol_confirm`, `mom_coherence`, `sharpe_ratio_24/72/168h`, `regime_composite` |
| `calendar` | 11 | `hour_sin/cos`, `dow_sin/cos`, `halving_cycle_sin/cos/pos`, `dom_sin/cos`, `quarter_sin/cos` |

**Łącznie: 196 cech** (193 modelowych + 3 backtestowe)

---

## Zestaw V3 — 21 cech zewnętrznych

**Plik źródłowy:** `data/features/BTCUSDT_1h_v3_features.parquet`  
**Skrypt generujący:** `src/hmats/notebooks/05_feature_selection_lgbm_v1.py`  
**Dane zewnętrzne:** multi-coin OHLCV (9 altcoinów 1h), CoinGecko market caps, Fear & Greed Index

Te cechy mają częściowe uzasadnienie przyczynowe (nie tylko statystyczne) —
są oparte na danych spoza historii BTC, co czyni je wartościowszymi w dyskusji
o korelacji vs. przyczynowości.

| Grupa | Prefix | Cechy | Opis |
|-------|--------|-------|------|
| Cross-asset | `cross_` | `cross_eth_btc_ratio`, `cross_eth_btc_mom_24h`, `cross_eth_btc_mom_72h`, `cross_altcoin_breadth_24h`, `cross_btc_relative_strength`, `cross_alt_correlation_24h` | ETH/BTC ratio, momentum ETH vs BTC, % altcoinów z dodatnim zwrotem 24h, siła relatywna BTC |
| Market structure | `mkt_` | `mkt_btc_dominance`, `mkt_btc_dominance_chg_7d`, `mkt_eth_dominance`, `mkt_total_mcap_chg_24h`, `mkt_stablecoin_pct` | Dominacja BTC/ETH, zmiana total market cap, udział stablecoinów |
| Sentiment | `sent_` | `sent_fear_greed`, `sent_fear_greed_ma7`, `sent_fear_greed_chg_7d` | Fear & Greed Index (0–1), 7d MA, zmiana 7d |
| Microstructure | `micro_` | `micro_amihud_illiq`, `micro_kyle_lambda`, `micro_roll_spread`, `micro_volume_clock` | Illiquidity Amihuda, Kyle lambda (price impact), Roll spread, regularność wolumenu |
| Enhanced OHLCV | — | `vol_term_structure`, `mom_normalized_24h`, `mom_normalized_72h` | Stosunek zmienności 24h/168h, znormalizowane momentum |

**Łącznie: 21 cech V3**

---

## Zestaw V2 — 39 cech strukturalnych 5m (eksperymentalny)

**Plik źródłowy:** `data/features/BTCUSDT_5m_structural.parquet`  
**Rejestr:** `data/features/feature_registry_v2.json`  
**Timeframe:** świece 5m (z kontekstem 1h i 4h)  
**Wygenerowano:** 2026-05-28

Zestaw przeznaczony do eksperymentów na granularności 5m. Nie był jeszcze używany
w głównym pipeline'ie modelowym.

| Grupa | Prefix | Liczba | Opis |
|-------|--------|--------|------|
| `A_structure` | `struct_` | 11 | Swing high/low (minor/major), body ratio, wick ratios, odległości od swingów |
| `B_liquidity` | `liq_` | 10 | Anchored VWAP (daily/weekly/rolling), POC distance, volume z-score, exhaustion |
| `C_volatility` | `volat_` | 8 | ATR 20/72 pct, BB position/width, Bollinger-Keltner squeeze, Garman-Klass |
| `D_mtf` | `mtf_` | 10 | EMA alignment 1h/4h, RSI 1h/4h, composite MTF score, session encoding |

**Łącznie: 39 cech V2 (5m)**

---

## Wczesny zestaw agentów — 12 cech (archiwalny)

**Plik źródłowy:** `src/hmats/data/features.py` → `FEATURE_COLS`  
**Używany w:** wczesnych agentach NEAT/PPO (przed przejściem na LGBM/TCN)

Prosty zestaw 12 cech używany w pierwszych eksperymentach z RL:

```
log_ret_1, vol_24, vol_72, sma_ratio_24_72, macd, macd_signal, macd_hist,
mom_24, mom_72, rsi_14, volu_z_72, z_close_72
```

Cechy są clip-owane do `[-10, 10]` i standaryzowane statystykami zbioru treningowego.

---

## Metoda selekcji cech (aktualny pipeline — v5)

Stosowany w: `src/hmats/notebooks/02_lgbm_omni_0fee_v5.ipynb`

Selekcja odbywa się **wyłącznie na danych pre-OOS** (przed 2024-01-01),
co zapobiega data leakage.

```
Pool V1+V3 (216 cech)
       │
       ▼
Stage 1 — Variance + Spearman filter
  • Usuwa cechy z wariancją < 1e-6
  • Usuwa cechy z korelacją Spearmana ρ > 0.85 między sobą
    (zachowuje tę z wyższą korelacją z etykietą)
  • Wynik: ~119 cech
       │
       ▼
Stage 2 — Mutual Information ranking (top 60)
  • sklearn mutual_info_classif, n_neighbors=5
  • Zachowuje 60 cech z najwyższym MI względem etykiety
       │
       ▼
Stage 3 — Walk-forward stability (≥50% okien)
  • Dzieli dane treningowe na nakładające się okna
  • W każdym oknie oblicza MI i wybiera top-K cech
  • Zachowuje cechy, które pojawiają się w ≥50% okien
  • KLUCZOWY etap: spurious correlations są niestabilne w czasie
  • Wynik: ~41 cech
       │
       ▼
Stage 4 — Permutation importance pruning
  • Trenuje LGBM na split 2021-2022 / val 2023
  • Permutation importance (10 powtórzeń, scoring=roc_auc)
  • Zachowuje cechy z importance_mean > 0.0005
  • Wynik: ~20 cech
```

### Parametry pipeline'u (v5)

| Parametr | Wartość |
|----------|---------|
| `VAR_THRESHOLD` | `1e-6` |
| `CORR_THRESHOLD` | `0.85` (Spearman) |
| `TOP_K_MI` | `60` |
| `MIN_STABLE_FRAC` | `0.50` |
| `PERM_THRESHOLD` | `0.0005` |
| Train dla Stage 4 | 2021-01-01 – 2022-12-31 |
| Val dla Stage 4 | 2023-01-01 – 2023-12-31 |
| OOS start | 2024-01-01 |

---

## Wybrane cechy — notebook 02_lgbm_omni_0fee_v5

**Data eksperymentu:** 2026-05-29  
**Artifacts:** `artifacts/02_lgbm_omni_0fee_v5/results.json`

### Wyniki selekcji

| Etap | Liczba cech | Uwagi |
|------|-------------|-------|
| Stage 0 (Pool) | 216 | 195 V1 + 21 V3 |
| Stage 1 (Variance+Corr) | 119 | |
| Stage 2 (MI Top-60) | 60 | |
| Stage 3 (Stability) | 41 | |
| Stage 4 (Perm prune) | **20** | finalny zestaw |

### 20 wybranych cech

| # | Cecha | Grupa V1 | Zewnętrzna? |
|---|-------|----------|-------------|
| 1 | `stoch_k_14` | oscillators | |
| 2 | `cross_altcoin_breadth_24h` | — | **V3** |
| 3 | `ret_2h` | returns | |
| 4 | `ad_z_48h` | volume_profile | |
| 5 | `macd_hist_5_13` | macd | |
| 6 | `bear_streak` | candlestick_patterns | |
| 7 | `ret_1h` | returns | |
| 8 | `close_vs_sma_7` | ma_ratios | |
| 9 | `ret_3h` | returns | |
| 10 | `rsi_divergence` | divergences | |
| 11 | `hour_cos` | calendar | |
| 12 | `candle_body` | candle_structure | |
| 13 | `close_vs_sma_720` | long_cycle_ma | |
| 14 | `vw_rsi_14` | volume_profile | |
| 15 | `atr_14_pct_rank` | volatility_regime | |
| 16 | `macd_divergence` | divergences | |
| 17 | `hl_position_48h` | price_position | |
| 18 | `vol_ratio_72h` | volume | |
| 19 | `close_vs_s1` | support_resistance | |
| 20 | `skew_24h` | statistical | |

**Jedyna cecha V3 w finalnym zbiorze:** `cross_altcoin_breadth_24h`
(% altcoinów z dodatnim zwrotem 24h — wskaźnik szerokości rynku altcoinów).

### Wyniki backtestowe (WFO EXP, OOS 2024-01-01+)

| Scenariusz | Threshold | Trady | Win rate | Total return | Sharpe | Max DD |
|------------|-----------|-------|----------|--------------|--------|--------|
| Bez opłat | 0.55 | 1021 | 45.2% | +54.6% | 0.63 | -23.9% |
| Bez opłat | 0.60 | 507 | 47.3% | +70.0% | 1.11 | -17.6% |
| Z opłatami (maker=0.02%, taker=0.05%) | 0.55 | 1021 | 45.1% | **-25.1%** | -0.42 | -43.7% |
| Z opłatami | 0.60 | 507 | 46.7% | **+19.4%** | 0.37 | -22.8% |

**Wniosek:** Model osiąga pozytywną stopę zwrotu bez opłat (threshold 0.6:
+70%). Z opłatami przy threshold 0.6 wynik nadal pozytywny (+19.4%), ale
Sharpe niski (0.37). Kluczowy problem: opłaty transakcyjne pochłaniają
większość zysku modelu przy niskim thresholdzie.

---

## Historia wersji

| Wersja | Notebook | Cechy wejściowe | Uwagi |
|--------|----------|-----------------|-------|
| Early agents | `src/hmats/data/features.py` | 12 cech OHLCV | NEAT/PPO, brak WFO |
| V1 | `data/features/feature_registry.json` | 196 cech OHLCV 1h | Pełny zestaw techniczny |
| V3 | `05_feature_selection_lgbm_v1.py` | V1 + 21 cech zewnętrznych | Cross-asset, sentiment, microstructure |
| V2 (5m, exp.) | `feature_registry_v2.json` | 39 cech strukturalnych 5m | MTF, liquidity, struktura swingów |
