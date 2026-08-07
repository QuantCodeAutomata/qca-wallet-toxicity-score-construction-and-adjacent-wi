# Wallet Toxicity Score Construction and Adjacent-Window Persistence Test

## Description

This repository implements a research pipeline for analyzing trader toxicity on the Hyperliquid decentralised perpetuals exchange. A wallet is considered *toxic* if its aggressive (taker) orders are systematically well-timed relative to short-horizon price moves — i.e., the wallet captures positive markout consistently after order submission.

The pipeline:
1. Ingests raw Level-4 Hyperliquid fill streams (hourly zstd-compressed NDJSON).
2. Reconstructs oracle prices and gates data quality via `OracleGate`.
3. Parses, pairs, and filters fills into qualifying aggressive orders.
4. Computes notional-weighted markout scores per wallet (`alpha_w`).
5. Identifies a top-decile toxic cohort and tests whether cohort membership and scores persist across adjacent non-overlapping windows (the *adjacent-window persistence test*).
6. Validates persistence significance using a size- and activity-matched placebo test.
7. Builds a 100 ms order-book feature grid and trains mid-price forecasting models (benchmark and identity-augmented) to quantify the information content of toxic order flow.

---

## Repository Structure

```
.
├── data/
│   ├── raw/               # Raw hourly NDJSON feeds (not included; see data/README.md)
│   └── README.md          # Data acquisition instructions
├── exp/
│   └── run_all.py         # Entry point: runs all experiments end-to-end
├── results/
│   └── RESULTS.md         # Populated after running experiments
├── src/
│   ├── oracle_gate.py     # Oracle price reconstruction and data-quality gate
│   ├── fills.py           # Fill parsing, pairing, and aggressive-order extraction
│   ├── wallet_scoring.py  # Per-wallet markout score computation
│   ├── persistence.py     # Adjacent-window Spearman correlation and persistence tests
│   ├── placebo.py         # Matched-sample placebo cohort construction and rank p-value
│   ├── features.py        # 100 ms order-book feature grid and forward-return targets
│   └── models.py          # OOS R², daily MSE, paired t-stat, and relative gain utilities
├── tests/
│   ├── __init__.py
│   ├── test_oracle_gate.py
│   ├── test_fills.py
│   ├── test_wallet_scoring.py
│   ├── test_persistence.py
│   ├── test_features.py
│   ├── test_models.py
│   └── test_placebo.py
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Installation

Requires Python 3.11+.

```bash
pip install -r requirements.txt
```

It is recommended to use a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

### Run the full experiment pipeline

```bash
python exp/run_all.py
```

Results are written to `results/`.

### Run the test suite

```bash
python -m pytest tests/ -v
```

### Run a specific experiment module

```bash
python -m src.wallet_scoring   # Score wallets for a given window
python -m src.persistence      # Run adjacent-window persistence analysis
```

---

## Experiments

### Experiment 1 — Wallet Toxicity Scoring

For each trading window, aggressive (taker) orders are extracted, TWAP and liquidation fills are excluded, and a notional-weighted markout score is computed per wallet:

```
alpha_w = sum(n_e * x_e) / sum(n_e)
```

where `n_e` is the notional of order `e` and `x_e` is the direction-signed mid-price return over a fixed horizon (default 5 s) expressed in basis points.

### Experiment 2 — Adjacent-Window Persistence Test

The top-fraction (default 10 %) of wallets by `alpha_w` in window A are compared against their scores in the immediately adjacent window B. Spearman rank correlation between window-A and window-B scores is computed across the cohort, and cohort participation rate (fraction of window-A toxic wallets that also appear in window B) is recorded.

### Experiment 3 — Placebo Test

A size- and activity-matched synthetic cohort is drawn from non-toxic wallets (matched on order-count decile and notional decile). The same persistence metrics are computed for 200 placebo draws, and a rank p-value is derived:

```
p = (1 + #{placebo increments >= real increment}) / (1 + 200)
```

### Experiment 4 — Mid-price Forecasting with Toxic Flow Features

A 100 ms order-book state grid is built and forward returns at multiple horizons (1 s, 5 s, 10 s) are computed as targets. Two XGBoost models are trained and evaluated out-of-sample:

- **Benchmark**: standard order-book features (spread, depth imbalance, etc.)
- **Identity-augmented**: benchmark features plus a toxic-wallet order-flow indicator

OOS R² and a paired t-test on daily MSE are used to assess whether toxic-flow features provide incremental predictive power.

---

## Data Requirements

Real Hyperliquid Level-4 data is **not publicly available**. The raw streams consist of hourly zstd-compressed NDJSON files captured from a non-validating Hyperliquid node. See [`data/README.md`](data/README.md) for details.

All tests use synthetically generated data and do not require real market data.

---

## Methodology References

- Collin-Dufresne, P. & Fos, V. (2015). Do prices reveal the presence of informed trading? *Journal of Finance*, 70(4), 1555–1582.
- Easley, D., López de Prado, M. & O'Hara, M. (2012). Flow toxicity and liquidity in a high-frequency world. *Review of Financial Studies*, 25(5), 1457–1493.
- Brogaard, J., Hendershott, T. & Riordan, R. (2014). High-frequency trading and price discovery. *Review of Financial Studies*, 27(8), 2267–2306.
- Glosten, L. R. & Milgrom, P. R. (1985). Bid, ask and transaction prices in a specialist market with heterogeneously informed traders. *Journal of Financial Economics*, 14(1), 71–100.
