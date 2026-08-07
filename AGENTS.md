# Repository Knowledge — Hyperliquid Wallet Toxicity Analysis

## Project Purpose
Research repository for detecting and measuring "toxic" (informed) order flow on
Hyperliquid perp markets.  A wallet is *toxic* if its aggressive orders
consistently generate positive direction-signed markouts (it moves the price
against passive liquidity providers).

## Directory Layout
```
src/                 Core Python modules (see below)
data/                Raw / processed market data (gitignored large files)
exp/                 Experiment notebooks / scripts
tests/               pytest test suite
results/             Output tables, figures
```

## Core Module Map (`src/`)

| File | Purpose |
|---|---|
| `data_generator.py` | Synthetic Level-4 data for tests (block states, oracle ticks, orders, fills) |
| `oracle_gate.py`    | `OracleGate` state-machine — requires 12 consecutive in-threshold ticks to open |
| `replay_engine.py`  | `LimitOrderBook` + `ReplayEngine` — LOB reconstruction with zero-corruption policy |
| `fills.py`          | `pair_fills`, `group_aggressive_orders`, `filter_qualifying_orders` |
| `wallet_scoring.py` | `compute_markout` (single order, bps), `compute_wallet_scores`, `get_toxic_cohort` |
| `persistence.py`    | Cross-window analysis: Spearman rank-correlation, decile markouts, peer adjustment |
| `features.py`       | 100 ms order-book feature grid + forward-return target construction |
| `models.py`         | OOS R², daily MSE, paired t-stat model evaluation helpers |
| `placebo.py`        | Matched-sample placebo cohort construction and rank p-value utilities |
| `payoff.py`         | Payoff-unit diagnostic: `compute_payoff_statistic` (bps), budget thresholds, `payoff_table` |

## Experiment Runner

`exp/run_all.py` is the master script — run with `python exp/run_all.py` from the project root.  It:
- Generates synthetic July 2026 (EXP1–5, 7) and December 2025 (EXP6) data.
- Runs EXP1–7 sequentially, saves CSV + PNG to `results/` and appends a Markdown
  summary to `results/RESULTS.md`.
- Full run takes ~5 minutes on a standard workstation (bottleneck: wallet scoring
  via per-order `compute_markout` loop).

Key constants in `run_all.py`:
- `GRID_FREQ_MS = 10_000` (10 s grid; change to 100 for 100 ms resolution, 100× slower)
- `N_WALLETS = 80`, `MIN_ORDERS = 50`
- `HORIZONS_S = [1.0, 5.0, 30.0]`

## Key Design Decisions

### fills.py
- Raw fills have an `is_taker` boolean column (True = aggressive taker).
- `pair_fills` joins on `(day, market, trade_id)` — the day key prevents
  spurious cross-midnight matches when trade IDs wrap.
- `group_aggressive_orders` takes the *paired* fills (output of `pair_fills`)
  and collapses partial fills by `order_id`.

### wallet_scoring.py
- `compute_markout(order: pd.Series, block_states: pd.DataFrame, ...)` works
  on a **single** order row and returns `Optional[float]`.
- Uses `np.searchsorted` (binary search) for O(log n) midpoint lookups.
- `compute_wallet_scores` pre-groups block_states by market before the inner
  loop to avoid O(n) filter on every order call.

### replay_engine.py
- Zero-corruption policy: reset book + oracle gate on block rewind, crossed
  book, or negative depth.  After reset, emit no state until gate reopens.
- Block envelope format:
  ```json
  {"block_number": int, "timestamp": float, "market": str,
   "updates": [{"side": "bid"|"ask", "price": float, "qty": float}],
   "oracle_price": float}
  ```

### oracle_gate.py
- Constructor params: `required_streak` (not `streak_required`) and
  `gap_seconds` (not `max_gap_s`).
- `update()` returns `bool` (gate open?).
- An out-of-threshold tick **closes** the gate (streak reset + `_open=False`).

### persistence.py
- `compute_peer_adjusted_scores` uses efficient group-total algebra to avoid
  O(n²) per-pair comparisons — group by (market, minute), subtract self
  contribution, re-aggregate by wallet.
- `compute_decile_markouts` accepts frozen t1 scores for bucketing and
  computes fresh markouts on t2 orders; returns 95 % CI via normal
  approximation on weighted variance.

## Date Windows Used in Experiments
- **t1 (training)**: December 1–31 2025
- **t2 (out-of-sample)**: July 1–20 2026

## Performance Notes

### Wallet Scoring
- `generate_block_states` over 10 days × 2 coins at 2 s resolution = 864 K rows.
- `compute_wallet_scores` passes the full block-state DataFrame to every
  `compute_markout` call, which internally re-filters by market (O(n) per call).
  **Fix**: downsample block states to every 15th row (30 s resolution) before
  passing to scoring; reduces runtime from ~∞ to ~32 s for 80 wallets / 10 days.
  See `_downsample_bs(bs, factor=15)` in `exp/run_all.py`.
- `compute_decile_markouts` and `compute_peer_adjusted_scores` pre-group by
  market internally so they are faster — pass the downsampled block states too.

### Feature Engineering
- `_rolling_breadth` used to call `.resample().apply(lambda x: buy-sell)` which
  is O(n_bins) Python-level iterations.  Now uses two vectorized `.resample().sum()`
  calls on boolean masks — **21× speedup** (41 s → 1.9 s for a 10-day / 2-coin build).
- `_rolling_flow` and `_rolling_breadth` compute at the grid's own period
  (controlled by `GRID_FREQ_MS`), never creating an internal 100 ms array longer
  than the grid itself.  At 10 s grid the internal array has ≤ 86 400 rows.
- Use `GRID_FREQ_MS = 10_000` (10 s) for experiments and `GRID_FREQ_MS = 100`
  for production-fidelity; the feature semantics snap to the nearest multiple of
  the grid period (minimum 1 period).

## Testing
```bash
pytest tests/
```
All tests should pass with the current `src/` implementation.
