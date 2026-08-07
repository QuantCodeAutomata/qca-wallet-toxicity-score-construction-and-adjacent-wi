"""Synthetic Hyperliquid-like Level-4 data generator for testing and experimentation.

Generates correlated block-close states, oracle ticks, aggressive order flow, and
paired fill records across BTC, ETH, and SOL for two configurable date windows.

Toxic cohorts are seeded with informed directional bias so that their orders exhibit
consistently positive notional-weighted markouts, enabling end-to-end pipeline tests.
"""

from __future__ import annotations

import hashlib
from typing import List

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Market constants
# ---------------------------------------------------------------------------

MARKETS: List[str] = ["BTC", "ETH", "SOL"]

_BASE_PRICES: dict[str, float] = {"BTC": 95_000.0, "ETH": 3_500.0, "SOL": 180.0}
_ANNUAL_VOL: dict[str, float] = {"BTC": 0.60, "ETH": 0.70, "SOL": 0.80}
_SPREAD_BPS: dict[str, float] = {"BTC": 1.5, "ETH": 2.0, "SOL": 3.0}

# Block interval (seconds) - Hyperliquid produces ~2 s blocks
_BLOCK_DT: float = 2.0

# Number of toxic wallets per cohort
_N_TOXIC_PER_COHORT: int = 5

# Lookahead for toxic-wallet signal (blocks = 5 × 2 s = 10 s horizon)
_LOOKAHEAD_BLOCKS: int = 5

# Minimum absolute future return to act on (noise filter)
_SIGNAL_THRESHOLD: float = 5e-5


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _wallet_id(idx: int) -> str:
    """Return a deterministic 42-char hex wallet address from an integer index."""
    digest = hashlib.md5(f"wallet_{idx}".encode()).hexdigest()
    return f"0x{digest[:40]}"


def _gbm_prices(
    base: float,
    annual_vol: float,
    n: int,
    dt: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate a Geometric Brownian Motion price series of length *n*."""
    per_step_vol = annual_vol / np.sqrt(252 * 86_400 / dt)
    log_returns = rng.normal(0.0, per_step_vol, n)
    return base * np.exp(np.cumsum(np.concatenate([[0.0], log_returns])))


def _date_range_seconds(start_date: str, end_date: str) -> tuple[float, float, int]:
    """Return (start_ts, end_ts, total_blocks) for a date pair."""
    start_ts = pd.Timestamp(start_date).timestamp()
    end_ts = pd.Timestamp(end_date).timestamp()
    total_blocks = int((end_ts - start_ts) / _BLOCK_DT)
    return start_ts, end_ts, total_blocks


# ---------------------------------------------------------------------------
# Public generators
# ---------------------------------------------------------------------------


def generate_block_states(
    start_date: str,
    end_date: str,
    markets: List[str],
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic block-close mid-price and spread snapshots.

    Each row represents the state captured at the close of a ~2-second block.

    Parameters
    ----------
    start_date:
        Inclusive start date string, e.g. ``'2025-12-01'``.
    end_date:
        Exclusive end date string.
    markets:
        List of market symbols to generate.
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    DataFrame with columns: timestamp, market, midpoint, spread,
    best_bid_qty, best_ask_qty.
    """
    rng = np.random.default_rng(seed)
    start_ts, _, n_blocks = _date_range_seconds(start_date, end_date)

    records: list[dict] = []
    for market in markets:
        prices = _gbm_prices(
            _BASE_PRICES[market], _ANNUAL_VOL[market], n_blocks, _BLOCK_DT, rng
        )
        spread_bps = _SPREAD_BPS[market]

        for i in range(n_blocks):
            price = prices[i]
            spread = price * spread_bps / 10_000.0
            records.append(
                {
                    "timestamp": start_ts + i * _BLOCK_DT,
                    "market": market,
                    "midpoint": price,
                    "spread": spread,
                    "best_bid_qty": float(rng.exponential(0.5)),
                    "best_ask_qty": float(rng.exponential(0.5)),
                }
            )

    return (
        pd.DataFrame(records)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def generate_oracle_ticks(
    start_date: str,
    end_date: str,
    markets: List[str],
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic oracle and mark-price ticks.

    Oracle prices lag the mark (LOB mid) by a small random amount to simulate
    real-world oracle latency.

    Parameters
    ----------
    start_date:
        Inclusive start date string.
    end_date:
        Exclusive end date string.
    markets:
        List of market symbols to generate.
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    DataFrame with columns: timestamp, market, oracle_price, mark_price.
    """
    rng = np.random.default_rng(seed + 1)
    start_ts, _, n_blocks = _date_range_seconds(start_date, end_date)

    records: list[dict] = []
    for market in markets:
        mark_prices = _gbm_prices(
            _BASE_PRICES[market], _ANNUAL_VOL[market], n_blocks, _BLOCK_DT, rng
        )
        # Oracle lags mark by up to ~10 bps with some smoothing
        lag_factor = rng.uniform(0.995, 1.005, n_blocks + 1)
        oracle_prices = mark_prices * lag_factor

        for i in range(n_blocks):
            records.append(
                {
                    "timestamp": start_ts + i * _BLOCK_DT,
                    "market": market,
                    "oracle_price": float(oracle_prices[i]),
                    "mark_price": float(mark_prices[i]),
                }
            )

    return (
        pd.DataFrame(records)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def generate_aggressive_orders(
    start_date: str,
    end_date: str,
    markets: List[str],
    n_wallets: int = 80,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate aggressive-order events across wallets, including two toxic cohorts.

    Toxic cohort 1 (wallets 0–4): predicts future 10 s price direction with ~80 %
    accuracy, producing consistently positive markouts.
    Toxic cohort 2 (wallets 5–9): ~65 % accuracy, weaker but still positive alpha.
    Remaining wallets: random direction (no edge).

    Parameters
    ----------
    start_date:
        Inclusive start date string.
    end_date:
        Exclusive end date string.
    markets:
        List of market symbols.
    n_wallets:
        Total number of wallets to simulate.
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    DataFrame with columns: order_id, wallet, market, direction, notional,
    event_time, is_twap, is_liquidation.
    """
    rng = np.random.default_rng(seed)

    wallets = [_wallet_id(i) for i in range(n_wallets)]
    toxic_1 = set(wallets[:_N_TOXIC_PER_COHORT])
    toxic_2 = set(wallets[_N_TOXIC_PER_COHORT: 2 * _N_TOXIC_PER_COHORT])

    start_ts, end_ts, n_blocks = _date_range_seconds(start_date, end_date)
    # Extra buffer so lookahead never exceeds array bounds
    extra = _LOOKAHEAD_BLOCKS + 10

    records: list[dict] = []
    order_counter = 0

    for market in markets:
        prices = _gbm_prices(
            _BASE_PRICES[market], _ANNUAL_VOL[market], n_blocks + extra, _BLOCK_DT, rng
        )

        # Future return at the 10 s lookahead horizon for each block
        future_returns = np.array(
            [
                (prices[i + _LOOKAHEAD_BLOCKS] - prices[i]) / prices[i]
                for i in range(n_blocks)
            ]
        )

        for wallet in wallets:
            # How many orders this wallet submits to this market
            n_orders = int(rng.integers(80, 220))
            # Sample event times uniformly, keep enough room for post-horizon
            event_times = np.sort(
                rng.uniform(start_ts, end_ts - _BLOCK_DT * extra, n_orders)
            )

            for t in event_times:
                block_idx = int((t - start_ts) / _BLOCK_DT)
                if block_idx >= n_blocks:
                    continue

                fr = future_returns[block_idx]
                abs_fr = abs(fr)

                if wallet in toxic_1:
                    # High-accuracy signal: follow the sign of the future return
                    if abs_fr >= _SIGNAL_THRESHOLD:
                        direction = 1 if fr > 0 else -1
                    else:
                        direction = int(rng.choice([-1, 1]))
                    # Small probability of noise trade
                    if rng.random() < 0.20:
                        direction = int(rng.choice([-1, 1]))

                elif wallet in toxic_2:
                    # Medium-accuracy signal
                    if abs_fr >= _SIGNAL_THRESHOLD and rng.random() < 0.65:
                        direction = 1 if fr > 0 else -1
                    else:
                        direction = int(rng.choice([-1, 1]))

                else:
                    direction = int(rng.choice([-1, 1]))

                notional = float(rng.exponential(12_000.0) + 1_000.0)

                records.append(
                    {
                        "order_id": f"ord_{order_counter:010d}",
                        "wallet": wallet,
                        "market": market,
                        "direction": direction,
                        "notional": notional,
                        "event_time": float(t),
                        "is_twap": bool(rng.random() < 0.04),
                        "is_liquidation": False,
                    }
                )
                order_counter += 1

    return (
        pd.DataFrame(records)
        .sort_values("event_time")
        .reset_index(drop=True)
    )


def generate_fills(
    aggressive_orders: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate paired taker and maker fill records from aggressive orders.

    Each aggressive order produces one to three partial fills; each fill gets a
    unique trade_id.  Maker wallets are drawn from a separate pool so they do
    not appear in the taker wallet set.

    Parameters
    ----------
    aggressive_orders:
        Output of :func:`generate_aggressive_orders`.
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    DataFrame with columns: fill_time, market, trade_id, order_id, wallet,
    side, notional, is_twap, is_liquidation, is_taker.
    """
    rng = np.random.default_rng(seed)

    # Dedicated market-maker wallet pool (indices 1000–1019)
    maker_pool = np.array([_wallet_id(1000 + i) for i in range(20)])

    records: list[dict] = []
    trade_counter = 0

    for _, order in aggressive_orders.iterrows():
        n_fills = int(rng.integers(1, 4))  # 1, 2, or 3 partial fills
        remaining = float(order["notional"])
        taker_side = "buy" if order["direction"] == 1 else "sell"
        maker_side = "sell" if order["direction"] == 1 else "buy"

        for f in range(n_fills):
            trade_id = f"trade_{trade_counter:012d}"
            trade_counter += 1

            if f < n_fills - 1:
                frac = float(rng.uniform(0.25, 0.75))
                fill_notional = remaining * frac
                remaining -= fill_notional
            else:
                fill_notional = remaining

            fill_time = float(order["event_time"]) + float(rng.uniform(0.0, 0.05))
            maker = str(rng.choice(maker_pool))

            # Taker fill
            records.append(
                {
                    "fill_time": fill_time,
                    "market": order["market"],
                    "trade_id": trade_id,
                    "order_id": order["order_id"],
                    "wallet": order["wallet"],
                    "side": taker_side,
                    "notional": fill_notional,
                    "is_twap": bool(order["is_twap"]),
                    "is_liquidation": bool(order["is_liquidation"]),
                    "is_taker": True,
                }
            )
            # Maker fill
            records.append(
                {
                    "fill_time": fill_time,
                    "market": order["market"],
                    "trade_id": trade_id,
                    "order_id": None,
                    "wallet": maker,
                    "side": maker_side,
                    "notional": fill_notional,
                    "is_twap": False,
                    "is_liquidation": False,
                    "is_taker": False,
                }
            )

    return (
        pd.DataFrame(records)
        .sort_values("fill_time")
        .reset_index(drop=True)
    )
