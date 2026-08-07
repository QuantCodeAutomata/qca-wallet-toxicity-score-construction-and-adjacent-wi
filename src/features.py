"""100 ms order-book feature grid and forward-return target construction."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_datetime_series(col: pd.Series) -> pd.Series:
    """Convert a column of unix-float seconds or strings to pd.Timestamp."""
    if pd.api.types.is_float_dtype(col) or pd.api.types.is_integer_dtype(col):
        return pd.to_datetime(col, unit="s", utc=False)
    return pd.to_datetime(col)


def _prep_block_states(block_states: pd.DataFrame, coin: str) -> pd.DataFrame:
    """Return coin-filtered, datetime-indexed copy of block_states, sorted by time."""
    market_col = "market" if "market" in block_states.columns else "coin"
    bs = block_states[block_states[market_col] == coin].copy()
    bs["_ts"] = _to_datetime_series(bs["timestamp"])
    bs = bs.sort_values("_ts").set_index("_ts")
    bs.index.name = "timestamp"
    # Normalise column aliases
    if "midpoint" in bs.columns and "mid" not in bs.columns:
        bs["mid"] = bs["midpoint"]
    if "mid" in bs.columns and "midpoint" not in bs.columns:
        bs["midpoint"] = bs["mid"]
    return bs


def _prep_orders(orders: pd.DataFrame, coin: str) -> pd.DataFrame:
    """Return coin-filtered, datetime-indexed orders with signed_notional."""
    market_col = "market" if "market" in orders.columns else "coin"
    ts_col = "event_time" if "event_time" in orders.columns else "timestamp"
    ord_ = orders[orders[market_col] == coin].copy()
    ord_["_ts"] = _to_datetime_series(ord_[ts_col])
    ord_ = ord_.sort_values("_ts").set_index("_ts")
    ord_.index.name = "timestamp"
    ord_["signed_notional"] = ord_["direction"].astype(float) * ord_["notional"]
    return ord_


def _reindex_to_grid(series: pd.Series, grid: pd.DatetimeIndex) -> pd.Series:
    """Forward-fill *series* onto *grid*, returning a Series indexed by grid."""
    combined = series.reindex(series.index.union(grid)).sort_index().ffill()
    return combined.reindex(grid)


def _completed_bar_feature(
    base: pd.Series, grid: pd.DatetimeIndex, bar_freq: str
) -> pd.Series:
    """Compute a 'completed-bar' feature by resampling then forward-filling to grid.

    At each grid point the value reflects the last *fully completed* bar.
    """
    resampled = base.resample(bar_freq).last()
    return _reindex_to_grid(resampled, grid)


def _grid_period(grid: pd.DatetimeIndex) -> pd.Timedelta:
    """Return the spacing of *grid*, defaulting to 100 ms for single-point grids."""
    if len(grid) >= 2:
        return grid[1] - grid[0]
    return pd.Timedelta("100ms")


def _rolling_flow(
    orders: pd.DataFrame, grid: pd.DatetimeIndex, window_str: str
) -> pd.Series:
    """Signed notional flow summed over *window_str* rolling windows at grid points.

    Uses the grid's own period as the computation resolution so that the
    internal index never exceeds the number of grid points regardless of
    the absolute time range spanned.  For 100 ms grids this preserves the
    original sub-second semantics; for coarser grids the window snaps to the
    nearest integer multiple of the grid period (minimum 1 period).
    """
    if orders.empty or len(grid) == 0:
        return pd.Series(0.0, index=grid, dtype=float)

    period = _grid_period(grid)
    period_ms = max(1, int(period.total_seconds() * 1000))
    freq_str = f"{period_ms}ms"

    sn = orders["signed_notional"].copy()
    idx = pd.date_range(grid[0], grid[-1], freq=freq_str)
    sn_binned = sn.resample(freq_str).sum().reindex(idx, fill_value=0.0)
    n_periods = max(1, int(pd.Timedelta(window_str) / period))
    flow = sn_binned.rolling(n_periods, min_periods=1).sum()
    combined = flow.reindex(flow.index.union(grid)).sort_index()
    return combined.ffill().reindex(grid).fillna(0.0)


def _rolling_breadth(
    orders: pd.DataFrame, grid: pd.DatetimeIndex, window_str: str
) -> pd.Series:
    """Net distinct-buyer minus distinct-seller count per rolling window.

    Vectorized: resamples buy- and sell-side indicators separately to avoid
    the slow per-bin ``apply`` path.
    """
    if orders.empty or len(grid) == 0:
        return pd.Series(0.0, index=grid, dtype=float)

    period = _grid_period(grid)
    period_ms = max(1, int(period.total_seconds() * 1000))
    freq_str = f"{period_ms}ms"

    sn = orders["signed_notional"]
    idx = pd.date_range(grid[0], grid[-1], freq=freq_str)

    # Count distinct buy and sell events per bin using vectorized resample.sum
    buy_count = (sn > 0).astype(float).resample(freq_str).sum().reindex(idx, fill_value=0.0)
    sell_count = (sn < 0).astype(float).resample(freq_str).sum().reindex(idx, fill_value=0.0)
    breadth_binned = buy_count - sell_count

    n_periods = max(1, int(pd.Timedelta(window_str) / period))
    breadth = breadth_binned.rolling(n_periods, min_periods=1).sum()
    combined = breadth.reindex(breadth.index.union(grid)).sort_index()
    return combined.ffill().reindex(grid).fillna(0.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_100ms_grid(
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    freq_ms: int = 100,
) -> pd.DatetimeIndex:
    """Build a regular timestamp grid between *start_time* and *end_time*.

    Parameters
    ----------
    start_time, end_time:
        Boundary timestamps (start inclusive, end exclusive).
    freq_ms:
        Grid spacing in milliseconds (default 100).

    Returns
    -------
    pd.DatetimeIndex with *freq_ms* ms spacing.
    """
    return pd.date_range(
        start=start_time, end=end_time, freq=f"{freq_ms}ms", inclusive="left"
    )


def build_targets(
    grid: pd.DatetimeIndex,
    block_states: pd.DataFrame,
    coin: str,
    horizons_s: List[float],
) -> pd.DataFrame:
    """Compute forward mid-price returns (in basis points) at each horizon.

    Returns ``(mid_{t+h} - mid_t) / mid_t * 10_000``.  Observations whose
    forward timestamp crosses a UTC calendar-day boundary are set to NaN.

    Parameters
    ----------
    grid:
        Evaluation timestamps (pd.DatetimeIndex, tz-naive UTC).
    block_states:
        LOB state DataFrame with ``timestamp`` (float unix s or datetime),
        ``midpoint`` / ``mid``, and ``market`` / ``coin`` columns.
    coin:
        Market symbol used to filter *block_states*.
    horizons_s:
        Forward horizons in seconds.

    Returns
    -------
    pd.DataFrame indexed by *grid*, one column per horizon (``'<h>s'``).
    """
    bs = _prep_block_states(block_states, coin)
    mid = bs["midpoint"]

    # Reindex midpoint onto the 100ms grid
    mid_grid = _reindex_to_grid(mid, grid)

    result: Dict[str, np.ndarray] = {}
    for h in horizons_s:
        n_steps = int(round(h * 1000 / 100))  # number of 100ms steps
        col = f"{h}s"
        mid_fwd = mid_grid.shift(-n_steps)
        ret = (mid_fwd - mid_grid) / mid_grid.replace(0, np.nan) * 10_000

        # NaN out returns that cross a calendar-day boundary
        day_now = grid.normalize()
        day_fwd = (grid + pd.Timedelta(seconds=h)).normalize()
        cross_day = day_now != day_fwd
        ret[cross_day] = np.nan
        result[col] = ret.values

    return pd.DataFrame(result, index=grid)


def build_anonymous_features(
    grid: pd.DatetimeIndex,
    block_states: pd.DataFrame,
    orders: pd.DataFrame,
    coin: str,
) -> pd.DataFrame:
    """Build anonymous (non-identity) order-book features on a 100 ms grid.

    Parameters
    ----------
    grid:
        Regular 100 ms timestamp grid.
    block_states:
        LOB state snapshots with ``timestamp``, ``market``, ``midpoint``,
        ``spread``, ``best_bid_qty``, ``best_ask_qty`` columns.
    orders:
        Aggressive-order events with ``event_time`` / ``timestamp``,
        ``market``, ``direction``, ``notional`` columns.
    coin:
        Market symbol to filter on.

    Returns
    -------
    pd.DataFrame indexed by *grid* with columns:
    imb, nearImb, ofi1s, ofi30s, print\\_, flow1s, flow30s,
    ret1m, vol1m, spread
    """
    bs = _prep_block_states(block_states, coin)
    ord_ = _prep_orders(orders, coin)

    # ── Depth-based features ────────────────────────────────────────────────
    bid_qty = _reindex_to_grid(bs["best_bid_qty"], grid)
    ask_qty = _reindex_to_grid(bs["best_ask_qty"], grid)
    mid_grid = _reindex_to_grid(bs["midpoint"], grid)
    spread_abs = _reindex_to_grid(bs["spread"], grid)

    denom = (bid_qty + ask_qty).replace(0, np.nan)
    imb = (bid_qty - ask_qty) / denom

    # nearImb: simplified as imb scaled by an estimate of 1-day intraday vol
    # (using 1-min realised vol from the current grid)
    ret_1s = np.log(mid_grid / mid_grid.shift(10).replace(0, np.nan))
    vol1m = ret_1s.rolling(600, min_periods=10).std()
    vol_scale = vol1m.fillna(vol1m.median() if vol1m.notna().any() else 0.02)
    nearImb = imb * vol_scale.clip(1e-6, 0.1)

    spread_bps = (spread_abs / mid_grid.replace(0, np.nan) * 10_000)

    # ── Return / volatility features ────────────────────────────────────────
    ret1m = np.log(mid_grid / mid_grid.shift(600).replace(0, np.nan)) * 10_000

    # ── OFI: completed-bar depth-change imbalance ────────────────────────────
    # At each completed 1-s bar: Δbid_qty - Δask_qty
    bid_1s = _completed_bar_feature(bs["best_bid_qty"], grid, "1s")
    ask_1s = _completed_bar_feature(bs["best_ask_qty"], grid, "1s")
    bid_1s_lag = bid_1s.shift(10)   # 10 * 100ms = 1s ago (prev completed bar)
    ask_1s_lag = ask_1s.shift(10)
    ofi1s = (bid_1s - bid_1s_lag) - (ask_1s - ask_1s_lag)

    bid_30s = _completed_bar_feature(bs["best_bid_qty"], grid, "30s")
    ask_30s = _completed_bar_feature(bs["best_ask_qty"], grid, "30s")
    bid_30s_lag = bid_30s.shift(300)
    ask_30s_lag = ask_30s.shift(300)
    ofi30s = (bid_30s - bid_30s_lag) - (ask_30s - ask_30s_lag)

    # ── Order-flow features ──────────────────────────────────────────────────
    flow1s = _rolling_flow(ord_, grid, "1s")
    flow30s = _rolling_flow(ord_, grid, "30s")

    # print_: signed notional of the most recent aggressive order
    if not ord_.empty:
        print_ = _reindex_to_grid(ord_["signed_notional"], grid).fillna(0.0)
    else:
        print_ = pd.Series(0.0, index=grid, dtype=float)

    return pd.DataFrame(
        {
            "imb": imb,
            "nearImb": nearImb,
            "ofi1s": ofi1s,
            "ofi30s": ofi30s,
            "print_": print_,
            "flow1s": flow1s,
            "flow30s": flow30s,
            "ret1m": ret1m,
            "vol1m": vol1m,
            "spread": spread_bps,
        },
        index=grid,
    )


def build_identity_features(
    grid: pd.DatetimeIndex,
    block_states: pd.DataFrame,
    orders: pd.DataFrame,
    cohort: List[str],
    coin: str,
) -> pd.DataFrame:
    """Build identity-twin features restricted to the toxic cohort.

    Parameters
    ----------
    grid:
        Regular 100 ms timestamp grid.
    block_states:
        LOB state snapshots.
    orders:
        Full aggressive-order stream (all wallets).
    cohort:
        List of toxic wallet identifiers.
    coin:
        Market symbol to filter on.

    Returns
    -------
    pd.DataFrame indexed by *grid* with columns:
    hotImb, hotNear, hotOfi1s, hotOfi30s, hotPrint,
    hotFlow1s, hotFlow30s, hotBreadth1s, hotBreadth30s,
    hotShare, hotContrib
    """
    bs = _prep_block_states(block_states, coin)
    all_ord = _prep_orders(orders, coin)

    wallet_col = "wallet" if "wallet" in all_ord.columns else None
    if wallet_col and not all_ord.empty:
        hot_ord = all_ord[all_ord[wallet_col].isin(cohort)]
    else:
        hot_ord = all_ord.iloc[:0].copy()

    # Cohort-restricted flow features
    hotFlow1s = _rolling_flow(hot_ord, grid, "1s")
    hotFlow30s = _rolling_flow(hot_ord, grid, "30s")

    # Breadth: net distinct buyers minus sellers
    hotBreadth1s = _rolling_breadth(hot_ord, grid, "1s")
    hotBreadth30s = _rolling_breadth(hot_ord, grid, "30s")

    # OFI restricted to cohort (approximated via order flow changes)
    if not hot_ord.empty:
        sn_1s = hot_ord["signed_notional"].resample("1s").sum()
        hotOfi1s_1s = sn_1s - sn_1s.shift(1)
        hotOfi1s = _reindex_to_grid(hotOfi1s_1s, grid).fillna(0.0)

        sn_30s = hot_ord["signed_notional"].resample("30s").sum()
        hotOfi30s_30s = sn_30s - sn_30s.shift(1)
        hotOfi30s = _reindex_to_grid(hotOfi30s_30s, grid).fillna(0.0)

        hotPrint = _reindex_to_grid(hot_ord["signed_notional"], grid).fillna(0.0)
    else:
        hotOfi1s = pd.Series(0.0, index=grid, dtype=float)
        hotOfi30s = pd.Series(0.0, index=grid, dtype=float)
        hotPrint = pd.Series(0.0, index=grid, dtype=float)

    # hotImb, hotNear: use proportion of flow vs total depth as proxy
    bid_qty = _reindex_to_grid(bs["best_bid_qty"], grid)
    ask_qty = _reindex_to_grid(bs["best_ask_qty"], grid)
    total_depth = (bid_qty + ask_qty).replace(0, np.nan)

    # hotShare: fraction of best-quote depth attributable to cohort (approx),
    # proxied by rolling cohort flow fraction of total flow
    total_flow1s = _rolling_flow(all_ord, grid, "1s").abs().replace(0, np.nan)
    hot_flow1s_abs = hotFlow1s.abs()
    hotShare = (hot_flow1s_abs / total_flow1s).clip(-1.0, 1.0).fillna(0.0)

    # hotContrib: toxic cohort's imbalance contribution scaled by total depth
    hotContrib = (hotFlow1s / total_depth).clip(-1.0, 1.0).fillna(0.0)

    hotImb = hotContrib.copy()
    hotNear = hotImb * 0.5  # simplified near-touch version

    return pd.DataFrame(
        {
            "hotImb": hotImb,
            "hotNear": hotNear,
            "hotOfi1s": hotOfi1s,
            "hotOfi30s": hotOfi30s,
            "hotPrint": hotPrint,
            "hotFlow1s": hotFlow1s,
            "hotFlow30s": hotFlow30s,
            "hotBreadth1s": hotBreadth1s,
            "hotBreadth30s": hotBreadth30s,
            "hotShare": hotShare,
            "hotContrib": hotContrib,
        },
        index=grid,
    )


def build_feature_matrix(
    grid: pd.DatetimeIndex,
    block_states: pd.DataFrame,
    orders: pd.DataFrame,
    cohort: List[str],
    coins: List[str],
    horizons_s: List[float],
    embargo_ms: int = 0,
) -> Dict[str, pd.DataFrame]:
    """Build pooled feature matrices for multiple coins.

    For each coin, anonymous features, identity features, and forward-return
    targets are computed on *grid*, then combined.  An optional embargo drops
    the first *embargo_ms* milliseconds after every calendar-day boundary to
    prevent feature-window leakage.

    Parameters
    ----------
    grid:
        Regular 100 ms timestamp index covering the desired window.
    block_states:
        LOB snapshots (all coins pooled).
    orders:
        Aggressive orders (all coins pooled).
    cohort:
        Toxic wallet identifiers for identity features.
    coins:
        Markets to include.
    horizons_s:
        Forward return horizons in seconds.
    embargo_ms:
        Milliseconds to drop from the start of each UTC day (default 0).

    Returns
    -------
    Dict mapping coin symbol to a pd.DataFrame with all features and targets,
    plus a ``'coin'`` column and a ``'date'`` column.
    """
    result: Dict[str, pd.DataFrame] = {}

    embargo_td = pd.Timedelta(milliseconds=embargo_ms)

    for coin in coins:
        anon = build_anonymous_features(grid, block_states, orders, coin)
        iden = build_identity_features(grid, block_states, orders, cohort, coin)
        tgts = build_targets(grid, block_states, coin, horizons_s)

        df = pd.concat([anon, iden, tgts], axis=1)
        df["coin"] = coin
        df["date"] = df.index.normalize()

        # Apply embargo: drop the first embargo_ms ms of each day
        if embargo_ms > 0:
            day_start = df.index.normalize()
            time_into_day = df.index - day_start
            df = df[time_into_day >= embargo_td]

        result[coin] = df

    return result
