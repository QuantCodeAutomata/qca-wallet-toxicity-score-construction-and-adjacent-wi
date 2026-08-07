"""Per-wallet notional-weighted markout score computation."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from .fills import filter_qualifying_orders


def _to_unix_seconds(val) -> float:
    """Convert a timestamp value to Unix seconds (float).

    Accepts ``float``/``int`` (already Unix seconds), ``pd.Timestamp``,
    and ``numpy.datetime64``.
    """
    if isinstance(val, (float, int, np.floating, np.integer)):
        return float(val)
    ts = pd.Timestamp(val)
    return ts.timestamp()


def _ts_array_to_unix_seconds(arr: np.ndarray) -> np.ndarray:
    """Convert a numpy array of timestamps to float Unix-second values."""
    if np.issubdtype(arr.dtype, np.datetime64):
        # datetime64[ns] → float seconds
        return arr.astype("datetime64[s]").astype(np.float64)
    return arr.astype(np.float64)


def compute_markout(
    order: pd.Series,
    block_states: pd.DataFrame,
    markout_horizon_s: float = 10.0,
    pre_staleness_s: float = 60.0,
    post_staleness_s: float = 120.0,
) -> Optional[float]:
    """Compute the direction-signed notional-weighted markout for a single order.

    The markout is defined as::

        x_e = 10 000 * direction * (m_post - m_pre) / m_pre

    where:

    * ``m_pre``  – midpoint of the last block **strictly before** the order's
      ``event_time``.  Returns ``None`` if that block is more than
      ``pre_staleness_s`` seconds old.
    * ``m_post`` – midpoint of the latest block at or before
      ``event_time + markout_horizon_s``.  Returns ``None`` if that block is
      more than ``post_staleness_s`` seconds old or if no such block exists.

    Parameters
    ----------
    order:
        A single row from an orders DataFrame; must contain ``event_time``
        (float Unix seconds), ``direction`` (+1 or -1), and ``market``.
    block_states:
        Block-close state DataFrame with columns ``timestamp``, ``market``,
        ``midpoint``.  Need not be sorted in advance.
    markout_horizon_s:
        Forward horizon in seconds (default 10 s).
    pre_staleness_s:
        Maximum age in seconds for ``m_pre`` before the markout is voided
        (default 60 s).
    post_staleness_s:
        Maximum staleness in seconds for ``m_post`` (default 120 s).

    Returns
    -------
    Float markout in basis points, or ``None`` when either reference
    midpoint is unavailable.
    """
    market: str = order["market"]
    event_time: float = _to_unix_seconds(order["event_time"])
    direction: int = int(order["direction"])

    mkt_states = block_states[block_states["market"] == market]
    if mkt_states.empty:
        return None

    # Sort only when necessary; callers may pass pre-sorted data.
    if not mkt_states["timestamp"].is_monotonic_increasing:
        mkt_states = mkt_states.sort_values("timestamp")

    # Normalise timestamps to float Unix seconds for arithmetic.
    ts = _ts_array_to_unix_seconds(mkt_states["timestamp"].values)
    mids = mkt_states["midpoint"].values

    # ── m_pre: last block STRICTLY before event_time (binary search) ────
    pre_idx = int(np.searchsorted(ts, event_time, side="left")) - 1
    if pre_idx < 0:
        return None
    pre_ts = float(ts[pre_idx])
    if event_time - pre_ts > pre_staleness_s:
        return None
    m_pre = float(mids[pre_idx])

    # ── m_post: latest block at or before event_time + horizon ──────────
    post_deadline = event_time + markout_horizon_s
    post_idx = int(np.searchsorted(ts, post_deadline, side="right")) - 1
    if post_idx < 0:
        return None
    post_ts = float(ts[post_idx])
    if post_deadline - post_ts > post_staleness_s:
        return None
    m_post = float(mids[post_idx])

    if m_pre == 0.0:
        return None

    return 10_000.0 * direction * (m_post - m_pre) / m_pre


def compute_wallet_scores(
    orders: pd.DataFrame,
    block_states: pd.DataFrame,
    min_orders: int = 100,
) -> pd.DataFrame:
    """Compute the notional-weighted markout score ``alpha_w`` for every wallet.

    Only qualifying orders (non-TWAP, non-liquidation) with a valid markout are
    counted.  Wallets with fewer than ``min_orders`` valid orders are excluded.

    ``alpha_w = sum(n_e * x_e) / sum(n_e)``

    Parameters
    ----------
    orders:
        Aggressive orders DataFrame with columns ``wallet``, ``market``,
        ``direction``, ``notional``, ``event_time``, ``is_twap``,
        ``is_liquidation``.
    block_states:
        Block-close state DataFrame with columns ``timestamp``, ``market``,
        ``midpoint``.
    min_orders:
        Minimum number of valid orders for a wallet to appear in the output.

    Returns
    -------
    DataFrame with columns ``wallet``, ``alpha_w``, ``order_count``,
    ``total_notional``, ``rank`` (1 = most toxic).
    """
    qualifying = filter_qualifying_orders(orders)
    if qualifying.empty:
        return pd.DataFrame(
            columns=["wallet", "alpha_w", "order_count", "total_notional", "rank"]
        )

    # Pre-sort and pre-group block states by market once for efficient lookups.
    market_states: dict[str, pd.DataFrame] = {
        mkt: grp.sort_values("timestamp").reset_index(drop=True)
        for mkt, grp in block_states.groupby("market")
    }

    records: list[dict] = []
    for wallet, wallet_orders in qualifying.groupby("wallet"):
        markouts: list[float] = []
        notionals: list[float] = []
        for _, order in wallet_orders.iterrows():
            mkt_bs = market_states.get(str(order["market"]), pd.DataFrame())
            x_e = compute_markout(order, mkt_bs)
            if x_e is not None:
                markouts.append(x_e)
                notionals.append(float(order["notional"]))

        if len(markouts) < min_orders:
            continue

        n_arr = np.array(notionals)
        x_arr = np.array(markouts)
        alpha_w = float(np.sum(n_arr * x_arr) / np.sum(n_arr))

        records.append(
            {
                "wallet": wallet,
                "alpha_w": alpha_w,
                "order_count": len(markouts),
                "total_notional": float(np.sum(n_arr)),
            }
        )

    df = pd.DataFrame(records)
    if df.empty:
        df["rank"] = pd.Series(dtype=int)
        return df

    df = df.sort_values("alpha_w", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df


def get_toxic_cohort(
    scores: pd.DataFrame,
    top_fraction: float = 0.10,
) -> List[str]:
    """Return wallet identifiers in the top ``top_fraction`` by ``alpha_w``.

    Parameters
    ----------
    scores:
        Output of :func:`compute_wallet_scores`; must have ``wallet`` and
        ``alpha_w`` columns.
    top_fraction:
        Fraction of wallets to include (default 0.1 = top decile).

    Returns
    -------
    List of wallet identifiers sorted by descending ``alpha_w``.
    """
    k = max(1, int(len(scores) * top_fraction))
    return scores.nlargest(k, "alpha_w")["wallet"].tolist()
