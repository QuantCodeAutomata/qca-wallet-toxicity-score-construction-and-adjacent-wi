"""Cross-window persistence analysis: Spearman correlation, decile markouts, peer-adjustment.

All functions are designed around two adjacent time windows (t1 = training,
t2 = out-of-sample).  Scores derived in t1 are used as frozen rankings;
forward performance is evaluated in t2.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .fills import filter_qualifying_orders
from .wallet_scoring import compute_markout


def spearman_correlation(
    scores_t1: pd.Series,
    scores_t2: pd.Series,
) -> Tuple[float, float]:
    """Compute Spearman rank correlation between two alpha-score Series.

    Only the intersection of indices (wallet identifiers) is used.

    Parameters
    ----------
    scores_t1:
        ``alpha_w`` values from window t1, indexed by wallet.
    scores_t2:
        ``alpha_w`` values from window t2, indexed by wallet.

    Returns
    -------
    ``(rho, p_value)`` tuple.
    """
    common = scores_t1.index.intersection(scores_t2.index)
    rho, p = stats.spearmanr(scores_t1.loc[common], scores_t2.loc[common])
    return float(rho), float(p)


def compute_decile_markouts(
    wallets: List[str],
    scores_t1: pd.DataFrame,
    orders_t2: pd.DataFrame,
    block_states_t2: pd.DataFrame,
    n_groups: int = 10,
) -> pd.DataFrame:
    """Sort wallets into groups by frozen t1 scores, then measure t2 markouts.

    Wallets are divided into ``n_groups`` groups (deciles when
    ``n_groups=10``) by their ``alpha_w`` in t1.  For each group the
    notional-weighted average markout is computed on the t2 order flow,
    together with a 95 % CI using the normal approximation.

    Parameters
    ----------
    wallets:
        Wallet identifiers to analyse.
    scores_t1:
        Output of :func:`~wallet_scoring.compute_wallet_scores` for window t1;
        must have ``wallet`` and ``alpha_w`` columns.
    orders_t2:
        Aggressive orders from window t2.
    block_states_t2:
        Block-close states from window t2.
    n_groups:
        Number of quantile groups (default 10 = deciles).

    Returns
    -------
    DataFrame with columns ``group`` (0-based), ``mean_markout``,
    ``ci_lower``, ``ci_upper``, ``n_orders``.
    """
    empty = pd.DataFrame(
        columns=["group", "mean_markout", "ci_lower", "ci_upper", "n_orders"]
    )

    wallet_scores = scores_t1[scores_t1["wallet"].isin(wallets)].copy()
    if wallet_scores.empty:
        return empty

    wallet_scores["group"] = pd.qcut(
        wallet_scores["alpha_w"], q=n_groups, labels=False, duplicates="drop"
    )
    wallet_to_group: dict[str, int] = dict(
        zip(wallet_scores["wallet"], wallet_scores["group"].astype(int))
    )

    qualifying_t2 = filter_qualifying_orders(orders_t2)
    qualifying_t2 = qualifying_t2[qualifying_t2["wallet"].isin(wallets)]
    if qualifying_t2.empty:
        return empty

    market_states_t2: dict[str, pd.DataFrame] = {
        mkt: grp.sort_values("timestamp").reset_index(drop=True)
        for mkt, grp in block_states_t2.groupby("market")
    }

    markout_rows: list[dict] = []
    for _, order in qualifying_t2.iterrows():
        wallet = str(order["wallet"])
        grp = wallet_to_group.get(wallet)
        if grp is None:
            continue
        mkt_bs = market_states_t2.get(str(order["market"]), pd.DataFrame())
        x_e = compute_markout(order, mkt_bs)
        if x_e is not None:
            markout_rows.append(
                {
                    "group": grp,
                    "notional": float(order["notional"]),
                    "markout": x_e,
                }
            )

    if not markout_rows:
        return empty

    mdf = pd.DataFrame(markout_rows)

    records: list[dict] = []
    for grp_id, grp in mdf.groupby("group"):
        n = len(grp)
        w = grp["notional"].values
        x = grp["markout"].values
        w_sum = w.sum()
        wmean = float(np.dot(w, x) / w_sum)
        # Weighted variance (frequency weights approximation)
        wvar = float(np.dot(w, (x - wmean) ** 2) / w_sum)
        ci_half = 1.96 * np.sqrt(wvar) / np.sqrt(n)
        records.append(
            {
                "group": int(grp_id),
                "mean_markout": wmean,
                "ci_lower": wmean - ci_half,
                "ci_upper": wmean + ci_half,
                "n_orders": n,
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values("group")
        .reset_index(drop=True)
    )


def participation_persistence(
    cohort_t1: List[str],
    orders_t2: pd.DataFrame,
) -> float:
    """Fraction of t1 cohort wallets that submit orders in t2.

    Parameters
    ----------
    cohort_t1:
        Wallet identifiers belonging to the t1 toxic cohort.
    orders_t2:
        Orders DataFrame from window t2; must have a ``wallet`` column.

    Returns
    -------
    Float in ``[0, 1]``.
    """
    if not cohort_t1:
        return 0.0
    active_t2 = set(orders_t2["wallet"].unique())
    n_present = sum(1 for w in cohort_t1 if w in active_t2)
    return n_present / len(cohort_t1)


def compute_peer_adjusted_scores(
    orders: pd.DataFrame,
    block_states: pd.DataFrame,
    min_orders: int = 100,
) -> pd.DataFrame:
    """Compute peer-adjusted wallet toxicity scores.

    For every qualifying order the notional-weighted mean markout of *all
    other wallets* that traded the same market in the same minute is subtracted
    from the order's own markout.  This removes systematic intra-coin-minute
    price drift from the score, leaving only idiosyncratic alpha.

    The adjustment is computed efficiently using group totals::

        peer_mean_e = (group_weighted_sum - n_e * x_e) / (group_notional - n_e)

    Wallets with fewer than ``min_orders`` valid peer-adjusted observations
    are excluded from the output.

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
        Minimum valid orders per wallet for inclusion.

    Returns
    -------
    DataFrame with columns ``wallet``, ``alpha_w``, ``order_count``,
    ``total_notional``, ``rank`` (1 = most toxic by peer-adjusted alpha).
    """
    qualifying = filter_qualifying_orders(orders)
    if qualifying.empty:
        return pd.DataFrame(
            columns=["wallet", "alpha_w", "order_count", "total_notional", "rank"]
        )

    # Pre-sort and pre-group block states by market once.
    market_states: dict[str, pd.DataFrame] = {
        mkt: grp.sort_values("timestamp").reset_index(drop=True)
        for mkt, grp in block_states.groupby("market")
    }

    # ── Compute raw markout for every qualifying order ───────────────────
    markout_vals: List[Optional[float]] = []
    for _, order in qualifying.iterrows():
        mkt_bs = market_states.get(str(order["market"]), pd.DataFrame())
        markout_vals.append(compute_markout(order, mkt_bs))

    df = qualifying.copy()
    df["markout"] = markout_vals
    df = df.dropna(subset=["markout"]).copy()
    if df.empty:
        return pd.DataFrame(
            columns=["wallet", "alpha_w", "order_count", "total_notional", "rank"]
        )

    # Minute bucket for peer grouping
    df["minute"] = (df["event_time"] // 60).astype(np.int64)

    # ── Group-level weighted sums ────────────────────────────────────────
    df["weighted_markout"] = df["notional"] * df["markout"]
    grp_stats = (
        df.groupby(["market", "minute"])[["notional", "weighted_markout"]]
        .sum()
        .rename(columns={"notional": "grp_notional", "weighted_markout": "grp_wmarkout"})
        .reset_index()
    )
    df = df.merge(grp_stats, on=["market", "minute"], how="left")

    # ── Peer mean (leave-one-out) ────────────────────────────────────────
    peer_notional = df["grp_notional"] - df["notional"]
    peer_wmarkout = df["grp_wmarkout"] - df["weighted_markout"]
    df["peer_mean"] = np.where(
        peer_notional > 0,
        peer_wmarkout / peer_notional,
        0.0,
    )
    df["adj_markout"] = df["markout"] - df["peer_mean"]

    # ── Re-aggregate by wallet ───────────────────────────────────────────
    records: list[dict] = []
    for wallet, grp in df.groupby("wallet"):
        if len(grp) < min_orders:
            continue
        n_arr = grp["notional"].values
        x_arr = grp["adj_markout"].values
        alpha_w = float(np.dot(n_arr, x_arr) / n_arr.sum())
        records.append(
            {
                "wallet": wallet,
                "alpha_w": alpha_w,
                "order_count": len(grp),
                "total_notional": float(n_arr.sum()),
            }
        )

    result = pd.DataFrame(records)
    if result.empty:
        result["rank"] = pd.Series(dtype=int)
        return result

    result = result.sort_values("alpha_w", ascending=False).reset_index(drop=True)
    result["rank"] = range(1, len(result) + 1)
    return result


def compute_rank_pvalue_or_placeholder(
    real_increment: float,
    placebo_increments: List[float],
) -> float:
    """Rank p-value of a real metric against a placebo distribution.

    Uses Laplace smoothing: ``p = (1 + #{placebo >= real}) / (1 + N)``.

    Parameters
    ----------
    real_increment:
        Observed persistence increment for the real toxic cohort.
    placebo_increments:
        Distribution of placebo persistence increments.

    Returns
    -------
    Float in ``(0, 1]``.
    """
    n_above = sum(1 for x in placebo_increments if x >= real_increment)
    return (1 + n_above) / (len(placebo_increments) + 1)


