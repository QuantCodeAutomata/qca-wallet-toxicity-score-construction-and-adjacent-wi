"""Payoff-unit diagnostic for evaluating forecast quality in return-space.

The payoff statistic measures the average realised return per unit of absolute
forecast, analogous to the Sharpe ratio of a signal-proportional strategy.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def compute_payoff_statistic(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Compute the payoff statistic in basis points.

    Payoff = (sum(yhat_i * y_i) / sum(|yhat_i|)) * 10_000

    This equals the average signed return per unit of absolute forecast, which
    is positive for a skilful forecast and zero for a noise forecast.

    Parameters
    ----------
    y_true:
        Realised forward returns (in any consistent unit).
    y_pred:
        Model forecasts aligned with *y_true*.

    Returns
    -------
    Float payoff in basis points.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]

    abs_pred_sum = np.sum(np.abs(y_pred))
    if abs_pred_sum == 0:
        return 0.0

    return float(np.sum(y_pred * y_true) / abs_pred_sum * 10_000)


def compute_threshold_budget(
    y_pred: np.ndarray,
    budget: float,
) -> float:
    """Compute the absolute-forecast threshold corresponding to a signal budget.

    The threshold is the ``(1 - budget)``-quantile of ``|yhat|``.  Applying
    this threshold retains the *budget* fraction of observations with the
    largest absolute forecasts.

    Parameters
    ----------
    y_pred:
        Model forecasts.
    budget:
        Fraction of observations to retain (e.g. 0.1 retains the top 10 %).

    Returns
    -------
    Float threshold on |yhat|.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    abs_pred = np.abs(y_pred[np.isfinite(y_pred)])
    if len(abs_pred) == 0:
        return 0.0
    return float(np.quantile(abs_pred, 1.0 - budget))


def compute_thresholded_payoff(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
) -> float:
    """Payoff statistic restricted to observations where ``|yhat| > threshold``.

    Parameters
    ----------
    y_true:
        Realised returns.
    y_pred:
        Model forecasts.
    threshold:
        Minimum absolute forecast to include.

    Returns
    -------
    Float payoff in basis points for the filtered subset.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = (np.abs(y_pred) > threshold) & np.isfinite(y_true) & np.isfinite(y_pred)
    return compute_payoff_statistic(y_true[mask], y_pred[mask])


def payoff_table(
    y_true: pd.Series,
    forecasts_dict: Dict[str, np.ndarray],
    budgets: List[float],
) -> pd.DataFrame:
    """Build a payoff table across models and signal budgets.

    Parameters
    ----------
    y_true:
        Realised returns Series.
    forecasts_dict:
        Mapping from model name to forecast array (same length as *y_true*).
    budgets:
        List of budget fractions (e.g. ``[0.05, 0.10, 0.25, 1.0]``).

    Returns
    -------
    pd.DataFrame with rows = model names and columns = budget fractions,
    values = payoff in basis points.  A ``'full'`` column (budget=1.0,
    i.e. no threshold) is always appended when not already present.
    """
    y_arr = np.asarray(y_true, dtype=float)

    all_budgets = list(budgets)
    if 1.0 not in all_budgets:
        all_budgets.append(1.0)

    records = {}
    for model_name, y_pred in forecasts_dict.items():
        y_pred_arr = np.asarray(y_pred, dtype=float)
        row: Dict[str, float] = {}
        for bud in all_budgets:
            if bud >= 1.0:
                row[bud] = compute_payoff_statistic(y_arr, y_pred_arr)
            else:
                thresh = compute_threshold_budget(y_pred_arr, bud)
                row[bud] = compute_thresholded_payoff(y_arr, y_pred_arr, thresh)
        records[model_name] = row

    df = pd.DataFrame(records).T
    df.index.name = "model"
    df.columns.name = "budget"
    return df
