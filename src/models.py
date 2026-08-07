"""Ridge/GBM modeling, cross-validation, and forecasting experiment pipeline."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

# Anonymous feature columns (must match features.py)
_ANON_COLS = [
    "imb", "nearImb", "ofi1s", "ofi30s", "print_",
    "flow1s", "flow30s", "ret1m", "vol1m", "spread",
]
# Identity feature columns
_IDEN_COLS = [
    "hotImb", "hotNear", "hotOfi1s", "hotOfi30s", "hotPrint",
    "hotFlow1s", "hotFlow30s", "hotBreadth1s", "hotBreadth30s",
    "hotShare", "hotContrib",
]


# ---------------------------------------------------------------------------
# Feature preparation utilities
# ---------------------------------------------------------------------------

def standardize_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Z-score-standardize features using train-set statistics only.

    Parameters
    ----------
    X_train, X_test:
        Feature matrices; columns must match.

    Returns
    -------
    (X_train_std, X_test_std, means, stds) where means/stds are pd.Series
    indexed by feature name.
    """
    means = X_train.mean()
    stds = X_train.std().replace(0, 1.0)
    X_train_std = (X_train - means) / stds
    X_test_std = (X_test - means) / stds
    return X_train_std, X_test_std, means, stds


def add_coin_intercepts(X: pd.DataFrame, coin_col: str = "coin") -> pd.DataFrame:
    """Append one-hot coin dummy columns to *X*.

    The first coin level is dropped (reference category).  The original
    ``coin_col`` column is removed.

    Parameters
    ----------
    X:
        Feature matrix containing the coin identifier column.
    coin_col:
        Name of the coin column (default ``'coin'``).

    Returns
    -------
    pd.DataFrame with dummy columns appended and ``coin_col`` removed.
    """
    dummies = pd.get_dummies(X[coin_col], prefix="coin", drop_first=True)
    return pd.concat([X.drop(columns=[coin_col]), dummies], axis=1)


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

def fit_ridge(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    alpha: float,
) -> Ridge:
    """Fit a Ridge regression with the given regularisation strength.

    Parameters
    ----------
    X_train:
        Training feature matrix (already standardised, NaN-dropped).
    y_train:
        Training targets aligned with *X_train*.
    alpha:
        Ridge penalty (L2 regularisation strength).

    Returns
    -------
    Fitted ``sklearn.linear_model.Ridge`` model.
    """
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X_train.values, y_train.values)
    return model


def fit_gbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    seed: int = 42,
) -> object:
    """Fit a LightGBM gradient-boosted machine with early stopping on a validation set.

    Falls back to Ridge(alpha=1.0) when LightGBM is unavailable.

    Parameters
    ----------
    X_train, y_train:
        Training data.
    X_val, y_val:
        Validation data used for early stopping.
    seed:
        Random seed.

    Returns
    -------
    Fitted LightGBM model (or Ridge fallback).
    """
    if not _HAS_LGB:
        return fit_ridge(X_train, y_train, alpha=1.0)

    params = {
        "objective": "regression",
        "metric": "mse",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": seed,
        "verbose": -1,
        "n_jobs": -1,
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train.values,
        y_train.values,
        eval_set=[(X_val.values, y_val.values)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    return model


def predict(model: object, X_test: pd.DataFrame) -> np.ndarray:
    """Generate predictions from any fitted model (Ridge or LightGBM).

    Parameters
    ----------
    model:
        Fitted model with a ``predict`` method.
    X_test:
        Feature matrix matching the training schema.

    Returns
    -------
    np.ndarray of predictions.
    """
    return np.asarray(model.predict(X_test.values), dtype=float)


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def expanding_day_cv_ridge(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    alphas: List[float],
) -> float:
    """Select ridge alpha via expanding-window day-by-day cross-validation.

    For each validation day d (starting from day 2), the model is trained on
    all days 1 … d−1 and validated on day d.  The alpha minimising mean
    validation MSE is returned.

    Parameters
    ----------
    X:
        Feature matrix (NaN-free).
    y:
        Target Series aligned with *X*.
    dates:
        Calendar date for each row.
    alphas:
        Candidate regularisation values.

    Returns
    -------
    Best alpha (float).
    """
    unique_days = sorted(dates.unique())
    if len(unique_days) < 2:
        return float(alphas[len(alphas) // 2])

    mse_by_alpha: Dict[float, List[float]] = {a: [] for a in alphas}

    for i, val_day in enumerate(unique_days[1:], start=1):
        train_mask = dates < val_day
        val_mask = dates == val_day

        X_tr, y_tr = X[train_mask], y[train_mask]
        X_va, y_va = X[val_mask], y[val_mask]

        if X_tr.empty or X_va.empty:
            continue

        X_tr_s, X_va_s, _, _ = standardize_features(X_tr, X_va)

        for a in alphas:
            m = fit_ridge(X_tr_s, y_tr, a)
            yhat = predict(m, X_va_s)
            mse_by_alpha[a].append(float(np.mean((y_va.values - yhat) ** 2)))

    mean_mses = {a: np.mean(v) for a, v in mse_by_alpha.items() if v}
    if not mean_mses:
        return float(alphas[len(alphas) // 2])
    return min(mean_mses, key=mean_mses.__getitem__)


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def compute_oos_r2(y: np.ndarray, yhat: np.ndarray) -> float:
    """Compute out-of-sample R² against a zero-mean benchmark.

    R² = 1 - sum((y - yhat)²) / sum(y²)

    Parameters
    ----------
    y:
        Realised target values.
    yhat:
        Model forecasts.

    Returns
    -------
    Scalar R² value (can be negative).
    """
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum(y ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def compute_daily_mse(
    y: pd.Series,
    yhat: pd.Series,
    dates: pd.Series,
) -> pd.Series:
    """Compute mean squared error grouped by calendar date.

    Parameters
    ----------
    y:
        Realised targets.
    yhat:
        Forecasts.
    dates:
        Date label for each observation (same length as *y* and *yhat*).

    Returns
    -------
    pd.Series indexed by unique date with MSE values.
    """
    sq_err = (y.values - yhat.values) ** 2
    df = pd.DataFrame({'sq_err': sq_err, 'date': dates.values})
    return df.groupby('date')['sq_err'].mean()


def compute_paired_tstat(
    mse1: pd.Series,
    mse2: pd.Series,
) -> tuple[float, float]:
    """Paired t-test comparing two daily MSE series (model vs. benchmark).

    Parameters
    ----------
    mse1:
        Daily MSE for the first model.
    mse2:
        Daily MSE for the second model.

    Returns
    -------
    (t_statistic, two-sided p_value) tuple.
    """
    t, p = stats.ttest_rel(mse1.values, mse2.values)
    return float(t), float(p)


def compute_relative_gain(r2_bench: float, r2_identity: float) -> float:
    """Relative R² improvement of identity-augmented model over benchmark.

    gain = r2_identity / r2_bench - 1

    Parameters
    ----------
    r2_bench:
        OOS R² of the benchmark model.
    r2_identity:
        OOS R² of the identity-augmented model.

    Returns
    -------
    Fractional gain (e.g. 0.2 = 20 % improvement).
    """
    return r2_identity / r2_bench - 1.0


# ---------------------------------------------------------------------------
# Full experiment pipeline
# ---------------------------------------------------------------------------

def _pool_feature_matrix(
    data: Dict[str, pd.DataFrame],
    horizon_col: str,
    feature_cols: List[str],
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Pool multi-coin feature matrix, drop NaN rows, return (X, y, dates)."""
    frames = []
    for coin, df in data.items():
        sub = df[[*feature_cols, "coin", "date", horizon_col]].copy()
        sub = sub.dropna(subset=[horizon_col] + feature_cols)
        frames.append(sub)
    if not frames:
        return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=object)
    pooled = pd.concat(frames, ignore_index=True)
    X = pooled[feature_cols + ["coin"]].copy()
    y = pooled[horizon_col].copy()
    dates = pooled["date"].copy()
    return X, y, dates


def run_forecasting_experiment(
    train_data: Dict[str, pd.DataFrame],
    eval_data: Dict[str, pd.DataFrame],
    horizons: List[float],
    embargo_ms: int = 0,
) -> pd.DataFrame:
    """Fit ridge benchmark and identity-augmented challenger; evaluate OOS.

    For each horizon the function:

    1. Pools the multi-coin train feature matrix.
    2. Selects ridge alpha via expanding-day CV on the training set.
    3. Fits an anonymous-only Ridge (benchmark) and an anonymous+identity
       Ridge (challenger) using the best alpha.
    4. Evaluates both on the evaluation set.
    5. Reports R², ΔR², relative gain, and paired-MSE t-statistic.

    Parameters
    ----------
    train_data:
        Dict mapping coin → feature DataFrame (output of
        ``build_feature_matrix``).
    eval_data:
        Same structure for the evaluation window.
    horizons:
        Forward-return horizons in seconds (must match target column names
        ``'<h>s'`` in the feature matrices).
    embargo_ms:
        Informational only; embargo is applied upstream by
        ``build_feature_matrix``.

    Returns
    -------
    pd.DataFrame with columns: horizon, r2_bench, r2_identity, delta_r2,
    relative_gain, t_stat, p_value.
    """
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1_000.0]
    rows: List[dict] = []

    for h in horizons:
        target_col = f"{h}s"

        # ── Training data ────────────────────────────────────────────────────
        X_tr_anon, y_tr, dates_tr = _pool_feature_matrix(
            train_data, target_col, _ANON_COLS
        )
        X_tr_full, _, _ = _pool_feature_matrix(
            train_data, target_col, _ANON_COLS + _IDEN_COLS
        )

        if X_tr_anon.empty or y_tr.empty:
            rows.append({"horizon": h, "r2_bench": np.nan, "r2_identity": np.nan,
                         "delta_r2": np.nan, "relative_gain": np.nan,
                         "t_stat": np.nan, "p_value": np.nan})
            continue

        # Add coin dummies and align indices
        X_tr_anon_d = add_coin_intercepts(X_tr_anon, "coin")
        X_tr_full_d = add_coin_intercepts(X_tr_full, "coin")

        # Cross-validate alpha on anonymous features
        best_alpha = expanding_day_cv_ridge(X_tr_anon_d, y_tr, dates_tr, alphas)

        # Standardize
        X_tr_anon_s, _, means_a, stds_a = standardize_features(
            X_tr_anon_d, X_tr_anon_d
        )
        X_tr_full_s, _, means_f, stds_f = standardize_features(
            X_tr_full_d, X_tr_full_d
        )

        # Fit models
        m_bench = fit_ridge(X_tr_anon_s, y_tr, best_alpha)
        m_ident = fit_ridge(X_tr_full_s, y_tr, best_alpha)

        # ── Evaluation data ──────────────────────────────────────────────────
        X_ev_anon, y_ev, dates_ev = _pool_feature_matrix(
            eval_data, target_col, _ANON_COLS
        )
        X_ev_full, _, _ = _pool_feature_matrix(
            eval_data, target_col, _ANON_COLS + _IDEN_COLS
        )

        if X_ev_anon.empty or y_ev.empty:
            rows.append({"horizon": h, "r2_bench": np.nan, "r2_identity": np.nan,
                         "delta_r2": np.nan, "relative_gain": np.nan,
                         "t_stat": np.nan, "p_value": np.nan})
            continue

        X_ev_anon_d = add_coin_intercepts(X_ev_anon, "coin")
        X_ev_full_d = add_coin_intercepts(X_ev_full, "coin")

        # Align columns (eval may have different dummy columns than train)
        for col in X_tr_anon_d.columns:
            if col not in X_ev_anon_d.columns:
                X_ev_anon_d[col] = 0.0
        X_ev_anon_d = X_ev_anon_d[X_tr_anon_d.columns]

        for col in X_tr_full_d.columns:
            if col not in X_ev_full_d.columns:
                X_ev_full_d[col] = 0.0
        X_ev_full_d = X_ev_full_d[X_tr_full_d.columns]

        # Apply train-set standardization to eval data
        X_ev_anon_s = (X_ev_anon_d - means_a) / stds_a
        X_ev_full_s = (X_ev_full_d - means_f) / stds_f

        yhat_bench = predict(m_bench, X_ev_anon_s)
        yhat_ident = predict(m_ident, X_ev_full_s)

        y_ev_arr = y_ev.values
        r2_b = compute_oos_r2(y_ev_arr, yhat_bench)
        r2_i = compute_oos_r2(y_ev_arr, yhat_ident)
        delta_r2 = r2_i - r2_b

        rel_gain = compute_relative_gain(r2_b, r2_i) if r2_b != 0 else np.nan

        # Paired MSE t-test
        mse_b = compute_daily_mse(y_ev, pd.Series(yhat_bench, index=y_ev.index),
                                  dates_ev)
        mse_i = compute_daily_mse(y_ev, pd.Series(yhat_ident, index=y_ev.index),
                                  dates_ev)
        common_days = mse_b.index.intersection(mse_i.index)
        if len(common_days) >= 2:
            t_stat, p_val = compute_paired_tstat(
                mse_b.loc[common_days], mse_i.loc[common_days]
            )
        else:
            t_stat, p_val = np.nan, np.nan

        rows.append({
            "horizon": h,
            "r2_bench": r2_b,
            "r2_identity": r2_i,
            "delta_r2": delta_r2,
            "relative_gain": rel_gain,
            "t_stat": t_stat,
            "p_value": p_val,
        })

    return pd.DataFrame(rows)
