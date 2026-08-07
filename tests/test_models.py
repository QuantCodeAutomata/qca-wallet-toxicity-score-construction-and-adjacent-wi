import pytest
import pandas as pd
import numpy as np
from src.models import compute_oos_r2, compute_daily_mse, compute_paired_tstat, compute_relative_gain

def test_r2_perfect_forecast():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    r2 = compute_oos_r2(y, y)
    assert abs(r2 - 1.0) < 1e-6

def test_r2_zero_forecast():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    yhat = np.zeros(5)
    r2 = compute_oos_r2(y, yhat)
    assert abs(r2) < 1e-6

def test_r2_negative_for_worse_than_zero():
    y = np.array([1.0, 1.0, 1.0])
    yhat = np.array([-10.0, -10.0, -10.0])
    r2 = compute_oos_r2(y, yhat)
    assert r2 < 0

def test_daily_mse_shape():
    n = 100
    dates = pd.Series(pd.date_range('2026-07-21', periods=n, freq='100ms').date)
    y = pd.Series(np.random.randn(n))
    yhat = pd.Series(np.random.randn(n))
    daily_mse = compute_daily_mse(y, yhat, dates)
    assert len(daily_mse) == len(dates.unique())

def test_paired_tstat_returns_tuple():
    mse1 = pd.Series([0.1, 0.2, 0.15, 0.12, 0.18, 0.14, 0.16])
    mse2 = pd.Series([0.09, 0.19, 0.14, 0.11, 0.17, 0.13, 0.15])
    t, p = compute_paired_tstat(mse1, mse2)
    assert isinstance(t, float)
    assert 0 <= p <= 1

def test_relative_gain_formula():
    r2_bench = 0.10
    r2_identity = 0.12
    gain = compute_relative_gain(r2_bench, r2_identity)
    assert abs(gain - 0.2) < 1e-6  # 0.12/0.10 - 1 = 0.2

def test_relative_gain_zero_when_equal():
    r2 = 0.10
    gain = compute_relative_gain(r2, r2)
    assert abs(gain) < 1e-6

def test_r2_formula_consistency():
    # R^2 = 1 - sum((y-yhat)^2) / sum(y^2)
    y = np.array([1.0, 2.0, 3.0])
    yhat = np.array([0.9, 2.1, 2.8])
    expected = 1 - np.sum((y - yhat)**2) / np.sum(y**2)
    assert abs(compute_oos_r2(y, yhat) - expected) < 1e-10
